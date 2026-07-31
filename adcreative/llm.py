"""LLM 客户端抽象。

管线只依赖 LLMClient 协议: complete(system, user, schema) -> schema 实例。
    - ClaudeLLM: Anthropic SDK 的 messages.parse 结构化输出。
    - DeepSeekLLM: OpenAI 兼容 API（deepseek-chat），走 JSON 模式 + Pydantic
      校验，成本远低于 Claude。仅用 httpx，不加新依赖。
    - FakeLLM: 确定性假实现，测试与离线演示用，零 API 成本、完全可复现。
    - make_live_llm(): 按环境变量 ADPILOT_PROVIDER 选 claude/deepseek。

不同 provider 的结构化输出机制不同，但都收敛到「返回一个通过 schema 校验的
Pydantic 实例」；不合规时由 complete_with_retry 回喂错误重试。
"""
from __future__ import annotations

import os
from typing import Protocol, Sequence, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def lang_note(lang: str) -> str:
    """按 UI 语言返回强制输出语言的前置指令块（拼在系统提示词最前面）。

    en / zh 都给对称强指令：en 压过中文系统提示词惯性；zh 防止 DeepSeek 等模型
    在输入含英文商品页时把文案写成英文，保证全平台语言统一。
    """
    if (lang or "zh").lower() == "en":
        return ("【OUTPUT LANGUAGE — HIGHEST PRIORITY】\n"
                "You MUST write EVERY word of your entire output in ENGLISH: "
                "all field values, product names, selling points, headlines, "
                "descriptions, reasons, suggestions, questions and narratives. "
                "Do NOT output any Chinese characters under any circumstances, "
                "even if the instructions or input below are in Chinese.\n\n")
    return ("【输出语言 — 最高优先级】\n"
            "你必须用简体中文写出全部输出：所有字段值、卖点、标题、描述、理由、"
            "建议、追问与叙述都要是中文。即使输入的商品页或关键词是英文，也不要"
            "整句输出英文（品牌名、产品型号、以及 ROAS/CTR/USB-C 这类专有名词或"
            "缩写可保留原文）。\n\n")


def is_en(lang: str) -> bool:
    return (lang or "zh").lower() == "en"


def has_cjk(text: str) -> bool:
    """文本是否含中日韩统一表意文字（用于英文模式下的语言泄漏兜底检测）。"""
    return any("一" <= ch <= "鿿" for ch in (text or ""))


class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: type[T]) -> T: ...


class ClaudeLLM:
    """Anthropic API 实现。需要 ANTHROPIC_API_KEY（或 ant auth login 的凭据）。"""

    def __init__(self, model: str = "claude-opus-4-8", client=None,
                 max_tokens: int = 16000, api_key: str | None = None):
        import anthropic  # 延迟导入: 只有真实调用才需要装 anthropic
        if client is None:
            # 清洗 key 的首尾空白：从文件/环境读入的 key 常带尾随空格或换行，
            # 未清洗会变成非法 HTTP header 值 -> 误报为 APIConnectionError。
            import os
            key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, schema: type[T]) -> T:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return response.parsed_output


def _strip_code_fence(s: str) -> str:
    """去掉 LLM 偶尔套的 ```json ... ``` 围栏，只留 JSON 主体。"""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


class DeepSeekLLM:
    """DeepSeek（OpenAI 兼容 API）实现，成本远低于 Claude。

    走 chat/completions 的 JSON 模式：把目标 schema 的 JSON Schema 注入系统
    提示，要求只输出匹配的 JSON，再用 Pydantic 校验。不合规交给
    complete_with_retry 回喂重试。仅依赖 httpx。
    """

    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com",
                 timeout: float = 120.0, http_client=None):
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = http_client  # 注入 httpx.Client 便于测试

    def complete(self, system: str, user: str, schema: type[T]) -> T:
        import json

        import httpx

        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        # 格式说明用英文（纯结构指令，不影响内容语言）：中文格式说明会把系统
        # 提示词末尾锚定成中文，诱导英文模式下的内容也输出中文。
        sys_full = (f"{system}\n\nOutput ONLY a single JSON object that strictly "
                    f"matches this JSON Schema. No explanations, no markdown code "
                    f"fences. Keep the content language exactly as instructed "
                    f"above.\n{schema_json}")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_full},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        finally:
            if self._client is None:
                client.close()
        return schema.model_validate_json(_strip_code_fence(content))


def make_live_llm(provider: str | None = None) -> LLMClient:
    """按 provider（或环境变量 ADPILOT_PROVIDER）构造真实 LLM 客户端。
    缺对应 key 时抛 ValueError，供上层决定降级。"""
    provider = (provider or os.environ.get("ADPILOT_PROVIDER") or "claude").lower()
    if provider == "deepseek":
        if not (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
            raise ValueError("DEEPSEEK_API_KEY 未设置")
        return DeepSeekLLM()
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise ValueError("ANTHROPIC_API_KEY 未设置")
    return ClaudeLLM()


class FakeLLM:
    """测试/离线演示用。responses 按调用顺序弹出；Exception 项被 raise。"""

    def __init__(self, responses: Sequence[BaseModel | Exception]):
        self._queue: list[BaseModel | Exception] = list(responses)
        self.calls: list[tuple[str, str, type]] = []

    def complete(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((system, user, schema))
        if not self._queue:
            raise RuntimeError(
                f"FakeLLM 响应耗尽（第 {len(self.calls)} 次调用, schema={schema.__name__}）")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, schema):
            raise TypeError(
                f"FakeLLM 预设响应类型 {type(item).__name__} 与请求 schema "
                f"{schema.__name__} 不符")
        return item


def complete_with_retry(
    llm: LLMClient, system: str, user: str, schema: type[T],
    max_attempts: int = 2,
) -> T:
    """带重试的结构化调用: 失败时把错误信息附回提示词再试一次。

    LLM 结构化输出偶发不合 schema 是已知高频风险（设计文档风险清单），
    重试 + 错误回喂是标准缓解。超过 max_attempts 原样抛出。
    """
    last_err: Exception | None = None
    prompt = user
    for _ in range(max_attempts):
        try:
            return llm.complete(system, prompt, schema)
        except Exception as e:  # ValidationError / API 错误统一处理
            last_err = e
            prompt = (f"{user}\n\n上一次输出未通过校验，错误如下，请修正后严格按 "
                      f"schema 重新输出:\n{e}")
    assert last_err is not None
    raise last_err
