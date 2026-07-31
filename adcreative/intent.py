"""意图解析 Agent: 关键词 / 商品 URL -> 结构化投放 Brief。

产品体验约束（设计文档 4.1）: 缺失关键字段最多一轮追问，其余给推断默认值
让用户一键确认。实现为返回 (brief, questions) —— brief 总是可用（缺项已填
默认值），questions 非空时由调用方决定是否追问后重跑。
"""
from __future__ import annotations

from typing import Callable

from adcreative.llm import LLMClient, complete_with_retry, lang_note
from adcreative.schema import Brief

INTENT_SYSTEM = """你是广告投放的意图解析器。根据用户输入（关键词或商品页内容）\
输出结构化投放 brief。要求:
- 卖点从输入中提取，不要编造具体数字参数
- 金额一律 USD；用户未提供预算时 total_usd/daily_usd 填 0（不要猜）
- 未提到的受众/平台字段给合理默认值
- landing_url: 输入是 URL 时用原 URL，否则留默认值"""

# 追问上限一轮，默认值兜底
_DEFAULT_TOTAL_USD = 3000.0
_DEFAULT_DAILY_USD = 100.0


def _default_fetch(url: str) -> str:
    """抓取商品页文本（截断防超长）。测试注入假 fetcher 替代。"""
    import httpx

    # 带浏览器 UA：不少站点对无 UA 请求返回 202 空响应（反爬），会导致
    # 只能从 URL slug 猜产品，抓不到页面真实卖点。
    ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    resp = httpx.get(url, timeout=15.0, follow_redirects=True,
                     headers={"User-Agent": ua})
    resp.raise_for_status()
    return resp.text[:20_000]


def parse_brief(
    user_input: str,
    llm: LLMClient,
    fetch_page: Callable[[str], str] | None = None,
    lang: str = "zh",
) -> tuple[Brief, list[str]]:
    """解析用户输入为 Brief。

    Returns:
        (brief, questions): questions 是需要向用户确认的追问（最多一轮）；
        对应字段已填默认值，用户不回答也能继续跑。
    """
    is_url = user_input.startswith(("http://", "https://"))
    if is_url:
        page = (fetch_page or _default_fetch)(user_input)
        prompt = f"商品页 URL: {user_input}\n\n页面内容:\n{page}"
    else:
        prompt = f"投放关键词: {user_input}"

    brief = complete_with_retry(llm, lang_note(lang) + INTENT_SYSTEM, prompt, Brief)

    en = (lang or "zh").lower() == "en"
    questions: list[str] = []
    if brief.budget.total_usd <= 0:
        questions.append(
            (f"What is the total budget (USD)? Defaulting to "
             f"${_DEFAULT_TOTAL_USD:.0f} total / ${_DEFAULT_DAILY_USD:.0f} daily "
             f"if unconfirmed." if en else
             f"投放总预算是多少（USD）？未确认将按默认 ${_DEFAULT_TOTAL_USD:.0f} "
             f"总预算 / ${_DEFAULT_DAILY_USD:.0f} 日预算执行。"))
        brief.budget.total_usd = _DEFAULT_TOTAL_USD
        brief.budget.daily_usd = _DEFAULT_DAILY_USD
    if not brief.audience.geo:
        questions.append("Which regions to target? Defaulting to US." if en
                         else "投放地区是哪里？未确认将默认投放 US。")
        brief.audience.geo = ["US"]
    if is_url and brief.landing_url in ("", "https://example.com"):
        brief.landing_url = user_input

    return brief, questions
