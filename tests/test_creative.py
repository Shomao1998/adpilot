"""创意管线测试（全程 FakeLLM，零 API 依赖、完全确定性）。

覆盖:
    1. schema 层: USD-only (ADR-5)、RSA 30/90 与资产数量、Meta/TikTok 长度限制。
    2. LLM 抽象: FakeLLM 顺序/类型契约、complete_with_retry 的错误回喂重试。
    3. 意图解析: 关键词/URL 两条路径、缺预算的一轮追问 + 默认值兜底。
    4. 审核: L1 违禁词 reject（不进 L2）、绝对化用语 revise、干净文案进 L2。
    5. 小闭环: revise 回流重试、reject 不回流、重试轮数上限、W3 验收 ≥6 组过审。
    6. 排版: 三尺寸输出、文件落盘。
"""
from __future__ import annotations

import pytest
from PIL import Image
from pydantic import ValidationError

from adcreative.copywriter import generate_copy
from adcreative.intent import parse_brief
from adcreative.layout import CREATIVE_SIZES, render_all_sizes, render_creative
from adcreative.llm import (
    DeepSeekLLM, FakeLLM, complete_with_retry, make_live_llm,
)
from adcreative.pipeline import generate_creatives
from adcreative.review import review_variant, run_l1_rules
from adcreative.schema import (
    Brief, Budget, CopyBatch, CopyVariant, Product, ReviewVerdict,
)


# ---------------------------------------------------------------------------
# fixture 构造器
# ---------------------------------------------------------------------------

def _brief(**kw) -> Brief:
    return Brief(product=Product(name="便携咖啡机", category="厨房小电",
                                 selling_points=["3分钟出咖啡", "500g 轻量"]),
                 **kw)


def _meta_variant(headline="Fresh coffee in 3 minutes",
                  body="Loved by thousands of campers.") -> CopyVariant:
    return CopyVariant(platform="meta", variant_id="m-0",
                       headline=headline, body=body)


def _google_variant() -> CopyVariant:
    return CopyVariant(
        platform="google", variant_id="g-0",
        rsa_headlines=["Portable Coffee Maker", "Brew In 3 Minutes", "Only 500g"],
        rsa_descriptions=["Fresh espresso anywhere you go.",
                          "Compact design for travel and camping."])


def _batch(platform: str, n: int, headline_prefix="Nice clean copy") -> CopyBatch:
    if platform == "google":
        return CopyBatch(variants=[_google_variant() for _ in range(n)])
    return CopyBatch(variants=[
        CopyVariant(platform=platform, headline=f"{headline_prefix} {i}",
                    body="A concrete, verifiable selling point.")
        for i in range(n)])


PASS = ReviewVerdict(verdict="pass")


# ---------------------------------------------------------------------------
# 1. schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_currency_usd_only(self):
        with pytest.raises(ValidationError):
            Budget(total_usd=100, daily_usd=10, currency="CNY")
        assert Budget().currency == "USD"

    def test_rsa_headline_length_and_count(self):
        with pytest.raises(ValidationError, match="超长"):
            CopyVariant(platform="google",
                        rsa_headlines=["x" * 31, "ok", "ok2"],
                        rsa_descriptions=["d1", "d2"])
        with pytest.raises(ValidationError, match="标题数"):
            CopyVariant(platform="google", rsa_headlines=["only", "two"],
                        rsa_descriptions=["d1", "d2"])
        with pytest.raises(ValidationError, match="描述数"):
            CopyVariant(platform="google",
                        rsa_headlines=["a", "b", "c"],
                        rsa_descriptions=["only one"])

    def test_rsa_description_length(self):
        with pytest.raises(ValidationError, match="描述超长"):
            CopyVariant(platform="google",
                        rsa_headlines=["a", "b", "c"],
                        rsa_descriptions=["x" * 91, "ok"])

    def test_meta_headline_limits(self):
        with pytest.raises(ValidationError, match="Meta 标题超长"):
            CopyVariant(platform="meta", headline="x" * 41)
        with pytest.raises(ValidationError, match="缺少 headline"):
            CopyVariant(platform="meta")

    def test_tiktok_text_limit(self):
        with pytest.raises(ValidationError, match="TikTok"):
            CopyVariant(platform="tiktok", headline="x" * 101)


# ---------------------------------------------------------------------------
# 2. LLM 抽象
# ---------------------------------------------------------------------------

class TestFakeLLM:
    def test_returns_in_order_and_records_calls(self):
        llm = FakeLLM([PASS, ReviewVerdict(verdict="reject")])
        assert llm.complete("s", "u", ReviewVerdict).verdict == "pass"
        assert llm.complete("s", "u", ReviewVerdict).verdict == "reject"
        assert len(llm.calls) == 2

    def test_schema_mismatch_raises(self):
        llm = FakeLLM([PASS])
        with pytest.raises(TypeError):
            llm.complete("s", "u", CopyBatch)

    def test_exhausted_raises(self):
        with pytest.raises(RuntimeError, match="耗尽"):
            FakeLLM([]).complete("s", "u", ReviewVerdict)

    def test_retry_feeds_error_back(self):
        """第一次失败 -> 重试提示词携带错误信息 -> 第二次成功。"""
        llm = FakeLLM([ValueError("RSA 标题超长"), PASS])
        out = complete_with_retry(llm, "s", "user-prompt", ReviewVerdict)
        assert out.verdict == "pass"
        assert "RSA 标题超长" in llm.calls[1][1]   # 错误回喂进了第二次 prompt

    def test_retry_exhausted_reraises(self):
        llm = FakeLLM([ValueError("bad"), ValueError("bad again")])
        with pytest.raises(ValueError, match="bad again"):
            complete_with_retry(llm, "s", "u", ReviewVerdict)


class TestDeepSeek:
    """DeepSeek（OpenAI 兼容 JSON 模式）走 mock httpx，不触网。"""

    def _client(self, content: str, capture: dict | None = None):
        import json as _json
        import httpx

        def handler(request):
            if capture is not None:
                capture["body"] = _json.loads(request.content)
                capture["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]})
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_parses_json_and_sends_json_mode(self):
        import json
        cap: dict = {}
        content = json.dumps({"verdict": "pass", "reasons": [], "suggestions": ""})
        llm = DeepSeekLLM(api_key="test-key",
                          http_client=self._client(content, cap))
        v = llm.complete("审核", "文案", ReviewVerdict)
        assert v.verdict == "pass"
        assert cap["body"]["response_format"]["type"] == "json_object"
        assert cap["body"]["model"] == "deepseek-chat"
        assert cap["auth"] == "Bearer test-key"
        # schema 被注入系统提示
        assert "JSON Schema" in cap["body"]["messages"][0]["content"]

    def test_strips_markdown_fence(self):
        content = '```json\n{"verdict":"revise","reasons":["x"],"suggestions":"y"}\n```'
        llm = DeepSeekLLM(api_key="k", http_client=self._client(content))
        assert llm.complete("s", "u", ReviewVerdict).verdict == "revise"

    def test_invalid_json_raises_for_retry(self):
        """返回不合 schema -> 抛错，交给 complete_with_retry 重试。"""
        llm = DeepSeekLLM(api_key="k",
                          http_client=self._client('{"verdict":"nope"}'))
        with pytest.raises(Exception):
            llm.complete("s", "u", ReviewVerdict)


class TestProviderFactory:
    def test_deepseek_selected(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        assert isinstance(make_live_llm("deepseek"), DeepSeekLLM)

    def test_env_var_selects_provider(self, monkeypatch):
        monkeypatch.setenv("ADPILOT_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        assert isinstance(make_live_llm(), DeepSeekLLM)

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            make_live_llm("deepseek")


# ---------------------------------------------------------------------------
# 3. 意图解析
# ---------------------------------------------------------------------------

class TestIntent:
    def test_keyword_with_budget_no_questions(self):
        brief_out = _brief(budget=Budget(total_usd=3000, daily_usd=100))
        llm = FakeLLM([brief_out])
        brief, questions = parse_brief("便携咖啡机 促销 北美", llm)
        assert questions == []
        assert brief.budget.total_usd == 3000
        assert "便携咖啡机" in llm.calls[0][1]

    def test_missing_budget_asks_once_and_defaults(self):
        llm = FakeLLM([_brief()])  # budget 全 0
        brief, questions = parse_brief("咖啡机", llm)
        assert len(questions) == 1 and "预算" in questions[0]
        assert brief.budget.total_usd == 3000.0   # 默认值兜底，可继续跑

    def test_url_path_fetches_page(self):
        fetched: list[str] = []

        def fake_fetch(url: str) -> str:
            fetched.append(url)
            return "<html>便携咖啡机 $79.9 三分钟出咖啡</html>"

        llm = FakeLLM([_brief(budget=Budget(total_usd=1000, daily_usd=50))])
        brief, _ = parse_brief("https://shop.example.com/coffee", llm,
                               fetch_page=fake_fetch)
        assert fetched == ["https://shop.example.com/coffee"]
        assert "三分钟出咖啡" in llm.calls[0][1]      # 页面内容进了 prompt
        assert brief.landing_url == "https://shop.example.com/coffee"


# ---------------------------------------------------------------------------
# 4. 审核
# ---------------------------------------------------------------------------

class TestReview:
    def test_banned_term_rejects_without_llm(self):
        v = _meta_variant(headline="Best gun holster deals")
        llm = FakeLLM([])   # 队列为空: 若进了 L2 会炸
        verdict = review_variant(v, llm)
        assert verdict.verdict == "reject"
        assert llm.calls == []

    def test_superlative_gets_revise(self):
        verdict = run_l1_rules(_meta_variant(headline="The best coffee maker"))
        assert verdict.verdict == "revise"
        assert any("绝对化" in r for r in verdict.reasons)

    def test_chinese_superlative(self):
        assert run_l1_rules(_meta_variant(headline="全网第一咖啡机")).verdict == "revise"

    def test_restricted_category_revise(self):
        v = _meta_variant(body="Pair it with our weight loss pill bundle")
        assert run_l1_rules(v).verdict == "revise"

    def test_clean_copy_goes_to_l2(self):
        llm = FakeLLM([ReviewVerdict(verdict="revise",
                                     reasons=["隐含疗效承诺"],
                                     suggestions="softened")])
        verdict = review_variant(_meta_variant(), llm)
        assert verdict.verdict == "revise"      # 来自 L2
        assert len(llm.calls) == 1

    def test_l1_reason_english(self):
        v = _meta_variant(headline="The best coffee maker")
        assert run_l1_rules(v, lang="en").verdict == "revise"
        assert any("superlative" in r.lower()
                   for r in run_l1_rules(v, lang="en").reasons)

    def test_google_variant_all_text_reviewed(self):
        v = _google_variant()
        assert "Portable Coffee Maker" in v.all_text()
        assert run_l1_rules(v).verdict == "pass"


# ---------------------------------------------------------------------------
# 5. 小闭环（生成 -> 审核 -> 回流）
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_all_pass_first_round_meets_w3(self):
        brief = _brief()
        responses = []
        for platform in brief.platforms:            # 每平台: 1 次生成 + 3 次 L2
            responses.append(_batch(platform, 3))
            responses.extend([PASS] * 3)
        llm = FakeLLM(responses)
        result = generate_creatives(brief, llm, n_variants=3)
        assert result.ok                            # ≥6 组过审 (实际 9)
        assert len(result.approved) == 9
        assert result.rounds_used == {"meta": 1, "google": 1, "tiktok": 1}

    def test_revise_flows_back_with_feedback(self):
        """meta 平台: 第 1 轮 1 组 revise -> 第 2 轮携带审核意见重新生成并过审。"""
        brief = _brief(platforms=["meta"])
        revise = ReviewVerdict(verdict="revise", reasons=["夸大"],
                               suggestions="use concrete claims")
        llm = FakeLLM([
            _batch("meta", 3),          # 第 1 轮生成
            PASS, PASS, revise,          # 审核: 2 过 1 revise
            _batch("meta", 1),          # 第 2 轮只补 1 组
            PASS,
        ])
        result = generate_creatives(brief, llm, n_variants=3)
        assert len(result.approved) == 3
        assert result.rounds_used["meta"] == 2
        # 回流的生成 prompt 应携带审核意见
        regen_prompt = llm.calls[4][1]
        assert "use concrete claims" in regen_prompt

    def test_reject_does_not_flow_back(self):
        """reject（硬违规）不回流: 只应有 1 轮生成。"""
        brief = _brief(platforms=["meta"])
        llm = FakeLLM([
            _batch("meta", 2),
            PASS,
            ReviewVerdict(verdict="reject", reasons=["违禁类目"]),
        ])
        result = generate_creatives(brief, llm, n_variants=2)
        assert len(result.approved) == 1
        assert len(result.rejected) == 1
        assert result.rounds_used["meta"] == 1      # 没有第 2 轮

    def test_retry_rounds_capped(self):
        """永远 revise: 初始 1 轮 + 回流 2 轮后强制停止。"""
        brief = _brief(platforms=["meta"])
        revise = ReviewVerdict(verdict="revise", reasons=["r"], suggestions="s")
        llm = FakeLLM([
            _batch("meta", 1), revise,   # 初始
            _batch("meta", 1), revise,   # 回流 1
            _batch("meta", 1), revise,   # 回流 2
        ])
        result = generate_creatives(brief, llm, n_variants=1)
        assert result.approved == []
        assert result.rounds_used["meta"] == 3
        assert not result.ok

    def test_one_platform_failure_does_not_crash_others(self):
        """某平台生成屡次抛错（DeepSeek 输出不合 schema）：跳过它，其他平台照常。"""
        brief = _brief(platforms=["meta", "google"])
        # meta 正常过审；google 每轮生成都抛错（3 轮都失败）
        llm = FakeLLM([
            _batch("meta", 2), PASS, PASS,          # meta: 2 组过审
            ValueError("RSA 超长"),                  # google r0
            ValueError("RSA 超长"),                  # google r1
            ValueError("RSA 超长"),                  # google r2
        ])
        result = generate_creatives(brief, llm, n_variants=2)
        assert len(result.approved) == 2            # meta 的 2 组不受影响
        assert all(a.variant.platform == "meta" for a in result.approved)
        assert result.rounds_used["google"] == 3    # google 尝试满 3 轮后放弃

    def test_review_failure_counts_as_revise(self):
        """审核调用连续失败（complete_with_retry 2 次都挂）当作 revise，不崩。"""
        err = RuntimeError("审核 API 挂了")
        brief = _brief(platforms=["meta"])
        llm = FakeLLM([
            _batch("meta", 1), err, err,   # r0: 审核 2 次都抛错 -> 当作 revise
            _batch("meta", 1), PASS,       # r1: 补回并过审
        ])
        result = generate_creatives(brief, llm, n_variants=1)
        assert len(result.approved) == 1
        assert result.rounds_used["meta"] == 2

    def test_render_dir_produces_images(self, tmp_path):
        brief = _brief(platforms=["google"])
        llm = FakeLLM([_batch("google", 2), PASS, PASS])
        result = generate_creatives(brief, llm, n_variants=2,
                                    render_dir=tmp_path)
        assert len(result.approved) == 2
        for item in result.approved:
            assert set(item.image_paths) == set(CREATIVE_SIZES)
            for p in item.image_paths.values():
                assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. 排版
# ---------------------------------------------------------------------------

class TestLayout:
    def test_all_sizes_rendered(self, tmp_path):
        paths = render_all_sizes("Fresh Coffee In 3 Minutes",
                                 "Compact 500g design for travel.",
                                 "Shop Now", out_dir=tmp_path, stem="v0")
        assert set(paths) == {"feed_1x1", "story_9x16", "land_1911"}
        for name, p in paths.items():
            img = Image.open(p)
            assert img.size == CREATIVE_SIZES[name]
            assert img.mode == "RGB"

    def test_product_image_composited(self):
        prod = Image.new("RGBA", (400, 400), (255, 0, 0, 255))
        img = render_creative("Headline", "Body", "Buy",
                              size=(1080, 1080), product_image=prod)
        assert img.size == (1080, 1080)

    def test_copywriter_variant_ids(self):
        brief = _brief(platforms=["meta"])
        llm = FakeLLM([_batch("meta", 2)])
        variants = generate_copy(brief, llm, "meta", n_variants=2, round_tag=1)
        assert [v.variant_id for v in variants] == ["meta-r1-0", "meta-r1-1"]
