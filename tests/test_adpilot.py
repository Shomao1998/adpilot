"""编排层测试（W4）。

确定性核心（指标/出价/诊断/CTR权重）用手工构造精确断言；LangGraph 编排跑
真实 3 日闭环做集成 + 复现性。LLM 依赖走 FakeLLM。
"""
from __future__ import annotations

import pytest

from adcreative.llm import FakeLLM
from adcreative.schema import CopyVariant
from adpilot.bidding import BiddingAgent, BidConfig, PlatformPlan
from adpilot.ctr_ranking import CtrScore, CtrScoreBatch, rank_creatives
from adpilot.decision import Decision, DecisionLog
from adpilot.diagnosis import DiagnosisNarrative, diagnose_day
from adpilot.metrics import DayMetrics, MICROS
from adpilot.orchestrator import AdPilotOrchestrator, PilotConfig
from adsim.models import Platform

META, GOOGLE, TIKTOK = (Platform.META_SIM, Platform.GOOGLE_SIM,
                        Platform.TIKTOK_SIM)


def _m(platform=META, day=0, imps=10_000, clicks=200, convs=10,
       spend_usd=100.0, rev_usd=300.0) -> DayMetrics:
    return DayMetrics(platform=platform, day_index=day, impressions=imps,
                      clicks=clicks, conversions=convs,
                      spend_micros=int(spend_usd * MICROS),
                      revenue_micros=int(rev_usd * MICROS))


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

class TestDayMetrics:
    def test_derived_ratios(self):
        m = _m(imps=1000, clicks=50, convs=5, spend_usd=100, rev_usd=250)
        assert m.ctr == pytest.approx(0.05)
        assert m.cvr == pytest.approx(0.10)
        assert m.cpa_usd == pytest.approx(20.0)
        assert m.roas == pytest.approx(2.5)

    def test_zero_guards(self):
        m = DayMetrics(META, 0, 0, 0, 0, 0, 0)
        assert m.ctr == 0.0 and m.cvr == 0.0 and m.roas == 0.0
        assert m.cpa_micros == float("inf")   # 无转化 -> 下游判超成本


# ---------------------------------------------------------------------------
# 决策日志
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_filter_and_applied(self):
        log = DecisionLog()
        log.add(Decision(0, "meta_sim", "set_initial_bid", "r", "e", 0.6,
                         applied=True))
        log.add(Decision(1, "meta_sim", "hold", "r", "e", 0.5))
        assert len(log.for_day(0)) == 1
        assert len(log.applied()) == 1


# ---------------------------------------------------------------------------
# 出价 Agent
# ---------------------------------------------------------------------------

class TestBiddingInitial:
    def test_initial_bid_formula(self):
        agent = BiddingAgent(BidConfig(target_cpa_usd=25.0, aggressiveness=0.8))
        plans = {META: PlatformPlan(META, bid_micros=5_000_000,
                                    budget_micros=100 * MICROS)}
        [d] = agent.initial_bids(plans)
        # 25 × 0.012 × 0.020 × 1000 × 0.8 = 4.8 USD
        assert d.action == "set_initial_bid"
        assert d.new_value == pytest.approx(4_800_000, rel=1e-6)


class TestBiddingLang:
    def test_english_decision_text(self):
        import re
        agent = BiddingAgent(BidConfig(lang="en"))
        plans = {META: PlatformPlan(META, 5 * MICROS, 100 * MICROS)}
        [d] = agent.initial_bids(plans)
        assert "Target CPA" in d.reason
        assert not re.search(r"[一-鿿]", d.reason + d.expected_impact)


class TestBiddingDaily:
    def _plan(self, budget_usd=100.0, bid_usd=8.0):
        return PlatformPlan(META, bid_micros=int(bid_usd * MICROS),
                            budget_micros=int(budget_usd * MICROS))

    def test_cost_overrun_lowers_bid(self):
        agent = BiddingAgent(BidConfig(target_cpa_usd=25.0))
        plans = {META: self._plan(bid_usd=8.0)}
        # CPA = 300/5 = $60 > 25×1.5=$37.5；convs=5 ≥ 学习期护栏(3)
        hist = {META: [_m(convs=5, spend_usd=300, rev_usd=120)]}
        decs = agent.daily_decisions(1, plans, hist)
        assert any(d.action == "lower_bid" for d in decs)
        low = next(d for d in decs if d.action == "lower_bid")
        assert low.new_value == pytest.approx(8_000_000 * 0.8)  # -20%

    def test_efficient_raises_bid_when_budget_maxed(self):
        agent = BiddingAgent(BidConfig(target_cpa_usd=25.0))
        plans = {META: self._plan(budget_usd=100, bid_usd=8.0)}
        # CPA = 100/20 = $5 < 25×0.7 且花满预算
        hist = {META: [_m(convs=20, spend_usd=100, rev_usd=600)]}
        decs = agent.daily_decisions(1, plans, hist)
        assert any(d.action == "raise_bid" for d in decs)

    def test_learning_guard_holds(self):
        agent = BiddingAgent(BidConfig(learning_min_conversions=3))
        plans = {META: self._plan()}
        hist = {META: [_m(convs=1, spend_usd=100, rev_usd=10)]}  # CPA 巨高但样本不足
        decs = agent.daily_decisions(1, plans, hist)
        assert all(d.action == "hold" for d in decs if d.platform == "meta_sim")

    def test_fatigue_triggers_refresh(self):
        agent = BiddingAgent(BidConfig(fatigue_window=3))
        plans = {META: self._plan()}
        # CTR 连续 4 天严格下滑
        hist = {META: [_m(day=i, imps=10_000, clicks=c)
                       for i, c in enumerate([500, 400, 300, 200])]}
        decs = agent.daily_decisions(4, plans, hist)
        assert any(d.action == "trigger_creative_refresh" for d in decs)

    def test_no_fatigue_when_ctr_stable(self):
        agent = BiddingAgent(BidConfig(fatigue_window=3))
        plans = {META: self._plan()}
        hist = {META: [_m(day=i, imps=10_000, clicks=300, convs=10)
                       for i in range(4)]}
        decs = agent.daily_decisions(4, plans, hist)
        assert not any(d.action == "trigger_creative_refresh" for d in decs)


class TestReallocation:
    def test_shifts_budget_high_gap(self):
        agent = BiddingAgent(BidConfig(realloc_roas_gap=0.5, max_change_pct=0.2))
        plans = {META: PlatformPlan(META, 8 * MICROS, 100 * MICROS),
                 GOOGLE: PlatformPlan(GOOGLE, 8 * MICROS, 100 * MICROS)}
        # meta ROAS 5, google ROAS 1 -> gap 0.8 > 0.5
        hist = {META: [_m(META, spend_usd=100, rev_usd=500)],
                GOOGLE: [_m(GOOGLE, spend_usd=100, rev_usd=100)]}
        decs = agent.daily_decisions(3, plans, hist)
        lower = next(d for d in decs if d.action == "lower_budget")
        raise_ = next(d for d in decs if d.action == "raise_budget")
        assert lower.platform == "google_sim"
        assert raise_.platform == "meta_sim"
        # 总量守恒: 转出 == 转入
        assert (lower.old_value - lower.new_value
                == raise_.new_value - raise_.old_value)

    def test_cooldown_blocks_realloc(self):
        agent = BiddingAgent(BidConfig(realloc_cooldown_days=2))
        plans = {META: PlatformPlan(META, 8 * MICROS, 100 * MICROS,
                                    last_realloc_day=2),
                 GOOGLE: PlatformPlan(GOOGLE, 8 * MICROS, 100 * MICROS,
                                      last_realloc_day=2)}
        hist = {META: [_m(META, spend_usd=100, rev_usd=500)],
                GOOGLE: [_m(GOOGLE, spend_usd=100, rev_usd=100)]}
        decs = agent.daily_decisions(3, plans, hist)  # 3-2=1 <= 2 冷却中
        assert not any(d.action in ("lower_budget", "raise_budget") for d in decs)

    def test_small_gap_no_realloc(self):
        agent = BiddingAgent(BidConfig(realloc_roas_gap=0.5))
        plans = {META: PlatformPlan(META, 8 * MICROS, 100 * MICROS),
                 GOOGLE: PlatformPlan(GOOGLE, 8 * MICROS, 100 * MICROS)}
        hist = {META: [_m(META, spend_usd=100, rev_usd=300)],
                GOOGLE: [_m(GOOGLE, spend_usd=100, rev_usd=280)]}  # gap 小
        decs = agent.daily_decisions(3, plans, hist)
        assert not any(d.action in ("lower_budget", "raise_budget") for d in decs)


class TestApply:
    def test_copilot_only_applies_approved(self):
        agent = BiddingAgent()
        plans = {META: PlatformPlan(META, 8 * MICROS, 100 * MICROS)}
        d = Decision(1, "meta_sim", "lower_bid", "r", "e", 0.7,
                     old_value=8 * MICROS, new_value=6 * MICROS)
        agent.apply(plans, [d], auto_approve=False)  # 未批准
        assert plans[META].bid_micros == 8 * MICROS and not d.applied
        d.approved = True
        agent.apply(plans, [d], auto_approve=False)
        assert plans[META].bid_micros == 6 * MICROS and d.applied

    def test_auto_approve_applies_all(self):
        agent = BiddingAgent()
        plans = {META: PlatformPlan(META, 8 * MICROS, 100 * MICROS)}
        d = Decision(1, "meta_sim", "lower_bid", "r", "e", 0.7,
                     old_value=8 * MICROS, new_value=6 * MICROS)
        agent.apply(plans, [d], auto_approve=True)
        assert plans[META].bid_micros == 6 * MICROS and d.applied


# ---------------------------------------------------------------------------
# CTR 预排序
# ---------------------------------------------------------------------------

class TestCtrRanking:
    def _variants(self, n):
        return [CopyVariant(platform="meta", variant_id=f"v{i}",
                            headline=f"Headline number {i}") for i in range(n)]

    def test_ranks_by_score_and_weights_top_heavy(self):
        variants = self._variants(5)
        scores = [CtrScore(variant_id=f"v{i}", visual_appeal=x, clarity=x,
                           cta_strength=x, platform_fit=x)
                  for i, x in enumerate([0.9, 0.2, 0.7, 0.3, 0.5])]
        llm = FakeLLM([CtrScoreBatch(scores=scores)])
        ranked = rank_creatives(variants, llm, top_k=3, top_share=0.70)
        # 降序
        assert [r.variant.variant_id for r in ranked] == ["v0", "v2", "v4",
                                                          "v3", "v1"]
        # 权重归一 & Top3 拿约 70%
        assert sum(r.budget_weight for r in ranked) == pytest.approx(1.0)
        top3 = sum(r.budget_weight for r in ranked[:3])
        assert top3 == pytest.approx(0.70, abs=1e-6)

    def test_missing_score_gets_neutral(self):
        variants = self._variants(2)
        llm = FakeLLM([CtrScoreBatch(scores=[
            CtrScore(variant_id="v0", visual_appeal=0.8, clarity=0.8,
                     cta_strength=0.8, platform_fit=0.8)])])  # v1 缺分
        ranked = rank_creatives(variants, llm)
        assert ranked[-1].variant.variant_id == "v1"
        assert ranked[-1].score == 0.5

    def test_empty_input(self):
        assert rank_creatives([], FakeLLM([])) == []


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------

class TestDiagnosis:
    def test_best_worst_and_blended(self):
        today = {META: _m(META, spend_usd=100, rev_usd=500),   # ROAS 5
                 GOOGLE: _m(GOOGLE, spend_usd=100, rev_usd=100)}  # ROAS 1
        rpt = diagnose_day(1, today, {p: [m] for p, m in today.items()}, [])
        assert rpt.best_platform == "meta_sim"
        assert rpt.worst_platform == "google_sim"
        assert rpt.blended_roas == pytest.approx(3.0)  # 600/200

    def test_empty_day(self):
        rpt = diagnose_day(0, {}, {}, [])
        assert "无投放数据" in rpt.observations[0]

    def test_flags_cpa_overrun(self):
        today = {META: _m(META, convs=2, spend_usd=100, rev_usd=40)}  # CPA $50
        rpt = diagnose_day(1, today, {META: [today[META]]}, [],
                           target_cpa_usd=25.0)
        assert any("超标" in o for o in rpt.observations)

    def test_no_llm_leaves_narrative_empty(self):
        today = {META: _m(META, spend_usd=100, rev_usd=300)}
        rpt = diagnose_day(1, today, {META: [today[META]]}, [])
        assert rpt.narrative == ""

    def test_llm_populates_narrative(self):
        today = {META: _m(META, spend_usd=100, rev_usd=500),
                 GOOGLE: _m(GOOGLE, spend_usd=100, rev_usd=100)}
        llm = FakeLLM([DiagnosisNarrative(
            narrative="meta 效率领先", next_step="向 meta 倾斜")])
        rpt = diagnose_day(1, today, {p: [m] for p, m in today.items()}, [],
                           llm=llm)
        assert "meta 效率领先" in rpt.narrative
        assert "建议：向 meta 倾斜" in rpt.narrative
        assert len(llm.calls) == 1

    def test_narrative_lang_en_separator(self):
        """lang=en 时建议用英文分隔符 'Suggestion:'。"""
        today = {META: _m(META, spend_usd=100, rev_usd=500),
                 GOOGLE: _m(GOOGLE, spend_usd=100, rev_usd=100)}
        llm = FakeLLM([DiagnosisNarrative(narrative="meta leads",
                                          next_step="shift to meta")])
        rpt = diagnose_day(1, today, {p: [m] for p, m in today.items()}, [],
                           llm=llm, lang="en")
        assert "meta leads" in rpt.narrative and "Suggestion:" in rpt.narrative

    def test_llm_failure_does_not_break_report(self):
        """LLM 偶发失败: 结构化诊断照常，narrative 退回确定性兜底句（不依赖 LLM）。"""
        today = {META: _m(META, spend_usd=100, rev_usd=300)}
        llm = FakeLLM([])   # 耗尽 -> complete_with_retry 抛错 -> 被吞 -> 兜底复述
        rpt = diagnose_day(1, today, {META: [today[META]]}, [], llm=llm)
        assert rpt.best_platform == "meta_sim"   # 结构化结论仍在
        assert rpt.narrative                     # 非空：兜底句而非空串
        assert "meta_sim" in rpt.narrative       # 兜底句复述了结构化结论


# ---------------------------------------------------------------------------
# LangGraph 编排（集成 + 复现性）
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_end_to_end_7_days(self):
        orch = AdPilotOrchestrator(PilotConfig(total_days=7))
        final = orch.run()
        assert len(final["diagnoses"]) == 7
        for p in (META, GOOGLE, TIKTOK):
            assert len(final["history"][p]) == 7
            assert final["diagnoses"][-1].day_index == 6
        # 决策日志非空且含初始出价
        assert any(d.action == "set_initial_bid" for d in orch.log.entries)

    def test_adjustments_take_effect(self):
        """W4 验收核心: 自动调整确实改变了投放参数（指标可见变化）。"""
        orch = AdPilotOrchestrator(PilotConfig(total_days=7))
        orch.run()
        applied = [d for d in orch.log.entries
                   if d.applied and d.action != "hold"
                   and d.action != "set_initial_bid"]
        assert applied, "7 天内应至少有一次生效的自动调整"
        # 至少一次预算再分配或出价调整改变了数值
        assert any(d.old_value != d.new_value for d in applied
                   if d.old_value is not None)

    def test_determinism(self):
        f1 = AdPilotOrchestrator(PilotConfig(total_days=5, seed=7)).run()
        f2 = AdPilotOrchestrator(PilotConfig(total_days=5, seed=7)).run()
        s1 = [(p.value, m.impressions, m.clicks, m.conversions)
              for p, h in f1["history"].items() for m in h]
        s2 = [(p.value, m.impressions, m.clicks, m.conversions)
              for p, h in f2["history"].items() for m in h]
        assert s1 == s2

    def test_diagnosis_llm_narrates_each_day(self):
        """诊断 LLM 复述接入编排: 每天一段 narrative。"""
        days = 3
        narrations = [DiagnosisNarrative(narrative=f"day {i} 诊断",
                                         next_step="调预算") for i in range(days)]
        llm = FakeLLM(narrations)
        cfg = PilotConfig(total_days=days, diagnosis_llm=llm)
        final = AdPilotOrchestrator(cfg).run()
        assert all(rpt.narrative for rpt in final["diagnoses"])
        assert len(llm.calls) == days

    def test_creative_qualities_flow_to_ads(self):
        """CTR 预排序分 -> creative_quality: 高质量应拿到更高 CTR。"""
        cfg = PilotConfig(total_days=3,
                          creative_qualities={META: [0.95], GOOGLE: [0.9],
                                              TIKTOK: [0.9]})
        orch = AdPilotOrchestrator(cfg)
        final = orch.run()
        assert len(final["history"][META]) == 3
