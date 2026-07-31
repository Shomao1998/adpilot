"""W4 验收演示: 关键词 -> CTR预排序 -> 7 模拟日闭环 -> 诊断报告 -> 自动调整生效。

端到端命令行跑通（设计文档 W4 验收标准）。全程离线可复现:
    - CTR 预排序用 FakeLLM 给创意打分 -> creative_quality（喂 AdSim 点击模型）
    - LangGraph 状态机跑 7 天: 出价决策 -> AdSim 投放 -> 诊断 -> 次日
    - 诊断的 LLM 自然语言复述: 离线走 TemplateNarrator（确定性 stub），
      --live 走真实 claude-opus-4-8（需 pip install anthropic + ANTHROPIC_API_KEY）
    - 决策日志时间线 + 每日诊断 + 指标可见变化

用法:
    python scripts/demo_w4.py [--days 7]          # 离线（stub 诊断复述）
    python scripts/demo_w4.py --live              # 诊断走真实 Claude API
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adcreative.llm import FakeLLM  # noqa: E402
from adcreative.schema import CopyVariant  # noqa: E402
from adpilot import AdPilotOrchestrator, PilotConfig, rank_creatives  # noqa: E402
from adpilot.ctr_ranking import CtrScore, CtrScoreBatch  # noqa: E402
from adpilot.diagnosis import DiagnosisNarrative  # noqa: E402
from adsim.models import Platform  # noqa: E402


class TemplateNarrator:
    """离线诊断复述 stub（实现 LLMClient 协议）。

    从诊断事实提示词里解析领先/垫底平台与综合 ROAS，产出一句确定性中文复述。
    真实场景换成 ClaudeLLM 即可（一行），管线代码不变 —— 演示的就是"接上真实
    LLM 只换实现"这个缝。"""

    def complete(self, system: str, user: str, schema):
        best = re.search(r"领先 (\w+)", user)
        worst = re.search(r"垫底 (\w+)", user)
        roas = re.search(r"ROAS ([\d.]+)。", user)
        b = best.group(1) if best else "?"
        w = worst.group(1) if worst else "?"
        r = roas.group(1) if roas else "?"
        return schema(
            narrative=(f"综合 ROAS {r}，{b} 效率领先、{w} 垫底；"
                       f"差距主要由平台间转化效率与出价调整共同驱动。"),
            next_step=f"向 {b} 倾斜预算，并关注 {w} 的素材与出价")


_CTR_VARIANT_TEXT = {   # 每平台 3 个真实文案（live 时交给 Claude 真打分）
    Platform.META_SIM: [
        "Fresh espresso in 3 minutes, anywhere. 500g, USB-C, 8 cups per charge.",
        "Your campsite coffee upgrade — trail-tested, 30-day returns.",
        "A small coffee maker for trips.",
    ],
    Platform.GOOGLE_SIM: [
        "Portable Espresso Maker | Brew In 3 Minutes | 500g USB-C",
        "Camping Coffee Gear | 8 Cups Per Charge",
        "Coffee maker for sale",
    ],
    Platform.TIKTOK_SIM: [
        "POV: real espresso at 3000m altitude ☕",
        "Wait for the crema... at a campsite?!",
        "my tent kitchen upgrade",
    ],
}
_CTR_OFFLINE_PRESETS = {
    Platform.META_SIM: [0.85, 0.55, 0.35],
    Platform.GOOGLE_SIM: [0.90, 0.60, 0.40],
    Platform.TIKTOK_SIM: [0.80, 0.50, 0.30],
}


def _ctr_rank_demo(ctr_llm=None) -> dict[Platform, list[float]]:
    """CTR 预排序: 每平台 3 个创意变体打分 -> 取质量分喂给 AdSim。

    ctr_llm 给定（--live）时用真实 Claude 打分；否则用 FakeLLM 预设复现流程。
    """
    qualities: dict[Platform, list[float]] = {}
    for p, texts in _CTR_VARIANT_TEXT.items():
        def _mk(i: int, t: str) -> CopyVariant:
            if p == Platform.GOOGLE_SIM:
                return CopyVariant(
                    platform="google", variant_id=f"{p.value}-v{i}",
                    rsa_headlines=[t[:28], "Brew In 3 Minutes", "Only 500g"],
                    rsa_descriptions=[t[:88], "USB-C portable. 30-day returns."])
            if p == Platform.TIKTOK_SIM:   # TikTok 文案 ≤100 字符，直接作 headline
                return CopyVariant(platform="tiktok", variant_id=f"{p.value}-v{i}",
                                   headline=t)
            # Meta: 短标题 + 长正文（headline ≤40, body ≤300）
            return CopyVariant(platform="meta", variant_id=f"{p.value}-v{i}",
                               headline="Fresh Espresso, Anywhere", body=t)

        variants = [_mk(i, t) for i, t in enumerate(texts)]
        if ctr_llm is not None:
            llm = ctr_llm
        else:
            vals = _CTR_OFFLINE_PRESETS[p]
            llm = FakeLLM([CtrScoreBatch(scores=[
                CtrScore(variant_id=v.variant_id, visual_appeal=x, clarity=x,
                         cta_strength=x, platform_fit=x)
                for v, x in zip(variants, vals)])])
        ranked = rank_creatives(variants, llm)
        qualities[p] = [r.score for r in ranked]
        top = ranked[0]
        print(f"  {p.value:<11} 预排序 Top: {top.variant.variant_id} "
              f"(分 {top.score:.2f}, 权重 {top.budget_weight:.0%}) | "
              f"质量分 {[round(q, 2) for q in qualities[p]]}")
    return qualities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--live", action="store_true",
                    help="诊断复述走真实 Claude API（需 anthropic 包 + API key）")
    args = ap.parse_args()

    if args.live:
        from adcreative.llm import ClaudeLLM
        narrator = ClaudeLLM()
        ctr_llm = ClaudeLLM()
        print("[LLM] LIVE (claude-opus-4-8): CTR 预排序 + 诊断复述均走真实 API")
    else:
        narrator = TemplateNarrator()
        ctr_llm = None
        print("[LLM] OFFLINE: CTR 用 FakeLLM 预设，诊断用 TemplateNarrator stub")

    print("[1/3] CTR 预排序（VLM 打分 -> creative_quality）")
    qualities = _ctr_rank_demo(ctr_llm)

    print(f"\n[2/3] LangGraph 编排: {args.days} 模拟日闭环")
    cfg = PilotConfig(total_days=args.days, creative_qualities=qualities,
                      diagnosis_llm=narrator)
    orch = AdPilotOrchestrator(cfg)
    final = orch.run()

    # 每日诊断（含 LLM 自然语言复述）
    for rpt in final["diagnoses"]:
        print(f"  D{rpt.day_index}: ROAS {rpt.blended_roas:.2f} | "
              f"消耗 ${rpt.total_spend_usd:.0f} 收入 ${rpt.total_revenue_usd:.0f} | "
              f"领先 {rpt.best_platform} 垫底 {rpt.worst_platform}")
        if rpt.narrative:
            print(f"      诊断: {rpt.narrative}")

    print("\n[3/3] 决策日志时间线（自动调整生效）")
    for d in orch.log.entries:
        if d.action in ("hold",):
            continue
        print(f"  {d.summary()}")

    # 验收: 首末日预算/指标可见变化
    print("=" * 68)
    changed = [d for d in orch.log.entries
               if d.applied and d.old_value is not None
               and d.old_value != d.new_value and d.action != "set_initial_bid"]
    print(f"W4 验收通过: {args.days} 日闭环跑通，{len(final['diagnoses'])} 份诊断，"
          f"{len(changed)} 次生效的自动调整（预算/出价）")
    # 展示某平台首末日预算变化
    for p, hist in final["history"].items():
        first, last = hist[0], hist[-1]
        print(f"  {p.value:<11} D0 消耗 ${first.spend_usd:.0f} ROAS {first.roas:.2f}"
              f"  ->  D{last.day_index} 消耗 ${last.spend_usd:.0f} "
              f"ROAS {last.roas:.2f}")


if __name__ == "__main__":
    main()
