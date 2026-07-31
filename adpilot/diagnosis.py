"""评估诊断 Agent（设计文档 4.8）。

每模拟日生成结构化诊断: 哪个平台在涨跌、原因假设（结合当日决策日志）、
下一步建议。核心用规则从指标算出可证伪的观察（涨跌方向、超成本、ROAS 排名），
LLM 只做可选的自然语言复述 —— 保证诊断结论确定、可测，不靠 LLM 编故事。

诊断结论回喂出价 Agent 与创意生成 Agent，完成大闭环。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from adcreative.llm import LLMClient, complete_with_retry
from adpilot.decision import Decision
from adpilot.metrics import DayMetrics
from adsim.models import Platform


class DiagnosisNarrative(BaseModel):
    """LLM 复述的结构化输出。narrative: 自然语言诊断; next_step: 下一步建议。"""
    narrative: str
    next_step: str = ""


DIAG_SYSTEM_ZH = """你是广告投放的诊断分析师。基于给定的当日结构化指标与已采取的\
决策，用 2-3 句中文写出诊断：哪个平台/指标在涨跌、可能原因（结合已采取的调整），\
以及下一步建议。只依据给定事实，不要编造数字。"""

DIAG_SYSTEM_EN = """You are an ad-delivery diagnosis analyst. Based on the given \
structured daily metrics and the actions already taken, write a 2-3 sentence \
diagnosis IN ENGLISH: which platform/metric is rising or falling, the likely \
cause (tie it to the actions taken), and the next-step recommendation. Use only \
the given facts; do not invent numbers. Write every word in English."""


def _diag_system(lang: str) -> str:
    return DIAG_SYSTEM_EN if (lang or "zh").lower() == "en" else DIAG_SYSTEM_ZH


@dataclass
class DiagnosisReport:
    day_index: int
    observations: list[str] = field(default_factory=list)
    best_platform: str | None = None      # 当日 ROAS 最高
    worst_platform: str | None = None
    total_spend_usd: float = 0.0
    total_revenue_usd: float = 0.0
    blended_roas: float = 0.0
    narrative: str = ""                    # LLM 可选复述

    def text(self) -> str:
        lines = [f"== 诊断 D{self.day_index} =="]
        lines += [f"  • {o}" for o in self.observations]
        lines.append(f"  综合: 消耗 ${self.total_spend_usd:.0f} / 收入 "
                     f"${self.total_revenue_usd:.0f} / ROAS {self.blended_roas:.2f}")
        if self.narrative:
            lines.append(f"  诊断: {self.narrative}")
        return "\n".join(lines)


def diagnose_day(
    day_index: int,
    today: dict[Platform, DayMetrics],
    history: dict[Platform, list[DayMetrics]],
    decisions: list[Decision],
    target_cpa_usd: float = 25.0,
    llm: LLMClient | None = None,
    lang: str = "zh",
) -> DiagnosisReport:
    """从当日指标 + 历史 + 决策日志产出结构化诊断（纯规则，确定性）。

    llm 给定时追加一段 LLM 自然语言复述（可选层，失败不影响结构化结论）——
    结构化 observations 始终确定、可测；narrative 只是把它翻成人话。
    """
    en = (lang or "zh").lower() == "en"
    rpt = DiagnosisReport(day_index=day_index)
    if not today:
        rpt.observations.append("No delivery data" if en else "无投放数据")
        return rpt

    total_spend = sum(m.spend_micros for m in today.values())
    total_rev = sum(m.revenue_micros for m in today.values())
    rpt.total_spend_usd = total_spend / 1_000_000
    rpt.total_revenue_usd = total_rev / 1_000_000
    rpt.blended_roas = total_rev / total_spend if total_spend else 0.0

    ranked = sorted(today.items(), key=lambda kv: kv[1].roas, reverse=True)
    rpt.best_platform = ranked[0][0].value
    rpt.worst_platform = ranked[-1][0].value

    for p, m in today.items():
        # 环比方向
        hist = history.get(p, [])
        trend = ""
        if len(hist) >= 2:
            prev = hist[-2]
            d_roas = m.roas - prev.roas
            arrow = '↑' if d_roas >= 0 else '↓'
            trend = (f", ROAS {arrow}{abs(d_roas):.2f}" if en
                     else f"，ROAS {arrow}{abs(d_roas):.2f}")
        cpa_flag = ""
        if m.conversions > 0 and m.cpa_usd > target_cpa_usd * 1.5:
            cpa_flag = (f", CPA ${m.cpa_usd:.0f} over target" if en
                        else f"，CPA ${m.cpa_usd:.0f} 超标")
        rpt.observations.append(
            f"{p.value}: CTR {m.ctr:.2%} CVR {m.cvr:.2%} ROAS {m.roas:.2f}"
            f"{trend}{cpa_flag}")

    # 结合决策日志给下一步建议
    actions = [d for d in decisions if d.action != "hold"]
    if actions:
        taken = "; ".join(f"{d.platform} {d.action}" for d in actions)
        rpt.observations.append((f"Actions taken: {taken}" if en
                                 else "已采取: " + taken))
    rpt.observations.append(
        (f"ROAS top {rpt.best_platform}, bottom {rpt.worst_platform}; "
         f"next step leans toward shifting budget to {rpt.best_platform}" if en
         else f"ROAS 领先 {rpt.best_platform}、垫底 {rpt.worst_platform}，"
              f"下一步倾向向 {rpt.best_platform} 倾斜预算"))

    if llm is not None:
        rpt.narrative = _narrate(rpt, llm, lang)
    return rpt


def _narrate(rpt: DiagnosisReport, llm: LLMClient, lang: str = "zh") -> str:
    """把结构化诊断交给 LLM 复述成自然语言。失败则返回空串（不阻断闭环）。

    lang 透传两路: LLM 走系统提示的语言指令；stub（TemplateNarrator）自带语言。
    """
    from adcreative.llm import has_cjk, lang_note

    en = (lang or "zh").lower() == "en"
    facts = "\n".join(rpt.observations)
    if en:
        user = (f"Day {rpt.day_index} delivery diagnosis. Facts:\n{facts}\n"
                f"Blended spend ${rpt.total_spend_usd:.0f}, revenue "
                f"${rpt.total_revenue_usd:.0f}, ROAS {rpt.blended_roas:.2f}.")
    else:
        user = (f"第 {rpt.day_index} 天投放诊断，事实如下：\n{facts}\n"
                f"综合消耗 ${rpt.total_spend_usd:.0f}、收入 "
                f"${rpt.total_revenue_usd:.0f}、ROAS {rpt.blended_roas:.2f}。")
    system = lang_note(lang) + _diag_system(lang)
    try:
        out = complete_with_retry(llm, system, user, DiagnosisNarrative)
    except Exception:
        return _fallback_narrative(rpt, en)  # 失败不阻断闭环
    text = out.narrative.strip()
    if out.next_step:
        sep = " Suggestion: " if en else " 建议："
        text += f"{sep}{out.next_step.strip()}"
    # 英文模式硬兜底：LLM 仍漏中文时，退回确定性英文句，保证全平台语言统一。
    if en and has_cjk(text):
        return _fallback_narrative(rpt, en)
    return text


def _fallback_narrative(rpt: DiagnosisReport, en: bool) -> str:
    """不依赖 LLM 的确定性复述。英文模式下保证零中文；也用于 LLM 失败兜底。"""
    if en:
        return (f"Blended ROAS {rpt.blended_roas:.2f} on ${rpt.total_spend_usd:.0f} "
                f"spend / ${rpt.total_revenue_usd:.0f} revenue. {rpt.best_platform} "
                f"leads on ROAS while {rpt.worst_platform} lags. Suggestion: shift "
                f"budget toward {rpt.best_platform}.")
    return (f"综合 ROAS {rpt.blended_roas:.2f}，消耗 ${rpt.total_spend_usd:.0f}、"
            f"收入 ${rpt.total_revenue_usd:.0f}。{rpt.best_platform} ROAS 领先、"
            f"{rpt.worst_platform} 垫底。建议：向 {rpt.best_platform} 倾斜预算。")
