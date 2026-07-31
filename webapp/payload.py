"""编排结果 -> 看板 JSON 序列化（纯函数，可测）。

把 AdPilotOrchestrator 跑完的 state + 决策日志摊平成前端好消费的结构:
    platforms[]  每平台逐日指标时序（ROAS/CPA/CTR/spend...）
    diagnoses[]  每日诊断（含 LLM narrative）
    decisions[]  决策日志时间线
    summary      全局汇总 + 平台对比（跨平台对比看板的数据源）
金额统一以 USD 浮点给前端（micros 是存储口径，展示层换算，ADR-5）。
"""
from __future__ import annotations

from typing import Any

from adpilot.decision import DecisionLog
from adpilot.metrics import DayMetrics
from adsim.models import Platform


def _metrics_row(m: DayMetrics) -> dict[str, Any]:
    return dict(
        day=m.day_index,
        impressions=m.impressions, clicks=m.clicks, conversions=m.conversions,
        spend_usd=round(m.spend_usd, 2),
        revenue_usd=round(m.revenue_micros / 1_000_000, 2),
        ctr=round(m.ctr, 5), cvr=round(m.cvr, 5),
        cpa_usd=(None if m.conversions == 0 else round(m.cpa_usd, 2)),
        roas=round(m.roas, 3),
    )


def build_dashboard_payload(final_state: dict, log: DecisionLog,
                            target_cpa_usd: float = 25.0) -> dict[str, Any]:
    history: dict[Platform, list[DayMetrics]] = final_state["history"]
    diagnoses = final_state["diagnoses"]

    platforms = []
    for p, rows in history.items():
        series = [_metrics_row(m) for m in rows]
        tot_spend = sum(m.spend_usd for m in rows)
        tot_rev = sum(m.revenue_micros / 1_000_000 for m in rows)
        tot_conv = sum(m.conversions for m in rows)
        platforms.append(dict(
            platform=p.value,
            series=series,
            totals=dict(
                spend_usd=round(tot_spend, 2),
                revenue_usd=round(tot_rev, 2),
                conversions=tot_conv,
                roas=round(tot_rev / tot_spend, 3) if tot_spend else 0.0,
                cpa_usd=round(tot_spend / tot_conv, 2) if tot_conv else None,
            ),
        ))
    platforms.sort(key=lambda x: x["platform"])

    diag = [dict(
        day=r.day_index,
        blended_roas=round(r.blended_roas, 3),
        spend_usd=round(r.total_spend_usd, 2),
        revenue_usd=round(r.total_revenue_usd, 2),
        best_platform=r.best_platform, worst_platform=r.worst_platform,
        observations=list(r.observations),
        narrative=r.narrative,
    ) for r in diagnoses]

    decisions = [dict(
        day=d.day_index, platform=d.platform, action=d.action,
        reason=d.reason, expected_impact=d.expected_impact,
        confidence=round(d.confidence, 2),
        old_value=d.old_value, new_value=d.new_value,
        approved=d.approved, applied=d.applied,
    ) for d in log.entries]

    total_spend = sum(pl["totals"]["spend_usd"] for pl in platforms)
    total_rev = sum(pl["totals"]["revenue_usd"] for pl in platforms)
    summary = dict(
        days=len(diag),
        target_cpa_usd=target_cpa_usd,
        total_spend_usd=round(total_spend, 2),
        total_revenue_usd=round(total_rev, 2),
        blended_roas=round(total_rev / total_spend, 3) if total_spend else 0.0,
        n_decisions=len(decisions),
        n_applied_adjustments=sum(
            1 for d in decisions if d["applied"] and d["action"] not in
            ("hold", "set_initial_bid")),
    )
    return dict(summary=summary, platforms=platforms,
               diagnoses=diag, decisions=decisions)
