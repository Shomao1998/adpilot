"""归一化指标层（DayMetrics）。

一个平台一天的可比化指标。金额一律 micros USD (ADR-5)。派生指标（CTR/CVR/
CPA/ROAS）以属性给出，避免落库冗余、也避免除零。出价 Agent 与诊断 Agent
都消费这个对象 —— 它是"归一化"之后的统一口径（时区/货币已由 reporting 层拉齐）。
"""
from __future__ import annotations

from dataclasses import dataclass

from adsim.models import Platform

MICROS = 1_000_000


@dataclass(frozen=True)
class DayMetrics:
    platform: Platform
    day_index: int
    impressions: int
    clicks: int
    conversions: int
    spend_micros: int
    revenue_micros: int

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def cpa_micros(self) -> float:
        """每转化成本；无转化时返回 inf（下游据此判超成本）。"""
        return self.spend_micros / self.conversions if self.conversions else float("inf")

    @property
    def roas(self) -> float:
        return self.revenue_micros / self.spend_micros if self.spend_micros else 0.0

    @property
    def spend_usd(self) -> float:
        return self.spend_micros / MICROS

    @property
    def cpa_usd(self) -> float:
        return self.cpa_micros / MICROS

    @property
    def marginal_roas(self) -> float:
        """边际 ROAS 的可观测代理: 本期 revenue/spend。真实边际需要多点拟合，
        demo 用当期 ROAS 作近似 —— 预算再分配比较的是平台间相对水位。"""
        return self.roas
