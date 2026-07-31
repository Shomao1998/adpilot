"""决策日志（设计文档 4.6 / 10.2）。

Copilot 模式的核心资产: 每条 Agent 建议都带「理由 + 预期影响 + 置信度」，
用户批准后才 applied。演示可切"自动放权"（auto_approve）跑全自动闭环。
决策全部留痕 —— 面试叙事里"知道边界、可解释"的证据就在这张表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal[
    "set_initial_bid", "raise_bid", "lower_bid",
    "raise_budget", "lower_budget", "pause_ad", "trigger_creative_refresh",
    "hold",
]


@dataclass
class Decision:
    day_index: int
    platform: str
    action: Action
    reason: str                 # 为什么这么建议
    expected_impact: str        # 预期影响
    confidence: float           # [0,1]
    old_value: float | None = None   # 调整前（micros 或计数）
    new_value: float | None = None   # 调整后
    approved: bool = False      # Copilot: 用户是否批准
    applied: bool = False       # 是否已执行

    def summary(self) -> str:
        chg = ""
        if self.old_value is not None and self.new_value is not None:
            chg = f" [{self.old_value:.0f} -> {self.new_value:.0f}]"
        state = "已执行" if self.applied else ("已批准" if self.approved else "待批")
        return (f"D{self.day_index} {self.platform} {self.action}{chg} "
                f"(conf={self.confidence:.0%}, {state}): {self.reason}")


@dataclass
class DecisionLog:
    entries: list[Decision] = field(default_factory=list)

    def add(self, d: Decision) -> Decision:
        self.entries.append(d)
        return d

    def for_day(self, day_index: int) -> list[Decision]:
        return [d for d in self.entries if d.day_index == day_index]

    def applied(self) -> list[Decision]:
        return [d for d in self.entries if d.applied]
