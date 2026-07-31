"""出价 Agent: 可解释规则策略（设计文档 4.6）。

明确不做 RL —— 数据量不支撑 off-policy 学习，且黑盒策略在 Copilot 阶段有信任
问题。规则策略在真实冷启动期也是主流。每条建议带理由/预期/置信度，进决策日志。

规则（每模拟日结束触发）:
    1. 初始出价（D0）: CPM = 目标CPA × 预估CTR × 预估CVR × 1000 × aggressiveness。
       等价于"目标CPA ÷ 预估CVR 得目标CPC，再 × CTR × 1000 换算到 CPM"。
    2. 超成本预警: CPA > 目标×1.5 且过了学习期 -> 降价（每次 ≤20%）。
    3. 有余量提效: CPA < 目标×0.7 且预算花满 -> 提价抢量（≤20%）。
    4. 边际ROAS再分配: 平台间 ROAS 差距 > 阈值 -> 低效平台预算挪给高效平台
       （每次 ≤20%，带冷却期防抖）。
    5. 素材疲劳: CTR 连续下滑 fatigue_window 天 -> 触发创意迭代建议。
    6. 否则 hold。

所有调整幅度硬上限 max_change_pct，且预算再分配保持总量守恒。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from adpilot.decision import Decision
from adpilot.metrics import DayMetrics
from adsim.models import Platform

MICROS = 1_000_000


@dataclass
class PlatformPlan:
    """出价 Agent 操作的可变计划: 一平台的当前出价/预算/状态。"""
    platform: Platform
    bid_micros: int
    budget_micros: int
    status: str = "active"        # active | paused
    last_realloc_day: int = -99   # 上次参与再分配的日 index（冷却用）


@dataclass
class BidConfig:
    target_cpa_usd: float = 25.0
    aggressiveness: float = 0.8       # 初始出价相对盈亏平衡的折扣
    overrun_mult: float = 1.5         # CPA 超过目标此倍数触发降价
    efficient_mult: float = 0.7       # CPA 低于目标此倍数触发提价
    max_change_pct: float = 0.20      # 单次调整幅度上限
    learning_min_conversions: int = 3 # 学习期护栏: 转化不足不据 CPA 决策
    fatigue_window: int = 3           # CTR 连降天数阈值
    realloc_roas_gap: float = 0.5     # 平台间 ROAS 相对差距触发再分配
    realloc_cooldown_days: int = 2    # 再分配冷却
    # 预估漏斗（初始出价用；来自行业 benchmark，与 simulate.DEFAULT_PROFILES 对齐）
    est_ctr: dict[Platform, float] = field(default_factory=lambda: {
        Platform.META_SIM: 0.012, Platform.TIKTOK_SIM: 0.018,
        Platform.GOOGLE_SIM: 0.032})
    est_cvr: dict[Platform, float] = field(default_factory=lambda: {
        Platform.META_SIM: 0.020, Platform.TIKTOK_SIM: 0.008,
        Platform.GOOGLE_SIM: 0.050})
    lang: str = "zh"                  # 决策文本语言（zh/en）


class BiddingAgent:
    def __init__(self, config: BidConfig | None = None):
        self.cfg = config or BidConfig()
        self.en = (self.cfg.lang or "zh").lower() == "en"

    # ---- D0: 初始出价 ----
    def initial_bids(
        self, plans: dict[Platform, PlatformPlan]
    ) -> list[Decision]:
        cfg = self.cfg
        out: list[Decision] = []
        for p, plan in plans.items():
            ctr, cvr = cfg.est_ctr[p], cfg.est_cvr[p]
            cpm_usd = cfg.target_cpa_usd * ctr * cvr * 1000 * cfg.aggressiveness
            new_bid = max(int(cpm_usd * MICROS), 1)
            reason = (f"Target CPA ${cfg.target_cpa_usd:.0f} × est. CTR {ctr:.1%} "
                      f"× CVR {cvr:.1%} → CPM ${cpm_usd:.2f}" if self.en else
                      f"目标CPA ${cfg.target_cpa_usd:.0f} × 预估CTR {ctr:.1%} "
                      f"× CVR {cvr:.1%} 折算 CPM ${cpm_usd:.2f}")
            out.append(Decision(
                day_index=0, platform=p.value, action="set_initial_bid",
                reason=reason,
                expected_impact=("establish cold-start baseline bid" if self.en
                                 else "建立冷启动基线出价"), confidence=0.6,
                old_value=plan.bid_micros, new_value=new_bid))
        return out

    # ---- D>0: 日终调整 ----
    def daily_decisions(
        self,
        day_index: int,
        plans: dict[Platform, PlatformPlan],
        history: dict[Platform, list[DayMetrics]],
    ) -> list[Decision]:
        cfg = self.cfg
        out: list[Decision] = []
        yest = {p: h[-1] for p, h in history.items() if h}

        # 单平台规则: 超成本 / 提效 / 疲劳
        for p, plan in plans.items():
            if plan.status != "active" or p not in yest:
                continue
            m = yest[p]

            # 疲劳: CTR 连降
            if self._ctr_declining(history[p], cfg.fatigue_window):
                out.append(Decision(
                    day_index=day_index, platform=p.value,
                    action="trigger_creative_refresh",
                    reason=(f"CTR declined {cfg.fatigue_window} days running "
                            f"(creative fatigue)" if self.en else
                            f"CTR 连续 {cfg.fatigue_window} 天下滑（素材疲劳）"),
                    expected_impact=("refresh creatives to recover CTR" if self.en
                                     else "换新创意以恢复点击率"), confidence=0.65))
                continue  # 疲劳优先，本平台本日不再动出价

            # 学习期护栏: 转化太少不据 CPA 决策
            if m.conversions < cfg.learning_min_conversions:
                out.append(self._hold(day_index, p,
                    "too few conversions; hold bid through learning phase"
                    if self.en else "转化样本不足，维持出价过学习期"))
                continue

            target_micros = cfg.target_cpa_usd * MICROS
            if m.cpa_micros > target_micros * cfg.overrun_mult:
                new_bid = int(plan.bid_micros * (1 - cfg.max_change_pct))
                out.append(Decision(
                    day_index=day_index, platform=p.value, action="lower_bid",
                    reason=(f"CPA ${m.cpa_usd:.1f} exceeds "
                            f"{cfg.overrun_mult:.1f}× target ${cfg.target_cpa_usd:.0f}"
                            if self.en else
                            f"CPA ${m.cpa_usd:.1f} 超目标 "
                            f"${cfg.target_cpa_usd:.0f} 的 {cfg.overrun_mult:.1f} 倍"),
                    expected_impact=(f"lower bid {cfg.max_change_pct:.0%} to cut CPA"
                                     if self.en else
                                     f"降价 {cfg.max_change_pct:.0%} 压低获客成本"),
                    confidence=0.75,
                    old_value=plan.bid_micros, new_value=new_bid))
            elif (m.cpa_micros < target_micros * cfg.efficient_mult
                  and m.spend_micros >= plan.budget_micros * 0.95):
                new_bid = int(plan.bid_micros * (1 + cfg.max_change_pct))
                out.append(Decision(
                    day_index=day_index, platform=p.value, action="raise_bid",
                    reason=("CPA well below target and budget fully spent — "
                            "room to scale" if self.en else
                            f"CPA ${m.cpa_usd:.1f} 远低于目标且预算花满，有提量空间"),
                    expected_impact=(f"raise bid {cfg.max_change_pct:.0%} for more reach"
                                     if self.en else
                                     f"提价 {cfg.max_change_pct:.0%} 抢更多曝光"),
                    confidence=0.7,
                    old_value=plan.bid_micros, new_value=new_bid))
            else:
                out.append(self._hold(day_index, p,
                    "CPA within target range" if self.en else "CPA 在目标区间内"))

        # 跨平台: 边际 ROAS 预算再分配
        realloc = self._reallocate(day_index, plans, yest)
        if realloc:
            out.extend(realloc)
        return out

    # ---- 应用（Copilot: 仅执行 approved） ----
    def apply(
        self, plans: dict[Platform, PlatformPlan],
        decisions: list[Decision], auto_approve: bool = False,
    ) -> None:
        by_platform = {p.value: p for p in plans}
        for d in decisions:
            if auto_approve:
                d.approved = True
            if not d.approved:
                continue
            plan = plans[by_platform[d.platform]]
            if d.action in ("set_initial_bid", "raise_bid", "lower_bid"):
                plan.bid_micros = int(d.new_value)
            elif d.action in ("raise_budget", "lower_budget"):
                plan.budget_micros = int(d.new_value)
                plan.last_realloc_day = d.day_index
            elif d.action == "pause_ad":
                plan.status = "paused"
            d.applied = True

    # ---- helpers ----
    def _hold(self, day_index: int, p: Platform, reason: str) -> Decision:
        return Decision(day_index=day_index, platform=p.value, action="hold",
                        reason=reason,
                        expected_impact=("no change" if self.en else "无变化"),
                        confidence=0.5)

    @staticmethod
    def _ctr_declining(hist: list[DayMetrics], window: int) -> bool:
        if len(hist) < window + 1:
            return False
        recent = hist[-(window + 1):]
        return all(recent[i].ctr < recent[i - 1].ctr
                   for i in range(1, len(recent)))

    def _reallocate(
        self, day_index: int,
        plans: dict[Platform, PlatformPlan],
        yest: dict[Platform, DayMetrics],
    ) -> list[Decision]:
        cfg = self.cfg
        eligible = [p for p, plan in plans.items()
                    if plan.status == "active" and p in yest
                    and day_index - plan.last_realloc_day > cfg.realloc_cooldown_days]
        if len(eligible) < 2:
            return []
        best = max(eligible, key=lambda p: yest[p].marginal_roas)
        worst = min(eligible, key=lambda p: yest[p].marginal_roas)
        if best is worst:
            return []
        r_best, r_worst = yest[best].marginal_roas, yest[worst].marginal_roas
        if r_best <= 0:
            return []
        gap = (r_best - r_worst) / r_best
        if gap < cfg.realloc_roas_gap:
            return []

        delta = int(plans[worst].budget_micros * cfg.max_change_pct)
        if delta <= 0:
            return []
        worst_new = plans[worst].budget_micros - delta
        best_new = plans[best].budget_micros + delta
        gapstr = (f"{best.value} ROAS {r_best:.2f} vs {worst.value} "
                  f"{r_worst:.2f}, relative gap {gap:.0%}" if self.en else
                  f"{best.value} ROAS {r_best:.2f} vs {worst.value} "
                  f"{r_worst:.2f}，相对差 {gap:.0%}")
        return [
            Decision(day_index=day_index, platform=worst.value,
                     action="lower_budget",
                     reason=(f"Marginal ROAS lagging: {gapstr}" if self.en
                             else f"边际ROAS偏低：{gapstr}"),
                     expected_impact=(f"budget -{cfg.max_change_pct:.0%}, "
                                      f"reallocate to top platform" if self.en else
                                      f"预算 -{cfg.max_change_pct:.0%} 转投高效平台"),
                     confidence=0.7,
                     old_value=plans[worst].budget_micros, new_value=worst_new),
            Decision(day_index=day_index, platform=best.value,
                     action="raise_budget",
                     reason=(f"Marginal ROAS leading: {gapstr}" if self.en
                             else f"边际ROAS领先：{gapstr}"),
                     expected_impact=(f"absorb {cfg.max_change_pct:.0%} budget "
                                      f"from lagging platform" if self.en else
                                      f"承接低效平台转出的 {cfg.max_change_pct:.0%} 预算"),
                     confidence=0.7,
                     old_value=plans[best].budget_micros, new_value=best_new),
        ]
