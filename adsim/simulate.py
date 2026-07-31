"""模拟日引擎：把拍卖引擎、点击/转化模型串成一天的事件流。

流程（每个竞价机会）:
    竞价机会 -> 拍卖(二价/GSP) -> AuctionEvent(必写, ground truth)
        -> won? -> 点击采样 -> ClickEvent -> 转化采样(带延迟) -> ConversionEvent

关键口径:
    - 全程美元计价 (ADR-5)。事件表价格列统一 CPM micros (1e-6 USD / 千次曝光)；
      每次曝光的实际扣费 = paying_cpm / 1000，预算按此累计。
    - 点击率 = base_ctr(平台) × 质量乘数(0.5 + creative_quality) × 日噪声 × 槽位衰减。
      日噪声是刻意的：CTR 预排序分数与真实点击率之间必须有差距，评估环节才有
      真东西可诊断（设计文档 4.7）。
    - 转化延迟 ~ LogNormal(小时)。默认参数下约 3-4% 转化落在 7 天窗口之外 ——
      这是三平台归因口径差异(7d vs 30d)可观测的前提。
    - 预算控制用朴素贪心（花完即停）。真实平台有 pacing 算法，这里刻意简化，
      并把"学习期/pacing"留给出价 Agent 层面讨论。

复现性: 引擎持有自己的 rng，同一 seed + 同一输入 -> 完全相同的事件流。

TODO(W2+): base_ctr/base_cvr 目前取公开行业 benchmark 量级（WordStream 等），
    待 Criteo/Avazu 拟合脚本完成后替换为数据驱动的分布采样。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Mapping, Sequence

import numpy as np
from sqlalchemy.orm import Session

from adsim.auction import GSPAuction, MarketModel, SecondPriceAuction
from adsim.models import (
    Ad,
    AdGroup,
    AuctionEvent,
    AuctionMechanism,
    Campaign,
    ClickEvent,
    ConversionEvent,
    Platform,
)

MICROS = 1_000_000


def usd_to_micros(x: float) -> int:
    return int(round(x * MICROS))


# 默认小时流量曲线：早低、午平、晚高峰（20-22 点），归一化为占比
_DEFAULT_HOURLY = np.array(
    [1.5, 1.0, 0.7, 0.5, 0.5, 0.7, 1.2, 2.0, 3.0, 3.8, 4.2, 4.5,
     4.8, 4.6, 4.4, 4.5, 4.8, 5.2, 5.8, 6.5, 7.2, 7.0, 5.5, 3.2]
)
_DEFAULT_HOURLY = _DEFAULT_HOURLY / _DEFAULT_HOURLY.sum()


@dataclass(frozen=True)
class PlatformProfile:
    """一个平台的流量与用户行为参数。

    量级参考公开行业 benchmark 的相对水位（与设计文档 4.7 的定性描述一致）:
        Meta:   量大、CTR/CVR 中等
        TikTok: 量大、CTR 高、CVR 低（冲动点击多、转化差）
        Google: 量小（搜索意图）、CTR/CVR 都高
    """
    platform: Platform
    mechanism: AuctionMechanism
    daily_opportunities: int
    base_ctr: float
    base_cvr: float
    # 转化延迟 LogNormal(小时): conv_delay_mu=2.0, sigma=1.8
    # -> 中位数 ~7.4h, P(>7天) ~4% —— 归因窗口差异的可观测来源
    conv_delay_mu: float = 2.0
    conv_delay_sigma: float = 1.8
    # 订单金额 (AOV) LogNormal(USD): 默认均值 ~$60
    aov_log_mu: float = 4.014
    aov_log_sigma: float = 0.4
    ctr_noise_sigma: float = 0.15   # 每广告每日的 CTR 乘性噪声
    hourly_traffic: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.hourly_traffic is None:
            object.__setattr__(self, "hourly_traffic", _DEFAULT_HOURLY)


DEFAULT_PROFILES: dict[Platform, PlatformProfile] = {
    Platform.META_SIM: PlatformProfile(
        platform=Platform.META_SIM,
        mechanism=AuctionMechanism.SECOND_PRICE,
        daily_opportunities=30_000, base_ctr=0.012, base_cvr=0.020,
    ),
    Platform.TIKTOK_SIM: PlatformProfile(
        platform=Platform.TIKTOK_SIM,
        mechanism=AuctionMechanism.SECOND_PRICE,
        daily_opportunities=36_000, base_ctr=0.018, base_cvr=0.008,
    ),
    Platform.GOOGLE_SIM: PlatformProfile(
        platform=Platform.GOOGLE_SIM,
        mechanism=AuctionMechanism.GSP,
        daily_opportunities=6_000, base_ctr=0.032, base_cvr=0.050,
    ),
}


@dataclass(frozen=True)
class DayStats:
    """一天模拟的汇总（金额为 micros USD）。返回给调用方做冒烟检查，
    正式指标一律从事件表重新聚合（事件流是唯一事实来源）。"""
    auctions: int
    wins: int
    clicks: int
    conversions: int
    spend_micros: int
    revenue_micros: int


class AdSimEngine:
    """无外部状态的模拟引擎：markets/profiles 注入，事件写入传入的 Session。"""

    def __init__(
        self,
        markets: Mapping[Platform, MarketModel],
        profiles: Mapping[Platform, PlatformProfile] | None = None,
        seed: int = 0,
    ):
        self.markets = markets
        self.profiles = dict(profiles or DEFAULT_PROFILES)
        self.rng = np.random.default_rng(seed)

    def run_day(
        self,
        session: Session,
        sim_date: date,
        campaign: Campaign,
        ad_group: AdGroup,
        ads: Sequence[Ad],
    ) -> DayStats:
        platform = campaign.platform
        profile = self.profiles[platform]
        market = self.markets[platform]
        rng = self.rng

        active_ads = [a for a in ads if a.status == "active"]
        if not active_ads:
            return DayStats(0, 0, 0, 0, 0, 0)

        second_price = SecondPriceAuction(market)
        gsp = GSPAuction(market)
        use_gsp = profile.mechanism == AuctionMechanism.GSP

        bid_usd_cpm = ad_group.bid_micros / MICROS
        budget_micros = campaign.daily_budget_micros

        # 每广告每日 CTR 噪声：预排序分数 -> 真实点击率之间刻意保留的差距
        day_noise = {
            a.id: float(rng.lognormal(0.0, profile.ctr_noise_sigma))
            for a in active_ads
        }

        hourly_opps = rng.multinomial(
            profile.daily_opportunities, profile.hourly_traffic
        )

        stats = dict(auctions=0, wins=0, clicks=0, conversions=0,
                     spend_micros=0, revenue_micros=0)
        events: list = []
        budget_exhausted = False

        for hour in range(24):
            if budget_exhausted:
                break
            for _ in range(int(hourly_opps[hour])):
                if stats["spend_micros"] >= budget_micros:
                    budget_exhausted = True
                    break
                ad = active_ads[int(rng.integers(0, len(active_ads)))]
                ts = datetime(sim_date.year, sim_date.month, sim_date.day,
                              hour, int(rng.integers(0, 60)),
                              int(rng.integers(0, 60)))

                if use_gsp:
                    outcome = gsp.run(bid_usd_cpm, ad.creative_quality,
                                      hour, rng)
                else:
                    outcome = second_price.run(bid_usd_cpm, hour, rng)

                stats["auctions"] += 1
                # 先取整落库，扣费从同一个整数推导 —— 避免 float/int 两条
                # 取整路径产生分位级偏差（micros 存在的意义）
                paying_cpm_micros = usd_to_micros(outcome.paying_price)
                auction_ev = AuctionEvent(
                    platform=platform, ad_id=ad.id, sim_ts=ts,
                    hour_of_day=hour,
                    bid_micros=ad_group.bid_micros,
                    floor_micros=usd_to_micros(market.floor),
                    market_price_micros=usd_to_micros(outcome.market_price),
                    won=outcome.won,
                    paying_micros=paying_cpm_micros,
                )
                events.append(auction_ev)
                if not outcome.won:
                    continue

                stats["wins"] += 1
                # 每次曝光实际扣费 = CPM 价 / 1000
                stats["spend_micros"] += int(round(paying_cpm_micros / 1000))

                # --- 点击模型 ---
                slot_factor = 0.85 ** max(outcome.slot, 0)  # GSP 槽位衰减
                p_click = (profile.base_ctr
                           * (0.5 + ad.creative_quality)
                           * day_noise[ad.id]
                           * slot_factor)
                if rng.random() >= min(p_click, 0.95):
                    continue

                stats["clicks"] += 1
                click_ts = ts + timedelta(
                    seconds=float(min(rng.exponential(30.0), 300.0)))
                click_ev = ClickEvent(platform=platform, ad_id=ad.id,
                                      sim_ts=click_ts)
                # 需要先拿到 auction_event 主键 -> flush 见下方批量处理
                click_ev._auction_ref = auction_ev  # type: ignore[attr-defined]
                events.append(click_ev)

                # --- 转化模型 ---
                if rng.random() >= profile.base_cvr:
                    continue
                delay_h = float(rng.lognormal(profile.conv_delay_mu,
                                              profile.conv_delay_sigma))
                conv_ts = click_ts + timedelta(hours=delay_h)
                value = usd_to_micros(
                    float(rng.lognormal(profile.aov_log_mu,
                                        profile.aov_log_sigma)))
                stats["conversions"] += 1
                stats["revenue_micros"] += value
                conv_ev = ConversionEvent(
                    platform=platform, ad_id=ad.id, sim_ts=conv_ts,
                    click_ts=click_ts, delay_hours=delay_h,
                    value_micros=value,
                )
                conv_ev._click_ref = click_ev  # type: ignore[attr-defined]
                events.append(conv_ev)

        # 批量写入：先落拍卖事件拿主键，再回填 click/conversion 的外键
        auction_evs = [e for e in events if isinstance(e, AuctionEvent)]
        click_evs = [e for e in events if isinstance(e, ClickEvent)]
        conv_evs = [e for e in events if isinstance(e, ConversionEvent)]
        session.add_all(auction_evs)
        session.flush()
        for c in click_evs:
            c.auction_event_id = c._auction_ref.id
        session.add_all(click_evs)
        session.flush()
        for v in conv_evs:
            v.click_event_id = v._click_ref.id
        session.add_all(conv_evs)
        session.commit()

        return DayStats(**stats)
