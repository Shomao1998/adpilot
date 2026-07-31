"""报表生成器：从 ground truth 事件流物化"平台自己报的数"。

这是 ADR-4（三平台故意不同口径）的落地点。同一份事件流，三平台报出三份
对不上的日报表，差异来自三个正交的口径维度:

    1. 归因窗口: 点击后多少天内的转化算数。
       meta_sim 7d / google_sim 30d / tiktok_sim 7d。
       延迟 10 天的转化 -> Google 记入、Meta/TikTok 丢弃。
    2. 报表时区: "一天"的边界在哪。
       meta_sim 用 America/Los_Angeles、google_sim 用 America/New_York、
       tiktok_sim 用 UTC。UTC 早上的事件在 LA 口径下属于"昨天"。
    3. 转化记账日 (conversion_date_basis): 转化算在哪一天头上。
       click_date（Meta/Google 口径: 归到点击发生日，导致历史报表回填）
       vs conv_date（TikTok 口径: 归到转化发生日）。

回填 (backfill) 语义:
    click_date 口径下，今天发生的转化会改写"三天前"那一行报表 —— 真实平台
    报表就是这样滚动回填的。因此本模块的物化策略是 **整段重算覆盖**:
    每次调用删掉该平台指定日期范围的旧行、按当前事件流重新聚合。
    `as_of` 参数模拟"在某时刻拉报表"：晚于 as_of 的转化尚未发生、不可见 ——
    同一天的报表在不同时刻拉取会长大，这正是归一化层要处理的现实。

Ground truth 对照:
    ground_truth_daily() 按 UTC 日、无归因窗口、转化归点击日聚合 —— 唯一真值。
    归一化层的输出应与它对得上，三平台报表则各自偏离它（可解释的偏离）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from adsim.models import (
    Ad,
    AdGroup,
    AttributionSpec,
    AuctionEvent,
    ClickEvent,
    ConversionEvent,
    Platform,
    PlatformDailyReport,
)

# 三平台的默认归因配置（ADR-4 的具体数值）
DEFAULT_SPECS: dict[Platform, dict] = {
    Platform.META_SIM: dict(
        click_window_days=7, view_window_days=1,
        report_timezone="America/Los_Angeles",
        conversion_date_basis="click_date",
    ),
    Platform.GOOGLE_SIM: dict(
        click_window_days=30, view_window_days=0,
        report_timezone="America/New_York",
        conversion_date_basis="click_date",
    ),
    Platform.TIKTOK_SIM: dict(
        click_window_days=7, view_window_days=0,
        report_timezone="UTC",
        conversion_date_basis="conv_date",
    ),
}


def seed_attribution_specs(session: Session) -> None:
    """写入三平台默认归因配置（幂等：已存在则跳过）。"""
    for platform, kw in DEFAULT_SPECS.items():
        if session.get(AttributionSpec, platform) is None:
            session.add(AttributionSpec(platform=platform, **kw))
    session.commit()


def _local_date(ts_utc: datetime, tz: ZoneInfo) -> date:
    """事件时间（naive, UTC 语义）-> 平台报表时区的日期。"""
    return ts_utc.replace(tzinfo=timezone.utc).astimezone(tz).date()


def _ad_to_campaign(session: Session) -> dict[int, int]:
    rows = session.execute(
        select(Ad.id, AdGroup.campaign_id).join(AdGroup, Ad.ad_group_id == AdGroup.id)
    ).all()
    return {ad_id: campaign_id for ad_id, campaign_id in rows}


def generate_platform_reports(
    session: Session,
    platform: Platform,
    as_of: datetime | None = None,
) -> list[PlatformDailyReport]:
    """按平台自己的口径物化日报表（整段重算覆盖，模拟平台报表回填）。

    Args:
        as_of: 模拟"在该时刻(UTC)拉报表"。晚于它的转化不可见。
            None = 上帝视角（所有转化已回填完毕）。
    """
    spec = session.get(AttributionSpec, platform)
    if spec is None:
        raise ValueError(f"缺少 {platform} 的 AttributionSpec，先跑 seed_attribution_specs")
    tz = ZoneInfo(spec.report_timezone)
    window_hours = spec.click_window_days * 24
    ad2camp = _ad_to_campaign(session)

    # (local_date, ad_id) -> 累计指标
    buckets: dict[tuple[date, int], dict[str, int]] = {}

    def bucket(d: date, ad_id: int) -> dict[str, int]:
        return buckets.setdefault(
            (d, ad_id),
            dict(impressions=0, clicks=0, spend_micros=0,
                 conversions=0, revenue_micros=0),
        )

    # 曝光与花费：竞胜的拍卖事件
    for ev in session.scalars(
        select(AuctionEvent).where(
            AuctionEvent.platform == platform, AuctionEvent.won
        )
    ):
        b = bucket(_local_date(ev.sim_ts, tz), ev.ad_id)
        b["impressions"] += 1
        b["spend_micros"] += int(round(ev.paying_micros / 1000))

    # 点击
    for ev in session.scalars(
        select(ClickEvent).where(ClickEvent.platform == platform)
    ):
        bucket(_local_date(ev.sim_ts, tz), ev.ad_id)["clicks"] += 1

    # 转化：归因窗口过滤 + as_of 可见性 + 记账日口径
    for ev in session.scalars(
        select(ConversionEvent).where(ConversionEvent.platform == platform)
    ):
        if ev.delay_hours > window_hours:
            continue  # 窗口外 -> 平台"看不见"这笔转化
        if as_of is not None and ev.sim_ts > as_of:
            continue  # 尚未发生 -> 报表此刻还没回填到
        basis_ts = ev.click_ts if spec.conversion_date_basis == "click_date" else ev.sim_ts
        b = bucket(_local_date(basis_ts, tz), ev.ad_id)
        b["conversions"] += 1
        b["revenue_micros"] += ev.value_micros

    # 整段重算覆盖：删旧插新
    session.execute(
        delete(PlatformDailyReport).where(PlatformDailyReport.platform == platform)
    )
    reports = [
        PlatformDailyReport(
            platform=platform,
            report_date_local=d.isoformat(),
            campaign_id=ad2camp[ad_id],
            ad_id=ad_id,
            **metrics,
        )
        for (d, ad_id), metrics in sorted(buckets.items())
    ]
    session.add_all(reports)
    session.commit()
    return reports


@dataclass(frozen=True)
class GroundTruthRow:
    """唯一真值：UTC 日、无归因窗口、转化归点击日。"""
    utc_date: date
    ad_id: int
    impressions: int
    clicks: int
    spend_micros: int
    conversions: int
    revenue_micros: int


def ground_truth_daily(
    session: Session, platform: Platform
) -> list[GroundTruthRow]:
    """从事件流聚合 ground truth 日指标（归一化层的对照真值）。"""
    buckets: dict[tuple[date, int], dict[str, int]] = {}

    def bucket(d: date, ad_id: int) -> dict[str, int]:
        return buckets.setdefault(
            (d, ad_id),
            dict(impressions=0, clicks=0, spend_micros=0,
                 conversions=0, revenue_micros=0),
        )

    for ev in session.scalars(
        select(AuctionEvent).where(
            AuctionEvent.platform == platform, AuctionEvent.won
        )
    ):
        b = bucket(ev.sim_ts.date(), ev.ad_id)
        b["impressions"] += 1
        b["spend_micros"] += int(round(ev.paying_micros / 1000))

    for ev in session.scalars(
        select(ClickEvent).where(ClickEvent.platform == platform)
    ):
        bucket(ev.sim_ts.date(), ev.ad_id)["clicks"] += 1

    for ev in session.scalars(
        select(ConversionEvent).where(ConversionEvent.platform == platform)
    ):
        b = bucket(ev.click_ts.date(), ev.ad_id)
        b["conversions"] += 1
        b["revenue_micros"] += ev.value_micros

    return [
        GroundTruthRow(utc_date=d, ad_id=ad_id, **m)
        for (d, ad_id), m in sorted(buckets.items())
    ]
