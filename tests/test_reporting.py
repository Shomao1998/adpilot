"""报表生成器测试。

口径差异必须是 **可预测的、可对照真值验证的**，所以测试用手工构造的
最小事件流精确断言每个口径维度（窗口/时区/记账日/as_of 回填），
再用真实引擎跑一天做守恒性集成测试。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from adsim.auction import MarketModel
from adsim.models import (
    Ad, AdGroup, AttributionSpec, AuctionEvent, Base, Campaign,
    ClickEvent, ConversionEvent, Platform, PlatformDailyReport,
)
from adsim.reporting import (
    generate_platform_reports, ground_truth_daily, seed_attribution_specs,
)
from adsim.simulate import AdSimEngine, PlatformProfile, usd_to_micros
from adsim.models import AuctionMechanism


def _fresh_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    seed_attribution_specs(s)
    return s


def _entity(s, platform):
    c = Campaign(platform=platform, name=f"c-{platform.value}",
                 daily_budget_micros=usd_to_micros(100), external_ref="x")
    s.add(c); s.flush()
    g = AdGroup(campaign_id=c.id, name="g", bid_micros=usd_to_micros(8))
    s.add(g); s.flush()
    a = Ad(ad_group_id=g.id, creative_id="cr", creative_quality=0.5)
    s.add(a); s.flush()
    return c, g, a


def _impression(s, platform, ad, ts, paying_cpm_micros=5_000_000):
    ev = AuctionEvent(platform=platform, ad_id=ad.id, sim_ts=ts,
                      hour_of_day=ts.hour, bid_micros=8_000_000,
                      floor_micros=10_000, market_price_micros=paying_cpm_micros,
                      won=True, paying_micros=paying_cpm_micros)
    s.add(ev); s.flush()
    return ev

def _click(s, platform, ad, auction_ev, ts):
    ev = ClickEvent(auction_event_id=auction_ev.id, platform=platform,
                    ad_id=ad.id, sim_ts=ts)
    s.add(ev); s.flush()
    return ev

def _conversion(s, platform, ad, click_ev, delay_hours, value=60_000_000):
    conv_ts = click_ev.sim_ts + timedelta(hours=delay_hours)
    ev = ConversionEvent(click_event_id=click_ev.id, platform=platform,
                         ad_id=ad.id, sim_ts=conv_ts, click_ts=click_ev.sim_ts,
                         delay_hours=delay_hours, value_micros=value)
    s.add(ev); s.flush()
    return ev


class TestSpecs:
    def test_seed_idempotent(self):
        s = _fresh_session()
        seed_attribution_specs(s)  # 第二次调用
        assert len(s.scalars(select(AttributionSpec)).all()) == 3

    def test_missing_spec_raises(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        s = Session(engine)  # 未 seed
        with pytest.raises(ValueError, match="AttributionSpec"):
            generate_platform_reports(s, Platform.META_SIM)


class TestAttributionWindow:
    def test_10day_delay_seen_by_google_not_meta(self):
        """延迟 10 天的转化：30d 窗口计入、7d 窗口丢弃 —— ADR-4 的核心可观测差异。"""
        for platform, expected_convs in [(Platform.GOOGLE_SIM, 1),
                                         (Platform.META_SIM, 0)]:
            s = _fresh_session()
            _, _, ad = _entity(s, platform)
            ts = datetime(2026, 7, 1, 12, 0, 0)
            imp = _impression(s, platform, ad, ts)
            clk = _click(s, platform, ad, imp, ts)
            _conversion(s, platform, ad, clk, delay_hours=240)  # 10 天
            reports = generate_platform_reports(s, platform)
            assert sum(r.conversions for r in reports) == expected_convs
            # ground truth 永远看得见全部转化
            gt = ground_truth_daily(s, platform)
            assert sum(r.conversions for r in gt) == 1


class TestTimezone:
    def test_same_event_lands_on_different_local_dates(self):
        """UTC 07-01 06:00 的曝光: LA(-7) 属 06-30、NY(-4) 属 07-01、UTC 属 07-01。"""
        ts = datetime(2026, 7, 1, 6, 0, 0)
        expected = {Platform.META_SIM: "2026-06-30",    # America/Los_Angeles
                    Platform.GOOGLE_SIM: "2026-07-01",  # America/New_York
                    Platform.TIKTOK_SIM: "2026-07-01"}  # UTC
        for platform, expected_date in expected.items():
            s = _fresh_session()
            _, _, ad = _entity(s, platform)
            _impression(s, platform, ad, ts)
            reports = generate_platform_reports(s, platform)
            assert [r.report_date_local for r in reports] == [expected_date]


class TestConversionDateBasis:
    def test_click_date_vs_conv_date(self):
        """延迟 48h 的转化: Meta(click_date) 记在点击日, TikTok(conv_date) 记在转化日。"""
        ts = datetime(2026, 7, 1, 12, 0, 0)
        expected = {Platform.META_SIM: "2026-07-01",
                    Platform.TIKTOK_SIM: "2026-07-03"}
        for platform, expected_date in expected.items():
            s = _fresh_session()
            _, _, ad = _entity(s, platform)
            imp = _impression(s, platform, ad, ts)
            clk = _click(s, platform, ad, imp, ts)
            _conversion(s, platform, ad, clk, delay_hours=48)
            reports = generate_platform_reports(s, platform)
            conv_dates = [r.report_date_local for r in reports if r.conversions]
            assert conv_dates == [expected_date]


class TestBackfill:
    def test_as_of_hides_future_conversions_then_backfills(self):
        """同一天的报表，晚点再拉会"长大" —— 平台报表回填的模拟。"""
        s = _fresh_session()
        platform = Platform.META_SIM
        _, _, ad = _entity(s, platform)
        ts = datetime(2026, 7, 1, 12, 0, 0)
        imp = _impression(s, platform, ad, ts)
        clk = _click(s, platform, ad, imp, ts)
        _conversion(s, platform, ad, clk, delay_hours=72)  # 07-04 转化

        early = generate_platform_reports(s, platform,
                                          as_of=datetime(2026, 7, 2))
        assert sum(r.conversions for r in early) == 0
        late = generate_platform_reports(s, platform,
                                         as_of=datetime(2026, 7, 10))
        assert sum(r.conversions for r in late) == 1
        # 回填改写的是点击日(07-01 LA 口径)那一行，而不是新增一行
        assert {r.report_date_local for r in late} == \
               {r.report_date_local for r in early}

    def test_regeneration_overwrites_no_duplicates(self):
        s = _fresh_session()
        platform = Platform.TIKTOK_SIM
        _, _, ad = _entity(s, platform)
        _impression(s, platform, ad, datetime(2026, 7, 1, 12))
        generate_platform_reports(s, platform)
        generate_platform_reports(s, platform)
        rows = s.scalars(select(PlatformDailyReport)).all()
        assert len(rows) == 1 and rows[0].impressions == 1


class TestConservationWithEngine:
    """真实引擎跑一天：报表与 ground truth 在窗口无关的指标上必须完全一致。"""

    def test_totals_match_ground_truth(self):
        s = _fresh_session()
        market = {p: MarketModel(mu=float(np.log(5.0)), sigma=0.6, floor=0.01)
                  for p in Platform}
        c, g, ads = self._setup(s, Platform.META_SIM)
        prof = PlatformProfile(platform=Platform.META_SIM,
                               mechanism=AuctionMechanism.SECOND_PRICE,
                               daily_opportunities=4000,
                               base_ctr=0.2, base_cvr=0.3)
        eng = AdSimEngine(market, {Platform.META_SIM: prof}, seed=42)
        st = eng.run_day(s, date(2026, 7, 1), c, g, ads)
        assert st.conversions > 20  # 样本量护栏

        reports = generate_platform_reports(s, Platform.META_SIM)
        gt = ground_truth_daily(s, Platform.META_SIM)

        for metric in ("impressions", "clicks", "spend_micros"):
            assert (sum(getattr(r, metric) for r in reports)
                    == sum(getattr(r, metric) for r in gt))
        # 平台转化 = 恰好窗口内的真值转化（7d = 168h），不多不少
        in_window = len([v for v in s.scalars(select(ConversionEvent))
                         if v.delay_hours <= 168])
        assert sum(r.conversions for r in reports) == in_window
        assert in_window <= sum(r.conversions for r in gt)

    @staticmethod
    def _setup(s, platform):
        c = Campaign(platform=platform, name="t",
                     daily_budget_micros=usd_to_micros(500), external_ref="x")
        s.add(c); s.flush()
        g = AdGroup(campaign_id=c.id, name="g", bid_micros=usd_to_micros(8))
        s.add(g); s.flush()
        ads = [Ad(ad_group_id=g.id, creative_id=f"cr{i}", creative_quality=0.5)
               for i in range(2)]
        s.add_all(ads); s.flush()
        return c, g, ads
