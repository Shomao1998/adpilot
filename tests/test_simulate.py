"""模拟日引擎测试。

断言思路与 test_adsim 一致 —— 可证伪的行为性质，而非魔法数字:
    1. 事件层级守恒: clicks <= wins <= auctions, conversions <= clicks，外键成链。
    2. 预算约束: 花费不超预算 + 单次曝光成本（贪心停止的最大过冲）。
    3. 复现性: 同 seed 同输入 -> 事件流完全一致。
    4. 质量分单调性: 高 creative_quality 的广告拿到更高 CTR。
    5. 延迟长尾: 部分转化落在 7 天归因窗口外（W2 口径差异的前提）。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from adsim.auction import MarketModel
from adsim.models import (
    Ad, AdGroup, AuctionEvent, AuctionMechanism, Base, Campaign,
    ClickEvent, ConversionEvent, Platform,
)
from adsim.simulate import (
    MICROS, AdSimEngine, DayStats, PlatformProfile, usd_to_micros,
)

DAY = date(2026, 7, 1)
# 市场价中位数 $5/CPM
MARKET = {p: MarketModel(mu=float(np.log(5.0)), sigma=0.6, floor=0.01)
          for p in Platform}


def _profile(platform=Platform.META_SIM,
             mechanism=AuctionMechanism.SECOND_PRICE,
             opps=3000, ctr=0.10, cvr=0.20) -> PlatformProfile:
    return PlatformProfile(platform=platform, mechanism=mechanism,
                           daily_opportunities=opps,
                           base_ctr=ctr, base_cvr=cvr)


def _setup(session, platform=Platform.META_SIM, budget_usd=500.0,
           bid_usd_cpm=8.0, qualities=(0.5, 0.5)):
    c = Campaign(platform=platform, name="t",
                 daily_budget_micros=usd_to_micros(budget_usd),
                 external_ref="x")
    session.add(c); session.flush()
    g = AdGroup(campaign_id=c.id, name="g",
                bid_micros=usd_to_micros(bid_usd_cpm))
    session.add(g); session.flush()
    ads = [Ad(ad_group_id=g.id, creative_id=f"cr{i}", creative_quality=q)
           for i, q in enumerate(qualities)]
    session.add_all(ads); session.flush()
    return c, g, ads


def _fresh_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


class TestEventStream:
    def test_hierarchy_and_fk_chain(self):
        s = _fresh_session()
        c, g, ads = _setup(s)
        eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()}, seed=1)
        st = eng.run_day(s, DAY, c, g, ads)

        assert st.auctions > 0
        assert 0 < st.wins <= st.auctions
        assert 0 < st.clicks <= st.wins
        assert 0 < st.conversions <= st.clicks

        # DB 行数与统计一致
        n_auc = len(s.scalars(select(AuctionEvent)).all())
        n_clk = len(s.scalars(select(ClickEvent)).all())
        n_cnv = len(s.scalars(select(ConversionEvent)).all())
        assert (n_auc, n_clk, n_cnv) == (st.auctions, st.clicks, st.conversions)

        # 外键成链: click -> won auction; conversion -> click
        won_ids = {a.id for a in s.scalars(
            select(AuctionEvent).where(AuctionEvent.won)).all()}
        clicks = s.scalars(select(ClickEvent)).all()
        assert all(k.auction_event_id in won_ids for k in clicks)
        click_ids = {k.id for k in clicks}
        for v in s.scalars(select(ConversionEvent)).all():
            assert v.click_event_id in click_ids
            assert v.sim_ts > v.click_ts
            assert v.sim_ts - v.click_ts == pytest.approx(
                timedelta(hours=v.delay_hours), abs=timedelta(seconds=1))

    def test_all_prices_are_int_micros_and_second_price_holds(self):
        s = _fresh_session()
        c, g, ads = _setup(s, bid_usd_cpm=8.0)
        eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()}, seed=2)
        eng.run_day(s, DAY, c, g, ads)
        for a in s.scalars(select(AuctionEvent)).all():
            assert isinstance(a.paying_micros, int)
            assert isinstance(a.bid_micros, int)
            if a.won:
                assert 0 < a.paying_micros < a.bid_micros  # 二价 < 出价
            else:
                assert a.paying_micros == 0


class TestBudget:
    def test_spend_respects_daily_budget(self):
        s = _fresh_session()
        # 预算 $2，出价 $8/CPM -> 单曝光最高 8000 micros，预算必然先耗尽
        c, g, ads = _setup(s, budget_usd=2.0, bid_usd_cpm=8.0)
        eng = AdSimEngine(MARKET,
                          {Platform.META_SIM: _profile(opps=20_000, ctr=0.01)},
                          seed=3)
        st = eng.run_day(s, DAY, c, g, ads)
        max_single_imp = g.bid_micros // 1000
        assert st.spend_micros <= c.daily_budget_micros + max_single_imp
        assert st.auctions < 20_000  # 确实提前停了

    def test_spend_matches_event_stream(self):
        """DayStats 只是冒烟口径，真值必须能从事件表重新聚合出来。"""
        s = _fresh_session()
        c, g, ads = _setup(s)
        eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()}, seed=4)
        st = eng.run_day(s, DAY, c, g, ads)
        recomputed = sum(
            int(round(a.paying_micros / 1000))
            for a in s.scalars(select(AuctionEvent).where(AuctionEvent.won)))
        assert recomputed == pytest.approx(st.spend_micros, rel=1e-6)


class TestDeterminism:
    def test_same_seed_same_stream(self):
        def run(seed):
            s = _fresh_session()
            c, g, ads = _setup(s)
            eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()},
                              seed=seed)
            st = eng.run_day(s, DAY, c, g, ads)
            evs = [(a.won, a.paying_micros, a.hour_of_day)
                   for a in s.scalars(select(AuctionEvent)).all()]
            return st, evs

        st1, ev1 = run(seed=7)
        st2, ev2 = run(seed=7)
        st3, _ = run(seed=8)
        assert st1 == st2 and ev1 == ev2
        assert st1 != st3  # 不同 seed 应产生不同流（防"假随机"）


class TestClickModel:
    def test_higher_quality_gets_higher_ctr(self):
        s = _fresh_session()
        c, g, ads = _setup(s, qualities=(0.9, 0.1))
        eng = AdSimEngine(MARKET,
                          {Platform.META_SIM: _profile(opps=8000, ctr=0.15)},
                          seed=5)
        eng.run_day(s, DAY, c, g, ads)
        hi, lo = ads[0].id, ads[1].id

        def ctr_of(ad_id):
            wins = len(s.scalars(select(AuctionEvent).where(
                AuctionEvent.ad_id == ad_id, AuctionEvent.won)).all())
            clicks = len(s.scalars(select(ClickEvent).where(
                ClickEvent.ad_id == ad_id)).all())
            return clicks / max(wins, 1)

        assert ctr_of(hi) > ctr_of(lo)

    def test_paused_ads_get_no_traffic(self):
        s = _fresh_session()
        c, g, ads = _setup(s)
        ads[1].status = "paused"
        eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()}, seed=6)
        eng.run_day(s, DAY, c, g, ads)
        assert not s.scalars(select(AuctionEvent).where(
            AuctionEvent.ad_id == ads[1].id)).all()

    def test_all_paused_returns_empty(self):
        s = _fresh_session()
        c, g, ads = _setup(s)
        for a in ads:
            a.status = "paused"
        eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()}, seed=6)
        assert eng.run_day(s, DAY, c, g, ads) == DayStats(0, 0, 0, 0, 0, 0)


class TestConversionModel:
    def test_delay_tail_crosses_7day_window(self):
        """默认延迟参数下必须有转化落在 7 天窗口外，
        否则 Meta-sim(7d) 与 Google-sim(30d) 报表将无差异可展示。"""
        s = _fresh_session()
        c, g, ads = _setup(s)
        eng = AdSimEngine(MARKET,
                          {Platform.META_SIM: _profile(opps=6000,
                                                       ctr=0.3, cvr=0.5)},
                          seed=9)
        st = eng.run_day(s, DAY, c, g, ads)
        assert st.conversions > 100  # 样本量护栏
        delays = [v.delay_hours for v in s.scalars(select(ConversionEvent))]
        frac_beyond_7d = np.mean([d > 168 for d in delays])
        assert 0.005 < frac_beyond_7d < 0.15

    def test_revenue_positive_int_micros(self):
        s = _fresh_session()
        c, g, ads = _setup(s)
        eng = AdSimEngine(MARKET, {Platform.META_SIM: _profile()}, seed=10)
        st = eng.run_day(s, DAY, c, g, ads)
        values = [v.value_micros for v in s.scalars(select(ConversionEvent))]
        assert all(isinstance(v, int) and v > 0 for v in values)
        assert sum(values) == st.revenue_micros


class TestGSP:
    def test_google_uses_gsp_and_produces_events(self):
        s = _fresh_session()
        c, g, ads = _setup(s, platform=Platform.GOOGLE_SIM,
                           bid_usd_cpm=12.0, qualities=(0.8, 0.8))
        prof = _profile(platform=Platform.GOOGLE_SIM,
                        mechanism=AuctionMechanism.GSP,
                        opps=3000, ctr=0.1, cvr=0.2)
        eng = AdSimEngine(MARKET, {Platform.GOOGLE_SIM: prof}, seed=11)
        st = eng.run_day(s, DAY, c, g, ads)
        assert st.wins > 0 and st.clicks > 0
        # GSP 实付不超过出价
        for a in s.scalars(select(AuctionEvent).where(AuctionEvent.won)):
            assert a.paying_micros <= a.bid_micros
