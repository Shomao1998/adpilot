"""AdSim 核心正确性测试。

关键测试思想（面试可讲）: 模拟器的可信度不靠肉眼，靠三类可证伪的断言 ——
    1. 统计恢复: 用已知参数造数据，拟合应恢复真值（删失版），朴素版应显著低估。
    2. 机制一致: 出价 b 的经验竞胜率应等于理论 CDF(b)。
    3. 定价性质: 二价实付 <= 出价；GSP 中质量分更高 -> 平均实付更低。
"""
from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect

from adsim.auction import GSPAuction, MarketModel, SecondPriceAuction
from adsim.fitting import fit_lognormal_censored, fit_lognormal_naive
from adsim.models import Base

TRUE_MU, TRUE_SIGMA = 4.0, 0.8  # 市场价 ~ LogNormal(4.0, 0.8), E[X]≈75


def _make_censored_dataset(n: int, seed: int = 0):
    """模拟真实数据生成过程: 市场价 ~ 真分布; 我方按某历史策略出价;
    赢 -> 观测市场价; 输 -> 只留下出价作为删失点。"""
    rng = np.random.default_rng(seed)
    market = rng.lognormal(TRUE_MU, TRUE_SIGMA, size=n)
    bids = rng.lognormal(TRUE_MU - 0.1, 0.5, size=n)  # 出价偏低 -> 胜率 <50%，删失严重
    won = bids > market
    return market[won], bids[~won]


class TestCensoredFitting:
    def test_censored_mle_recovers_truth(self):
        obs, cens = _make_censored_dataset(60_000)
        fit = fit_lognormal_censored(obs, cens)
        assert fit.mu == pytest.approx(TRUE_MU, abs=0.03)
        assert fit.sigma == pytest.approx(TRUE_SIGMA, abs=0.03)

    def test_naive_fit_is_biased_low(self):
        """朴素拟合系统性低估 —— 这是删失校正存在的理由。"""
        obs, cens = _make_censored_dataset(60_000)
        naive = fit_lognormal_naive(obs)
        censored = fit_lognormal_censored(obs, cens)
        assert naive.mean() < censored.mean() * 0.85  # 低估至少 15%
        assert naive.mu < TRUE_MU - 0.1

    def test_win_rate_prediction(self):
        obs, cens = _make_censored_dataset(60_000)
        fit = fit_lognormal_censored(obs, cens)
        # 中位数出价的理论胜率应为 0.5
        assert fit.win_rate_at(fit.quantile(0.5)) == pytest.approx(0.5, abs=0.01)


class TestSecondPriceAuction:
    def test_empirical_win_rate_matches_cdf(self):
        market = MarketModel(mu=TRUE_MU, sigma=TRUE_SIGMA)
        auction = SecondPriceAuction(market)
        rng = np.random.default_rng(7)
        from scipy import stats
        for bid_q in (0.3, 0.5, 0.8):
            bid = float(stats.lognorm.ppf(bid_q, s=TRUE_SIGMA, scale=np.exp(TRUE_MU)))
            wins = sum(auction.run(bid, hour=12, rng=rng).won for _ in range(20_000))
            assert wins / 20_000 == pytest.approx(bid_q, abs=0.02)

    def test_pays_second_price_not_bid(self):
        market = MarketModel(mu=TRUE_MU, sigma=TRUE_SIGMA, floor=1.0)
        auction = SecondPriceAuction(market)
        rng = np.random.default_rng(7)
        outcomes = [auction.run(bid=500.0, hour=0, rng=rng) for _ in range(5_000)]
        won = [o for o in outcomes if o.won]
        assert won, "高出价应有竞胜"
        assert all(o.paying_price <= 500.0 for o in won)
        assert all(o.paying_price >= 1.0 for o in won)          # 不低于底价
        assert np.mean([o.paying_price for o in won]) < 500.0 * 0.5  # 显著低于出价

    def test_hourly_multiplier_shifts_price(self):
        hourly = np.ones(24); hourly[20] = 2.0  # 晚八点行情翻倍
        market = MarketModel(mu=TRUE_MU, sigma=TRUE_SIGMA, hourly_multipliers=hourly)
        auction = SecondPriceAuction(market)
        rng = np.random.default_rng(7)
        wr_noon = np.mean([auction.run(60.0, 12, rng).won for _ in range(10_000)])
        wr_prime = np.mean([auction.run(60.0, 20, rng).won for _ in range(10_000)])
        assert wr_prime < wr_noon - 0.1  # 同样出价，高峰时段胜率显著下降


class TestGSPAuction:
    def test_higher_quality_pays_less(self):
        market = MarketModel(mu=TRUE_MU, sigma=TRUE_SIGMA)
        gsp = GSPAuction(market)
        rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
        bid = 150.0
        pay_hi = [o.paying_price for _ in range(8_000)
                  if (o := gsp.run(bid, quality=0.9, hour=12, rng=rng_a)).won]
        pay_lo = [o.paying_price for _ in range(8_000)
                  if (o := gsp.run(bid, quality=0.4, hour=12, rng=rng_b)).won]
        assert len(pay_hi) > len(pay_lo)                 # 高质量 -> 更多竞胜
        assert np.mean(pay_hi) < np.mean(pay_lo)         # 高质量 -> 实付更低
        assert all(p <= bid + 1e-9 for p in pay_hi + pay_lo)


class TestModels:
    def test_schema_creates_on_sqlite(self):
        """BigInteger 主键经 with_variant 后应能在 SQLite 建表并自增。"""
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        tables = set(inspect(engine).get_table_names())
        assert {"campaigns", "ad_groups", "ads", "auction_events",
                "click_events", "conversion_events", "attribution_specs",
                "platform_daily_reports"} <= tables

    def test_autoincrement_works(self):
        from sqlalchemy.orm import Session
        from adsim.models import Campaign, Platform
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with Session(engine) as s:
            c1 = Campaign(platform=Platform.META_SIM, name="a",
                          daily_budget_micros=10, external_ref="x1")
            c2 = Campaign(platform=Platform.GOOGLE_SIM, name="b",
                          daily_budget_micros=10, external_ref="x2")
            s.add_all([c1, c2]); s.commit()
            assert c1.id is not None and c2.id == c1.id + 1
