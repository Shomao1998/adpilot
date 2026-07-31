"""拍卖引擎。

两种机制，对应三个模拟平台:
    - SecondPriceAuction (Meta-sim / TikTok-sim):
        市场价 M 直接从拟合的对数正态分布采样（分小时缩放）。
        出价 b > max(M, floor) 则竞胜，实付 max(M, floor) —— 标准二价规则。
        直接采样市场价而非逐个模拟 N 个对手，是 iPinYou 文献里的标准做法:
        我们拟合的 paying_price 本身就是"最高对手价"的分布，再去分解成
        单个对手反而引入无依据的假设。
    - GSPAuction (Google-sim):
        广义二价 + 质量分。采样 K 个对手 (bid_i, quality_i)，按 AdRank=bid*quality
        排序，取前 S 个槽位；第 k 名实付 AdRank_{k+1} / quality_k + epsilon。
        这还原了 Google "质量分能降低实付 CPC" 的机制 —— 创意质量在 Google-sim
        上不仅影响点击率，还影响成本，跨平台对比因此更有戏剧性。

设计约束: 引擎无状态、入参显式传 rng —— 同一 seed 全链路可复现（测试与演示都依赖这点）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AuctionOutcome:
    won: bool
    paying_price: float      # 竞胜时的实付；竞败为 0
    market_price: float      # 最高对手价（ground truth 记录用）
    slot: int = -1           # GSP: 赢得的槽位序号（0 最优）；二价拍卖恒为 0/-1


@dataclass
class MarketModel:
    """一个平台的市场行情参数（由 iPinYou 拟合 + 平台档位缩放得到）。"""
    mu: float
    sigma: float
    floor: float = 0.0
    hourly_multipliers: np.ndarray = field(
        default_factory=lambda: np.ones(24)
    )

    def sample_market_price(self, hour: int, rng: np.random.Generator) -> float:
        m = rng.lognormal(mean=self.mu, sigma=self.sigma)
        return float(m * self.hourly_multipliers[hour % 24])


class SecondPriceAuction:
    """Meta / TikTok 风格：单槽位密封二价。"""

    def __init__(self, market: MarketModel):
        self.market = market

    def run(self, bid: float, hour: int, rng: np.random.Generator) -> AuctionOutcome:
        market_price = self.market.sample_market_price(hour, rng)
        clearing = max(market_price, self.market.floor)
        if bid > clearing:
            return AuctionOutcome(won=True, paying_price=clearing,
                                  market_price=market_price, slot=0)
        return AuctionOutcome(won=False, paying_price=0.0,
                              market_price=market_price)


class GSPAuction:
    """Google 风格：K 个对手、S 个槽位、质量分加权的广义二价。"""

    def __init__(
        self,
        market: MarketModel,
        n_competitors_mean: float = 6.0,
        n_slots: int = 4,
        quality_alpha: float = 8.0,
        quality_beta: float = 4.0,
        epsilon: float = 0.01,
    ):
        self.market = market
        self.n_competitors_mean = n_competitors_mean
        self.n_slots = n_slots
        # 对手质量分 ~ Beta(alpha, beta)，均值 ~0.67，制造质量差异
        self.quality_alpha = quality_alpha
        self.quality_beta = quality_beta
        self.epsilon = epsilon

    def run(
        self, bid: float, quality: float, hour: int, rng: np.random.Generator
    ) -> AuctionOutcome:
        n = max(1, rng.poisson(self.n_competitors_mean))
        # 对手出价：围绕市场价采样（各自独立的对数正态抖动）
        base = self.market.sample_market_price(hour, rng)
        comp_bids = base * rng.lognormal(mean=0.0, sigma=0.35, size=n)
        comp_quality = rng.beta(self.quality_alpha, self.quality_beta, size=n)
        comp_rank = comp_bids * comp_quality

        my_rank = bid * quality
        market_price = float(comp_bids.max())

        # 我方排名（比我 AdRank 高的对手数）
        position = int(np.sum(comp_rank > my_rank))
        if position >= self.n_slots or my_rank <= 0:
            return AuctionOutcome(won=False, paying_price=0.0,
                                  market_price=market_price)

        # GSP 定价: 下一名的 AdRank / 我的质量分 + epsilon
        lower_ranks = comp_rank[comp_rank <= my_rank]
        next_rank = float(lower_ranks.max()) if lower_ranks.size else 0.0
        price = min(bid, next_rank / max(quality, 1e-6) + self.epsilon)
        price = max(price, self.market.floor)
        return AuctionOutcome(won=True, paying_price=price,
                              market_price=market_price, slot=position)


def make_platform_markets(
    base_fit_mu: float, base_fit_sigma: float, hourly: np.ndarray | None = None
) -> dict[str, MarketModel]:
    """从单一 iPinYou 拟合结果派生三平台差异化行情。

    缩放系数依据公开行业 benchmark 的相对水位（Meta CPM 为基准 1.0，
    TikTok 约 0.6-0.7 倍，Google 搜索按点击计费、换算到曝光口径更高）。
    log 空间下乘性缩放 = mu 加常数: scale k 倍 -> mu + ln(k)。
    """
    h = hourly if hourly is not None else np.ones(24)
    return {
        "meta_sim": MarketModel(mu=base_fit_mu, sigma=base_fit_sigma,
                                floor=0.01, hourly_multipliers=h),
        "tiktok_sim": MarketModel(mu=base_fit_mu + np.log(0.65),
                                  sigma=base_fit_sigma * 1.10,
                                  floor=0.01, hourly_multipliers=h),
        "google_sim": MarketModel(mu=base_fit_mu + np.log(1.8),
                                  sigma=base_fit_sigma * 0.90,
                                  floor=0.05, hourly_multipliers=h),
    }
