"""市场价格分布拟合：删失感知（censoring-aware）的对数正态 MLE。

为什么需要删失处理（面试高频考点）:
    iPinYou 日志里，市场价格(paying_price，即除你之外的最高竞价)只有在你 **竞胜**
    的场次才可观测（imp 日志）。竞败场次（出现在 bid 日志但没有对应 imp 记录）
    只能知道 market_price >= 你的出价 —— 这是右删失(right-censored)观测。

    如果只拿竞胜样本做朴素拟合，样本天然偏向低价场次（价格低你才赢得了），
    估出的分布会 **系统性低估** 市场价格 —— 用它做出价模拟会让 Agent 误以为
    钱很好花。正确做法是把两类观测都写进似然:

        L(mu, sigma) =  Π_wins  pdf(m_i)          # 竞胜: 精确观测到市场价 m_i
                      × Π_losses (1 - CDF(b_j))    # 竞败: 只知市场价 >= 出价 b_j

    对数正态是 RTB 市场价建模的经典选择（文献: Cui et al. KDD'11 等），
    右偏、支撑为正、两参数易拟合。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class LogNormalFit:
    mu: float          # 对数空间均值
    sigma: float       # 对数空间标准差
    n_observed: int    # 竞胜（精确观测）样本数
    n_censored: int    # 竞败（右删失）样本数
    neg_log_lik: float

    def mean(self) -> float:
        """市场价均值 E[X] = exp(mu + sigma^2/2)"""
        return float(np.exp(self.mu + self.sigma**2 / 2))

    def quantile(self, q: float) -> float:
        return float(stats.lognorm.ppf(q, s=self.sigma, scale=np.exp(self.mu)))

    def win_rate_at(self, bid: float) -> float:
        """给定出价的理论竞胜率 P(market < bid) = CDF(bid)。"""
        return float(stats.lognorm.cdf(bid, s=self.sigma, scale=np.exp(self.mu)))


def fit_lognormal_censored(
    observed_prices: np.ndarray,
    censored_at_bids: np.ndarray | None = None,
) -> LogNormalFit:
    """删失感知 MLE。

    Args:
        observed_prices: 竞胜场次观测到的市场价（paying_price），必须 > 0。
        censored_at_bids: 竞败场次我方的出价（市场价 >= 该值），可为空。

    Returns:
        LogNormalFit。当 censored_at_bids 为空时退化为普通截断样本 MLE
        （仍有偏，函数会照常返回 —— 偏差对比正是测试用例展示的内容）。
    """
    obs = np.asarray(observed_prices, dtype=float)
    obs = obs[obs > 0]
    if obs.size == 0:
        raise ValueError("observed_prices 为空或全部非正")

    cens = (
        np.asarray(censored_at_bids, dtype=float)
        if censored_at_bids is not None
        else np.empty(0)
    )
    cens = cens[cens > 0]

    log_obs = np.log(obs)

    def neg_log_lik(params: np.ndarray) -> float:
        mu, log_sigma = params
        sigma = np.exp(log_sigma)  # 参数化保证 sigma > 0，免约束优化
        # 竞胜项: lognormal logpdf
        ll = np.sum(stats.norm.logpdf(log_obs, loc=mu, scale=sigma) - log_obs)
        # 竞败项: log 生存函数 log P(X >= b) —— 用 logsf 保证数值稳定
        if cens.size:
            ll += np.sum(stats.norm.logsf(np.log(cens), loc=mu, scale=sigma))
        return -ll

    # 初值：竞胜样本的对数矩（有偏但足够做起点）
    x0 = np.array([log_obs.mean(), np.log(max(log_obs.std(), 1e-3))])
    res = optimize.minimize(neg_log_lik, x0, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"MLE 未收敛: {res.message}")

    mu_hat, sigma_hat = res.x[0], float(np.exp(res.x[1]))
    return LogNormalFit(
        mu=float(mu_hat),
        sigma=sigma_hat,
        n_observed=int(obs.size),
        n_censored=int(cens.size),
        neg_log_lik=float(res.fun),
    )


def fit_lognormal_naive(observed_prices: np.ndarray) -> LogNormalFit:
    """朴素拟合（只用竞胜样本、无删失校正）—— 用于对照展示偏差。"""
    obs = np.asarray(observed_prices, dtype=float)
    obs = obs[obs > 0]
    log_obs = np.log(obs)
    mu, sigma = float(log_obs.mean()), float(log_obs.std(ddof=1))
    ll = float(np.sum(stats.norm.logpdf(log_obs, loc=mu, scale=sigma) - log_obs))
    return LogNormalFit(mu=mu, sigma=sigma, n_observed=int(obs.size),
                        n_censored=0, neg_log_lik=-ll)


def fit_hourly_multipliers(
    hours: np.ndarray, prices: np.ndarray, floor: float = 0.25, cap: float = 4.0
) -> np.ndarray:
    """按小时的市场热度乘数: multiplier[h] = median(price|hour=h) / median(price)。
    用中位数抗极端值；clip 防止小样本小时段爆炸。返回 shape=(24,)。"""
    hours = np.asarray(hours, dtype=int)
    prices = np.asarray(prices, dtype=float)
    overall = np.median(prices)
    mult = np.ones(24)
    for h in range(24):
        p = prices[hours == h]
        if p.size >= 30:  # 样本太少的小时段保持 1.0
            mult[h] = np.clip(np.median(p) / overall, floor, cap)
    return mult
