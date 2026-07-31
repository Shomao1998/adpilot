"""iPinYou 日志 -> 市场价分布参数 拟合脚本。

用法:
    python scripts/fit_ipinyou.py --bid-log bid.20130606.txt \
        --imp-log imp.20130606.txt --sample 500000 --out market_params.json

数据准备（推荐路径）:
    使用社区标准预处理仓库 make-ipinyou-data (github.com/wnzhang/make-ipinyou-data)
    处理后的 season 2/3 数据。原始数据 ~15GB，本脚本流式读取 + 蓄水池采样，
    单日单广告主切片即可（拟合两参数分布不需要全量）。

列格式（make-ipinyou-data 处理后的 tab 分隔格式，运行前务必抽几行核对！）:
    idx 1  timestamp   (yyyyMMddHHmmssSSS)
    idx 19 bidding_price  (整数价格 / CPM)
    idx 20 paying_price   (仅 imp 日志有意义; 即市场价)

删失样本构造逻辑:
    - imp 日志的每一行 = 一次竞胜 -> paying_price 是精确观测的市场价
    - bid 日志中 BidID 不在 imp 日志的行 = 竞败 -> 市场价 >= bidding_price（右删失）

货币口径 (ADR-5): 全项目统一美元计价，不做任何汇率/币种转换。iPinYou 日志只用于
    提取市场价的分布形状（mu/sigma、分时热度），原始整数价格除以 100 得到主单位
    数值后直接按 USD/CPM 解读；各平台行情的绝对水位由行业 benchmark 缩放校准
    （见 auction.make_platform_markets），与原始数据的币种无关。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adsim.fitting import (  # noqa: E402
    fit_hourly_multipliers,
    fit_lognormal_censored,
    fit_lognormal_naive,
)

COL_BID_ID = 0
COL_TIMESTAMP = 1
COL_BIDDING_PRICE = 19
COL_PAYING_PRICE = 20
PRICE_SCALE = 0.01  # 原始整数价格 -> 主单位（按 USD 口径解读，ADR-5）


def reservoir_sample_lines(path: Path, k: int, rng: np.random.Generator) -> list[str]:
    """蓄水池采样：单遍流式读取，内存 O(k)，适配 GB 级日志。"""
    sample: list[str] = []
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i < k:
                sample.append(line)
            else:
                j = rng.integers(0, i + 1)
                if j < k:
                    sample[j] = line
    return sample


def parse_imp(lines: list[str]) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """imp 日志 -> (市场价数组, 小时数组, 竞胜 BidID 集合)"""
    prices, hours, ids = [], [], set()
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if len(cols) <= COL_PAYING_PRICE:
            continue
        try:
            p = float(cols[COL_PAYING_PRICE]) * PRICE_SCALE
            h = int(cols[COL_TIMESTAMP][8:10])
        except (ValueError, IndexError):
            continue
        if p > 0:
            prices.append(p)
            hours.append(h)
            ids.add(cols[COL_BID_ID])
    return np.array(prices), np.array(hours), ids


def parse_losing_bids(lines: list[str], won_ids: set[str]) -> np.ndarray:
    """bid 日志中未竞胜的出价 -> 右删失点"""
    censored = []
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if len(cols) <= COL_BIDDING_PRICE:
            continue
        if cols[COL_BID_ID] in won_ids:
            continue
        try:
            b = float(cols[COL_BIDDING_PRICE]) * PRICE_SCALE
        except ValueError:
            continue
        if b > 0:
            censored.append(b)
    return np.array(censored)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp-log", type=Path, required=True)
    ap.add_argument("--bid-log", type=Path, default=None,
                    help="可选；缺省则退化为无删失校正的截断拟合（结果有偏，脚本会警告）")
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("market_params.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[1/4] 采样 imp 日志: {args.imp_log}")
    imp_lines = reservoir_sample_lines(args.imp_log, args.sample, rng)
    prices, hours, won_ids = parse_imp(imp_lines)
    print(f"      竞胜样本 {prices.size:,} 条, 市场价中位数 {np.median(prices):.2f} USD/CPM")

    censored = np.empty(0)
    if args.bid_log:
        print(f"[2/4] 采样 bid 日志: {args.bid_log}")
        bid_lines = reservoir_sample_lines(args.bid_log, args.sample, rng)
        censored = parse_losing_bids(bid_lines, won_ids)
        print(f"      竞败(右删失)样本 {censored.size:,} 条")
    else:
        print("[2/4] 警告: 未提供 bid 日志，拟合将系统性低估市场价")

    print("[3/4] 拟合")
    naive = fit_lognormal_naive(prices)
    fit = fit_lognormal_censored(prices, censored if censored.size else None)
    print(f"      朴素拟合:  mu={naive.mu:.4f} sigma={naive.sigma:.4f} "
          f"E[X]={naive.mean():.2f}")
    print(f"      删失校正:  mu={fit.mu:.4f} sigma={fit.sigma:.4f} "
          f"E[X]={fit.mean():.2f}")
    hourly = fit_hourly_multipliers(hours, prices)

    print(f"[4/4] 写出 {args.out}")
    args.out.write_text(json.dumps({
        "mu": fit.mu, "sigma": fit.sigma,
        "mu_naive": naive.mu, "sigma_naive": naive.sigma,
        "hourly_multipliers": hourly.tolist(),
        "n_observed": fit.n_observed, "n_censored": fit.n_censored,
        "price_unit": "USD_per_CPM",
        "source": str(args.imp_log.name),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
