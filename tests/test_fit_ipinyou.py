"""fit_ipinyou.py 脚本测试。

不依赖真实 iPinYou 数据（~15GB）：用内联 fixture 构造 make-ipinyou-data
格式的 tab 分隔日志行，覆盖解析、删失样本构造、蓄水池采样与端到端 main。
价格口径为 USD（ADR-5）：原始整数价格 * PRICE_SCALE(0.01) 直接按 USD/CPM 解读。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_ipinyou  # noqa: E402
from fit_ipinyou import (  # noqa: E402
    PRICE_SCALE,
    parse_imp,
    parse_losing_bids,
    reservoir_sample_lines,
)

N_COLS = 25  # make-ipinyou-data 行的列数下限（够覆盖 idx 20 即可）


def _make_line(bid_id: str, ts: str = "20130606120000000",
               bidding_price: str = "300", paying_price: str = "150") -> str:
    cols = ["0"] * N_COLS
    cols[fit_ipinyou.COL_BID_ID] = bid_id
    cols[fit_ipinyou.COL_TIMESTAMP] = ts
    cols[fit_ipinyou.COL_BIDDING_PRICE] = bidding_price
    cols[fit_ipinyou.COL_PAYING_PRICE] = paying_price
    return "\t".join(cols) + "\n"


class TestParseImp:
    def test_extracts_price_hour_and_ids(self):
        lines = [
            _make_line("b1", ts="20130606090000000", paying_price="150"),
            _make_line("b2", ts="20130606230000000", paying_price="80"),
        ]
        prices, hours, ids = parse_imp(lines)
        assert prices.tolist() == [150 * PRICE_SCALE, 80 * PRICE_SCALE]
        assert hours.tolist() == [9, 23]
        assert ids == {"b1", "b2"}

    def test_skips_zero_and_malformed(self):
        lines = [
            _make_line("b1", paying_price="0"),        # 非正价格
            _make_line("b2", paying_price="abc"),      # 数值解析失败
            "short\tline\n",                            # 列数不足
            _make_line("b3", ts="bad_ts"),             # 小时解析失败
            _make_line("b4", paying_price="100"),      # 唯一合法行
        ]
        prices, hours, ids = parse_imp(lines)
        assert prices.size == 1
        assert ids == {"b4"}

    def test_empty_input(self):
        prices, hours, ids = parse_imp([])
        assert prices.size == 0 and hours.size == 0 and ids == set()


class TestParseLosingBids:
    def test_won_ids_excluded(self):
        lines = [
            _make_line("w1", bidding_price="200"),  # 竞胜 -> 排除
            _make_line("l1", bidding_price="300"),  # 竞败 -> 右删失点
            _make_line("l2", bidding_price="250"),
        ]
        censored = parse_losing_bids(lines, won_ids={"w1"})
        assert sorted(censored.tolist()) == [250 * PRICE_SCALE, 300 * PRICE_SCALE]

    def test_skips_nonpositive_and_malformed(self):
        lines = [
            _make_line("l1", bidding_price="0"),
            _make_line("l2", bidding_price="x"),
            "short\n",
            _make_line("l3", bidding_price="100"),
        ]
        censored = parse_losing_bids(lines, won_ids=set())
        assert censored.tolist() == [100 * PRICE_SCALE]


class TestReservoirSample:
    def test_returns_all_when_file_smaller_than_k(self, tmp_path):
        p = tmp_path / "log.txt"
        lines = [f"line{i}\n" for i in range(10)]
        p.write_text("".join(lines))
        out = reservoir_sample_lines(p, k=100, rng=np.random.default_rng(0))
        assert out == lines

    def test_sample_size_and_membership(self, tmp_path):
        p = tmp_path / "log.txt"
        universe = [f"line{i}\n" for i in range(1000)]
        p.write_text("".join(universe))
        out = reservoir_sample_lines(p, k=50, rng=np.random.default_rng(0))
        assert len(out) == 50
        assert set(out) <= set(universe)

    def test_sampling_is_approximately_uniform(self, tmp_path):
        """前半段行被选中的比例应接近 50%（蓄水池无位置偏倚）。"""
        p = tmp_path / "log.txt"
        p.write_text("".join(f"{i}\n" for i in range(2000)))
        rng = np.random.default_rng(42)
        picks_front = 0
        rounds, k = 200, 20
        for _ in range(rounds):
            out = reservoir_sample_lines(p, k=k, rng=rng)
            picks_front += sum(int(x) < 1000 for x in out)
        frac = picks_front / (rounds * k)
        assert frac == pytest.approx(0.5, abs=0.05)


class TestMainEndToEnd:
    def _write_logs(self, tmp_path, rng, n=3000, mu=4.0, sigma=0.8):
        """按真实生成过程造日志: 市场价 ~ LogNormal; 赢->imp 行, 输->bid 行。
        价格写入前 /PRICE_SCALE 反缩放，保证解析后回到原尺度。"""
        market = rng.lognormal(mu, sigma, size=n)
        bids = rng.lognormal(mu, 0.5, size=n)
        imp_lines, bid_lines = [], []
        for i, (m, b) in enumerate(zip(market, bids)):
            bid_id = f"id{i}"
            hour = int(rng.integers(0, 24))
            ts = f"20130606{hour:02d}00000000"[:17]
            bid_lines.append(_make_line(bid_id, ts=ts,
                                        bidding_price=str(int(b / PRICE_SCALE))))
            if b > m:
                imp_lines.append(_make_line(bid_id, ts=ts,
                                            paying_price=str(int(m / PRICE_SCALE))))
        imp = tmp_path / "imp.txt"
        bid = tmp_path / "bid.txt"
        imp.write_text("".join(imp_lines))
        bid.write_text("".join(bid_lines))
        return imp, bid

    def test_main_writes_usd_params(self, tmp_path, monkeypatch, capsys):
        rng = np.random.default_rng(7)
        imp, bid = self._write_logs(tmp_path, rng)
        out = tmp_path / "params.json"
        monkeypatch.setattr(sys, "argv", [
            "fit_ipinyou.py", "--imp-log", str(imp), "--bid-log", str(bid),
            "--sample", "10000", "--out", str(out),
        ])
        fit_ipinyou.main()

        params = json.loads(out.read_text())
        assert params["price_unit"] == "USD_per_CPM"
        assert params["n_censored"] > 0
        assert len(params["hourly_multipliers"]) == 24
        # 删失校正应把均值拉回到朴素拟合之上（方向性断言，容忍采样噪声）
        assert params["mu"] > params["mu_naive"]
        # 拟合应大致恢复造数用的真值
        assert params["mu"] == pytest.approx(4.0, abs=0.15)
        assert "USD/CPM" in capsys.readouterr().out

    def test_main_without_bid_log_warns(self, tmp_path, monkeypatch, capsys):
        rng = np.random.default_rng(7)
        imp, _ = self._write_logs(tmp_path, rng, n=500)
        out = tmp_path / "params.json"
        monkeypatch.setattr(sys, "argv", [
            "fit_ipinyou.py", "--imp-log", str(imp), "--out", str(out),
        ])
        fit_ipinyou.main()
        assert "警告" in capsys.readouterr().out
        assert json.loads(out.read_text())["n_censored"] == 0
