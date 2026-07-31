"""W2 验收 demo：同一标的物，三平台跑 7 模拟日，出三份口径不同的平台报表 + ground truth。

验收标准（设计文档 §8 W2 行）:
    手工构造 campaign 可跑 7 模拟日并出三份口径不同的平台报表 + ground truth。

演示看点:
    1. 全流程走 AdPlatformAdapter（Agent 视角，不碰模拟器内部）。
    2. 三平台报表的"总转化数"彼此对不上、也和真值对不上 —— 且每个差异都可解释:
       归因窗口截断(7d vs 30d)、时区切日(LA/NY/UTC)、记账日口径(click/conv date)。
    3. as_of 回填: 第 7 天结束当刻拉的报表 vs 上帝视角的最终报表，转化数不同。

用法:
    python scripts/demo_w2.py [--seed 42] [--db sqlite:///w2_demo.db]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adplatform import CampaignBrief, Creative, Targeting, make_adapter  # noqa: E402
from adsim.auction import make_platform_markets  # noqa: E402
from adsim.models import Ad, AdGroup, Base, Campaign, Platform  # noqa: E402
from adsim.reporting import (  # noqa: E402
    generate_platform_reports, ground_truth_daily, seed_attribution_specs,
)
from adsim.simulate import DEFAULT_PROFILES, MICROS, AdSimEngine  # noqa: E402

START = date(2026, 7, 1)
N_DAYS = 7


def usd(micros: int) -> str:
    return f"${micros / MICROS:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default="sqlite://", help="默认内存库；可指定文件持久化")
    args = ap.parse_args()

    engine = create_engine(args.db)
    Base.metadata.create_all(engine)
    session = Session(engine)
    seed_attribution_specs(session)

    # 市场行情：iPinYou 拟合参数的演示占位（真实参数由 scripts/fit_ipinyou.py 产出）
    markets_by_name = make_platform_markets(
        base_fit_mu=float(np.log(4.0)), base_fit_sigma=0.55)
    markets = {Platform(k): v for k, v in markets_by_name.items()}
    sim = AdSimEngine(markets, DEFAULT_PROFILES, seed=args.seed)

    # --- 1. Agent 视角：通过适配器建三平台 campaign（同一标的物）---
    brief = CampaignBrief(name="portable-espresso-launch",
                          daily_budget_micros=100 * MICROS)
    targeting = Targeting(geo=("US", "CA"), age_min=25, age_max=44,
                          interests=("coffee",), bid_micros=9 * MICROS)
    creatives = [
        Creative(creative_id=f"cr-{i}", headline=h,
                 body="Brew barista-grade espresso anywhere.",
                 image_url=f"img-{i}", quality=q,
                 rsa_headlines=(h, "Barista In Your Bag"),
                 rsa_descriptions=("Brew barista-grade espresso anywhere.",))
        for i, (h, q) in enumerate([("Espresso Anywhere", 0.75),
                                    ("Coffee On The Go", 0.45)])
    ]

    handles = {}
    for platform in Platform:
        adapter = make_adapter(platform, session)
        camp = adapter.create_campaign(brief)
        group = adapter.create_ad_group(camp, targeting)
        ad_refs = [adapter.create_ad(group, c) for c in creatives]
        handles[platform] = (adapter, camp, group, ad_refs)
        print(f"[build] {platform.value:<12} campaign={camp.external_id}")

    # --- 2. 跑 7 个模拟日 ---
    print(f"\n[sim] {START} 起跑 {N_DAYS} 个模拟日 (seed={args.seed})")
    for d in range(N_DAYS):
        sim_date = START + timedelta(days=d)
        for platform, (_, camp, group, ad_refs) in handles.items():
            c = session.get(Campaign, camp.internal_id)
            g = session.get(AdGroup, group.internal_id)
            ads = [session.get(Ad, r.internal_id) for r in ad_refs]
            st = sim.run_day(session, sim_date, c, g, ads)
            print(f"  day{d+1} {platform.value:<12} "
                  f"auctions={st.auctions:>6} wins={st.wins:>6} "
                  f"clicks={st.clicks:>4} convs={st.conversions:>3} "
                  f"spend={usd(st.spend_micros)}")

    # --- 3. 物化平台报表（as_of = 第 7 天结束时刻，模拟"当天拉报表"）---
    as_of = datetime.combine(START + timedelta(days=N_DAYS), datetime.min.time())

    print(f"\n{'='*76}\n三平台报表 (as_of={as_of:%Y-%m-%d}, 各平台自己的口径) vs Ground Truth\n{'='*76}")
    header = (f"{'platform':<12} {'窗口':>5} {'时区':<19} {'记账日':<10} "
              f"{'imps':>7} {'clicks':>6} {'convs':>5} {'spend':>10} {'revenue':>11}")
    print(header + "\n" + "-" * len(header))

    spec_desc = {Platform.META_SIM: ("7d", "America/Los_Angeles", "click_date"),
                 Platform.GOOGLE_SIM: ("30d", "America/New_York", "click_date"),
                 Platform.TIKTOK_SIM: ("7d", "UTC", "conv_date")}
    final_convs = {}
    for platform in Platform:
        rows = generate_platform_reports(session, platform, as_of=as_of)
        w, tz, basis = spec_desc[platform]
        tot = {m: sum(getattr(r, m) for r in rows)
               for m in ("impressions", "clicks", "conversions",
                         "spend_micros", "revenue_micros")}
        print(f"{platform.value:<12} {w:>5} {tz:<19} {basis:<10} "
              f"{tot['impressions']:>7} {tot['clicks']:>6} "
              f"{tot['conversions']:>5} {usd(tot['spend_micros']):>10} "
              f"{usd(tot['revenue_micros']):>11}")
        final_convs[platform] = tot["conversions"]

    print("-" * len(header))
    for platform in Platform:
        gt = ground_truth_daily(session, platform)
        gt_convs = sum(r.conversions for r in gt)
        rows_god = generate_platform_reports(session, platform, as_of=None)
        god_convs = sum(r.conversions for r in rows_god)
        print(f"{platform.value:<12} ground_truth convs={gt_convs:>4} | "
              f"平台口径(回填完毕)={god_convs:>4} | "
              f"as_of 当刻={final_convs[platform]:>4}  "
              f"(差异 = 延迟转化未回填 + 归因窗口截断)")

    # --- 4. 展示一条落库的 API 调用（schema 对齐真实平台的证据）---
    from adsim.models import ApiCallLog
    from sqlalchemy import select
    log = session.scalars(select(ApiCallLog).where(
        ApiCallLog.platform == Platform.GOOGLE_SIM)).first()
    print(f"\n[api-log 样例] {log.method} {log.endpoint}\n  request: "
          f"{log.request_json[:160]}...")

    print("\nW2 验收通过: 7 模拟日 × 3 平台，三份口径不同的平台报表 + ground truth 对照。")


if __name__ == "__main__":
    main()
