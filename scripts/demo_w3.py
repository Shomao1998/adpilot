"""W3 验收演示: 商品 URL -> Brief -> 分平台创意 -> 审核回流 -> ≥6 组过审 + 三尺寸图。

用法:
    离线模式（默认, 脚本化 FakeLLM, 零 API 依赖）:
        python scripts/demo_w3.py
    真实 API 模式（需 pip install anthropic + ANTHROPIC_API_KEY）:
        python scripts/demo_w3.py --live https://your-shop.example.com/product

离线模式的价值: 完整走一遍 URL 抓页 -> 意图解析 -> 生成 -> L1/L2 审核 ->
revise 回流 -> 排版出图 的全链路（含一个故意埋的绝对化用语触发回流），
验证管线机器本身；文案质量的验收留给 --live。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adcreative import (  # noqa: E402
    Brief, Budget, FakeLLM, generate_creatives, parse_brief,
)
from adcreative.schema import (  # noqa: E402
    Audience, CopyBatch, CopyVariant, Product, ReviewVerdict,
)

DEMO_URL = "https://shop.example.com/portable-coffee-maker"
FAKE_PAGE = """<html><title>PeakBrew 便携咖啡机 - $79.9</title>
<body>3 分钟出浓缩咖啡。仅重 500g。USB-C 充电，一次充电做 8 杯。
户外露营、办公差旅皆宜。30 天无理由退换。</body></html>"""


def _offline_llm() -> FakeLLM:
    """脚本化响应序列: 与管线调用顺序严格对齐。
    meta 第 1 轮埋一个 "best"（L1 拦下 revise, 不进 L2）触发回流重试。"""
    brief = Brief(
        product=Product(name="PeakBrew 便携咖啡机", category="户外小家电",
                        selling_points=["3分钟出浓缩", "仅重500g",
                                        "USB-C充电一次8杯"], price_usd=79.9),
        audience=Audience(geo=["US", "CA"], age_min=25, age_max=44,
                          interests=["camping", "coffee", "travel"]),
        budget=Budget(total_usd=3000, daily_usd=100),
        brand_tone="young_energetic", landing_url=DEMO_URL)

    def meta(i, hl, body):
        return CopyVariant(platform="meta", headline=hl, body=body)

    meta_r0 = CopyBatch(variants=[
        meta(0, "Espresso In 3 Minutes, Anywhere",
             "PeakBrew weighs just 500g. 8 cups per USB-C charge."),
        meta(1, "Your Campsite Coffee Upgrade",
             "Trail-tested by thousands of campers. 30-day returns."),
        meta(2, "The best coffee maker for camping",       # <- L1 revise
             "Nothing beats fresh espresso outdoors."),
    ])
    meta_r1 = CopyBatch(variants=[
        meta(0, "Fresh Espresso On Every Trail",
             "3-minute brew, 500g light. Loved by campers across the US."),
    ])
    google = CopyBatch(variants=[CopyVariant(
        platform="google",
        rsa_headlines=["Portable Espresso Maker", "Brew In 3 Minutes",
                       "Only 500g, USB-C Powered", "Camping Coffee Gear",
                       "8 Cups Per Charge"],
        rsa_descriptions=["Fresh espresso anywhere: campsite, office, or hotel.",
                          "500g portable coffee maker with USB-C. 30-day returns."],
    ) for _ in range(3)])
    tiktok = CopyBatch(variants=[CopyVariant(
        platform="tiktok",
        headline=h, body="PeakBrew. 500g. USB-C. Real espresso.")
        for h in ("POV: real espresso at 3000m altitude",
                  "Wait for the crema... at a campsite?!",
                  "My tent kitchen just got an upgrade")])

    P = ReviewVerdict(verdict="pass")
    return FakeLLM([
        brief,                      # 意图解析
        meta_r0, P, P,              # meta r0: 2 过审, 第 3 个被 L1 revise(无LLM调用)
        meta_r1, P,                 # meta r1 回流: 补 1 过审
        google, P, P, P,            # google: 3 过审
        tiktok, P, P, P,            # tiktok: 3 过审
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default=DEMO_URL)
    ap.add_argument("--live", action="store_true",
                    help="走真实 Claude API（需 anthropic 包与 API key）")
    ap.add_argument("--out", type=Path, default=Path("demo_w3_out"))
    args = ap.parse_args()

    if args.live:
        from adcreative import ClaudeLLM
        llm = ClaudeLLM()
        fetch = None            # 真实抓页
        mode = "LIVE (claude-opus-4-8)"
    else:
        llm = _offline_llm()
        fetch = lambda url: FAKE_PAGE  # noqa: E731
        mode = "OFFLINE (FakeLLM 脚本化)"

    print(f"[mode] {mode}")
    print(f"[1/3] 意图解析: {args.url}")
    brief, questions = parse_brief(args.url, llm, fetch_page=fetch)
    print(f"      产品: {brief.product.name} | 预算: ${brief.budget.total_usd:.0f} "
          f"(${brief.budget.daily_usd:.0f}/日) | 受众: {','.join(brief.audience.geo)} "
          f"{brief.audience.age_min}-{brief.audience.age_max}")
    for q in questions:
        print(f"      [追问] {q}")

    print("[2/3] 创意生成 + 审核（revise 回流 ≤2 轮）")
    result = generate_creatives(brief, llm, n_variants=3, render_dir=args.out)
    for platform, rounds in result.rounds_used.items():
        n_ok = sum(1 for a in result.approved if a.variant.platform == platform)
        print(f"      {platform:<8} 过审 {n_ok} 组, 用了 {rounds} 轮生成")
    for v, verdict in result.rejected:
        print(f"      [未过审] {v.variant_id}: {verdict.verdict} "
              f"({'; '.join(verdict.reasons)})")

    n_img = sum(len(a.image_paths) for a in result.approved)
    print(f"[3/3] 排版: {n_img} 张图 -> {args.out}/")
    print("=" * 60)
    status = "通过" if result.ok else "未达标"
    print(f"W3 验收{status}: 过审创意 {len(result.approved)} 组 (要求 ≥6)")


if __name__ == "__main__":
    main()
