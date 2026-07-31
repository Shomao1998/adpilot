"""看板前三页（输入向导/创意画廊/Campaign 管理）的确定性离线数据。

不依赖真实 LLM: brief 直接构造，创意用固定变体 + L1 规则审核（确定性）+ 预设
CTR 分 + 真实 Pillow 缩略图，Campaign 走真实 Mock 适配器并读取 ApiCallLog。
这些页展示的是「机器与数据结构」，文案/打分质量的真实验证在 --live 清单里。

注: 离线创意审核只跑 L1 规则引擎（违禁/绝对化/限制类目，确定性）；L2 LLM 语义
复审需真实 LLM，此处跳过 —— 与 demo_w3/demo_w4 的离线口径一致。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from adcreative.layout import CREATIVE_SIZES, render_creative
from adcreative.review import run_l1_rules
from adcreative.schema import Audience, Brief, Budget, CopyVariant, Product
from adplatform import CampaignBrief, Creative, Targeting, make_adapter
from adsim.models import ApiCallLog, Base, Platform

MICROS = 1_000_000
BRAND_COLOR = (55, 53, 47)  # Notion 近黑，缩略图配色与看板一致

# ---- 统一 brief（输入向导展示 + campaign 构建输入）----
DEMO_BRIEF = Brief(
    product=Product(name="PeakBrew 便携咖啡机", category="户外小家电",
                    selling_points=["3 分钟出浓缩", "仅重 500g", "USB-C 一次充 8 杯"],
                    price_usd=79.9),
    audience=Audience(geo=["US", "CA"], age_min=25, age_max=44,
                      interests=["camping", "coffee", "travel"]),
    budget=Budget(total_usd=3000, daily_usd=100),
    platforms=["meta", "google", "tiktok"],
    objective="conversions", brand_tone="young_energetic",
    landing_url="https://shop.example.com/portable-coffee-maker")

# ---- 固定创意变体（含 1 个绝对化用语触发 revise，展示审核状态）----
_VARIANTS = [
    CopyVariant(platform="meta", variant_id="meta-0",
                headline="Fresh Espresso, Anywhere",
                body="PeakBrew weighs just 500g. 8 cups per USB-C charge."),
    CopyVariant(platform="meta", variant_id="meta-1",
                headline="Your Campsite Coffee Upgrade",
                body="Trail-tested by thousands of campers. 30-day returns."),
    CopyVariant(platform="meta", variant_id="meta-2",
                headline="The best coffee maker for camping",   # -> L1 revise
                body="Nothing beats fresh espresso outdoors."),
    CopyVariant(platform="google", variant_id="google-0",
                rsa_headlines=["Portable Espresso Maker", "Brew In 3 Minutes",
                               "Only 500g, USB-C", "8 Cups Per Charge"],
                rsa_descriptions=["Fresh espresso anywhere you go.",
                                  "Compact design for travel and camping."]),
    CopyVariant(platform="google", variant_id="google-1",
                rsa_headlines=["Camping Coffee Gear", "Espresso On The Trail",
                               "Lightweight & USB-C"],
                rsa_descriptions=["Real crema at the campsite.",
                                  "30-day returns, ships free."]),
    CopyVariant(platform="tiktok", variant_id="tiktok-0",
                headline="POV: real espresso at 3000m altitude"),
    CopyVariant(platform="tiktok", variant_id="tiktok-1",
                headline="Wait for the crema... at a campsite?!"),
    CopyVariant(platform="tiktok", variant_id="tiktok-2",
                headline="my tent kitchen just got an upgrade"),
]
_CTR_SCORES = {  # CTR 预排序分（离线预设；--live 由 Claude 真打分）
    "meta-0": 0.86, "meta-1": 0.71, "meta-2": 0.44,
    "google-0": 0.90, "google-1": 0.68,
    "tiktok-0": 0.83, "tiktok-1": 0.77, "tiktok-2": 0.52,
}


def brief_to_dict(b: Brief) -> dict:
    return dict(
        product=dict(name=b.product.name, category=b.product.category,
                     selling_points=b.product.selling_points,
                     price_usd=b.product.price_usd),
        audience=dict(geo=b.audience.geo, age_min=b.audience.age_min,
                      age_max=b.audience.age_max, interests=b.audience.interests),
        budget=dict(total_usd=b.budget.total_usd, daily_usd=b.budget.daily_usd,
                    currency=b.budget.currency),
        platforms=b.platforms, objective=b.objective,
        brand_tone=b.brand_tone, landing_url=b.landing_url)


def build_brief_payload() -> dict:
    return brief_to_dict(DEMO_BRIEF)


_DIMS = ("visual_appeal", "clarity", "cta_strength", "platform_fit")


def _scores_of(detail: dict | None) -> dict | None:
    """从明细 dict 取四维分（翻转卡用）；缺失返回 None。"""
    if not detail:
        return None
    return {k: round(float(detail.get(k, 0.0)), 2) for k in _DIMS} | {
        "visual_from_image": bool(detail.get("visual_from_image", False))}


def pipeline_to_gallery(result, detail_map: dict, url_prefix: str) -> dict:
    """PipelineResult -> 创意画廊 payload（与 build_creatives_payload 同 schema）。

    detail_map: {variant_id: {overall, visual_appeal, clarity, cta_strength,
    platform_fit, visual_from_image}} —— 综合分 + 四维明细（翻转卡展示）。
    """
    items = []
    for a in result.approved:
        v = a.variant
        fp = a.image_paths.get("feed_1x1")
        d = detail_map.get(v.variant_id, {})
        items.append(dict(
            variant_id=v.variant_id, platform=v.platform,
            headline=v.headline, body=v.body,
            rsa_headlines=list(v.rsa_headlines),
            rsa_descriptions=list(v.rsa_descriptions), cta=v.cta,
            ctr_score=round(float(d.get("overall", 0.5)), 2),
            scores=_scores_of(d),
            verdict="pass", reasons=[],
            thumb=f"{url_prefix}/{fp.name}" if fp else None))
    for v, verdict in result.rejected:
        items.append(dict(
            variant_id=v.variant_id, platform=v.platform,
            headline=v.headline, body=v.body,
            rsa_headlines=list(v.rsa_headlines),
            rsa_descriptions=list(v.rsa_descriptions), cta=v.cta,
            ctr_score=0.0, scores=None, verdict=verdict.verdict,
            reasons=verdict.reasons, thumb=None))
    # 过审在前（按 CTR 降序），未过审在后
    items.sort(key=lambda x: (x["verdict"] != "pass", -x["ctr_score"]))
    return dict(items=items, n_total=len(items),
                n_pass=sum(1 for i in items if i["verdict"] == "pass"))


def _synth_breakdown(score: float, seed: str) -> dict:
    """离线示例的四维明细：确定性地在综合分附近抖动，均值≈综合分（保持排序）。"""
    import hashlib
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    offs = [((h >> (i * 8)) & 0xFF) / 255 * 0.24 - 0.12 for i in range(4)]  # ±0.12
    m = sum(offs) / 4
    dims = [max(0.03, min(0.99, score + o - m)) for o in offs]  # 去均值→均值保持
    return dict(zip(_DIMS, [round(d, 2) for d in dims])) | {
        "visual_from_image": False}


def build_creatives_payload(gen_dir: Path) -> dict:
    gen_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for v in _VARIANTS:
        verdict = run_l1_rules(v)
        score = _CTR_SCORES.get(v.variant_id, 0.5)
        headline = v.headline or (v.rsa_headlines[0] if v.rsa_headlines else "")
        body = v.body or (v.rsa_descriptions[0] if v.rsa_descriptions else "")
        thumb = None
        scores = None
        if verdict.verdict == "pass":
            fp = gen_dir / f"{v.variant_id}.png"
            render_creative(headline, body, v.cta, size=CREATIVE_SIZES["feed_1x1"],
                            brand_color=BRAND_COLOR, out_path=fp)
            thumb = f"/static/gen/{v.variant_id}.png"
            scores = _synth_breakdown(score, v.variant_id)
        items.append(dict(
            variant_id=v.variant_id, platform=v.platform,
            headline=headline, body=body,
            rsa_headlines=list(v.rsa_headlines),
            rsa_descriptions=list(v.rsa_descriptions),
            cta=v.cta, ctr_score=round(score, 2), scores=scores,
            verdict=verdict.verdict, reasons=verdict.reasons, thumb=thumb))
    # 过审在前（按 CTR 降序），未过审在后
    items.sort(key=lambda x: (x["verdict"] != "pass", -x["ctr_score"]))
    n_pass = sum(1 for i in items if i["verdict"] == "pass")
    return dict(items=items, n_total=len(items), n_pass=n_pass)


def live_campaign_inputs(brief, variants, score_map: dict):
    """把向导生成的过审创意整理成 campaign 构建入参：每平台取分最高的一条。"""
    by_plat: dict = {}
    for v in variants:
        by_plat.setdefault(v.platform, []).append(v)
    picks = {name: max(vs, key=lambda x: score_map.get(x.variant_id, 0.0))
             for name, vs in by_plat.items()}
    return picks, (lambda cv: score_map.get(cv.variant_id, 0.5))


def build_campaigns_payload(brief=None, creatives_by_plat=None,
                            quality_of=None) -> dict:
    """走真实 Mock 适配器建三平台 campaign 层级，读取 ApiCallLog（API 查看器数据源）。

    缺省用离线示例（DEMO_BRIEF + 固定变体）；给定 brief/creatives 时构建「投放你
    这批创意」的 campaign —— 让第 3 步与漏斗其余步骤同一个商品，不再割裂。
    """
    b = brief or DEMO_BRIEF
    if creatives_by_plat is None:
        creatives_by_plat = {"meta": _VARIANTS[0], "google": _VARIANTS[3],
                             "tiktok": _VARIANTS[5]}
    if quality_of is None:
        quality_of = lambda cv: _CTR_SCORES.get(cv.variant_id, 0.5)  # noqa: E731
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    plat_map = {"meta": Platform.META_SIM, "google": Platform.GOOGLE_SIM,
                "tiktok": Platform.TIKTOK_SIM}
    tree = []
    with Session(engine) as s:
        for name in b.platforms:
            p = plat_map.get(name)
            cv = creatives_by_plat.get(name)
            if p is None or cv is None:      # 该平台无过审创意则跳过
                continue
            ad = make_adapter(p, s)
            cref = ad.create_campaign(CampaignBrief(
                name=f"{b.product.name} · {name}",
                objective=b.objective,
                daily_budget_micros=int(b.budget.daily_usd * MICROS)))
            gref = ad.create_ad_group(cref, Targeting(
                geo=tuple(b.audience.geo), age_min=b.audience.age_min,
                age_max=b.audience.age_max,
                interests=tuple(b.audience.interests)))
            aref = ad.create_ad(gref, Creative(
                creative_id=cv.variant_id, headline=cv.headline, body=cv.body,
                landing_url=b.landing_url, quality=quality_of(cv),
                rsa_headlines=tuple(cv.rsa_headlines),
                rsa_descriptions=tuple(cv.rsa_descriptions)))
            s.commit()
            tree.append(dict(platform=name, campaign_id=cref.external_id,
                             ad_group_id=gref.external_id, ad_id=aref.external_id))
        api_log = [dict(
            platform=row.platform.value, method=row.method, endpoint=row.endpoint,
            request=json.loads(row.request_json),
            response=json.loads(row.response_json))
            for row in s.scalars(select(ApiCallLog).order_by(ApiCallLog.id))]
    return dict(tree=tree, api_log=api_log)
