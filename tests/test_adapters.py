"""适配器层测试。

断言分三层:
    1. 统一接口行为: 三个 Mock 对同一组调用产生同构的实体行（Campaign/AdGroup/Ad），
       Agent 侧换平台零改动 (ADR-3)。
    2. schema 对齐: 每个 Mock 落库的请求体必须带上对应真实 API 的标志性字段与
       金额单位（Meta 分 / Google micros / TikTok 美元浮点）—— 这是 ADR-2 的验收。
    3. 与模拟器/报表生成器的集成: 适配器建的 campaign 能直接被引擎消费，
       fetch_report 返回各平台自己的响应形状。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from adplatform import (
    CampaignBrief, Creative, Targeting, make_adapter,
)
from adsim.auction import MarketModel
from adsim.models import (
    Ad, AdGroup, ApiCallLog, Base, Campaign, Platform,
)
from adsim.reporting import generate_platform_reports, seed_attribution_specs
from adsim.simulate import (
    MICROS, AdSimEngine, DEFAULT_PROFILES, usd_to_micros,
)

BRIEF = CampaignBrief(name="portable-espresso-launch",
                      daily_budget_micros=150 * MICROS)
TARGETING = Targeting(geo=("US", "CA"), age_min=25, age_max=44,
                      interests=("coffee",), bid_micros=8 * MICROS)
CREATIVE = Creative(
    creative_id="cr-001", headline="Espresso Anywhere",
    body="Brew barista-grade shots on the go.",
    image_url="img-001", quality=0.7,
    rsa_headlines=("Espresso Anywhere", "Barista In Your Bag"),
    rsa_descriptions=("Brew barista-grade shots on the go.",),
)


def _fresh_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    seed_attribution_specs(s)
    return s


def _build_full_structure(s, platform):
    adapter = make_adapter(platform, s)
    camp = adapter.create_campaign(BRIEF)
    group = adapter.create_ad_group(camp, TARGETING)
    ad = adapter.create_ad(group, CREATIVE)
    return adapter, camp, group, ad


class TestUniformInterface:
    @pytest.mark.parametrize("platform", list(Platform))
    def test_full_structure_and_entities(self, platform):
        s = _fresh_session()
        adapter, camp, group, ad = _build_full_structure(s, platform)

        row = s.get(Campaign, camp.internal_id)
        assert row.platform == platform
        assert row.daily_budget_micros == BRIEF.daily_budget_micros
        assert s.get(AdGroup, group.internal_id).bid_micros == TARGETING.bid_micros
        assert s.get(Ad, ad.internal_id).creative_quality == CREATIVE.quality

        # 层级 create ×3 各留一条 API 日志
        logs = s.scalars(select(ApiCallLog)).all()
        assert len(logs) == 3
        assert all(log.platform == platform for log in logs)

    @pytest.mark.parametrize("platform", list(Platform))
    def test_update_budget_and_pause(self, platform):
        s = _fresh_session()
        adapter, camp, group, ad = _build_full_structure(s, platform)
        assert adapter.update_budget(camp, 300 * MICROS).success
        assert s.get(Campaign, camp.internal_id).daily_budget_micros == 300 * MICROS
        assert adapter.pause_ad(ad).success
        assert s.get(Ad, ad.internal_id).status == "paused"


class TestSchemaAlignment:
    """请求体必须长得像真实 API —— 字段名与金额单位是硬标准。"""

    def _campaign_request(self, s, platform):
        _build_full_structure(s, platform)
        log = s.scalars(select(ApiCallLog).where(
            ApiCallLog.platform == platform)).first()
        return json.loads(log.request_json), log.endpoint

    def test_meta_campaign_request(self):
        s = _fresh_session()
        req, endpoint = self._campaign_request(s, Platform.META_SIM)
        assert "/campaigns" in endpoint and endpoint.startswith("/act_")
        assert req["objective"] == "OUTCOME_SALES"       # v21 ODAX 枚举
        assert "special_ad_categories" in req            # v21 必填
        assert req["daily_budget"] == 150 * 100          # 单位: 分

    def test_google_campaign_request(self):
        s = _fresh_session()
        req, endpoint = self._campaign_request(s, Platform.GOOGLE_SIM)
        assert endpoint.endswith(":mutate") and "customers/" in endpoint
        create = req["operations"][0]["create"]
        assert create["advertisingChannelType"] == "SEARCH"
        assert create["campaignBudgetAmountMicros"] == 150 * MICROS  # micros
        assert create["campaignBudget"].startswith("customers/")     # 资源名引用

    def test_tiktok_campaign_request_and_envelope(self):
        s = _fresh_session()
        req, endpoint = self._campaign_request(s, Platform.TIKTOK_SIM)
        assert endpoint == "/open_api/v1.3/campaign/create/"
        assert req["budget_mode"] == "BUDGET_MODE_DAY"
        assert req["budget"] == pytest.approx(150.0)     # 美元浮点
        log = s.scalars(select(ApiCallLog)).first()
        resp = json.loads(log.response_json)
        assert resp["code"] == 0 and "data" in resp      # 统一响应包

    def test_google_rsa_asset_limits_enforced(self):
        s = _fresh_session()
        adapter = make_adapter(Platform.GOOGLE_SIM, s)
        camp = adapter.create_campaign(BRIEF)
        group = adapter.create_ad_group(camp, TARGETING)
        too_long = Creative(creative_id="bad",
                            rsa_headlines=("x" * 31,),          # >30 字符
                            rsa_descriptions=("ok",))
        with pytest.raises(ValueError, match="headlines"):
            adapter.create_ad(group, too_long)


class TestIntegrationWithEngine:
    def test_adapter_campaign_runs_and_reports(self):
        """适配器建的 campaign 直接被引擎消费 -> 报表 -> 三平台各自响应形状。"""
        s = _fresh_session()
        market = {p: MarketModel(mu=float(np.log(5.0)), sigma=0.6, floor=0.01)
                  for p in Platform}
        eng = AdSimEngine(market, DEFAULT_PROFILES, seed=21)

        reports = {}
        for platform in Platform:
            adapter, camp, group, ad = _build_full_structure(s, platform)
            c = s.get(Campaign, camp.internal_id)
            g = s.get(AdGroup, group.internal_id)
            ads = [s.get(Ad, ad.internal_id)]
            st = eng.run_day(s, date(2026, 7, 1), c, g, ads)
            assert st.wins > 0
            generate_platform_reports(s, platform)
            reports[platform] = adapter.fetch_report(
                camp, date(2026, 6, 30), date(2026, 7, 2))

        # 各平台响应形状不同 —— 这本身就是归一化层存在的理由
        meta = reports[Platform.META_SIM]
        assert isinstance(meta["data"][0]["impressions"], str)
        assert meta["data"][0]["actions"][0]["action_type"].startswith(
            "offsite_conversion")
        google = reports[Platform.GOOGLE_SIM]
        assert "costMicros" in google["results"][0]["metrics"]
        tiktok = reports[Platform.TIKTOK_SIM]
        assert tiktok["code"] == 0
        assert "stat_time_day" in tiktok["data"]["list"][0]["dimensions"]
