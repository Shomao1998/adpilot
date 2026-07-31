"""AdPlatformAdapter：统一接口 + 三平台 Mock 实现。

设计要点 (ADR-2):
    - 接口是统一的领域语言（campaign / ad_group / ad），但每个 Mock 构造的
      请求体字段名、层级、枚举值、金额单位 **照抄对应真实平台文档**:

        Meta Marketing API v21   POST /act_{account}/campaigns|adsets|ads
            金额单位: 分 (cents)；层级叫 adset；定向嵌在 adset.targeting。
        Google Ads API v17       customers/{cid}/campaigns|adGroups|adGroupAds
            金额单位: micros；资源用 resource_name 引用；广告是 RSA 资产结构
            （headlines<=15 条且各<=30 字符 / descriptions<=4 条且各<=90 字符）。
        TikTok Business API v1.3 /campaign|adgroup|ad/create/
            金额单位: 美元浮点；响应统一包 {"code":0,"message":"OK","data":...}。

    - 每次调用的请求/响应全量写入 ApiCallLog（UI 的 API 查看器数据源，
      也是"读过真实文档"的直接证据）。
    - Mock 的"平台后端"就是模拟器的实体表: create_* 落 Campaign/AdGroup/Ad 行，
      fetch_report 读报表生成器物化的 PlatformDailyReport，并按各平台自己的
      响应格式返回（Meta insights 的数值是字符串、转化藏在 actions 列表里；
      Google 是 searchStream 的 results/metrics；TikTok 是 data.list）——
      连"报表响应长什么样"的口径差异也一并还原。

金额口径 (ADR-5): 接口层统一 micros 整数；各 Mock 在构造请求体时才换算成
    平台自己的单位（cents / micros / 美元浮点），换算只发生在边界上。
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from adsim.models import (
    Ad,
    AdGroup,
    ApiCallLog,
    Campaign,
    Platform,
    PlatformDailyReport,
)

MICROS = 1_000_000


# ---------------------------------------------------------------------------
# 领域对象（Agent 侧的统一语言，与平台无关）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CampaignBrief:
    name: str
    objective: str = "conversions"
    daily_budget_micros: int = 100 * MICROS


@dataclass(frozen=True)
class Targeting:
    geo: tuple[str, ...] = ("US",)
    age_min: int = 18
    age_max: int = 65
    interests: tuple[str, ...] = ()
    bid_micros: int = 8 * MICROS  # CPM 口径


@dataclass(frozen=True)
class Creative:
    creative_id: str
    headline: str = ""
    body: str = ""
    image_url: str = ""
    landing_url: str = "https://example.com"
    quality: float = 0.5  # CTR 预排序分数，模拟器点击模型的输入
    # Google RSA 资产（其余平台忽略）
    rsa_headlines: tuple[str, ...] = ()
    rsa_descriptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CampaignRef:
    platform: Platform
    external_id: str
    internal_id: int


@dataclass(frozen=True)
class AdGroupRef:
    platform: Platform
    external_id: str
    internal_id: int
    campaign_internal_id: int


@dataclass(frozen=True)
class AdRef:
    platform: Platform
    external_id: str
    internal_id: int


@dataclass(frozen=True)
class Result:
    success: bool
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------

class AdPlatformAdapter(ABC):
    """Agent 侧唯一依赖的接口。Mock 与真实实现互换时 Agent 代码零改动 (ADR-3)。"""

    platform: Platform

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create_campaign(self, brief: CampaignBrief) -> CampaignRef: ...

    @abstractmethod
    def create_ad_group(
        self, campaign: CampaignRef, targeting: Targeting
    ) -> AdGroupRef: ...

    @abstractmethod
    def create_ad(self, group: AdGroupRef, creative: Creative) -> AdRef: ...

    @abstractmethod
    def update_budget(
        self, campaign: CampaignRef, daily_budget_micros: int
    ) -> Result: ...

    @abstractmethod
    def pause_ad(self, ad: AdRef) -> Result: ...

    @abstractmethod
    def fetch_report(
        self, campaign: CampaignRef, date_from: date, date_to: date
    ) -> dict: ...

    # ---- 共用工具 ----

    def _log(self, method: str, endpoint: str, request: dict, response: dict) -> None:
        self.session.add(ApiCallLog(
            platform=self.platform, method=method, endpoint=endpoint,
            request_json=json.dumps(request, ensure_ascii=False, sort_keys=True),
            response_json=json.dumps(response, ensure_ascii=False, sort_keys=True),
        ))
        self.session.commit()

    def _report_rows(
        self, campaign: CampaignRef, date_from: date, date_to: date
    ) -> list[PlatformDailyReport]:
        return list(self.session.scalars(
            select(PlatformDailyReport).where(
                PlatformDailyReport.platform == self.platform,
                PlatformDailyReport.campaign_id == campaign.internal_id,
                PlatformDailyReport.report_date_local >= date_from.isoformat(),
                PlatformDailyReport.report_date_local <= date_to.isoformat(),
            ).order_by(PlatformDailyReport.report_date_local)
        ))


# ---------------------------------------------------------------------------
# Meta Mock（Marketing API v21: campaign -> adset -> ad，金额单位: 分）
# ---------------------------------------------------------------------------

class MockMetaAdapter(AdPlatformAdapter):
    platform = Platform.META_SIM
    ACCOUNT = "act_10152000000000000"

    def create_campaign(self, brief: CampaignBrief) -> CampaignRef:
        request = {
            "name": brief.name,
            # Meta v21 的 ODAX objective 枚举
            "objective": "OUTCOME_SALES" if brief.objective == "conversions"
                         else "OUTCOME_TRAFFIC",
            "status": "ACTIVE",
            "special_ad_categories": [],   # 必填字段，即便为空
            "daily_budget": brief.daily_budget_micros // 10_000,  # micros -> 分
        }
        row = Campaign(platform=self.platform, name=brief.name,
                       objective=brief.objective,
                       daily_budget_micros=brief.daily_budget_micros,
                       external_ref=f"1207{uuid.uuid4().int % 10**13:013d}")
        self.session.add(row); self.session.commit()
        response = {"id": row.external_ref}
        self._log("POST", f"/{self.ACCOUNT}/campaigns", request, response)
        return CampaignRef(self.platform, row.external_ref, row.id)

    def create_ad_group(self, campaign: CampaignRef, t: Targeting) -> AdGroupRef:
        request = {
            "name": "adset-1",
            "campaign_id": campaign.external_id,
            "billing_event": "IMPRESSIONS",
            "optimization_goal": "OFFSITE_CONVERSIONS",
            "bid_amount": t.bid_micros // 10_000,  # 分
            "targeting": {
                "geo_locations": {"countries": list(t.geo)},
                "age_min": t.age_min,
                "age_max": t.age_max,
                "flexible_spec": [
                    {"interests": [{"name": i} for i in t.interests]}
                ] if t.interests else [],
            },
            "status": "ACTIVE",
        }
        row = AdGroup(campaign_id=campaign.internal_id, name=request["name"],
                      targeting_json=json.dumps(request["targeting"]),
                      bid_micros=t.bid_micros)
        self.session.add(row); self.session.commit()
        ext = f"2384{uuid.uuid4().int % 10**13:013d}"
        self._log("POST", f"/{self.ACCOUNT}/adsets", request, {"id": ext})
        return AdGroupRef(self.platform, ext, row.id, campaign.internal_id)

    def create_ad(self, group: AdGroupRef, c: Creative) -> AdRef:
        request = {
            "name": f"ad-{c.creative_id}",
            "adset_id": group.external_id,
            "creative": {
                "object_story_spec": {
                    "link_data": {
                        "message": c.body,
                        "name": c.headline,
                        "picture": c.image_url,
                        "link": c.landing_url,
                        "call_to_action": {"type": "SHOP_NOW"},
                    }
                }
            },
            "status": "ACTIVE",
        }
        row = Ad(ad_group_id=group.internal_id, creative_id=c.creative_id,
                 creative_quality=c.quality)
        self.session.add(row); self.session.commit()
        ext = f"6047{uuid.uuid4().int % 10**13:013d}"
        self._log("POST", f"/{self.ACCOUNT}/ads", request, {"id": ext})
        return AdRef(self.platform, ext, row.id)

    def update_budget(self, campaign: CampaignRef, daily_budget_micros: int) -> Result:
        request = {"daily_budget": daily_budget_micros // 10_000}
        row = self.session.get(Campaign, campaign.internal_id)
        row.daily_budget_micros = daily_budget_micros
        self.session.commit()
        self._log("POST", f"/{campaign.external_id}", request, {"success": True})
        return Result(True, {"daily_budget_micros": daily_budget_micros})

    def pause_ad(self, ad: AdRef) -> Result:
        row = self.session.get(Ad, ad.internal_id)
        row.status = "paused"
        self.session.commit()
        self._log("POST", f"/{ad.external_id}",
                  {"status": "PAUSED"}, {"success": True})
        return Result(True)

    def fetch_report(self, campaign: CampaignRef, date_from: date, date_to: date) -> dict:
        """Meta insights 口径：数值全是字符串、金额是美元、转化藏在 actions。"""
        rows = self._report_rows(campaign, date_from, date_to)
        data = [{
            "date_start": r.report_date_local,
            "date_stop": r.report_date_local,
            "impressions": str(r.impressions),
            "clicks": str(r.clicks),
            "spend": f"{r.spend_micros / MICROS:.2f}",
            "actions": [{"action_type": "offsite_conversion.fb_pixel_purchase",
                         "value": str(r.conversions)}],
            "action_values": [{"action_type": "offsite_conversion.fb_pixel_purchase",
                               "value": f"{r.revenue_micros / MICROS:.2f}"}],
        } for r in rows]
        response = {"data": data, "paging": {"cursors": {}}}
        self._log("GET", f"/{campaign.external_id}/insights",
                  {"time_range": {"since": date_from.isoformat(),
                                  "until": date_to.isoformat()},
                   "time_increment": 1}, response)
        return response


# ---------------------------------------------------------------------------
# Google Mock（Ads API: campaign -> ad group -> RSA，金额单位: micros）
# ---------------------------------------------------------------------------

class MockGoogleAdapter(AdPlatformAdapter):
    platform = Platform.GOOGLE_SIM
    CUSTOMER = "customers/4590000000"

    RSA_MAX_HEADLINES = 15
    RSA_MAX_HEADLINE_LEN = 30
    RSA_MAX_DESCRIPTIONS = 4
    RSA_MAX_DESCRIPTION_LEN = 90

    def create_campaign(self, brief: CampaignBrief) -> CampaignRef:
        ext = f"{self.CUSTOMER}/campaigns/{uuid.uuid4().int % 10**10}"
        request = {
            "operations": [{"create": {
                "name": brief.name,
                "advertisingChannelType": "SEARCH",
                "status": "ENABLED",
                "campaignBudget": f"{self.CUSTOMER}/campaignBudgets/"
                                  f"{uuid.uuid4().int % 10**10}",
                # Google 原生就是 micros —— 与 ADR-5 的存储口径同源
                "campaignBudgetAmountMicros": brief.daily_budget_micros,
                "biddingStrategyType": "MAXIMIZE_CONVERSIONS",
            }}]
        }
        row = Campaign(platform=self.platform, name=brief.name,
                       objective=brief.objective,
                       daily_budget_micros=brief.daily_budget_micros,
                       external_ref=ext)
        self.session.add(row); self.session.commit()
        response = {"results": [{"resourceName": ext}]}
        self._log("POST", f"{self.CUSTOMER}/campaigns:mutate", request, response)
        return CampaignRef(self.platform, ext, row.id)

    def create_ad_group(self, campaign: CampaignRef, t: Targeting) -> AdGroupRef:
        ext = f"{self.CUSTOMER}/adGroups/{uuid.uuid4().int % 10**10}"
        request = {
            "operations": [{"create": {
                "name": "adgroup-1",
                "campaign": campaign.external_id,
                "type": "SEARCH_STANDARD",
                "status": "ENABLED",
                "cpcBidMicros": t.bid_micros,
                # 搜索定向以 criteria 单独 mutate，此处内联摘要（Mock 简化）
                "_targetingCriteria": {
                    "geoTargetConstants": list(t.geo),
                    "ageRanges": [f"AGE_RANGE_{t.age_min}_{t.age_max}"],
                },
            }}]
        }
        row = AdGroup(campaign_id=campaign.internal_id, name="adgroup-1",
                      targeting_json=json.dumps(request["operations"][0]["create"]
                                                ["_targetingCriteria"]),
                      bid_micros=t.bid_micros)
        self.session.add(row); self.session.commit()
        self._log("POST", f"{self.CUSTOMER}/adGroups:mutate", request,
                  {"results": [{"resourceName": ext}]})
        return AdGroupRef(self.platform, ext, row.id, campaign.internal_id)

    def create_ad(self, group: AdGroupRef, c: Creative) -> AdRef:
        headlines = list(c.rsa_headlines) or [c.headline]
        descriptions = list(c.rsa_descriptions) or [c.body]
        # RSA 资产硬约束：超限直接拒绝 —— 真实 API 也会报错，创意生成 Agent
        # 必须按平台结构产出（设计文档 4.2）
        if (len(headlines) > self.RSA_MAX_HEADLINES
                or any(len(h) > self.RSA_MAX_HEADLINE_LEN for h in headlines)):
            raise ValueError("RSA headlines 超限: 最多 15 条、每条 <= 30 字符")
        if (len(descriptions) > self.RSA_MAX_DESCRIPTIONS
                or any(len(d) > self.RSA_MAX_DESCRIPTION_LEN for d in descriptions)):
            raise ValueError("RSA descriptions 超限: 最多 4 条、每条 <= 90 字符")

        ext = f"{self.CUSTOMER}/adGroupAds/{uuid.uuid4().int % 10**10}"
        request = {
            "operations": [{"create": {
                "adGroup": group.external_id,
                "status": "ENABLED",
                "ad": {
                    "finalUrls": [c.landing_url],
                    "responsiveSearchAd": {
                        "headlines": [{"text": h} for h in headlines],
                        "descriptions": [{"text": d} for d in descriptions],
                    },
                },
            }}]
        }
        row = Ad(ad_group_id=group.internal_id, creative_id=c.creative_id,
                 creative_quality=c.quality)
        self.session.add(row); self.session.commit()
        self._log("POST", f"{self.CUSTOMER}/adGroupAds:mutate", request,
                  {"results": [{"resourceName": ext}]})
        return AdRef(self.platform, ext, row.id)

    def update_budget(self, campaign: CampaignRef, daily_budget_micros: int) -> Result:
        request = {"operations": [{"update": {
            "amountMicros": daily_budget_micros}}]}
        row = self.session.get(Campaign, campaign.internal_id)
        row.daily_budget_micros = daily_budget_micros
        self.session.commit()
        self._log("POST", f"{self.CUSTOMER}/campaignBudgets:mutate", request,
                  {"results": [{"resourceName": "..."}]})
        return Result(True, {"daily_budget_micros": daily_budget_micros})

    def pause_ad(self, ad: AdRef) -> Result:
        row = self.session.get(Ad, ad.internal_id)
        row.status = "paused"
        self.session.commit()
        self._log("POST", f"{self.CUSTOMER}/adGroupAds:mutate",
                  {"operations": [{"update": {
                      "resourceName": ad.external_id, "status": "PAUSED"}}]},
                  {"results": [{"resourceName": ad.external_id}]})
        return Result(True)

    def fetch_report(self, campaign: CampaignRef, date_from: date, date_to: date) -> dict:
        """GAQL searchStream 口径：metrics 嵌套、costMicros、conversions 是浮点。"""
        rows = self._report_rows(campaign, date_from, date_to)
        results = [{
            "campaign": {"resourceName": campaign.external_id},
            "segments": {"date": r.report_date_local},
            "metrics": {
                "impressions": str(r.impressions),
                "clicks": str(r.clicks),
                "costMicros": str(r.spend_micros),
                "conversions": float(r.conversions),
                "conversionsValue": r.revenue_micros / MICROS,
            },
        } for r in rows]
        response = {"results": results}
        self._log("POST", f"{self.CUSTOMER}/googleAds:searchStream",
                  {"query": "SELECT metrics.impressions, metrics.clicks, "
                            "metrics.cost_micros, metrics.conversions, "
                            "metrics.conversions_value, segments.date "
                            "FROM campaign WHERE segments.date BETWEEN "
                            f"'{date_from}' AND '{date_to}'"},
                  response)
        return response


# ---------------------------------------------------------------------------
# TikTok Mock（Business API v1.3，金额单位: 美元浮点，响应包 code/message/data）
# ---------------------------------------------------------------------------

class MockTikTokAdapter(AdPlatformAdapter):
    platform = Platform.TIKTOK_SIM
    ADVERTISER = "6900000000000000000"

    def _envelope(self, data: dict) -> dict:
        return {"code": 0, "message": "OK",
                "request_id": uuid.uuid4().hex[:16], "data": data}

    def create_campaign(self, brief: CampaignBrief) -> CampaignRef:
        request = {
            "advertiser_id": self.ADVERTISER,
            "campaign_name": brief.name,
            "objective_type": "CONVERSIONS" if brief.objective == "conversions"
                              else "TRAFFIC",
            "budget_mode": "BUDGET_MODE_DAY",
            "budget": brief.daily_budget_micros / MICROS,  # 美元浮点
        }
        ext = str(1700000000000000000 + uuid.uuid4().int % 10**15)
        row = Campaign(platform=self.platform, name=brief.name,
                       objective=brief.objective,
                       daily_budget_micros=brief.daily_budget_micros,
                       external_ref=ext)
        self.session.add(row); self.session.commit()
        response = self._envelope({"campaign_id": ext})
        self._log("POST", "/open_api/v1.3/campaign/create/", request, response)
        return CampaignRef(self.platform, ext, row.id)

    def create_ad_group(self, campaign: CampaignRef, t: Targeting) -> AdGroupRef:
        request = {
            "advertiser_id": self.ADVERTISER,
            "campaign_id": campaign.external_id,
            "adgroup_name": "adgroup-1",
            "placements": ["PLACEMENT_TIKTOK"],
            "location_ids": list(t.geo),
            "age_groups": ["AGE_18_24", "AGE_25_34", "AGE_35_44"],
            "interest_category_ids": list(t.interests),
            "billing_event": "CPM",
            "bid_type": "BID_TYPE_CUSTOM",
            "bid_price": t.bid_micros / MICROS,  # 美元浮点
            "optimization_goal": "CONVERT",
        }
        ext = str(1710000000000000000 + uuid.uuid4().int % 10**15)
        row = AdGroup(campaign_id=campaign.internal_id, name="adgroup-1",
                      targeting_json=json.dumps(
                          {k: request[k] for k in
                           ("location_ids", "age_groups",
                            "interest_category_ids")}),
                      bid_micros=t.bid_micros)
        self.session.add(row); self.session.commit()
        self._log("POST", "/open_api/v1.3/adgroup/create/", request,
                  self._envelope({"adgroup_id": ext}))
        return AdGroupRef(self.platform, ext, row.id, campaign.internal_id)

    def create_ad(self, group: AdGroupRef, c: Creative) -> AdRef:
        request = {
            "advertiser_id": self.ADVERTISER,
            "adgroup_id": group.external_id,
            "creatives": [{
                "ad_name": f"ad-{c.creative_id}",
                "ad_format": "SINGLE_IMAGE",
                "ad_text": c.body,
                "image_ids": [c.image_url],
                "landing_page_url": c.landing_url,
                "call_to_action": "SHOP_NOW",
            }],
        }
        ext = str(1720000000000000000 + uuid.uuid4().int % 10**15)
        row = Ad(ad_group_id=group.internal_id, creative_id=c.creative_id,
                 creative_quality=c.quality)
        self.session.add(row); self.session.commit()
        self._log("POST", "/open_api/v1.3/ad/create/", request,
                  self._envelope({"ad_ids": [ext]}))
        return AdRef(self.platform, ext, row.id)

    def update_budget(self, campaign: CampaignRef, daily_budget_micros: int) -> Result:
        request = {"advertiser_id": self.ADVERTISER,
                   "campaign_id": campaign.external_id,
                   "budget": daily_budget_micros / MICROS}
        row = self.session.get(Campaign, campaign.internal_id)
        row.daily_budget_micros = daily_budget_micros
        self.session.commit()
        self._log("POST", "/open_api/v1.3/campaign/update/", request,
                  self._envelope({}))
        return Result(True, {"daily_budget_micros": daily_budget_micros})

    def pause_ad(self, ad: AdRef) -> Result:
        request = {"advertiser_id": self.ADVERTISER,
                   "ad_ids": [ad.external_id],
                   "operation_status": "DISABLE"}
        row = self.session.get(Ad, ad.internal_id)
        row.status = "paused"
        self.session.commit()
        self._log("POST", "/open_api/v1.3/ad/status/update/", request,
                  self._envelope({}))
        return Result(True)

    def fetch_report(self, campaign: CampaignRef, date_from: date, date_to: date) -> dict:
        """TikTok 报表口径：data.list、维度带 'stat_time_day 00:00:00'、数值字符串。"""
        rows = self._report_rows(campaign, date_from, date_to)
        lst = [{
            "dimensions": {"stat_time_day": f"{r.report_date_local} 00:00:00"},
            "metrics": {
                "impressions": str(r.impressions),
                "clicks": str(r.clicks),
                "spend": f"{r.spend_micros / MICROS:.2f}",
                "conversions": str(r.conversions),
                "total_purchase_value": f"{r.revenue_micros / MICROS:.2f}",
            },
        } for r in rows]
        response = self._envelope({"list": lst,
                                   "page_info": {"total_number": len(lst)}})
        self._log("GET", "/open_api/v1.3/report/integrated/get/",
                  {"advertiser_id": self.ADVERTISER,
                   "report_type": "BASIC", "data_level": "AUCTION_CAMPAIGN",
                   "start_date": date_from.isoformat(),
                   "end_date": date_to.isoformat()},
                  response)
        return response


_ADAPTERS: dict[Platform, type[AdPlatformAdapter]] = {
    Platform.META_SIM: MockMetaAdapter,
    Platform.GOOGLE_SIM: MockGoogleAdapter,
    Platform.TIKTOK_SIM: MockTikTokAdapter,
}


def make_adapter(platform: Platform, session: Session) -> AdPlatformAdapter:
    return _ADAPTERS[platform](session)
