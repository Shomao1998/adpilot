"""创意管线的数据契约。

两个设计点:
    1. ADR-5 在类型层生效: Budget.currency 是 Literal["USD"]，任何其他币种
       直接 ValidationError —— 不靠约定靠类型。
    2. 平台格式约束（Google RSA 30/90 字符、资产数量等）写成 Pydantic 校验器。
       LLM 结构化输出经过这里，格式不合规 = 校验失败 = 触发重试 —— 审核 Agent
       的 L1 格式校验由 schema 承担，规则引擎只管内容合规。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PlatformName = Literal["meta", "google", "tiktok"]

# 平台文案硬限制（取自真实 API 文档的资产约束）
META_HEADLINE_MAX = 40        # Meta: headline 建议 ≤40 字符
META_PRIMARY_TEXT_MAX = 300   # primary text 超过会被折叠，硬限制放宽到 300
TIKTOK_TEXT_MAX = 100         # TikTok: ad text 12-100 字符
RSA_HEADLINE_MAX = 30         # Google RSA: 标题 ≤30 字符
RSA_DESCRIPTION_MAX = 90      # Google RSA: 描述 ≤90 字符
RSA_HEADLINES_RANGE = (3, 15)     # RSA 要求 3-15 个标题
RSA_DESCRIPTIONS_RANGE = (2, 4)   # RSA 要求 2-4 条描述


class Product(BaseModel):
    name: str
    category: str = ""
    selling_points: list[str] = Field(default_factory=list)
    price_usd: float | None = None


class Audience(BaseModel):
    geo: list[str] = Field(default_factory=lambda: ["US"])
    age_min: int = 18
    age_max: int = 65
    interests: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    total_usd: float = 0.0
    daily_usd: float = 0.0
    currency: Literal["USD"] = "USD"  # ADR-5: 只有美元


class Brief(BaseModel):
    """结构化投放 brief（设计文档 4.1）。意图解析 Agent 的输出。"""

    product: Product
    audience: Audience = Field(default_factory=Audience)
    budget: Budget = Field(default_factory=Budget)
    platforms: list[PlatformName] = Field(
        default_factory=lambda: ["meta", "google", "tiktok"])
    objective: str = "conversions"
    brand_tone: str = "neutral"
    landing_url: str = "https://example.com"


class CopyVariant(BaseModel):
    """一组平台文案。google 用 rsa_* 资产字段，meta/tiktok 用 headline/body。"""

    platform: PlatformName
    variant_id: str = ""
    headline: str = ""
    body: str = ""
    cta: str = "Shop Now"
    rsa_headlines: list[str] = Field(default_factory=list)
    rsa_descriptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _platform_format_rules(self) -> "CopyVariant":
        if self.platform == "google":
            lo, hi = RSA_HEADLINES_RANGE
            if not lo <= len(self.rsa_headlines) <= hi:
                raise ValueError(f"RSA 标题数须在 {lo}-{hi} 之间")
            lo, hi = RSA_DESCRIPTIONS_RANGE
            if not lo <= len(self.rsa_descriptions) <= hi:
                raise ValueError(f"RSA 描述数须在 {lo}-{hi} 之间")
            for h in self.rsa_headlines:
                if len(h) > RSA_HEADLINE_MAX:
                    raise ValueError(f"RSA 标题超长(>{RSA_HEADLINE_MAX}): {h!r}")
            for d in self.rsa_descriptions:
                if len(d) > RSA_DESCRIPTION_MAX:
                    raise ValueError(f"RSA 描述超长(>{RSA_DESCRIPTION_MAX}): {d!r}")
        else:
            if not self.headline:
                raise ValueError(f"{self.platform} 文案缺少 headline")
            if self.platform == "meta" and len(self.headline) > META_HEADLINE_MAX:
                raise ValueError(f"Meta 标题超长(>{META_HEADLINE_MAX})")
            if self.platform == "meta" and len(self.body) > META_PRIMARY_TEXT_MAX:
                raise ValueError(f"Meta 正文超长(>{META_PRIMARY_TEXT_MAX})")
            if self.platform == "tiktok" and len(self.headline) > TIKTOK_TEXT_MAX:
                raise ValueError(f"TikTok 文案超长(>{TIKTOK_TEXT_MAX})")
        return self

    def all_text(self) -> str:
        """审核用: 拼接全部可见文本。"""
        parts = [self.headline, self.body, self.cta,
                 *self.rsa_headlines, *self.rsa_descriptions]
        return " ".join(p for p in parts if p)


class CopyBatch(BaseModel):
    """LLM 单次文案生成的输出容器。"""
    variants: list[CopyVariant]


class ReviewVerdict(BaseModel):
    """审核结论: pass / reject(硬违规) / revise(可修改后重试)。"""
    verdict: Literal["pass", "reject", "revise"]
    reasons: list[str] = Field(default_factory=list)
    suggestions: str = ""
