"""AdSim 数据模型。

设计原则（事件溯源，Event Sourcing）:
    1. Ground Truth 事件流是唯一事实来源: auction -> (win) impression -> click -> conversion。
       事件表只追加、不更新，保证模拟可复现、可审计。
    2. "平台报表"是派生视图: 由报表生成器按各平台的 AttributionSpec（归因窗口/时区/口径）
       从事件流物化而来。刻意让三平台口径不同 —— 这是跨平台归一化模块存在的理由。
    3. Ground truth 与平台报表并存，使归一化层的正确性可以对照真值验证 —— 真实业务
       做不到这一点，是模拟器独有的展示优势。

工程注意:
    - 主键用 BigInteger 以支撑千万级事件；SQLite 的 AUTOINCREMENT 只认 INTEGER，
      故使用 with_variant 做方言兼容（生产 Postgres 用 BIGINT，本地 SQLite 退化为 INTEGER）。
    - 全项目统一美元(USD)计价，不涉及汇率或币种转换 (ADR-5)；金额一律以
      micros (1e-6 美元) 的整数存储，避免浮点累加误差 —— 与 Google Ads API
      的 cost_micros 口径一致。
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# SQLite 方言兼容的 BigInteger 主键类型
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------

class Platform(str, enum.Enum):
    META_SIM = "meta_sim"
    GOOGLE_SIM = "google_sim"
    TIKTOK_SIM = "tiktok_sim"


class AuctionMechanism(str, enum.Enum):
    SECOND_PRICE = "second_price"   # Meta / TikTok 风格
    GSP = "gsp"                     # Google 风格（质量分 + 广义二价）


# ---------------------------------------------------------------------------
# 实体表（campaign 层级，schema 对齐真实平台三层结构）
# ---------------------------------------------------------------------------

class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    platform: Mapped[Platform] = mapped_column(Enum(Platform), index=True)
    name: Mapped[str] = mapped_column(String(255))
    objective: Mapped[str] = mapped_column(String(64), default="conversions")
    daily_budget_micros: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), default="active")
    # 平台侧引用（Mock API 返回的 id，模拟真实对接时的外部引用）
    external_ref: Mapped[str] = mapped_column(String(64), index=True)


class AdGroup(Base):
    __tablename__ = "ad_groups"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    # 定向摘要（demo 简化为 JSON 字符串；生产应拆维度表）
    targeting_json: Mapped[str] = mapped_column(String(2048), default="{}")
    bid_micros: Mapped[int] = mapped_column(BigInteger)  # 当前出价（CPM 口径）


class Ad(Base):
    __tablename__ = "ads"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    ad_group_id: Mapped[int] = mapped_column(ForeignKey("ad_groups.id"), index=True)
    creative_id: Mapped[str] = mapped_column(String(64), index=True)
    # CTR 预排序给出的创意质量分 [0,1]，模拟器用其作为点击率乘数的输入
    creative_quality: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(16), default="active")


# ---------------------------------------------------------------------------
# Ground Truth 事件流（append-only）
# ---------------------------------------------------------------------------

class AuctionEvent(Base):
    """每次竞价机会一行。won=True 即产生一次曝光（impression 内联于此表，
    避免 1:1 拆表带来的海量 join）。"""

    __tablename__ = "auction_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    platform: Mapped[Platform] = mapped_column(Enum(Platform))
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id"))
    sim_ts: Mapped[datetime] = mapped_column(DateTime, index=True)  # 模拟时钟 UTC
    hour_of_day: Mapped[int] = mapped_column(Integer)               # 冗余列，便于分时分析

    user_segment: Mapped[str] = mapped_column(String(32), default="generic")

    bid_micros: Mapped[int] = mapped_column(BigInteger)      # 我方出价 (CPM micros)
    floor_micros: Mapped[int] = mapped_column(BigInteger)    # 底价
    market_price_micros: Mapped[int] = mapped_column(BigInteger)  # 最高竞争对手价
    won: Mapped[bool] = mapped_column(Boolean)
    paying_micros: Mapped[int] = mapped_column(BigInteger, default=0)  # 实付（二价，CPM 口径；每次曝光扣费 = 本值/1000）

    __table_args__ = (
        Index("ix_auction_platform_ts", "platform", "sim_ts"),
        Index("ix_auction_ad_ts", "ad_id", "sim_ts"),
    )


class ClickEvent(Base):
    __tablename__ = "click_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    auction_event_id: Mapped[int] = mapped_column(
        ForeignKey("auction_events.id"), index=True
    )
    platform: Mapped[Platform] = mapped_column(Enum(Platform))
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id"), index=True)
    sim_ts: Mapped[datetime] = mapped_column(DateTime, index=True)


class ConversionEvent(Base):
    """转化挂在点击上（点击归因）。delay_hours 制造跨归因窗口的可观测差异:
    延迟 20 天的转化会被 Google-sim(30d) 记入、被 Meta-sim(7d) 丢弃。"""

    __tablename__ = "conversion_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    click_event_id: Mapped[int] = mapped_column(
        ForeignKey("click_events.id"), index=True
    )
    platform: Mapped[Platform] = mapped_column(Enum(Platform))
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id"), index=True)
    sim_ts: Mapped[datetime] = mapped_column(DateTime, index=True)  # 转化实际发生时刻
    click_ts: Mapped[datetime] = mapped_column(DateTime)            # 冗余：归因窗口判断
    delay_hours: Mapped[float] = mapped_column(Float)
    value_micros: Mapped[int] = mapped_column(BigInteger, default=0)  # 订单金额


# ---------------------------------------------------------------------------
# 平台归因配置 + 派生报表（"平台自己报的数"）
# ---------------------------------------------------------------------------

class AttributionSpec(Base):
    """每个平台一行。报表生成器据此从事件流物化 PlatformDailyReport。
    刻意的差异示例:
        meta_sim:   click_window=7d,  报表时区 America/Los_Angeles
        google_sim: click_window=30d, 报表时区 随账户设置 (demo 用 America/New_York)
        tiktok_sim: click_window=7d,  报表时区 UTC
    时区不同 -> "同一天"的边界不同 -> 三平台日报表对不上，归一化层负责拉齐。"""

    __tablename__ = "attribution_specs"

    platform: Mapped[Platform] = mapped_column(Enum(Platform), primary_key=True)
    click_window_days: Mapped[int] = mapped_column(Integer)
    view_window_days: Mapped[int] = mapped_column(Integer, default=0)
    report_timezone: Mapped[str] = mapped_column(String(64))
    # 转化计入哪一天: "click_date"(Meta口径, 归到点击日) 或 "conv_date"(归到转化日)
    conversion_date_basis: Mapped[str] = mapped_column(String(16), default="click_date")


class ApiCallLog(Base):
    """适配器层每次"平台 API 调用"的请求/响应全量留痕（设计文档 4.5）。
    UI 的 API 调用记录查看器直接读本表 —— 证明 schema 对齐真实平台文档。"""

    __tablename__ = "api_call_logs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    platform: Mapped[Platform] = mapped_column(Enum(Platform), index=True)
    method: Mapped[str] = mapped_column(String(8))          # POST/GET
    endpoint: Mapped[str] = mapped_column(String(255))
    request_json: Mapped[str] = mapped_column(Text)
    response_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )


class PlatformDailyReport(Base):
    """派生表：模拟"从平台 API 拉回来的日报表"。与 ground truth 事件流并存，
    归一化层的输出可与真值对照。注意: 本表允许重算覆盖（模拟平台报表回填）。"""

    __tablename__ = "platform_daily_reports"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    platform: Mapped[Platform] = mapped_column(Enum(Platform))
    report_date_local: Mapped[str] = mapped_column(String(10))  # 平台本地时区的日期
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id"))
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id"))

    impressions: Mapped[int] = mapped_column(BigInteger, default=0)
    clicks: Mapped[int] = mapped_column(BigInteger, default=0)
    spend_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    conversions: Mapped[int] = mapped_column(BigInteger, default=0)
    revenue_micros: Mapped[int] = mapped_column(BigInteger, default=0)

    __table_args__ = (
        Index(
            "ix_report_platform_date_ad",
            "platform", "report_date_local", "ad_id",
            unique=True,
        ),
    )
