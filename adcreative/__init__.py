"""adcreative: 创意管线（W3）—— 意图解析 -> 分平台文案 -> 审核 -> 多尺寸排版。

设计要点:
    - LLM 依赖通过 LLMClient 协议注入: ClaudeLLM(真实) / FakeLLM(测试与离线演示)。
      管线代码不感知背后是谁 —— 与 ADR-3（模拟器与 Agent 解耦）同一哲学。
    - 图片管线实现的是设计文档 W3 的降级预案（渐变背景+排版+CTA，纯 Pillow 本地渲染），
      fal.ai 生图 / RMBG 抠图留接口（ImageSource 协议）不实现。
    - 全部金额 USD (ADR-5)，Brief.currency 用 Literal["USD"] 在类型层禁止其他币种。
"""
from adcreative.schema import (
    Audience,
    Brief,
    Budget,
    CopyVariant,
    Product,
    ReviewVerdict,
)
from adcreative.llm import (
    ClaudeLLM,
    DeepSeekLLM,
    FakeLLM,
    LLMClient,
    make_live_llm,
)
from adcreative.intent import parse_brief
from adcreative.copywriter import generate_copy
from adcreative.review import review_variant, run_l1_rules
from adcreative.pipeline import generate_creatives
from adcreative.layout import CREATIVE_SIZES, render_creative
from adcreative.imagery import (
    FalImageSource,
    ImageSource,
    OpenAICompatImageSource,
    make_image_source,
    resolve_product_image,
    scrape_product_image,
)

__all__ = [
    "Audience", "Brief", "Budget", "CopyVariant", "Product", "ReviewVerdict",
    "ClaudeLLM", "DeepSeekLLM", "FakeLLM", "LLMClient", "make_live_llm",
    "parse_brief", "generate_copy", "review_variant", "run_l1_rules",
    "generate_creatives", "CREATIVE_SIZES", "render_creative",
    "FalImageSource", "ImageSource", "OpenAICompatImageSource",
    "make_image_source", "resolve_product_image", "scrape_product_image",
]
