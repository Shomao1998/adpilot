"""adplatform: Campaign 构建层（适配器模式，ADR-2/ADR-3）。

Agent 侧只依赖本包的 AdPlatformAdapter 接口，不知道背后是 Mock 还是真实 API。
"转真"只需新增一个 adapter 实现（如 Meta 开发者沙盒），Agent 代码零改动。
"""
from adplatform.adapters import (
    AdPlatformAdapter,
    AdGroupRef,
    AdRef,
    CampaignBrief,
    CampaignRef,
    Creative,
    MockGoogleAdapter,
    MockMetaAdapter,
    MockTikTokAdapter,
    Result,
    Targeting,
    make_adapter,
)

__all__ = [
    "AdPlatformAdapter", "AdGroupRef", "AdRef", "CampaignBrief",
    "CampaignRef", "Creative", "MockGoogleAdapter", "MockMetaAdapter",
    "MockTikTokAdapter", "Result", "Targeting", "make_adapter",
]
