"""创意生成 Agent（文案侧）: 按平台调性分化生成（设计文档 4.2）。

平台差异不是措辞装饰，而是硬结构差异:
    meta   -> 信息流长文案 + 短标题，社交证明话术
    tiktok -> 3 秒钩子句式、口语化短文案
    google -> RSA 严格资产结构（3-15 标题 ≤30 字符、2-4 描述 ≤90 字符），
              由 schema 校验器强制，LLM 输出不合规会触发重试。
"""
from __future__ import annotations

from adcreative.llm import LLMClient, complete_with_retry, has_cjk, lang_note
from adcreative.schema import Brief, CopyBatch, CopyVariant, PlatformName

_PLATFORM_GUIDES: dict[str, str] = {
    "meta": """平台: Meta (Facebook/Instagram 信息流)
- headline: 短标题, ≤40 字符
- body: 信息流长文案, 可用社交证明话术（如 "10,000+ 用户的选择"），≤300 字符
- 填 headline/body/cta 字段, rsa_* 留空""",
    "tiktok": """平台: TikTok
- headline: 3 秒钩子句式开头、口语化, ≤100 字符（TikTok 的 ad text）
- body 可留空或一句补充
- 填 headline/body/cta 字段, rsa_* 留空""",
    "google": """平台: Google 搜索 (Responsive Search Ad)
- rsa_headlines: 3-15 个标题, 每个 ≤30 字符（严格！超长会被拒）
- rsa_descriptions: 2-4 条描述, 每条 ≤90 字符（严格！）
- headline/body 留空, 填 rsa_* 与 cta 字段""",
}

COPY_SYSTEM = """你是分平台的广告合规文案生成器。根据投放 brief 为指定平台生成文案变体。
文案会经过严格的合规审核，违规一律被拒。请从一开始就写合规文案。

【绝对禁止 —— 命中即被拒】
1. 绝对化 / 最高级用语：best、#1、no.1、perfect、guaranteed、最好、第一、
   唯一、治愈、根治、保证、100%、百分百。
2. 编造的数字或社会证明：不得写具体用户数、评分、销量或人群背书，
   例如 "10,000+ users"、"thousands of customers"、"loved by health
   enthusiasts"、"trusted by millions"、"畅销 XX 万"。除非 brief 明确提供了
   这类数据，否则一律不许出现。
3. 不可验证的性能声明：不得编造具体时长或效果数字，例如 "blend in 10
   seconds"、"3 秒榨好"、"一周见效"。除非该数字是 brief 里给出的卖点。
4. 夸大或隐含承诺：不要暗示适用一切场景/极端环境（如 "anywhere"、
   "beat the heat"），不要臆造 brief 未提供的产品功能。

【应当这样写】
- 只基于 brief 提供的卖点，用具体、可验证的产品特征（轻便、可充电、易清洗等）。
- 情绪和句式可以有感染力、有钩子，但每一句事实都要站得住、不夸大。
- 想体现受欢迎时，用中性表述（如 "designed for..."、"made for..."），
  不要编造用户数或背书。

【格式】
- 每个变体角度不同（卖点侧重/情绪/句式），不要同义改写。
- platform 字段必须与要求的平台一致，variant_id 留空。
- cta 用一句本地化的号召性用语（与输出语言一致）。"""


def _brief_summary(brief: Brief) -> str:
    p = brief.product
    return (f"产品: {p.name}（{p.category}）\n"
            f"卖点: {', '.join(p.selling_points) or '未提供'}\n"
            f"价格: {f'${p.price_usd}' if p.price_usd else '未提供'}\n"
            f"受众: {', '.join(brief.audience.geo)} "
            f"{brief.audience.age_min}-{brief.audience.age_max}岁 "
            f"兴趣: {', '.join(brief.audience.interests) or '泛人群'}\n"
            f"品牌调性: {brief.brand_tone}\n"
            f"落地页: {brief.landing_url}")


def generate_copy(
    brief: Brief,
    llm: LLMClient,
    platform: PlatformName,
    n_variants: int = 3,
    feedback: str = "",
    round_tag: int = 0,
    lang: str = "zh",
) -> list[CopyVariant]:
    """为单个平台生成 n 组文案变体。feedback 是审核回流的修改意见（≤2 轮重试）。"""
    prompt = (f"{_brief_summary(brief)}\n\n{_PLATFORM_GUIDES[platform]}\n\n"
              f"生成 {n_variants} 个变体。")
    if feedback:
        prompt += f"\n\n上一轮审核未通过，修改意见（务必规避）:\n{feedback}"

    batch = complete_with_retry(llm, lang_note(lang) + COPY_SYSTEM, prompt, CopyBatch)
    variants = [v for v in batch.variants if v.platform == platform]
    en = (lang or "zh").lower() == "en"
    for i, v in enumerate(variants):
        v.variant_id = f"{platform}-r{round_tag}-{i}"
        # CTA 语言兜底：模型漏填或沿用英文默认时，按 UI 语言给本地化号召语，
        # 避免中文创意里蹦出英文 "Shop Now"（图上 CTA 也随之统一）。
        c = (v.cta or "").strip()
        if en and (not c or has_cjk(c)):
            v.cta = "Shop Now"
        elif not en and (not c or not has_cjk(c)):
            v.cta = "立即选购"
    return variants
