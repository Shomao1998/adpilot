"""创意审核 Agent: L1 规则引擎 + L2 LLM 复审（设计文档 4.3）。

分层理由:
    L1 (确定性规则): 违禁词 -> reject（硬违规不给 LLM 讨价还价的机会）；
        限制类目 / 绝对化用语 -> revise。规则语料参照 Meta Advertising
        Standards 与 Google Ads Policy 公开文档的类目。格式校验不在这里 ——
        已由 schema 校验器承担（不合规文案根本构造不出来）。
    L2 (LLM 复审): 夸大宣传、图文相关性、品牌一致性等语义判断。
        只有 L1 通过才进 L2，省 token 也省时延。
"""
from __future__ import annotations

import re

from adcreative.llm import LLMClient, complete_with_retry, lang_note
from adcreative.schema import CopyVariant, ReviewVerdict

# L1-a 违禁类目 -> reject（Meta/Google 政策: 武器、成人、烟草）
BANNED_TERMS = [
    # 武器
    "枪支", "弹药", "武器", "gun", "guns", "firearm", "ammunition", "weapon",
    # 成人
    "色情", "成人内容", "porn", "adult content", "xxx",
    # 烟草
    "香烟", "烟草", "电子烟", "cigarette", "tobacco", "vape", "e-cigarette",
]

# L1-b 限制类目 -> revise（需资质/附加声明，demo 简化为要求改写规避）
RESTRICTED_TERMS = [
    "保健品", "减肥药", "处方药", "贷款", "加密货币",
    "supplement", "weight loss pill", "prescription", "crypto", "loan",
]

# L1-c 绝对化用语 -> revise（夸大宣传的词面层拦截，语义层交给 L2）
_SUPERLATIVE_RE = re.compile(
    r"最好|第一|治愈|根治|百分之?百|保证|"
    r"\b(?:best|#1|no\.?\s?1|cure[sd]?|guaranteed|perfect)\b",
    re.IGNORECASE,
)


def run_l1_rules(variant: CopyVariant, lang: str = "zh") -> ReviewVerdict:
    """确定性规则审核。reject > revise > pass。reasons/suggestions 双语。"""
    en = (lang or "zh").lower() == "en"
    text = variant.all_text()
    text_lower = text.lower()

    banned = [t for t in BANNED_TERMS if t in text_lower]
    if banned:
        joined = ", ".join(banned)
        return ReviewVerdict(
            verdict="reject",
            reasons=[(f"Prohibited category term: {joined}" if en
                      else f"违禁类目词: {joined}")],
            suggestions=("This product category cannot be advertised; do not retry."
                         if en else "该产品类目不可投放，不要改写重试。"))

    reasons: list[str] = []
    restricted = [t for t in RESTRICTED_TERMS if t in text_lower]
    if restricted:
        joined = ", ".join(restricted)
        reasons.append(f"Restricted category term (needs qualification): {joined}"
                       if en else f"限制类目词（需资质）: {joined}")
    superlatives = _SUPERLATIVE_RE.findall(text)
    if superlatives:
        joined = ", ".join(set(superlatives))
        reasons.append(f"Absolute/superlative wording: {joined}" if en
                       else f"绝对化用语: {joined}")

    if reasons:
        return ReviewVerdict(
            verdict="revise", reasons=reasons,
            suggestions=("Remove restricted-category claims and superlatives; "
                         "use concrete, verifiable selling points." if en else
                         "移除限制类目表述与绝对化用语，改用可验证的具体卖点。"))
    return ReviewVerdict(verdict="pass")


REVIEW_SYSTEM = """你是广告创意复审员（平台预审视角）。对给定文案输出审核结论:
- pass: 合规
- revise: 有可修复问题（夸大但可软化、承诺无依据、与产品不符），给出具体修改建议
- reject: 不可修复的硬违规
注意: 违禁词/绝对化用语已由前置规则拦截，你负责语义层——隐含的夸大承诺、
误导性表述、与产品/受众明显不符的内容。宁可 revise 不要轻易 reject。"""


def review_variant(variant: CopyVariant, llm: LLMClient,
                   lang: str = "zh") -> ReviewVerdict:
    """两级审核: L1 未过直接返回；L1 通过才进 L2 LLM 复审。"""
    l1 = run_l1_rules(variant, lang)
    if l1.verdict != "pass":
        return l1
    prompt = (f"平台: {variant.platform}\n文案全文:\n{variant.all_text()}")
    return complete_with_retry(llm, lang_note(lang) + REVIEW_SYSTEM,
                               prompt, ReviewVerdict)
