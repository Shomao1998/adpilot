"""CTR 预排序（设计文档 4.4, P1）。

VLM/LLM 对同一 brief 下的创意变体打相对分（相对分比绝对分可靠），维度:
视觉吸引力、信息清晰度、CTA 强度、平台风格匹配。分数两个用途:
    1. 写进 Ad.creative_quality -> 喂给 AdSim 点击模型（质量乘数）。
    2. 决定初始预算权重（Top3 拿 70% 探索预算）。

刻意与真实点击率脱钩: 模拟器点击模型会在质量分上叠加日噪声（simulate.py），
制造"预估与现实的差距" —— 评估诊断环节才有真东西可诊断。

LLM 依赖走 LLMClient 协议（复用 adcreative.llm），测试用 FakeLLM。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from adcreative.llm import LLMClient, complete_with_retry
from adcreative.schema import CopyVariant
from adcreative.visual_score import VisualScorer, default_visual_scorer


class CtrScore(BaseModel):
    variant_id: str
    visual_appeal: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)
    cta_strength: float = Field(ge=0, le=1)
    platform_fit: float = Field(ge=0, le=1)

    @property
    def overall(self) -> float:
        return (self.visual_appeal + self.clarity
                + self.cta_strength + self.platform_fit) / 4


class CtrScoreBatch(BaseModel):
    scores: list[CtrScore]


@dataclass
class RankedCreative:
    variant: CopyVariant
    score: float           # [0,1] 综合质量分（四维平均）
    budget_weight: float   # 初始预算权重，同 brief 内归一到 1
    breakdown: CtrScore | None = None   # 四维明细（翻转卡展示；含图片视觉分）
    visual_from_image: bool = False     # visual_appeal 是否来自真实图像分析


CTR_SYSTEM = """你是广告创意的投前质量评审（CTR 预排序）。对每个创意变体在四个
维度打 0-1 分：visual_appeal(视觉吸引力)、clarity(信息清晰度)、cta_strength(CTA强度)、
platform_fit(平台风格匹配)。相对排序比绝对分更重要——拉开变体间差距。
variant_id 必须与输入一致。"""


def _top_heavy_weights(scores: list[float], top_k: int = 3,
                       top_share: float = 0.70) -> list[float]:
    """Top-K 拿 top_share 预算（按分数比例分配），其余均分剩余。"""
    n = len(scores)
    if n == 0:
        return []
    if n <= top_k:
        s = sum(scores) or 1.0
        return [x / s for x in scores]
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_idx = set(order[:top_k])
    top_sum = sum(scores[i] for i in top_idx) or 1.0
    rest = n - top_k
    weights = [0.0] * n
    for i in range(n):
        if i in top_idx:
            weights[i] = top_share * scores[i] / top_sum
        else:
            weights[i] = (1 - top_share) / rest
    return weights


def _load_image(src):
    """images 值可为 PIL.Image 或路径；加载失败返回 None（视觉分回退到 LLM）。"""
    from PIL import Image
    if src is None:
        return None
    if isinstance(src, (str, Path)):
        try:
            img = Image.open(src)
            img.load()
            return img
        except Exception:
            return None
    return src   # 已是 PIL.Image


def rank_creatives(
    variants: list[CopyVariant], llm: LLMClient,
    top_k: int = 3, top_share: float = 0.70,
    images: dict | None = None,
    visual_scorer: VisualScorer | None = None,
) -> list[RankedCreative]:
    """对一组创意打分并计算初始预算权重（降序返回）。

    文案三维（clarity/cta_strength/platform_fit）由 LLM 打；visual_appeal 若给了
    对应创意图（images: {variant_id: PIL/路径}），改用 visual_scorer 分析真实图像
    得出（默认本地 CV 打分器），否则沿用 LLM 从文案的估计。四维取平均为综合分。
    """
    if not variants:
        return []
    listing = "\n".join(
        f"- variant_id={v.variant_id} platform={v.platform}: {v.all_text()}"
        for v in variants)
    batch = complete_with_retry(
        llm, CTR_SYSTEM, f"为以下 {len(variants)} 个创意打分:\n{listing}",
        CtrScoreBatch)
    detail_map = {s.variant_id: s for s in batch.scores}
    images = images or {}
    scorer = visual_scorer or default_visual_scorer()

    ranked_raw: list[tuple[CopyVariant, CtrScore, bool]] = []
    for v in variants:
        # 缺分的变体给中性四维分，保证鲁棒
        detail = detail_map.get(v.variant_id) or CtrScore(
            variant_id=v.variant_id, visual_appeal=0.5, clarity=0.5,
            cta_strength=0.5, platform_fit=0.5)
        from_image = False
        img = _load_image(images.get(v.variant_id))
        if img is not None:
            detail.visual_appeal = round(scorer.score(img), 4)   # 真图覆盖视觉分
            from_image = True
        ranked_raw.append((v, detail, from_image))

    scores = [d.overall for _, d, _ in ranked_raw]
    weights = _top_heavy_weights(scores, top_k, top_share)
    ranked = [RankedCreative(variant=v, score=d.overall, budget_weight=w,
                             breakdown=d, visual_from_image=fi)
              for (v, d, fi), w in zip(ranked_raw, weights)]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
