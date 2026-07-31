"""创意管线编排: 生成 -> 审核 -> (reject/revise 回流重试 ≤2 轮) -> 排版。

这是设计文档流程图里的"小闭环"（审核→重新生成）。W4 的 LangGraph 编排
会把它接进全流程 DAG；本模块保持纯函数式，方便被状态机节点直接调用。

W3 验收口径: 输入 -> 产出 ≥6 组过审创意（3 平台 × 每平台 ≥2 组过审）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adcreative.copywriter import generate_copy
from adcreative.layout import render_all_sizes
from adcreative.llm import LLMClient
from adcreative.review import review_variant
from adcreative.schema import Brief, CopyVariant, ReviewVerdict

MAX_REVISE_ROUNDS = 2  # 审核回流重试上限（设计文档 4.3）


@dataclass
class ApprovedCreative:
    variant: CopyVariant
    verdict: ReviewVerdict
    image_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class PipelineResult:
    approved: list[ApprovedCreative]
    rejected: list[tuple[CopyVariant, ReviewVerdict]]
    rounds_used: dict[str, int]  # platform -> 实际生成轮数

    @property
    def ok(self) -> bool:
        """W3 验收: 全平台合计 ≥6 组过审。"""
        return len(self.approved) >= 6


def generate_creatives(
    brief: Brief,
    llm: LLMClient,
    n_variants: int = 3,
    render_dir: str | Path | None = None,
    product_image=None,
    lang: str = "zh",
) -> PipelineResult:
    """跑通单个 brief 的创意小闭环。

    每个平台: 生成 n 组 -> 逐组审核 -> 未过审的携带审核意见回流重新生成，
    最多 MAX_REVISE_ROUNDS 轮；reject（硬违规）不回流。
    render_dir 给定时为每组过审文案渲染三尺寸图片；product_image（PIL 图，可选）
    会合成进图中（抓取的主图或文生图），缺省则纯渐变兜底。
    """
    approved: list[ApprovedCreative] = []
    rejected: list[tuple[CopyVariant, ReviewVerdict]] = []
    rounds_used: dict[str, int] = {}

    for platform in brief.platforms:
        need = n_variants
        feedback = ""
        rounds = 0
        while need > 0 and rounds <= MAX_REVISE_ROUNDS:
            round_tag = rounds
            rounds += 1
            try:
                variants = generate_copy(brief, llm, platform,
                                         n_variants=need, feedback=feedback,
                                         round_tag=round_tag, lang=lang)
            except Exception:
                # 该平台本轮生成失败（LLM 输出屡次不合 schema，DeepSeek 常见）：
                # 保留 need/feedback，下一轮再试；耗尽轮次则该平台产出较少，
                # 但绝不拖垮其他平台的创意生成。
                continue
            fail_reasons: list[str] = []
            still_need = 0
            for v in variants:
                try:
                    verdict = review_variant(v, llm, lang=lang)
                except Exception:
                    # 审核调用失败：保守当作需重生成，不崩
                    rejected.append((v, ReviewVerdict(
                        verdict="revise", reasons=["审核调用失败"], suggestions="")))
                    still_need += 1
                    continue
                if verdict.verdict == "pass":
                    approved.append(ApprovedCreative(variant=v, verdict=verdict))
                elif verdict.verdict == "revise":
                    rejected.append((v, verdict))
                    fail_reasons.append(
                        f"[{v.variant_id}] {'; '.join(verdict.reasons)} "
                        f"建议: {verdict.suggestions}")
                    still_need += 1
                else:  # reject: 硬违规不回流
                    rejected.append((v, verdict))
            need = still_need
            feedback = "\n".join(fail_reasons)
        rounds_used[platform] = rounds

    if render_dir is not None:
        for item in approved:
            v = item.variant
            headline = v.headline or (v.rsa_headlines[0] if v.rsa_headlines else "")
            body = v.body or (v.rsa_descriptions[0] if v.rsa_descriptions else "")
            item.image_paths = render_all_sizes(
                headline, body, v.cta, out_dir=render_dir, stem=v.variant_id,
                product_image=product_image)

    return PipelineResult(approved=approved, rejected=rejected,
                          rounds_used=rounds_used)
