"""长任务的计算逻辑：Web 与 worker 共用。

Web 请求只往队列里塞任务并立刻返回；实际的 1–2 分钟计算（pilot 模拟 + LLM 诊断
复述）在这里，由 ThreadQueue（本地后台线程）或 RQ worker（云上独立容器）执行，
结果写进 Store，前端轮询取。这样 Web 层无状态、长任务不再阻塞 HTTP 请求。

诊断复述: 默认离线 TemplateNarrator；ADPILOT_LIVE=1 时走真实 LLM。
"""
from __future__ import annotations

import os
import re

from adpilot import AdPilotOrchestrator, PilotConfig
from adsim.models import Platform
from webapp.infra import STORE, persist_run
from webapp.payload import build_dashboard_payload

# 创意画廊平台名 -> 模拟平台枚举
_PLAT_MAP = {"meta": Platform.META_SIM, "google": Platform.GOOGLE_SIM,
             "tiktok": Platform.TIKTOK_SIM}


class TemplateNarrator:
    """离线诊断复述 stub（实现 LLMClient 协议），中英双语。"""

    def __init__(self, lang: str = "zh"):
        self.lang = (lang or "zh").lower()

    def complete(self, system: str, user: str, schema):
        best = re.search(r"领先 (\w+)", user)
        worst = re.search(r"垫底 (\w+)", user)
        roas = re.search(r"ROAS ([\d.]+)。", user)
        b = best.group(1) if best else "?"
        w = worst.group(1) if worst else "?"
        r = roas.group(1) if roas else "?"
        if self.lang == "en":
            return schema(
                narrative=(f"Blended ROAS {r}: {b} leads on efficiency while "
                           f"{w} lags; the gap is driven by cross-platform "
                           f"conversion efficiency and bid adjustments."),
                next_step=f"shift budget toward {b} and review {w}'s creative and bids")
        return schema(
            narrative=(f"综合 ROAS {r}，{b} 效率领先、{w} 垫底；"
                       f"差距主要由平台间转化效率与出价调整共同驱动。"),
            next_step=f"向 {b} 倾斜预算，并关注 {w} 的素材与出价")


def _narrator(lang: str = "zh"):
    if os.getenv("ADPILOT_LIVE") == "1":
        try:
            from adcreative import make_live_llm
            return make_live_llm()   # claude / deepseek 按 ADPILOT_PROVIDER；语言经提示词
        except Exception:
            pass
    return TemplateNarrator(lang)


def run_pilot(days: int = 7, seed: int = 42, lang: str = "zh",
              qualities: dict | None = None,
              daily_budget_usd: float = 100.0) -> dict:
    """跑一次投放模拟。

    qualities 给定时（来自向导生成的创意 CTR 预排序分），据此建各平台 Ad 的
    creative_quality —— 决策日志/看板/对比就变成「投放你这批创意」的结果，而非
    与输入无关的固定合成数据。缺省 {} 时回到默认质量，即示例模拟。
    """
    cfg = PilotConfig(total_days=days, seed=seed, lang=lang,
                      diagnosis_llm=_narrator(lang),
                      creative_qualities=qualities or {},
                      daily_budget_usd=daily_budget_usd)
    orch = AdPilotOrchestrator(cfg)
    final = orch.run()
    return build_dashboard_payload(final, orch.log, cfg.bid.target_cpa_usd)


def platform_qualities(gallery: dict) -> dict:
    """过审创意的 CTR 预排序分 -> {平台名: [quality,...]}（JSON 友好，供 Store 存）。"""
    q: dict = {}
    for it in gallery.get("items", []):
        if it.get("verdict") != "pass":
            continue
        name = it.get("platform")
        if name not in _PLAT_MAP:
            continue
        q.setdefault(name, []).append(max(0.05, min(1.0, float(it.get("ctr_score") or 0))))
    return q


def pilot_key(lang: str) -> str:
    """有向导创意则用 live 键（随 version 变，换创意即失效重算），否则示例键。"""
    if STORE.get("latest:qualities"):
        return f"pilot:live:{lang}:{STORE.get('latest:version') or 0}"
    return f"pilot:sample:{lang}"


def build_pilot(lang: str, days: int = 7, seed: int = 42) -> dict:
    """构造 pilot 结果并标注来源: live=投放你的创意 / sample=示例合成数据。"""
    q_json = STORE.get("latest:qualities")
    q = {_PLAT_MAP[k]: v for k, v in q_json.items()} if q_json else None
    payload = run_pilot(days=days, seed=seed, lang=lang, qualities=q,
                        daily_budget_usd=float(STORE.get("latest:daily_usd") or 100.0))
    payload["source"] = "live" if q else "sample"
    if q:
        payload["campaign"] = STORE.get("latest:campaign")
    return payload


def pilot_job(lang: str, key: str) -> None:
    """队列任务：算 pilot -> 写 Store -> 落库 -> 释放计算锁。

    本地由后台线程执行，云上由 RQ worker 容器执行——同一函数，只是调度不同。
    """
    try:
        payload = build_pilot(lang)
        STORE.set(key, payload)
        persist_run(key, payload)
    finally:
        STORE.release(key)
