"""看板后端（FastAPI）—— 五页面 SPA 的数据源。

云无关：`ADPILOT_BACKEND=local|cloud` 切换状态/队列/对象存储后端（见 webapp.infra）。
local（默认）= 进程内 dict + 后台线程 + static 目录，行为与单机版一致；
cloud = Redis（状态+队列）+ RQ worker 容器 + S3/MinIO（创意图）+ Postgres（结果持久化）。

端点:
    GET  /api/results               默认 pilot（后台/worker 算，未就绪返回 202 {pending}）
    GET  /api/run                   强制重跑 pilot（用户点“重新跑”，同步返回）
    GET  /api/brief|creatives|campaigns   向导/画廊/Campaign 数据（示例或联动本次输入）
    POST /api/wizard                URL/关键词 -> 意图 -> 生成+审核 -> CTR 打分 -> 画廊
    GET  /                          五页面 SPA

长任务（pilot 模拟）走队列不阻塞请求；诊断复述默认离线 stub，ADPILOT_LIVE=1 走真实 LLM。
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp import jobs, scenario
from webapp.infra import BLOB, QUEUE, STORE
# 兼容旧引用（测试从 server 导入）
from webapp.jobs import TemplateNarrator, run_pilot  # noqa: F401

STATIC_DIR = Path(__file__).parent / "static"
GEN_DIR = STATIC_DIR / "gen"
WIZARD_GEN = STATIC_DIR / "wizard_gen"

app = FastAPI(title="AdPilot Dashboard")


def _enqueue_pilot(lang: str) -> None:
    """若结果未就绪且无人在算，抢锁并派发 pilot 计算任务（本地线程 / 云 worker）。"""
    key = jobs.pilot_key(lang)
    if STORE.get(key) is None and STORE.acquire(key):
        QUEUE.enqueue(jobs.pilot_job, lang, key)


@app.on_event("startup")
def _startup():
    _enqueue_pilot("zh")     # 启动即预热示例 pilot，页面秒开


@app.get("/api/results")
def api_results(lang: str = Query("zh")):
    key = jobs.pilot_key(lang)
    cached = STORE.get(key)
    if cached is not None:
        return JSONResponse(cached)
    _enqueue_pilot(lang)     # 未就绪：派发计算，先返回 pending 让前端轮询
    return JSONResponse({"pending": True}, status_code=202)


@app.get("/api/run")
def api_run(days: int = Query(7, ge=1, le=30), seed: int = Query(42),
            lang: str = Query("zh")):
    data = jobs.build_pilot(lang, days, seed)
    if days == 7 and seed == 42:
        STORE.set(jobs.pilot_key(lang), data)
    return JSONResponse(data)


@app.get("/api/brief")
def api_brief():
    # 恒返回固定示例（输入向导页的“示例 Brief”用；实时 brief 已在向导页内联显示）
    b = STORE.get("scenario:brief")
    if b is None:
        b = scenario.build_brief_payload()
        STORE.set("scenario:brief", b)
    return JSONResponse(b)


@app.get("/api/creatives")
def api_creatives():
    live = STORE.get("latest:gallery")       # 创意画廊反映最近一次向导生成
    if live is not None:
        return JSONResponse({**live, "source": "live",
                             "campaign": STORE.get("latest:campaign")})
    s = STORE.get("scenario:creatives")
    if s is None:
        s = scenario.build_creatives_payload(GEN_DIR)
        STORE.set("scenario:creatives", s)
    return JSONResponse({**s, "source": "sample"})


@app.get("/api/campaigns")
def api_campaigns():
    live = STORE.get("latest:campaigns")     # 有向导创意：本次“上线”的 campaign
    if live is not None:
        return JSONResponse({**live, "source": "live",
                             "campaign": STORE.get("latest:campaign")})
    s = STORE.get("scenario:campaigns")
    if s is None:
        s = scenario.build_campaigns_payload()
        STORE.set("scenario:campaigns", s)
    return JSONResponse({**s, "source": "sample"})


def _live_llm():
    if os.getenv("ADPILOT_LIVE") != "1":
        return None
    try:
        from adcreative import make_live_llm
        return make_live_llm()   # claude / deepseek 按 ADPILOT_PROVIDER
    except Exception:
        return None


@app.post("/api/wizard")
def api_wizard(payload: dict = Body(...)):
    """URL/关键词 -> 意图解析 -> 生成+审核 -> CTR 打分 -> 画廊；结果写 Store 联动全平台。

    需真实 LLM（ADPILOT_LIVE=1 + 对应 provider key）；离线返回 503 提示。
    创意图经对象存储（本地 static / 云 MinIO）；pilot 重算派进队列。
    """
    from adcreative import (
        generate_creatives, make_image_source, parse_brief, resolve_product_image,
    )
    from adpilot import rank_creatives

    text = (payload.get("input") or "").strip()
    lang = (payload.get("lang") or "zh").lower()
    if not text:
        return JSONResponse({"error": "请输入商品 URL 或关键词"}, status_code=400)
    llm = _live_llm()
    if llm is None:
        return JSONResponse(
            {"error": "实时生成需真实 LLM：设置 ADPILOT_LIVE=1 + 对应 key"
                      "（ANTHROPIC_API_KEY 或 ADPILOT_PROVIDER=deepseek + "
                      "DEEPSEEK_API_KEY）后重启。下方为示例数据。"},
            status_code=503)
    try:
        brief, questions = parse_brief(text, llm, lang=lang)
        # 配图三级兜底：URL 抓主图 -> fal.ai 文生图(配了 FAL_KEY) -> 渐变
        product_img = resolve_product_image(brief, text, make_image_source())
        result = generate_creatives(brief, llm, n_variants=2,
                                    render_dir=WIZARD_GEN,
                                    product_image=product_img, lang=lang)
        # 创意图喂给打分器（visual_appeal 由真实图像分析得出），并上传对象存储
        images = {}
        for a in result.approved:
            fp = a.image_paths.get("feed_1x1")
            if fp:
                BLOB.put(fp.name, fp.read_bytes())     # 本地=static / 云=MinIO
                images[a.variant.variant_id] = fp
        ranked = rank_creatives([a.variant for a in result.approved], llm,
                                images=images)
        score_map = {r.variant.variant_id: r.score for r in ranked}
        detail_map = {r.variant.variant_id: dict(
            overall=r.score, visual_appeal=r.breakdown.visual_appeal,
            clarity=r.breakdown.clarity, cta_strength=r.breakdown.cta_strength,
            platform_fit=r.breakdown.platform_fit,
            visual_from_image=r.visual_from_image) for r in ranked}
        brief_d = scenario.brief_to_dict(brief)
        gallery = scenario.pipeline_to_gallery(result, detail_map, BLOB.base)
        # 第 3 步 Campaign 即时构建（与漏斗其余步骤同一商品）
        picks, qof = scenario.live_campaign_inputs(
            brief, [a.variant for a in result.approved], score_map)
        campaigns = scenario.build_campaigns_payload(brief, picks, qof)

        # 写 Store（全 JSON，跨 web/worker 共享）：一次输入贯穿全链路
        STORE.set("latest:gallery", gallery)
        STORE.set("latest:campaigns", campaigns)
        STORE.set("latest:qualities", jobs.platform_qualities(gallery))
        STORE.set("latest:daily_usd", brief.budget.daily_usd or 100.0)
        STORE.set("latest:campaign", dict(product=brief.product.name,
                                          daily_usd=brief.budget.daily_usd,
                                          n_pass=gallery.get("n_pass", 0)))
        STORE.set("latest:version", (STORE.get("latest:version") or 0) + 1)
        # 触发本次创意对应的投放模拟（本地线程 / 云 worker）
        if STORE.get("latest:qualities"):
            _enqueue_pilot(lang)
        return JSONResponse(
            {"brief": brief_d, "questions": questions, "creatives": gallery})
    except Exception as e:  # LLM/抓页失败: 返回可读错误，不 500
        return JSONResponse({"error": f"生成失败：{e}"}, status_code=502)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
