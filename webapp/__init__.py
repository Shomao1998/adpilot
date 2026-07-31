"""webapp: W5 看板后端（FastAPI）+ 跨平台对比看板前端。

payload.py  把编排层输出序列化成看板 JSON（纯函数、可测，不依赖 FastAPI）。
server.py   FastAPI: 跑一次 pilot -> 缓存 -> REST 暴露 + 静态看板页。
static/     自包含单页看板（vanilla JS + 内联 SVG 图，无外部依赖）。

前端选型说明: 本 MVP 用零依赖静态页实现设计文档 W5 的核心页「跨平台对比看板」，
可 uvicorn 一键起、当场验证。后端 REST 契约即为 Next.js SPA 的数据源 ——
升级到 Next.js 五页面时后端零改动。
"""
from webapp.payload import build_dashboard_payload

__all__ = ["build_dashboard_payload"]
