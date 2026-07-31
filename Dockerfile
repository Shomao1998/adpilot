# AdPilot 看板一键部署镜像。默认离线模式（无需 API key 即可跑全流程）；
# 需真实 LLM 诊断时设 ADPILOT_LIVE=1 + ANTHROPIC_API_KEY（anthropic 包已随 [llm] 装入）。
FROM python:3.12-slim

WORKDIR /app

# 中文字体：slim 镜像不带 CJK 字体，创意图上的中文会渲成豆腐块。装 Noto Sans CJK
# （落在 layout._FONT_CANDIDATES 里的 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc）。
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 先拷依赖清单，利用 Docker 层缓存
COPY pyproject.toml ./
COPY adsim ./adsim
COPY adplatform ./adplatform
COPY adcreative ./adcreative
COPY adpilot ./adpilot
COPY webapp ./webapp
COPY scripts ./scripts
COPY worker.py ./worker.py

# 一个镜像同时服务 Web 与 worker：[cloud] 含 fastapi/uvicorn + redis/rq/boto3/psycopg，
# 本地 docker-compose 不用云依赖也无妨。numpy/scipy/pillow 均有 manylinux wheel。
RUN pip install --no-cache-dir -e '.[web,cloud,llm]'

EXPOSE 8000

# 默认起 Web（local 后端）；云编排里 worker 容器覆盖 command 为 `python worker.py`
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000"]
