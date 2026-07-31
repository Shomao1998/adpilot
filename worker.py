"""RQ worker 入口（云模式）：消费队列里的长任务（pilot 模拟等）。

docker-compose 里作为独立容器运行，与 Web 容器共享同一 Redis。
用法: ADPILOT_BACKEND=cloud REDIS_URL=redis://redis:6379/0 python worker.py
"""
from __future__ import annotations

import os

if __name__ == "__main__":
    from redis import Redis
    from rq import Queue, Worker

    conn = Redis.from_url(os.environ["REDIS_URL"])
    Worker([Queue("adpilot", connection=conn)], connection=conn).work()
