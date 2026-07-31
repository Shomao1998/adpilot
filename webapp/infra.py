"""云无关基础设施抽象：一个 `ADPILOT_BACKEND` 开关切换本地/云后端。

三类可换后端（协议 + 两实现），刻意做成云无关——docker-compose 用开源等价物
（Redis / MinIO / Postgres）在本地"模拟云"，换任意云只替换实现，上层零改动：

    Store  状态存储：pilot/latest/scenario 缓存 + 计算锁
        - InMemoryStore  进程内 dict（默认，单机；测试与本地 dev）
        - RedisStore     外部化，多实例共享、重启不丢（云: Redis/ElastiCache/Azure Cache）
    Queue  任务队列：把 1–2 分钟的长任务（向导/模拟）从 Web 请求里挪走
        - ThreadQueue    守护线程（默认，等价于原来的后台预热）
        - RQQueue        Redis 队列 + 独立 worker 容器（云: SQS/Cloud Tasks/Service Bus）
    Blob   对象存储：创意图
        - LocalBlob      static 目录（默认）
        - S3Blob         S3 兼容（云: S3/GCS/Azure Blob；本地用 MinIO）

外部依赖（redis/rq/boto3）只在 cloud 模式惰性导入；local/测试不装也能跑。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

BACKEND = os.getenv("ADPILOT_BACKEND", "local").lower()
_NS = "adpilot"


# --------------------------------------------------------------------------- #
# Store：状态 + 计算锁
# --------------------------------------------------------------------------- #
class Store(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...
    def acquire(self, key: str) -> bool: ...      # 抢占计算锁，成功返回 True
    def release(self, key: str) -> None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._d: dict[str, Any] = {}
        self._locks: set[str] = set()
        self._lk = threading.Lock()

    def get(self, key: str) -> Any | None:
        return self._d.get(key)

    def set(self, key: str, value: Any) -> None:
        self._d[key] = value

    def acquire(self, key: str) -> bool:
        with self._lk:
            if key in self._locks:
                return False
            self._locks.add(key)
            return True

    def release(self, key: str) -> None:
        with self._lk:
            self._locks.discard(key)


class RedisStore:
    """值以 JSON 存（跨进程边界只走可序列化数据）。锁用 SET NX EX 防死锁。"""

    def __init__(self, url: str) -> None:
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        raw = self.r.get(f"{_NS}:v:{key}")
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any) -> None:
        self.r.set(f"{_NS}:v:{key}", json.dumps(value, ensure_ascii=False))

    def acquire(self, key: str) -> bool:
        return bool(self.r.set(f"{_NS}:lock:{key}", "1", nx=True, ex=900))

    def release(self, key: str) -> None:
        self.r.delete(f"{_NS}:lock:{key}")


# --------------------------------------------------------------------------- #
# Queue：长任务
# --------------------------------------------------------------------------- #
class Queue(Protocol):
    def enqueue(self, func: Callable, *args) -> None: ...


class ThreadQueue:
    """守护线程执行（等价于原 threading.Thread 后台预热）。单机即用。"""

    def enqueue(self, func: Callable, *args) -> None:
        threading.Thread(target=func, args=args, daemon=True).start()


class RQQueue:
    """Redis Queue：任务派到独立 worker 容器。按名字派发，避免序列化闭包。"""

    def __init__(self, url: str) -> None:
        import redis
        from rq import Queue as _RQ
        self._q = _RQ("adpilot", connection=redis.Redis.from_url(url))

    def enqueue(self, func: Callable, *args) -> None:
        # 用 "模块.函数" 字符串派发，worker 侧按同名导入执行
        self._q.enqueue(f"{func.__module__}.{func.__name__}", *args,
                        job_timeout=600)


# --------------------------------------------------------------------------- #
# Blob：对象存储（创意图）
# --------------------------------------------------------------------------- #
class Blob(Protocol):
    base: str   # URL 前缀，thumb = f"{base}/{name}"
    def put(self, name: str, data: bytes, content_type: str = "image/png") -> str: ...
    def url(self, name: str) -> str: ...


class LocalBlob:
    def __init__(self, root: Path, url_prefix: str) -> None:
        self.root = root
        self.base = url_prefix.rstrip("/")
        root.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, data: bytes, content_type: str = "image/png") -> str:
        (self.root / name).write_bytes(data)
        return self.url(name)

    def url(self, name: str) -> str:
        return f"{self.base}/{name}"


class S3Blob:
    """S3 兼容对象存储（本地 MinIO / 云 S3/GCS/Azure Blob）。桶设公共读，直接给 URL。"""

    def __init__(self, endpoint: str, bucket: str, key: str, secret: str,
                 public_url: str) -> None:
        import boto3
        self.s3 = boto3.client("s3", endpoint_url=endpoint,
                               aws_access_key_id=key, aws_secret_access_key=secret)
        self.bucket = bucket
        self.base = public_url.rstrip("/")

    def put(self, name: str, data: bytes, content_type: str = "image/png") -> str:
        self.s3.put_object(Bucket=self.bucket, Key=name, Body=data,
                           ContentType=content_type)
        return self.url(name)

    def url(self, name: str) -> str:
        return f"{self.base}/{name}"


# --------------------------------------------------------------------------- #
# 持久化：完成的模拟结果落 Postgres（云模式；本地 no-op）
# --------------------------------------------------------------------------- #
def persist_run(run_id: str, payload: dict) -> None:
    """把一次完成的 pilot 结果落库，供重启后查询历史（云: RDS/Cloud SQL/Azure PG）。"""
    url = os.getenv("DATABASE_URL")
    if BACKEND != "cloud" or not url:
        return
    try:
        import psycopg
        with psycopg.connect(url) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(),"
                "payload JSONB)")
            conn.execute(
                "INSERT INTO runs (id, payload) VALUES (%s, %s) "
                "ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload",
                (run_id, json.dumps(payload, ensure_ascii=False)))
            conn.commit()
    except Exception:
        pass   # 持久化失败不阻断主流程（结果仍在 Store 里）


# --------------------------------------------------------------------------- #
# 工厂：按 ADPILOT_BACKEND 选实现
# --------------------------------------------------------------------------- #
def _make() -> tuple[Store, Queue, Blob]:
    static_dir = Path(__file__).parent / "static"
    if BACKEND == "cloud":
        redis_url = os.environ["REDIS_URL"]
        blob: Blob = S3Blob(
            endpoint=os.environ["S3_ENDPOINT"], bucket=os.environ["S3_BUCKET"],
            key=os.environ["S3_KEY"], secret=os.environ["S3_SECRET"],
            public_url=os.environ["S3_PUBLIC_URL"])
        return RedisStore(redis_url), RQQueue(redis_url), blob
    return (InMemoryStore(), ThreadQueue(),
            LocalBlob(static_dir / "wizard_gen", "/static/wizard_gen"))


STORE, QUEUE, BLOB = _make()
