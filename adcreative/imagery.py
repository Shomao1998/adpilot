"""创意配图来源（图片管线的真实实现，替代纯渐变兜底）。

三级兜底（resolve_product_image）:
    A. URL 输入 -> 抓商品页主图（og:image / twitter:image），零额外 key。
    B. A 拿不到（纯关键词 / 页面无主图）且配了 fal.ai -> Flux 文生图。
    C. 都没有 -> 返回 None，渲染层（render_creative）回退到渐变模板。

文字与 CTA 始终由 Pillow 排版叠加，不依赖模型渲染文字（模型写字很烂）。
所有网络/解析失败都吞成 None -> 安全降级，绝不因配图问题中断创意生成。
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Callable, Protocol
from urllib.parse import urljoin

from PIL import Image

from adcreative.schema import Brief

_MIN_DIM = 200      # 小于此尺寸多半是图标/追踪像素，跳过
_MAX_DIM = 900      # 下载后封顶，避免超大图拖慢合成
# 不少站点对无 UA 的请求返回 202 空响应（反爬），必须带浏览器 UA
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/120.0.0.0 Safari/537.36")


def _load_image(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if min(img.size) < _MIN_DIM:
        return None
    img = img.convert("RGBA")
    if max(img.size) > _MAX_DIM:
        s = _MAX_DIM / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img


_META_PATTERNS = [
    r'<meta[^>]+property=["\']og:image(?::url)?["\'][^>]+content=["\']([^"\']+)["\']',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::url)?["\']',
    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
]


def _jsonld_image(block: str) -> str | None:
    """从一段 JSON-LD 里找 image（schema.org Product 常见），支持 str/list/ImageObject。"""
    try:
        data = json.loads(block)
    except Exception:
        return None
    found: list[str] = []

    def collect(v):
        if isinstance(v, str):
            found.append(v)
        elif isinstance(v, dict):
            u = v.get("url") or v.get("contentUrl")
            if u:
                found.append(u)
        elif isinstance(v, list):
            for x in v:
                collect(x)

    def walk(o):
        if isinstance(o, dict):
            for k, val in o.items():
                if k.lower() == "image":
                    collect(val)
                walk(val)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(data)
    return found[0] if found else None


def _find_image_url(html: str, base_url: str) -> str | None:
    # 1. og:image / twitter:image（最常见）
    for pat in _META_PATTERNS:
        m = re.search(pat, html, re.I)
        if m:
            return urljoin(base_url, m.group(1).strip())
    # 2. <link rel="image_src">
    m = re.search(r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
                  html, re.I)
    if m:
        return urljoin(base_url, m.group(1).strip())
    # 3. JSON-LD 结构化数据里的 image（电商标准）
    for block in re.findall(
            r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.I | re.S):
        u = _jsonld_image(block)
        if u:
            return urljoin(base_url, u.strip())
    return None


def scrape_product_image(url: str, http_client=None) -> Image.Image | None:
    """A：抓商品页主图。取 og:image / twitter:image，下载并转 RGBA。失败返回 None。"""
    import httpx

    client = http_client or httpx.Client(
        timeout=15.0, follow_redirects=True,
        headers={"User-Agent": _BROWSER_UA})
    try:
        r = client.get(url)
        r.raise_for_status()
        img_url = _find_image_url(r.text, url)
        if not img_url:
            return None
        ir = client.get(img_url)
        ir.raise_for_status()
        return _load_image(ir.content)
    except Exception:
        return None
    finally:
        if http_client is None:
            client.close()


class ImageSource(Protocol):
    def generate(self, prompt: str) -> Image.Image | None: ...


def build_image_prompt(brief: Brief) -> str:
    """由 brief 构造文生图提示词（只生成产品/场景图，文字交给 Pillow）。"""
    p = brief.product
    sp = ", ".join(p.selling_points[:3])
    base = (f"Professional commercial product photo of {p.name}"
            + (f", a {p.category}" if p.category else "")
            + ", clean studio lighting, minimal seamless background, centered, "
              "high detail, advertising photography, no text")
    return f"{base}. Highlights: {sp}" if sp else base


class FalImageSource:
    """B：fal.ai Flux 文生图。需环境变量 FAL_KEY。"""

    def __init__(self, model: str = "fal-ai/flux/schnell", api_key: str | None = None,
                 timeout: float = 90.0, http_client=None):
        self.model = model
        self.api_key = (api_key or os.environ.get("FAL_KEY") or "").strip()
        self.timeout = timeout
        self._client = http_client

    def generate(self, prompt: str) -> Image.Image | None:
        import httpx

        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            r = client.post(
                f"https://fal.run/{self.model}",
                headers={"Authorization": f"Key {self.api_key}"},
                json={"prompt": prompt, "image_size": "square_hd", "num_images": 1})
            r.raise_for_status()
            images = r.json().get("images") or []
            if not images:
                return None
            ir = client.get(images[0]["url"])
            ir.raise_for_status()
            return _load_image(ir.content)
        except Exception:
            return None
        finally:
            if self._client is None:
                client.close()


class OpenAICompatImageSource:
    """OpenAI 兼容的文生图（硅基流动 SiliconFlow / 智谱 等国内服务）。

    默认指向硅基流动的 FLUX.1-schnell（国内可用、人民币结算、便宜）。
    响应形状各家略有差异，这里容错解析 images[].url / data[].url / data[].b64_json。
    """

    def __init__(self, model: str = "black-forest-labs/FLUX.1-schnell",
                 base_url: str = "https://api.siliconflow.cn/v1",
                 api_key: str | None = None,
                 api_key_env: str = "SILICONFLOW_API_KEY",
                 image_size: str = "1024x1024",
                 timeout: float = 120.0, http_client=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or os.environ.get(api_key_env) or "").strip()
        self.image_size = image_size
        self.timeout = timeout
        self._client = http_client

    def _extract(self, data: dict, client) -> Image.Image | None:
        import base64
        for key in ("images", "data"):
            arr = data.get(key) or []
            if not arr:
                continue
            first = arr[0]
            url = b64 = None
            if isinstance(first, dict):
                url, b64 = first.get("url"), first.get("b64_json")
            elif isinstance(first, str):
                url = first
            if b64:
                return _load_image(base64.b64decode(b64))
            if url:
                ir = client.get(url)
                ir.raise_for_status()
                return _load_image(ir.content)
        return None

    def generate(self, prompt: str) -> Image.Image | None:
        import httpx

        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            r = client.post(
                f"{self.base_url}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "prompt": prompt,
                      "image_size": self.image_size, "batch_size": 1})
            r.raise_for_status()
            return self._extract(r.json(), client)
        except Exception:
            return None
        finally:
            if self._client is None:
                client.close()


def make_image_source() -> ImageSource | None:
    """按 ADPILOT_IMAGE_PROVIDER 或已设置的 key 自动选择文生图后端。

    siliconflow（SILICONFLOW_API_KEY）/ fal（FAL_KEY）；都没有则返回 None，
    只走 A 抓图 + 渐变兜底。
    """
    provider = (os.environ.get("ADPILOT_IMAGE_PROVIDER") or "").strip().lower()
    has_sf = bool((os.environ.get("SILICONFLOW_API_KEY") or "").strip())
    has_fal = bool((os.environ.get("FAL_KEY") or "").strip())
    if provider == "siliconflow" or (not provider and has_sf):
        return OpenAICompatImageSource()
    if provider == "fal" or (not provider and has_fal):
        return FalImageSource()
    return None


def resolve_product_image(
    brief: Brief, user_input: str,
    image_source: ImageSource | None = None,
    scrape: Callable[[str], Image.Image | None] = scrape_product_image,
) -> Image.Image | None:
    """三级兜底解析一张产品图（整个 brief 复用同一张，省成本）。"""
    img = None
    text = user_input.strip()
    if text.startswith(("http://", "https://")):
        img = scrape(text)                                  # A
    if img is None and image_source is not None:
        img = image_source.generate(build_image_prompt(brief))  # B
    return img                                              # 否则 C（None）
