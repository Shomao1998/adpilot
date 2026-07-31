"""配图来源测试（抓主图 / 文生图 / 三级兜底），全走 mock httpx，不触网。"""
from __future__ import annotations

import io

import httpx
import pytest
from PIL import Image

from adcreative.imagery import (
    FalImageSource, OpenAICompatImageSource, build_image_prompt,
    make_image_source, resolve_product_image, scrape_product_image,
)
from adcreative.schema import Brief, Product

BRIEF = Brief(product=Product(name="便携榨汁机", category="厨房电器",
                              selling_points=["便携", "可充电"]))
HTML_OG = ('<html><head><meta property="og:image" '
           'content="https://cdn.example.com/p.jpg"></head></html>')


def _png(size=(300, 300)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, (20, 30, 40)).save(b, "PNG")
    return b.getvalue()


class TestScrape:
    def _client(self, html, img_bytes):
        def handler(req):
            if req.url.path.endswith((".jpg", ".png")) or "cdn." in req.url.host:
                return httpx.Response(200, content=img_bytes,
                                     headers={"content-type": "image/png"})
            return httpx.Response(200, text=html)
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_scrapes_og_image(self):
        img = scrape_product_image(
            "https://shop.example.com/p", http_client=self._client(HTML_OG, _png()))
        assert img is not None and img.mode == "RGBA"

    def test_no_og_returns_none(self):
        assert scrape_product_image(
            "https://x.com/p", http_client=self._client("<html></html>", _png())) is None

    def test_tiny_image_skipped(self):
        """图标/追踪像素（<200px）跳过。"""
        assert scrape_product_image(
            "https://x.com/p", http_client=self._client(HTML_OG, _png((50, 50)))) is None

    def test_jsonld_image(self):
        """无 og:image，但有 JSON-LD Product image。"""
        html = ('<html><head><script type="application/ld+json">'
                '{"@type":"Product","name":"X",'
                '"image":["https://cdn.example.com/p.jpg"]}</script></head></html>')
        img = scrape_product_image(
            "https://shop.example.com/p", http_client=self._client(html, _png()))
        assert img is not None

    def test_link_image_src(self):
        html = '<html><head><link rel="image_src" href="https://cdn.example.com/p.png"></head></html>'
        img = scrape_product_image(
            "https://shop.example.com/p", http_client=self._client(html, _png()))
        assert img is not None

    def test_network_error_returns_none(self):
        def handler(req):
            raise httpx.ConnectError("boom")
        c = httpx.Client(transport=httpx.MockTransport(handler))
        assert scrape_product_image("https://x.com/p", http_client=c) is None


class TestFal:
    def _client(self, json_body, img_bytes):
        def handler(req):
            if req.method == "POST":
                return httpx.Response(200, json=json_body)
            return httpx.Response(200, content=img_bytes,
                                 headers={"content-type": "image/png"})
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_generates_image(self):
        c = self._client({"images": [{"url": "https://fal.media/x.png"}]}, _png())
        assert FalImageSource(api_key="k", http_client=c).generate("p") is not None

    def test_empty_images_returns_none(self):
        c = self._client({"images": []}, _png())
        assert FalImageSource(api_key="k", http_client=c).generate("p") is None


class TestResolveTiers:
    def _img(self):
        return Image.new("RGBA", (300, 300))

    def test_url_uses_scrape(self):
        img = self._img()
        got = resolve_product_image(BRIEF, "https://x.com/p",
                                    image_source=None, scrape=lambda u: img)
        assert got is img

    def test_url_falls_back_to_source_when_scrape_empty(self):
        img = self._img()
        src = type("S", (), {"generate": lambda self, p: img})()
        got = resolve_product_image(BRIEF, "https://x.com/p",
                                    image_source=src, scrape=lambda u: None)
        assert got is img

    def test_keyword_uses_source(self):
        img = self._img()
        src = type("S", (), {"generate": lambda self, p: img})()
        got = resolve_product_image(BRIEF, "便携榨汁机",
                                    image_source=src, scrape=lambda u: None)
        assert got is img

    def test_none_when_nothing_available(self):
        assert resolve_product_image(BRIEF, "关键词",
                                     image_source=None, scrape=lambda u: None) is None


class TestOpenAICompat:
    """硅基流动等 OpenAI 兼容文生图：容错解析多种响应形状。"""

    def _client(self, json_body, img_bytes=None):
        def handler(req):
            if req.method == "POST":
                return httpx.Response(200, json=json_body)
            return httpx.Response(200, content=img_bytes or _png(),
                                 headers={"content-type": "image/png"})
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_images_url_shape(self):
        c = self._client({"images": [{"url": "https://cdn/x.png"}]})
        src = OpenAICompatImageSource(api_key="k", http_client=c)
        assert src.generate("p") is not None

    def test_data_url_shape(self):
        c = self._client({"data": [{"url": "https://cdn/x.png"}]})
        assert OpenAICompatImageSource(api_key="k", http_client=c).generate("p") is not None

    def test_b64_shape(self):
        import base64
        c = self._client({"data": [{"b64_json": base64.b64encode(_png()).decode()}]})
        assert OpenAICompatImageSource(api_key="k", http_client=c).generate("p") is not None

    def test_empty_response_none(self):
        c = self._client({"images": []})
        assert OpenAICompatImageSource(api_key="k", http_client=c).generate("p") is None

    def test_sends_auth_and_model(self):
        import json as _json
        cap = {}

        def handler(req):
            if req.method == "POST":
                cap["body"] = _json.loads(req.content)
                cap["auth"] = req.headers.get("authorization")
                cap["url"] = str(req.url)
                return httpx.Response(200, json={"images": [{"url": "https://cdn/x.png"}]})
            return httpx.Response(200, content=_png(),
                                 headers={"content-type": "image/png"})
        c = httpx.Client(transport=httpx.MockTransport(handler))
        OpenAICompatImageSource(api_key="sk-test", http_client=c).generate("a pan")
        assert cap["auth"] == "Bearer sk-test"
        assert cap["body"]["prompt"] == "a pan"
        assert "FLUX" in cap["body"]["model"]
        assert cap["url"].endswith("/images/generations")


class TestFactory:
    def test_fal_key_present(self, monkeypatch):
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
        monkeypatch.delenv("ADPILOT_IMAGE_PROVIDER", raising=False)
        monkeypatch.setenv("FAL_KEY", "x")
        assert isinstance(make_image_source(), FalImageSource)

    def test_siliconflow_key_present(self, monkeypatch):
        monkeypatch.delenv("ADPILOT_IMAGE_PROVIDER", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-x")
        assert isinstance(make_image_source(), OpenAICompatImageSource)

    def test_explicit_provider_wins(self, monkeypatch):
        """两个 key 都在时，ADPILOT_IMAGE_PROVIDER 说了算。"""
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-x")
        monkeypatch.setenv("FAL_KEY", "x")
        monkeypatch.setenv("ADPILOT_IMAGE_PROVIDER", "fal")
        assert isinstance(make_image_source(), FalImageSource)

    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
        monkeypatch.delenv("ADPILOT_IMAGE_PROVIDER", raising=False)
        assert make_image_source() is None


def test_image_prompt_mentions_product():
    p = build_image_prompt(BRIEF)
    assert "便携榨汁机" in p and "no text" in p
