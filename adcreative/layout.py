"""图片管线（W3 降级预案实现）: 渐变背景 + 文案排版 + CTA 按钮，纯 Pillow 本地渲染。

设计文档 4.2 的完整管线是 文生图 -> 抠图 -> 场景合成 -> 排版；风险清单明确
降级预案为"抠图产品 + 纯色/渐变背景 + 排版"。本模块实现排版层（P0 核心，
零外部 API），并留 product_image 参数——上游生图/抠图模块就绪后直接传入。

三个标准尺寸（设计文档 4.2）:
    feed_1x1   1080x1080  Meta/TikTok Feed
    story_9x16 1080x1920  Story / TikTok 全屏
    land_1911  1200x628   Google 展示 (1.91:1)
"""
from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CREATIVE_SIZES: dict[str, tuple[int, int]] = {
    "feed_1x1": (1080, 1080),
    "story_9x16": (1080, 1920),
    "land_1911": (1200, 628),
}

# 支持中文的字体候选（按优先级）。Pillow 默认字体渲不了 CJK，中文文案会变豆腐块；
# 逐个探测系统字体，Linux/Docker 走 Noto，都没有才回退默认（仅 Latin）。
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)

# emoji / 象形符号：常规字体渲不出，落在图上是豆腐块 ▯，排版前统一剔除。
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF"   # emoji 主区（表情/物件/交通等）
    "\U00002600-\U000027BF"    # 杂项符号 + dingbats
    "\U0001F1E6-\U0001F1FF"    # 区域旗帜
    "\U00002B00-\U00002BFF"    # 杂项符号与箭头（含 ★ 等）
    "\U0000FE00-\U0000FE0F"    # 变体选择符
    "\U0000200D]+"             # 零宽连接符（ZWJ）
)


def _clean_text(s: str) -> str:
    """去掉 emoji 并收拢多余空白（emoji 被删后常留下双空格）。"""
    return re.sub(r"\s{2,}", " ", _EMOJI_RE.sub("", s or "")).strip()


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" or "　" <= c <= "ヿ"
               for c in (s or ""))


def _chars_per_line(w: int, px: int, s: str) -> int:
    """每行可容纳字符数：中文近全角（≈1.05×字号），Latin 约 0.62×。"""
    factor = 1.05 if _has_cjk(s) else 0.62
    return max(int(w / (px * factor)), 6)


def _gradient(size: tuple[int, int], top: tuple[int, int, int],
              bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, h)[:, None]
    col = np.asarray(top) * (1 - t) + np.asarray(bottom) * t   # (h, 3)
    arr = np.broadcast_to(col[:, None, :], (h, w, 3)).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def _font(px: int) -> ImageFont.FreeTypeFont:
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, px)
        except Exception:
            pass
    return ImageFont.load_default(size=px)


def render_creative(
    headline: str,
    body: str = "",
    cta: str = "Shop Now",
    size: tuple[int, int] = CREATIVE_SIZES["feed_1x1"],
    brand_color: tuple[int, int, int] = (28, 78, 128),
    product_image: Image.Image | None = None,
    out_path: str | Path | None = None,
) -> Image.Image:
    """渲染单张创意图。out_path 给定时同时存 PNG。"""
    headline, body, cta = _clean_text(headline), _clean_text(body), _clean_text(cta)
    w, h = size
    darker = tuple(int(c * 0.45) for c in brand_color)
    img = _gradient(size, brand_color, darker)  # type: ignore[arg-type]
    draw = ImageDraw.Draw(img)

    # 产品图（可选）: 居中偏上，等比缩放到画布 45% 高
    y_cursor = int(h * 0.10)
    if product_image is not None:
        target_h = int(h * 0.45)
        scale = target_h / product_image.height
        pw = int(product_image.width * scale)
        prod = product_image.resize((pw, target_h))
        img.paste(prod, ((w - pw) // 2, y_cursor),
                  prod if prod.mode == "RGBA" else None)
        y_cursor += target_h + int(h * 0.04)

    # 标题: 按画布宽度折行（中文全角字更宽，每行字数按 CJK 自适应，防溢出）
    hl_px = max(int(w * 0.055), 28)
    hl_font = _font(hl_px)
    chars_per_line = _chars_per_line(w, hl_px, headline)
    for line in textwrap.wrap(headline, chars_per_line)[:3]:
        tw = draw.textlength(line, font=hl_font)
        draw.text(((w - tw) / 2, y_cursor), line, font=hl_font, fill="white")
        y_cursor += int(hl_px * 1.25)

    # 正文（可选）
    if body:
        body_px = max(int(hl_px * 0.55), 18)
        body_font = _font(body_px)
        y_cursor += int(h * 0.02)
        for line in textwrap.wrap(body, _chars_per_line(w, body_px, body))[:4]:
            tw = draw.textlength(line, font=body_font)
            draw.text(((w - tw) / 2, y_cursor), line, font=body_font,
                      fill=(235, 235, 235))
            y_cursor += int(body_px * 1.35)

    # CTA 按钮: 底部居中的圆角矩形
    cta_px = max(int(w * 0.04), 22)
    cta_font = _font(cta_px)
    tw = draw.textlength(cta, font=cta_font)
    pad_x, pad_y = int(cta_px * 1.2), int(cta_px * 0.6)
    bw, bh = tw + 2 * pad_x, cta_px + 2 * pad_y
    bx, by = (w - bw) / 2, h - bh - int(h * 0.06)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh],
                           radius=bh / 2, fill="white")
    draw.text((bx + pad_x, by + pad_y), cta, font=cta_font, fill=darker)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
    return img


def render_all_sizes(
    headline: str, body: str, cta: str,
    out_dir: str | Path, stem: str,
    brand_color: tuple[int, int, int] = (28, 78, 128),
    product_image: Image.Image | None = None,
) -> dict[str, Path]:
    """一组文案 -> 三个标准尺寸的 PNG。返回 {尺寸名: 文件路径}。"""
    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}
    for name, size in CREATIVE_SIZES.items():
        p = out_dir / f"{stem}_{name}.png"
        render_creative(headline, body, cta, size=size,
                        brand_color=brand_color,
                        product_image=product_image, out_path=p)
        paths[name] = p
    return paths
