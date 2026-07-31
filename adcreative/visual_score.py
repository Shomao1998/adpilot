"""创意图视觉打分（可插拔）。

CTR 预排序的 visual_appeal 维度：以前是 LLM 从文案「猜」视觉分（模型看不到图），
现在改由真正分析渲染出的创意图得出。

设计成协议 + 实现，便于将来平滑升级：
    VisualScorer.score(image) -> [0,1]
    - LocalCVVisualScorer（默认，A 方案）：纯本地 CV 指标，零外部依赖、确定性、
      可写测试。度量的是「图像技术质量」（对比度/色彩/清晰度），是审美的代理。
    - 将来 B 方案：实现同一协议、内部调用 VLM 看图打分即可，上层 rank_creatives
      与前端翻转卡都无需改动。

指标全部归一到 [0,1] 再加权。归一参考常数按典型电商创意图标定，落在 0.4–0.9 区间。
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from PIL import Image


class VisualScorer(Protocol):
    def score(self, image: Image.Image) -> float: ...


def _metrics(image: Image.Image) -> dict[str, float]:
    """从图像算三项归一指标：对比度、色彩丰富度、清晰度。"""
    im = image.convert("RGB")
    im.thumbnail((256, 256))                    # 缩小提速，指标对分辨率不敏感
    a = np.asarray(im, dtype=np.float64)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    L = 0.299 * R + 0.587 * G + 0.114 * B       # 亮度

    # 1) 对比度：亮度标准差（画面明暗层次）。参考 ~95 为高对比（留梯度不早饱和）。
    contrast = min(L.std() / 95.0, 1.0)

    # 2) 色彩丰富度：Hasler–Süsstrunk colorfulness metric。参考 ~75 为很鲜艳。
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorful = (math.sqrt(rg.std() ** 2 + yb.std() ** 2)
                + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    colorful = min(colorful / 75.0, 1.0)

    # 3) 清晰度：拉普拉斯响应方差（边缘能量，越高越锐/越有主体细节）。
    lap = (L[:-2, 1:-1] + L[2:, 1:-1] + L[1:-1, :-2] + L[1:-1, 2:]
           - 4 * L[1:-1, 1:-1])
    sharp = min(lap.var() / 650.0, 1.0)

    return {"contrast": contrast, "colorful": colorful, "sharp": sharp}


class LocalCVVisualScorer:
    """A 方案：纯本地 CV 指标合成视觉分。确定性、零外部依赖。"""

    # 三项权重：对比度与色彩对「广告吸睛度」贡献大，清晰度其次。
    _W = {"contrast": 0.40, "colorful": 0.35, "sharp": 0.25}

    def score(self, image: Image.Image) -> float:
        m = _metrics(image)
        v = sum(self._W[k] * m[k] for k in self._W)
        return float(min(max(v, 0.05), 0.98))     # 收进 [0.05, 0.98]，避免极端


_DEFAULT = LocalCVVisualScorer()


def default_visual_scorer() -> VisualScorer:
    return _DEFAULT
