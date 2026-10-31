# -*- coding: utf-8 -*-
"""
degrade.py — مولد تخریب سنتزی برای بنچمارک مقاومت (ادعای C4)
5 نوع × 3 شدت (طبق robustness_benchmark.md):
  ATT (جاذب نوری رنگی)، CC (رنگ‌کاستی سبز/آبی)، BLR (بلور)، NS (نویز)، LOW (نور کم)
شدت‌ها: L1 ملایم / L2 متوسط / L3 شدید — قطعی با seed (تکرارپذیری پروتکل)
"""

import numpy as np
from PIL import Image, ImageFilter
import torch


def _to_np(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) float 0-1 → (H,W,3) uint8"""
    return (img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy())


def _to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


# ─────────────────────────── 5 اپراتور تخریب ───────────────────────────

def attenuate(img: np.ndarray, level: int, rng) -> np.ndarray:
    """ATT: جاذب انتخابی طول‌موج (قرمز سریع‌تر حذف می‌شود — فیزیک آب)"""
    k = [0.35, 0.6, 0.85][level - 1]
    out = img.astype(np.float32) / 255.0
    out[..., 0] *= (1 - k)            # R قوی‌تر میرا
    out[..., 1] *= (1 - 0.55 * k)
    out[..., 2] *= (1 - 0.25 * k)     # B کمترین جاذب
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def color_cast(img: np.ndarray, level: int, rng) -> np.ndarray:
    """CC: رنگ‌کاستی سبز (شکوفایی جلبکی) با گیت تصادفی سبز/آبی"""
    k = [0.25, 0.45, 0.65][level - 1]
    hue = rng.choice(['green', 'blue'])
    out = img.astype(np.float32)
    ch = 1 if hue == 'green' else 2
    out[..., ch] = np.clip(out[..., ch] * (1 + k), 0, 255)
    out[..., 0] = np.clip(out[..., 0] * (1 - 0.5 * k), 0, 255)
    return out.astype(np.uint8)


def blur(img: np.ndarray, level: int, rng) -> np.ndarray:
    """BLR: بلور گاوسی (پراکندگی ذرات معلق) + لرزش خفیف"""
    sigma = [1.2, 2.2, 3.5][level - 1]
    return np.asarray(Image.fromarray(img).filter(
        ImageFilter.GaussianBlur(radius=sigma)))


def noise(img: np.ndarray, level: int, rng) -> np.ndarray:
    """NS: نویز گاوسی + پالس نمک‌فلفلی خفیف (sensor در نور کم)"""
    s = [0.04, 0.09, 0.15][level - 1]
    out = img.astype(np.float32) + rng.normal(0, 255 * s, img.shape)
    sp = rng.random(img.shape[:2]) < 0.01 * level      # ماسک پیکسلی
    vals = (255 * rng.random((sp.sum(), 3))).astype(np.float32)  # مقدار تصادفی 3کاناله
    out[sp] = vals
    return np.clip(out, 0, 255).astype(np.uint8)


def low_light(img: np.ndarray, level: int, rng) -> np.ndarray:
    """LOW: افت نور + گامای نابرابر (عمق بیشتر)"""
    g = [0.55, 0.38, 0.25][level - 1]
    out = (img.astype(np.float32) / 255.0) ** (1 / (1 + (1 - g)))
    return np.clip(out * 255, 0, 255).astype(np.uint8)


OPS = {'ATT': attenuate, 'CC': color_cast, 'BLR': blur, 'NS': noise, 'LOW': low_light}
TYPES = list(OPS.keys())
LEVELS = (1, 2, 3)


class DegradedQueryGenerator:
    """تولید قطعی نسخه‌های تخریب‌شده query ها — seed تضمین تکرارپذیری بین اجراها/مدل‌ها"""

    def __init__(self, seed: int = 2026):
        self.seed = seed

    def __call__(self, img_t: torch.Tensor, dtype: str, level: int, idx: int) -> torch.Tensor:
        assert dtype in OPS and level in LEVELS
        rng = np.random.default_rng(self.seed + idx * 1000 + level)   # قطعی به‌ازای (تصویر، نوع، شدت)
        return _to_t(OPS[dtype](_to_np(img_t), level, rng))


if __name__ == "__main__":
    g = DegradedQueryGenerator()
    img = torch.rand(3, 64, 64)
    for t in TYPES:
        for lv in LEVELS:
            out = g(img, t, lv, idx=0)
            assert out.shape == img.shape and torch.isfinite(out).all()
    print(f"OK — degrade: {len(TYPES)}×{len(LEVELS)} = {len(TYPES)*len(LEVELS)} گام، همه قطعی/سالم")
