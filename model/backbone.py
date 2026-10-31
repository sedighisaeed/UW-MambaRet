# -*- coding: utf-8 -*-
"""
backbone.py — UW-MambaRet: معماری کامل سبک برای بازیابی تصویر زیرآبی
ساختار: Stem → UPAM → Stage1..4 (هرکدام VSS blocks + downsample) → EmbedHead
قید طراحی: total params < 12M (ادعای C1 مقاله) — assert خودکار
ابعاد سبک (dims بر اساس NOTE-04 کاهش‌یافته): (48, 96, 192, 384)
"""

import torch
import torch.nn as nn

from .upam import UPAM
from .ss2d import VSSBlock
from .embed_head import EmbedHead


class Downsample(nn.Module):
    """پچ‌مرج (2× کاهش) + کانال‌گستر — سبک‌تر از conv stride-2"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.rearrange = nn.PixelUnshuffle(2)      # (B,C,H,W)→(B,4C,H/2,W/2)
        self.proj = nn.Conv2d(4 * c_in, c_out, 1, bias=False)
        self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.norm(self.proj(self.rearrange(x)))


class UWMambaRet(nn.Module):
    """
    UW-MambaRet backbone + embedding head
    dims=(48,96,192,384)، depths=(2,2,4,2)، groups=4، d_state=8 → هدف: ~6-9M پارامتر
    ورودی: (B,3,224,224) → خروجی: (B, embed_dim) واحد-طول
    """

    def __init__(self, embed_dim: int = 512, depths=(2, 2, 4, 2), dims=(48, 96, 192, 384),
                 groups: int = 4, d_state: int = 8, use_upam: bool = True,
                 normalize: bool = True):
        super().__init__()
        self.use_upam = use_upam
        # Stem: conv 4×4/4 (پچ 16 پیکسلی — مثل VMamba) + BN
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )
        self.upam = UPAM(dims[0]) if use_upam else nn.Identity()

        stages = []
        for i, (c, d) in enumerate(zip(dims, depths)):
            blocks = nn.Sequential(*[VSSBlock(c, groups, d_state) for _ in range(d)])
            stages.append(blocks)
            if i < len(dims) - 1:
                stages.append(Downsample(c, dims[i + 1]))
        self.stages = nn.Sequential(*stages)

        self.head = EmbedHead(dims[-1], embed_dim, normalize=normalize)
        self._check_budget()

    def _check_budget(self):
        n = sum(p.numel() for p in self.parameters())
        assert n < 12_000_000, f"پارامتر {n/1e6:.2f}M از بودجه 12M فراتر است!"
        return n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.upam(x)
        x = self.stages(x)
        return self.head(x)     # (B, embed_dim)، L2-نرمال


if __name__ == "__main__":
    torch.manual_seed(0)
    for use_upam in (True, False):
        m = UWMambaRet(embed_dim=512, use_upam=use_upam)
        x = torch.randn(2, 3, 224, 224)
        e = m(x)
        n = sum(p.numel() for p in m.parameters())
        tag = "UPAM" if use_upam else "NO-UPAM"
        assert e.shape == (2, 512) and torch.isfinite(e).all()
        print(f"OK [{tag}] — embed={tuple(e.shape)}, total={n/1e6:.2f}M (<12M)")
