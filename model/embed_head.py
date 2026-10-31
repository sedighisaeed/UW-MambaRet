# -*- coding: utf-8 -*-
"""
embed_head.py — سر embedding برای بازیابی: GeM Pooling + FC + L2Norm
(پروتکل: NC-26 Metric Learning Reality Check — GeM استاندارد retrieval)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling — p قابل‌آموزش (شروع p=3 استاندارد)"""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)


class EmbedHead(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 512, normalize: bool = True):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_channels, embed_dim, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)   # نرمال‌سازی قبل از L2 (استاندارد DML)
        self.normalize = normalize            # False فقط برای SSL (VICReg به دامنه آزاد نیاز دارد)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.gem(x)
        v = self.bn(self.fc(v))
        return F.normalize(v, p=2, dim=1) if self.normalize else v


if __name__ == "__main__":
    torch.manual_seed(0)
    h = EmbedHead(384, 512)
    x = torch.randn(4, 384, 7, 7)
    e = h(x)
    assert e.shape == (4, 512)
    assert torch.allclose(e.norm(dim=1), torch.ones(4), atol=1e-5)
    n = sum(p.numel() for p in h.parameters())
    print(f"OK — EmbedHead(384→512): out={tuple(e.shape)}, |e|=1, params={n/1e3:.1f}K")
