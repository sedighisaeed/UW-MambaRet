# -*- coding: utf-8 -*-
"""
ss2d.py — اسکن انتخابی دوبعدی سبک (گروه‌محور) — UW-MambaRet
طراحی بر اساس: NOTE-04 (VMamba/SS2D) + NOTE-05 (MambaHash grouped scan)
تمایز سبکی: به‌جای اجرای موازی 4 مدل SSM کامل روی کل کانال‌ها (4× هزینه VMamba)،
هر گروه کانالی فقط یک جهت اسکن می‌شود (Idea MambaHash) → هزینه ≈ 1× با پوشش جهتی کامل.

مسیرهای 4گانه (مثل VMamba): LR (چپ→راست)، RL، TB (بالا→پایین)، BT
ترتیب گروه‌ها: چرخشی در مراحل مختلف (stage i از جهت i شروع) — پوشش متوازن
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .s6_block import S6

# بازچینش پچ‌ها به دنباله 1D بر اساس جهت
def _scan_paths(h: int, w: int, device):
    """برمی‌گرداند 4 ایندکس (L tensors) برای LR/RL/TB/BT + ایندکس‌های برگشت به مکان"""
    idx = torch.arange(h * w, device=device).view(h, w)
    lr = idx.reshape(-1)                      # چپ→راست، سطر به سطر
    rl = torch.flip(idx, dims=[1]).reshape(-1)
    tb = idx.t().contiguous().reshape(-1)     # بالا→پایین، ستون به ستون
    bt = torch.flip(idx, dims=[0]).t().contiguous().reshape(-1)
    inv_lr = torch.argsort(lr); inv_rl = torch.argsort(rl)
    inv_tb = torch.argsort(tb); inv_bt = torch.argsort(bt)
    return (lr, inv_lr), (rl, inv_rl), (tb, inv_tb), (bt, inv_bt)


class SS2DGrouped(nn.Module):
    """
    SS2D گروهی: ورودی (B, C, H, W) → خروجی هم‌شکل.
    C به G گروه تقسیم می‌شود؛ گروه g با یکی از 4 جهت اسکن می‌شود (g mod 4).
    هر گروه SSM سبک مستقل دارد (d_model = C/G).
    """

    def __init__(self, channels: int, groups: int = 4, d_state: int = 8):
        super().__init__()
        assert channels % groups == 0, "channels باید مضرب groups باشد"
        self.C, self.G = channels, groups
        self.Cg = channels // groups
        self.ssms = nn.ModuleList([S6(self.Cg, d_state) for _ in range(groups)])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        xs = x.view(B, self.G, self.Cg, H * W)   # گروه‌بندی کانالی
        paths = _scan_paths(H, W, device)
        outs = torch.empty_like(xs)

        for g in range(self.G):
            fwd, inv = paths[g % 4]
            seq = xs[:, g].index_select(2, fwd)         # (B, Cg, L)
            seq = seq.transpose(1, 2)                   # (B, L, Cg)
            y = self.ssms[g](seq)                       # SSM سبک
            y = y.transpose(1, 2).index_select(2, inv)  # بازگشت مکانی
            outs[:, g] = y

        out = outs.reshape(B, C, H, W)
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class VSSBlock(nn.Module):
    """بلوک Visual State-Space سبک: ConvDW → SS2DGrouped (با residual)"""

    def __init__(self, channels: int, groups: int = 4, d_state: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.ss2d = SS2DGrouped(channels, groups, d_state)
        self.ffn = nn.Conv2d(channels, channels, 1, bias=False)  # pointwise (جایگزین سبک MLP)

    def forward(self, x):
        x = x + self.dw(x)
        x = x + self.ss2d(x)
        return x + self.ffn(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    b = SS2DGrouped(48, groups=4, d_state=8)
    x = torch.randn(2, 48, 14, 14)
    y = b(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    n = sum(p.numel() for p in b.parameters())
    print(f"OK — SS2DGrouped(48,g=4): out={tuple(y.shape)}, params={n/1e3:.1f}K")
