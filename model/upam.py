# -*- coding: utf-8 -*-
"""
upam.py — ماژول کالیبراسیون فیزیک-آگاه (Underwater Physics-Aware Module)
طراحی بر اساس NOTE-07 (Mamba-UIE: تجزیه 4جزئی؛ UniUIR: گیت محتوامحور) + جدول تصمیم فاز 1
فرم نهایی: تخمین 4 پارامتر فیزیکی (γ_R, γ_G, γ_B برای جبران رنگ + β برای مه/کدوری)
با گیت محتوامحور — تصحیح در فضای ویژگی، بدون decode تصویر (تمایز اصلی با UIE ها).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UPAM(nn.Module):
    """ورودی: feature map پس از stem (B, C, H, W) → خروجی کالیبره‌شده هم‌شکل"""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(                     # گیت محتوامحور (UniUIR-style سبک)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 4),               # (γ_R, γ_G, γ_B, β)
            nn.Sigmoid(),                              # بازه [0,1] → scale-safe
        )
        # پیش‌فرض‌های خنثی: γ=1 (بدون تصحیح)، β=0 (بدون مه‌زدایی)
        self.register_buffer("neutral", torch.tensor([1.0, 1.0, 1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C % 3 == 0, "C باید مضرب 3 باشد (گروه‌های RGB-like)"
        p = self.gate(x)                               # (B, 4)
        dev = p - self.neutral                         # انحراف از حالت خنثی
        cg = C // 3
        # ① جبران رنگ کانالی: γ_c برای هر گروه RGB-like
        gamma = 1.0 + 0.5 * dev[:, :3].unsqueeze(-1).unsqueeze(-1)   # (B,3,1,1)
        gamma_ch = gamma.repeat_interleave(cg, dim=1)                # (B,C,1,1)
        out = x * gamma_ch
        # ② جبران مه/کدوری: شیفت کنتراست جهانی با β (ضریب 0.5 برای پایداری)
        beta = dev[:, 3].view(B, 1, 1, 1) * 0.5
        out = out * (1.0 + beta) - beta * out.mean(dim=(1, 2, 3), keepdim=True)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    m = UPAM(48)
    x = torch.randn(2, 48, 28, 28)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    n = sum(p.numel() for p in m.parameters())
    print(f"OK — UPAM(48): out={tuple(y.shape)}, params={n/1e3:.1f}K")
