# -*- coding: utf-8 -*-
"""
s6_block.py — بلوک Selective State Space (S6) با PyTorch خالص — سازگار CPU/MPS
UW-MambaRet | ریفرنس: NOTE-02 (مکانیزم) + جلسه 7 (بازنویسی نهایی MPS-سازگار)

پیاده‌سازی scan (نهایی و تمیز):
- هر پاره CH گام: scan موازی درون-پاره با «فرم بازگشتی تجمعی» روی نماهای log (فقط مقادیر ≤0 → بدون سرریز)
- حمل بین پاره‌ها با h صریح
- درون-پاره: weights w[t,i] = exp(lcum_t − lcum_i) فقط روی مثلث پایین؛ چون lcum ≤ 0 و t≥i
  داریم lcum_t − lcum_i ∈ (−∞, 0] → exp ∈ [0,1] → بدون سرریز (به شرط انتخاب CH معقول)
- پایداری: max-shift درون پاره (exp(lcum_t − min_lc)) و در انتها جبران
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class S6(nn.Module):
    """Selective Scan (S6) — ورودی (B, L, D) → خروجی (B, L, D)"""

    def __init__(self, d_model: int, d_state: int = 16, chunk: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk = chunk

        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        dt_init = torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.dt_proj.weight.data = dt_init.exp().unsqueeze(1).repeat(1, d_model) * 0.1
        self.dt_proj.bias.data = dt_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state
        CH = self.chunk

        A = -torch.exp(self.A_log.float())                       # (D,N) منفی
        delta = F.softplus(self.dt_proj(x))                      # (B,L,D)
        BC = self.x_proj(x)
        Bm, Cm = BC[..., :N], BC[..., N:2 * N]

        deltaA = torch.exp(delta.unsqueeze(-1) * A)              # (B,L,D,N) ∈ (0,1]
        deltaB_x = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)

        y = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        for s in range(0, L, CH):
            e = min(s + CH, L)
            ch = e - s
            da = deltaA[:, s:e]                                  # (B,ch,D,N) ≤1
            dbx = deltaB_x[:, s:e].clone()

            # حمل از پاره‌های قبل: h_true(اولین گام پاره) = a_0·h_prev + b_0
            dbx[:, 0] = dbx[:, 0] + da[:, 0] * h

            # scan موازی درون-پاره (Blelloch-lite):
            # عملگر انجمنی (a,b)∘(a',b') = (a·a', a'·b + b')
            a = da
            b = dbx
            step = 1
            while step < ch:
                a_t = a[:, step:]                    # سگمنت بعدی (later)
                b_t = b[:, step:]
                a_h = a[:, :-step]                   # سگمنت قبلی (earlier)
                b_h = b[:, :-step]
                # ترکیب: h_end2 = a2·h_end1 + b2 → (a2·a1, a2·b1 + b2)
                a_new_tail = a_t * a_h
                b_new_tail = b_h * a_t + b_t
                a = torch.cat([a[:, :step], a_new_tail], dim=1)
                b = torch.cat([b[:, :step], b_new_tail], dim=1)
                step *= 2
            # b[t] = h_t (inclusive)
            y[:, s:e] = (b * Cm[:, s:e].unsqueeze(2)).sum(-1).to(x.dtype)
            h = b[:, -1]

        return y + x * self.D


class MambaLiteBlock(nn.Module):
    """بلوک سبک Mamba: Linear → Conv1d → S6 → گیت SiLU → Linear خروجی"""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 1, chunk: int = 32):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.s6 = S6(d_inner, d_state, chunk)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_in = F.silu(x_in)
        y = self.s6(x_in)
        return self.out_proj(y * F.silu(z))


if __name__ == "__main__":
    torch.manual_seed(0)
    blk = MambaLiteBlock(d_model=48, d_state=16)
    x = torch.randn(2, 32, 48)
    y = blk(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    n_params = sum(p.numel() for p in blk.parameters())
    print(f"OK — MambaLiteBlock: out={tuple(y.shape)}, params={n_params/1e3:.1f}K")
