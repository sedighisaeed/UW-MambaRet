# -*- coding: utf-8 -*-
"""
losses.py — توابع هزینه metric learning — UW-MambaRet
ترکیب method_design.md: Multi-Similarity (اصلی) + Triplet batch-hard (کمکی)
ریفرنس: SP-09 (Kaya survey)، NC-26 (Reality Check)، Wang et al. CVPR 2019
"""

import torch
import torch.nn.functional as F


def multi_similarity_loss(emb, labels, scale_pos=2.0, scale_neg=40.0, lambda_inv=0.5):
    """Multi-Similarity Loss (Wang et al., CVPR 2019) — نسخه استاندارد:
    L_i = log(1 + Σ_{p∈P_i} exp(−α(S_ip − λ))) + log(1 + Σ_{n∈N_i} exp(−β(λ − S_in)))
    با mining: فقط مثبت با Sp > (max_neg − ε) و منفی با Sn < (min_pos + ε) نگه داشته می‌شوند.
    """
    sim = emb @ emb.t()
    B = len(labels)
    labels = labels.to(emb.device)
    eye = torch.eye(B, dtype=torch.bool, device=emb.device)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    mask_neg = ~mask_pos & ~eye

    # آستانه‌های mining به‌ازای هر ردیف
    neg_max = sim.masked_fill(~mask_neg, -2.0).max(dim=1).values if mask_neg.any() else torch.full((B,), -2.0, device=emb.device)
    pos_min = sim.masked_fill(~mask_pos, 2.0).min(dim=1).values if mask_pos.any() else torch.full((B,), 2.0, device=emb.device)

    keep_pos = mask_pos & (sim > (neg_max - lambda_inv).unsqueeze(1))
    keep_neg = mask_neg & (sim < (pos_min + lambda_inv).unsqueeze(1))

    pos_sp = sim.masked_fill(~keep_pos, 0.0)     # جفت‌های دورریخته صفر → exp(−α(Sp−λ))→exp(αλ) ثابت کوچک؛ دقیق‌تر: حل جمع فقط روی keep
    # فرم پایدار: mask کردن با −inf در logsumexp
    pos_terms = torch.where(keep_pos, -scale_pos * (sim - lambda_inv), torch.full_like(sim, -1e12))
    neg_terms = torch.where(keep_neg, -scale_neg * (lambda_inv - sim), torch.full_like(sim, -1e12))

    lse_pos = torch.logsumexp(pos_terms, dim=1)  # Σ exp(−α(Sp−λ))
    lse_neg = torch.logsumexp(neg_terms, dim=1)  # Σ exp(−β(λ−Sn))
    loss = torch.log1p(lse_pos.exp().clamp(max=1e12)) + torch.log1p(lse_neg.exp().clamp(max=1e12))
    return loss.mean()


def batch_hard_triplet_loss(emb, labels, margin=0.2):
    """Triplet batch-hard (سخت‌ترین مثبت/منفی) — Hermans et al. 2017
    نکته MPS: cdist_backward پیاده نشده → فاصله دستی با广播 (سازگار autograd)
    """
    dist = torch.sqrt(
        (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(-1).clamp_min(1e-12)
    )
    labels = labels.to(emb.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)

    hard_pos = dist.masked_fill(~(same & ~eye), 0.0).max(dim=1).values
    if (~same).any():
        hard_neg = dist.masked_fill(~(~same), float('inf')).min(dim=1).values
    else:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    return F.relu(hard_pos - hard_neg + margin).mean()


class UWRetrievalLoss(torch.nn.Module):
    """Loss نهایی مقاله: MS (وزن 1) + Triplet-hard (وزن w) — ablation ±Triplet"""

    def __init__(self, triplet_weight: float = 0.3, triplet_margin: float = 0.2):
        super().__init__()
        self.w = triplet_weight
        self.margin = triplet_margin

    def forward(self, emb, labels):
        l_ms = multi_similarity_loss(emb, labels)
        l_tri = batch_hard_triplet_loss(emb, labels, self.margin)
        total = l_ms + self.w * l_tri
        return total, {"ms": l_ms.item(), "triplet": l_tri.item()}


if __name__ == "__main__":
    torch.manual_seed(0)
    emb = F.normalize(torch.randn(8, 64, requires_grad=True), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    crit = UWRetrievalLoss()
    loss, parts = crit(emb, labels)
    assert torch.isfinite(loss)
    loss.backward()
    print(f"OK — UWRetrievalLoss: total={loss.item():.4f}, {parts}")
