# -*- coding: utf-8 -*-
"""
evaluate_retrieval.py — ارزیابی بازیابی: mAP@K و Recall@K (بدون FAISS — صراحت ساده و MPS-پسند)
پروتکل: embedding های L2-نرمال → شباهت کسینوسی = ضرب ماتریسی → رتبه‌بندی
"""

import torch


@torch.no_grad()
def evaluate_retrieval(query_emb: torch.Tensor, db_emb: torch.Tensor,
                       query_labels: torch.Tensor, db_labels: torch.Tensor,
                       ks=(1, 5, 10, 20)):
    """
    query_emb: (Q,d)، db_emb: (N,d) — هر دو L2-normalized
    خروجی: dict {mAP, R@1, R@5, R@10, R@20}
    """
    # ارزیابی در CPU انجام می‌شود (کم‌هزینه است و از مشکل indexing MPS پرهیز)
    query_emb = query_emb.detach().cpu()
    db_emb = db_emb.detach().cpu()
    query_labels = query_labels.cpu()
    db_labels = db_labels.cpu()

    sim = query_emb @ db_emb.t()                     # (Q,N) کسینوس
    Q, N = sim.shape

    # حذف خود query وقتی در db است (همان تصویر): تطابق بر اساس ایندکس نیست؛
    # فرض: caller از قبل db بدون query ساخته یا query∈db را با label-check می‌پذیریم (پروتکل ماتریس کامل).
    order = torch.argsort(sim, dim=1, descending=True)          # (Q,N) رتبه‌ها
    match = db_labels[order] == query_labels.unsqueeze(1)       # (Q,N) بولی

    # ── Recall@K
    res = {}
    for k in ks:
        res[f"R@{k}"] = match[:, :k].any(dim=1).float().mean().item()

    # ── mAP (کات K=همه؛ فرم استاندارد retrieval با multiple-good)
    ap = torch.zeros(Q)
    for i in range(Q):
        m = match[i]
        n_rel = m.sum()
        if n_rel == 0:
            continue
        ranks = torch.arange(1, N + 1, dtype=torch.float32)[m]  # رتبه‌های مرتبط
        prec = torch.arange(1, len(ranks) + 1, dtype=torch.float32) / ranks
        ap[i] = prec.mean()
    res["mAP"] = (ap[ap > 0].mean().item() if (ap > 0).any() else 0.0)
    res["mAP@10"] = _map_at_k(match, 10)
    return res


def _map_at_k(match: torch.Tensor, k: int) -> float:
    Q, N = match.shape
    aps = []
    for i in range(Q):
        m = match[i, :k]
        n_rel = m.sum()
        if n_rel == 0:
            continue
        ranks = torch.arange(1, k + 1, dtype=torch.float32)[m]
        aps.append((torch.arange(1, len(ranks) + 1, dtype=torch.float32) / ranks).mean())
    return float(torch.tensor(aps).mean()) if aps else 0.0


if __name__ == "__main__":
    torch.manual_seed(0)
    # سناریوی ساختگی: 6 query، 20 db، 4 کلاس — کیفیت بالا با جابه‌جایی کم
    centers = F_onehot = torch.eye(4)[torch.tensor([0, 1, 2, 3])].repeat(5, 1) * 10.0
    db_labels = torch.arange(4).repeat(5)
    db_emb = F_onehot + torch.randn(20, 4) * 0.1
    db_emb = torch.nn.functional.normalize(db_emb, dim=1)
    q_labels = torch.arange(4)
    q_emb = torch.nn.functional.normalize(torch.eye(4)[q_labels] * 10 + torch.randn(4, 4) * 0.1, dim=1)
    r = evaluate_retrieval(q_emb, db_emb, q_labels, db_labels)
    assert r["mAP"] > 0.9, f"sanity mAP کم: {r}"
    print(f"OK — evaluate: {r}")
