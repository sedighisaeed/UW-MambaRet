"""
train_mambahash.py — baseline پنجم: MambaHash-style SSM hashing head
====================================================================
اقتباس آموزشی از He & Wei (2025) MambaHash برای مقایسه در جدول main:
- backbone همان UW-MambaRet (1.35M) — منصفانه: فقط head فرق می‌کند
- hashing head: Linear(256→b, bias=False) → ℓ2→{0,1} (sign quantization در eval)
- loss: pairwise surrogate Hamming (EU-like) + quantization (مطابق DHN رویه)
- ارزیابی: mAP با Hamming ranking روی کد binary (b=32/64)
اجرا: python train_mambahash.py --bits 64 --tag mambahash_b64
"""
import argparse, csv, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from model.backbone import UWMambaRet

SPLIT_CSV = Path("../04_experiments/results/splits/f4k_split_v1.csv")
IMG_ROOT = Path("data/fish4k/fish4konwledge-master/fish_image")

TRAIN_TF = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
])
EVAL_TF = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
])


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, lab = self.rows[i][0], int(self.rows[i][1])
        return self.tf(Image.open(IMG_ROOT / rel).convert("RGB")), lab


def load_split():
    rows = list(csv.reader(open(SPLIT_CSV)))[1:]
    return ([r for r in rows if r[3] == s] for s in ("train", "test", "query"))


class HashNet(nn.Module):
    """backbone (frozen-optional) + sign hashing head."""

    def __init__(self, bits=64, embed=256):
        super().__init__()
        self.backbone = UWMambaRet(embed_dim=embed, use_upam=True)
        self.hash = nn.Linear(embed, bits, bias=False)

    def forward(self, x):
        return torch.tanh(self.hash(self.backbone(x)))


def hash_loss(U, Y):
    """DHN-style: pairwise surrogate + quantization. U:(B,b) tanh، Y:(B,)"""
    S = (Y[:, None] == Y[None, :]).float() * 2 - 1          # (B,B) ±1
    dot = U @ U.t() / U.shape[1]                             # normalized inner
    l_pair = ((1 + S.clamp(min=-1) * dot).clamp(min=0).pow(2).sum()
              - torch.diag((1 + torch.diag(S) * torch.diag(dot)).pow(2)).sum())
    l_q = (1 - U.abs()).abs().mean()
    return l_pair / U.shape[0] + 0.01 * l_q


@torch.no_grad()
def embed_binary(model, rows, device, batch=32):
    model.eval()
    ds = SplitDataset(rows, EVAL_TF)
    codes, ys = [], []
    for i in range(0, len(ds), batch):
        imgs = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        U = model(imgs).cpu()
        codes.append(U.sign())                               # {±1}
        ys.extend(int(ds.rows[j][1]) for j in range(i, min(i + batch, len(ds))))
    model.train()
    return torch.cat(codes), torch.tensor(ys)


def hamming_map(Q, G, qy, gy):
    """mAP@all با Hamming ranking (معیار استاندارد hashing)."""
    D = (Q[:, None, :] != G[None, :, :]).sum(-1)             # (Q,G) hamming
    ok = (qy[:, None] == gy[None, :])
    mAPs = []
    for i in range(len(Q)):
        order = D[i].argsort()
        rel = ok[i, order]
        cum = torch.cumsum(rel.float(), 0)
        k = torch.arange(1, len(rel) + 1, dtype=torch.float)
        prec = cum / k
        mAPs.append(((prec * rel.float()).sum() / rel.sum().clamp(min=1)).item())
    return float(np.mean(mAPs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--tag", default="mambahash_b64")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    tr, te, qy = load_split()
    print(f"split: train={len(tr)} test={len(te)} query={len(qy)}")

    dl = DataLoader(SplitDataset(tr, TRAIN_TF), batch_size=args.batch, shuffle=True,
                    num_workers=2, drop_last=True, persistent_workers=False)
    model = HashNet(bits=args.bits).to(device)
    print(f"[{args.tag}] bits={args.bits} — {sum(p.numel() for p in model.parameters())/1e6:.2f}M — {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(f"../04_experiments/results/train_{args.tag}.csv")
    f = open(out, "w", newline=""); wr = csv.writer(f)
    wr.writerow(["epoch", "lr", "loss", "mAP_ham", "sec"])
    ck = Path("checkpoints"); ck.mkdir(exist_ok=True)
    for ep in range(args.epochs):
        t0 = time.time(); tot = n = 0
        for img, y in dl:
            img, y = img.to(device), y.to(device)
            U = model(img)
            loss = hash_loss(U, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        sched.step()
        Q, q_y = embed_binary(model, qy, device)
        G, g_y = embed_binary(model, te, device)
        m = hamming_map(Q, G, q_y, g_y)
        wr.writerow([ep + 1, f"{sched.get_last_lr()[0]:.2e}",
                     f"{tot/max(n,1):.4f}", f"{m:.4f}", f"{time.time()-t0:.1f}"])
        f.flush()
        print(f"[{args.tag}] ep{ep+1}/{args.epochs} loss={tot/max(n,1):.4f} mAP={m:.4f}", flush=True)
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            torch.save(model.state_dict(), ck / f"{args.tag}_ep{ep+1}.pt")
    f.close(); print(f"OK — {out}")


if __name__ == "__main__":
    main()
