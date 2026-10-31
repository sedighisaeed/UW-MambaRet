# -*- coding: utf-8 -*-
"""
train_full.py — آموزش کامل با split رسمی freeze‌شده (فاز 3)
- train: f4k_split_v1.csv[split=train]
- eval:  database=test (gallery) / query=split=query  → mAP واقعی cross-set
- گزینه resume برای ادامه آموزش
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T

from model.backbone import UWMambaRet
from losses import UWRetrievalLoss
from evaluate_retrieval import evaluate_retrieval

SPLIT_CSV = Path("../04_experiments/results/splits/f4k_split_v1.csv")
IMG_ROOT = Path("data/fish4k/fish4konwledge-master/fish_image")

TRAIN_TF = T.Compose([
    T.Resize(256, antialias=True),
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.1, 0.02),
    T.ToTensor(),
])
EVAL_TF = T.Compose([
    T.Resize(256, antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
])


class SplitDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, li = self.rows[i][0], int(self.rows[i][1])
        img = Image.open(IMG_ROOT / rel).convert("RGB")
        return self.tf(img), li


def load_split():
    rows = list(csv.reader(open(SPLIT_CSV)))[1:]
    train = [r for r in rows if r[3] == "train"]
    test = [r for r in rows if r[3] == "test"]
    query = [r for r in rows if r[3] == "query"]
    return train, test, query


@torch.no_grad()
def embed_rows(model, rows, device, batch=32):
    model.eval()
    ds = SplitDataset(rows, EVAL_TF)
    embs, ys = [], []
    for i in range(0, len(ds), batch):
        imgs = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        embs.append(model(imgs).cpu())
        ys.extend(int(ds.rows[j][1]) for j in range(i, min(i + batch, len(ds))))
    model.train()
    return torch.cat(embs), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--triplet-w", type=float, default=0.3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="f4k_full")
    ap.add_argument("--no-upam", action="store_true",
                    help="آبلاشن: حذف UPAM (پیش‌فرض: فعال)")
    ap.add_argument("--ssl-init", default=None,
                    help="مسیر checkpoint سلف‌سایپرویژن برای init backbone")
    args = ap.parse_args()

    device = args.device if torch.backends.mps.is_available() else "cpu"
    train_rows, test_rows, query_rows = load_split()
    print(f"split: train={len(train_rows)} test={len(test_rows)} query={len(query_rows)}")

    train_ds = SplitDataset(train_rows, TRAIN_TF)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

    model = UWMambaRet(embed_dim=args.embed, use_upam=not args.no_upam).to(device)
    if args.ssl_init:
        sd = torch.load(args.ssl_init, map_location="cpu")
        sd = sd.get("backbone", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"ssl-init: {args.ssl_init} | missing={len(missing)} unexpected={len(unexpected)}")
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | device={device}")

    crit = UWRetrievalLoss(triplet_weight=args.triplet_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_csv = Path(f"../04_experiments/results/train_{args.tag}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    f = open(out_csv, "a", newline="")
    wr = csv.writer(f)
    if new:
        wr.writerow(["epoch", "lr", "loss", "mAP", "mAP@10", "R@1", "R@5", "sec"])

    ck_dir = Path("checkpoints")
    ck_dir.mkdir(exist_ok=True)

    for ep in range(args.epochs):
        t0 = time.time()
        tot = 0.0
        nb = 0
        for img, y in dl:
            img = img.to(device)
            emb = model(img)
            loss, _ = crit(emb, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        q_emb, q_y = embed_rows(model, query_rows, device)
        g_emb, g_y = embed_rows(model, test_rows, device)
        r = evaluate_retrieval(q_emb, g_emb, q_y, g_y, ks=(1, 5, 10))
        row = [ep + 1, f"{sched.get_last_lr()[0]:.2e}", f"{tot/max(nb,1):.4f}",
               f"{r['mAP']:.4f}", f"{r['mAP@10']:.4f}", f"{r['R@1']:.4f}", f"{r['R@5']:.4f}",
               f"{time.time()-t0:.1f}"]
        wr.writerow(row)
        f.flush()
        print(f"epoch {row[0]}/{args.epochs}: loss={row[2]} mAP={row[3]} mAP@10={row[4]} R@1={row[5]} ({row[7]}s)")

        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            torch.save(model.state_dict(), ck_dir / f"{args.tag}_ep{ep+1}.pt")
    f.close()
    print(f"OK — CSV: {out_csv}")


if __name__ == "__main__":
    main()
