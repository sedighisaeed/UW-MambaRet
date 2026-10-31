# -*- coding: utf-8 -*-
"""
train_curriculum.py — آموزش با Curriculum Degradation (ادعای C4: مقاومت‌سازی)
Ablation جفت train_full.py: هر دو از scratch، همین recipe (AdamW lr=3e-4، cosine، batch=32،
30 epoch، MS+Triplet w=0.3) — تنها تفاوت: اعوجاج سنتزی تدریجی روی train.

برنامه curriculum (پس از ep5 پایه پایدار، 0-based):
  - احتمال تخریب p: 0.3 → 0.8 خطی
  - سقف شدت max_lv: L1 (ابتدا) → L1..L2 (میانه) → L1..L3 (انتها)
  - نوع تخریب: یکنواخت روی {ATT, CC, BLR, NS, LOW}

⚠️ نکات پیاده‌سازی (باگ‌های نسخه 1 که اینجا رفع شدند):
  1) persistent_workers=True مانع دیدن set_epoch توسط worker ها می‌شد → curriculum
     هرگز فعال نمی‌شد. راه‌حل: persistent_workers=False (بازتولید worker هر epoch).
  2) torch.Generator مشترک بین worker ها → تصمیم‌های تخریب همبسته. راه‌حل:
     ماژول random (DataLoader خودش per-worker seed می‌کند).
  3) lr پیش‌فرض 1e-4 با baseline (3e-4) ناسازگار بود → confound در ablation.
  ستون deg_frac در CSV ثابت می‌کند تخریب در عمل فعال بوده (قابل حسابرسی).
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T

from model.backbone import UWMambaRet
from losses import UWRetrievalLoss
from evaluate_retrieval import evaluate_retrieval
from data.degrade import DegradedQueryGenerator, TYPES

SPLIT_CSV = Path("../04_experiments/results/splits/f4k_split_v1.csv")
IMG_ROOT = Path("data/fish4k/fish4konwledge-master/fish_image")

# دقیقاً برابر train_full.py — انصاف ablation
TRAIN_TF = T.Compose([
    T.Resize(256, antialias=True),
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.1, 0.02),
    T.ToTensor(),
])
EVAL_TF = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToTensor()])

CURR_START_EP = 5   # 0-based: epoch های 0..4 تمیز؛ از ep5 برنامه تخریب شروع می‌شود


class SplitDataset(Dataset):
    """خروجی: (img, label, deg_flag) — deg_flag برای حسابرسی فعال‌بودن curriculum"""

    def __init__(self, rows, tf, curriculum: bool = False, max_epochs: int = 30, seed: int = 2026):
        self.rows = rows
        self.tf = tf
        self.curriculum = curriculum
        self.max_epochs = max_epochs
        self.epoch = 0
        self.g = DegradedQueryGenerator(seed=seed)

    def set_epoch(self, ep):
        self.epoch = ep

    def schedule(self, ep):
        """(p_deg, max_lv) در epoch داده‌شده — قبل از شروع: (0, 0)"""
        if ep < CURR_START_EP:
            return 0.0, 0
        # مخرج (max_epochs-1-START): آخرین epoch دقیقاً p=0.8 و L3 (مطابق spec)
        frac = (ep - CURR_START_EP) / (self.max_epochs - 1 - CURR_START_EP)
        p_deg = 0.3 + 0.5 * frac                       # 0.3 → 0.8
        max_lv = 1 + min(2, int(frac * 3))             # همیشه ∈ {1,2,3} (گارد crash)
        return p_deg, max_lv

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, li = self.rows[i][0], int(self.rows[i][1])
        img_t = self.tf(Image.open(IMG_ROOT / rel).convert("RGB"))
        deg = 0
        p_deg, max_lv = self.schedule(self.epoch)
        if self.curriculum and p_deg > 0 and random.random() < p_deg:
            level = random.randint(1, max_lv)
            dtype = TYPES[random.randrange(len(TYPES))]
            img_t = self.g(img_t, dtype, level, idx=i)
            deg = 1
        return img_t, li, deg


def load_split():
    rows = list(csv.reader(open(SPLIT_CSV)))[1:]
    return ([r for r in rows if r[3] == "train"],
            [r for r in rows if r[3] == "test"],
            [r for r in rows if r[3] == "query"])


@torch.no_grad()
def embed_rows(model, rows, device, batch=32):
    model.eval()
    ds = SplitDataset(rows, EVAL_TF)          # ارزیابی همیشه تمیز
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
    ap.add_argument("--lr", type=float, default=3e-4, help="برابری با train_full (confound نباید)")
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--triplet-w", type=float, default=0.3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="f4k_curr")
    args = ap.parse_args()

    device = args.device if torch.backends.mps.is_available() else "cpu"
    train_rows, test_rows, query_rows = load_split()
    print(f"split: train={len(train_rows)} test={len(test_rows)} query={len(query_rows)}")

    train_ds = SplitDataset(train_rows, TRAIN_TF, curriculum=True, max_epochs=args.epochs)
    # persistent_workers=False — ضروری: worker هر epoch با epoch جدید بازتولید شود
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2,
                    drop_last=True, persistent_workers=False)

    model = UWMambaRet(embed_dim=args.embed, use_upam=True).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | device={device} | from-scratch (ablation parity)")

    crit = UWRetrievalLoss(triplet_weight=args.triplet_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_csv = Path(f"../04_experiments/results/train_{args.tag}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, "a", newline="")
    wr = csv.writer(f)
    if f.tell() == 0:
        wr.writerow(["epoch", "lr", "loss", "mAP", "mAP@10", "R@1", "R@5", "deg_frac", "sec"])
    ck_dir = Path("checkpoints")
    ck_dir.mkdir(exist_ok=True)

    for ep in range(args.epochs):
        train_ds.set_epoch(ep)
        p_deg, max_lv = train_ds.schedule(ep)
        t0 = time.time()
        tot, nb, n_deg, n_seen = 0.0, 0, 0, 0
        for img, y, deg in dl:
            img = img.to(device)
            emb = model(img)
            loss, _ = crit(emb, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            n_deg += int(deg.sum().item())
            n_seen += len(deg)
        sched.step()
        deg_frac = n_deg / max(n_seen, 1)
        q_e, q_y = embed_rows(model, query_rows, device)
        g_e, g_y = embed_rows(model, test_rows, device)
        r = evaluate_retrieval(q_e, g_e, q_y, g_y, ks=(1, 5, 10))
        row = [ep + 1, f"{sched.get_last_lr()[0]:.2e}", f"{tot/max(nb,1):.4f}",
               f"{r['mAP']:.4f}", f"{r['mAP@10']:.4f}", f"{r['R@1']:.4f}", f"{r['R@5']:.4f}",
               f"{deg_frac:.3f}", f"{time.time()-t0:.1f}"]
        wr.writerow(row)
        f.flush()
        print(f"[{args.tag}] ep {row[0]}/{args.epochs}: loss={row[2]} mAP={row[3]} R@1={row[5]} "
              f"deg={deg_frac:.2f}(p={p_deg:.2f},L≤{max_lv}) ({row[8]}s)")
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            torch.save(model.state_dict(), ck_dir / f"{args.tag}_ep{ep+1}.pt")
    f.close()
    print(f"OK — CSV: {out_csv}")


if __name__ == "__main__":
    main()
