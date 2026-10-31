"""
train_baseline.py — آموزش baseline های استاندارد با همان پروتکل UW-MambaRet
=================================================================
هدف: جدول main مقاله — مقایسه منصفانه روی split فریز f4k_split_v1 (همان recipe
train_full.py: MS+triplet loss، batch/lr/epochs/eval یکسان؛ فقط backbone متفاوت)
مدل‌ها: resnet50 / mobilenetv3 / vit_b16 (torchvision، وزن ImageNet-V2)
اجرا:
    python train_baseline.py --model resnet50    --tag base_r50
    python train_baseline.py --model mobilenetv3 --tag base_mbv3
    python train_baseline.py --model vit_b16     --tag base_vit
"""
import argparse, csv, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.models as tvm
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from losses import UWRetrievalLoss
from evaluate_retrieval import evaluate_retrieval

SPLIT_CSV = Path("../04_experiments/results/splits/f4k_split_v1.csv")
IMG_ROOT = Path("data/fish4k/fish4konwledge-master/fish_image")

# همان transform های train_full.py (باید یکسان باشد برای مقایسه منصفانه)
TRAIN_TF = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
])
EVAL_TF = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
])


class SplitDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, lab = self.rows[i][0], int(self.rows[i][1])
        return self.tf(Image.open(IMG_ROOT / rel).convert("RGB")), lab


def build_backbone(name: str, embed_dim: int = 256) -> nn.Module:
    """backbone torchvision → GeM → Linear(embed) — خروجی L2-normalized در loss"""
    if name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        ch, trunk = net.fc.in_features, nn.Sequential(*list(net.children())[:-2])
    elif name == "mobilenetv3":
        net = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        ch, trunk = net.classifier[0].in_features, net.features
    elif name == "vit_b16":
        net = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        ch = net.heads[0].in_features
        trunk = nn.Sequential(net.conv_proj,
                              nn.Sequential(*net.encoder.layers),
                              nn.Sequential(nn.LazyLinear(ch)))  # placeholder — below handled
        # ViT خروجی CLS-token است؛ ساده‌تر: مدل را wrapper می‌کنیم
        return ViTWrapper(net, ch, embed_dim)
    else:
        raise ValueError(name)
    return nn.Sequential(trunk, GeM(), nn.Flatten(), nn.Linear(ch, embed_dim))


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), 1).pow(1.0 / self.p)


class ViTWrapper(nn.Module):
    """vit_b_16 → CLS token → Linear(embed)"""

    def __init__(self, vit, ch, embed_dim):
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(ch, embed_dim)

    def forward(self, x):
        feats = self.vit(x)  # (B, 1000) با heads پیش‌فرض؟ نه — پس از جایگزینی heads:
        return self.head(feats)


def _prep_vit(vit, embed_dim):
    vit.heads = nn.Identity()
    return ViTWrapper(vit, vit.hidden_dim, embed_dim)


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
    return F.normalize(torch.cat(embs), dim=1), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50", "mobilenetv3", "vit_b16"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--triplet-w", type=float, default=0.3)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = args.device if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    train_rows, test_rows, query_rows = load_split()
    print(f"split: train={len(train_rows)} test={len(test_rows)} query={len(query_rows)}")

    train_ds = SplitDataset(train_rows, TRAIN_TF)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2,
                    drop_last=True, persistent_workers=False)

    if args.model == "vit_b16":
        model = _prep_vit(tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1), args.embed)
    else:
        model = build_backbone(args.model, args.embed)
    model = model.to(device)
    print(f"[{args.tag}] {args.model} — {sum(p.numel() for p in model.parameters())/1e6:.2f}M — {device}")

    crit = UWRetrievalLoss(triplet_weight=args.triplet_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_csv = Path(f"../04_experiments/results/train_{args.tag}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["epoch", "lr", "loss", "mAP", "mAP@10", "R@1", "R@5", "sec"])

    ck = Path("checkpoints"); ck.mkdir(exist_ok=True)
    for ep in range(args.epochs):
        t0 = time.time(); tot = nb = 0
        for img, y in dl:
            img, y = img.to(device), y.to(device)
            emb = model(img)
            loss, _ = crit(emb, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        q_emb, q_y = embed_rows(model, query_rows, device)
        g_emb, g_y = embed_rows(model, test_rows, device)
        r = evaluate_retrieval(q_emb, g_emb, q_y, g_y, ks=(1, 5, 10))
        row = [ep + 1, f"{sched.get_last_lr()[0]:.2e}", f"{tot/max(nb,1):.4f}",
               f"{r['mAP']:.4f}", f"{r['mAP@10']:.4f}", f"{r['R@1']:.4f}", f"{r['R@5']:.4f}",
               f"{time.time()-t0:.1f}"]
        wr.writerow(row); f.flush()
        print(f"[{args.tag}] ep{row[0]}/{args.epochs}: loss={row[2]} mAP={row[3]} R@1={row[5]} ({row[7]}s)", flush=True)
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            torch.save(model.state_dict(), ck / f"{args.tag}_ep{ep+1}.pt")
    f.close()
    print(f"OK — {out_csv}")


if __name__ == "__main__":
    main()
