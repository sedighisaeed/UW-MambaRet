# -*- coding: utf-8 -*-
"""
run_robustness.py — بنچمارک مقاومت 5×3 (ادعای C4) — روی checkpoint آموخته‌شده
پروتکل: query های سالم از split رسمی → 15 نسخه تخریب‌شده (5 نوع × 3 شدت، قطعی)
embedding تخریب‌شده → جستجو در gallery سالم (test split) → افت mAP/R@K نسبت به سالم
خروجی: robustness_f4k.csv (ماتریس 5×3 + baseline سالم)
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from model.backbone import UWMambaRet
from data.degrade import DegradedQueryGenerator, TYPES, LEVELS
from evaluate_retrieval import evaluate_retrieval

SPLIT_CSV = Path("../04_experiments/results/splits/f4k_split_v1.csv")
IMG_ROOT = Path("data/fish4k/fish4konwledge-master/fish_image")
EVAL_TF = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToTensor()])


def load_split():
    rows = list(csv.reader(open(SPLIT_CSV)))[1:]
    test = [r for r in rows if r[3] == "test"]
    query = [r for r in rows if r[3] == "query"]
    return test, query


def load_img(rel):
    return EVAL_TF(Image.open(IMG_ROOT / rel).convert("RGB"))


@torch.no_grad()
def embed(model, imgs, device, batch=32):
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        out.append(model(torch.stack(imgs[i:i + batch]).to(device)).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/f4k_full_ep30.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="../04_experiments/results/robustness_f4k.csv")
    ap.add_argument("--no-upam", action="store_true",
                    help="build the model without the UPAM module")
    args = ap.parse_args()

    device = args.device if torch.backends.mps.is_available() else "cpu"
    test_rows, query_rows = load_split()

    model = UWMambaRet(embed_dim=256, use_upam=not args.no_upam).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    print(f"ckpt: {args.ckpt} loaded | device={device}")

    g = DegradedQueryGenerator(seed=2026)

    # gallery سالم (همیشه ثابت)
    g_imgs = [load_img(r[0]) for r in test_rows]
    g_emb = embed(model, g_imgs, device)
    g_y = torch.tensor([int(r[1]) for r in test_rows])

    # baseline سالم
    q_imgs_clean = [load_img(r[0]) for r in query_rows]
    q_y = torch.tensor([int(r[1]) for r in query_rows])
    q_emb = embed(model, q_imgs_clean, device)
    base = evaluate_retrieval(q_emb, g_emb, q_y, g_y, ks=(1, 5, 10))

    rows = [{"dtype": "CLEAN", "level": 0, "mAP": round(base["mAP"], 4),
             "mAP@10": round(base["mAP@10"], 4), "R@1": round(base["R@1"], 4),
             "R@5": round(base["R@5"], 4)}]
    print("baseline:", rows[0])

    # 5×3
    for dtype in TYPES:
        for level in LEVELS:
            deg = [g(load_img(r[0]), dtype, level, idx=i) for i, r in enumerate(query_rows)]
            q_e = embed(model, deg, device)
            r = evaluate_retrieval(q_e, g_emb, q_y, g_y, ks=(1, 5, 10))
            rows.append({"dtype": dtype, "level": level, "mAP": round(r["mAP"], 4),
                         "mAP@10": round(r["mAP@10"], 4), "R@1": round(r["R@1"], 4),
                         "R@5": round(r["R@5"], 4)})
            print(rows[-1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["dtype", "level", "mAP", "mAP@10", "R@1", "R@5"])
        wr.writeheader()
        wr.writerows(rows)

    # خلاصه افت
    bm = base["mAP"]
    print("\n=== افت نسبی mAP (نسبت به سالم) ===")
    for r in rows[1:]:
        drop = (1 - r["mAP"] / bm) * 100 if bm > 0 else 0
        print(f"{r['dtype']}-L{r['level']}: {drop:.1f}%")
    print(f"OK — ذخیره: {out}")


if __name__ == "__main__":
    main()
