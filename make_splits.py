# -*- coding: utf-8 -*-
"""
make_splits.py — ساخت split رسمی Fish4Knowledge (طبق پروتکل 04_experiments/protocol.md)
① dedup با dHash (64-bit perceptual hash — بدون کتابخانه خارجی)
② split stratified: 70% database / 20% gallery-test / 10% query (کف‌ها رعایت شود)
③ freeze به CSV (تکرارپذیری — هرگز دوباره تغییر نمی‌کند؛ نکته anti-leak: dedup قبل از split)
"""

import csv
import hashlib
import random
from pathlib import Path
from PIL import Image

ROOT = Path("data/fish4k/fish4konwledge-master/fish_image")
OUT = Path("../04_experiments/results/splits")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_PER_CLS = 20
MAX_PER_CLS = 400          # سقف برای توازن (کلاس 12هزارتایی به 400 محدود — حذف توازن شدید)
HAMMING_DUP = 6            # آستانه dHash (از 64 بیت)
SEED = 2026


def dhash(img: Image.Image) -> int:
    """Difference hash 64-bit — سریع و مقاوم به تغییرات خفیف"""
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main():
    rng = random.Random(SEED)
    rows = []           # (relpath, label, split, hash_hex)
    seen = []           # hash های قبلی (جهانی — anti-leak بین همه split ها)
    n_dup = 0

    classes = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    for ci, cname in enumerate(classes):
        files = sorted(p for p in (ROOT / cname).iterdir() if p.suffix.lower() in EXTS)
        rng.shuffle(files)
        files = files[:MAX_PER_CLS]
        kept_hashes = []
        kept_files = []
        for p in files:
            try:
                img = Image.open(p)
                h = dhash(img)
            except Exception:
                continue
            if any(hamming(h, prev) <= HAMMING_DUP for prev in kept_hashes[-200:]):
                n_dup += 1
                continue
            kept_hashes.append(h)
            kept_files.append(p)
        if len(kept_files) < MIN_PER_CLS:
            continue
        # split stratified 70/20/10 (کف query=2)
        n = len(kept_files)
        n_q = max(2, round(n * 0.10))
        n_t = max(2, round(n * 0.20))
        queries = kept_files[:n_q]
        tests = kept_files[n_q:n_q + n_t]
        trains = kept_files[n_q + n_t:]
        for split, lst in (("query", queries), ("test", tests), ("train", trains)):
            for p in lst:
                rows.append([str(p.relative_to(ROOT)), ci, cname, split])

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "f4k_split_v1.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["relpath", "label_idx", "label_name", "split"])
        wr.writerows(rows)

    # خلاصه
    from collections import Counter
    cnt = Counter(r[3] for r in rows)
    n_cls = len({r[1] for r in rows})
    sig = hashlib.md5(csv_path.read_bytes()).hexdigest()[:12]
    print(f"OK — {len(rows)} تصویر | {cnt} | {n_cls} کلاس | dedup={n_dup} | md5={sig}")
    print(f"freeze: {csv_path}")


if __name__ == "__main__":
    main()
