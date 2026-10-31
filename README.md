# UW-MambaRet

**UW-MambaRet: A Lightweight State-Space Framework for Robust Underwater Image Retrieval**

Official repository for the paper (under review, *Expert Systems with Applications*).

UW-MambaRet is an underwater image retrieval framework built on a
visual state-space (Mamba) backbone — the state-space backbone reaches retrieval,
complementing existing underwater Mamba models for enhancement and detection. It couples:

- a **1.26M-parameter grouped two-direction SS2D encoder** (dims 48/96/192/384, depths 2/2/4/2, state dim N=8),
- a **portable chunked associative scan** — pure PyTorch, no CUDA-specific kernel; runs unmodified on CPU / Apple MPS / CUDA with exact sequential-recurrence equivalence (verified to <1e-6),
- a **0.64K-parameter physics-aware gate (UPAM)** inserted after the stem (reported transparently, including its neutrality at corpus scale),
- a **degradation-robustness benchmark**: 5 physically modelled distortion families x 3 seeded severities, query-distorted / gallery-clean protocol, Retention Rate and AUC-of-severity metrics,
- a **degradation curriculum** that lifts mean AUC-S from 76.4 to 95.4 at a transparent 3.9-pp clean-mAP cost.

Headline result on the frozen Fish4Knowledge split (3,726 images / 21 classes, MD5-pinned):
**mAP 0.664 / R@1 0.802 at 33.9 img/s (batch 32) on an Apple M4 Pro (MPS).**

## Repository layout

```
model/          backbone.py (UWMambaRet), ss2d.py, s6_block.py (chunked scan),
                upam.py (physics gate), embed_head.py (GeM + FC + BN + L2)
data/           degrade.py — the seeded degradation generator (ATT/CC/BLR/NS/LOW, L1-L3)
losses.py       multi-similarity (alpha=2, beta=40, mining 0.5) + batch-hard triplet (w=0.3, margin 0.2)
make_splits.py  deterministic F4K split (dHash dedup, md5 b833dc454cdd)
train_full.py   30-epoch base training (batch 32, AdamW wd 1e-4, LR 3e-4, cosine)
train_curriculum.py  same 30-epoch recipe; epochs 1-5 clean, then p 0.3->0.8, severity cap L1->L3
train_baseline.py    ResNet50 / MobileNetV3-L / ViT-B/16 under the identical protocol
train_mambahash.py   hashing-head variant (64-bit codes, DHN-style surrogate) — objective ablation
evaluate_retrieval.py  mAP / mAP@10 / R@1/5/20, cosine ranking on L2-normalised embeddings
run_robustness.py      the 5x3 benchmark (add --model, --ckpt, --no-upam)
measure_submission.py  params / GFLOPs / MPS throughput for all models
splits/f4k_split_v1.csv  the frozen split (release with every experiment)
results/           per-cell robustness CSVs for base / curriculum / no-UPAM runs
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins torch 2.8 (works with 2.7+), torchvision, Pillow and numpy.
No CUDA-specific packages are required.

## Data

1. Download Fish4Knowledge (fish species images; boom2014fish4k) and note its root.
2. Build the frozen split (adjust the data-root flag to your layout):

```bash
python make_splits.py --data-root <F4K_ROOT> --out splits/f4k_split_v1.csv
```

The released `splits/f4k_split_v1.csv` is the exact split used in the paper
(md5 `b833dc454cdd...`); skip this step and use it directly to reproduce numbers.

## Reproducing the paper

```bash
# base model (30 epochs, ~2 h on M4 Pro / MPS)
python train_full.py --embed-dim 256 --epochs 30

# robustness benchmark for any checkpoint
python run_robustness.py --ckpt checkpoints/f4k_full_ep29.pt

# curriculum run (same 30-epoch budget)
python train_curriculum.py

# baselines under the identical protocol
python train_baseline.py --arch resnet50|mobilenetv3|vit_b16

# UPAM ablation
python train_full.py --no-upam

# budget/throughput table
python measure_submission.py
```

## Results summary (F4K frozen split)

| Model | Params | mAP | R@1 | img/s |
|---|---|---|---|---|
| ResNet50 | 24.0M | 0.454 | 0.719 | 337 |
| MobileNetV3-L | 5.4M | 0.589 | 0.807 | 1214 |
| ViT-B/16 | 86.6M | **0.980** | **0.987** | 105 |
| MambaHash (same encoder, hashing head) | 1.36M | 0.105 (Hamming) | 0.105 | -- |
| **UW-MambaRet (base)** | **1.26M** | 0.664 | 0.802 | 33.9 |
| + curriculum | 1.26M | 0.624 | 0.780 | 33.9 |

Curriculum robustness (mean retention over severities, AUC-S): base 76.4 -> **95.4**
(ATT +35.5, CC +30.8, NS +21.7). UPAM ablation: performance-neutral at this scale
(clean 0.672 vs 0.664; benefit confined to NS-L3).

## License

MIT (see LICENSE). The Fish4Knowledge dataset follows its own terms — download it
separately; this repository ships only the frozen split index.
