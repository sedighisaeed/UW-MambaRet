# -*- coding: utf-8 -*-
"""
measure_submission.py — اعداد نهایی برای پاسخ به ریویو:
1) پارامتر دقیق هر مدل (ours/r50/mbv3/vit)
2) throughput (img/s) هر 4 مدل روی همان دستگاه (MPS) — parity جدول 3
3) FLOPs تقریبی ours با torch.profiler (با فلاپس‌شمار) روی CPU
خروجی: ../04_experiments/results/submission_metrics.csv + چاپ
"""
import sys, time, csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).parent))
from model.backbone import UWMambaRet

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def build_ours():
    return UWMambaRet(embed_dim=256, use_upam=True)


def build_r50():
    net = tvm.resnet50(weights=None)
    trunk = nn.Sequential(*list(net.children())[:-2])
    return nn.Sequential(trunk, GeM(), nn.Flatten(), nn.Linear(2048, 256))


def build_mbv3():
    net = tvm.mobilenet_v3_large(weights=None)
    return nn.Sequential(net.features, GeM(), nn.Flatten(),
                         nn.Linear(net.classifier[0].in_features, 256))


class ViTW(nn.Module):
    def __init__(self):
        super().__init__()
        net = tvm.vit_b_16(weights=None)
        net.heads = nn.Identity()
        self.vit = net
        self.head = nn.Linear(net.hidden_dim, 256)

    def forward(self, x):
        return self.head(self.vit(x))


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), 1).pow(1.0 / self.p)


def count(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def throughput(model, dev, reps=12, n=32):
    model = model.to(dev).eval()
    x = torch.randn(n, 3, 224, 224, device=dev)
    for _ in range(4):
        model(x)
    if dev == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(x)
    if dev == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    return n * reps / dt


def flops_ours():
    """FLOPs را با profiler روی CPU — یک forward با batch=1"""
    from torch.profiler import profile, ProfilerActivity
    m = build_ours().eval()
    x = torch.randn(1, 3, 224, 224)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        m(x)
    tot = sum(e.flops for e in prof.key_averages())
    return tot


def main():
    rows = []
    builders = {"UW-MambaRet": build_ours, "ResNet50": build_r50,
                "MobileNetV3-L": build_mbv3, "ViT-B/16": ViTW}
    for name, b in builders.items():
        m = b()
        n = count(m)
        if name == "UW-MambaRet":
            fl = flops_ours()
        else:
            fl = None
        th = throughput(m, DEV)
        rows.append({"model": name, "params_M": round(n / 1e6, 3),
                     "GFLOPs": round(fl / 1e9, 3) if fl else "",
                     "imgps_MPS": round(th, 1)})
        print(rows[-1], flush=True)
        del m
    out = Path("../04_experiments/results/submission_metrics.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["model", "params_M", "GFLOPs", "imgps_MPS"])
        wr.writeheader(); wr.writerows(rows)
    print("OK —", out)


if __name__ == "__main__":
    main()
