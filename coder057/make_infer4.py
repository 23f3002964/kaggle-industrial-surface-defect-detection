#!/usr/bin/env python3
"""Generate infer-v4: test probs ONLY for folds 4 and 5 (models from dl-train-v6 output).

Runs right after training. Saves test_probs_fold{4,5}.npy so the CPU post kernel can
combine them with the existing fold1/2/3 + baseline probs at a matched mask budget.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "infer-v4"
OUT.mkdir(exist_ok=True)

cells = []


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s})


code(r'''
import time, gc
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F

t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| GPUs:", torch.cuda.device_count(), flush=True)
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
TEST_DIR, SAMPLE = ROOT / "test", ROOT / "sample_submission.csv"
BATCH = 64
''')

code(r'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.down(x)


class AttentionGate(nn.Module):
    def __init__(self, gating_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gating_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.W_x = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 1))
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attention(x, skip)], dim=1))


class AttentionDefectUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(DoubleConv(in_channels, 32), DownBlock(32, 64),
                                     DownBlock(64, 128), DownBlock(128, 256))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.decoder = nn.ModuleList([AttentionUpBlock(512, 256, 256), AttentionUpBlock(256, 128, 128),
                                      AttentionUpBlock(128, 64, 64), AttentionUpBlock(64, 32, 32)])
        self.outc = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x); skips.append(x)
        x = self.bottleneck(x)
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = layer(x, skip)
        return self.outc(x)
''')

code(r'''
sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800

test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {time.time()-t_start:.0f}s", flush=True)

models = {int(p.stem[-1]): p for p in sorted(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD*.pth"))}
print("models found:", {k: str(v) for k, v in models.items()}, flush=True)
assert 4 in models, f"fold 4 model missing; found {sorted(models)}"
todo = [f for f in (4, 5) if f in models]
print("will run folds:", todo, flush=True)


@torch.inference_mode()
def probs_tta(model, u8, tag=""):
    out = np.empty((len(u8), 4, H, W), dtype=np.float16)
    t0 = time.time()
    for s in range(0, len(u8), BATCH):
        x = torch.from_numpy(u8[s:s + BATCH]).unsqueeze(1).float().div_(255).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            logits = (logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])) / 2
        out[s:s + BATCH] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float16)
        done = min(s + BATCH, len(u8))
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {tag} {done}/{len(u8)} | {el:.0f}s | ETA {el/done*(len(u8)-done):.0f}s", flush=True)
    return out


for fold in todo:
    model = torch.load(models[fold], map_location=device, weights_only=False).to(device).eval()
    tp = probs_tta(model, test_u8, tag=f"test{fold}")
    np.save(f"/kaggle/working/test_probs_fold{fold}.npy", tp)
    print(f"fold {fold} probs saved | mean {tp.mean():.4f} | {time.time()-t_start:.0f}s", flush=True)
    del model, tp
    gc.collect(); torch.cuda.empty_cache()

scores = Path("/kaggle/input")
for p in scores.rglob("fold_scores.json"):
    print("training val scores:", p.read_text(), flush=True)
print(f"DONE | total {time.time()-t_start:.0f}s", flush=True)
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "infer-v4.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "coder057/dl-infer-v4", "title": "DL-infer-v4", "code_file": "infer-v4.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [], "kernel_sources": ["coder057/dl-train-v6"],
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("written:", OUT / "infer-v4.ipynb")
