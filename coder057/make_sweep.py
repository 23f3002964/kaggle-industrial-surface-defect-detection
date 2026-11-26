#!/usr/bin/env python3
"""Build sweep-v1 kernel: copy coder057 notebook, patch for CPU postproc-threshold sweep + TTA."""
import json, copy
from pathlib import Path

SRC = Path("/home/adi/CODE/Kaggle/industrial-surface-defect-detection/coder057/dl-23f1000351-notebook-t32026.ipynb")
DST_DIR = Path("/home/adi/CODE/Kaggle/industrial-surface-defect-detection/coder057/sweep-v1")
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC.read_text())

def set_cell_src(i, src):
    nb["cells"][i]["source"] = src.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None

# --- cell 2: drop wandb import (training skipped), add nullcontext ---
src = "".join(nb["cells"][2]["source"])
src = src.replace("# Experiment Tracking, Kaggle Utils\nimport wandb\nimport kagglehub\n",
                  "# Kaggle utils\nimport kagglehub\nfrom contextlib import nullcontext\n")
set_cell_src(2, src)

# --- cell 4: real image size ---
src = "".join(nb["cells"][4]["source"]).replace('"image_height": 180, "image_width": 800', '"image_height": 128, "image_width": 800')
set_cell_src(4, src)

# --- cell 6: no wandb login needed in inference mode ---
set_cell_src(6, "# wandb login skipped: inference/sweep mode (no training)\n")

# --- cell 24: small batches for CPU ---
src = "".join(nb["cells"][24]["source"]).replace("BatchSize = 64", "BatchSize = 8").replace("num_workers = 4", "num_workers = 2")
set_cell_src(24, src)

# --- cell 54: parametrized inference + postproc ---
set_cell_src(54, '''# Inference + postproc (parametrized, CPU-friendly)

@torch.inference_mode()
def model_probs(model, images, device, use_tta=False):
    model.eval()
    images = images.to(device, non_blocking=True)
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            if use_tta:
                logits = (logits + torch.flip(model(torch.flip(images, dims=[-1])), dims=[-1])) / 2
    else:
        logits = model(images)
        if use_tta:
            logits = (logits + torch.flip(model(torch.flip(images, dims=[-1])), dims=[-1])) / 2
    return torch.sigmoid(logits).cpu().numpy().astype(np.float16)


def cc_filter_mask(mask, min_component):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            out[labels == i] = 1
    return out


def postprocess(probs, thresholds, min_component, min_pixels):
    # probs (B,C,H,W) float16/float32 -> filtered uint8 masks
    B, C, H, W = probs.shape
    out = np.zeros((B, C, H, W), dtype=np.uint8)
    for c in range(C):
        base = (probs[:, c].astype(np.float32) >= thresholds[c]).astype(np.uint8)
        if min_component[c] > 0 and base.any():
            for b in range(B):
                base[b] = cc_filter_mask(base[b], min_component[c])
        for b in range(B):
            if base[b].sum() >= min_pixels[c]:
                out[b, c] = base[b]
    return out


@torch.inference_mode()
def predict_test(model, test_loader, device, thresholds, min_component, min_pixels, use_tta=False):
    model.eval()
    rle_dict = {}
    pbar = tqdm(test_loader, desc="Generating [Test Preds]", dynamic_ncols=True)
    for images, image_ids in pbar:
        probs = model_probs(model, images, device, use_tta=use_tta)
        preds = postprocess(probs, thresholds, min_component, min_pixels)
        for b_idx, img_id in enumerate(image_ids):
            for c_idx, defect_type in enumerate(DEFECT_TYPES):
                mask = preds[b_idx, c_idx]
                rle_dict[(str(img_id), str(defect_type))] = rle_encode(mask) if mask.sum() > 0 else " "
        pbar.set_postfix({"predictions": len(rle_dict)})
    return rle_dict
''')

# --- cell 57: full sweep pipeline ---
set_cell_src(57, '''# ===== Threshold / postproc sweep on validation + submission =====
import time
t_start = time.time()
print("STAGE 1/5 collect validation probs (plain + TTA)", flush=True)

val_probs = {"plain": [], "tta": []}
val_targets = []
for images, targets in tqdm(valid_loader, desc="[Val probs]", dynamic_ncols=True):
    val_targets.append(targets.numpy().astype(np.uint8))
    val_probs["plain"].append(model_probs(model, images, device, use_tta=False))
    val_probs["tta"].append(model_probs(model, images, device, use_tta=True))
val_probs = {k: np.concatenate(v) for k, v in val_probs.items()}
val_targets = np.concatenate(val_targets)
print(f"val probs: {val_probs['plain'].shape}, targets: {val_targets.shape} | {time.time()-t_start:.0f}s elapsed", flush=True)

THRESH_GRID = np.round(np.arange(0.05, 0.801, 0.025), 3)

def competition_score(pred_u8, tgt_u8):
    N, C, H, W = pred_u8.shape
    p = pred_u8.reshape(N, C, H * W).astype(np.float64)
    t = (tgt_u8.reshape(N, C, H * W) > 0).astype(np.float64)
    pred_px, tgt_px = p.sum(-1), t.sum(-1)
    inter = (p * t).sum(-1)
    eps = 1e-7
    pos, emp = tgt_px > 0, tgt_px == 0
    dice = (2 * inter + eps) / (pred_px + tgt_px + eps)
    fg = dice[pos].mean() if pos.any() else 0.0
    fps = 1.0 - (pred_px[emp] / (H * W)).mean() if emp.any() else 1.0
    ea = (pred_px[emp] == 0).mean() if emp.any() else 1.0
    return 0.7 * fg + 0.2 * fps + 0.1 * ea

def best_thresholds(probs, tgt):
    N, C, H, W = probs.shape
    P = probs.reshape(N, C, H * W).astype(np.float32)
    T = (tgt.reshape(N, C, H * W) > 0)
    tgt_px = T.sum(-1).astype(np.float64)
    pos, emp = tgt_px > 0, tgt_px == 0
    best = []
    for c in range(C):
        pc, tc = P[:, c], T[:, c]
        tgt_px_c = tgt_px[:, c]
        pos_c, emp_c = pos[:, c], emp[:, c]
        scores = []
        for thr in THRESH_GRID:
            pred = pc >= thr
            pred_px = pred.sum(-1).astype(np.float64)
            inter = (pred & tc).sum(-1).astype(np.float64)
            dice = (2 * inter + 1e-7) / (pred_px + tgt_px_c + 1e-7)
            fg = dice[pos_c].mean() if pos_c.any() else 0.0
            fps = 1.0 - (pred_px[emp_c] / (H * W)).mean() if emp_c.any() else 1.0
            ea = (pred_px[emp_c] == 0).mean() if emp_c.any() else 1.0
            scores.append(0.7 * fg + 0.2 * fps + 0.1 * ea)
        i = int(np.argmax(scores))
        best.append({"threshold": float(THRESH_GRID[i]), "sub_score": float(scores[i])})
        print(f"  class {DEFECT_TYPES[c]}: best thr={THRESH_GRID[i]:.3f} (score {scores[i]:.4f})", flush=True)
    return best

print("STAGE 2/5 per-class threshold sweep", flush=True)
thresholds = {k: best_thresholds(v, val_targets) for k, v in val_probs.items()}
print(f"{time.time()-t_start:.0f}s elapsed", flush=True)

MC_GRID = [0, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256]
MT_GRID = [0, 64, 128, 192, 256, 320, 384, 512]
rng = np.random.default_rng(CONFIG["seed"])
sub_idx = rng.choice(val_targets.shape[0], size=min(400, val_targets.shape[0]), replace=False)

def sweep_cc(probs, tgt, thr_list):
    results = []
    for c in range(probs.shape[1]):
        thr = thr_list[c]["threshold"]
        P = (probs[sub_idx, c].astype(np.float32) >= thr).astype(np.uint8)
        T = (tgt[sub_idx, c] > 0)
        n = len(sub_idx)
        tgt_px = T.reshape(n, -1).sum(-1).astype(np.float64)
        pos, emp = tgt_px > 0, tgt_px == 0
        T_u8 = T.astype(np.uint8)
        base_sums = P.reshape(n, -1).sum(-1).astype(np.float64)
        base_inter = (P * T_u8).reshape(n, -1).sum(-1).astype(np.float64)
        best = (0, 0, -1.0)
        for mc in MC_GRID:
            if mc == 0:
                sums = base_sums
                itr = base_inter
            else:
                masks = np.stack([cc_filter_mask(P[i], mc) for i in range(n)])
                sums = masks.reshape(n, -1).sum(-1).astype(np.float64)
                itr = (masks * T_u8).reshape(n, -1).sum(-1).astype(np.float64)
            for mt in MT_GRID:
                px = np.where(sums >= mt, sums, 0)
                itr2 = np.where(sums >= mt, itr, 0)
                dice = (2 * itr2 + 1e-7) / (px + tgt_px + 1e-7)
                fg = dice[pos].mean() if pos.any() else 0.0
                fps = 1.0 - (px[emp] / P.shape[-1] / P.shape[-2]).mean() if emp.any() else 1.0
                ea = (px[emp] == 0).mean() if emp.any() else 1.0
                s = 0.7 * fg + 0.2 * fps + 0.1 * ea
                if s > best[2]:
                    best = (mc, mt, s)
        results.append({"min_component": best[0], "min_pixels": best[1], "sub_score": best[2]})
        print(f"  class {DEFECT_TYPES[c]}: mc={best[0]} mt={best[1]} (sub score {best[2]:.4f})", flush=True)
    return results

print("STAGE 3/5 connected-component / min-pixel sweep (400-img subsample)", flush=True)
cc_cfg = {k: sweep_cc(v, val_targets, thresholds[k]) for k, v in val_probs.items()}
print(f"{time.time()-t_start:.0f}s elapsed", flush=True)

print("STAGE 4/5 full-validation evaluation, variant selection", flush=True)
variant_cfg = {}
for k in val_probs:
    thr = [d["threshold"] for d in thresholds[k]]
    mc = [d["min_component"] for d in cc_cfg[k]]
    mt = [d["min_pixels"] for d in cc_cfg[k]]
    pred = np.zeros_like(val_probs[k], dtype=np.uint8)
    for c in range(pred.shape[1]):
        base = (val_probs[k][:, c].astype(np.float32) >= thr[c]).astype(np.uint8)
        if mc[c] > 0:
            for i in range(base.shape[0]):
                base[i] = cc_filter_mask(base[i], mc[c])
        for i in range(base.shape[0]):
            if base[i].sum() >= mt[c]:
                pred[i, c] = base[i]
    sc = competition_score(pred, val_targets)
    variant_cfg[k] = {"thresholds": thr, "min_component": mc, "min_pixels": mt, "val_score": sc}
    print(f"  {k}: val score {sc:.4f} | thr={thr} mc={mc} mt={mt}", flush=True)

baseline_pred = postprocess(val_probs["plain"].astype(np.float32),
                            CONFIG["class_thresholds"],
                            [CONFIG["min_component_pixels"]] * 4,
                            [CONFIG["min_defect_pixels"]] * 4)
print(f"  baseline (old CONFIG, plain): val score {competition_score(baseline_pred, val_targets):.4f}", flush=True)

best_variant = max(variant_cfg, key=lambda k: variant_cfg[k]["val_score"])
cfg = variant_cfg[best_variant]
use_tta = best_variant == "tta"
print(f"BEST variant: {best_variant} | val score {cfg['val_score']:.4f} | {time.time()-t_start:.0f}s elapsed", flush=True)

print(f"STAGE 5/5 test prediction + submission (TTA={use_tta})", flush=True)
rle_dict = predict_test(model, test_loader, device, cfg["thresholds"], cfg["min_component"], cfg["min_pixels"], use_tta=use_tta)
submission_df = build_submission(rle_dict=rle_dict, sample_sub_path=SAMPLE_SUB_PATH, output_csv="/kaggle/working/submission.csv")
display(submission_df.head(12))

# submission validation
sub = pd.read_csv("/kaggle/working/submission.csv")
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
assert len(sub) == len(sample_sub), "row count mismatch"
assert set(sub["image_id"]) == set(sample_sub["image_id"]), "image_id set mismatch"
assert sub.notna().all().all(), "NaN in submission"
print(f"Submission validation PASSED | total {time.time()-t_start:.0f}s", flush=True)
''')

# --- cell 52: model attached via model_sources (kagglehub download fails in non-interactive sessions) ---
set_cell_src(52, '''def load_attached_model(device):
    model_files = list(Path("/kaggle/input").rglob("*.pth"))
    if not model_files: raise FileNotFoundError("No .pth model attached!")
    model_path = model_files[0]
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device).eval()
    print(f"Model loaded: {model_path}")
    return model
''')

# --- cell 56: use attached-model loader ---
src = "".join(nb["cells"][56]["source"]).replace(
    "model = load_kaggle_model(handle, device)", "model = load_attached_model(device)")
set_cell_src(56, src)

# clean outputs from all cells
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb_path = DST_DIR / "sweep-v1.ipynb"
nb_path.write_text(json.dumps(nb, indent=1))
json.dump({
    "id": "coder057/dl-sweep-v1",
    "title": "DL-sweep-v1",
    "code_file": "sweep-v1.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": ["industrial-surface-defect-detection"],
    "model_sources": ["coder057/attention-unet/pytorch/baseline-v1.0/1"],
    "machine_shape": "CPU",
}, open(DST_DIR / "kernel-metadata.json", "w"), indent=2)
print("written:", nb_path)
