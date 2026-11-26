#!/usr/bin/env python3
"""Build train-v1 kernel: 5-fold attention-UNet + aug + fold-ensemble + tuned postproc."""
import json, re
from pathlib import Path

SRC = Path("/home/adi/CODE/Kaggle/industrial-surface-defect-detection/coder057/dl-23f1000351-notebook-t32026.ipynb")
DST_DIR = Path("/home/adi/CODE/Kaggle/industrial-surface-defect-detection/coder057/train-v1")
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC.read_text())

def set_cell_src(i, src):
    nb["cells"][i]["source"] = src.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None

def patch(i, fn):
    src = "".join(nb["cells"][i]["source"])
    set_cell_src(i, fn(src))

# cell 2: no wandb, add nullcontext + Subset
patch(2, lambda s: s.replace(
    "# Experiment Tracking, Kaggle Utils\nimport wandb\nimport kagglehub\n",
    "# Kaggle utils\nfrom contextlib import nullcontext\nfrom torch.utils.data import Subset\nfrom iterstrat.ml_stratifiers import MultilabelStratifiedKFold\n"))

# cell 4: tuned thresholds from sweep-v1 (plain config)
patch(4, lambda s: s.replace('"class_thresholds": [0.25, 0.25, 0.60, 0.50]', '"class_thresholds": [0.20, 0.35, 0.575, 0.40]'))

# cell 5: training mode
patch(5, lambda s: s.replace("SUBMISSION = True", "SUBMISSION = False"))

# cell 6: no wandb login
set_cell_src(6, "# wandb skipped: plain console logging\n")

# cell 15: X,y only (folds made in training cell)
set_cell_src(15, '''X = image_df["image_id"].values
y = image_df[DEFECT_TYPES].values
print(f"Images: {len(X):,} | positive rate per class: {y.mean(0).round(3)}")''')

# cell 17: masks as uint8 (RAM: 18.7GB -> 4.7GB)
patch(17, lambda s: s.replace(
    'masks = np.zeros((n, CONFIG["num_classes"], height, width), dtype=np.float32)',
    'masks = np.zeros((n, CONFIG["num_classes"], height, width), dtype=np.uint8)'))

# cell 19: dataset with train augmentation
set_cell_src(19, '''def augment(image, mask):
    # image (1,H,W) float32, mask (C,H,W) uint8
    if random.random() < 0.5:
        image = image[:, :, ::-1]; mask = mask[:, :, ::-1]
    if random.random() < 0.25:
        image = image[:, ::-1, :]; mask = mask[:, ::-1, :]
    if random.random() < 0.3:
        image = image * (0.9 + 0.2 * random.random()) + (random.random() - 0.5) * 0.05
        image = np.clip(image, 0.0, 1.0)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class SurfaceDefectDataset(Dataset):
    def __init__(self, images, masks=None, image_ids=None, is_test=False, is_train=False):
        self.images, self.masks = images, masks
        self.image_ids, self.is_test, self.is_train = image_ids, is_test, is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.is_test:
            return torch.from_numpy(np.ascontiguousarray(image)), str(self.image_ids[idx])
        mask = self.masks[idx]
        if self.is_train:
            image, mask = augment(image, mask)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(mask.astype(np.float32))''')

# cell 21: preload ALL train once + test
set_cell_src(21, '''# Preload all training images + masks once (masks uint8)
all_images, all_masks, all_ids = preload_data(image_df=image_df, annotations_df=df, image_dir=TRAIN_DIR, desc="All train")
print(f"all: images {all_images.shape} {all_images.nbytes/1e9:.1f}GB, masks {all_masks.shape} {all_masks.nbytes/1e9:.1f}GB", flush=True)

# Test set
sub_df = pd.read_csv(SAMPLE_SUB_PATH)
test_image_df = pd.DataFrame({"image_id": sub_df["image_id"].unique()})
test_images, test_ids = preload_data(image_df=test_image_df, image_dir=TEST_DIR, desc="Test", is_test=True)
print(f"test: {test_images.shape} {test_images.nbytes/1e9:.1f}GB", flush=True)''')

# cell 24: base datasets (zero-copy subsets per fold)
set_cell_src(24, '''base_train = SurfaceDefectDataset(images=all_images, masks=all_masks, image_ids=all_ids, is_train=True)
base_val = SurfaceDefectDataset(images=all_images, masks=all_masks, image_ids=all_ids, is_train=False)
test_dataset = SurfaceDefectDataset(images=test_images, image_ids=test_ids, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
print(f"base datasets: {len(base_train)} imgs | test {len(test_dataset)} imgs", flush=True)''')

# cell 39: fit_model without wandb + elapsed/ETA prints
def fix_fit(s):
    s = "import time\n" + s
    s = re.sub(r"    # Initialize wandb.*?run\.watch\(model, log=\"all\", log_freq=7\)\n", "", s, flags=re.S)
    s = re.sub(r"        wandb\.log\(.*?step=epoch \+ 1\)\n", "", s, flags=re.S)
    s = s.replace('    run.finish()\n', '')
    s = s.replace("    patience = 7\n", "    patience = 5\n")
    s = s.replace(
        '    for epoch in range(epochs):\n        \n        train_metrics',
        '    t_fit = time.time()\n    for epoch in range(epochs):\n        e0 = time.time()\n        train_metrics')
    s = s.replace(
        "        print(f\"LR: [{optimizer.param_groups[0]['lr']:.7f}] Early Stopping: [{early_stop_counter}/{patience}] Best Valid Score: [{best_score:.4f}]\")",
        "        el = time.time() - t_fit\n        print(f\"epoch {epoch+1}/{epochs} | val score {current_score:.4f} best {best_score:.4f} | {el:.0f}s elapsed, ~{el/(epoch+1)*(epochs-epoch-1):.0f}s left\", flush=True)")
    return s
patch(39, fix_fit)

# cell 42: gc + sweep utilities (used by fold loop + ensemble)
set_cell_src(42, '''gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
        tgt_px_c, pos_c, emp_c = tgt_px[:, c], pos[:, c], emp[:, c]
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
    return out''')

# cell 54: utilities moved to cell 42
set_cell_src(54, "# inference/postproc utilities moved to the utils cell")

# cell 44: 5-fold training + per-fold probs accumulation (resumable artifacts)
set_cell_src(44, '''# ===== fold training (aug) + fold ensemble =====
import time, json
t_start = time.time()
TIME_BUDGET = 9.0 * 3600  # 4 folds @ ~3.25h; stops before a 5th fold could overrun the 12h session cap

kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
fold_splits = list(kfold.split(X, y))
print(f"{len(fold_splits)} folds | {time.time()-t_start:.0f}s elapsed", flush=True)

probs_sum, n_done, thr_per_fold = None, 0, []
H_, W_ = test_images.shape[-2:]
reuse_fold1 = list(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD1.pth"))

for fold, (tr_idx, va_idx) in enumerate(fold_splits, start=1):
    if time.time() - t_start > TIME_BUDGET:
        print(f"TIME BUDGET hit before fold {fold}; ensembling {n_done} folds", flush=True)
        break
    print(f"STAGE fold {fold}/{len(fold_splits)} training", flush=True)
    tr_loader = DataLoader(Subset(base_train, tr_idx), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(Subset(base_val, va_idx), batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    if fold == 1 and reuse_fold1:
        fold_model = torch.load(reuse_fold1[0], map_location=device, weights_only=False).to(device).eval()
        print(f"fold 1 REUSED: {reuse_fold1[0]}", flush=True)
        history = {"valid": {"score": [0.7147]}}
    else:
        fold_model = AttentionDefectUNet(in_channels=1, num_classes=4).to(device)
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        criterion = SoftDiceBCELoss(bce_weight=0.4, dice_weight=0.6)
        save_path = f"/kaggle/working/ATTENTION-UNET-FOLD{fold}.pth"
        history = fit_model(fold_model, tr_loader, va_loader, optimizer, criterion,
                            epochs=20, log_every=40, run_name=f"ATTN-FOLD{fold}", save_path=save_path)
    print(f"fold {fold} best val score {max(history['valid']['score']):.4f} | {time.time()-t_start:.0f}s elapsed", flush=True)

    print(f"fold {fold}: val probs (TTA) + threshold sweep", flush=True)
    vp, vt = [], []
    for images, targets in tqdm(va_loader, desc=f"[fold{fold} val]", dynamic_ncols=True):
        vp.append(model_probs(fold_model, images, device, use_tta=True))
        vt.append(targets.numpy().astype(np.uint8))
    vp, vt = np.concatenate(vp), np.concatenate(vt)
    fthr = best_thresholds(vp, vt)
    thr_per_fold.append([d["threshold"] for d in fthr])
    print(f"fold {fold} thresholds {[d['threshold'] for d in fthr]} | {time.time()-t_start:.0f}s elapsed", flush=True)
    del vp, vt, va_loader; gc.collect()

    print(f"fold {fold}: test probs (TTA)", flush=True)
    fold_probs = np.empty((len(test_dataset), 4, H_, W_), dtype=np.float16)
    off = 0
    for images, ids in tqdm(test_loader, desc=f"[fold{fold} test]", dynamic_ncols=True):
        p = model_probs(fold_model, images, device, use_tta=True)
        fold_probs[off:off + p.shape[0]] = p
        off += p.shape[0]
    if probs_sum is None:
        probs_sum = fold_probs.astype(np.float32)
    else:
        # chunked += : avoids a second 4.9GB float32 temporary
        for i0 in range(0, len(probs_sum), 256):
            probs_sum[i0:i0 + 256] += fold_probs[i0:i0 + 256]
    n_done = fold
    np.save("/kaggle/working/ensemble_probs_sum.npy", probs_sum)
    json.dump({"n_done": n_done, "thresholds": thr_per_fold}, open("/kaggle/working/ensemble_meta.json", "w"))
    del fold_model, fold_probs, tr_loader; gc.collect(); torch.cuda.empty_cache()
    print(f"fold {fold} DONE | {time.time()-t_start:.0f}s elapsed", flush=True)

# free the 10.6GB preload before the ensemble cell needs RAM
del base_train, base_val, all_images, all_masks; gc.collect()
print("preload freed", flush=True)''')

# cells 45-50: unused legacy training cells
for i in (45, 46, 49, 50):
    set_cell_src(i, "# (legacy single-model training cell removed)")

# cell 49/50 note: keep 48 optimizer helper? not needed
set_cell_src(48, "# (legacy fine-tune optimizer removed)")

# cell 52: no external model needed (training from scratch)
set_cell_src(52, "# training from scratch; no pretrained model load")

# cell 56: tuned config constants
set_cell_src(56, '''# tuned config from sweep-v1 (TTA variant) as fallback
TUNED = {
    "thresholds": [0.175, 0.05, 0.375, 0.2],
    "min_component": [48, 32, 160, 256],
    "min_pixels": [64, 0, 192, 384],
    "val_score": 0.7008,
}''')

# cell 57: ensemble + submission
set_cell_src(57, '''# ===== fold ensemble + submission =====
import time, json
t_start = time.time()
meta = json.load(open("/kaggle/working/ensemble_meta.json"))
n = meta["n_done"]
thr = np.mean(np.array(meta["thresholds"]), axis=0).tolist()
print(f"ensembling {n} folds | thresholds {thr}", flush=True)

mc, mt = TUNED["min_component"], TUNED["min_pixels"]
mean_probs = probs_sum
mean_probs /= n  # in-place: avoids a second 4.9GB array
pred = postprocess(mean_probs, thr, mc, mt)
print(f"postproc done | {time.time()-t_start:.0f}s elapsed", flush=True)

rle_dict = {}
for b in range(pred.shape[0]):
    img_id = str(test_dataset.image_ids[b])
    for c, dt in enumerate(DEFECT_TYPES):
        mask = pred[b, c]
        rle_dict[(img_id, dt)] = rle_encode(mask) if mask.sum() > 0 else " "
submission_df = build_submission(rle_dict=rle_dict, sample_sub_path=SAMPLE_SUB_PATH, output_csv="/kaggle/working/submission.csv")
display(submission_df.head(12))

sub = pd.read_csv("/kaggle/working/submission.csv")
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
assert len(sub) == len(sample_sub), "row count mismatch"
assert set(sub["image_id"]) == set(sample_sub["image_id"]), "image_id set mismatch"
assert sub.notna().all().all(), "NaN in submission"
print(f"Submission validation PASSED | total {time.time()-t_start:.0f}s", flush=True)''')

# clean outputs
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb_path = DST_DIR / "train-v1.ipynb"
nb_path.write_text(json.dumps(nb, indent=1))
json.dump({
    "id": "coder057/dl-train-v1",
    "title": "DL-train-v1",
    "code_file": "train-v1.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": ["coder057/fold1-model"],
    "kernel_sources": [],
    "competition_sources": ["industrial-surface-defect-detection"],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}, open(DST_DIR / "kernel-metadata.json", "w"), indent=2)
print("written:", nb_path)
