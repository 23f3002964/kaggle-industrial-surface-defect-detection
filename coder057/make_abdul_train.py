#!/usr/bin/env python3
"""Generate training kernels for the third account (abdulmajid786123).

usage:
  python3 make_abdul_train.py <acct> ft  <fold> <epochs>       # fine-tune an existing fold checkpoint
  python3 make_abdul_train.py <acct> all <seed> <epochs> <tag>  # train on ALL train images

Both variants end with hflip-TTA test inference and write the test probs + the checkpoint, so the
next stage never needs a second GPU session.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
from nbpatch import fix_root

ACCT = sys.argv[1]  # account to push under
MODE = sys.argv[2]
SRC = json.load(open(HERE / "train-v7" / "train-v7.ipynb"))["cells"]

INFER_CELL = '''
# ===== test inference (hflip TTA) -> probs for the blend =====
TEST_DIR, SAMPLE = ROOT / "test", ROOT / "sample_submission.csv"
assert TEST_DIR.exists() and SAMPLE.exists(), f"test data missing: {TEST_DIR}"
sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800
test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {time.time()-t_start:.0f}s", flush=True)

model = torch.load(save_path, map_location=device, weights_only=False).to(device).eval()
out = np.empty((len(test_u8), 4, H, W), dtype=np.float16)
t0 = time.time()
with torch.inference_mode():
    for s in range(0, len(test_u8), 64):
        x = torch.from_numpy(test_u8[s:s + 64]).unsqueeze(1).float().div_(255).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            logits = (logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])) / 2
        out[s:s + 64] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float16)
        print(f"  {min(s+64, len(test_u8))}/{len(test_u8)} | {time.time()-t0:.0f}s", flush=True)
np.save(f"/kaggle/working/test_probs_{TAG}.npy", out)
print(f"probs saved tag={TAG} | mean {out.mean():.4f} | {time.time()-t_start:.0f}s", flush=True)
print(f"DONE | total {time.time()-t_start:.0f}s", flush=True)
'''


def build(mode, slug, cell9, dataset_sources, kernel_sources=()):
    cells = [dict(c) for c in SRC[:9]]
    cells[1]["source"] = fix_root("".join(cells[1]["source"]), "train.csv")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell9})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": INFER_CELL})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"acct-{ACCT}-{slug}"
    out.mkdir(exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{ACCT}/{slug}", "title": slug.upper(), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
            "dataset_sources": dataset_sources, "kernel_sources": list(kernel_sources),
            "competition_sources": ["industrial-surface-defect-detection"], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    errs = []
    for i, c in enumerate(nb["cells"], 1):
        try:
            compile("".join(c["source"]), f"c{i}", "exec")
        except SyntaxError as e:
            errs.append((i, e.lineno, e.msg))
    print(f"written {out} | id {meta['id']} | syntax {errs}")
    return errs


if MODE in ("ft", "ftb"):
    fold, epochs = int(sys.argv[3]), int(sys.argv[4])
    b = MODE == "ftb"  # 2nd round: start from the fine-tuned checkpoint at lower lr
    TAG = f"ft{fold}b" if b else f"ft{fold}"
    cell9 = f'''# ===== fine-tune fold {fold} for {epochs} extra epochs (lr {"1.5e-4" if b else "3e-4"}) =====
t_start = time.time()
EPOCHS = {epochs}
TAG = "{TAG}"
FINETUNE_FROM = next(Path("/kaggle/input").rglob("ATTENTION-UNET-FOLD{fold}{"FT" if b else ""}.pth"))
print("fine-tune checkpoint:", FINETUNE_FROM, flush=True)
kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
fold_splits = list(kfold.split(X, y))
tr_idx, va_idx = fold_splits[{fold} - 1]
print(f"STAGE fold {fold} | train {{len(tr_idx)}} val {{len(va_idx)}} | {{time.time()-t_start:.0f}}s", flush=True)
tr_loader = DataLoader(Subset(SurfaceDefectDataset(images_u8, masks, is_train=True), tr_idx),
                       batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
va_loader = DataLoader(Subset(SurfaceDefectDataset(images_u8, masks, is_train=False), va_idx),
                       batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
model = torch.load(FINETUNE_FROM, map_location=device, weights_only=False).to(device)
print("loaded checkpoint:", type(model).__name__, flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr={"1.5e-4" if b else "3e-4"}, weight_decay=CONFIG["weight_decay"])
criterion = SoftDiceBCELoss(bce_weight=0.4, dice_weight=0.6)
save_path = "/kaggle/working/ATTENTION-UNET-FOLD{fold}FT{"B" if b else ""}.pth"
history = fit_model(model, tr_loader, va_loader, optimizer, criterion,
                    epochs=EPOCHS, log_every=40, run_name="ATTN-FOLD{fold}FT{"B" if b else ""}", save_path=save_path)
print(f"fold {fold}ft{"b" if b else ""} best val {{max(history['valid']['score']):.4f}} | {{time.time()-t_start:.0f}}s", flush=True)
'''
    # round-2 must start from the round-1 checkpoint, which only exists in our own kernel output
    build("ft", f"pt-ft{fold}{"b" if b else ""}", cell9, ["coder057/isdd-all-models"],
          [f"{sys.argv[1]}/pt-ft{fold}"] if b else [])
elif MODE == "allb":
    src_kernel, src_tag, epochs, tag = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
    cell9 = f'''# ===== continue training on ALL train images from {src_tag}, {epochs} epochs (lr 3e-4) =====
t_start = time.time()
EPOCHS = {epochs}
TAG = "{tag}"
SRC = next(Path("/kaggle/input").rglob("ATTENTION-UNET-{src_tag}.pth"))
print("continue from:", SRC, flush=True)
tr_loader = DataLoader(SurfaceDefectDataset(images_u8, masks, is_train=True),
                       batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
mon_loader = DataLoader(SurfaceDefectDataset(images_u8, masks, is_train=False),
                        batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
model = torch.load(SRC, map_location=device, weights_only=False).to(device)
print("loaded:", type(model).__name__, flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=CONFIG["weight_decay"])
criterion = SoftDiceBCELoss(bce_weight=0.4, dice_weight=0.6)
save_path = "/kaggle/working/ATTENTION-UNET-{tag}.pth"
history = fit_model(model, tr_loader, mon_loader, optimizer, criterion,
                    epochs=EPOCHS, log_every=40, run_name="ATTN-{tag}", save_path=save_path)
print(f"all-data continue done | best monitor score {{max(history['valid']['score']):.4f}} | {{time.time()-t_start:.0f}}s", flush=True)
'''
    build("allb", f"pt-all-{tag}", cell9, [], [sys.argv[1] + "/" + src_kernel])
else:
    seed, epochs, tag = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    TAG = tag
    cell9 = f'''# ===== train on ALL train images (100% data), {epochs} epochs, seed {seed} =====
t_start = time.time()
EPOCHS = {epochs}
TAG = "{tag}"
SEED = {seed}
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
CONFIG["seed"] = SEED
print(f"ALL-DATA training | images {{len(images_u8)}} | seed {{SEED}} | {{EPOCHS}} ep", flush=True)
tr_loader = DataLoader(SurfaceDefectDataset(images_u8, masks, is_train=True),
                       batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
mon_loader = DataLoader(SurfaceDefectDataset(images_u8, masks, is_train=False),
                        batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
model = AttentionDefectUNet(in_channels=1, num_classes=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
criterion = SoftDiceBCELoss(bce_weight=0.4, dice_weight=0.6)
save_path = "/kaggle/working/ATTENTION-UNET-{tag}.pth"
history = fit_model(model, tr_loader, mon_loader, optimizer, criterion,
                    epochs=EPOCHS, log_every=40, run_name="ATTN-{tag}", save_path=save_path)
print(f"all-data done | best monitor score {{max(history['valid']['score']):.4f}} | {{time.time()-t_start:.0f}}s", flush=True)
'''
    build("all", f"pt-all-{tag}", cell9, [])
