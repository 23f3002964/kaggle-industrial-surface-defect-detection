#!/usr/bin/env python3
"""Build the end-to-end best-score kernel: checkpoints -> hflip-TTA probs -> blend -> calibrated masks.

One GPU kernel, coder057, nothing precomputed: the 5 blend members are loaded from raw .pth files
(baseline from model_sources, ft1b/ft3b/ft4b/ft5b from the isdd-ftb-models dataset), inference is
uint8-quantised, then the exact postproc that scored 0.65581 runs in the same notebook.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
INFER = json.load(open(HERE / "infer-v3" / "infer-v3.ipynb"))["cells"]
POST = (HERE / "make_postv.py").read_text().split("NB_HEAD = r'''", 1)[1].split("\n'''", 1)[0]

SETTINGS = {
    "weights": {"baseline": 1.0, "ft1b": 1.0, "ft3b": 1.0, "ft4b": 1.0, "ft5b": 1.0},
    "mode": "mean",
    "target_masks": {"defect_a": 702, "defect_b": 593, "defect_c": 1498, "defect_d": 1322},
    "min_component": [48, 32, 160, 256],
    "min_pixels": [64, 0, 192, 384],
}
POST = POST.replace("__SETTINGS__", json.dumps(SETTINGS))
# the u8 dumps live in /kaggle/working in this kernel, not only under /kaggle/input
POST = POST.replace(
    'for p in Path("/kaggle/input").rglob("test_probs_*.npy"):',
    'for p in [q for root in ("/kaggle/input", "/kaggle/working") for q in Path(root).rglob("test_probs_*.npy")]:')

LOAD = '''# ===== stage 1/3 | resolve the 5 checkpoints and load the test images =====
CKPTS = {}
for p in sorted(Path("/kaggle/input").rglob("*.pth")):
    if "FTB" not in p.name:
        tag = "baseline"
    else:
        # ATTENTION-UNET-FOLD1FTB.pth -> fold num = 1 -> ft1b
        fold_num = p.name.split("FOLD")[1].split("FTB")[0]
        tag = f"ft{fold_num}b"
    CKPTS[tag] = p
print("checkpoints:", {k: str(v) for k, v in CKPTS.items()}, flush=True)
assert set(CKPTS) == {"baseline", "ft1b", "ft3b", "ft4b", "ft5b"}, f"unexpected members {sorted(CKPTS)}"

sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800
test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {len(test_ids)} imgs | {time.time()-t_start:.0f}s", flush=True)
'''

INFER_ALL = '''# ===== stage 2/3 | hflip-TTA inference for every blend member, uint8 dumps =====
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


for i, (tag, ckpt) in enumerate(sorted(CKPTS.items()), 1):
    model = torch.load(ckpt, map_location=device, weights_only=False).to(device).eval()
    print(f"\\n[{i}/5] {tag} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | "
          f"{time.time()-t_start:.0f}s elapsed", flush=True)
    tp = probs_tta(model, test_u8, tag=tag)
    u8 = np.clip(tp * 255.0, 0, 255).round().astype(np.uint8)  # same quantisation the earlier chain used
    np.save(f"/kaggle/working/test_probs_{tag}.npy", u8)
    print(f"[{i}/5] {tag} saved uint8 | mean {float(tp.mean()):.4f} | {time.time()-t_start:.0f}s", flush=True)
    del model, tp, u8
    gc.collect(); torch.cuda.empty_cache()
'''

cells = [{"cell_type": "markdown", "metadata": {}, "source": [
    "# Best-score pipeline, end to end (0.65581)\n",
    "\n",
    "One kernel, from raw checkpoints to `submission.csv`: **5-model hflip-TTA inference -> probability\n",
    "blend -> per-class mask-budget calibration -> RLE**. Nothing is mounted precomputed except the\n",
    "baseline model (Kaggle Models) and the four round-2 fine-tuned checkpoints (dataset).\n",
    "\n",
    "| stage | what it does |\n",
    "|---|---|\n",
    "| 1/3 | resolve 5 `.pth` checkpoints, load 3000 test images as uint8 |\n",
    "| 2/3 | hflip TTA, sigmoid, quantise to uint8 (`p*255`) per member |\n",
    "| 3/3 | average the 5 prob tensors, calibrate per-class thresholds to the mask budget, CC-filter, RLE |\n",
    "\n",
    "Expected output: **3903 masks** (a=670, b=546, c=1468, d=1219) = public LB **0.65581**.\n"]}]

for c in INFER[:2]:  # imports/config + model classes, verbatim
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": "".join(c["source"])})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": LOAD})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": INFER_ALL})
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 3/3 - blend and calibrated masks\n",
    "\n",
    "Equal-weight probability average, then per defect type the threshold is solved so that exactly\n",
    "`target_masks[class]` images get a mask, retried against the real connected-component filter.\n"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": POST})

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "acct-coder057-isdd-e2e-v75"
out.mkdir(exist_ok=True)
name = "isdd-e2e-v75"
(out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"coder057/{name}", "title": "ISDD-E2E-V75", "code_file": f"{name}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": False, "keywords": ["segmentation", "unet", "tta"],
        "dataset_sources": ["coder057/isdd-ftb-models"],
        "kernel_sources": [], "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": ["coder057/attention-unet/pytorch/baseline-v1.0/1"],
        "machine_shape": "NvidiaTeslaT4"}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

errs = []
for i, c in enumerate(cells, 1):
    if c["cell_type"] == "code":
        try:
            compile("".join(c["source"]), f"c{i}", "exec")
        except SyntaxError as e:
            errs.append((i, e.lineno, e.msg))
print(f"written {out} | cells {len(cells)} | syntax {errs}")
