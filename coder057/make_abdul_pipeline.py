#!/usr/bin/env python3
"""abdulmajid786123 side: infer test probs for the shared checkpoints and build blend kernels.

usage:
  python3 make_abdul_pipeline.py <acct> infer                     # shared checkpoints -> probs (GPU)
  python3 make_abdul_pipeline.py <acct> post <name> <weights-json> [scale] [mc_scale] [sources]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
from nbpatch import fix_root
ACCT = sys.argv[1]  # account that owns the kernels
MODE = sys.argv[2]  # infer | post

# ---------------- inference kernel ----------------
if MODE == "infer":
    SRC = json.load(open(HERE / "infer-v4" / "infer-v4.ipynb"))["cells"]
    cells = [dict(SRC[0]), dict(SRC[1])]
    cells[0]["source"] = fix_root("".join(cells[0]["source"]), "sample_submission.csv")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": '''sub_raw = pd.read_csv(SAMPLE)
test_ids = sub_raw["image_id"].unique()
H, W = 128, 800
test_u8 = np.empty((len(test_ids), H, W), dtype=np.uint8)
for i, iid in enumerate(tqdm(test_ids, desc="test imgs", dynamic_ncols=True)):
    name = iid if Path(iid).suffix else f"{iid}.jpg"
    with Image.open(TEST_DIR / name) as im:
        test_u8[i] = np.asarray(im.convert("L"), dtype=np.uint8)
print(f"test_u8 {test_u8.nbytes/1e9:.2f}GB | {time.time()-t_start:.0f}s", flush=True)

JOBS = [("5", "ATTENTION-UNET-FOLD5.pth"), ("ft1", "ATTENTION-UNET-FOLD1FT.pth"),
        ("ft4", "ATTENTION-UNET-FOLD4FT.pth")]
paths = {tag: sorted(Path("/kaggle/input").rglob(name))[0] for tag, name in JOBS}
# the `baseline` checkpoint is a private coder057 model -> no cross-account mount, so blends here are
# baseline-free (coder057 runs the baseline-including A/B on its own probs).
print("checkpoints:", {k: str(v) for k, v in paths.items()}, flush=True)


@torch.inference_mode()
def probs_tta(model, u8, tag=""):
    out = np.empty((len(u8), 4, H, W), dtype=np.float16)
    t0 = time.time()
    for s in range(0, len(u8), 64):
        x = torch.from_numpy(u8[s:s + 64]).unsqueeze(1).float().div_(255).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            logits = (logits + torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])) / 2
        out[s:s + 64] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float16)
        print(f"  {tag} {min(s+64, len(u8))}/{len(u8)} | {time.time()-t0:.0f}s", flush=True)
    return out


for tag, pth in paths.items():
    model = torch.load(pth, map_location=device, weights_only=False).to(device).eval()
    tp = probs_tta(model, test_u8, tag=tag)
    np.save(f"/kaggle/working/test_probs_{tag}.npy", tp)
    print(f"{tag} saved | mean {tp.mean():.4f} | {time.time()-t_start:.0f}s", flush=True)
    del model, tp
    gc.collect(); torch.cuda.empty_cache()
print(f"DONE | total {time.time()-t_start:.0f}s", flush=True)
'''})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"acct-{ACCT}-pt-infer-shared"
    out.mkdir(exist_ok=True)
    (out / "pt-infer-shared.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{ACCT}/pt-infer-shared", "title": "PT-infer-shared", "code_file": "pt-infer-shared.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": False, "dataset_sources": ["coder057/isdd-all-models"],
            "kernel_sources": [], "competition_sources": ["industrial-surface-defect-detection"],
            "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    errs = []
    for i, c in enumerate(nb["cells"], 1):
        try:
            compile("".join(c["source"]), f"c{i}", "exec")
        except SyntaxError as e:
            errs.append((i, e.lineno, e.msg))
    print(f"written {out} | syntax {errs}")
    raise SystemExit

# ---------------- postproc kernel ----------------
name = sys.argv[3]
weights = json.loads(sys.argv[4])
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
mc_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
sources = sys.argv[7].split(",") if len(sys.argv) > 7 else [
    f"{ACCT}/pt-infer-shared", f"{ACCT}/pt-ft2", f"{ACCT}/pt-ft3", f"{ACCT}/pt-all-s42", f"{ACCT}/pt-all-s7"]

BASE_TARGET = {"defect_a": 675, "defect_b": 570, "defect_c": 1440, "defect_d": 1271}
# TGT='{"defect_a":600,...}' overrides per-class target counts (global scale still applies to em? no: exact values)
TGT = json.loads(__import__("os").environ.get("TGT", "{}"))
src = (HERE / "make_postv.py").read_text()
NB_HEAD = src.split("NB_HEAD = r'''", 1)[1].split("\n'''", 1)[0]
settings = {"weights": weights, "mode": "mean",
            "target_masks": {k: int(TGT[k]) if k in TGT else int(round(n * scale)) for k, n in BASE_TARGET.items()},
            "min_component": [int(round(x * mc_scale)) for x in (48, 32, 160, 256)],
            "min_pixels": [int(round(x * mc_scale)) for x in (64, 0, 192, 384)]}
cell = NB_HEAD.replace("__SETTINGS__", json.dumps(settings))
nb2 = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11"}},
       "nbformat": 4, "nbformat_minor": 5}
out = HERE / f"acct-{ACCT}-{name}"
out.mkdir(exist_ok=True)
(out / f"{name}.ipynb").write_text(json.dumps(nb2, indent=1))
DSRC = [x for x in __import__("os").environ.get("DSRC", "").split(",") if x]
meta = {"id": f"{ACCT}/{name}", "title": name.upper(), "code_file": f"{name}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_internet": False, "dataset_sources": DSRC, "kernel_sources": sources,
        "competition_sources": ["industrial-surface-defect-detection"], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
errs = []
for i, c in enumerate(nb2["cells"], 1):
    try:
        compile("".join(c["source"]), f"c{i}", "exec")
    except SyntaxError as e:
        errs.append((i, e.lineno, e.msg))
print(f"written {out} | weights {list(weights)} | masks {sum(settings['target_masks'].values())} | syntax {errs}")
