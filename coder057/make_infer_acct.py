#!/usr/bin/env python3
"""Generate the all-folds inference kernel for a given account.

usage: python3 make_infer_acct.py <account> <n_folds>
produces <account>/dl-infer-all: test_probs_fold{1..n}.npy (own folds only)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
from nbpatch import fix_root
acct, n_folds = sys.argv[1], int(sys.argv[2])
slug = "dl-infer-all"
OUT = HERE / f"acct-{acct}-{slug}"
OUT.mkdir(exist_ok=True)

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

all_pth = sorted(Path("/kaggle/input").rglob("*.pth"))
fold_paths = {}
extra = []
for p in all_pth:
    stem = p.stem.upper()
    if "FOLD" in stem:
        try:
            fold_paths[int(stem.split("FOLD")[-1])] = p
        except ValueError:
            extra.append(p)
    else:
        extra.append(p)
print("fold models:", {k: str(v) for k, v in sorted(fold_paths.items())}, flush=True)
print("extra models:", [str(p) for p in extra], flush=True)
EXPECTED_FOLDS = list(range(1, __N_FOLDS__ + 1))
assert sorted(fold_paths) == EXPECTED_FOLDS, f"need folds {EXPECTED_FOLDS}, found {sorted(fold_paths)}"


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


jobs = [(f"fold{f}", p) for f, p in sorted(fold_paths.items())]
if extra:
    jobs.append(("baseline", extra[0]))

for tag, pth in jobs:
    try:
        model = torch.load(pth, map_location=device, weights_only=False).to(device).eval()
    except Exception as e:
        print(f"{tag} LOAD FAILED ({pth}): {e}", flush=True)
        continue
    tp = probs_tta(model, test_u8, tag=tag)
    np.save(f"/kaggle/working/test_probs_{tag}.npy", tp)
    print(f"{tag} probs saved | mean {tp.mean():.4f} | {time.time()-t_start:.0f}s", flush=True)
    del model, tp
    gc.collect(); torch.cuda.empty_cache()

print(f"DONE | total {time.time()-t_start:.0f}s", flush=True)
'''})

cells[-1]["source"] = cells[-1]["source"].replace("__N_FOLDS__", str(n_folds))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": f"{acct}/{slug}", "title": "DL-infer-all", "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [f"{acct}/dl-train-fold{i}" for i in range(1, n_folds + 1)],
        "competition_sources": ["industrial-surface-defect-detection"],
        "model_sources": []}  # cross-account model refs 404 -> baseline is blended on the coder057 side instead
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"written {OUT} | id {meta['id']} | sources {meta['kernel_sources']}")
