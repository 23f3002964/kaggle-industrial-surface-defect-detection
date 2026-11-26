import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "acct-bytescode0115-csv-resubmit"
D.mkdir(exist_ok=True)

CODE = '''import time
import pandas as pd
from pathlib import Path

t0 = time.time()
ROOT = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public/")
if not (ROOT / "sample_submission.csv").exists():  # Kaggle mounts the competition data at varying depths
    _hits = sorted(Path("/kaggle/input").rglob("sample_submission.csv"))
    if _hits:
        ROOT = _hits[0].parent
print("ROOT:", ROOT, flush=True)
assert (ROOT / "sample_submission.csv").exists(), f"competition data not mounted: {ROOT}"

sample = pd.read_csv(ROOT / "sample_submission.csv")
assert list(sample.columns) == ["ID", "image_id", "defect_type", "mask_rle"], list(sample.columns)
test_dir = ROOT / "test"
n_test = len(list(test_dir.iterdir())) if test_dir.exists() else -1
print(f"sample {len(sample)} rows | test dir {n_test} files | {time.time()-t0:.0f}s", flush=True)
assert n_test > 0, f"competition test data missing: {test_dir}"

srcs = sorted(Path("/kaggle/input").rglob("submission_v*.csv"))
print("dataset csvs:", [p.name for p in srcs], flush=True)
assert srcs, "csv dataset not mounted"

for p in srcs:
    df = pd.read_csv(p)
    assert list(df.columns) == list(sample.columns), f"{p.name}: columns {list(df.columns)}"
    assert df["ID"].tolist() == sample["ID"].tolist(), f"{p.name}: ID order mismatch"
    assert df["image_id"].tolist() == sample["image_id"].tolist(), f"{p.name}: image_id order mismatch"
    assert df["defect_type"].tolist() == sample["defect_type"].tolist(), f"{p.name}: defect_type order mismatch"
    out = Path("/kaggle/working") / p.name
    df.to_csv(out, index=False)
    n = int((df["mask_rle"].astype(str).str.strip() != "").sum())
    print(f"{out.name}: {len(df)} rows | {n} masks | {out.stat().st_size/1e6:.2f}MB", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
'''

nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "csv-resubmit", "metadata": {},
                 "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(D / "resubmit.ipynb").write_text(json.dumps(nb, indent=1))

meta = {"id": "bytescode0115/isdd-csv-resubmit",
        "title": "isdd csv resubmit",
        "code_file": "resubmit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ["bytescode0115/isdd-csvs-64294"],
        "competition_sources": ["industrial-surface-defect-detection"],
        "kernel_sources": []}
(D / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
compile(CODE, "resubmit", "exec")
print("written", D)
