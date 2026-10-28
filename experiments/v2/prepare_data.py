"""Per-seed data cache for the v2 (label-budget-honest) protocol.

For every seed and dataset: stratified 72/8/20 train/val/test split, a
DatasetPreprocessor fitted on that seed's TRAIN split only, and the processed
arrays written to dataset/processed/v2/seed{seed}/{dataset}.npz.

The raw transaction amount of each test row is kept (amt_te) for the
cost-sensitive analysis.

Usage
-----
    python -m experiments.v2.prepare_data --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import DatasetPreprocessor
from experiments.baseline.config import DATASET_CONFIG
from experiments.transfer.run_transfer import _raw_load

DATASETS = ("ulb", "ieee_cis", "paysim", "banksim")
AMOUNT_COL = {"ulb": "Amount", "ieee_cis": "TransactionAmt",
              "paysim": "amount", "banksim": "amount"}
CACHE_DIR = ROOT / "dataset" / "processed" / "v2"


def cache_path(dataset_id: str, seed: int) -> Path:
    return CACHE_DIR / f"seed{seed}" / f"{dataset_id}.npz"


def load_cached(dataset_id: str, seed: int) -> dict:
    with np.load(cache_path(dataset_id, seed), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["feature_names"] = [str(n) for n in d["feature_names"]]
    return d


def prepare(dataset_id: str, seed: int, force: bool = False) -> None:
    out = cache_path(dataset_id, seed)
    if out.exists() and not force:
        print(f"  [skip] {out.relative_to(ROOT)}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X_df, y, feature_types = _raw_load(dataset_id)
    cfg = DATASET_CONFIG[dataset_id]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_df, y, test_size=0.28, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.20 / 0.28, stratify=y_tmp, random_state=seed)

    amt_col = AMOUNT_COL[dataset_id]
    amt_te = (X_te[amt_col].to_numpy(dtype=np.float64)
              if amt_col in X_te.columns else np.full(len(X_te), np.nan))

    prep = DatasetPreprocessor(
        feature_types, dataset_id,
        missing_threshold=cfg.get("missing_threshold", 0.50),
        ohe_max_categories=cfg.get("ohe_max_categories", 50),
        scale_method=cfg.get("scale_method", "robust"),
    )
    np.savez(
        out,
        X_tr=prep.fit_transform(X_tr).astype(np.float32),
        y_tr=y_tr.to_numpy(dtype=np.int8),
        X_val=prep.transform(X_val).astype(np.float32),
        y_val=y_val.to_numpy(dtype=np.int8),
        X_te=prep.transform(X_te).astype(np.float32),
        y_te=y_te.to_numpy(dtype=np.int8),
        amt_te=amt_te,
        feature_names=np.array(prep.feature_names_out),
    )
    print(f"  {dataset_id} seed={seed}: {len(X_df):,} rows, "
          f"{prep.n_features_out} features ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for s in args.seeds:
        for ds in args.datasets:
            prepare(ds, s, args.force)
