"""DANN lambda-sensitivity sweep (Phase 7c).

Tests whether the DANN results of the main grid depend on the single
adaptation-loss weight (lambda_da = 1.0) used throughout: retrains DANN at
lambda in {0.1, 0.5, 2.0} on all 12 pairs (seed 42, zero-fill alignment,
same backbone/protocol as the main grid) and compares to the archived
lambda = 1.0 results.

Usage
-----
    cd E:/Research_Works/Fraud_Detection_Research/financial-fraud-transfer
    python -m experiments.transfer.run_lambda_sensitivity

Outputs (under experiments/transfer/results/)
---------------------------------------------
    lambda_sensitivity_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics
from src.models.transfer.adapter import TransferAdapter
from experiments.transfer.config import ALL_PAIRS, DEEP_CONFIG, RF_CONFIG
from experiments.transfer.run_transfer import DataCache
from experiments.cross_domain.feature_aligner import make_aligner

OUT_DIR = ROOT / "experiments" / "transfer" / "results"
RESULTS_FILE = OUT_DIR / "lambda_sensitivity_results.json"

LAMBDAS = (0.1, 0.5, 2.0)  # 1.0 already in transfer_results.json
SEED = 42


def main():
    print(f"=== DANN lambda sensitivity (seed={SEED}, lambdas={LAMBDAS}) ===")
    cache = DataCache(seed=SEED)
    results = []
    if RESULTS_FILE.exists():
        results = json.load(open(RESULTS_FILE))
    done = {(r["src"], r["tgt"], r["lambda_da"]) for r in results}

    for src, tgt in ALL_PAIRS:
        src_data = cache.get(src)
        tgt_data = cache.get(tgt)
        aligner = make_aligner("zero_fill", src_data["feature_names"],
                               tgt_data["feature_names"])
        X_src_tr = aligner.align_source(src_data["X_tr"])
        y_src_tr = src_data["y_tr"]
        X_tgt_tr = aligner.align_target(tgt_data["X_tr"])
        X_val = aligner.align_target(tgt_data["X_val"])
        y_val = tgt_data["y_val"]
        X_te = aligner.align_target(tgt_data["X_te"])
        y_te = tgt_data["y_te"]

        for lam in LAMBDAS:
            if (src, tgt, lam) in done:
                continue
            cfg = {**DEEP_CONFIG, "lambda_da": lam}
            adapter = TransferAdapter(
                method="dann", input_dim=aligner.n_features, **cfg,
                rf_n_estimators=RF_CONFIG["n_estimators"],
                rf_min_samples_leaf=RF_CONFIG["min_samples_leaf"],
                seed=SEED,
            )
            t0 = time.time()
            adapter.fit(X_src_tr, y_src_tr, X_tgt_unlab=X_tgt_tr,
                        X_val=X_val, y_val=y_val)
            y_prob = adapter.predict_proba(X_te)
            m = compute_metrics(y_te, (y_prob >= 0.5).astype(int), y_prob)
            rec = {"src": src, "tgt": tgt, "lambda_da": lam,
                   "f1": round(m["f1"], 4), "pr_auc": round(m["pr_auc"], 4),
                   "elapsed_s": round(time.time() - t0, 1)}
            results.append(rec)
            print(f"  {src}->{tgt} lam={lam}: F1={m['f1']:.4f} "
                  f"PR-AUC={m['pr_auc']:.4f} ({rec['elapsed_s']}s)", flush=True)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    # Summary incl. archived lambda=1.0
    import pandas as pd
    df = pd.DataFrame(results)
    base = pd.DataFrame(json.load(open(OUT_DIR / "transfer_results.json")))
    dann10 = base[(base.method == "dann") & (base.label_frac == 0.0)][["src", "tgt", "f1", "pr_auc"]].copy()
    dann10["lambda_da"] = 1.0
    full = pd.concat([df[["src", "tgt", "lambda_da", "f1", "pr_auc"]], dann10])
    print("\n=== Mean over 12 pairs by lambda ===")
    print(full.groupby("lambda_da")[["f1", "pr_auc"]].mean().round(4))
    print(f"\nResults -> {RESULTS_FILE} ({len(results)} records)")


if __name__ == "__main__":
    main()
