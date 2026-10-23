"""Threshold-tuned re-evaluation of saved transfer adapters (Phase 7b).

Tests the Chapter 5 claim that contrastive alignment's non-monotonic F1 is a
threshold-calibration artefact: reload each saved contrastive/finetune adapter,
choose the F1-optimal threshold on the TARGET VALIDATION split, and re-score
the untouched target test split at that threshold. No retraining occurs.

Usage
-----
    cd E:/Research_Works/Fraud_Detection_Research/financial-fraud-transfer
    python -m experiments.transfer.run_threshold_tuned

Outputs (under experiments/transfer/results/)
---------------------------------------------
    threshold_tuned_results.json
    f1_fixed_vs_tuned.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.transfer.encoder import FraudEncoder
from src.models.transfer.contrastive import ContrastiveEncoder
from experiments.transfer.config import ALL_PAIRS, DEEP_CONFIG
from experiments.transfer.run_transfer import DataCache
from experiments.cross_domain.feature_aligner import make_aligner

OUT_DIR = ROOT / "experiments" / "transfer" / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seed 42 has the full supervised grid; seeds 123/456 have the key cells only
# (matching the Section 4.4 multi-seed protocol).
CELLS_BY_SEED = {
    42:  {"contrastive": (0.01, 0.05, 0.10, 0.20),
          "finetune":    (0.01, 0.05, 0.10, 0.20),
          "tradaboost":  (0.01, 0.05, 0.10, 0.20)},
    123: {"contrastive": (0.01,), "finetune": (0.10,)},
    456: {"contrastive": (0.01,), "finetune": (0.10,)},
    789: {"contrastive": (0.01,), "finetune": (0.10,)},
    1024: {"contrastive": (0.01,), "finetune": (0.10,)},
}


def paths_for_seed(seed: int):
    models = ROOT / "experiments" / "transfer" / (
        "models" if seed == 42 else f"models_seed{seed}")
    suffix = "" if seed == 42 else f"_seed{seed}"
    return models, OUT_DIR / f"threshold_tuned_results{suffix}.json"


def rebuild_model(payload: dict):
    method = payload["method"]
    if method == "tradaboost":
        return payload["tradaboost"]  # sklearn-based, no torch rebuild
    if method == "contrastive":
        model = ContrastiveEncoder(
            input_dim=payload["input_dim"],
            encoder_dims=list(payload["hidden_dims"]),
            latent_dim=payload["repr_dim"],
            dropout=payload["dropout"],
        )
    else:  # finetune (FraudEncoder)
        model = FraudEncoder(
            payload["input_dim"], payload["hidden_dims"],
            payload["repr_dim"], payload["dropout"],
        )
    model.load_state_dict(payload["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict_proba(model, method: str, X: np.ndarray) -> np.ndarray:
    if method == "tradaboost":
        return model.predict_proba(X)
    if method == "contrastive":
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.FloatTensor(X[i:i + 4096]).to(DEVICE)
                probs.append(model.predict_proba(xb).cpu().numpy())
        return np.concatenate(probs)
    return model.predict_proba(X, device=DEVICE)


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """F1-optimal threshold on validation probabilities."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def main(seed: int = 42):
    print(f"=== Threshold-tuned re-evaluation (device={DEVICE}, seed={seed}) ===")
    models_dir, results_file = paths_for_seed(seed)
    cells = CELLS_BY_SEED[seed]
    cache = DataCache(seed=seed)
    results = []

    for src, tgt in ALL_PAIRS:
        src_data = cache.get(src)
        tgt_data = cache.get(tgt)
        aligner = make_aligner("zero_fill", src_data["feature_names"],
                               tgt_data["feature_names"])
        X_val = aligner.align_target(tgt_data["X_val"])
        y_val = tgt_data["y_val"]
        X_te = aligner.align_target(tgt_data["X_te"])
        y_te = tgt_data["y_te"]

        for method, fracs in cells.items():
            for frac in fracs:
                tag = f"{src}_to_{tgt}_{method}_{int(frac * 100):03d}pct"
                path = models_dir / f"{tag}.joblib"
                if not path.exists():
                    print(f"  [skip] missing {tag}")
                    continue
                t0 = time.time()
                payload = joblib.load(path)
                model = rebuild_model(payload)

                p_val = predict_proba(model, method, X_val)
                p_te = predict_proba(model, method, X_te)

                thr = tune_threshold(y_val, p_val)
                f1_fixed = f1_score(y_te, (p_te >= 0.5).astype(int), zero_division=0)
                f1_tuned = f1_score(y_te, (p_te >= thr).astype(int), zero_division=0)

                rec = {
                    "src": src, "tgt": tgt, "method": method, "label_frac": frac,
                    "threshold": round(thr, 4),
                    "f1_fixed": round(float(f1_fixed), 4),
                    "f1_tuned": round(float(f1_tuned), 4),
                    "f1_gain": round(float(f1_tuned - f1_fixed), 4),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results.append(rec)
                print(f"  {src}->{tgt} [{method}] {frac:.0%}: "
                      f"fixed={f1_fixed:.4f}  tuned={f1_tuned:.4f}  "
                      f"(thr={thr:.3f}, {rec['elapsed_s']}s)")

                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)

    # Summary + figure
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n=== Mean F1 across 12 pairs (fixed 0.5 vs val-tuned threshold) ===")
    summary = df.groupby(["method", "label_frac"])[["f1_fixed", "f1_tuned"]].mean().round(3)
    print(summary)

    if seed == 42:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            for method, style in (("contrastive", "-"), ("finetune", "--"), ("tradaboost", ":")):
                sub = df[df.method == method].groupby("label_frac")[["f1_fixed", "f1_tuned"]].mean()
                ax.plot(sub.index * 100, sub.f1_fixed, "o" + style, label=f"{method} (fixed 0.5)")
                ax.plot(sub.index * 100, sub.f1_tuned, "s" + style, label=f"{method} (val-tuned)")
            ax.set_xlabel("Labeled target data (%)")
            ax.set_ylabel("F1 (mean across 12 pairs)")
            ax.set_title("Fixed vs Validation-Tuned Decision Threshold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "f1_fixed_vs_tuned.png", dpi=150)
            print(f"\nSaved figure -> f1_fixed_vs_tuned.png")
        except ImportError:
            pass

    print(f"Results -> {results_file}  ({len(results)} records)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, choices=sorted(CELLS_BY_SEED))
    main(seed=parser.parse_args().seed)
