"""Phase 9 — Ablation study of the transfer learning framework.

One-factor-at-a-time ablation from the default configuration
(zero_fill alignment, repr_dim=32, lambda_da=1.0, src_cap=200K):

    alignment   : zero_fill | semantic | intersection      (all methods)
    repr_dim    : 8 | 16 | 32 | 64                          (all methods)
    lambda_da   : 0.1 | 0.5 | 1.0 | 2.0                     (dann only)
    src_cap     : 50K | 100K | 200K                         (all methods)

Methods: dann@0%, contrastive@10%, finetune@10% (headline methods).
Pairs: 6 representative directed pairs (each dataset appears as src and tgt).

Reports both fixed-0.5 and val-tuned-threshold F1 (cf. Phase 7b).

Usage
-----
    cd E:/Research_Works/Fraud_Detection_Research/financial-fraud-transfer
    fraud_env/Scripts/python.exe -m experiments.ablation.run_ablation [--seed 42]

Outputs (experiments/ablation/results/)
---------------------------------------
    ablation_results.json     — all metrics (appended cell-by-cell, resume-capable)
    ablation_{axis}.png       — mean tuned-F1 vs axis value per method
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import precision_recall_curve

from src.evaluation.metrics import compute_metrics
from src.models.transfer.adapter import TransferAdapter

from experiments.transfer.config import DEEP_CONFIG, RF_CONFIG
from experiments.transfer.run_transfer import (
    DataCache, _sample_labeled_target, _stratified_subsample,
)
from experiments.cross_domain.feature_aligner import make_aligner

# ── experiment grid ───────────────────────────────────────────────────────────

PAIRS = [
    ("ulb",      "ieee_cis"),
    ("ieee_cis", "ulb"),
    ("ieee_cis", "paysim"),
    ("paysim",   "ulb"),
    ("paysim",   "banksim"),
    ("banksim",  "paysim"),
]

# (method, label_frac)
METHODS = [
    ("dann",        0.00),
    ("contrastive", 0.10),
    ("finetune",    0.10),
]

DEFAULTS = {"alignment": "zero_fill", "repr_dim": 32, "lambda_da": 1.0, "src_cap": 200_000}

AXES = {
    "alignment": ["zero_fill", "semantic", "intersection"],
    "repr_dim":  [8, 16, 32, 64],
    "lambda_da": [0.1, 0.5, 1.0, 2.0],   # dann only
    "src_cap":   [50_000, 100_000, 200_000],
}

OUT_DIR      = ROOT / "experiments" / "ablation" / "results"
RESULTS_FILE = OUT_DIR / "ablation_results.json"


def build_configs() -> list[dict]:
    """One-factor-at-a-time configs, deduplicated (default counted once per method)."""
    configs = []
    seen = set()
    for method, frac in METHODS:
        for axis, values in AXES.items():
            if axis == "lambda_da" and method != "dann":
                continue
            for v in values:
                cfg = dict(DEFAULTS)
                cfg[axis] = v
                key = (method, cfg["alignment"], cfg["repr_dim"], cfg["lambda_da"], cfg["src_cap"])
                if key in seen:
                    continue
                seen.add(key)
                configs.append({
                    "method": method, "label_frac": frac,
                    "axis": axis, "value": v, **cfg,
                })
    return configs


# ── threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(y_val, prob_val) -> float:
    """Return threshold maximising F1 on validation set."""
    if len(np.unique(y_val)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, prob_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve returns len(thr) = len(prec) - 1
    best = int(np.nanargmax(f1[:-1]))
    return float(thr[best])


# ── run one cell ──────────────────────────────────────────────────────────────

def run_one(cfg: dict, src: str, tgt: str, cache: DataCache, seed: int) -> dict:
    src_data = cache.get(src)
    tgt_data = cache.get(tgt)

    aligner = make_aligner(cfg["alignment"], src_data["feature_names"],
                           tgt_data["feature_names"])
    input_dim = aligner.n_features

    X_src_tr = aligner.align_source(src_data["X_tr"])
    y_src_tr = src_data["y_tr"]

    # Source cap ablation (cache already capped at 200K)
    cap = cfg["src_cap"]
    if len(X_src_tr) > cap:
        idx = _stratified_subsample(X_src_tr, y_src_tr, cap, seed)
        X_src_tr, y_src_tr = X_src_tr[idx], y_src_tr[idx]

    X_tgt_tr  = aligner.align_target(tgt_data["X_tr"])
    y_tgt_tr  = tgt_data["y_tr"]
    X_tgt_val = aligner.align_target(tgt_data["X_val"])
    y_tgt_val = tgt_data["y_val"]
    X_tgt_te  = aligner.align_target(tgt_data["X_te"])
    y_tgt_te  = tgt_data["y_te"]

    X_lab, y_lab = _sample_labeled_target(X_tgt_tr, y_tgt_tr, cfg["label_frac"], seed)

    deep_cfg = dict(DEEP_CONFIG)
    deep_cfg["repr_dim"]  = cfg["repr_dim"]
    deep_cfg["lambda_da"] = cfg["lambda_da"]

    adapter = TransferAdapter(
        method=cfg["method"],
        input_dim=input_dim,
        **deep_cfg,
        rf_n_estimators=RF_CONFIG["n_estimators"],
        rf_min_samples_leaf=RF_CONFIG["min_samples_leaf"],
        seed=seed,
    )

    t0 = time.time()
    adapter.fit(
        X_src_tr, y_src_tr,
        X_tgt_unlab=X_tgt_tr,
        X_tgt_lab=X_lab if len(X_lab) > 0 else None,
        y_tgt_lab=y_lab if len(y_lab) > 0 else None,
        X_val=X_tgt_val,
        y_val=y_tgt_val,
    )
    elapsed = time.time() - t0

    prob_te  = adapter.predict_proba(X_tgt_te)
    prob_val = adapter.predict_proba(X_tgt_val)

    m_fixed = compute_metrics(y_tgt_te, (prob_te >= 0.5).astype(int), prob_te)
    thr = tune_threshold(y_tgt_val, prob_val)
    m_tuned = compute_metrics(y_tgt_te, (prob_te >= thr).astype(int), prob_te)

    return {
        "src": src, "tgt": tgt,
        "method": cfg["method"], "label_frac": cfg["label_frac"],
        "axis": cfg["axis"], "value": cfg["value"],
        "alignment": cfg["alignment"], "repr_dim": cfg["repr_dim"],
        "lambda_da": cfg["lambda_da"], "src_cap": cfg["src_cap"],
        "input_dim": input_dim,
        "n_src_train": len(X_src_tr), "n_tgt_labeled": len(X_lab),
        "f1": m_fixed["f1"], "pr_auc": m_fixed["pr_auc"],
        "roc_auc": m_fixed.get("roc_auc"),
        "threshold": thr, "f1_tuned": m_tuned["f1"],
        "precision_tuned": m_tuned.get("precision"),
        "recall_tuned": m_tuned.get("recall"),
        "elapsed_s": round(elapsed, 2),
    }


# ── results I/O ───────────────────────────────────────────────────────────────

def _key(r: dict) -> tuple:
    return (r["src"], r["tgt"], r["method"], r["alignment"],
            r["repr_dim"], r["lambda_da"], r["src_cap"])


def load_results() -> list[dict]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results: list[dict]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ── figures ───────────────────────────────────────────────────────────────────

def make_figures(results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty:
        return

    for axis, values in AXES.items():
        # Rows where the varied axis equals `axis`, plus defaults (which belong to every axis)
        rows = []
        for method, _ in METHODS:
            if axis == "lambda_da" and method != "dann":
                continue
            for v in values:
                cfg = dict(DEFAULTS)
                cfg[axis] = v
                sub = df[(df["method"] == method)
                         & (df["alignment"] == cfg["alignment"])
                         & (df["repr_dim"] == cfg["repr_dim"])
                         & (df["lambda_da"] == cfg["lambda_da"])
                         & (df["src_cap"] == cfg["src_cap"])]
                if not sub.empty:
                    rows.append({"method": method, "value": v,
                                 "f1_tuned": sub["f1_tuned"].mean(),
                                 "pr_auc": sub["pr_auc"].mean()})
        adf = pd.DataFrame(rows)
        if adf.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in adf["method"].unique():
            sub = adf[adf["method"] == method]
            x = np.arange(len(sub))
            ax.plot(x, sub["f1_tuned"], marker="o", label=method)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(axis)
        ax.set_ylabel("Tuned F1 (mean across pairs)")
        ax.set_title(f"Ablation: {axis}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = OUT_DIR / f"ablation_{axis}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname.name}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty:
        print("No results.")
        return
    print("\n=== Mean tuned F1 by axis value (across pairs) ===")
    for axis, values in AXES.items():
        print(f"\n[{axis}]")
        for method, _ in METHODS:
            if axis == "lambda_da" and method != "dann":
                continue
            row = f"  {method:<12}"
            for v in values:
                cfg = dict(DEFAULTS)
                cfg[axis] = v
                sub = df[(df["method"] == method)
                         & (df["alignment"] == cfg["alignment"])
                         & (df["repr_dim"] == cfg["repr_dim"])
                         & (df["lambda_da"] == cfg["lambda_da"])
                         & (df["src_cap"] == cfg["src_cap"])]
                row += f"  {v!s:>12}={sub['f1_tuned'].mean():.3f}" if not sub.empty else f"  {v!s:>12}=  —  "
            print(row)


# ── main ──────────────────────────────────────────────────────────────────────

def run_ablation(seed: int = 42, resume: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    total = sum(len(PAIRS) for _ in configs)
    print("\n=== Phase 9: Ablation Study ===")
    print(f"Configs : {len(configs)}  ({total} cells over {len(PAIRS)} pairs)")
    print(f"Seed    : {seed}\n")

    results = load_results() if resume else []
    done = {_key(r) for r in results}
    print(f"  [resume] {len(results)} results on disk\n")

    cache = DataCache(seed=seed)
    for ds in {d for p in PAIRS for d in p}:
        cache.get(ds)

    run_count = 0
    for src, tgt in PAIRS:
        for cfg in configs:
            probe = {**cfg, "src": src, "tgt": tgt}
            if _key(probe) in done:
                continue
            tag = (f"{src}→{tgt} [{cfg['method']}] {cfg['axis']}={cfg['value']}")
            print(f"  {tag} ...", end=" ", flush=True)
            try:
                r = run_one(cfg, src, tgt, cache, seed)
                print(f"F1={r['f1']:.4f} tuned={r['f1_tuned']:.4f} "
                      f"PR-AUC={r['pr_auc']:.4f} ({r['elapsed_s']:.0f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                r = {**{k: cfg[k] for k in
                        ("method", "label_frac", "axis", "value",
                         "alignment", "repr_dim", "lambda_da", "src_cap")},
                     "src": src, "tgt": tgt, "error": str(e),
                     "f1": 0.0, "f1_tuned": 0.0, "pr_auc": 0.0}
            results.append(r)
            done.add(_key(r))
            run_count += 1
            save_results(results)

    save_results(results)
    print(f"\nResults → {RESULTS_FILE}  ({len(results)} records)")

    print("\n[Figures]")
    make_figures(results)
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 ablation study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_ablation(seed=args.seed, resume=not args.no_resume)
