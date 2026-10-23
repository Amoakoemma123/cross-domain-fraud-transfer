"""Phase 10 — Imbalance strategy study.

Compares class-imbalance handling strategies on the two strongest baseline
models (RF, LightGBM) across all four datasets:

    none         no correction (raw imbalanced training)
    class_weight balanced class weights (the Phase 4 baseline setting)
    undersample  RandomUnderSampler to 1:1
    smote        SMOTE to pos:neg = 0.1
    smote_under  SMOTE to 0.1, then RandomUnderSampler to 0.5

Reports fixed-0.5 and val-tuned-threshold F1 plus PR-AUC.

Usage
-----
    cd E:/Research_Works/Fraud_Detection_Research/financial-fraud-transfer
    fraud_env/Scripts/python.exe -m experiments.imbalance.run_imbalance [--seed 42]

Outputs (experiments/imbalance/results/)
----------------------------------------
    imbalance_results.json    — all metrics (resume-capable)
    imbalance_f1.png          — tuned F1 per strategy, grouped by dataset
    imbalance_prauc.png       — PR-AUC per strategy, grouped by dataset
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

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.data.preprocessing import DatasetPreprocessor
from src.evaluation.metrics import compute_metrics
from experiments.baseline.config import DATASET_CONFIG, MODEL_CONFIG
from experiments.baseline.run_baseline import load_dataset, stratified_split, subsample
from experiments.ablation.run_ablation import tune_threshold

DATASETS   = ["ulb", "ieee_cis", "paysim", "banksim"]
MODELS     = ["rf", "lgbm"]
STRATEGIES = ["none", "class_weight", "undersample", "smote", "smote_under"]

OUT_DIR      = ROOT / "experiments" / "imbalance" / "results"
RESULTS_FILE = OUT_DIR / "imbalance_results.json"


# ── resampling ────────────────────────────────────────────────────────────────

def resample(strategy: str, X, y, seed: int):
    if strategy in ("none", "class_weight"):
        return X, y
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed).fit_resample(X, y)
    if strategy == "smote":
        return SMOTE(sampling_strategy=0.1, random_state=seed).fit_resample(X, y)
    if strategy == "smote_under":
        X_s, y_s = SMOTE(sampling_strategy=0.1, random_state=seed).fit_resample(X, y)
        return RandomUnderSampler(sampling_strategy=0.5, random_state=seed).fit_resample(X_s, y_s)
    raise ValueError(strategy)


# ── model builders (with/without class weighting) ─────────────────────────────

def build_model(model_name: str, strategy: str, seed: int):
    weighted = strategy == "class_weight"
    if model_name == "rf":
        cfg = MODEL_CONFIG["rf"]
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"] if weighted else None,
            random_state=seed, n_jobs=-1,
        )
    if model_name == "lgbm":
        cfg = MODEL_CONFIG["lgbm"]
        return LGBMClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
            class_weight=cfg["class_weight"] if weighted else None,
            early_stopping_round=cfg["early_stopping_round"],
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    raise ValueError(model_name)


# ── run one cell ──────────────────────────────────────────────────────────────

def run_one(ds_id, model_name, strategy, data, seed) -> dict:
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    t0 = time.time()

    X_rs, y_rs = resample(strategy, X_tr, y_tr, seed)
    resample_s = time.time() - t0

    model = build_model(model_name, strategy, seed)
    if model_name == "lgbm":
        model.fit(X_rs, y_rs, eval_set=[(data["X_val"], data["y_val"])])
    else:
        model.fit(X_rs, y_rs)
    elapsed = time.time() - t0

    prob_te  = model.predict_proba(data["X_te"])[:, 1]
    prob_val = model.predict_proba(data["X_val"])[:, 1]

    m_fixed = compute_metrics(data["y_te"], (prob_te >= 0.5).astype(int), prob_te)
    thr = tune_threshold(data["y_val"], prob_val)
    m_tuned = compute_metrics(data["y_te"], (prob_te >= thr).astype(int), prob_te)

    return {
        "dataset": ds_id, "model": model_name, "strategy": strategy,
        "n_train_orig": len(X_tr), "n_train_resampled": len(X_rs),
        "fraud_rate_resampled": float(np.mean(y_rs)),
        "f1": m_fixed["f1"], "pr_auc": m_fixed["pr_auc"],
        "roc_auc": m_fixed["roc_auc"],
        "precision": m_fixed["precision"], "recall": m_fixed["recall"],
        "threshold": thr, "f1_tuned": m_tuned["f1"],
        "precision_tuned": m_tuned["precision"], "recall_tuned": m_tuned["recall"],
        "resample_s": round(resample_s, 2), "elapsed_s": round(elapsed, 2),
    }


# ── data prep (once per dataset) ──────────────────────────────────────────────

def prepare_dataset(ds_id: str, seed: int) -> dict:
    cfg = DATASET_CONFIG[ds_id]
    X, y, feature_types = load_dataset(ds_id, cfg)
    X_tr_df, X_val_df, X_te_df, y_tr, y_val, y_te = stratified_split(X, y, seed=seed)

    prep = DatasetPreprocessor(
        feature_types, ds_id,
        missing_threshold=cfg["missing_threshold"],
        ohe_max_categories=cfg["ohe_max_categories"],
        scale_method=cfg["scale_method"],
    )
    X_tr  = prep.fit_transform(X_tr_df)
    X_val = prep.transform(X_val_df)
    X_te  = prep.transform(X_te_df)

    # Cap large training sets (same caps as baseline; use rf cap as the common cap)
    cap = cfg.get("rf_max_train")
    X_tr, y_tr = subsample(X_tr, y_tr, cap, seed)

    print(f"  [prep] {ds_id}: train={len(X_tr):,}  features={X_tr.shape[1]}")
    return {"X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "y_val": y_val,
            "X_te": X_te, "y_te": y_te}


# ── results I/O ───────────────────────────────────────────────────────────────

def _key(r): return (r["dataset"], r["model"], r["strategy"])


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ── figures ───────────────────────────────────────────────────────────────────

def make_figures(results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty:
        return

    for metric, fname in (("f1_tuned", "imbalance_f1.png"),
                          ("pr_auc",   "imbalance_prauc.png")):
        fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.5),
                                 sharey=True)
        for ax, model in zip(np.atleast_1d(axes), MODELS):
            sub = df[df["model"] == model]
            x = np.arange(len(STRATEGIES))
            width = 0.8 / len(DATASETS)
            palette = sns.color_palette("Set2", len(DATASETS))
            for i, (ds, color) in enumerate(zip(DATASETS, palette)):
                vals = [sub[(sub["dataset"] == ds) & (sub["strategy"] == s)][metric].mean()
                        for s in STRATEGIES]
                ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                       label=ds, color=color, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(STRATEGIES, rotation=20, ha="right", fontsize=8)
            ax.set_title(model.upper())
            ax.grid(True, axis="y", alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel(
            "Tuned F1" if metric == "f1_tuned" else "PR-AUC")
        np.atleast_1d(axes)[-1].legend(title="Dataset", fontsize=8)
        plt.tight_layout()
        out = OUT_DIR / fname
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out.name}")


def print_summary(results: list[dict]):
    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty:
        print("No results.")
        return
    for metric in ("f1", "f1_tuned", "pr_auc"):
        print(f"\n=== {metric} ===")
        pivot = df.pivot_table(index=["model", "dataset"], columns="strategy",
                               values=metric)
        print(pivot.round(4).to_string())


# ── main ──────────────────────────────────────────────────────────────────────

def run_imbalance(datasets=None, seed: int = 42, resume: bool = True):
    datasets = datasets or DATASETS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 10: Imbalance Strategy Study ===")
    print(f"Datasets   : {datasets}")
    print(f"Models     : {MODELS}")
    print(f"Strategies : {STRATEGIES}")
    print(f"Seed       : {seed}\n")

    results = load_results() if resume else []
    done = {_key(r) for r in results}
    print(f"  [resume] {len(results)} results on disk\n")

    for ds_id in datasets:
        todo = [(m, s) for m in MODELS for s in STRATEGIES
                if (ds_id, m, s) not in done]
        if not todo:
            continue
        data = prepare_dataset(ds_id, seed)
        for model_name, strategy in todo:
            print(f"  {ds_id} [{model_name}] {strategy} ...", end=" ", flush=True)
            try:
                r = run_one(ds_id, model_name, strategy, data, seed)
                print(f"F1={r['f1']:.4f} tuned={r['f1_tuned']:.4f} "
                      f"PR-AUC={r['pr_auc']:.4f} ({r['elapsed_s']:.0f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                r = {"dataset": ds_id, "model": model_name, "strategy": strategy,
                     "error": str(e), "f1": 0.0, "f1_tuned": 0.0, "pr_auc": 0.0}
            results.append(r)
            done.add(_key(r))
            save_results(results)

    save_results(results)
    print(f"\nResults → {RESULTS_FILE}  ({len(results)} records)")
    print("\n[Figures]")
    make_figures(results)
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 10 imbalance study")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_imbalance(datasets=args.datasets, seed=args.seed, resume=not args.no_resume)
