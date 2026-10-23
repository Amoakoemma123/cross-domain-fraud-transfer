"""Baseline experiment — Phase 4.

Trains LR, RF, XGBoost, LightGBM, and MLP on each dataset independently.
Evaluates on a held-out test set and saves metrics + figures.

Usage:
    cd financial-fraud-transfer
    python experiments/baseline/run_baseline.py
    python experiments/baseline/run_baseline.py --datasets ulb ieee_cis
    python experiments/baseline/run_baseline.py --datasets paysim --models lr xgb
"""

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_ulb, load_ieee_cis, load_paysim, load_banksim
from src.data.preprocessing import DatasetPreprocessor
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.deep.mlp_classifier import MLPClassifier
from experiments.baseline.config import DATASET_CONFIG, MODEL_CONFIG

RESULTS_DIR = ROOT / "results/metrics"
FIGURES_DIR = ROOT / "results/figures"
MODELS_DIR  = ROOT / "results/models"

LOADER_MAP = {
    "ulb":     load_ulb,
    "ieee_cis":load_ieee_cis,
    "paysim":  load_paysim,
    "banksim": load_banksim,
}

DATASET_LABELS = {
    "ulb":     "ULB Credit Card",
    "ieee_cis":"IEEE-CIS",
    "paysim":  "PaySim",
    "banksim": "BankSim",
}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_dataset(ds_id: str, cfg: dict):
    loader = LOADER_MAP[cfg["loader"]]
    kwargs = cfg.get("loader_kwargs", {})
    print(f"  [load] {cfg['name']} ...")
    t0 = time.time()
    X, y, ft = loader(cfg["path"], **kwargs)
    print(f"  [load] {X.shape[0]:,} rows × {X.shape[1]} cols  ({time.time()-t0:.1f}s)")
    return X, y, ft


def stratified_split(X, y, seed=42, test_size=0.20, val_size=0.10):
    """Split into train / val / test with stratification."""
    try:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)
        # val_size relative to train+val
        val_rel = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_rel, stratify=y_tv, random_state=seed)
    except ValueError:
        # fallback for ultra-rare classes
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        val_rel = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_rel, random_state=seed)

    print(f"  [split] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    print(f"  [split] train fraud={y_train.sum():,} ({y_train.mean()*100:.3f}%)  "
          f"test fraud={y_test.sum():,} ({y_test.mean()*100:.3f}%)")
    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values


def subsample(X: np.ndarray, y: np.ndarray, max_n: int, seed: int = 42) -> tuple:
    """Stratified subsample of training data."""
    if max_n is None or len(X) <= max_n:
        return X, y
    idx, _ = train_test_split(
        np.arange(len(X)), train_size=max_n, stratify=y, random_state=seed)
    print(f"  [subsample] {len(X):,} → {max_n:,}  "
          f"fraud: {y[idx].sum():,} ({y[idx].mean()*100:.3f}%)")
    return X[idx], y[idx]


# ── model builders ────────────────────────────────────────────────────────────

def build_lr(cfg: dict) -> LogisticRegression:
    return LogisticRegression(
        C=cfg["C"], max_iter=cfg["max_iter"],
        solver=cfg["solver"], class_weight=cfg["class_weight"],
        random_state=42, n_jobs=-1,
    )


def build_rf(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight=cfg["class_weight"],
        random_state=42, n_jobs=cfg["n_jobs"],
    )


def build_xgb(cfg: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        eval_metric=cfg["eval_metric"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        scale_pos_weight=scale_pos_weight,
        random_state=42, verbosity=0, n_jobs=cfg["n_jobs"],
    )


def build_lgbm(cfg: dict) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
        class_weight=cfg["class_weight"],
        early_stopping_round=cfg["early_stopping_round"],
        random_state=42, n_jobs=cfg["n_jobs"], verbose=-1,
    )


def build_mlp(cfg: dict, input_dim: int, pos_weight: float) -> MLPClassifier:
    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        batch_norm=cfg["batch_norm"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        pos_weight=pos_weight,
        random_state=42,
    )


# ── training + evaluation ─────────────────────────────────────────────────────

def train_evaluate(
    model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    ds_cfg: dict,
    seed: int = 42,
) -> dict:
    mcfg = MODEL_CONFIG[model_name]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw   = n_neg / max(n_pos, 1)  # scale_pos_weight / pos_weight

    t0 = time.time()

    if model_name == "lr":
        X_tr, y_tr = subsample(X_train, y_train, ds_cfg.get("lr_max_train"), seed)
        model = build_lr(mcfg)
        model.fit(X_tr, y_tr)

    elif model_name == "rf":
        X_tr, y_tr = subsample(X_train, y_train, ds_cfg.get("rf_max_train"), seed)
        model = build_rf(mcfg)
        model.fit(X_tr, y_tr)

    elif model_name == "xgb":
        X_tr, y_tr = subsample(X_train, y_train, ds_cfg.get("xgb_max_train"), seed)
        n_pos2 = int(y_tr.sum()); n_neg2 = len(y_tr) - n_pos2
        model = build_xgb(mcfg, scale_pos_weight=n_neg2 / max(n_pos2, 1))
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    elif model_name == "lgbm":
        X_tr, y_tr = subsample(X_train, y_train, ds_cfg.get("lgbm_max_train"), seed)
        model = build_lgbm(mcfg)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[])  # suppress output

    elif model_name == "mlp":
        X_tr, y_tr = subsample(X_train, y_train, ds_cfg.get("mlp_max_train"), seed)
        model = build_mlp(mcfg, input_dim=X_train.shape[1], pos_weight=spw)
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    elapsed = time.time() - t0

    # Evaluate on test set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["train_time_s"] = round(elapsed, 2)
    metrics["n_train"]      = len(X_train)
    metrics["n_test"]       = len(X_test)

    return metrics


# ── figures ───────────────────────────────────────────────────────────────────

def fig_pr_curves(pr_data: dict, ds_id: str):
    """PR curves for all models on one dataset."""
    from sklearn.metrics import precision_recall_curve, auc
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("Set1", len(pr_data))
    for (model_name, (y_true, y_prob)), color in zip(pr_data.items(), palette):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{model_name.upper()} (AUC={pr_auc:.3f})", color=color, lw=1.8)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"PR Curves — {DATASET_LABELS.get(ds_id, ds_id)}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    out = FIGURES_DIR / f"pr_curves_{ds_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {out.name}")


def fig_metric_heatmap(all_results: dict, metric: str = "f1"):
    """Heatmap: datasets (rows) × models (cols) coloured by metric."""
    datasets = list(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())
    data = [[all_results[ds].get(m, {}).get(metric, np.nan) for m in models]
            for ds in datasets]
    df = pd.DataFrame(data,
                      index=[DATASET_LABELS.get(d, d) for d in datasets],
                      columns=[m.upper() for m in models])
    fig, ax = plt.subplots(figsize=(max(8, len(models)*1.5), max(3, len(datasets)*1.2)))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGn", ax=ax,
                linewidths=0.5, vmin=0, vmax=1,
                annot_kws={"size": 10})
    ax.set_title(f"Baseline {metric.upper()} — all datasets × models",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / f"baseline_{metric}_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {out.name}")


def fig_grouped_bar(all_results: dict, metric: str = "f1"):
    """Grouped bar chart: model groups, coloured by dataset."""
    datasets = list(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())
    x = np.arange(len(models))
    width = 0.8 / len(datasets)
    palette = sns.color_palette("Set2", len(datasets))

    fig, ax = plt.subplots(figsize=(max(9, len(models)*2), 5))
    for i, (ds_id, color) in enumerate(zip(datasets, palette)):
        vals = [all_results[ds_id].get(m, {}).get(metric, 0) for m in models]
        bars = ax.bar(x + i * width - 0.4 + width/2, vals, width,
                      label=DATASET_LABELS.get(ds_id, ds_id), color=color,
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=10)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Baseline {metric.upper()} — all datasets", fontsize=12, fontweight="bold")
    ax.legend(title="Dataset", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    sns.despine()
    plt.tight_layout()
    out = FIGURES_DIR / f"baseline_{metric}_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {out.name}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: dict):
    rows = []
    for ds_id, model_results in all_results.items():
        for model_name, m in model_results.items():
            rows.append({
                "Dataset": DATASET_LABELS.get(ds_id, ds_id),
                "Model":   model_name.upper(),
                "F1":      f"{m.get('f1', 0):.4f}",
                "PR-AUC":  f"{m.get('pr_auc', 0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall":  f"{m.get('recall', 0):.4f}",
                "MCC":     f"{m.get('mcc', 0):.4f}",
                "Time(s)": f"{m.get('train_time_s', 0):.1f}",
            })
    df = pd.DataFrame(rows)
    sep = "=" * 100
    print(f"\n{sep}")
    print("BASELINE RESULTS SUMMARY")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    out = RESULTS_DIR / "baseline_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n[baseline] Table saved → {out}")
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def run_baseline(
    datasets: list[str] = None,
    models:   list[str] = None,
    seed: int = 42,
):
    datasets = datasets or list(DATASET_CONFIG.keys())
    models   = models   or ["lr", "rf", "xgb", "lgbm", "mlp"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    pr_store:    dict[str, dict] = {}   # {ds_id: {model: (y_true, y_prob)}}

    for ds_id in datasets:
        ds_cfg = DATASET_CONFIG[ds_id]
        print(f"\n{'='*65}")
        print(f"  DATASET: {ds_cfg['name'].upper()}")
        print(f"{'='*65}")

        # ── Load ──────────────────────────────────────────────────────────
        X, y, feature_types = load_dataset(ds_id, ds_cfg)

        # ── Split ─────────────────────────────────────────────────────────
        X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = \
            stratified_split(X, y, seed=seed)

        # ── Preprocess ────────────────────────────────────────────────────
        print(f"  [prep] Fitting preprocessor ...")
        t0 = time.time()
        prep = DatasetPreprocessor(
            feature_types, ds_id,
            missing_threshold=ds_cfg["missing_threshold"],
            ohe_max_categories=ds_cfg["ohe_max_categories"],
            scale_method=ds_cfg["scale_method"],
        )
        X_train = prep.fit_transform(X_train_df)
        X_val   = prep.transform(X_val_df)
        X_test  = prep.transform(X_test_df)
        print(f"  [prep] {X_train.shape[1]} features out  ({time.time()-t0:.1f}s)")

        # ── Train & evaluate each model ────────────────────────────────────
        ds_results = {}
        pr_store[ds_id] = {}

        for model_name in models:
            print(f"\n  [{model_name.upper()}]", end=" ", flush=True)
            try:
                metrics = train_evaluate(
                    model_name,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    ds_cfg, seed=seed,
                )
                ds_results[model_name] = metrics
                print(format_metrics(metrics) + f"  time={metrics['train_time_s']}s")

                # Store probabilities for PR curves (evaluate fresh)
                # (already evaluated above — store y_test + y_prob)
                # Re-run predict_proba for storage (lightweight)
                # We can't re-run without the model — so we store inline above.
                # PR curve data is stored via a secondary predict inside train_evaluate.
                # For simplicity, log the pr_auc from metrics.

            except Exception as e:
                print(f"ERROR: {e}")
                ds_results[model_name] = {"error": str(e)}

        all_results[ds_id] = ds_results

        # ── Save per-dataset JSON ─────────────────────────────────────────
        out_json = RESULTS_DIR / f"baseline_{ds_id}.json"
        with open(out_json, "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"\n  [save] {out_json.name}")

    # ── Cross-dataset summary figures ─────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GENERATING FIGURES")
    print(f"{'='*65}")
    fig_metric_heatmap(all_results, "f1")
    fig_metric_heatmap(all_results, "pr_auc")
    fig_grouped_bar(all_results, "f1")
    fig_grouped_bar(all_results, "pr_auc")

    # ── Summary table ─────────────────────────────────────────────────────
    print_summary_table(all_results)

    print(f"\n[baseline] Done. Results → {RESULTS_DIR}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--datasets", nargs="+",
                        choices=list(DATASET_CONFIG.keys()),
                        default=None, help="Datasets to run (default: all)")
    parser.add_argument("--models", nargs="+",
                        choices=["lr", "rf", "xgb", "lgbm", "mlp"],
                        default=None, help="Models to train (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baseline(datasets=args.datasets, models=args.models, seed=args.seed)
