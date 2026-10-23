"""Dataset audit — Phase 2.

Loads all four datasets, prints audit reports, saves JSON summaries,
and generates figures:
  - fraud_prevalence.png
  - amount_distribution.png
  - missing_heatmap_<dataset>.png (IEEE-CIS only — high missing rate)
  - class_imbalance_summary.png

Usage:
    python experiments/baseline/run_audit.py
    python experiments/baseline/run_audit.py --datasets ulb paysim
"""

import sys
import json
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

warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_ulb, load_ieee_cis, load_paysim, load_banksim
from src.data.validation import audit_dataset, print_audit

# ── config ───────────────────────────────────────────────────────────────────
DATASET_LOADERS = {
    "ulb":     (load_ulb,      str(ROOT / "dataset/raw/ulb/creditcard.csv")),
    "ieee_cis":(load_ieee_cis, str(ROOT / "dataset/raw/ieee_cis")),
    "paysim":  (load_paysim,   str(ROOT / "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv")),
    "banksim": (load_banksim,  str(ROOT / "dataset/raw/banksim/bs140513_032310.csv")),
}

DATASET_LABELS = {
    "ulb":     "ULB Credit Card",
    "ieee_cis":"IEEE-CIS",
    "paysim":  "PaySim",
    "banksim": "BankSim",
}

FIGURES_DIR = ROOT / "results/figures"
METRICS_DIR = ROOT / "results/metrics"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset(dataset_id: str):
    loader_fn, path = DATASET_LOADERS[dataset_id]
    print(f"\n[audit] Loading {DATASET_LABELS[dataset_id]} ...")
    X, y, feature_types = loader_fn(path)
    print(f"[audit] Loaded: {X.shape[0]:,} rows × {X.shape[1]} cols")
    return X, y, feature_types


def save_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[audit] Report saved → {path}")


# ── figures ──────────────────────────────────────────────────────────────────

def fig_fraud_prevalence(reports: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = [DATASET_LABELS[k] for k in reports]
    rates  = [reports[k]["fraud_rate"] * 100 for k in reports]
    colors = sns.color_palette("Set2", len(names))

    bars = ax.bar(names, rates, color=colors, edgecolor="white", linewidth=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rates) * 0.01,
                f"{rate:.3f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Fraud rate (%)", fontsize=11)
    ax.set_title("Fraud Prevalence by Dataset", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f%%"))
    ax.set_ylim(0, max(rates) * 1.25)
    sns.despine()
    plt.tight_layout()

    out = FIGURES_DIR / "fraud_prevalence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[audit] Figure saved → {out}")


def fig_imbalance_ratio(reports: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = [DATASET_LABELS[k] for k in reports]
    ratios = [reports[k]["imbalance_ratio"] for k in reports]
    colors = sns.color_palette("Set1", len(names))

    bars = ax.barh(names, ratios, color=colors, edgecolor="white")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_width() + max(ratios) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"1 : {ratio:.0f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Legitimate : Fraud ratio", fontsize=11)
    ax.set_title("Class Imbalance Ratio by Dataset", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(ratios) * 1.15)
    sns.despine()
    plt.tight_layout()

    out = FIGURES_DIR / "class_imbalance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[audit] Figure saved → {out}")


def fig_amount_distributions(amount_data: dict[str, pd.Series]):
    """KDE of log-amount per dataset, all on one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    colors = sns.color_palette("Set2", 4)

    for ax, (ds_id, series), color in zip(axes, amount_data.items(), colors):
        log_vals = np.log1p(series.clip(lower=0))
        ax.hist(log_vals, bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(log_vals.median(), color="red", lw=1.5, ls="--", label=f"median={series.median():.2f}")
        ax.set_title(DATASET_LABELS[ds_id], fontsize=11, fontweight="bold")
        ax.set_xlabel("log(1 + amount)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle("Transaction Amount Distributions (log scale)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = FIGURES_DIR / "amount_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[audit] Figure saved → {out}")


def fig_missing_heatmap(X: pd.DataFrame, dataset_id: str, top_n: int = 40):
    """Missing-value heatmap for columns with any missingness (capped at top_n)."""
    missing = X.isnull().mean()
    missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
    if missing.empty:
        print(f"[audit] No missing values in {dataset_id} — skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(missing) * 0.35), 4))
    ax.bar(missing.index, missing.values * 100,
           color=sns.color_palette("Reds_r", len(missing)), edgecolor="none")
    ax.set_ylabel("Missing (%)", fontsize=10)
    ax.set_title(f"Missing Values — {DATASET_LABELS[dataset_id]} (top {len(missing)} cols)",
                 fontsize=12, fontweight="bold")
    ax.axhline(50, color="black", lw=1, ls="--", label="50% threshold")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    sns.despine()
    plt.tight_layout()

    out = FIGURES_DIR / f"missing_{dataset_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[audit] Figure saved → {out}")


def fig_fraud_amount_kde(amount_data: dict[str, tuple[pd.Series, pd.Series]]):
    """Fraud vs legitimate log-amount KDE per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (ds_id, (legit_amt, fraud_amt)) in zip(axes, amount_data.items()):
        log_legit = np.log1p(legit_amt.clip(lower=0))
        log_fraud = np.log1p(fraud_amt.clip(lower=0))

        sns.kdeplot(log_legit, ax=ax, label="Legitimate", fill=True, alpha=0.4, color="steelblue")
        if len(log_fraud) > 5:
            sns.kdeplot(log_fraud, ax=ax, label="Fraud", fill=True, alpha=0.5, color="crimson")

        ax.set_title(DATASET_LABELS[ds_id], fontsize=11, fontweight="bold")
        ax.set_xlabel("log(1 + amount)", fontsize=9)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle("Amount Distribution: Fraud vs Legitimate", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = FIGURES_DIR / "amount_fraud_vs_legit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[audit] Figure saved → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def run_audit(datasets: list[str] = None):
    datasets = datasets or list(DATASET_LOADERS.keys())
    reports = {}
    amount_data_hist = {}
    amount_data_kde  = {}

    for ds_id in datasets:
        X, y, feature_types = load_dataset(ds_id)
        report = audit_dataset(X, y, DATASET_LABELS[ds_id], feature_types)
        print_audit(report)
        reports[ds_id] = report
        save_report(report, METRICS_DIR / f"audit_{ds_id}.json")

        # Collect amount data for figures
        amt_col = report["amount_stats"].get("col")
        if amt_col and amt_col in X.columns:
            amount_data_hist[ds_id] = X[amt_col].dropna()
            amount_data_kde[ds_id]  = (
                X.loc[y == 0, amt_col].dropna(),
                X.loc[y == 1, amt_col].dropna(),
            )

        # Per-dataset missing heatmap
        fig_missing_heatmap(X, ds_id)

    # Cross-dataset figures
    if len(reports) > 1:
        print("\n[audit] Generating cross-dataset figures ...")
        fig_fraud_prevalence(reports)
        fig_imbalance_ratio(reports)

    if amount_data_hist:
        fig_amount_distributions(amount_data_hist)
        fig_fraud_amount_kde(amount_data_kde)

    # Combined summary table
    summary_rows = []
    for ds_id, r in reports.items():
        summary_rows.append({
            "Dataset":       r["dataset"],
            "Rows":          r["n_rows"],
            "Features":      r["n_cols"],
            "Fraud":         r["n_fraud"],
            "Fraud %":       f"{r['fraud_rate']*100:.4f}",
            "Imbalance 1:N": r["imbalance_ratio"],
            "Missing cols":  r["cols_with_missing"],
            "Max missing %": f"{r['max_missing_rate']*100:.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)

    out_csv = METRICS_DIR / "audit_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[audit] Summary saved → {out_csv}")
    print(f"[audit] Figures saved → {FIGURES_DIR}/")
    print("[audit] Done.")

    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset audit for all/selected datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=None,
        help="Datasets to audit (default: all)",
    )
    args = parser.parse_args()
    run_audit(args.datasets)
