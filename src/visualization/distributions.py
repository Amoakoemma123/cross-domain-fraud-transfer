"""Distribution comparison plots across datasets."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_feature_distributions(
    datasets: dict[str, pd.DataFrame],
    feature: str,
    output_dir: str = "results/figures",
) -> None:
    """Overlay feature distributions from multiple datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, X in datasets.items():
        if feature in X.columns:
            vals = X[feature].dropna()
            sns.kdeplot(vals, ax=ax, label=name, fill=True, alpha=0.25)

    ax.set_xlabel(feature)
    ax.set_title(f"Distribution of '{feature}' across datasets")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f"dist_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fraud_prevalence(prevalences: dict[str, float], output_dir: str = "results/figures") -> None:
    """Bar chart of fraud prevalence per dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(prevalences.keys())
    rates = [v * 100 for v in prevalences.values()]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, rates, color=sns.color_palette("Set2", len(names)))
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.3f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud prevalence by dataset")
    plt.tight_layout()
    fig.savefig(output_dir / "fraud_prevalence.png", dpi=150, bbox_inches="tight")
    plt.close()
