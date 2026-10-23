"""Performance comparison plots: PR curves, ROC curves, bar charts."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def plot_pr_curve(
    results: dict[str, tuple],
    title: str = "Precision-Recall Curves",
    output_path: str = None,
) -> None:
    """Plot PR curves for multiple models.

    Args:
        results: {model_name: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_true, y_prob) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(
    metrics_df,
    metric: str = "f1",
    title: str = None,
    output_path: str = None,
) -> None:
    """Grouped bar chart comparing a metric across models and datasets."""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} comparison")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Dataset", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
