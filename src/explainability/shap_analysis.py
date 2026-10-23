"""SHAP-based explainability for fraud detection models."""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def compute_shap_values(model, X: pd.DataFrame, model_type: str = "tree") -> np.ndarray:
    """Compute SHAP values for a fitted model.

    Args:
        model:      Fitted model (sklearn tree-based or neural).
        X:          Feature DataFrame (sample for explanation).
        model_type: 'tree' for RF/XGB/LGBM, 'linear' for LR, 'deep' for PyTorch.

    Returns:
        SHAP values array of shape (n_samples, n_features).
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, take class-1 values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
    elif model_type == "deep":
        background = shap.sample(X, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X, nsamples=200)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    dataset_name: str,
    output_dir: str = "results/figures",
    top_n: int = 20,
) -> None:
    """Save a SHAP beeswarm summary plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=top_n, show=False)
    plt.tight_layout()
    fig.savefig(output_dir / f"shap_summary_{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def get_feature_importance(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Return mean absolute SHAP values as a ranked importance DataFrame."""
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
