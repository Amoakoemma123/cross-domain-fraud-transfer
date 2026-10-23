"""Probability calibration analysis."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_report(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute calibration metrics.

    Returns:
        Dict with ECE, MCE, Brier score, and calibration curve arrays.
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = bin_sizes / bin_sizes.sum()
    ece = float(np.sum(weights * np.abs(fraction_of_positives - mean_predicted)))

    # Maximum Calibration Error
    mce = float(np.max(np.abs(fraction_of_positives - mean_predicted)))

    return {
        "ece": ece,
        "mce": mce,
        "brier": brier_score_loss(y_true, y_prob),
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }
