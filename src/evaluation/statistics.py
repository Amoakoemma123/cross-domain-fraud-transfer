"""Statistical testing for model comparison."""

import numpy as np
from scipy import stats
from typing import Optional


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iter: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Returns:
        (mean, lower, upper)
    """
    rng = np.random.RandomState(random_state)
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return float(np.mean(scores)), float(lower), float(upper)


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Wilcoxon signed-rank test between two sets of scores.

    Returns:
        Dict with statistic, p_value, and significant flag.
    """
    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {"statistic": stat, "p_value": p, "significant": p < 0.05}


def effect_size_cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between two score lists."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled_std + 1e-10))


def distribution_shift_metrics(
    X_source: np.ndarray, X_target: np.ndarray
) -> dict:
    """Quantify distribution shift between source and target.

    Returns per-feature KS statistic mean, JS divergence mean, and MMD.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.special import kl_div

    n_features = min(X_source.shape[1], X_target.shape[1])
    ks_stats = []
    js_divs = []

    for i in range(n_features):
        ks_stat, _ = stats.ks_2samp(X_source[:, i], X_target[:, i])
        ks_stats.append(ks_stat)

        # JS divergence via histograms
        bins = np.linspace(
            min(X_source[:, i].min(), X_target[:, i].min()),
            max(X_source[:, i].max(), X_target[:, i].max()),
            50,
        )
        p, _ = np.histogram(X_source[:, i], bins=bins, density=True)
        q, _ = np.histogram(X_target[:, i], bins=bins, density=True)
        p = p + 1e-10
        q = q + 1e-10
        js_divs.append(jensenshannon(p, q))

    return {
        "ks_mean": float(np.mean(ks_stats)),
        "ks_max": float(np.max(ks_stats)),
        "js_mean": float(np.mean(js_divs)),
        "js_max": float(np.max(js_divs)),
    }
