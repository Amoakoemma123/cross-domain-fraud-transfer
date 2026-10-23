"""TrAdaBoost — instance-based transfer boosting (Dai et al., ICML 2007).

Binary-classification variant. Trains base learners on combined
(source + labeled target) data. Each round:
  * source instances misclassified by the current learner are DOWN-weighted
    (fixed rate beta = 1 / (1 + sqrt(2 ln n_src / N)))
  * target instances misclassified are UP-weighted (AdaBoost rate beta_t)
Error is measured on the labeled target only. Prediction uses the ensemble
of the second half of the learners, weighted by ln(1/beta_t).

Reference: Dai, Yang, Xue, Yu. "Boosting for Transfer Learning." ICML 2007.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class TrAdaBoost:
    """TrAdaBoost for binary classification with decision-tree base learners."""

    def __init__(
        self,
        n_iters: int = 20,
        max_depth: int = 6,
        min_samples_leaf: int = 4,
        seed: int = 42,
    ):
        self.n_iters = n_iters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self._learners: list[DecisionTreeClassifier] = []
        self._betas_t: list[float] = []

    def fit(self, X_src, y_src, X_tgt, y_tgt) -> "TrAdaBoost":
        X_src = np.asarray(X_src, dtype=np.float32)
        X_tgt = np.asarray(X_tgt, dtype=np.float32)
        y_src = np.asarray(y_src, dtype=np.int64)
        y_tgt = np.asarray(y_tgt, dtype=np.int64)

        n_src, n_tgt = len(X_src), len(X_tgt)
        X = np.vstack([X_src, X_tgt])
        y = np.concatenate([y_src, y_tgt])

        w = np.ones(n_src + n_tgt)
        # Fixed source down-weighting rate (Dai et al. eq. for beta)
        beta_src = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(n_src, 2)) / self.n_iters))

        self._learners, self._betas_t = [], []
        for t in range(self.n_iters):
            p = w / w.sum()
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.seed + t,
            )
            clf.fit(X, y, sample_weight=p)
            pred = clf.predict(X)
            err_vec = (pred != y).astype(float)

            # Weighted error on the labeled TARGET portion only
            w_tgt = w[n_src:]
            eps_t = float((w_tgt * err_vec[n_src:]).sum() / max(w_tgt.sum(), 1e-12))
            eps_t = float(np.clip(eps_t, 1e-10, 0.499))
            beta_t = eps_t / (1.0 - eps_t)

            self._learners.append(clf)
            self._betas_t.append(beta_t)

            # Source: down-weight misclassified; Target: up-weight misclassified
            w[:n_src] *= beta_src ** err_vec[:n_src]
            w[n_src:] *= beta_t ** (-err_vec[n_src:])

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return (N,) positive-class probabilities from the second-half ensemble."""
        X = np.asarray(X, dtype=np.float32)
        start = int(np.ceil(self.n_iters / 2))
        alphas, probs = [], []
        for clf, beta_t in zip(self._learners[start:], self._betas_t[start:]):
            alpha = np.log(1.0 / max(beta_t, 1e-12))
            pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else None
            p = clf.predict_proba(X)[:, pos_col] if pos_col is not None else np.zeros(len(X))
            alphas.append(alpha)
            probs.append(p)
        alphas = np.asarray(alphas)
        probs = np.asarray(probs)
        return (alphas[:, None] * probs).sum(axis=0) / max(alphas.sum(), 1e-12)
