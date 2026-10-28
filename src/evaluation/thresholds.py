"""Decision-threshold policies for transferred fraud scorers (v2 protocol).

Every policy maps a model's raw target scores to hard decisions using only
the information its name says it uses:

  fixed            0.5 on the raw score                       (no labels)
  src_val          F1-optimal threshold on SOURCE validation  (no target labels)
  em_plugin        Platt on source val -> Saerens EM target prior estimate on
                   unlabelled target -> plug-in expected-F1 threshold
                                                              (no target labels)
  bbse_plugin      as em_plugin, prior estimated by BBSE      (no target labels)
  prior_plugin     as em_plugin, true target prior supplied   (base rate only)
  budget_f1        F1-optimal on the held-out part of the labelled budget
  budget_plugin    Platt on budget hold-out -> plug-in threshold on unlabelled
                   target                                     (budget labels)
  oracle_val       F1-optimal on the full labelled target validation split
                   (upper bound, NOT a deployable policy)

References: Saerens et al. (2002) Neural Computation 14(1); Lipton et al.
(2018) ICML (BBSE); Lipton et al. (2014) / Nan et al. (2012) for plug-in F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

_EPS = 1e-6


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(s / (1 - s))


def f1_opt_threshold(y: np.ndarray, s: np.ndarray, fallback: float = 0.5) -> float:
    """Threshold maximising F1 on (y, s); fallback when y has no positives."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return fallback
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


class Platt:
    """1-D logistic calibration on the score logit."""

    def __init__(self):
        self.lr = None

    def fit(self, y: np.ndarray, s: np.ndarray) -> "Platt | None":
        y = np.asarray(y)
        if y.sum() == 0 or y.sum() == len(y):
            return None
        self.lr = LogisticRegression(C=1e4, max_iter=1000)
        self.lr.fit(_logit(s)[:, None], y)
        if self.lr.coef_[0, 0] <= 0:  # scores anti-correlated with labels
            return None
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(s)[:, None])[:, 1]


def adjust_prior(p: np.ndarray, pi_s: float, pi_t: float) -> np.ndarray:
    """Bayes prior-shift correction of calibrated posteriors (Saerens et al.)."""
    a = (pi_t / pi_s) * p
    b = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
    return a / np.maximum(a + b, 1e-300)


def em_prior(p: np.ndarray, pi_s: float, max_iter: int = 200, tol: float = 1e-7) -> float:
    """Saerens-Latinne-Decaestecker EM estimate of the target positive prior."""
    pi = pi_s
    for _ in range(max_iter):
        new = float(adjust_prior(p, pi_s, pi).mean())
        new = min(max(new, 1e-6), 1 - 1e-6)
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def bbse_prior(y_src: np.ndarray, s_src: np.ndarray, s_tgt: np.ndarray,
               thr: float) -> float:
    """Black-box shift estimation (binary) with a hard classifier at thr."""
    yhat = s_src >= thr
    tpr = yhat[y_src == 1].mean() if (y_src == 1).any() else 0.0
    fpr = yhat[y_src == 0].mean() if (y_src == 0).any() else 0.0
    q = float((s_tgt >= thr).mean())
    if tpr - fpr <= 1e-6:
        return float("nan")
    return float(np.clip((q - fpr) / (tpr - fpr), 1e-6, 0.5))


def plugin_f1_threshold(p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising expected F1 when p are calibrated posteriors.

    Predicting the top-k rows positive gives E[TP] = sum of their p and
    E[F1] ~= 2 E[TP] / (k + sum(p)).  Returns (threshold, estimated F1).
    """
    ps = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    tp = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    f1 = 2 * tp / (k + ps.sum())
    i = int(np.argmax(f1))
    return float(ps[i]), float(f1[i])


def _prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def label_free_policies(s_te, y_te, s_src_val, y_src_val, s_pool, true_prior) -> dict:
    """Policies needing no target labels. Returns {policy: metrics}."""
    s_te = np.asarray(s_te); y_te = np.asarray(y_te)
    out = {}
    out["fixed"] = {**_prf(y_te, s_te >= 0.5), "thr": 0.5}

    t_src = f1_opt_threshold(y_src_val, s_src_val)
    out["src_val"] = {**_prf(y_te, s_te >= t_src), "thr": t_src}

    platt = Platt().fit(y_src_val, s_src_val)
    if platt is None:
        return out
    pi_s = float(np.mean(y_src_val))
    p_pool, p_te = platt(s_pool), platt(s_te)
    priors = {
        "em_plugin": em_prior(p_pool, pi_s),
        "bbse_plugin": bbse_prior(y_src_val, s_src_val, s_pool, t_src),
        "prior_plugin": true_prior,
    }
    for name, pi_t in priors.items():
        if not np.isfinite(pi_t):
            continue
        thr, f1_hat = plugin_f1_threshold(adjust_prior(p_pool, pi_s, pi_t))
        pred = adjust_prior(p_te, pi_s, pi_t) >= thr
        out[name] = {**_prf(y_te, pred), "thr": thr, "prior_est": pi_t, "f1_est": f1_hat}
    return out


def budget_policies(s_te, y_te, s_hold, y_hold, s_pool, fallback: float) -> dict:
    """Policies spending the held-out part of the labelled target budget."""
    s_te = np.asarray(s_te); y_te = np.asarray(y_te)
    out = {}
    t = f1_opt_threshold(y_hold, s_hold, fallback=fallback)
    out["budget_f1"] = {**_prf(y_te, s_te >= t), "thr": t,
                        "n_hold_pos": int(np.sum(y_hold))}
    platt = Platt().fit(y_hold, s_hold)
    if platt is not None:
        thr, f1_hat = plugin_f1_threshold(platt(s_pool))
        out["budget_plugin"] = {**_prf(y_te, platt(s_te) >= thr), "thr": thr,
                                "f1_est": f1_hat}
    return out


def oracle_policy(s_te, y_te, s_val, y_val) -> dict:
    t = f1_opt_threshold(y_val, s_val)
    return {"oracle_val": {**_prf(np.asarray(y_te), np.asarray(s_te) >= t), "thr": t}}
