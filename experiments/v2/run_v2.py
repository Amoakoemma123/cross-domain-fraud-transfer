"""v2 cross-domain benchmark under a label-budget-honest protocol.

Rules (the v1 runner broke all three):
  1. Target labels come ONLY from the budget: frac x |target train| rows,
     split 70/30 into a fit part and a hold-out part.  Training uses the fit
     part; early stopping and threshold selection use the hold-out part.
  2. Methods that use no target labels early-stop on SOURCE validation.
  3. The labelled target validation split is used only for the `oracle_val`
     threshold, reported as an upper bound and never for model selection.

Methods
  source-only / unsupervised DA (source feature space, `zero_fill`):
      no_adapt (RF), src_lgbm, coral_rf, coral_deep, mmd, dann
  supervised transfer (`union` space: source + target-only features):
      finetune, contrastive, tradaboost, pooled_lgbm
  target-only (target's native features, budget labels only):
      tgt_rf, tgt_lgbm, tgt_mlp
  v1-design controls (`zero_fill`, target-only features dropped; --v1-controls):
      finetune_zf, contrastive_zf

Usage
    python -m experiments.v2.run_v2 --seed 42 [--v1-controls]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.transfer.adapter import TransferAdapter
from src.models.transfer.encoder import FraudEncoder
from src.evaluation.thresholds import (
    label_free_policies, budget_policies, oracle_policy, _prf, f1_opt_threshold,
)
from experiments.cross_domain.feature_aligner import make_aligner
from experiments.transfer.config import ALL_PAIRS, DEEP_CONFIG, RF_CONFIG
from experiments.v2.prepare_data import load_cached, DATASETS

FRACS = (0.01, 0.05, 0.10, 0.20)
RANDOM_DRAWS = False         # --random-budget: draw budget and pool without looking at labels
BUDGET_REP = 0               # --budget-rep: index of an independent label-blind budget draw
MAX_SRC_TRAIN = 100_000     # source rows for training
MAX_SRC_VAL = 100_000       # source val rows (early stopping, calibration)
MAX_TGT_UNLAB = 200_000     # unlabelled target rows (DA training, EM pool)
SCORE_CHUNK = 100_000

UNSUP = ("no_adapt", "src_lgbm", "coral_rf", "coral_deep", "mmd", "dann")
SUP = ("finetune", "contrastive", "tradaboost", "pooled_lgbm")
TGT_ONLY = ("tgt_rf", "tgt_lgbm", "tgt_mlp")
V1_CONTROLS = ("finetune_zf", "contrastive_zf")

OUT_DIR = ROOT / "experiments" / "v2" / "results"
SCORE_DIR = ROOT / "experiments" / "v2" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                   class_weight="balanced", subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, n_jobs=-1, verbose=-1)


# ── sampling ──────────────────────────────────────────────────────────────────

def random_idx(n_rows: int, n: int, seed: int) -> np.ndarray:
    """Label-blind sample of n row indices (all rows if n >= n_rows)."""
    if n >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, n, replace=False))


def strat_idx(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Stratified sample of n row indices (all rows if n >= len)."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n,
                              stratify=y, random_state=seed)
    return np.sort(idx)


def budget_split(y_tr: np.ndarray, frac: float, seed: int):
    """Budget rows from target train, split 70/30 into fit / hold-out."""
    n = max(4, int(round(frac * len(y_tr))))
    bseed = seed * 1000 + int(round(frac * 1000)) + 7919 * BUDGET_REP
    idx = random_idx(len(y_tr), n, bseed) if RANDOM_DRAWS else strat_idx(y_tr, n, bseed)
    yb = y_tr[idx]
    strat = yb if np.bincount(yb, minlength=2).min() >= 2 else None
    fit, hold = train_test_split(idx, test_size=0.30, stratify=strat,
                                 random_state=bseed)
    return np.sort(fit), np.sort(hold)


# ── data ──────────────────────────────────────────────────────────────────────

class Data:
    def __init__(self, seed: int):
        self.seed = seed
        self.d = {}
        for ds in DATASETS:
            z = load_cached(ds, seed)
            z["y_tr"] = z["y_tr"].astype(np.int64)
            z["y_val"] = z["y_val"].astype(np.int64)
            z["y_te"] = z["y_te"].astype(np.int64)
            z["src_idx"] = strat_idx(z["y_tr"], MAX_SRC_TRAIN, seed)
            z["srcval_idx"] = strat_idx(z["y_val"], MAX_SRC_VAL, seed)
            z["pool_idx"] = (random_idx(len(z["y_tr"]), MAX_TGT_UNLAB, seed + 1) if RANDOM_DRAWS
                             else strat_idx(z["y_tr"], MAX_TGT_UNLAB, seed + 1))
            z["prior"] = float(z["y_tr"].mean())
            z["budgets"] = {f: budget_split(z["y_tr"], f, seed) for f in FRACS}
            self.d[ds] = z
            print(f"  [data] {ds}: train={len(z['y_tr']):,} test={len(z['y_te']):,} "
                  f"dim={z['X_tr'].shape[1]} prior={z['prior']:.5f}")

    def __getitem__(self, ds):
        return self.d[ds]


# ── scoring helpers ───────────────────────────────────────────────────────────

def score(predict, X: np.ndarray, align=None) -> np.ndarray:
    out = []
    for i in range(0, len(X), SCORE_CHUNK):
        xb = X[i:i + SCORE_CHUNK]
        out.append(predict(align(xb) if align else xb))
    return np.concatenate(out).astype(np.float64)


def evaluate(scores: dict, src, tgt, frac, label_free: bool) -> dict:
    """All threshold policies + threshold-free metrics for one model/frac."""
    y_te = tgt["y_te"]
    s_te = scores["te"]
    row = {"pr_auc": float(average_precision_score(y_te, s_te)),
           "roc_auc": float(roc_auc_score(y_te, s_te)),
           "policies": {}}
    pol = row["policies"]
    if label_free and src is not None:
        pol.update(label_free_policies(
            s_te, y_te, scores["src_val"], src["y_val"][src["srcval_idx"]],
            scores["pool"], tgt["prior"]))
    else:
        pol["fixed"] = {**_prf(y_te, s_te >= 0.5), "thr": 0.5}
    if frac > 0:
        _, hold = tgt["budgets"][frac]
        s_h, y_h = scores[f"hold_{frac}"], tgt["y_tr"][hold]
        pol.update(budget_policies(s_te, y_te, s_h, y_h, scores["pool"],
                                   fallback=pol.get("src_val", pol["fixed"])["thr"]))
        # hold-out evidence for budget-only model selection baselines
        mixed = 0 < y_h.sum() < len(y_h)
        row["hold"] = {"n": int(len(y_h)), "n_pos": int(y_h.sum()),
                       "pr_auc": float(average_precision_score(y_h, s_h)) if mixed else None,
                       "f1_opt": float(_prf(y_h, s_h >= f1_opt_threshold(y_h, s_h))["f1"]) if mixed else None}
    pol.update(oracle_policy(s_te, y_te, scores["val"], tgt["y_val"]))
    return row


# ── model builders ────────────────────────────────────────────────────────────

def fit_rf(X, y, seed):
    rf = RandomForestClassifier(n_estimators=RF_CONFIG["n_estimators"],
                                min_samples_leaf=RF_CONFIG["min_samples_leaf"],
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=seed)
    rf.fit(X, y)
    return lambda Z: rf.predict_proba(Z)[:, 1]


def fit_lgbm(X, y, X_es, y_es, seed, weight=None):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    m.fit(X, y, sample_weight=weight, eval_set=[(X_es, y_es)],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return lambda Z: m.predict_proba(Z)[:, 1]


def _helper(input_dim, seed):
    return TransferAdapter("finetune", input_dim, device=DEVICE, seed=seed, **DEEP_CONFIG)


def _pos_weight_loss(y):
    n_pos = int(y.sum())
    pw = torch.tensor([(len(y) - n_pos) / max(n_pos, 1)], device=DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def fit_mlp(X, y, X_es, y_es, seed, init: FraudEncoder | None = None,
            lr_mult=1.0, epochs=None, patience=None):
    """Train (or continue training) a FraudEncoder classifier."""
    torch.manual_seed(seed); np.random.seed(seed)
    h = _helper(X.shape[1], seed)
    model = copy.deepcopy(init) if init is not None else FraudEncoder(
        X.shape[1], DEEP_CONFIG["hidden_dims"], DEEP_CONFIG["repr_dim"],
        DEEP_CONFIG["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=DEEP_CONFIG["lr"] * lr_mult)
    h._train_classifier(model, opt, _pos_weight_loss(y), X, y, X_es, y_es,
                        epochs or DEEP_CONFIG["epochs"], patience=patience)
    return model


def mlp_predict(model):
    return lambda Z: model.predict_proba(Z, device=DEVICE)


# ── per-pair runs ─────────────────────────────────────────────────────────────

def run_unsup(method, s_id, t_id, data, seed):
    src, tgt = data[s_id], data[t_id]
    al = make_aligner("zero_fill", src["feature_names"], tgt["feature_names"])
    Xs = al.align_source(src["X_tr"][src["src_idx"]]); ys = src["y_tr"][src["src_idx"]]
    Xsv = al.align_source(src["X_val"][src["srcval_idx"]]); ysv = src["y_val"][src["srcval_idx"]]
    Xpool = al.align_target(tgt["X_tr"][tgt["pool_idx"]])

    if method == "src_lgbm":
        predict = fit_lgbm(Xs, ys, Xsv, ysv, seed)
    else:
        ad = TransferAdapter(method, al.n_features, device=DEVICE, seed=seed,
                             rf_n_estimators=RF_CONFIG["n_estimators"],
                             rf_min_samples_leaf=RF_CONFIG["min_samples_leaf"],
                             **DEEP_CONFIG)
        ad.fit(Xs, ys, X_tgt_unlab=Xpool, X_val=Xsv, y_val=ysv)
        predict = ad.predict_proba

    sc = {"src_val": score(predict, Xsv), "pool": score(predict, Xpool),
          "val": score(predict, tgt["X_val"], al.align_target),
          "te": score(predict, tgt["X_te"], al.align_target)}
    for f in FRACS:
        sc[f"hold_{f}"] = score(predict, tgt["X_tr"][tgt["budgets"][f][1]], al.align_target)
    rows = [{"frac": 0.0, **evaluate(sc, src, tgt, 0.0, True)}]
    rows += [{"frac": f, **evaluate(sc, src, tgt, f, True)} for f in FRACS]
    return rows, sc["te"]


def run_sup_pair(method, s_id, t_id, data, seed):
    """Supervised transfer across all budgets. Yields (frac, row, test scores)."""
    src, tgt = data[s_id], data[t_id]
    strategy = "zero_fill" if method.endswith("_zf") else "union"
    base = method.removesuffix("_zf")
    al = make_aligner(strategy, src["feature_names"], tgt["feature_names"])
    Xs = al.align_source(src["X_tr"][src["src_idx"]]); ys = src["y_tr"][src["src_idx"]]
    Xsv = al.align_source(src["X_val"][src["srcval_idx"]]); ysv = src["y_val"][src["srcval_idx"]]
    Xpool = al.align_target(tgt["X_tr"][tgt["pool_idx"]])

    pre = None
    if base == "finetune":  # source phase is budget-independent: train once
        pre = fit_mlp(Xs, ys, Xsv, ysv, seed)

    for f in FRACS:
        fit, hold = tgt["budgets"][f]
        Xf, yf = al.align_target(tgt["X_tr"][fit]), tgt["y_tr"][fit]
        Xh, yh = al.align_target(tgt["X_tr"][hold]), tgt["y_tr"][hold]
        t0 = time.time()
        if base == "finetune":
            # same schedule as tgt_mlp: only the initialisation differs
            model = fit_mlp(Xf, yf, Xh, yh, seed, init=pre)
            predict = mlp_predict(model)
        elif base == "pooled_lgbm":
            w = np.concatenate([np.ones(len(ys)), np.full(len(yf), max(1.0, len(ys) / len(yf)))])
            predict = fit_lgbm(np.vstack([Xs, Xf]), np.concatenate([ys, yf]), Xh, yh, seed, w)
        else:  # contrastive, tradaboost
            ad = TransferAdapter(base, al.n_features, device=DEVICE, seed=seed, **DEEP_CONFIG)
            ad.fit(Xs, ys, X_tgt_unlab=Xpool, X_tgt_lab=Xf, y_tgt_lab=yf,
                   X_val=Xh, y_val=yh)
            predict = ad.predict_proba
        sc = {"pool": score(predict, Xpool), f"hold_{f}": score(predict, Xh),
              "val": score(predict, tgt["X_val"], al.align_target),
              "te": score(predict, tgt["X_te"], al.align_target)}
        row = {"frac": f, **evaluate(sc, None, tgt, f, False),
               "elapsed_s": round(time.time() - t0, 1)}
        yield f, row, sc["te"]


def run_tgt_only(method, t_id, data, seed):
    tgt = data[t_id]
    Xpool = tgt["X_tr"][tgt["pool_idx"]]
    for f in FRACS:
        fit, hold = tgt["budgets"][f]
        Xf, yf = tgt["X_tr"][fit], tgt["y_tr"][fit]
        Xh, yh = tgt["X_tr"][hold], tgt["y_tr"][hold]
        t0 = time.time()
        if yf.sum() == 0:
            predict = lambda Z: np.zeros(len(Z))
        elif method == "tgt_rf":
            predict = fit_rf(Xf, yf, seed)
        elif method == "tgt_lgbm":
            predict = fit_lgbm(Xf, yf, Xh, yh, seed)
        else:
            predict = mlp_predict(fit_mlp(Xf, yf, Xh, yh, seed))
        sc = {"pool": score(predict, Xpool), f"hold_{f}": score(predict, Xh),
              "val": score(predict, tgt["X_val"]), "te": score(predict, tgt["X_te"])}
        row = {"frac": f, **evaluate(sc, None, tgt, f, False),
               "elapsed_s": round(time.time() - t0, 1)}
        yield f, row, sc["te"]


# ── driver ────────────────────────────────────────────────────────────────────

def main(seed: int, methods: list[str], pairs: list[tuple]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"_randbudget_r{BUDGET_REP}" if RANDOM_DRAWS else "")
    sdir = SCORE_DIR / f"seed{seed}{tag}"
    sdir.mkdir(parents=True, exist_ok=True)
    res_file = OUT_DIR / f"results_seed{seed}{tag}.json"
    results = json.load(open(res_file)) if res_file.exists() else []
    done = {(r["src"], r["tgt"], r["method"], r["frac"]) for r in results}

    def save():
        tmp = res_file.with_suffix(".tmp")
        json.dump(results, open(tmp, "w"), indent=1)
        tmp.replace(res_file)

    print(f"\n=== v2 benchmark  seed={seed}  device={DEVICE} ===")
    data = Data(seed)
    for ds in DATASETS:
        p = sdir / f"_truth_{ds}.npz"
        if not p.exists():
            np.savez_compressed(p, y_te=data[ds]["y_te"].astype(np.int8),
                                amt_te=data[ds]["amt_te"])

    def emit(s_id, t_id, method, f, row, s_te, save_scores=True):
        results.append({"seed": seed, "src": s_id, "tgt": t_id, "method": method, **row})
        done.add((s_id, t_id, method, f))
        if save_scores:
            np.savez_compressed(sdir / f"{s_id}_to_{t_id}_{method}_{int(f*100):03d}.npz",
                                s_te=s_te.astype(np.float32))
        fx = row["policies"]["fixed"]["f1"]
        bud = row["policies"].get("budget_f1", row["policies"].get("em_plugin", {})).get("f1", float("nan"))
        orc = row["policies"]["oracle_val"]["f1"]
        print(f"    {s_id}->{t_id} [{method}] {f:.0%}: PR-AUC={row['pr_auc']:.3f} "
              f"F1 fixed={fx:.3f} honest={bud:.3f} oracle={orc:.3f}", flush=True)

    for method in methods:
        if method in TGT_ONLY:
            for t_id in sorted({t for _, t in pairs}):
                if all(("none", t_id, method, f) in done for f in FRACS):
                    continue
                try:
                    for f, row, s_te in run_tgt_only(method, t_id, data, seed):
                        emit("none", t_id, method, f, row, s_te)
                except Exception as e:  # e.g. no fraud case in the fit part of a label-blind budget
                    print(f"    ERROR none->{t_id} [{method}]: {e!r}", flush=True)
                    results.append({"seed": seed, "src": "none", "tgt": t_id,
                                    "method": method, "frac": -1, "error": repr(e)})
                save()
            continue
        for s_id, t_id in pairs:
            if (s_id, t_id, method, FRACS[-1]) in done:
                continue
            t0 = time.time()
            try:
                if method in UNSUP:
                    rows, s_te = run_unsup(method, s_id, t_id, data, seed)
                    for r in rows:
                        r["elapsed_s"] = round(time.time() - t0, 1)
                        emit(s_id, t_id, method, r["frac"], r, s_te,
                             save_scores=r["frac"] == 0.0)  # model is budget-independent
                else:
                    for f, row, s_te in run_sup_pair(method, s_id, t_id, data, seed):
                        if (s_id, t_id, method, f) not in done:
                            emit(s_id, t_id, method, f, row, s_te)
            except Exception as e:  # record and continue
                print(f"    ERROR {s_id}->{t_id} [{method}]: {e!r}", flush=True)
                results.append({"seed": seed, "src": s_id, "tgt": t_id,
                                "method": method, "frac": -1, "error": repr(e)})
            save()
    print(f"\nSaved {len(results)} rows -> {res_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--v1-controls", action="store_true")
    ap.add_argument("--src"); ap.add_argument("--tgt")
    ap.add_argument("--fracs", nargs="+", type=float, default=None)
    ap.add_argument("--random-budget", action="store_true",
                    help="draw budget and unlabelled pool label-blind; writes results_seed<s>_randbudget_r<k>.json")
    ap.add_argument("--budget-rep", type=int, default=0)
    a = ap.parse_args()
    if a.fracs:
        FRACS = tuple(a.fracs)
    RANDOM_DRAWS = a.random_budget
    BUDGET_REP = a.budget_rep
    methods = a.methods or list(TGT_ONLY + UNSUP + SUP) + (list(V1_CONTROLS) if a.v1_controls else [])
    pairs = ([(a.src, a.tgt)] if a.src else
             [(s, t) for s, t in ALL_PAIRS if t == a.tgt] if a.tgt else ALL_PAIRS)
    main(a.seed, methods, pairs)
