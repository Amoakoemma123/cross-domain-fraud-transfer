"""Cross-domain experiment: train on source dataset, evaluate on target (zero-shot).

Usage
-----
    cd E:/Research_Works/Fraud_Detection_Research/financial-fraud-transfer
    python -m experiments.cross_domain.run_cross_domain [--datasets ulb paysim ...] [--seed 42]

Outputs (under experiments/cross_domain/results/)
--------------------------------------------------
  cross_domain_results.json        — all metrics for every pair × strategy
  f1_matrix_{strategy}.png         — 4×4 heatmap (source rows, target columns)
  prauc_matrix_{strategy}.png      — same for PR-AUC
  degradation_{strategy}.png       — F1 drop vs in-domain baseline per target
  shift_metrics.json               — KS / JS divergence on shared features
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loaders.ulb      import load_ulb
from src.data.loaders.ieee_cis import load_ieee_cis
from src.data.loaders.paysim   import load_paysim
from src.data.loaders.banksim  import load_banksim
from src.data.preprocessing    import DatasetPreprocessor
from src.evaluation.metrics    import compute_metrics

from experiments.cross_domain.feature_aligner import make_aligner, canonicalize
from experiments.baseline.config import DATASET_CONFIG

# ── constants ─────────────────────────────────────────────────────────────────

DATASETS = ["ulb", "ieee_cis", "paysim", "banksim"]

STRATEGIES = ["semantic", "intersection", "zero_fill"]

# Models used for cross-domain (fast, robust to small feature sets)
# RF only for speed; LGBM added if lightgbm is installed
CROSS_DOMAIN_MODELS = ["rf"]

# Training-data cap for cross-domain source training
MAX_TRAIN = 200_000

OUT_DIR    = ROOT / "experiments" / "cross_domain" / "results"
MODELS_DIR = ROOT / "experiments" / "cross_domain" / "models"

# ── loaders ───────────────────────────────────────────────────────────────────

def _load_dataset(dataset_id: str):
    """Returns (X_df, y_series, feature_types)."""
    cfg = DATASET_CONFIG[dataset_id]
    raw_path = cfg["path"]

    if dataset_id == "ulb":
        return load_ulb(raw_path)
    elif dataset_id == "ieee_cis":
        return load_ieee_cis(raw_path, split="train")
    elif dataset_id == "paysim":
        return load_paysim(raw_path)
    elif dataset_id == "banksim":
        return load_banksim(raw_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_id}")


# ── preprocessing & caching ───────────────────────────────────────────────────

class _DataCache:
    """Load + preprocess each dataset once, cache in memory."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._cache: dict[str, dict] = {}

    def get(self, dataset_id: str) -> dict:
        if dataset_id not in self._cache:
            self._cache[dataset_id] = self._load_and_split(dataset_id)
        return self._cache[dataset_id]

    def _load_and_split(self, dataset_id: str) -> dict:
        print(f"  [load] {dataset_id} ...", end=" ", flush=True)
        t0 = time.time()
        X_df, y, feature_types = _load_dataset(dataset_id)
        print(f"{len(X_df):,} rows, {X_df.shape[1]} cols  ({time.time()-t0:.1f}s)")

        cfg = DATASET_CONFIG[dataset_id]

        # Stratified 72/8/20 split
        try:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_df, y, test_size=0.28, stratify=y, random_state=self._seed)
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.20/0.28, stratify=y_tmp,
                random_state=self._seed)
        except ValueError:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_df, y, test_size=0.28, random_state=self._seed)
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.20/0.28, random_state=self._seed)

        # Subsample training if too large
        if len(X_tr) > MAX_TRAIN:
            idx = _stratified_sample(y_tr, MAX_TRAIN, self._seed)
            X_tr, y_tr = X_tr.iloc[idx], y_tr.iloc[idx]

        # Preprocess (fit on train)
        prep = DatasetPreprocessor(
            feature_types,
            dataset_id,
            missing_threshold=cfg.get("missing_threshold", 0.50),
            ohe_max_categories=cfg.get("ohe_max_categories", 50),
            scale_method=cfg.get("scale_method", "robust"),
        )
        print(f"    preprocessing {dataset_id} ...", end=" ", flush=True)
        t0 = time.time()
        X_tr_np  = prep.fit_transform(X_tr)
        X_val_np = prep.transform(X_val)
        X_te_np  = prep.transform(X_te)
        print(f"done ({time.time()-t0:.1f}s)  features={prep.n_features_out}")

        # Save fitted preprocessor for reuse in Phase 6
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        prep_path = MODELS_DIR / f"preprocessor_{dataset_id}.joblib"
        joblib.dump(prep, prep_path)
        print(f"    preprocessor saved → {prep_path.name}")

        return {
            "X_tr":   X_tr_np,
            "y_tr":   y_tr.values,
            "X_val":  X_val_np,
            "y_val":  y_val.values,
            "X_te":   X_te_np,
            "y_te":   y_te.values,
            "feature_names": prep.feature_names_out,
            "prep":   prep,
        }


def _stratified_sample(y: pd.Series, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    indices = []
    for cls, frac in zip(classes, fracs):
        cls_idx = np.where(y.values == cls)[0]
        k = max(1, int(round(n * frac)))
        k = min(k, len(cls_idx))
        indices.append(rng.choice(cls_idx, size=k, replace=False))
    return np.concatenate(indices)


# ── model training ────────────────────────────────────────────────────────────

def _train_rf(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> RandomForestClassifier:
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ── distribution shift ────────────────────────────────────────────────────────

def _ks_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-feature KS statistic between two datasets."""
    from scipy.stats import ks_2samp
    n_feat = min(a.shape[1], b.shape[1])
    ks_vals = []
    for i in range(n_feat):
        stat, _ = ks_2samp(a[:, i], b[:, i])
        ks_vals.append(stat)
    return float(np.mean(ks_vals))


def _js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Mean per-feature Jensen-Shannon divergence."""
    from scipy.spatial.distance import jensenshannon
    n_feat = min(a.shape[1], b.shape[1])
    js_vals = []
    for i in range(n_feat):
        mn = min(a[:, i].min(), b[:, i].min())
        mx = max(a[:, i].max(), b[:, i].max())
        if mx <= mn:
            js_vals.append(0.0)
            continue
        ha, _ = np.histogram(a[:, i], bins=bins, range=(mn, mx), density=True)
        hb, _ = np.histogram(b[:, i], bins=bins, range=(mn, mx), density=True)
        ha = ha + 1e-10
        hb = hb + 1e-10
        ha /= ha.sum()
        hb /= hb.sum()
        js_vals.append(float(jensenshannon(ha, hb)))
    return float(np.mean(js_vals))


# ── main cross-domain loop ────────────────────────────────────────────────────

def run_cross_domain(
    datasets: list[str] = None,
    seed: int = 42,
) -> dict:
    datasets = datasets or DATASETS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Cross-Domain Experiment (zero-shot transfer) ===")
    print(f"Datasets : {datasets}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Seed     : {seed}\n")

    cache = _DataCache(seed=seed)

    # Pre-load all datasets
    print("[Phase 1] Loading and preprocessing all datasets...")
    for ds in datasets:
        cache.get(ds)

    # In-domain baselines (train + test on same dataset)
    print("\n[Phase 2] In-domain baselines...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    in_domain: dict[str, dict] = {}
    for ds in datasets:
        d = cache.get(ds)
        print(f"  {ds}: training RF ...", end=" ", flush=True)
        t0 = time.time()
        clf = _train_rf(d["X_tr"], d["y_tr"], seed)
        y_prob = clf.predict_proba(d["X_te"])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_metrics(d["y_te"], y_pred, y_prob)
        in_domain[ds] = m
        # Save in-domain model
        model_path = MODELS_DIR / f"rf_indomain_{ds}.joblib"
        joblib.dump(clf, model_path)
        print(f"F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  ({time.time()-t0:.1f}s)  saved→{model_path.name}")

    # Cross-domain pairs
    print("\n[Phase 3] Cross-domain transfer (all pairs × strategies)...")
    results: dict = {}   # results[src][tgt][strategy] = metrics dict

    for src in datasets:
        results[src] = {}
        src_data = cache.get(src)
        src_names = src_data["feature_names"]

        for tgt in datasets:
            if src == tgt:
                continue
            results[src][tgt] = {}
            tgt_data = cache.get(tgt)
            tgt_names = tgt_data["feature_names"]

            for strategy in STRATEGIES:
                try:
                    aligner = make_aligner(strategy, src_names, tgt_names)
                except ValueError as e:
                    print(f"  {src}→{tgt} [{strategy}] SKIP: {e}")
                    results[src][tgt][strategy] = {"error": str(e)}
                    continue

                n_feat = aligner.n_features
                print(f"  {src}→{tgt} [{strategy}] {n_feat} features ...", end=" ", flush=True)
                t0 = time.time()

                # Align source train
                X_src_tr = aligner.align_source(src_data["X_tr"])
                y_src_tr = src_data["y_tr"]

                # Train
                clf = _train_rf(X_src_tr, y_src_tr, seed)

                # Align target test
                X_tgt_te = aligner.align_target(tgt_data["X_te"])
                y_tgt_te = tgt_data["y_te"]

                y_prob = clf.predict_proba(X_tgt_te)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                m = compute_metrics(y_tgt_te, y_pred, y_prob)

                # Distribution shift on aligned features (sample for speed)
                n_shift = min(5000, len(X_src_tr), len(X_tgt_te))
                rng = np.random.default_rng(seed)
                idx_s = rng.choice(len(X_src_tr), n_shift, replace=False)
                idx_t = rng.choice(len(X_tgt_te), n_shift, replace=False)
                ks  = _ks_divergence(X_src_tr[idx_s], X_tgt_te[idx_t])
                js  = _js_divergence(X_src_tr[idx_s], X_tgt_te[idx_t])

                elapsed = time.time() - t0
                f1_gap = in_domain[tgt]["f1"] - m["f1"]

                # Save cross-domain model
                model_path = MODELS_DIR / f"rf_{src}_to_{tgt}_{strategy}.joblib"
                joblib.dump(clf, model_path)

                print(f"F1={m['f1']:.4f}  gap={f1_gap:+.4f}  KS={ks:.3f}  JS={js:.3f}  ({elapsed:.1f}s)")

                results[src][tgt][strategy] = {
                    **m,
                    "n_features": n_feat,
                    "ks_divergence": ks,
                    "js_divergence": js,
                    "f1_gap": f1_gap,
                    "elapsed_s": round(elapsed, 2),
                    "model_path": str(model_path),
                    "aligner_feature_names": aligner.feature_names,
                }

    # Save raw results
    out_json = OUT_DIR / "cross_domain_results.json"
    payload = {
        "in_domain": in_domain,
        "cross_domain": results,
        "metadata": {
            "datasets": datasets,
            "strategies": STRATEGIES,
            "seed": seed,
            "max_train": MAX_TRAIN,
        }
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Generate figures
    print("\n[Phase 4] Generating figures...")
    _plot_matrices(results, in_domain, datasets)
    _plot_degradation(results, in_domain, datasets)
    _plot_shift_scatter(results, datasets)

    print("\n=== Done ===")
    return payload


# ── figures ───────────────────────────────────────────────────────────────────

def _f1_matrix(results: dict, datasets: list[str], strategy: str) -> np.ndarray:
    """Build a (n_ds × n_ds) F1 matrix; diagonal=NaN."""
    n = len(datasets)
    mat = np.full((n, n), np.nan)
    for i, src in enumerate(datasets):
        for j, tgt in enumerate(datasets):
            if src == tgt:
                continue
            cell = results.get(src, {}).get(tgt, {}).get(strategy, {})
            if "f1" in cell:
                mat[i, j] = cell["f1"]
    return mat


def _plot_matrices(results: dict, in_domain: dict, datasets: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available — skipping matrix plots")
        return

    labels = [d.upper() if d != "ieee_cis" else "IEEE-CIS" for d in datasets]

    for strategy in STRATEGIES:
        for metric_key, metric_label in [("f1", "F1"), ("pr_auc", "PR-AUC")]:
            n = len(datasets)
            mat = np.full((n, n), np.nan)
            for i, src in enumerate(datasets):
                for j, tgt in enumerate(datasets):
                    if src == tgt:
                        # diagonal = in-domain
                        mat[i, j] = in_domain[tgt].get(metric_key, np.nan)
                        continue
                    cell = results.get(src, {}).get(tgt, {}).get(strategy, {})
                    if metric_key in cell:
                        mat[i, j] = cell[metric_key]

            fig, ax = plt.subplots(figsize=(6, 5))
            mask = np.eye(n, dtype=bool)
            sns.heatmap(
                mat, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor="gray",
            )
            ax.set_title(f"Cross-domain {metric_label}  [{strategy}]\n(rows=source, cols=target, diagonal=in-domain)")
            ax.set_xlabel("Target dataset")
            ax.set_ylabel("Source dataset")
            plt.tight_layout()
            fname = OUT_DIR / f"{metric_key.replace('-','_')}_matrix_{strategy}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Saved {fname.name}")


def _plot_degradation(results: dict, in_domain: dict, datasets: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [d.upper() if d != "ieee_cis" else "IEEE-CIS" for d in datasets]

    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(5 * len(STRATEGIES), 5), sharey=True)
    if len(STRATEGIES) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#9C27B0"]

    for ax, strategy, color in zip(axes, STRATEGIES, colors):
        # For each target dataset, collect F1 gaps from all sources
        tgt_gaps: dict[str, list[float]] = {tgt: [] for tgt in datasets}
        for src in datasets:
            for tgt in datasets:
                if src == tgt:
                    continue
                cell = results.get(src, {}).get(tgt, {}).get(strategy, {})
                if "f1_gap" in cell:
                    tgt_gaps[tgt].append(cell["f1_gap"])

        # Box/bar per target
        tgt_list = list(datasets)
        means = [np.mean(tgt_gaps[t]) if tgt_gaps[t] else np.nan for t in tgt_list]
        stds  = [np.std(tgt_gaps[t])  if len(tgt_gaps[t]) > 1 else 0.0 for t in tgt_list]

        x = np.arange(len(tgt_list))
        ax.bar(x, means, yerr=stds, color=color, alpha=0.75, capsize=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"[{strategy}]")
        ax.set_xlabel("Target dataset")

    axes[0].set_ylabel("F1 gap  (in-domain − cross-domain)")
    fig.suptitle("Performance Degradation under Distribution Shift", fontsize=13)
    plt.tight_layout()
    fname = OUT_DIR / "degradation_all_strategies.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


def _plot_shift_scatter(results: dict, datasets: list[str]):
    """Scatter: KS divergence vs F1 gap coloured by strategy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"semantic": "#2196F3", "intersection": "#FF9800", "zero_fill": "#9C27B0"}
    markers = {"semantic": "o", "intersection": "s", "zero_fill": "^"}

    for strategy in STRATEGIES:
        ks_vals, js_vals, gap_vals = [], [], []
        for src in datasets:
            for tgt in datasets:
                if src == tgt:
                    continue
                cell = results.get(src, {}).get(tgt, {}).get(strategy, {})
                if "ks_divergence" in cell and "f1_gap" in cell:
                    ks_vals.append(cell["ks_divergence"])
                    js_vals.append(cell["js_divergence"])
                    gap_vals.append(cell["f1_gap"])

        kw = dict(label=strategy, color=colors[strategy],
                  marker=markers[strategy], alpha=0.8, s=60)
        axes[0].scatter(ks_vals, gap_vals, **kw)
        axes[1].scatter(js_vals, gap_vals, **kw)

    for ax, xlabel in zip(axes, ["Mean KS divergence", "Mean JS divergence"]):
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F1 gap  (in-domain − cross-domain)")
        ax.legend()

    fig.suptitle("Distribution Shift vs Performance Degradation", fontsize=13)
    plt.tight_layout()
    fname = OUT_DIR / "shift_vs_degradation.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain fraud detection experiment")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS, help="Datasets to include")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cross_domain(datasets=args.datasets, seed=args.seed)
