"""Transfer learning experiment runner.

Trains each method (no_adapt, coral_rf, coral_deep, mmd, dann, finetune,
contrastive) across all 12 source→target pairs and 5 label fractions.

Usage
-----
    cd E:/Research_Works/Fraud_Detection_Research/financial-fraud-transfer
    python -m experiments.transfer.run_transfer [options]

    # Run all pairs, all methods
    python -m experiments.transfer.run_transfer

    # Single pair, specific methods
    python -m experiments.transfer.run_transfer --src ulb --tgt ieee_cis \\
        --methods no_adapt coral_rf coral_deep mmd dann

    # Skip slow deep methods
    python -m experiments.transfer.run_transfer --methods no_adapt coral_rf

Outputs (under experiments/transfer/results/)
---------------------------------------------
    transfer_results.json          — all metrics (appended run-by-run)
    models/{src}_{tgt}_{method}_{frac}.joblib — saved adapters
    f1_vs_labelfrac.png            — F1 vs label fraction per method (averaged)
    method_heatmap_f1.png          — method × dataset-pair heatmap at best frac
    gain_over_no_adapt.png         — improvement of each method over no_adapt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loaders.ulb       import load_ulb
from src.data.loaders.ieee_cis  import load_ieee_cis
from src.data.loaders.paysim    import load_paysim
from src.data.loaders.banksim   import load_banksim
from src.data.preprocessing     import DatasetPreprocessor
from src.evaluation.metrics     import compute_metrics
from src.models.transfer.adapter import TransferAdapter

from experiments.baseline.config  import DATASET_CONFIG
from experiments.transfer.config  import (
    ALL_PAIRS, LABEL_FRACS, ALL_METHODS, DEFAULT_METHODS,
    UNSUPERVISED_METHODS, MAX_SRC_TRAIN, DEEP_CONFIG, RF_CONFIG,
)
from experiments.cross_domain.feature_aligner import make_aligner

# ── paths ─────────────────────────────────────────────────────────────────────

OUT_DIR        = ROOT / "experiments" / "transfer" / "results"
MODELS_DIR     = ROOT / "experiments" / "transfer" / "models"
PREP_CACHE_DIR = ROOT / "experiments" / "cross_domain" / "models"  # reuse saved preprocessors
RESULTS_FILE   = OUT_DIR / "transfer_results.json"

# ── dataset loading & caching ─────────────────────────────────────────────────

def _raw_load(dataset_id: str):
    cfg   = DATASET_CONFIG[dataset_id]
    path  = cfg["path"]
    if dataset_id == "ulb":
        return load_ulb(path)
    elif dataset_id == "ieee_cis":
        return load_ieee_cis(path, split="train")
    elif dataset_id == "paysim":
        return load_paysim(path)
    elif dataset_id == "banksim":
        return load_banksim(path)
    raise ValueError(dataset_id)


def _stratified_subsample(X, y, n, seed):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    idx = []
    for cls, frac in zip(classes, fracs):
        cls_idx = np.where(y == cls)[0]
        k = max(1, min(int(round(n * frac)), len(cls_idx)))
        idx.append(rng.choice(cls_idx, k, replace=False))
    return np.concatenate(idx)


class DataCache:
    """Load + preprocess each dataset once; reuse saved preprocessors where possible."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._cache: dict[str, dict] = {}

    def get(self, dataset_id: str) -> dict:
        if dataset_id not in self._cache:
            self._cache[dataset_id] = self._load(dataset_id)
        return self._cache[dataset_id]

    def _load(self, dataset_id: str) -> dict:
        print(f"  [data] {dataset_id} ...", end=" ", flush=True)
        t0 = time.time()

        # Try to reuse fitted preprocessor saved by cross-domain experiment
        prep_path = PREP_CACHE_DIR / f"preprocessor_{dataset_id}.joblib"

        X_df, y, feature_types = _raw_load(dataset_id)
        cfg = DATASET_CONFIG[dataset_id]

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

        if prep_path.exists():
            prep = joblib.load(prep_path)
            X_tr_np  = prep.transform(X_tr)
            X_val_np = prep.transform(X_val)
            X_te_np  = prep.transform(X_te)
            print(f"(prep cached)  {len(X_df):,} rows  ({time.time()-t0:.1f}s)")
        else:
            prep = DatasetPreprocessor(
                feature_types, dataset_id,
                missing_threshold=cfg.get("missing_threshold", 0.50),
                ohe_max_categories=cfg.get("ohe_max_categories", 50),
                scale_method=cfg.get("scale_method", "robust"),
            )
            X_tr_np  = prep.fit_transform(X_tr)
            X_val_np = prep.transform(X_val)
            X_te_np  = prep.transform(X_te)
            print(f"{len(X_df):,} rows  features={prep.n_features_out}  ({time.time()-t0:.1f}s)")

        # Subsample source train
        y_tr_np = y_tr.values
        if len(X_tr_np) > MAX_SRC_TRAIN:
            idx = _stratified_subsample(X_tr_np, y_tr_np, MAX_SRC_TRAIN, self._seed)
            X_tr_np  = X_tr_np[idx]
            y_tr_np  = y_tr_np[idx]

        return {
            "X_tr":   X_tr_np,
            "y_tr":   y_tr_np,
            "X_val":  X_val_np,
            "y_val":  y_val.values,
            "X_te":   X_te_np,
            "y_te":   y_te.values,
            "feature_names": prep.feature_names_out,
        }


# ── label-fraction sampling ───────────────────────────────────────────────────

def _sample_labeled_target(X_tgt_tr, y_tgt_tr, frac: float, seed: int):
    """Return (X_lab, y_lab) — stratified sample of frac of target train."""
    if frac <= 0 or len(X_tgt_tr) == 0:
        return np.zeros((0, X_tgt_tr.shape[1])), np.zeros(0)
    n = max(2, int(round(frac * len(X_tgt_tr))))
    n = min(n, len(X_tgt_tr))
    try:
        idx, _ = train_test_split(
            np.arange(len(X_tgt_tr)), train_size=n,
            stratify=y_tgt_tr, random_state=seed)
    except ValueError:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tgt_tr), n, replace=False)
    return X_tgt_tr[idx], y_tgt_tr[idx]


# ── run one experiment cell ───────────────────────────────────────────────────

def _run_one(
    src: str, tgt: str, method: str, label_frac: float,
    cache: DataCache, seed: int,
) -> dict | None:
    """Train one (src, tgt, method, frac) cell. Returns metrics dict or None on skip."""

    # Skip semi-supervised methods at 0% label fraction
    if label_frac == 0.0 and method in ("finetune", "tradaboost"):
        return None

    # RF-based methods don't use labeled target at all — label_frac > 0 is redundant.
    # Record as duplicate of frac=0.0 result to avoid re-training.
    if method in ("no_adapt", "coral_rf") and label_frac > 0.0:
        return None  # handled by cloning frac=0 result in run_transfer()

    src_data = cache.get(src)
    tgt_data = cache.get(tgt)

    src_names = src_data["feature_names"]
    tgt_names = tgt_data["feature_names"]

    # Use zero_fill alignment: full source feature space; target zeros out absent cols
    aligner = make_aligner("zero_fill", src_names, tgt_names)
    input_dim = aligner.n_features

    X_src_tr = aligner.align_source(src_data["X_tr"])
    y_src_tr = src_data["y_tr"]

    X_tgt_tr_unlab = aligner.align_target(tgt_data["X_tr"])  # all target train (unlabeled)
    y_tgt_tr       = tgt_data["y_tr"]
    X_tgt_val      = aligner.align_target(tgt_data["X_val"])
    y_tgt_val      = tgt_data["y_val"]
    X_tgt_te       = aligner.align_target(tgt_data["X_te"])
    y_tgt_te       = tgt_data["y_te"]

    # Labeled target subset
    X_tgt_lab, y_tgt_lab = _sample_labeled_target(X_tgt_tr_unlab, y_tgt_tr, label_frac, seed)

    # Build adapter
    is_deep = method in ("coral_deep", "mmd", "dann", "finetune", "contrastive")
    adapter = TransferAdapter(
        method=method,
        input_dim=input_dim,
        **DEEP_CONFIG,
        rf_n_estimators=RF_CONFIG["n_estimators"],
        rf_min_samples_leaf=RF_CONFIG["min_samples_leaf"],
        seed=seed,
    )

    t0 = time.time()
    adapter.fit(
        X_src_tr, y_src_tr,
        X_tgt_unlab=X_tgt_tr_unlab,
        X_tgt_lab=X_tgt_lab if len(X_tgt_lab) > 0 else None,
        y_tgt_lab=y_tgt_lab if len(y_tgt_lab) > 0 else None,
        X_val=X_tgt_val,
        y_val=y_tgt_val,
    )
    elapsed = time.time() - t0

    y_prob = adapter.predict_proba(X_tgt_te)
    y_pred = (y_prob >= 0.5).astype(int)
    m = compute_metrics(y_tgt_te, y_pred, y_prob)

    # Save model
    frac_tag = f"{int(label_frac * 100):03d}pct"
    model_name = f"{src}_to_{tgt}_{method}_{frac_tag}.joblib"
    model_path = MODELS_DIR / model_name
    adapter.save(str(model_path))

    return {
        "src": src, "tgt": tgt,
        "method": method,
        "label_frac": label_frac,
        "n_src_train": len(X_src_tr),
        "n_tgt_labeled": len(X_tgt_lab),
        "n_tgt_test": len(X_tgt_te),
        "input_dim": input_dim,
        **m,
        "elapsed_s": round(elapsed, 2),
        "model_path": str(model_path),
    }


# ── results I/O (append incrementally) ───────────────────────────────────────

def _load_existing_results() -> list[dict]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def _save_results(results: list[dict]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def _result_key(r: dict) -> tuple:
    return (r["src"], r["tgt"], r["method"], r["label_frac"])


# ── figures ───────────────────────────────────────────────────────────────────

def _plot_f1_vs_labelfrac(results: list[dict], methods: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Average F1 across all pairs for each (method, label_frac)
    df = pd.DataFrame(results)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        grp = sub.groupby("label_frac")["f1"].mean().reset_index()
        ax.plot(grp["label_frac"] * 100, grp["f1"],
                marker="o", label=method, color=colors[i % len(colors)])

    ax.set_xlabel("Labeled target data (%)")
    ax.set_ylabel("F1 score (mean across all pairs)")
    ax.set_title("Transfer Performance vs. Label Fraction")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = OUT_DIR / "f1_vs_labelfrac.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


def _plot_method_heatmap(results: list[dict], methods: list[str], pairs: list[tuple]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    df = pd.DataFrame(results)
    if df.empty:
        return

    # Use best label_frac F1 for each (method, pair)
    pair_labels = [f"{s}→{t}" for s, t in pairs]
    mat = np.full((len(methods), len(pairs)), np.nan)
    for j, (src, tgt) in enumerate(pairs):
        for i, method in enumerate(methods):
            sub = df[(df["src"] == src) & (df["tgt"] == tgt) & (df["method"] == method)]
            if not sub.empty:
                mat[i, j] = sub["f1"].max()  # best over label fracs

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.9), max(4, len(methods) * 0.7)))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=pair_labels, yticklabels=methods,
                vmin=0, vmax=1, ax=ax, linewidths=0.4)
    ax.set_title("Best F1 per Method × Transfer Pair")
    ax.set_xlabel("Transfer direction")
    ax.set_ylabel("Method")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    fname = OUT_DIR / "method_heatmap_f1.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


def _plot_gain_over_no_adapt(results: list[dict], methods: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.DataFrame(results)
    if df.empty or "no_adapt" not in df["method"].values:
        return

    # Compute F1 gain vs no_adapt at same (src, tgt, label_frac)
    no_adapt_df = df[df["method"] == "no_adapt"][["src","tgt","label_frac","f1"]]
    no_adapt_df = no_adapt_df.rename(columns={"f1": "f1_no_adapt"})
    merged = df.merge(no_adapt_df, on=["src","tgt","label_frac"], how="left")
    merged["f1_gain"] = merged["f1"] - merged["f1_no_adapt"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    adapt_methods = [m for m in methods if m != "no_adapt"]
    for i, method in enumerate(adapt_methods):
        sub = merged[merged["method"] == method]
        if sub.empty:
            continue
        grp = sub.groupby("label_frac")["f1_gain"].mean().reset_index()
        ax.plot(grp["label_frac"] * 100, grp["f1_gain"],
                marker="o", label=method, color=colors[i % len(colors)])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Labeled target data (%)")
    ax.set_ylabel("F1 gain over no-adapt baseline")
    ax.set_title("Adaptation Gain vs. Label Fraction")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = OUT_DIR / "gain_over_no_adapt.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


def _plot_pr_auc_heatmap(results: list[dict], methods: list[str], pairs: list[tuple]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    df = pd.DataFrame(results)
    if df.empty:
        return

    pair_labels = [f"{s}→{t}" for s, t in pairs]
    mat = np.full((len(methods), len(pairs)), np.nan)
    for j, (src, tgt) in enumerate(pairs):
        for i, method in enumerate(methods):
            sub = df[(df["src"] == src) & (df["tgt"] == tgt) & (df["method"] == method)]
            if not sub.empty and "pr_auc" in sub.columns:
                mat[i, j] = sub["pr_auc"].max()

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.9), max(4, len(methods) * 0.7)))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=pair_labels, yticklabels=methods,
                vmin=0, vmax=1, ax=ax, linewidths=0.4)
    ax.set_title("Best PR-AUC per Method × Transfer Pair")
    ax.set_xlabel("Transfer direction")
    ax.set_ylabel("Method")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    fname = OUT_DIR / "method_heatmap_prauc.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_transfer(
    pairs: list[tuple] = None,
    methods: list[str] = None,
    label_fracs: list[float] = None,
    seed: int = 42,
    resume: bool = True,
):
    pairs       = pairs       or ALL_PAIRS
    methods     = methods     or DEFAULT_METHODS
    label_fracs = label_fracs or LABEL_FRACS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Transfer Learning Experiments ===")
    print(f"Pairs   : {len(pairs)}")
    print(f"Methods : {methods}")
    print(f"Fracs   : {label_fracs}")
    print(f"Seed    : {seed}\n")

    # Load existing results (resume support)
    all_results = _load_existing_results() if resume else []
    done_keys = {_result_key(r) for r in all_results}
    print(f"  [resume] {len(all_results)} results already on disk\n")

    # Pre-load all needed datasets
    needed_datasets = list({ds for pair in pairs for ds in pair})
    print("[Phase 1] Loading datasets...")
    cache = DataCache(seed=seed)
    for ds in needed_datasets:
        cache.get(ds)

    # Main loop
    print("\n[Phase 2] Running transfer experiments...")
    total = len(pairs) * len(methods) * len(label_fracs)
    done_count = len(all_results)
    run_count = 0

    # RF-method cache: train once per (src,tgt), copy result across fracs
    rf_result_cache: dict[tuple, dict] = {}

    for src, tgt in pairs:
        for method in methods:
            for frac in label_fracs:
                key = (src, tgt, method, frac)
                if key in done_keys:
                    continue  # resume: skip completed

                tag = f"{src}→{tgt} [{method}] frac={frac:.0%}"

                # RF methods: only train once (at frac=0.0), copy for other fracs
                if method in ("no_adapt", "coral_rf"):
                    rf_key = (src, tgt, method)
                    if rf_key not in rf_result_cache:
                        # Check if frac=0 already in done_keys
                        base_key = (src, tgt, method, 0.0)
                        if base_key in done_keys:
                            # find it in all_results
                            base = next((r for r in all_results
                                         if _result_key(r) == base_key), None)
                            rf_result_cache[rf_key] = base
                        else:
                            print(f"  {src}→{tgt} [{method}] frac=0% (once) ...", end=" ", flush=True)
                            try:
                                base = _run_one(src, tgt, method, 0.0, cache, seed)
                            except Exception as e:
                                print(f"ERROR: {e}")
                                base = {"src": src, "tgt": tgt, "method": method,
                                        "label_frac": 0.0, "error": str(e),
                                        "f1": 0.0, "pr_auc": 0.0}
                            if base:
                                f1  = base.get("f1", 0.0)
                                auc = base.get("pr_auc", 0.0)
                                print(f"F1={f1:.4f}  PR-AUC={auc:.4f}  ({base.get('elapsed_s',0):.1f}s)")
                                base["label_frac"] = 0.0
                                all_results.append(base)
                                done_keys.add(base_key)
                                run_count += 1
                            rf_result_cache[rf_key] = base

                    # Copy result for this frac (different label_frac tag, same metrics)
                    base = rf_result_cache.get(rf_key)
                    if base and frac > 0.0:
                        cloned = {**base, "label_frac": frac}
                        all_results.append(cloned)
                        done_keys.add(key)
                        run_count += 1
                    continue

                # All other methods: run normally
                print(f"  {tag} ...", end=" ", flush=True)
                try:
                    result = _run_one(src, tgt, method, frac, cache, seed)
                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {
                        "src": src, "tgt": tgt, "method": method,
                        "label_frac": frac, "error": str(e),
                        "f1": 0.0, "pr_auc": 0.0,
                    }

                if result is None:
                    continue

                f1  = result.get("f1", 0.0)
                auc = result.get("pr_auc", 0.0)
                elapsed = result.get("elapsed_s", 0.0)
                print(f"F1={f1:.4f}  PR-AUC={auc:.4f}  ({elapsed:.1f}s)")

                all_results.append(result)
                done_keys.add(key)
                run_count += 1

                # Save incrementally every 5 runs
                if run_count % 5 == 0:
                    _save_results(all_results)

    _save_results(all_results)
    print(f"\nResults saved → {RESULTS_FILE}  ({len(all_results)} records total)")

    # Figures
    print("\n[Phase 3] Generating figures...")
    _plot_f1_vs_labelfrac(all_results, methods)
    _plot_method_heatmap(all_results, methods, pairs)
    _plot_pr_auc_heatmap(all_results, methods, pairs)
    _plot_gain_over_no_adapt(all_results, methods)

    print("\n=== Done ===")
    return all_results


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict], methods: list[str] = None):
    """Print a compact summary table of mean F1 per method × label_frac."""
    df = pd.DataFrame(results)
    if df.empty:
        print("No results yet.")
        return
    methods = methods or sorted(df["method"].unique())
    fracs   = sorted(df["label_frac"].unique())

    print("\n=== Mean F1 across all pairs ===")
    header = f"{'Method':<14}" + "".join(f"  {f:.0%}" for f in fracs)
    print(header)
    print("-" * len(header))
    for method in methods:
        sub = df[df["method"] == method]
        row = f"{method:<14}"
        for f in fracs:
            cell = sub[sub["label_frac"] == f]["f1"]
            row += f"  {cell.mean():.3f}" if not cell.empty else "   —  "
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning experiment runner")
    parser.add_argument("--src",     type=str, default=None,
                        help="Single source dataset (use with --tgt)")
    parser.add_argument("--tgt",     type=str, default=None,
                        help="Single target dataset (use with --src)")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=list(ALL_METHODS),
                        help="Methods to run (default: all)")
    parser.add_argument("--fracs",   nargs="+", type=float, default=None,
                        help="Label fractions (default: 0 0.01 0.05 0.10 0.20)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing results and re-run everything")
    args = parser.parse_args()

    if args.src and args.tgt:
        pairs = [(args.src, args.tgt)]
    elif args.src or args.tgt:
        parser.error("--src and --tgt must be used together")
        pairs = ALL_PAIRS
    else:
        pairs = ALL_PAIRS

    results = run_transfer(
        pairs=pairs,
        methods=args.methods,
        label_fracs=args.fracs,
        seed=args.seed,
        resume=not args.no_resume,
    )
    print_summary(results, args.methods)
