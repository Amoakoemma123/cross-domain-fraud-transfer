"""Phase 7 statistical validation for transfer learning results.

1. Wilcoxon signed-rank tests: each method vs no_adapt, paired across the
   12 source->target pairs (seed 42, best label_frac per method-pair).
2. Multi-seed mean +/- 95% CI for key cells (seeds 42/123/456), once
   transfer_results_seed{123,456}.json exist.

Usage
-----
    python -m experiments.transfer.run_stats

Outputs
-------
    experiments/transfer/results/stats_wilcoxon.json
    experiments/transfer/results/stats_multiseed.json (if seed files present)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RES_DIR = ROOT / "experiments" / "transfer" / "results"

KEY_CELLS = [
    ("no_adapt",    0.00),
    ("dann",        0.00),
    ("finetune",    0.10),
    ("contrastive", 0.01),
]


def wilcoxon_vs_no_adapt() -> dict:
    df = pd.DataFrame(json.load(open(RES_DIR / "transfer_results.json")))
    # Best F1 / PR-AUC per (src, tgt, method) over label fracs
    best = df.groupby(["src", "tgt", "method"])[["f1", "pr_auc"]].max().reset_index()
    base = best[best.method == "no_adapt"].set_index(["src", "tgt"])

    out = {}
    for method in sorted(best.method.unique()):
        if method == "no_adapt":
            continue
        sub = best[best.method == method].set_index(["src", "tgt"])
        idx = sub.index.intersection(base.index)
        res = {}
        for metric in ("f1", "pr_auc"):
            a = sub.loc[idx, metric].values
            b = base.loc[idx, metric].values
            diff = a - b
            if np.allclose(diff, 0):
                res[metric] = {"n": len(idx), "mean_gain": 0.0, "p": 1.0}
                continue
            try:
                stat, p = sps.wilcoxon(a, b, alternative="greater")
            except ValueError as e:
                stat, p = np.nan, np.nan
            res[metric] = {
                "n": len(idx),
                "mean_gain": round(float(diff.mean()), 4),
                "median_gain": round(float(np.median(diff)), 4),
                "wins": int((diff > 0).sum()),
                "W": float(stat),
                "p_one_sided": round(float(p), 5),
            }
        out[method] = res

    with open(RES_DIR / "stats_wilcoxon.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== Wilcoxon signed-rank vs no_adapt (paired over 12 pairs, best frac) ===")
    print(f"{'Method':<13}{'F1 gain':>9}{'p':>10}{'wins':>6}   "
          f"{'PR-AUC gain':>12}{'p':>10}{'wins':>6}")
    for m, r in out.items():
        f1, pr = r["f1"], r["pr_auc"]
        print(f"{m:<13}{f1['mean_gain']:>9.4f}{f1['p_one_sided']:>10.5f}"
              f"{f1.get('wins', 0):>6}   "
              f"{pr['mean_gain']:>12.4f}{pr['p_one_sided']:>10.5f}"
              f"{pr.get('wins', 0):>6}")
    return out


def multiseed_cis() -> dict | None:
    frames = []
    df42 = pd.DataFrame(json.load(open(RES_DIR / "transfer_results.json")))
    df42["seed"] = 42
    frames.append(df42)
    found_extra = False
    for seed in (123, 456):
        p = RES_DIR / f"transfer_results_seed{seed}.json"
        if p.exists():
            d = pd.DataFrame(json.load(open(p)))
            if not d.empty:
                d["seed"] = seed
                frames.append(d)
                found_extra = True
    if not found_extra:
        print("\n[multiseed] No extra-seed result files yet — skipping CIs.")
        return None

    df = pd.concat(frames, ignore_index=True)
    out = {}
    print("\n=== Multi-seed mean F1 across pairs, per seed -> mean ± 95% CI ===")
    for method, frac in KEY_CELLS:
        sub = df[(df.method == method) & (np.isclose(df.label_frac, frac))]
        per_seed = sub.groupby("seed")["f1"].mean()
        if len(per_seed) < 2:
            continue
        m = per_seed.mean()
        ci = sps.t.ppf(0.975, len(per_seed) - 1) * per_seed.std(ddof=1) / np.sqrt(len(per_seed))
        out[f"{method}@{frac:.0%}"] = {
            "seeds": {int(s): round(float(v), 4) for s, v in per_seed.items()},
            "mean": round(float(m), 4),
            "ci95": round(float(ci), 4),
        }
        print(f"  {method}@{frac:.0%}: mean={m:.4f} ± {ci:.4f}  "
              f"(per-seed: {dict((int(s), round(float(v), 4)) for s, v in per_seed.items())})")

    with open(RES_DIR / "stats_multiseed.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    wilcoxon_vs_no_adapt()
    multiseed_cis()
