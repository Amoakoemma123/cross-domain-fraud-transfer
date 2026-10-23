"""Multi-seed transfer runs for Phase 7 statistical validation.

Re-runs key (method, label_frac) cells across all 12 pairs with extra seeds.
Key cells (chosen from seed-42 results):
    no_adapt    @ 0%    — baseline
    dann        @ 0%    — best unsupervised
    finetune    @ 10%   — best semi-supervised (monotonic)
    contrastive @ 1%    — best single F1

Usage
-----
    python -m experiments.transfer.run_multiseed --seeds 123 456

Outputs
-------
    experiments/transfer/results/transfer_results_seed{seed}.json
    experiments/transfer/models_seed{seed}/ (saved adapters)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import experiments.transfer.run_transfer as rt
from experiments.transfer.config import ALL_PAIRS

KEY_CELLS = [
    ("no_adapt",    0.00),
    ("dann",        0.00),
    ("finetune",    0.10),
    ("contrastive", 0.01),
]


def run_seed(seed: int, pairs: list[tuple]) -> list[dict]:
    out_file = rt.OUT_DIR / f"transfer_results_seed{seed}.json"
    models_dir = rt.ROOT / "experiments" / "transfer" / f"models_seed{seed}"
    models_dir.mkdir(parents=True, exist_ok=True)
    rt.MODELS_DIR = models_dir  # redirect model saving away from seed-42 models

    results = []
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
    done = {rt._result_key(r) for r in results}

    cache = rt.DataCache(seed=seed)
    total = len(pairs) * len(KEY_CELLS)
    print(f"\n=== Seed {seed}: {total} cells ({len(done)} already done) ===")

    for src, tgt in pairs:
        for method, frac in KEY_CELLS:
            key = (src, tgt, method, frac)
            if key in done:
                continue
            print(f"  [seed {seed}] {src}->{tgt} [{method}] frac={frac:.0%} ...",
                  end=" ", flush=True)
            t0 = time.time()
            try:
                r = rt._run_one(src, tgt, method, frac, cache, seed)
            except Exception as e:
                print(f"ERROR: {e}")
                r = {"src": src, "tgt": tgt, "method": method,
                     "label_frac": frac, "error": str(e),
                     "f1": 0.0, "pr_auc": 0.0}
            if r is None:
                continue
            r["seed"] = seed
            print(f"F1={r.get('f1', 0):.4f}  PR-AUC={r.get('pr_auc', 0):.4f}"
                  f"  ({time.time() - t0:.1f}s)")
            results.append(r)
            done.add(key)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

    print(f"[seed {seed}] saved -> {out_file}  ({len(results)} records)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456])
    args = parser.parse_args()

    rt.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_seed(seed, ALL_PAIRS)
    print("\n=== Multi-seed done ===")
