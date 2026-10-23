"""Run additional label fractions for learning-curve granularity.

Runs contrastive, finetune, and tradaboost at extra fractions: 0.02, 0.03, 0.07, 0.15, 0.30
across all 12 pairs, seed 42. Appends to transfer_results.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.transfer.run_transfer import (
    _run_one, DataCache, _load_existing_results, _save_results, _result_key,
    RESULTS_FILE,
)
from experiments.transfer.config import ALL_PAIRS

EXTRA_FRACS = [0.02, 0.03, 0.07, 0.15, 0.30]
METHODS = ['contrastive', 'finetune', 'tradaboost']
SEED = 42


def main():
    results = _load_existing_results()
    done = {_result_key(r) for r in results}

    cache = DataCache(seed=SEED)
    total = 0
    ran = 0

    for src, tgt in ALL_PAIRS:
        for method in METHODS:
            for frac in EXTRA_FRACS:
                key = (src, tgt, method, frac)
                if key in done:
                    continue
                total += 1
                print(f"[{ran+1}] {src}->{tgt} {method}@{frac:.0%}", flush=True)
                t0 = time.time()
                try:
                    result = _run_one(src, tgt, method, frac, cache, SEED)
                    if result is not None:
                        result['elapsed_s'] = round(time.time() - t0, 2)
                        results.append(result)
                        ran += 1
                        _save_results(results)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        'src': src, 'tgt': tgt, 'method': method,
                        'label_frac': frac, 'error': str(e),
                        'elapsed_s': round(time.time() - t0, 2),
                    })
                    _save_results(results)

    print(f"\nDone. Ran {ran} new cells.")


if __name__ == '__main__':
    main()
