"""TabPFN foundation model benchmark for cross-domain fraud detection.

Evaluates TabPFN as a zero-shot and few-shot baseline on the same 12 transfer
pairs and label budgets used in the main benchmark. TabPFN does not use a source
domain — it relies entirely on its pretrained prior and the target labels provided.

This tests whether a foundation model can match alignment-based transfer
under extreme imbalance and schema heterogeneity.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.transfer.run_transfer import DataCache
from experiments.cross_domain.feature_aligner import make_aligner
from experiments.transfer.config import ALL_PAIRS
from src.evaluation.metrics import compute_metrics

OUT_DIR = ROOT / 'experiments' / 'foundation' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TabPFN has a row limit — typically 10K for training, 10K for prediction
TABPFN_MAX_TRAIN = 3000
TABPFN_MAX_TEST = 10000
LABEL_FRACS = [0.01, 0.05, 0.10, 0.20]
SEED = 42


def main():
    from tabpfn import TabPFNClassifier

    cache = DataCache(seed=SEED)
    results = []
    out_file = OUT_DIR / 'tabpfn_results.json'

    # Resume
    if out_file.exists():
        results = json.load(open(out_file))
    done = {(r['src'], r['tgt'], r['label_frac']) for r in results}

    for src, tgt in ALL_PAIRS:
        tgt_data = cache.get(tgt)
        src_data = cache.get(src)
        aligner = make_aligner('zero_fill', src_data['feature_names'], tgt_data['feature_names'])

        X_tgt_tr = aligner.align_target(tgt_data['X_tr'])
        y_tgt_tr = tgt_data['y_tr']
        X_tgt_te = aligner.align_target(tgt_data['X_te'])
        y_tgt_te = tgt_data['y_te']

        # Subsample test if too large
        if len(X_tgt_te) > TABPFN_MAX_TEST:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(len(X_tgt_te), TABPFN_MAX_TEST, replace=False)
            X_tgt_te = X_tgt_te[idx]
            y_tgt_te = y_tgt_te[idx]

        for frac in LABEL_FRACS:
            if (src, tgt, frac) in done:
                print(f"  SKIP {src}->{tgt} @{frac:.0%}")
                continue

            # Sample labeled target
            n_label = max(1, int(len(X_tgt_tr) * frac))
            rng = np.random.RandomState(SEED)

            # Stratified sample
            fraud_idx = np.where(y_tgt_tr == 1)[0]
            legit_idx = np.where(y_tgt_tr == 0)[0]
            n_fraud = max(1, int(n_label * y_tgt_tr.mean()))
            n_legit = n_label - n_fraud

            if n_fraud > len(fraud_idx):
                n_fraud = len(fraud_idx)
                n_legit = n_label - n_fraud
            if n_legit > len(legit_idx):
                n_legit = len(legit_idx)

            sel_fraud = rng.choice(fraud_idx, min(n_fraud, len(fraud_idx)), replace=False)
            sel_legit = rng.choice(legit_idx, min(n_legit, len(legit_idx)), replace=False)
            sel = np.concatenate([sel_fraud, sel_legit])

            X_train = X_tgt_tr[sel]
            y_train = y_tgt_tr[sel]

            # Cap training size for TabPFN
            if len(X_train) > TABPFN_MAX_TRAIN:
                sub = rng.choice(len(X_train), TABPFN_MAX_TRAIN, replace=False)
                X_train = X_train[sub]
                y_train = y_train[sub]

            print(f"  {src}->{tgt} TabPFN @{frac:.0%}: {len(X_train)} train ({y_train.sum():.0f} fraud), {len(X_tgt_te)} test", flush=True)

            t0 = time.time()
            try:
                clf = TabPFNClassifier(device='cpu', n_estimators=8, random_state=SEED)
                clf.fit(X_train, y_train)
                y_proba = clf.predict_proba(X_tgt_te)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)

                metrics = compute_metrics(y_tgt_te, y_pred, y_proba)
                metrics.update({
                    'src': src, 'tgt': tgt, 'method': 'tabpfn',
                    'label_frac': frac, 'n_train': len(X_train),
                    'n_fraud_train': int(y_train.sum()),
                    'elapsed_s': round(time.time() - t0, 2),
                })
                results.append(metrics)
                print(f"    F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  ({metrics['elapsed_s']:.0f}s)")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    'src': src, 'tgt': tgt, 'method': 'tabpfn',
                    'label_frac': frac, 'error': str(e),
                    'elapsed_s': round(time.time() - t0, 2),
                })

            # Save after each cell
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} cells saved.")

    # Summary
    df = pd.DataFrame([r for r in results if 'error' not in r])
    if len(df) > 0:
        print("\n=== TabPFN Mean Results ===")
        piv = df.pivot_table(index='method', columns='label_frac',
                             values=['f1', 'pr_auc'], aggfunc='mean').round(3)
        print(piv)


if __name__ == '__main__':
    main()
