"""SHAP feature importance analysis for transfer-adapted models.

Loads saved contrastive, finetune, and DANN adapters, reconstructs the encoder,
computes SHAP values on target test data, and produces summary plots.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import torch

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.transfer.run_transfer import DataCache
from experiments.cross_domain.feature_aligner import make_aligner
from src.models.transfer.encoder import FraudEncoder
from src.models.transfer.contrastive import ContrastiveEncoder
from src.models.transfer.dann import DANN

MODELS_DIR = ROOT / 'experiments' / 'transfer' / 'models'
OUT_DIR = ROOT / 'experiments' / 'interpretability' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PAIRS_TO_ANALYZE = [
    ('paysim', 'banksim', 'contrastive', '001pct'),
    ('ieee_cis', 'ulb', 'contrastive', '010pct'),
    ('paysim', 'banksim', 'finetune', '010pct'),
    ('paysim', 'banksim', 'dann', '000pct'),
    ('ieee_cis', 'banksim', 'contrastive', '001pct'),
]


def load_model(src, tgt, method, frac_str):
    """Load saved model dict and reconstruct the encoder."""
    fname = f"{src}_to_{tgt}_{method}_{frac_str}.joblib"
    path = MODELS_DIR / fname
    if not path.exists():
        print(f"  MISSING: {fname}")
        return None

    saved = joblib.load(path)
    if not isinstance(saved, dict) or 'state_dict' not in saved:
        print(f"  Unexpected format: {type(saved)}")
        return None

    model_class = saved.get('model_class', 'FraudEncoder')
    input_dim = saved['input_dim']
    hidden_dims = saved.get('hidden_dims', (128, 64))
    repr_dim = saved.get('repr_dim', 32)
    dropout = saved.get('dropout', 0.3)

    if model_class == 'ContrastiveEncoder':
        model = ContrastiveEncoder(input_dim, encoder_dims=list(hidden_dims),
                                   latent_dim=repr_dim, dropout=dropout)
    elif model_class == 'DANN':
        model = DANN(input_dim, encoder_dims=list(hidden_dims),
                     latent_dim=repr_dim, dropout=dropout)
    else:
        model = FraudEncoder(input_dim, hidden_dims=hidden_dims,
                             repr_dim=repr_dim, dropout=dropout)

    model.load_state_dict(saved['state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def make_predict_fn(model):
    """Create a numpy-in numpy-out prediction function."""
    has_fraud_proba = hasattr(model, 'predict_fraud_proba')
    def predict(X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            if has_fraud_proba:
                probs = model.predict_fraud_proba(t).cpu().numpy().flatten()
            else:
                logits = model(t)
                if logits.dim() == 1 or logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                else:
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return probs
    return predict


def main():
    cache = DataCache(seed=42)
    all_shap_results = []

    for src, tgt, method, frac_str in PAIRS_TO_ANALYZE:
        print(f"\n=== {src}->{tgt} {method}@{frac_str} ===")

        model = load_model(src, tgt, method, frac_str)
        if model is None:
            continue

        src_data = cache.get(src)
        tgt_data = cache.get(tgt)
        aligner = make_aligner('zero_fill', src_data['feature_names'], tgt_data['feature_names'])
        feat_names = aligner.feature_names

        X_te = aligner.align_target(tgt_data['X_te'])

        # Sample for SHAP
        rng = np.random.RandomState(42)
        n_bg = min(100, len(X_te))
        n_ex = min(100, len(X_te))
        bg_idx = rng.choice(len(X_te), n_bg, replace=False)
        ex_idx = rng.choice(len(X_te), n_ex, replace=False)

        pred_fn = make_predict_fn(model)

        # Test prediction
        test_pred = pred_fn(X_te[bg_idx[:5]])
        print(f"  Test predictions: {test_pred[:5]}")

        try:
            explainer = shap.KernelExplainer(pred_fn, X_te[bg_idx])
            shap_values = explainer.shap_values(X_te[ex_idx], nsamples=50)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_k = 20
            top_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]

            print(f"  Top 10 features by mean |SHAP|:")
            for i, idx in enumerate(top_idx[:10]):
                name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
                print(f"    {i+1}. {name}: {mean_abs_shap[idx]:.4f}")

            # Save SHAP summary plot
            fig_name = f"shap_{src}_{tgt}_{method}_{frac_str}.png"
            feature_names_arr = [feat_names[i] if i < len(feat_names) else f'feat_{i}'
                                 for i in range(X_te.shape[1])]
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_te[ex_idx],
                              feature_names=feature_names_arr,
                              max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(OUT_DIR / fig_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {fig_name}")

            top_features = []
            for idx in top_idx:
                name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
                top_features.append({'feature': name, 'mean_abs_shap': float(mean_abs_shap[idx])})

            all_shap_results.append({
                'src': src, 'tgt': tgt, 'method': method, 'frac': frac_str,
                'top_features': top_features,
            })

        except Exception as e:
            print(f"  SHAP failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    with open(OUT_DIR / 'shap_results.json', 'w') as f:
        json.dump(all_shap_results, f, indent=2)
    print(f"\nSaved shap_results.json ({len(all_shap_results)} analyses)")

    # Cross-method comparison figure
    if len(all_shap_results) >= 2:
        n_plots = min(len(all_shap_results), 4)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        for ax, result in zip(axes, all_shap_results[:n_plots]):
            top10 = result['top_features'][:10]
            names = [f['feature'][:30] for f in top10]
            values = [f['mean_abs_shap'] for f in top10]
            ax.barh(range(len(names)), values[::-1], color='steelblue')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names[::-1], fontsize=7)
            ax.set_xlabel('Mean |SHAP|')
            ax.set_title(f"{result['src']}\u2192{result['tgt']}\n{result['method']}@{result['frac']}", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'shap_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved shap_comparison.png")


if __name__ == '__main__':
    main()
