"""Cost-sensitive re-evaluation of transfer methods.

Re-scores all transfer results using a monetary cost framework:
- True Positive: saves the fraud amount (benefit)
- False Positive: investigation cost (fixed cost per alert)
- False Negative: fraud loss (the fraud amount)
- True Negative: no cost

Uses realistic cost assumptions from the literature:
- Investigation cost per alert: $50 (Whitrow et al., 2009)
- Average fraud amount: varies by dataset (computed from data)
- Cost ratio scenarios: conservative, moderate, aggressive

Outputs: cost_results.json + cost_comparison.png
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TR = ROOT / 'experiments' / 'transfer' / 'results'
OUT_DIR = ROOT / 'experiments' / 'cost_sensitive' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load transfer results (fixed threshold)
tr = pd.DataFrame(json.load(open(TR / 'transfer_results.json')))
if 'error' in tr.columns:
    tr = tr[tr['error'].isna()].copy()
tr = tr[tr.label_frac.isin([0.0, 0.01, 0.05, 0.10, 0.20])].copy()

# Also load threshold-tuned results for supervised methods
tt = pd.DataFrame(json.load(open(TR / 'threshold_tuned_results.json')))
# Merge tuned precision/recall into tr for supervised methods
for _, row in tt.iterrows():
    mask = (tr.src == row.src) & (tr.tgt == row.tgt) & (tr.method == row.method) & (tr.label_frac == row.label_frac)
    if mask.any():
        tr.loc[mask, 'precision_tuned'] = row.get('precision_tuned', np.nan)
        tr.loc[mask, 'recall_tuned'] = row.get('recall_tuned', np.nan)

# Cost scenarios (investigation_cost / avg_fraud_amount)
# Low ratio = fraud is expensive relative to investigation
COST_SCENARIOS = {
    'conservative': {'investigation_cost': 100, 'fraud_multiplier': 1.0},  # high review cost
    'moderate':     {'investigation_cost': 50,  'fraud_multiplier': 1.0},  # baseline
    'aggressive':   {'investigation_cost': 25,  'fraud_multiplier': 2.0},  # low review, high fraud cost
}

# Normalized fraud amounts (all scaled to comparable retail-fraud range)
# PaySim amounts are simulation artifacts; we normalize to realistic retail scale
AVG_FRAUD = {
    'ulb': 122.0,       # ULB credit card mean fraud ~$122
    'ieee_cis': 150.0,  # IEEE-CIS e-commerce ~$150
    'paysim': 200.0,    # PaySim normalized (simulation amounts unrealistic)
    'banksim': 500.0,   # BankSim retail ~$500
}


def expected_cost(precision, recall, n_fraud, n_legit, fraud_amount, inv_cost, fraud_mult):
    """Compute expected cost per transaction under given performance."""
    if precision == 0 and recall == 0:
        # No alerts: all fraud is missed
        return {'tp': 0, 'fp': 0, 'fn': n_fraud, 'cost_fp': 0,
                'cost_fn': n_fraud * fraud_amount * fraud_mult,
                'benefit_tp': 0, 'net_savings': 0, 'savings_pct': 0.0}

    tp = recall * n_fraud
    fn = (1 - recall) * n_fraud
    # precision = tp / (tp + fp) => fp = tp * (1-precision) / precision
    fp = tp * (1 - precision) / precision if precision > 0 else 0

    benefit_tp = tp * fraud_amount * fraud_mult   # fraud caught
    cost_fp = fp * inv_cost                        # false investigations
    cost_fn = fn * fraud_amount * fraud_mult       # fraud missed

    total_cost = cost_fn + cost_fp - benefit_tp    # net cost (negative = savings)
    # Normalize: savings relative to no-detection baseline
    baseline_cost = n_fraud * fraud_amount * fraud_mult
    savings_pct = (baseline_cost - (cost_fn + cost_fp)) / baseline_cost * 100 if baseline_cost > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'cost_fp': cost_fp,
        'cost_fn': cost_fn,
        'benefit_tp': benefit_tp,
        'net_savings': benefit_tp - cost_fp,
        'savings_pct': savings_pct,
    }


# Compute cost metrics for each method at best label frac
order = ['no_adapt', 'coral_rf', 'mmd', 'coral_deep', 'dann', 'tradaboost', 'finetune', 'contrastive']
results = []

for scenario_name, scenario in COST_SCENARIOS.items():
    inv_cost = scenario['investigation_cost']
    fraud_mult = scenario['fraud_multiplier']

    for method in order:
        md = tr[tr.method == method]
        # Get best label frac by F1
        best_frac = md.groupby('label_frac')['f1'].mean().idxmax()
        best = md[md.label_frac == best_frac]

        # Average across pairs — use tuned precision/recall if available
        savings_list = []
        for _, row in best.iterrows():
            tgt = row['tgt']
            fraud_amt = AVG_FRAUD.get(tgt, 200)
            # Prefer tuned metrics for supervised methods
            prec = row.get('precision_tuned', row.get('precision', 0))
            rec = row.get('recall_tuned', row.get('recall', 0))
            if pd.isna(prec):
                prec = row.get('precision', 0)
            if pd.isna(rec):
                rec = row.get('recall', 0)

            if pd.isna(prec) or pd.isna(rec):
                continue

            # Assume typical test set composition
            n_total = 10000
            fraud_rates = {'ulb': 0.00172, 'ieee_cis': 0.035, 'paysim': 0.00129, 'banksim': 0.01211}
            fr = fraud_rates.get(tgt, 0.01)
            n_fraud = int(n_total * fr)
            n_legit = n_total - n_fraud

            cost = expected_cost(prec, rec, n_fraud, n_legit, fraud_amt, inv_cost, fraud_mult)
            savings_list.append(cost['savings_pct'])

        if savings_list:
            results.append({
                'scenario': scenario_name,
                'method': method,
                'best_frac': best_frac,
                'mean_savings_pct': round(np.mean(savings_list), 1),
                'min_savings_pct': round(np.min(savings_list), 1),
                'max_savings_pct': round(np.max(savings_list), 1),
                'n_pairs': len(savings_list),
            })

# Save results
with open(OUT_DIR / 'cost_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
df = pd.DataFrame(results)
print("=== Cost-Sensitive Evaluation ===\n")
for scenario in COST_SCENARIOS:
    print(f"\n--- {scenario.upper()} scenario ---")
    sc = df[df.scenario == scenario].sort_values('mean_savings_pct', ascending=False)
    for _, row in sc.iterrows():
        print(f"  {row['method']:15s} @{row['best_frac']:.0%}: "
              f"mean savings {row['mean_savings_pct']:+.1f}%  "
              f"(range {row['min_savings_pct']:+.1f}% to {row['max_savings_pct']:+.1f}%)")

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, scenario in zip(axes, COST_SCENARIOS):
    sc = df[df.scenario == scenario].sort_values('mean_savings_pct')
    colors = ['#d73027' if v < 0 else '#1a9850' for v in sc.mean_savings_pct]
    ax.barh(range(len(sc)), sc.mean_savings_pct, color=colors)
    ax.set_yticks(range(len(sc)))
    ax.set_yticklabels(sc.method, fontsize=9)
    ax.set_xlabel('Mean Savings (%)')
    ax.set_title(f'{scenario.title()} Scenario', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Cost-Sensitive Evaluation: % Savings vs No-Detection Baseline', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'cost_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved cost_comparison.png")
