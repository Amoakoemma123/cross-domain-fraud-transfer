# Is Transfer Worth Its Labels? A Label-Budget-Honest Evaluation of Transfer Learning, Decision Threshold Calibration and Performance Estimation for Cross-Institution Financial Fraud Detection

**Emmanuel Kwame Amoako**<sup>a*</sup>, Mahamudu Amadu<sup>b</sup>, James Quayson<sup>b</sup>, Kingsford Donkor<sup>c</sup>, Daniel Asamoah Apraku<sup>d</sup>

<sup>a</sup> Independent Researcher, Dormaa Ahenkro, Ghana
<sup>b</sup> Department of Information Technology and Decision Sciences, University of Energy and Natural Resources, Sunyani, Ghana
<sup>c</sup> Department of Computer Engineering, University of Energy and Natural Resources, Sunyani, Ghana
<sup>d</sup> Department of Computer Science, University of Energy and Natural Resources, Sunyani, Ghana

\* Corresponding author: amoakopuyol123@gmail.com
ORCID: [0009-0004-3477-6832](https://orcid.org/0009-0004-3477-6832)

---

## Overview

This repository contains the code and results for a label-budget-honest benchmark of cross-institution fraud transfer. Every target label used for training, early stopping or threshold selection is charged to an explicit budget (1%, 5%, 10% or 20% of the target training partition). Methods that claim to use no target labels select models on source data only. The labelled target validation set is used only to report an oracle upper bound.

Thirteen methods (label-free adaptation, supervised transfer and target-only baselines) are evaluated with eight threshold policies on all 12 directed pairs of four public fraud datasets, over 5 seeds. Seed 42 is the development seed; seeds 123, 456, 789 and 1024 are confirmation seeds.

## Key results (5 seeds)

| Finding | Value |
|---------|-------|
| Best label-free method, default threshold | F1 ≤ 0.10 |
| Best transfer vs best target-only at 5% budget | F1 0.709 vs 0.714 (no budget significant) |
| Budget plug-in F1 estimate: mean error | +0.009 (hold-out F1: +0.048) |
| Choosing transfer vs target-only by the estimate | +0.009 F1, mostly from few-fraud cells |

## Reproducing the v2 benchmark

```bash
python -m experiments.v2.prepare_data              # per-seed cached splits (dataset/processed/v2)
python -m experiments.v2.run_v2 --seed 42          # repeat for 123 456 789 1024
python -m experiments.v2.analyze_v2                # tables in experiments/v2/analysis/
```

- Results: `experiments/v2/results/results_seed*.json`
- Threshold policies: `src/evaluation/thresholds.py`
- Label-blind budget sensitivity run (ULB, 1%): `experiments/v2/queue_randbudget.ps1`, which uses `run_v2 --random-budget --budget-rep k`.

## Earlier version (v1, superseded)

The directories `experiments/{cross_domain,transfer,ablation,imbalance,cost_sensitive,interpretability}` and `run_experiment.py` hold the first version of this study ("Calibration Is All You Need"). **Its results should not be used.** That protocol tuned thresholds and early-stopped every deep method, including the "label-free" ones, on the full labelled target validation set. Its zero-fill alignment also dropped target-only features. The v2 benchmark above replaces it. The v1 description below is kept for reference.

## Datasets

| ID | Dataset | Transactions | Fraud Rate | Features | Source |
|----|---------|-------------|------------|----------|--------|
| ulb | ULB Credit Card | 284,807 | 0.172% | 32 | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| ieee_cis | IEEE-CIS | 590,540 | 3.499% | 292 | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) |
| paysim | PaySim | 6,362,620 | 0.129% | 15 | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| banksim | BankSim | 594,643 | 1.211% | 30 | [Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/banksim1) |

**Download instructions:** Place raw CSV files under `dataset/raw/<dataset_name>/` (e.g., `dataset/raw/ulb/creditcard.csv`). Datasets are not included in this repository due to size constraints.

## Project Structure

```
financial-fraud-transfer/
├── src/                          # Source code
│   ├── data/                     # Loaders, cleaning, harmonisation
│   ├── features/                 # Engineering, encoding, scaling, alignment
│   ├── models/                   # Baselines (RF, MLP) and transfer methods
│   ├── training/                 # Trainer, losses, sampling, early stopping
│   ├── evaluation/               # Metrics, cross-domain, statistics, calibration
│   ├── explainability/           # SHAP analysis
│   └── visualization/            # Plotting utilities
├── configs/                      # YAML configuration files
├── experiments/                  # Experiment scripts and results
│   ├── cross_domain/results/     # Zero-shot transfer results + figures
│   ├── transfer/results/         # Transfer benchmark results (636 runs) + figures
│   ├── ablation/results/         # Ablation analysis (162 cells) + figures
│   ├── imbalance/results/        # Imbalance sensitivity results + figures
│   ├── cost_sensitive/results/   # Cost-sensitive evaluation + figures
│   └── interpretability/results/ # SHAP results + figures
├── results/                      # Aggregated metrics
├── run_experiment.py             # Main entry point
├── requirements.txt              # Python dependencies
└── README.md
```

## Transfer Methods Benchmarked

| Family | Method | Labels Required |
|--------|--------|----------------|
| No adaptation | no_adapt (RF baseline) | 0% |
| Moment matching | Shallow CORAL, Deep CORAL, MMD | 0% |
| Adversarial | DANN | 0% |
| Supervised transfer | Fine-tuning, Contrastive alignment, TrAdaBoost | 1–20% |

## Experiment Phases

| Phase | Description | Runs |
|-------|-------------|------|
| Baselines | In-domain (5 models × 4 datasets) | 20 |
| Zero-shot | Cross-domain (12 pairs × 3 strategies) | 36 |
| Transfer benchmark | 8 methods × 12 pairs × 10 fractions | 636 |
| Multi-seed | 5 seeds (42/123/456/789/1024) | 240 |
| Threshold tuning | Re-evaluation with calibrated thresholds | all |
| TrAdaBoost | SOTA instance-transfer baseline | 48 |
| Ablation | 27 configs × 6 pairs | 162 |
| Imbalance | 5 strategies × 4 datasets × 2 models | 40 |
| SHAP | Feature importance (5 models, 3 methods) | 5 |
| Cost-sensitive | 3 cost scenarios | 24 |

## Setup & Reproduction

```bash
# Create virtual environment
python -m venv fraud_env
source fraud_env/bin/activate  # Linux/Mac
# fraud_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run baseline experiments
python run_experiment.py --experiment baseline --dataset ulb

# Run transfer benchmark
python run_experiment.py --experiment transfer

# Reproduce figures from saved results (no retraining needed)
python run_experiment.py --experiment visualise
```

All experiments use fixed random seeds configured in `configs/experiments.yaml`. Result JSON files are included in this repository, enabling reproduction of all tables and figures without retraining.

## Pre-computed Results (v1, superseded)

The v1 result directories contain:

- `transfer_results.json` — 636 transfer runs (seed 42)
- `transfer_results_seed{123,456,789,1024}.json` — multi-seed replication
- `threshold_tuned_results.json` — calibrated results
- `stats_multiseed.json` / `stats_multiseed_tuned.json` — 5-seed statistics
- `stats_wilcoxon.json` — Wilcoxon signed-rank test results
- `ablation_results.json` — 162-cell ablation
- `cross_domain_results.json` — zero-shot transfer
- `imbalance_results.json` — imbalance sensitivity
- `cost_results.json` — cost-sensitive evaluation
- `shap_results.json` — SHAP feature importance

## Citation

If you use this code or results, please cite:

```
Amoako, E.K., Amadu, M., Quayson, J., Donkor, K., & Apraku, D.A. (2026).
Is Transfer Worth Its Labels? A Label-Budget-Honest Evaluation of Transfer
Learning, Decision Threshold Calibration and Performance Estimation for
Cross-Institution Financial Fraud Detection. Manuscript under review.
```

## License

This project is released for academic and research purposes. See [LICENSE](LICENSE) for details.

## Acknowledgements

The authors acknowledge the use of Claude (Anthropic) as an AI coding assistant during the implementation of the experimental pipeline, generation of figures, and drafting of the manuscript. All experimental design decisions, interpretation of results, and scientific conclusions are the sole responsibility of the authors.
