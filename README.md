# Calibration Is All You Need: Threshold Tuning as the Highest-Leverage Intervention for Cross-Domain Financial Fraud Transfer

**Emmanuel Kwame Amoako**<sup>a*</sup>, Mahamudu Amadu<sup>b</sup>, James Quayson<sup>b</sup>, Kingsford Donkor<sup>c</sup>, Daniel Asamoah Apraku<sup>d</sup>

<sup>a</sup> Independent Researcher, Dormaa Ahenkro, Ghana
<sup>b</sup> Department of Information Technology and Decision Sciences, University of Energy and Natural Resources, Sunyani, Ghana
<sup>c</sup> Department of Computer Engineering, University of Energy and Natural Resources, Sunyani, Ghana
<sup>d</sup> Department of Computer Science, University of Energy and Natural Resources, Sunyani, Ghana

\* Corresponding author: amoakopuyol123@gmail.com
ORCID: [0009-0004-3477-6832](https://orcid.org/0009-0004-3477-6832)

---

## Overview

This repository contains the full experimental pipeline for the paper *"Calibration Is All You Need: Threshold Tuning as the Highest-Leverage Intervention for Cross-Domain Financial Fraud Transfer"*, submitted to **Expert Systems with Applications**.

Machine learning fraud detectors achieve strong in-domain performance but collapse under cross-domain distribution shift. We benchmark seven transfer-learning methods across four heterogeneous fraud datasets (12 directed pairs, 636 runs, 5-seed replication) and show that **decision-threshold calibration — a zero-cost post-hoc step — is the single highest-leverage intervention**, more than doubling supervised transfer F1 (0.130 → 0.292).

## Key Results

| Finding | Value |
|---------|-------|
| Zero-shot transfer F1 | 0.00–0.14 (all 12 pairs) |
| Best unsupervised (DANN) | F1 0.110 ± 0.041 |
| Best supervised fixed threshold | F1 0.130 (contrastive @ 1%) |
| **Best calibrated** | **F1 0.292 (contrastive @ 10%)** |
| Seed variance reduction | ±0.029 → ±0.013 |

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

## Pre-computed Results

The `experiments/*/results/` directories contain all result JSONs and figures from the paper:

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
Calibration Is All You Need: Threshold Tuning as the Highest-Leverage
Intervention for Cross-Domain Financial Fraud Transfer.
Expert Systems with Applications (submitted).
```

## License

This project is released for academic and research purposes. See [LICENSE](LICENSE) for details.

## Acknowledgements

The authors acknowledge the use of Claude (Anthropic) as an AI coding assistant during the implementation of the experimental pipeline, generation of figures, and drafting of the manuscript. All experimental design decisions, interpretation of results, and scientific conclusions are the sole responsibility of the authors.
