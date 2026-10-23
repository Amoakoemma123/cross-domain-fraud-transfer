"""Main experiment entry point.

Usage examples:
    python run_experiment.py --experiment baseline --dataset ulb
    python run_experiment.py --experiment cross_domain --source ulb --target paysim
    python run_experiment.py --experiment adaptation --source ieee_cis --target paysim --method dann
    python run_experiment.py --experiment imbalance --dataset ulb
    python run_experiment.py --experiment ablation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Financial Fraud Transfer Learning Experiments")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "cross_domain", "adaptation", "imbalance", "ablation"],
        required=True,
        help="Experiment type to run",
    )
    parser.add_argument(
        "--dataset",
        choices=["ulb", "ieee_cis", "paysim", "banksim", "all"],
        default="all",
        help="Dataset for baseline/imbalance experiments",
    )
    parser.add_argument("--source", default=None, help="Source domain dataset ID")
    parser.add_argument("--target", default=None, help="Target domain dataset ID")
    parser.add_argument(
        "--method",
        choices=["coral", "mmd", "dann", "contrastive", "all"],
        default="all",
        help="Transfer learning method",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config_dir", default="configs", help="Path to config directory"
    )
    parser.add_argument(
        "--output_dir", default="results", help="Path to output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == "baseline":
        from experiments.baseline.run_baseline import run_baseline
        run_baseline(
            dataset=args.dataset,
            seed=args.seed,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
        )

    elif args.experiment == "cross_domain":
        if not args.source or not args.target:
            raise ValueError("--source and --target are required for cross_domain experiments")
        from experiments.cross_domain.run_cross_domain import run_cross_domain
        run_cross_domain(
            source=args.source,
            target=args.target,
            seed=args.seed,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
        )

    elif args.experiment == "adaptation":
        if not args.source or not args.target:
            raise ValueError("--source and --target are required for adaptation experiments")
        from experiments.adaptation.run_adaptation import run_adaptation
        run_adaptation(
            source=args.source,
            target=args.target,
            method=args.method,
            seed=args.seed,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
        )

    elif args.experiment == "imbalance":
        from experiments.imbalance.run_imbalance import run_imbalance
        run_imbalance(
            dataset=args.dataset,
            seed=args.seed,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
        )

    elif args.experiment == "ablation":
        from experiments.ablation.run_ablation import run_ablation
        run_ablation(
            seed=args.seed,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
