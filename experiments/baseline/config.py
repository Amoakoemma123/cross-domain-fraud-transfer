"""Per-dataset configuration for baseline experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG = {
    "ulb": {
        "name":         "ULB Credit Card",
        "loader":       "ulb",
        "path":         str(ROOT / "dataset/raw/ulb/creditcard.csv"),
        "loader_kwargs": {},
        # Training-data caps (None = use all)
        "rf_max_train":  None,
        "lr_max_train":  None,
        "mlp_max_train": None,
        # Preprocessing
        "missing_threshold":  0.50,
        "ohe_max_categories": 50,
        "scale_method":       "robust",
    },
    "ieee_cis": {
        "name":         "IEEE-CIS",
        "loader":       "ieee_cis",
        "path":         str(ROOT / "dataset/raw/ieee_cis"),
        "loader_kwargs": {"split": "train"},
        "rf_max_train":  None,
        "lr_max_train":  None,
        "mlp_max_train": None,
        "missing_threshold":  0.50,
        "ohe_max_categories": 50,
        "scale_method":       "robust",
    },
    "paysim": {
        "name":         "PaySim",
        "loader":       "paysim",
        "path":         str(ROOT / "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"),
        "loader_kwargs": {},
        # PaySim has 6.36M rows — cap all models on training data
        "rf_max_train":  300_000,
        "lr_max_train":  1_000_000,
        "mlp_max_train": 500_000,
        "xgb_max_train": 1_000_000,
        "lgbm_max_train":1_000_000,
        "missing_threshold":  0.50,
        "ohe_max_categories": 50,
        "scale_method":       "robust",
    },
    "banksim": {
        "name":         "BankSim",
        "loader":       "banksim",
        "path":         str(ROOT / "dataset/raw/banksim/bs140513_032310.csv"),
        "loader_kwargs": {},
        "rf_max_train":  None,
        "lr_max_train":  None,
        "mlp_max_train": None,
        "missing_threshold":  0.50,
        "ohe_max_categories": 50,
        "scale_method":       "robust",
    },
}

# Model hyperparameters — conservative settings for fast first runs
MODEL_CONFIG = {
    "lr": {
        "C": 1.0,
        "max_iter": 200,
        "solver": "saga",          # saga scales to large datasets; 200 passes is enough
        "class_weight": "balanced",
    },
    "rf": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "xgb": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "verbosity": 0,
        "n_jobs": -1,
    },
    "lgbm": {
        "n_estimators": 500,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "class_weight": "balanced",
        "early_stopping_round": 30,
        "n_jobs": -1,
        "verbose": -1,
    },
    "mlp": {
        "hidden_dims": (256, 128, 64),
        "dropout": 0.3,
        "batch_norm": True,
        "lr": 1e-3,
        "batch_size": 512,
        "epochs": 50,
        "patience": 10,
    },
}
