"""Configuration for transfer learning experiments."""

# All 12 directed source→target pairs
ALL_PAIRS = [
    ("ulb",      "ieee_cis"),
    ("ulb",      "paysim"),
    ("ulb",      "banksim"),
    ("ieee_cis", "ulb"),
    ("ieee_cis", "paysim"),
    ("ieee_cis", "banksim"),
    ("paysim",   "ulb"),
    ("paysim",   "ieee_cis"),
    ("paysim",   "banksim"),
    ("banksim",  "ulb"),
    ("banksim",  "ieee_cis"),
    ("banksim",  "paysim"),
]

# Fraction of labeled target-train data available at adaptation time
LABEL_FRACS = [0.00, 0.01, 0.05, 0.10, 0.20]

# Which methods are valid at each label fraction
# (no_adapt / coral_rf / coral_deep / mmd / dann: 0% OK)
# (finetune / contrastive: need >0%)
UNSUPERVISED_METHODS  = ("no_adapt", "coral_rf", "coral_deep", "mmd", "dann")
SEMISUPERVISED_METHODS = ("finetune", "contrastive", "tradaboost")
ALL_METHODS = UNSUPERVISED_METHODS + SEMISUPERVISED_METHODS

# Default subset to run (all methods)
DEFAULT_METHODS = list(ALL_METHODS)

# Max source training rows (keep fast)
MAX_SRC_TRAIN = 200_000

# Deep model architecture (shared across coral_deep / mmd / dann / finetune / contrastive)
DEEP_CONFIG = {
    "hidden_dims":  (128, 64),
    "repr_dim":     32,
    "dropout":      0.3,
    "lr":           1e-3,
    "batch_size":   256,
    "epochs":       50,
    "patience":     10,
    "lambda_da":    1.0,   # weight of domain alignment loss
}

# RF config (no_adapt, coral_rf)
RF_CONFIG = {
    "n_estimators":     200,
    "min_samples_leaf": 4,
}
