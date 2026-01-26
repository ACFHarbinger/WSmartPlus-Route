"""
Hyper-Parameter Optimization (HPO) constants.
"""
from typing import Tuple

# Hyper-Parameter Optimization
HOP_KEYS: Tuple[str, ...] = (
    "hpo_method",
    "hpo_range",
    "hpo_epochs",
    "metric",
    "n_trials",
    "timeout",
    "n_startup_trials",
    "n_warmup_steps",
    "interval_steps",
    "eta",
    "indpb",
    "tournsize",
    "cxpb",
    "mutpb",
    "n_pop",
    "n_gen",
    "fevals",
    "cpu_cores",
    "verbose",
    "train_best",
    "local_mode",
    "num_samples",
    "max_tres",
    "reduction_factor",
    "max_failures",
    "grid",
    "max_conc",
)
