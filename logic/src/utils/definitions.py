from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Paths
path: Path = Path(os.getcwd())
parts: tuple[str, ...] = path.parts
ROOT_DIR: Path = Path(*parts[: parts.index("WSmart-Route") + 1])
ICON_FILE: str = os.path.join(ROOT_DIR, "assets", "images", "logo-wsmartroute-white.png")

# Multi-core processing settings
CORE_LOCK_WAIT_TIME: int = 10
LOCK_TIMEOUT: int = CORE_LOCK_WAIT_TIME


def update_lock_wait_time(num_cpu_cores: Optional[int] = None) -> int:
    """
    Updates the global LOCK_TIMEOUT based on the number of CPU cores.

    Args:
        num_cpu_cores: Number of CPU cores to scale the timeout by.

    Returns:
        The new (or default) value of LOCK_TIMEOUT.
    """
    global LOCK_TIMEOUT
    global CORE_LOCK_WAIT_TIME
    if num_cpu_cores is None:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME
    else:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME * num_cpu_cores
    return LOCK_TIMEOUT


# Waste management information
MAP_DEPOTS: Dict[str, str] = {
    "mixrmbac": "CTEASO",  # Rio Maior, Bombarral, Azambuja, Cadaval
    "riomaior": "CTEASO",
    "figueiradafoz": "CITVRSU",
}

WASTE_TYPES: Dict[str, str] = {
    "glass": "Embalagens de Vidro",
    "plastic": "Mistura de embalagens",
    "paper": "Embalagens de papel e cartão",
}

# Distance matrix
EARTH_RADIUS: int = 6371
EARTH_WMP_RADIUS: int = 6378137

# WSmart+ route simulation settings
PBAR_WAIT_TIME: float = 0.1

METRICS: List[str] = ["overflows", "kg", "ncol", "kg_lost", "km", "kg/km", "cost", "profit"]
SIM_METRICS: List[str] = METRICS + ["days", "time"]
DAY_METRICS: List[str] = ["day"] + METRICS + ["tour"]
LOSS_KEYS: List[str] = ["nll", "reinforce_loss", "baseline_loss"]
TQDM_COLOURS: List[str] = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "cyan",
    "white",
]

# Plotting
MARKERS: List[str] = ["P", "s", "^", "8", "*"]
PLOT_COLOURS: List[str] = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "cyan",
    "black",
]
LINESTYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "dotted",
    "dashed",
    "dashdot",
    (0, (3, 5, 1, 5, 1, 5)),
    "solid",
]

# Problem definition
MAX_WASTE: float = 1.0
MAX_LENGTHS: Dict[int, int] = {20: 2, 50: 3, 100: 4, 150: 5, 225: 6, 317: 7}
VEHICLE_CAPACITY: float = 100.0

# Model configurations
SUB_NET_ENCS: List[str] = ["tgc"]
PRED_ENC_MODELS: List[str] = ["tam"]
ENC_DEC_MODELS: List[str] = ["ddam"]

# Hyper-Parameter Optimization
HOP_KEYS: Tuple[str, ...] = (
    "hop_method",
    "hop_range",
    "hop_epochs",
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

# File system settings
CONFIRM_TIMEOUT: int = 30

FS_COMMANDS: List[str] = ["create", "read", "update", "delete", "cryptography"]

OPERATION_MAP: Dict[str, Callable[[Any, Any], Any]] = {
    "=": lambda x, y: y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "+": lambda x, y: x + y,
    "+=": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "-=": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "*=": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "/=": lambda x, y: x / y,
    "**": lambda x, y: x**y,
    "**=": lambda x, y: x**y,
    "//": lambda x, y: x // y,
    "//=": lambda x, y: x // y,
    "%": lambda x, y: x % y,
    "%=": lambda x, y: x % y,
    "": lambda x, y: x,
    "<<": lambda x, y: x << y,
    "<<=": lambda x, y: x << y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">>": lambda x, y: x >> y,
    ">>=": lambda x, y: x >> y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "|": lambda x, y: x | y,
    "|=": lambda x, y: x | y,
    "&": lambda x, y: x & y,
    "&=": lambda x, y: x & y,
    "^": lambda x, y: x ^ y,
    "^=": lambda x, y: x ^ y,
    "is": lambda x, y: x is y,
    "isnot": lambda x, y: x is not y,
    "in": lambda x, y: x in y,
    "notin": lambda x, y: x not in y,
    "@": lambda x, y: x @ y,
    "@=": lambda x, y: x @ y,
    "divmod": lambda x, y: divmod(x, y),
}

STATS_FUNCTION_MAP: Dict[str, Callable[..., Any]] = {
    "mean": statistics.mean,
    "stdev": statistics.stdev,
    "median": statistics.median,
    "mode": statistics.mode,
    "var": statistics.variance,
    "quant": statistics.quantiles,
    "size": len,
    "sum": sum,
    "min": min,
    "max": max,
}

# GUI settings
CTRL_C_TIMEOUT: float = 2.0

APP_STYLES: List[str] = ["fusion", "windows", "windowsxp", "macintosh"]

# Test suite settings
TEST_MODULES: Dict[str, str] = {
    "parser": "test_configs_parser.py",
    "train": "test_train_command.py",
    "mrl": "test_mrl_train_command.py",
    "hp_optim": "test_hp_optim_command.py",
    "gen_data": "test_gen_data_command.py",
    "eval": "test_eval_command.py",
    "test_sim": "test_test_command.py",
    "file_system": "test_file_system_command.py",
    "gui": "test_gui_command.py",
    "actions": "test_custom_actions.py",
    "edge_cases": "test_edge_cases.py",
    "layers": "test_model_layers.py",
    "scheduler": "test_lr_scheduler.py",
    "optimizer": "test_optimizer.py",
    "integration": "test_integration.py",
}
