"""
Mathematical and environment-related helper functions (Facade).
"""

from .attention import add_attention_hooks
from .factory import load_problem
from .model import (
    get_inner_model,
    load_args,
    load_data,
    load_model,
    parse_softmax_temperature,
    torch_load_cpu,
)
from .parallel import run_all_in_pool
from .path import get_path_until_string
from .sampling import sample_many
from .tensors import compute_in_batches, do_batch_rep, move_to

__all__ = [
    "get_inner_model",
    "load_problem",
    "torch_load_cpu",
    "load_data",
    "move_to",
    "load_args",
    "load_model",
    "parse_softmax_temperature",
    "run_all_in_pool",
    "get_path_until_string",
    "compute_in_batches",
    "add_attention_hooks",
    "do_batch_rep",
    "sample_many",
]
