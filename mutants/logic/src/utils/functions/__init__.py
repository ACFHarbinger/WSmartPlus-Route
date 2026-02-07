"""
Mathematical and environment-related helper functions sub-package.
"""

# Re-exports from relocated packages for backward compatibility
from ..hooks.attention import add_attention_hooks
from ..model.loading import load_args, load_data, load_model, torch_load_cpu
from ..model.problem_factory import load_problem
from ..model.processing import get_inner_model, parse_softmax_temperature
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
    "add_attention_hooks",
    "run_all_in_pool",
    "get_path_until_string",
    "compute_in_batches",
    "do_batch_rep",
    "sample_many",
]
