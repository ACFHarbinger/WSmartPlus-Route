"""
Model utilities package.

Attributes:
    _load_model_file: Loads the model with parameters from the file and returns optimizer state dict if it is in the file.
    load_data: Loads data from a path or resume checkpoint.
    torch_load_cpu: Loads a checkpoint file mapping all tensors to CPU.
    load_args: Loads argument configuration from a JSON file.
    load_model: Loads a trained model from a checkpoint file.
    load_problem: Loads a problem instance from a file.
    get_inner_model: Extracts the inner model from a wrapped model.
    parse_softmax_temperature: Parses the softmax temperature from a string.

Example:
    >>> from logic.src.utils.model import load_model, load_problem, get_inner_model, parse_softmax_temperature
    >>> model = load_model("path/to/checkpoint.pt")
    >>> problem = load_problem("path/to/problem.pt")
    >>> inner_model = get_inner_model(model)
    >>> temperature = parse_softmax_temperature("1.0")
"""

from .checkpoint_utils import _load_model_file, load_data, torch_load_cpu
from .config_utils import load_args
from .loader import load_model, load_problem
from .processing import get_inner_model, parse_softmax_temperature

__all__ = [
    "get_inner_model",
    "torch_load_cpu",
    "load_data",
    "_load_model_file",
    "load_args",
    "load_model",
    "load_problem",
    "parse_softmax_temperature",
]
