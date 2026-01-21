"""
Mathematical and environment-related helper functions.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from multiprocessing.dummy import Pool as ThreadPool
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

T = TypeVar("T")


# Attention, Learn to Solve Routing Problems
def get_inner_model(model: nn.Module) -> nn.Module:
    """
    Returns the underlying model from a DataParallel wrapper if present.

    Args:
        model: The model (potentially wrapped).

    Returns:
        The inner model.
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_problem(name: str) -> Type[Any]:
    """
    Factory function to load a problem class by name.

    Args:
        name: The problem name (e.g., 'vrpp', 'wcvrp').

    Returns:
        The problem class.

    Raises:
        AssertionError: If problem name is unsupported.
    """
    from logic.src.tasks import CVRPP, CWCVRP, SCWCVRP, SDWCVRP, VRPP, WCVRP

    problem = {
        "vrpp": VRPP,
        "cvrpp": CVRPP,
        "wcvrp": WCVRP,
        "cwcvrp": CWCVRP,
        "sdwcvrp": SDWCVRP,
        "scwcvrp": SCWCVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path: str) -> Any:
    """
    Loads a checkpoint file mapping all tensors to CPU.

    Args:
        load_path: Path to the checkpoint file.

    Returns:
        The loaded data.
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def load_data(load_path: Optional[str], resume: Optional[str]) -> Any:
    """
    Loads data from a path or resume checkpoint.

    Args:
        load_path: Explicit path to data.
        resume: Path to resume checkpoint.

    Returns:
        Loaded data or empty dict if neither is provided.
    """
    data = {}
    assert load_path is None or resume is None, "Only one of load path and resume can be given"

    load_path = load_path if load_path is not None else resume
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        data = torch_load_cpu(load_path)
    return data


def move_to(var: T, device: torch.device, non_blocking: bool = False) -> T:
    """
    Recursively moves variables to the specified device.
    Supports dicts and Tensors.

    Args:
        var: The variable to move.
        device: The target device.
        non_blocking: If True and if the variable is a Tensor on pinned memory,
            the copy will be asynchronous with respect to the host. Defaults to False.

    Returns:
        The variable on the new device.
    """
    if var is None:
        return var  # type: ignore
    if isinstance(var, dict):
        return {k: move_to(v, device, non_blocking=non_blocking) for k, v in var.items()}  # type: ignore
    if isinstance(var, torch.Tensor):
        return var.to(device, non_blocking=non_blocking)  # type: ignore
    return var


def _load_model_file(load_path: str, model: nn.Module) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Loads the model with parameters from the file and returns optimizer state dict if it is in the file.

    Args:
        load_path: Path to the checkpoint.
        model: Model to load parameters into.

    Returns:
        Tuple of (model, optimizer state dict).
    """
    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print("  [*] Loading model from {}".format(load_path))

    load_data = torch.load(os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage)
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict


def load_args(filename: str) -> Dict[str, Any]:
    """
    Loads argument configuration from a JSON file.
    Handles deprecated keys for backward compatibility.

    Args:
        filename: Path to args.json.

    Returns:
        The loaded arguments.
    """
    with open(filename, "r") as f:
        args = json.load(f)

    # Backwards compatibility
    if "data_distribution" not in args:
        args["data_distribution"] = None
        probl, *dist = args["problem"].split("_")
        if probl in ("vrpp", "wcvrp"):
            args["problem"] = probl
            args["data_distribution"] = dist[0]

    if "aggregation_graph" not in args:
        args["aggregation_graph"] = "avg"
    return args


def load_model(path: str, epoch: Optional[int] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads the entire model from a checkpoint or directory.

    Args:
        path: Path to checkpoint file or directory containing checkpoints.
        epoch: Specific epoch to load if path is a directory. If None, loads latest.

    Returns:
        tuple: (model, args)

    Raises:
        ValueError: If no valid epoch files found in directory.
    """
    from logic.src.models import (
        AttentionModel,
        DeepDecoderAttentionModel,
        TemporalAttentionModel,
    )
    from logic.src.models.model_factory import (
        AttentionComponentFactory,
        GACComponentFactory,
        GCNComponentFactory,
        GGACComponentFactory,
        MLPComponentFactory,
        TGCComponentFactory,
    )

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            pt_files = [f for f in os.listdir(path) if f.endswith(".pt")]
            epochs = []
            for f in pt_files:
                name = os.path.splitext(f)[0]
                if "-" in name:
                    parts = name.split("-")
                    if len(parts) == 2 and parts[0] == "epoch" and parts[1].isdigit():
                        epochs.append(int(parts[1]))

            if not epochs:
                raise ValueError("No valid epoch files (epoch-N.pt) found in directory: {}".format(path))
            epoch = max(epochs)
        model_filename = os.path.join(path, "epoch-{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, "args.json"))
    problem = load_problem(args["problem"])

    # Map encoder name to Factory Class
    factory_class = {
        "gat": AttentionComponentFactory,
        "gac": GACComponentFactory,
        "tgc": TGCComponentFactory,
        "ggac": GGACComponentFactory,
        "mlp": MLPComponentFactory,
        "gcn": GCNComponentFactory,
    }.get(args.get("encoder", "gat"), None)

    # Fallback/Check
    assert factory_class is not None, "Unknown encoder type: {}".format(args.get("encoder", "gat"))

    component_factory = factory_class()

    model_class = {
        "am": AttentionModel,
        "tam": TemporalAttentionModel,
        "ddam": DeepDecoderAttentionModel,
    }.get(args.get("model", "am"), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        args["embedding_dim"],
        args["hidden_dim"],
        problem,
        component_factory,
        args["n_encode_layers"],
        args["n_encode_sublayers"],
        args["n_decode_layers"],
        n_heads=args["n_heads"],
        normalization=args["normalization"],
        norm_learn_affine=args["learn_affine"],
        norm_track_stats=args["track_stats"],
        norm_eps_alpha=args["epsilon_alpha"],
        norm_momentum_beta=args["momentum_beta"],
        lrnorm_k=args["lrnorm_k"],
        gnorm_groups=args["gnorm_groups"],
        activation_function=args["activation"],
        af_param=args["af_param"],
        af_threshold=args["af_threshold"],
        af_replacement_value=args["af_replacement"],
        af_num_params=args["af_nparams"],
        af_uniform_range=args["af_urange"],
        dropout_rate=args["dropout"],
        aggregation=args["aggregation"],
        aggregation_graph=args["aggregation_graph"],
        tanh_clipping=args["tanh_clipping"],
        mask_inner=args.get("mask_inner", True),
        mask_logits=args.get("mask_logits", True),
        mask_graph=args.get("mask_graph", False),
        checkpoint_encoder=args.get("checkpoint_encoder", False),
        shrink_size=args.get("shrink_size", None),
        temporal_horizon=args.get("temporal_horizon", 0),
        spatial_bias=args.get("spatial_bias", False),
        spatial_bias_scale=args.get("spatial_bias_scale", 1.0),
        entropy_weight=args.get("entropy_weight", 0.0),
        predictor_layers=args.get("n_predict_layers", None),
    )

    # Overwrite model parameters by parameters to load
    data = torch_load_cpu(model_filename)
    loaded_state_dict = data.get("model", {})

    # Migration for Abstract Factory Refactoring
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key in model_state_dict:
            new_state_dict[key] = value
        elif "decoder." + key in model_state_dict:
            new_state_dict["decoder." + key] = value
        elif "context_embedder." + key in model_state_dict:
            new_state_dict["context_embedder." + key] = value
        else:
            # Keep original key (might cause error if strict=True, but let's try)
            new_state_dict[key] = value

    model.load_state_dict({**model.state_dict(), **new_state_dict})
    print("  [*] Loaded model from {}".format(model_filename))
    model.eval()  # Put in eval mode
    return model, args


def parse_softmax_temperature(raw_temp: Union[str, float]) -> float:
    """
    Parses softmax temperature, supporting loading from a file (schedule) or a float.

    Args:
        raw_temp: The raw temperature argument.

    Returns:
        The parsed temperature.
    """
    # Load from file
    if isinstance(raw_temp, str) and os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(
    func: Callable[..., Any],
    directory: str,
    dataset: List[Any],
    opts: Any,
    use_multiprocessing: bool = True,
) -> Tuple[List[Any], int]:
    """
    Runs a function over a dataset in parallel using multiprocessing or threading.

    Args:
        func: The function to execute.
        directory: Directory context for the function.
        dataset: List of problem instances.
        opts: Options including 'cpus', 'offset', 'n'.
        use_multiprocessing: Whether to use process pool or thread pool. Defaults to True.

    Returns:
        tuple: (results, num_cpus)
    """
    num_cpus = (os.cpu_count() or 1) if opts.cpus is None else opts.cpus
    w = len(str(len(dataset) - 1))
    offset = getattr(opts, "offset", None)
    if offset is None:
        offset = 0

    ds = dataset[offset : (offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = mp.Pool if use_multiprocessing and num_cpus > 1 else ThreadPool
    with pool_cls(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    func,
                    [(directory, str(i + offset).zfill(w), *problem) for i, problem in enumerate(ds)],
                ),
                total=len(ds),
                mininterval=opts.progress_bar_mininterval,
            )
        )

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def get_path_until_string(path: str, end_str: str) -> Optional[str]:
    """
    Truncates a path up to the first occurrence of a specific directory component.

    Args:
        path: The full path.
        end_str: The directory name to truncate at.

    Returns:
        The truncated path or None if end_str is not found.
    """
    path_ls = str.split(path, os.sep)
    try:
        idx = path_ls.index(end_str)
        return os.sep.join(path_ls[: idx + 1])
    except ValueError:
        print(f"Path '{path}' does not contain '{end_str}'")
        return None


# Tensor functions
def compute_in_batches(
    f: Callable[..., Any], calc_batch_size: int, *args: torch.Tensor, n: Optional[int] = None
) -> Any:
    """
    Computes memory heavy function f(*args) in batches.

    Args:
        f: The function that is computed, should take only tensors as arguments and
            return tensor or tuple of tensors.
        calc_batch_size: The batch size to use when computing this function.
        *args: Tensor arguments with equally sized first batch dimension.
        n: the total number of elements.

    Returns:
        f(*args), this should be one or multiple tensors.
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    all_res = [f(*(arg[i * calc_batch_size : (i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks: List[Optional[torch.Tensor]], dim: int = 0) -> Optional[torch.Tensor]:
        """Concatenates tensors safely, handling empty chunks."""
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)  # type: ignore

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)


def add_attention_hooks(model_module: nn.Module) -> Dict[str, Any]:
    """
    Registers forward hooks on Multi-Head Attention layers to capture weights and masks.

    Args:
        model_module: The model to hook.

    Returns:
        dict: Dictionary containing lists of 'weights', 'masks', and 'handles'.
    """
    graph_masks = []
    attention_weights = []

    def hook(module: nn.Module, input: Any, output: Any) -> None:
        """Forward hook to capture attention weights."""
        if hasattr(module, "last_attn") and module.last_attn is not None:
            graph_masks.append(module.last_attn[-1])
            attention_weights.append(module.last_attn[0])

    # Register hooks on all MHA layers
    hook_data = {"weights": attention_weights, "masks": graph_masks, "handles": []}
    for layer in model_module.layers:  # type: ignore
        # Get the actual attention module (skip the SkipConnection wrapper), if layer has attention
        if not hasattr(layer, "att"):
            continue
        attention_module = layer.att.module

        # Register hook and store the handle
        hook_handle = attention_module.register_forward_hook(hook)
        hook_data["handles"].append(hook_handle)
    return hook_data


# Sampling functions
def do_batch_rep(v: Any, n: int) -> Any:
    """
    Replicates a variable n times along the batch dimension.

    Args:
        v: The variable (tensor, or structure containing tensors).
        n: Number of repetitions.

    Returns:
        Replicated variable.
    """
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)
    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(
    inner_func: Callable[..., Any],
    get_cost_func: Callable[..., Any],
    input: Any,
    batch_rep: int = 1,
    iter_rep: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples many solutions by repeated execution.

    Args:
        inner_func: Function producing policy and log probabilities.
        get_cost_func: Function computing costs and mask.
        input: Input node features.
        batch_rep: Batch replication factor. Defaults to 1.
        iter_rep: Iteration replication. Defaults to 1.

    Returns:
        tuple: (min_policies, min_costs)
    """
    input = do_batch_rep(input, batch_rep)
    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)

        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)

    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat([F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis], 1)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)

    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    return minpis, mincosts
