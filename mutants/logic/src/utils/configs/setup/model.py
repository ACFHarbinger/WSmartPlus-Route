"""
Model and baseline setup utilities.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from logic.src.models import GATLSTManager
from logic.src.utils.functions.function import (
    get_inner_model,
    load_model,
    torch_load_cpu,
)


def setup_hrl_manager(
    opts: Dict[str, Any],
    device: torch.device,
    configs: Optional[Dict[str, Any]] = None,
    policy: Optional[str] = None,
    base_path: Optional[str] = None,
    worker_model: Optional[nn.Module] = None,
) -> Optional[GATLSTManager]:
    """
    Initializes and loads the Manager model for Hierarchical RL.

    Args:
        opts: Options dictionary.
        device: Torch device.
        configs: Configuration dictionary.
        policy: Policy name.
        base_path: Base path for models.
        worker_model: Worker model instance for shared encoder.

    Returns:
        The initialized manager model, or None if not applicable.
    """
    if configs is None:
        configs = {}

    hrl_path: Optional[str] = None
    if opts.get("model_path") is not None and policy is not None:
        model_paths: Dict[str, str] = opts["model_path"]
        if policy in model_paths:
            hrl_path = model_paths[policy]
        else:
            base_policy: str = policy.split("_gamma")[0].split("_emp")[0]
            if base_policy in model_paths:
                hrl_path = model_paths[base_policy]
            else:
                for key in model_paths.keys():
                    if policy.startswith(key):
                        hrl_path = model_paths[key]
                        break

    if configs.get("mrl_method") != "hrl":
        return None

    if hrl_path is None:
        return None

    if base_path is not None and not os.path.exists(hrl_path):
        hrl_path = os.path.join(base_path, hrl_path)

    if os.path.isfile(hrl_path):
        pass
    elif os.path.isdir(hrl_path):
        pt_files: List[str] = [f for f in os.listdir(hrl_path) if f.endswith(".pt") and "epoch" in f]
        if not pt_files:
            return None
        epoch: int = max(int(os.path.splitext(f)[0].split("-")[1]) for f in pt_files)
        hrl_path = os.path.join(hrl_path, f"epoch-{epoch}.pt")
    else:
        return None

    mrl_history: int = configs.get("mrl_history", opts.get("mrl_history", 10))
    gat_hidden: int = configs.get("gat_hidden", opts.get("gat_hidden", 128))
    lstm_hidden: int = configs.get("lstm_hidden", opts.get("lstm_hidden", 64))
    global_input_dim: int = configs.get("global_input_dim", opts.get("global_input_dim", 3))

    load_data: Any = torch_load_cpu(hrl_path)
    state_dict: Dict[str, torch.Tensor] = (
        load_data["manager"] if isinstance(load_data, dict) and "manager" in load_data else load_data
    )

    if "gate_head.0.weight" in state_dict:
        weight_shape: torch.Size = state_dict["gate_head.0.weight"].shape
        in_dim: int = weight_shape[1]
        detected_dim: int = in_dim - gat_hidden
        if detected_dim > 0 and detected_dim != global_input_dim:
            global_input_dim = detected_dim

    manager: GATLSTManager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=mrl_history,
        hidden_dim=gat_hidden,
        lstm_hidden=lstm_hidden,
        device=device,
        global_input_dim=global_input_dim,
        shared_encoder=(
            worker_model.embedder
            if (worker_model is not None and opts.get("shared_encoder", True) and hasattr(worker_model, "embedder"))
            else None  # type: ignore
        ),
    ).to(device)

    manager.load_state_dict(state_dict)
    manager.eval()
    return manager


def setup_model(
    policy: str,
    general_path: str,
    model_paths: Dict[str, str],
    device: torch.device,
    lock: threading.Lock,
    temperature: float = 1.0,
    decode_type: str = "greedy",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Sets up and loads a specific model based on policy.

    Args:
        policy: Policy identifier.
        general_path: Base path for models.
        model_paths: Mapping of policy names to file paths.
        device: Torch device.
        lock: Threading lock for safe loading.
        temperature: Softmax temperature. Defaults to 1.
        decode_type: Decoding strategy. Defaults to "greedy".

    Returns:
        tuple: (model, configs)
    """

    def _load_model(
        general_path: str,
        model_name: str,
        device: torch.device,
        temperature: float,
        decode_type: str,
        lock: threading.Lock,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        # Robust path handling: only join if model_name does not exist on its own
        if os.path.isabs(model_name) or os.path.exists(model_name):
            model_path = model_name
        else:
            model_path = os.path.join(general_path, model_name)

        if not os.path.exists(model_path):
            # Try joining with ROOT_DIR as fallback if it's a semi-absolute path like 'assets/...'
            from logic.src.constants import ROOT_DIR

            root_joined = os.path.join(ROOT_DIR, model_name)
            if os.path.exists(root_joined):
                model_path = root_joined

        with lock:
            model, configs = load_model(model_path)

        model.to(device)
        model.eval()
        if hasattr(model, "set_decode_type"):
            model.set_decode_type(decode_type, temp=temperature)
        return model, configs

    pol_strip: str = policy.rsplit("_", 1)[0]
    model_name = model_paths.get(pol_strip)

    if model_name is None:
        # Robust lookup: Try to find a key in model_paths that is exactly matching or a subset of the policy string
        # e.g. key 'amgat' should match 'means_std0.84_neural_amgat_gamma1'
        # Sort keys by length (desc) to find the most specific match first
        for key in sorted(model_paths.keys(), key=len, reverse=True):
            if key in policy:
                model_name = model_paths[key]
                break

    if model_name is None:
        raise KeyError(
            f"Could not find model path for policy '{policy}'. "
            f"Available keys in model_paths: {list(model_paths.keys())}"
        )

    return _load_model(general_path, model_name, device, temperature, decode_type, lock)


def setup_model_and_baseline(
    problem: Any, data_load: Dict[str, Any], use_cuda: bool, opts: Dict[str, Any]
) -> Tuple[nn.Module, Any]:
    """
    Sets up the neural model and the reinforcement learning baseline.

    Args:
        problem: The problem instance (or class).
        data_load: Loaded checkpoint data.
        use_cuda: Whether to use CUDA.
        opts: Options dictionary.

    Returns:
        tuple: (model, baseline)
    """
    from logic.src.models import (
        AttentionModel,
        CriticBaseline,
        CriticNetwork,
        ExponentialBaseline,
        NoBaseline,
        POMOBaseline,
        RolloutBaseline,
        TemporalAttentionModel,
        WarmupBaseline,
    )
    from logic.src.models.model_factory import (
        AttentionComponentFactory,
        GACComponentFactory,
        GCNComponentFactory,
        GGACComponentFactory,
        TGCComponentFactory,
    )

    factory_cls: Optional[Type[Any]] = {
        "gat": AttentionComponentFactory,
        "gcn": GCNComponentFactory,
        "gac": GACComponentFactory,
        "tgc": TGCComponentFactory,
        "ggac": GGACComponentFactory,
    }.get(opts["encoder"], None)

    assert factory_cls is not None, "Unknown encoder: {}".format(opts["encoder"])

    factory: Any = factory_cls()

    # Map model name to decoder_type for backward compatibility
    decoder_type_map: Dict[str, str] = {
        "am": "attention",
        "tam": "attention",
        "ddam": "deep",
    }
    decoder_type: str = decoder_type_map.get(opts["model"], "attention")

    # Use TemporalAttentionModel only for 'tam', otherwise AttentionModel
    model_cls: Type[nn.Module] = TemporalAttentionModel if opts["model"] == "tam" else AttentionModel

    model: nn.Module = model_cls(
        opts["embed_dim"],
        opts["hidden_dim"],
        problem,
        factory,
        opts["n_encode_layers"],
        opts["n_encode_sublayers"],
        opts["n_decode_layers"],
        n_heads=opts["n_heads"],
        normalization=opts["normalization"],
        norm_learn_affine=opts["learn_affine"],
        norm_track_stats=opts["track_stats"],
        norm_eps_alpha=opts["epsilon_alpha"],
        norm_momentum_beta=opts["momentum_beta"],
        lrnorm_k=opts["lrnorm_k"],
        gnorm_groups=opts["gnorm_groups"],
        activation_function=opts["activation"],
        af_param=opts["af_param"],
        af_threshold=opts["af_threshold"],
        af_replacement_value=opts["af_replacement"],
        af_num_params=opts["af_nparams"],
        af_uniform_range=opts["af_urange"],
        dropout_rate=opts["dropout"],
        aggregation=opts["aggregation"],
        aggregation_graph=opts["aggregation_graph"],
        tanh_clipping=opts["tanh_clipping"],
        mask_inner=opts["mask_inner"],
        mask_logits=opts["mask_logits"],
        mask_graph=opts["mask_graph"],
        checkpoint_encoder=opts["checkpoint_encoder"],
        shrink_size=opts["shrink_size"],
        pomo_size=opts.get("pomo_size", 0),
        temporal_horizon=opts["temporal_horizon"],
        spatial_bias=opts.get("spatial_bias", False),
        spatial_bias_scale=opts.get("spatial_bias_scale", 1.0),
        entropy_weight=opts.get("entropy_weight", 0.0),
        predictor_layers=opts["n_predict_layers"],
        decoder_type=decoder_type,
    ).to(opts["device"])

    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_inner: nn.Module = get_inner_model(model)
    model_inner.load_state_dict({**model_inner.state_dict(), **data_load.get("model", {})})

    baseline: Any
    if opts["baseline"] == "exponential":
        baseline = ExponentialBaseline(opts["exp_beta"])
    elif opts["baseline"] in ["critic", "critic_lstm"]:
        baseline = CriticBaseline(
            (
                CriticNetwork(
                    problem,
                    factory,
                    opts["embed_dim"],
                    opts["hidden_dim"],
                    opts["n_encode_layers"],
                    opts["n_other_layers"],
                    opts["normalization"],
                    opts["activation"],
                    temporal_horizon=opts["temporal_horizon"],
                )
            ).to(opts["device"])
        )
    elif opts["baseline"] == "rollout":
        baseline = RolloutBaseline(policy=model, update_every=opts.get("bl_update_every", 1))
    elif opts["baseline"] == "pomo":
        baseline = POMOBaseline()
    else:
        assert opts["baseline"] is None, "Unknown baseline: {}".format(opts["baseline"])
        baseline = NoBaseline()

    if opts["bl_warmup_epochs"] > 0:
        baseline = WarmupBaseline(baseline, opts["bl_warmup_epochs"], warmup_exp_beta=opts["exp_beta"])

    if "baseline" in data_load:
        baseline.load_state_dict(data_load["baseline"])

    return model, baseline
