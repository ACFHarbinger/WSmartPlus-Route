from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gurobipy as gp
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import dotenv_values

import logic.src.utils.definitions as udef
from logic.src.models import GATLSTManager
from logic.src.utils.crypto_utils import decrypt_file_data
from logic.src.utils.definitions import ROOT_DIR
from logic.src.utils.functions.function import (
    get_inner_model,
    load_model,
    torch_load_cpu,
)


def setup_cost_weights(opts: Dict[str, Any], def_val: float = 1.0) -> Dict[str, float]:
    """
    Sets up the cost weights dictionary based on problem type.

    Args:
        opts: Options dictionary.
        def_val: Default weight value. Defaults to 1.0.

    Returns:
        Dictionary of cost weights (waste, length, overflows, etc.).
    """

    def _set_val(cost_weight: Optional[float], default_value: float) -> float:
        return default_value if cost_weight is None else cost_weight

    cw_dict: Dict[str, float] = {}
    if opts["problem"] in udef.PROBLEMS:  # type: ignore
        cw_dict["waste"] = opts["w_waste"] = _set_val(opts["w_waste"], def_val)
        cw_dict["length"] = opts["w_length"] = _set_val(opts["w_length"], def_val)
        if "overflows" in opts or opts["problem"] in ["wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp", "swcvrp"]:
            cw_dict["overflows"] = opts["w_overflows"] = _set_val(opts.get("w_overflows"), def_val)
    return cw_dict


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
        model_path: str = os.path.join(general_path, model_name)
        with lock:
            model, configs = load_model(model_path)

        model.to(device)
        model.eval()
        if hasattr(model, "set_decode_type"):
            model.set_decode_type(decode_type, temp=temperature)
        return model, configs

    pol_strip: str = policy.rsplit("_", 1)[0]
    return _load_model(general_path, model_paths[pol_strip], device, temperature, decode_type, lock)


def setup_env(
    policy: str,
    server: bool = False,
    gplic_filename: Optional[str] = None,
    symkey_name: Optional[str] = None,
    env_filename: Optional[str] = None,
) -> Optional[gp.Env]:
    """
    Sets up the solver environment (e.g., Gurobi).

    Args:
        policy: Policy name to determine environment type.
        server: Whether running on a server (requires specific license handling).
        gplic_filename: Gurobi license filename.
        symkey_name: Symmetric key name for decryption.
        env_filename: Environment variables filename.

    Returns:
        The Gurobi environment, or None if not applicable.
    """
    if "vrpp" in policy and "hexaly" not in policy:
        params: Dict[str, Any] = {}
        if server:

            def convert_int(param: str) -> Union[int, str]:
                """Helper to convert string parameters to int if possible."""
                return int(param) if param.isdigit() else param

            if gplic_filename is not None:
                gplic_path: str = os.path.join(ROOT_DIR, "assets", "api", gplic_filename)
                if symkey_name:
                    data: str = decrypt_file_data(gplic_path, symkey_name=symkey_name, env_filename=env_filename)
                else:
                    with open(gplic_path, "r") as gp_file:
                        data = gp_file.read()
                params = {
                    line.split("=")[0]: convert_int(line.split("=")[1]) for line in data.split("\n") if "=" in line
                }
            else:
                assert env_filename is not None
                env_path: str = os.path.join(ROOT_DIR, "env", env_filename)
                config: Dict[str, Optional[str]] = dotenv_values(env_path)
                glp_ls: List[str] = ["WLSACCESSID", "WLSSECRET", "LICENSEID"]
                params = {glp: convert_int(config.get(glp, "")) for glp in glp_ls}  # type: ignore
                for glp_key, glp_val in params.items():
                    if isinstance(glp_val, str) and glp_val == "":
                        raise ValueError(f"Missing parameter {glp_key} for Gurobi license")
        else:
            if gplic_filename is not None:
                gplic_path = os.path.join(ROOT_DIR, "assets", "api", gplic_filename)
                if os.path.exists(gplic_path):
                    os.environ["GRB_LICENSE_FILE"] = gplic_path
        params["OutputFlag"] = 0
        return gp.Env(params=params)
    return None


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
        DeepDecoderAttentionModel,
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

    model_cls: Optional[Type[nn.Module]] = {
        "am": AttentionModel,
        "tam": TemporalAttentionModel,
        "ddam": DeepDecoderAttentionModel,
    }.get(opts["model"], None)
    assert model_cls is not None, "Unknown model: {}".format(opts["model"])

    model: nn.Module = model_cls(
        opts["embedding_dim"],
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
                    opts["embedding_dim"],
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
        baseline = RolloutBaseline(model, problem, opts)
    elif opts["baseline"] == "pomo":
        baseline = POMOBaseline(opts.get("pomo_size", 0))
    else:
        assert opts["baseline"] is None, "Unknown baseline: {}".format(opts["baseline"])
        baseline = NoBaseline()

    if opts["bl_warmup_epochs"] > 0:
        baseline = WarmupBaseline(baseline, opts["bl_warmup_epochs"], warmup_exp_beta=opts["exp_beta"])

    if "baseline" in data_load:
        baseline.load_state_dict(data_load["baseline"])

    return model, baseline


def setup_optimizer_and_lr_scheduler(
    model: nn.Module, baseline: Any, data_load: Dict[str, Any], opts: Dict[str, Any]
) -> Tuple[optim.Optimizer, Any]:
    """
    Sets up the optimizer and learning rate scheduler.

    Args:
        model: The actor model.
        baseline: The RL baseline.
        data_load: Loaded checkpoint data.
        opts: Options dictionary.

    Returns:
        tuple: (optimizer, lr_scheduler)
    """
    optimizer_params: List[Dict[str, Any]] = [{"params": model.parameters(), "lr": opts["lr_model"]}]
    learnable_params: List[Any] = baseline.get_learnable_parameters()
    if len(learnable_params) > 0:
        optimizer_params.append({"params": learnable_params, "lr": opts["lr_critic_value"]})

    optimizer_cls: Optional[Type[optim.Optimizer]] = {
        "adam": optim.Adam,
        "adamax": optim.Adamax,
        "adamw": optim.AdamW,
        "radam": optim.RAdam,
        "nadam": optim.NAdam,
        "sadam": optim.SparseAdam,
        "adadelta": optim.Adadelta,
        "adagrad": optim.Adagrad,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "lbfgs": optim.LBFGS,
        "asgd": optim.ASGD,
        "sgd": optim.SGD,
    }.get(opts["optimizer"], None)
    assert optimizer_cls is not None, "Unknown optimizer: {}".format(opts["optimizer"])

    optimizer: optim.Optimizer = optimizer_cls(optimizer_params)

    if "optimizer" in data_load:
        optimizer.load_state_dict(data_load["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts["device"])

    scheduler_factory: Optional[Callable[[optim.Optimizer], Any]] = {
        "exp": lambda opt: optim.lr_scheduler.ExponentialLR(opt, opts["lr_decay"]),
        "step": lambda opt: optim.lr_scheduler.StepLR(opt, opts["lrs_step_size"], opts["lr_decay"]),
        "mult": lambda opt: optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: opts["lr_decay"]),
        "lambda": lambda opt: optim.lr_scheduler.LambdaLR(opt, lambda epoch: opts["lr_decay"] ** epoch),
        "const": lambda opt: optim.lr_scheduler.ConstantLR(opt, opts["lr_decay"], opts["lrs_total_steps"]),
        "poly": lambda opt: optim.lr_scheduler.PolynomialLR(opt, opts["lrs_total_steps"], opts["lr_decay"]),
        "multistep": lambda opt: optim.lr_scheduler.MultiStepLR(opt, opts["lrs_milestones"], opts["lr_decay"]),
        "cosan": lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, opts["lrs_total_steps"], opts["lr_min_value"]),
        "linear": lambda opt: optim.lr_scheduler.LinearLR(
            opt, opts["lr_min_decay"], opts["lr_decay"], opts["lrs_total_steps"]
        ),
        "cosanwr": lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            opts["lrs_restart_steps"],
            opts["lrs_rfactor"],
            opts["lr_min_value"],
        ),
        "plateau": lambda opt: optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            opts["lrs_mode"],
            opts["lrs_dfactor"],
            opts["lrs_patience"],
            opts["lrs_thresh"],
            opts["lrs_thresh_mode"],
            opts["lrs_cooldown"],
            opts["lr_min_value"],
            opts["lr_min_decay"],
        ),
    }.get(opts["lr_scheduler"], None)
    assert scheduler_factory is not None, "Unknown learning rate scheduler: {}".format(opts["lr_scheduler"])

    lr_scheduler: Any = scheduler_factory(optimizer)
    return optimizer, lr_scheduler
