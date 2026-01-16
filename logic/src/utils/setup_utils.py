"""
Setup utilities for initializing models, environments, and optimizers.

This module encapsulates the logic for:
- Configuring cost function weights
- Initializing Hierarchical RL (HRL) managers
- Loading pre-trained models
- Setting up solver environments (Gurobi)
- Creating model/baseline pairs
- Configuring optimizers and learning rate schedulers
"""

import os

import gurobipy as gp
import torch
import torch.optim as optim
from dotenv import dotenv_values

from logic.src.models import GATLSTManager
from logic.src.utils.crypto_utils import decrypt_file_data
from logic.src.utils.functions import get_inner_model, load_model, torch_load_cpu

from .definitions import ROOT_DIR


def setup_cost_weights(opts, def_val=1.0):
    """
    Sets up the cost weights dictionary based on problem type.

    Args:
        opts (dict): Options dictionary.
        def_val (float, optional): Default weight value. Defaults to 1.0.

    Returns:
        dict: Dictionary of cost weights (waste, length, overflows, etc.).
    """

    def _set_val(cost_weight, default_value):
        return default_value if cost_weight is None else cost_weight

    # def _set_weight(opts, cost_weight, default_value=1.):
    # return opts.get(cost_weight, default_value)

    cw_dict = {}
    if opts["problem"] in ["wcvrp", "cwcvrp", "sdwcvrp"]:
        cw_dict["waste"] = opts["w_waste"] = _set_val(opts["w_waste"], def_val)
        cw_dict["length"] = opts["w_length"] = _set_val(opts["w_length"], def_val)
        cw_dict["overflows"] = opts["w_overflows"] = _set_val(opts["w_overflows"], def_val)
    elif opts["problem"] in ["vrpp", "cvrpp"]:
        cw_dict["waste"] = opts["w_waste"] = _set_val(opts["w_waste"], def_val)
        cw_dict["length"] = opts["w_length"] = _set_val(opts["w_length"], def_val)
    return cw_dict


def setup_hrl_manager(opts, device, configs=None, policy=None, base_path=None, worker_model=None):
    """
    Initializes and loads the Manager model for Hierarchical RL.

    Args:
        opts (dict): Options dictionary.
        device: Torch device.
        configs (dict, optional): Configuration dictionary.
        policy (str, optional): Policy name.
        base_path (str, optional): Base path for models.
        worker_model (nn.Module, optional): Worker model instance for shared encoder.

    Returns:
        GATLSTManager or None: The initialized manager model, or None if not applicable.
    """
    hrl_path = None
    if opts.get("model_path") is not None:
        if policy in opts["model_path"]:
            hrl_path = opts["model_path"][policy]
        else:
            # Fallback: Try to find a match by stripping suffixes like _gamma1, _emp, etc.
            # This is common in the simulator where policy names are augmented.
            base_policy = policy.split("_gamma")[0].split("_emp")[0]
            if base_policy in opts["model_path"]:
                hrl_path = opts["model_path"][base_policy]
            else:
                # Last resort: check if any key in model_path is a prefix of our policy name
                for key in opts["model_path"].keys():
                    if policy.startswith(key):
                        hrl_path = opts["model_path"][key]
                        break

    if "mrl_method" not in configs or configs["mrl_method"] != "hrl":
        return None

    if base_path is not None and not os.path.exists(hrl_path):
        hrl_path = os.path.join(base_path, hrl_path)

    # --- Logic from load_model to handle directory ---
    if os.path.isfile(hrl_path):
        pass  # hrl_path is already the file
    elif os.path.isdir(hrl_path):
        # Find latest epoch
        epoch = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(hrl_path)
            if os.path.splitext(filename)[1] == ".pt" and "epoch" in filename
        )
        hrl_path = os.path.join(hrl_path, "epoch-{}.pt".format(epoch))
    else:
        # If explicitly requested but not found, maybe valid to return None or raise error?
        # For robustness, if we can't find it, we skip HRL
        return None

    # Get params from configs if available, else opts
    if configs is not None:
        mrl_history = configs.get("mrl_history", opts.get("mrl_history", 10))
        gat_hidden = configs.get("gat_hidden", opts.get("gat_hidden", 128))
        lstm_hidden = configs.get("lstm_hidden", opts.get("lstm_hidden", 64))
        global_input_dim = configs.get("global_input_dim", opts.get("global_input_dim", 3))
    else:
        mrl_history = opts.get("mrl_history", 10)
        gat_hidden = opts.get("gat_hidden", 128)
        lstm_hidden = opts.get("lstm_hidden", 64)
        global_input_dim = opts.get("global_input_dim", 3)

    # Load data first to inspect dimensions
    load_data = torch_load_cpu(hrl_path)
    if isinstance(load_data, dict) and "manager" in load_data:
        state_dict = load_data["manager"]
    else:
        state_dict = load_data

    # Detect global_input_dim from checkpoint if possible
    # weight shape: (hidden_dim, hidden_dim + global_input_dim)
    if "gate_head.0.weight" in state_dict:
        weight_shape = state_dict["gate_head.0.weight"].shape
        # weight_shape[1] is hidden_dim + global_input_dim
        # hidden_dim is usually 128 (gat_hidden)
        in_dim = weight_shape[1]
        detected_dim = in_dim - gat_hidden
        if detected_dim > 0 and detected_dim != global_input_dim:
            # print(f"Detected global_input_dim {detected_dim} from checkpoint (was {global_input_dim})")
            global_input_dim = detected_dim

    manager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=mrl_history,
        hidden_dim=gat_hidden,
        lstm_hidden=lstm_hidden,
        device=device,
        global_input_dim=global_input_dim,
        shared_encoder=(
            worker_model.embedder if (worker_model is not None and opts.get("shared_encoder", True)) else None
        ),
    ).to(device)

    manager.load_state_dict(state_dict)
    manager.eval()
    return manager


def setup_model(policy, general_path, model_paths, device, lock, temperature=1, decode_type="greedy"):
    """
    Sets up and loads a specific model based on policy.

    Args:
        policy (str): Policy identifier.
        general_path (str): Base path for models.
        model_paths (dict): Mapping of policy names to file paths.
        device: Torch device.
        lock: Threading lock for safe loading.
        temperature (float, optional): Softmax temperature. Defaults to 1.
        decode_type (str, optional): Decoding strategy. Defaults to "greedy".

    Returns:
        tuple: (model, configs)
    """

    def _load_model(general_path, model_name, device, temperature, decode_type, lock):
        model_path = os.path.join(general_path, model_name)
        with lock:
            model, configs = load_model(model_path)

        model.to(device)
        model.eval()
        model.set_decode_type(decode_type, temp=temperature)
        return model, configs

    pol_strip, _ = policy.rsplit("_", 1)
    return _load_model(general_path, model_paths[pol_strip], device, temperature, decode_type, lock)


def setup_env(policy, server=False, gplic_filename=None, symkey_name=None, env_filename=None):
    """
    Sets up the solver environment (e.g., Gurobi).

    Args:
        policy (str): Policy name to determine environment type.
        server (bool, optional): Whether running on a server (requires specific license handling).
        gplic_filename (str, optional): Gurobi license filename.
        symkey_name (str, optional): Symmetric key name for decryption.
        env_filename (str, optional): Environment variables filename.

    Returns:
        gp.Env or None: The Gurobi environment, or None if not applicable.
    """
    if "vrpp" in policy and "hexaly" not in policy:
        if server:

            def convert_int(param):
                """Helper to convert string parameters to int if possible."""
                return int(param) if param.isdigit() else param

            if gplic_filename is not None:
                gplic_path = os.path.join(ROOT_DIR, "assets", "api", gplic_filename)
                if symkey_name:
                    data = decrypt_file_data(gplic_path, symkey_name=symkey_name, env_filename=env_filename)
                else:
                    with open(gplic_path, "r") as gp_file:
                        data = gp_file.read()
                params = {
                    line.split("=")[0]: convert_int(line.split("=")[1]) for line in data.split("\n") if "=" in line
                }
            else:
                assert env_filename is not None
                env_path = os.path.join(ROOT_DIR, "env", env_filename)
                config = dotenv_values(env_path)
                glp_ls = ["WLSACCESSID", "WLSSECRET", "LICENSEID"]
                params = {glp: convert_int(config.get(glp, "")) for glp in glp_ls}
                for glp_key, glp_val in params.items():
                    if isinstance(glp_val, str) and glp_val == "":
                        raise ValueError(f"Missing parameter {glp_key} for Gurobi license")
        else:
            params = {}
            if gplic_filename is not None:
                gplic_path = os.path.join(ROOT_DIR, "assets", "api", gplic_filename)
                if os.path.exists(gplic_path):
                    os.environ["GRB_LICENSE_FILE"] = gplic_path
        params["OutputFlag"] = 0
        return gp.Env(params=params)


def setup_model_and_baseline(problem, data_load, use_cuda, opts):
    """
    Sets up the neural model and the reinforcement learning baseline.

    Args:
        problem: The problem instance (or class).
        data_load (dict): Loaded checkpoint data.
        use_cuda (bool): Whether to use CUDA.
        opts (dict): Options dictionary.

    Returns:
        tuple: (model, baseline)
    """
    from logic.src.models import (
        AttentionModel,
        CriticBaseline,
        CriticNetwork,
        DeepDecoderAttentionModel,
        # Encoders removed from import as they are handled by factory
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

    factory_class = {
        "gat": AttentionComponentFactory,
        "gcn": GCNComponentFactory,
        "gac": GACComponentFactory,
        "tgc": TGCComponentFactory,
        "ggac": GGACComponentFactory,
    }.get(opts["encoder"], None)

    assert factory_class is not None, "Unknown encoder: {}".format(opts["encoder"])

    factory = factory_class()

    model_class = {
        "am": AttentionModel,
        "tam": TemporalAttentionModel,
        "ddam": DeepDecoderAttentionModel,
    }.get(opts["model"], None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        opts["embedding_dim"],
        opts["hidden_dim"],
        problem,
        factory,  # Changed from encoder_class to factory
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

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **data_load.get("model", {})})

    # Initialize baseline
    if opts["baseline"] == "exponential":
        baseline = ExponentialBaseline(opts["exp_beta"])
    elif opts["baseline"] == "critic" or opts["baseline"] == "critic_lstm":
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

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in data_load:
        baseline.load_state_dict(data_load["baseline"])

    return model, baseline


def setup_optimizer_and_lr_scheduler(model, baseline, data_load, opts):
    """
    Sets up the optimizer and learning rate scheduler.

    Args:
        model (nn.Module): The actor model.
        baseline (Baseline): The RL baseline.
        data_load (dict): Loaded checkpoint data.
        opts (dict): Options dictionary.

    Returns:
        tuple: (optimizer, lr_scheduler)
    """
    optimizer_params = [{"params": model.parameters(), "lr": opts["lr_model"]}] + (
        [{"params": baseline.get_learnable_parameters(), "lr": opts["lr_critic_value"]}]
        if len(baseline.get_learnable_parameters()) > 0
        else []
    )
    optimizer_cls = {
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

    optimizer = optimizer_cls(optimizer_params)

    # Load optimizer state, make sure script is called with same type of optimizer
    if "optimizer" in data_load:
        optimizer.load_state_dict(data_load["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts["device"])

    # Initialize learning rate scheduler!
    scheduler_factory = {
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

    lr_scheduler = scheduler_factory(optimizer)
    return optimizer, lr_scheduler
