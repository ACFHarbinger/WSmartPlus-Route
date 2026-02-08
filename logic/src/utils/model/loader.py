"""
Central model loading utility.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple, Type, cast

import torch.nn as nn
from omegaconf import OmegaConf

from .checkpoint_utils import torch_load_cpu
from .config_utils import load_args
from .problem_factory import load_problem


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
        TemporalAttentionModel,
    )
    from logic.src.models.subnets.factories.attention import AttentionComponentFactory
    from logic.src.models.subnets.factories.base import NeuralComponentFactory
    from logic.src.models.subnets.factories.gac import GACComponentFactory
    from logic.src.models.subnets.factories.gcn import GCNComponentFactory
    from logic.src.models.subnets.factories.ggac import GGACComponentFactory
    from logic.src.models.subnets.factories.mlp import MLPComponentFactory
    from logic.src.models.subnets.factories.tgc import TGCComponentFactory

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            pt_files = [f for f in os.listdir(path) if f.endswith(".pt")]
            # Filter for epoch-N.pt or epochN.pt
            epochs = []
            for f in pt_files:
                name = os.path.splitext(f)[0]
                m = re.match(r"epoch-?(\d+)", name)
                if m:
                    epochs.append(int(m.group(1)))

            if not epochs:
                raise ValueError("No valid epoch files (epoch-N.pt or epochN.pt) found in directory: {}".format(path))
            epoch = max(epochs)

        # Check if version with hyphen exists first
        hyphen_path = os.path.join(path, "epoch-{}.pt".format(epoch))
        if os.path.exists(hyphen_path):
            model_filename = hyphen_path
        else:
            model_filename = os.path.join(path, "epoch{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    # Load hyperparameters
    args = {}
    config_yaml_path = os.path.join(path, "config.yaml")
    hparams_yaml_path = os.path.join(path, "hparams.yaml")
    args_json_path = os.path.join(path, "args.json")

    if os.path.exists(config_yaml_path) or os.path.exists(hparams_yaml_path):
        yaml_path = config_yaml_path if os.path.exists(config_yaml_path) else hparams_yaml_path
        cfg = OmegaConf.load(yaml_path)

        if "model" in cfg and "env" in cfg:
            # Full Hydra Config
            args = {
                "problem": cfg.env.name,
                "encoder": cfg.model.encoder_type,
                "model": cfg.model.name,
                "embed_dim": cfg.model.embed_dim,
                "hidden_dim": cfg.model.hidden_dim,
                "n_heads": cfg.model.n_heads,
                "n_encode_layers": cfg.model.n_encoder_layers,
                "n_encode_sublayers": cfg.model.n_encoder_sublayers,
                "n_decode_layers": cfg.model.n_decoder_layers,
                "n_predict_layers": cfg.model.n_predictor_layers,
                "normalization": cfg.model.normalization,
                "activation": cfg.model.activation,
                "dropout": cfg.model.dropout,
                "tanh_clipping": cfg.model.tanh_clipping,
                "learn_affine": cfg.model.learn_affine,
                "track_stats": cfg.model.track_stats,
                "epsilon_alpha": cfg.model.epsilon_alpha,
                "momentum_beta": cfg.model.momentum_beta,
                "lrnorm_k": cfg.model.lrnorm_k,
                "gnorm_groups": cfg.model.gnorm_groups,
                "af_param": cfg.model.activation_param,
                "af_threshold": cfg.model.activation_threshold,
                "af_replacement": cfg.model.activation_replacement,
                "af_num_params": cfg.model.activation_num_parameters,
                "af_urange": cfg.model.activation_uniform_range,
                "aggregation": cfg.model.aggregation_node,
                "aggregation_graph": cfg.model.aggregation_graph,
                "spatial_bias": cfg.model.spatial_bias,
                "spatial_bias_scale": cfg.model.spatial_bias_scale,
                "shrink_size": cfg.model.shrink_size,
                "temporal_horizon": cfg.model.temporal_horizon,
                "mask_inner": cfg.model.mask_inner,
                "mask_logits": cfg.model.mask_logits,
                "mask_graph": cfg.model.mask_graph,
                "checkpoint_encoder": cfg.model.checkpoint_encoder,
            }
            if "rl" in cfg:
                args["entropy_weight"] = cfg.rl.entropy_weight
        else:
            # Maybe it's a flat DictConfig or Lightning hparams
            args = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            if "embed_dim" in args and "embed_dim" not in args:
                args["embed_dim"] = args["embed_dim"]
            if "num_encoder_layers" in args and "n_encode_layers" not in args:
                args["n_encode_layers"] = args["num_encoder_layers"]
            if "n_heads" in args and "n_heads" not in args:
                args["n_heads"] = args["n_heads"]
    elif os.path.exists(args_json_path):
        args = load_args(args_json_path)

    if not args:
        raise ValueError("Could not find hyperparameters in directory: {}".format(path))

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

    factory_type = cast(Type[NeuralComponentFactory], factory_class)
    component_factory = factory_type()

    # Map model name to decoder_type for backward compatibility
    decoder_type_map = {
        "am": "attention",
        "tam": "attention",
        "ddam": "deep",
    }
    decoder_type = decoder_type_map.get(args.get("model", "am"), "attention")

    # Use TemporalAttentionModel only for 'tam', otherwise AttentionModel
    model_class = TemporalAttentionModel if args.get("model") == "tam" else AttentionModel
    model = model_class(
        args["embed_dim"],
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
        predictor_layers=args.get("n_predict_layers", 0),
        decoder_type=decoder_type,
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
            new_state_dict[key] = value

    model.load_state_dict({**model.state_dict(), **new_state_dict})
    print("  [*] Loaded model from {}".format(model_filename))
    model.eval()
    return model, args
