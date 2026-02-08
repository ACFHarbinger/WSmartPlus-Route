"""
Training model and baseline setup utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from logic.src.utils.functions import get_inner_model


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
    from logic.src.models.subnets.factories.attention import AttentionComponentFactory
    from logic.src.models.subnets.factories.gac import GACComponentFactory
    from logic.src.models.subnets.factories.gcn import GCNComponentFactory
    from logic.src.models.subnets.factories.ggac import GGACComponentFactory
    from logic.src.models.subnets.factories.tgc import TGCComponentFactory

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
