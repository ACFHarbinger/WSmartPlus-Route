"""
Optimizer and learning rate scheduler setup utilities.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim


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
    assert optimizer_cls is not None, (
        f"Unknown optimizer: '{opts['optimizer']}'. "
        f"Valid options are: adam, adamw, adamax, nadam, radam, sadam, adadelta, adagrad, rmsprop, rprop, lbfgs, asgd, sgd"
    )

    optimizer: optim.Optimizer = optimizer_cls(optimizer_params)  # type: ignore

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
    assert scheduler_factory is not None, (
        f"Unknown learning rate scheduler: '{opts['lr_scheduler']}'. "
        f"Valid options are: exp, step, mult, lambda, const, poly, multistep, cosan, linear, cosanwr, plateau"
    )

    lr_scheduler: Any = scheduler_factory(optimizer)
    return optimizer, lr_scheduler
