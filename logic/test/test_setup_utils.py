"""Tests for setup_utils.py."""

from unittest.mock import MagicMock

import torch

from logic.src.utils.setup_utils import setup_cost_weights, setup_optimizer_and_lr_scheduler


def test_setup_cost_weights():
    """Verify cost weight setup."""
    opts = {
        "problem": "vrpp",
        "w_waste": None,
        "w_length": 2.0,
    }
    cw = setup_cost_weights(opts, def_val=1.0)
    assert cw["waste"] == 1.0
    assert cw["length"] == 2.0
    assert opts["w_waste"] == 1.0


def test_setup_optimizer_and_lr_scheduler():
    """Verify optimizer and scheduler setup."""
    model = torch.nn.Linear(10, 1)
    baseline = MagicMock()
    baseline.get_learnable_parameters.return_value = []

    opts = {"optimizer": "adam", "lr_model": 1e-3, "lr_scheduler": "exp", "lr_decay": 0.99, "device": "cpu"}

    opt, sched = setup_optimizer_and_lr_scheduler(model, baseline, {}, opts)
    assert isinstance(opt, torch.optim.Adam)
    assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)


def test_setup_optimizer_available_types():
    """Verify various optimizer types."""
    model = torch.nn.Linear(1, 1)
    baseline = MagicMock()
    baseline.get_learnable_parameters.return_value = []

    for opt_name in ["adamax", "adamw", "rmsprop", "sgd"]:
        opts = {
            "optimizer": opt_name,
            "lr_model": 1e-3,
            "lr_scheduler": "step",
            "lrs_step_size": 1,
            "lr_decay": 0.9,
            "device": "cpu",
        }
        opt, _ = setup_optimizer_and_lr_scheduler(model, baseline, {}, opts)
        assert opt.__class__.__name__.lower().startswith(opt_name)
