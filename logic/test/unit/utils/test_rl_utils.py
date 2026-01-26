"""Tests for rl/common/utils.py."""

from unittest.mock import MagicMock

import pytest
import torch
from logic.src.pipeline.rl.common.utils import get_lightning_device, get_optimizer, get_scheduler


def test_get_optimizer():
    """Verify optimizer factory."""
    params = [torch.nn.Parameter(torch.randn(1))]

    # Adam
    opt = get_optimizer("adam", params, lr=0.01)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 0.01

    # SGD
    opt = get_optimizer("sgd", params, momentum=0.9)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["momentum"] == 0.9

    # Fail
    with pytest.raises(ValueError, match="Unknown optimizer"):
        get_optimizer("invalid", params)


def test_get_scheduler():
    """Verify scheduler factory."""
    params = [torch.nn.Parameter(torch.randn(1))]
    opt = torch.optim.Adam(params)

    # None
    assert get_scheduler("none", opt) is None

    # Step
    sched = get_scheduler("step", opt, step_size=10)
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    # Lambda
    sched = get_scheduler("lambda", opt, lr_lambda=lambda x: 1.0)
    assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    # Fail
    with pytest.raises(ValueError, match="Unknown scheduler"):
        get_scheduler("invalid", opt)


def test_get_lightning_device():
    """Verify device determination from trainer mockup."""
    trainer = MagicMock()

    # CPU
    trainer.accelerator = "cpu"
    assert get_lightning_device(trainer) == torch.device("cpu")

    # GPU
    trainer.accelerator = "gpu"
    trainer.strategy.root_device.index = 0
    assert get_lightning_device(trainer) == torch.device("cuda:0")
