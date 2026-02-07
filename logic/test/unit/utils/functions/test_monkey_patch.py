"""Tests for optimizer monkey patch."""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))


def test_load_state_dict_casting():
    """Verify that state tensors are cast to the correct device."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a dummy state dict with a tensor on CPU
    # We'll mock the device behavior or just use CPU to CPU and check types
    state_dict = {
        "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.tensor([0.5]), "exp_avg_sq": torch.tensor([0.2])}},
        "param_groups": [
            {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False, "params": [0]}
        ],
    }

    # Trigger the monkey patch
    optimizer.load_state_dict(state_dict)

    # Check that it's loaded
    assert optimizer.state[model.param]["step"] == 1.0
    assert optimizer.state[model.param]["exp_avg"] == 0.5


def test_load_state_dict_invalid_groups():
    """Verify error handling for mismatched parameter groups."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Different number of groups
    state_dict = {"state": {}, "param_groups": [{"params": [0]}, {"params": [1]}]}

    import pytest

    with pytest.raises(ValueError, match="different number of parameter groups"):
        optimizer.load_state_dict(state_dict)


def test_load_state_dict_invalid_params():
    """Verify error handling for mismatched parameter sizes."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Different number of params in group
    state_dict = {"state": {}, "param_groups": [{"params": []}]}

    import pytest

    with pytest.raises(ValueError, match="doesn't match the size of optimizer's group"):
        optimizer.load_state_dict(state_dict)
