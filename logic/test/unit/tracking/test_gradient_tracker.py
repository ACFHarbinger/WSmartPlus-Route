"""Unit tests for GradientTrackerCallback in logic/src/tracking/integrations/gradient_tracker.py."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from logic.src.tracking.integrations.gradient_tracker import GradientTrackerCallback

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_tracker_init():
    tracker = GradientTrackerCallback(
        log_freq=10,
        hist_freq=50,
        norm_type=2.0,
        include_bias=True,
        layer_filter=["fc1"],
        prefix="test/",
        max_grad_norm_alert=10.0,
    )
    assert tracker.log_freq == 10
    assert tracker.hist_freq == 50
    assert tracker.include_bias is True
    assert tracker.prefix == "test/"


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_tracker_should_track():
    tracker_no_bias = GradientTrackerCallback(include_bias=False, layer_filter=["fc1"])
    assert not tracker_no_bias._should_track("fc1.bias")
    assert tracker_no_bias._should_track("fc1.weight")
    assert not tracker_no_bias._should_track("fc2.weight")

    tracker_with_bias = GradientTrackerCallback(include_bias=True, layer_filter=None)
    assert tracker_with_bias._should_track("fc1.bias")
    assert tracker_with_bias._should_track("fc2.weight")


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_tracker_grad_norm():
    tracker = GradientTrackerCallback(norm_type=2.0)
    param = nn.Parameter(torch.tensor([1.0, 2.0]))
    assert tracker._grad_norm(param) is None

    param.grad = torch.tensor([3.0, 4.0])
    assert tracker._grad_norm(param) == pytest.approx(5.0)

    tracker_inf = GradientTrackerCallback(norm_type=float("inf"))
    tracker_inf.norm_type = float("inf")
    assert tracker_inf._grad_norm(param) == pytest.approx(4.0)


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_tracker_on_after_backward():
    model = SimpleModule()
    x = torch.randn(2, 5)
    loss = model(x).sum()
    loss.backward()

    tracker = GradientTrackerCallback(log_freq=1, max_grad_norm_alert=0.001)
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.global_step = 1

    mock_pl_module = MagicMock()
    mock_pl_module.named_parameters.return_value = model.named_parameters()

    with pytest.warns(RuntimeWarning, match="global grad norm"):
        tracker.on_after_backward(mock_trainer, mock_pl_module)

    assert mock_pl_module.log_dict.called


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_tracker_on_train_batch_end():
    tracker = GradientTrackerCallback(log_freq=1, hist_freq=1, log_histograms=True)
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.global_step = 1
    mock_trainer.loggers = []

    mock_pl_module = MagicMock()
    mock_pl_module.named_parameters.return_value = []

    # Should safely complete when no loggers attached
    tracker.on_train_batch_end(mock_trainer, mock_pl_module, outputs={}, batch={}, batch_idx=0)
