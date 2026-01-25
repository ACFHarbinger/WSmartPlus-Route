"""Tests for RewardScaler."""

import torch

from logic.src.pipeline.rl.common.reward_scaler import BatchRewardScaler, RewardScaler


def test_reward_scaler_norm():
    """Verify normalization scaling."""
    scaler = RewardScaler(scale="norm")
    data = torch.tensor([1.0, 2.0, 3.0])

    # Update and check mean
    scaler.update(data)
    assert abs(scaler.mean - 2.0) < 1e-6
    # Population variance: ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
    assert abs(scaler.variance - 2 / 3) < 1e-6

    scaled = scaler(data, update=False)
    # Mean of scaled should be 0, std should be 1
    assert abs(scaled.mean().item()) < 1e-6
    assert abs(scaled.std(correction=0).item() - 1.0) < 1e-6


def test_reward_scaler_ema():
    """Verify EMA updates."""
    scaler = RewardScaler(running_momentum=0.5)
    data1 = torch.tensor([10.0])
    data2 = torch.tensor([20.0])

    scaler.update(data1)
    assert scaler.mean == 10.0

    scaler.update(data2)
    # alpha=0.5: 0.5*20 + 0.5*10 = 15
    assert scaler.mean == 15.0


def test_batch_reward_scaler():
    """Verify batch-based scaling."""
    scaler = BatchRewardScaler()
    data = torch.tensor([10.0, 20.0, 30.0])
    scaled = scaler(data)

    assert abs(scaled.mean().item()) < 1e-6
    assert abs(scaled.std(correction=0).item() - 1.0) < 1e-6
