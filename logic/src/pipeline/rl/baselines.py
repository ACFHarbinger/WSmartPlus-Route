"""
Baseline implementations for policy gradient methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


class Baseline(ABC):
    """Base class for baselines."""

    @abstractmethod
    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """Compute baseline value."""
        raise NotImplementedError

    def epoch_callback(self, policy: nn.Module, epoch: int):
        """Optional callback at epoch end."""
        pass

    def setup(self, policy: nn.Module):
        """Optional setup with policy reference."""
        pass


class NoBaseline(Baseline):
    """No baseline (vanilla REINFORCE)."""

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(reward)


class ExponentialBaseline(Baseline):
    """Exponential moving average baseline."""

    def __init__(self, beta: float = 0.8):
        self.beta = beta
        self.running_mean: Optional[torch.Tensor] = None

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        if self.running_mean is None:
            self.running_mean = reward.mean().detach()
        else:
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward.mean().detach()
        return self.running_mean.expand_as(reward)


class RolloutBaseline(Baseline):
    """
    Greedy rollout baseline.

    Uses greedy decoding with a frozen policy copy as baseline.
    """

    def __init__(self, policy: Optional[nn.Module] = None, update_every: int = 1):
        self.update_every = update_every
        self.baseline_policy = None
        if policy is not None:
            self.setup(policy)

    def setup(self, policy: nn.Module):
        """Copy policy for baseline."""
        import copy

        self.baseline_policy = copy.deepcopy(policy)
        for param in self.baseline_policy.parameters():
            param.requires_grad = False

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """Run greedy baseline rollout."""
        if self.baseline_policy is None:
            return torch.zeros_like(reward)

        # This assumes we have access to env and td
        # In practice, this would be called differently
        return torch.zeros_like(reward)  # Placeholder

    def epoch_callback(self, policy: nn.Module, epoch: int):
        """Update baseline policy periodically."""
        if (epoch + 1) % self.update_every == 0:
            self.setup(policy)


class CriticBaseline(Baseline):
    """Learned critic baseline."""

    def __init__(self, critic: Optional[nn.Module] = None):
        self.critic = critic

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        if self.critic is None:
            return torch.zeros_like(reward)
        return self.critic(td).squeeze(-1)


# Baseline registry
BASELINE_REGISTRY = {
    "none": NoBaseline,
    "exponential": ExponentialBaseline,
    "rollout": RolloutBaseline,
    "critic": CriticBaseline,
}


def get_baseline(name: str, policy: Optional[nn.Module] = None, **kwargs) -> Baseline:
    """Get baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")

    baseline = BASELINE_REGISTRY[name](**kwargs)
    if policy is not None:
        baseline.setup(policy)
    return baseline
