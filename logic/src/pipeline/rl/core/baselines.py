"""
Baseline implementations for policy gradient methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch.utils.data import Dataset

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
        """
        Return zero baseline (no variance reduction).

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.

        Returns:
            torch.Tensor: Zeros matching the reward shape.
        """
        return torch.zeros_like(reward)


class ExponentialBaseline(Baseline):
    """Exponential moving average baseline."""

    def __init__(self, beta: float = 0.8):
        """
        Initialize ExponentialBaseline.

        Args:
            beta: Decay factor for exponential moving average (default: 0.8).
        """
        self.beta = beta
        self.running_mean: Optional[torch.Tensor] = None

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline value using exponential moving average.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.

        Returns:
            torch.Tensor: Baseline value expanded to match reward shape.
        """
        if self.running_mean is None:
            self.running_mean = reward.mean().detach()
        else:
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward.mean().detach()
        return self.running_mean.expand_as(reward)


class RolloutBaseline(Baseline):
    """
    Greedy rollout baseline with significance-based updates.

    Uses greedy decoding with a frozen policy copy as baseline.
    The baseline policy is updated only if the current policy outperforms it
    significantly according to a T-test.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        update_every: int = 1,
        bl_alpha: float = 0.05,
        **kwargs,
    ):
        """
        Initialize RolloutBaseline.

        Args:
            policy: Policy to use as initial baseline (will be copied).
            update_every: Update baseline every N epochs.
            bl_alpha: Significance level for T-test to decide on updates.
            **kwargs: Additional keyword arguments.
        """
        self.update_every = update_every
        self.bl_alpha = bl_alpha
        self.baseline_policy: Optional[nn.Module] = None
        if policy is not None:
            self.setup(policy)

    def setup(self, policy: nn.Module):
        """Copy policy for baseline."""
        import copy

        self.baseline_policy = copy.deepcopy(policy)
        if self.baseline_policy is not None:
            self.baseline_policy.eval()
            for param in self.baseline_policy.parameters():
                param.requires_grad = False

    def _rollout(self, policy: nn.Module, td_or_dataset: any, env: Optional[any] = None) -> torch.Tensor:
        """Run greedy rollout on a batch or dataset."""
        if env is None:
            raise ValueError("Environment (env) is required for RolloutBaseline evaluation")

        from torch.utils.data import DataLoader, Dataset

        from logic.src.data.datasets import tensordict_collate_fn

        if isinstance(td_or_dataset, Dataset):
            # We use a simple DataLoader to batch the dataset
            loader = DataLoader(
                td_or_dataset,
                batch_size=64,
                collate_fn=tensordict_collate_fn,
                num_workers=0,
            )
            rewards = []
            policy.eval()
            with torch.no_grad():
                for batch in loader:
                    # Move batch to device
                    device = next(policy.parameters()).device
                    batch = batch.to(device)
                    # Reset environment for the batch
                    batch = env.reset(batch)
                    out = policy(batch, env, decode_type="greedy")
                    rewards.append(out["reward"].cpu())
            return torch.cat(rewards)

        # If it's a TensorDict (single batch)
        td_copy = env.reset(td_or_dataset.clone())
        policy.eval()
        with torch.no_grad():
            out = policy(td_copy, env, decode_type="greedy")
        return out["reward"]

    def wrap_dataset(self, policy: nn.Module, dataset: Dataset, env: any) -> Dataset:
        """Wrap the dataset with rollout baseline values."""
        from logic.src.data.datasets import BaselineDataset

        print("Evaluating baseline on dataset...")
        bl_vals = self._rollout(self.baseline_policy, dataset.data if hasattr(dataset, "data") else dataset, env)
        return BaselineDataset(dataset, bl_vals.view(-1, 1))

    def unwrap_batch(self, batch: any) -> tuple[any, any]:
        """Unwrap the batch."""
        if isinstance(batch, dict) and "data" in batch and "baseline" in batch:
            return batch["data"], batch["baseline"].view(-1)
        return batch, None

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline value.
        """
        # If we have a baseline policy, run it to get the baseline value
        if self.baseline_policy is not None:
            # Note: This is computationally expensive if done every step
            # Ideally use wrap_dataset/unwrap_batch flow
            with torch.no_grad():
                out = self.baseline_policy(td, None, decode_type="greedy")
            return out["reward"]
        return torch.zeros_like(reward)

    def epoch_callback(
        self, policy: nn.Module, epoch: int, val_dataset: Optional[any] = None, env: Optional[any] = None
    ):
        """Update baseline policy if current policy improves significantly."""
        if (epoch + 1) % self.update_every == 0:
            if val_dataset is not None and self.baseline_policy is not None and env is not None:
                from scipy import stats

                # Evaluate candidate
                candidate_vals = self._rollout(policy, val_dataset, env)
                candidate_mean = candidate_vals.mean().item()

                # Evaluate baseline
                baseline_vals = self._rollout(self.baseline_policy, val_dataset, env)
                baseline_mean = baseline_vals.mean().item()

                # T-test for significance
                t_stat, p_val = stats.ttest_rel(candidate_vals.cpu().numpy(), baseline_vals.cpu().numpy())

                if candidate_mean > baseline_mean and p_val / 2 < self.bl_alpha:
                    print(f"Update baseline: {baseline_mean:.4f} -> {candidate_mean:.4f} (p={p_val/2:.4f})")
                    self.setup(policy)
            else:
                self.setup(policy)


class WarmupBaseline(Baseline):
    """Gradual transition from ExponentialBaseline to target baseline."""

    def __init__(self, baseline: Baseline, warmup_epochs: int = 1, beta: float = 0.8):
        """
        Initialize WarmupBaseline.

        Args:
            baseline: Target baseline to transition to.
            warmup_epochs: Number of epochs for warmup transition.
            beta: Beta parameter for the warmup exponential baseline.
        """
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(beta=beta)
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute blended baseline value based on warmup progress.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.

        Returns:
            torch.Tensor: Blended baseline value.
        """
        if self.alpha >= 1.0:
            return self.baseline.eval(td, reward)
        if self.alpha <= 0.0:
            return self.warmup_baseline.eval(td, reward)

        v_target = self.baseline.eval(td, reward)
        v_warmup = self.warmup_baseline.eval(td, reward)
        return self.alpha * v_target + (1 - self.alpha) * v_warmup

    def epoch_callback(self, policy: nn.Module, epoch: int):
        """
        Update warmup alpha and call inner baseline callback.

        Args:
            policy: Current policy.
            epoch: Current epoch number.
        """
        self.baseline.epoch_callback(policy, epoch)
        if epoch < self.warmup_epochs:
            self.alpha = (epoch + 1) / float(self.warmup_epochs)
        else:
            self.alpha = 1.0


class CriticBaseline(Baseline):
    """Learned critic baseline."""

    def __init__(self, critic: Optional[nn.Module] = None):
        """
        Initialize CriticBaseline.

        Args:
            critic: Critic neural network module.
        """
        self.critic = critic

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline value using learned critic.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards (used for shape if critic is None).

        Returns:
            torch.Tensor: Critic value predictions.
        """
        if self.critic is None:
            return torch.zeros_like(reward)
        return self.critic(td).squeeze(-1)


class POMOBaseline(Baseline):
    """
    POMO baseline: mean reward across starts of the SAME instance.
    """

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute POMO baseline as mean reward across starting points.

        Args:
            td: TensorDict with environment state.
            reward: Reward tensor with shape [batch, num_starts].

        Returns:
            torch.Tensor: Mean reward expanded to match input shape.
        """
        # Reward shape: [batch, num_starts]
        if reward.dim() > 1:
            return reward.mean(dim=1, keepdim=True).expand_as(reward)
        return reward.mean()


# Baseline registry
BASELINE_REGISTRY = {
    "none": NoBaseline,
    "exponential": ExponentialBaseline,
    "rollout": RolloutBaseline,
    "critic": CriticBaseline,
    "warmup": WarmupBaseline,
    "pomo": POMOBaseline,
}


def get_baseline(name: str, policy: Optional[nn.Module] = None, **kwargs) -> Baseline:
    """Get baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")

    baseline = BASELINE_REGISTRY[name](**kwargs)
    if policy is not None:
        baseline.setup(policy)
    return baseline
