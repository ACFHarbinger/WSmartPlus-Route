"""
Baseline implementations for policy gradient methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import Dataset

import torch
import torch.nn as nn
from tensordict import TensorDict


class Baseline(nn.Module, ABC):
    """Base class for baselines."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """Compute baseline value."""
        raise NotImplementedError

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch if it's wrapped with baseline values."""
        if isinstance(batch, (dict, TensorDict)):
            keys = batch.keys()
            if "data" in keys and "baseline" in keys:
                return batch["data"], batch["baseline"]
        return batch, None

    def unwrap_dataset(self, dataset: Any) -> Any:
        """Unwrap the dataset if it's wrapped."""
        from logic.src.data.datasets import BaselineDataset

        if isinstance(dataset, BaselineDataset):
            return dataset.dataset
        return dataset

    def epoch_callback(
        self, policy: nn.Module, epoch: int, val_dataset: Optional[Any] = None, env: Optional[Any] = None
    ):
        """Optional callback at epoch end."""
        pass

    def setup(self, policy: nn.Module):
        """Optional setup with policy reference."""
        pass


class NoBaseline(Baseline):
    """No baseline (vanilla REINFORCE)."""

    def __init__(self, **kwargs):
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """
        Return zero baseline (no variance reduction).

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

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
        super().__init__()
        self.beta = beta
        self.running_mean: Optional[torch.Tensor] = None

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """
        Compute baseline value using exponential moving average.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.
            env: Environment (unused).

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
        super().__init__()
        self.update_every = update_every
        self.bl_alpha = bl_alpha
        self.baseline_policy = None
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

    def _rollout(self, policy: nn.Module, td_or_dataset: Any, env: Optional[Any] = None) -> torch.Tensor:
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
                    # Get device from policy
                    device = next(policy.parameters()).device

                    # Reset environment for the batch
                    # if it's a dict-like, we only want the data part for reset
                    if isinstance(batch, (dict, TensorDict)) and "data" in batch.keys():
                        td_data = batch["data"]
                    else:
                        td_data = batch

                    if isinstance(td_data, dict):
                        td_data = TensorDict(td_data, batch_size=[len(next(iter(td_data.values())))])

                    # Move to device after converting to TensorDict
                    td_data = td_data.to(device)

                    try:
                        td_data = env.reset(td_data)
                        td_data = td_data.to(device)  # Ensure on policy device after reset
                    except Exception as e:
                        print(
                            f"DEBUG: batch type: {type(batch)}, keys: {batch.keys() if hasattr(batch, 'keys') else 'N/A'}"
                        )
                        print(
                            f"DEBUG: td_data type: {type(td_data)}, keys: {td_data.keys() if hasattr(td_data, 'keys') else 'N/A'}"
                        )
                        raise e
                    out = policy(td_data, env, decode_type="greedy")
                    rewards.append(out["reward"].cpu())
            return torch.cat(rewards)

        # If it's a TensorDict (single batch)
        td_copy = td_or_dataset.clone()
        if isinstance(td_copy, (dict, TensorDict)) and "data" in td_copy.keys():
            td_copy = td_copy["data"]

        if isinstance(td_copy, dict):
            td_copy = TensorDict(td_copy, batch_size=[len(next(iter(td_copy.values())))])

        # Get device from policy
        device = next(policy.parameters()).device
        td_copy = td_copy.to(device)
        td_copy = env.reset(td_copy)
        td_copy = td_copy.to(device)  # Ensure on policy device after reset
        policy.eval()
        with torch.no_grad():
            out = policy(td_copy, env, decode_type="greedy")
        return out["reward"]

    def wrap_dataset(self, policy: nn.Module, dataset: Dataset, env: Any) -> Dataset:
        """Wrap the dataset with rollout baseline values."""
        from logic.src.data.datasets import BaselineDataset

        print("Evaluating baseline on dataset...")
        bl_vals = self._rollout(self.baseline_policy, dataset.data if hasattr(dataset, "data") else dataset, env)
        return BaselineDataset(dataset, bl_vals.view(-1, 1))

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """
        Compute baseline value.
        """
        # If we have a baseline policy, run it to get the baseline value
        if self.baseline_policy is not None:
            # Note: This is computationally expensive if done every step
            # Ideally use wrap_dataset/unwrap_batch flow
            with torch.no_grad():
                # We expect td to be already unwrapped or we unwrap it here
                if isinstance(td, (dict, TensorDict)) and "data" in td.keys():
                    td = td["data"]

                if isinstance(td, dict):
                    td = TensorDict(td, batch_size=[len(next(iter(td.values())))])

                # Check for "done" key to avoid re-resetting if already in loop
                # but _rollout calls env.reset so we are safe.
                # Actually, let's just call _rollout
                return self._rollout(self.baseline_policy, td, env)

        return torch.zeros_like(reward)

    def epoch_callback(
        self, policy: nn.Module, epoch: int, val_dataset: Optional[Any] = None, env: Optional[Any] = None
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
        super().__init__()
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(beta=beta)
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """
        Compute blended baseline value based on warmup progress.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.
            env: Environment.

        Returns:
            torch.Tensor: Blended baseline value.
        """
        if self.alpha >= 1.0:
            return self.baseline.eval(td, reward, env)
        if self.alpha <= 0.0:
            return self.warmup_baseline.eval(td, reward, env)

        v_target = self.baseline.eval(td, reward, env)
        v_warmup = self.warmup_baseline.eval(td, reward, env)
        return self.alpha * v_target + (1 - self.alpha) * v_warmup

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        return self.baseline.unwrap_batch(batch)

    def epoch_callback(
        self, policy: nn.Module, epoch: int, val_dataset: Optional[Any] = None, env: Optional[Any] = None
    ):
        """
        Update warmup alpha and call inner baseline callback.

        Args:
            policy: Current policy.
            epoch: Current epoch number.
            val_dataset: Validation dataset.
            env: Environment.
        """
        self.baseline.epoch_callback(policy, epoch, val_dataset, env)
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
        super().__init__()
        self.critic = critic

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """
        Compute baseline value using learned critic.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards (used for shape if critic is None).
            env: Environment (unused).

        Returns:
            torch.Tensor: Critic value predictions.
        """
        if self.critic is None:
            return torch.zeros_like(reward)

        # Unwrap td if needed
        if isinstance(td, (dict, TensorDict)) and "data" in td.keys():
            td = td["data"]

        if isinstance(td, dict):
            td = TensorDict(td, batch_size=[len(next(iter(td.values())))])

        return self.critic(td).squeeze(-1)


class POMOBaseline(Baseline):
    """
    POMO baseline: mean reward across starts of the SAME instance.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:
        """
        Compute POMO baseline as mean reward across starting points.

        Args:
            td: TensorDict with environment state.
            reward: Reward tensor with shape [batch, num_starts].
            env: Environment (unused).

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
