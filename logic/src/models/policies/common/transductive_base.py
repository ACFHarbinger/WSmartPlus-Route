from __future__ import annotations

import copy
from abc import ABC
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase


class TransductiveModel(nn.Module, ABC):
    """
    Base class for transductive methods.

    Transductive models perform search or adaptation at inference time.
    Common examples include Active Search (Bello et al.), EAS (Hottung et al.),
    and other instance-specific fine-tuning methods.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 10,
        **kwargs: Any,
    ):
        """Initialize TransductiveModel."""
        super().__init__()
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.n_search_steps = n_search_steps
        self.kwargs = kwargs

    def _setup_optimizer(self, params: Any) -> torch.optim.Optimizer:
        """Initialize the optimizer for the search phase."""
        return torch.optim.Adam(params, **self.optimizer_kwargs)

    def _get_search_params(self) -> Any:
        """
        Define which parameters to optimize during search.
        Default is all model parameters. Subclasses can override.
        """
        return self.model.parameters()

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        decode_type: str = "greedy",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute search or adaptation on the given instances.

        Args:
            td: TensorDict containing problem instance(s).
            env: Optional environment for transition logic.
            decode_type: Final decoding strategy.
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary containing final results (reward, actions, etc.).
        """
        # Save original state to restore after search
        original_state = copy.deepcopy(self.model.state_dict())

        optimizer = self._setup_optimizer(self._get_search_params())

        # Track best solution found during search
        best_reward = None
        best_actions = None

        search_history = []

        for _ in range(self.n_search_steps):
            optimizer.zero_grad()

            # Forward pass on the instances
            # We use sampling during search to explore
            # We pass return_pi=True to get actions
            out = self.model(td, env, decode_type="sampling", return_pi=True, **kwargs)

            # Support both Tuple and TensorDict returns from AttentionModel
            if isinstance(out, tuple):
                # (cost, ll, cost_dict, pi, entropy)
                cost = out[0]
                log_p = out[1]
                actions = out[3]
                reward = -cost
            else:
                reward = out.get("reward")
                log_p = out.get("log_p")
                actions = out.get("actions")

                if reward is None and "cost" in out:
                    reward = -out["cost"]

            if reward is None or log_p is None:
                raise ValueError("Reward or log_p not found in model output")

            # Update best found
            if best_reward is None:
                best_reward = reward.clone()
                best_actions = actions.clone() if actions is not None else None
            else:
                better = reward > best_reward
                best_reward = torch.where(better, reward, best_reward)
                if best_actions is not None and actions is not None:
                    # For actions, we need to mask correctly
                    # actions: [B, seq_len]
                    better_expanded = better.view(-1, 1).expand_as(best_actions)
                    best_actions = torch.where(better_expanded, actions, best_actions)

            # Compute loss (REINFORCE with best-so-far baseline)
            # Minimize -(reward - best_reward) * log_p
            # We use a simple mean baseline for better stability if best_reward is not per-instance
            advantage = reward - reward.mean()
            loss = -(advantage * log_p).mean()

            loss.backward()
            optimizer.step()

            search_history.append({"mean_reward": reward.mean().item(), "best_reward": best_reward.mean().item()})

        # Final output
        final_out = {"reward": best_reward, "actions": best_actions, "search_history": search_history}

        # Restore original state
        self.model.load_state_dict(original_state)

        return final_out
