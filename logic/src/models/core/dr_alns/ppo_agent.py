"""PPO Agent and State for DR-ALNS control.

This module provides the neural architecture (`DRALNSPPOAgent`) and state
representation (`DRALNSState`) used for online control of Adaptive Large
Neighborhood Search (AAAI 2024).

Attributes:
    DRALNSPPOAgent: Multi-head actor-critic network.
    DRALNSState: Logic for problem-agnostic search state transformation.

Example:
    >>> from logic.src.models.core.dr_alns.ppo_agent import DRALNSPPOAgent
    >>> agent = DRALNSPPOAgent(state_dim=7)
    >>> actions, log_probs, val = agent.get_action(state_tensor)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class DRALNSPPOAgent(nn.Module):
    """Multi-headed PPO Actor-Critic for meta-heuristic control.

    The agent consists of a shared feature extractor followed by four discrete
    action heads (destroy, repair, severity, temperature) and a scalar value
    head for baseline estimation.

    Attributes:
        state_dim (int): Input feature count.
        hidden_dim (int): Width of hidden layers.
        n_destroy_ops (int): Number of available removal heuristics.
        n_repair_ops (int): Number of available insertion heuristics.
        n_severity_levels (int): Granularity of removal severity selection.
        n_temp_levels (int): Granularity of SA temperature selection.
        shared_net (nn.Sequential): Core MLP feature extractor.
        destroy_head (nn.Linear): Policy head for removal operator.
        repair_head (nn.Linear): Policy head for insertion operator.
        severity_head (nn.Linear): Policy head for destruction magnitude.
        temp_head (nn.Linear): Policy head for acceptance threshold.
        value_head (nn.Linear): Critic head for state value estimation.
    """

    def __init__(
        self,
        state_dim: int = 7,
        hidden_dim: int = 64,
        n_destroy_ops: int = 3,
        n_repair_ops: int = 3,
        n_severity_levels: int = 10,
        n_temp_levels: int = 50,
    ) -> None:
        """Initializes the PPO agent.

        Args:
            state_dim: Input dimensionality (default 7 based on search dynamics).
            hidden_dim: MLP hidden width.
            n_destroy_ops: Categorical size for destroy selection.
            n_repair_ops: Categorical size for repair selection.
            n_severity_levels: Categorical size for severity mapping.
            n_temp_levels: Categorical size for temperature mapping.
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_destroy_ops = n_destroy_ops
        self.n_repair_ops = n_repair_ops
        self.n_severity_levels = n_severity_levels
        self.n_temp_levels = n_temp_levels

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.destroy_head = nn.Linear(hidden_dim, n_destroy_ops)
        self.repair_head = nn.Linear(hidden_dim, n_repair_ops)
        self.severity_head = nn.Linear(hidden_dim, n_severity_levels)
        self.temp_head = nn.Linear(hidden_dim, n_temp_levels)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass to obtain logits and state value.

        Args:
            state: State tensor of shape [..., state_dim].

        Returns:
            Tuple: (destroy_logits, repair_logits, severity_logits, temp_logits, value).
        """
        features = self.shared_net(state)

        destroy_logits = self.destroy_head(features)
        repair_logits = self.repair_head(features)
        severity_logits = self.severity_head(features)
        temp_logits = self.temp_head(features)
        value = self.value_head(features)

        return destroy_logits, repair_logits, severity_logits, temp_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], torch.Tensor]:
        """Samples or selects actions for a given search state.

        Args:
            state: Input features [7].
            deterministic: Whether to use argmax (True) or sampling (False).

        Returns:
            Tuple[Dict[str, Any], Dict[str, torch.Tensor], torch.Tensor]:
                - actions: integer indices for each head.
                - log_probs: log-probabilities of selected actions.
                - value: scalar state value estimate.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        destroy_logits, repair_logits, severity_logits, temp_logits, value = self(state)

        destroy_dist = Categorical(logits=destroy_logits)
        repair_dist = Categorical(logits=repair_logits)
        severity_dist = Categorical(logits=severity_logits)
        temp_dist = Categorical(logits=temp_logits)

        if deterministic:
            destroy_action = destroy_logits.argmax(dim=-1)
            repair_action = repair_logits.argmax(dim=-1)
            severity_action = severity_logits.argmax(dim=-1)
            temp_action = temp_logits.argmax(dim=-1)
        else:
            destroy_action = destroy_dist.sample()
            repair_action = repair_dist.sample()
            severity_action = severity_dist.sample()
            temp_action = temp_dist.sample()

        destroy_log_prob = destroy_dist.log_prob(destroy_action)
        repair_log_prob = repair_dist.log_prob(repair_action)
        severity_log_prob = severity_dist.log_prob(severity_action)
        temp_log_prob = temp_dist.log_prob(temp_action)

        if squeeze_output:
            actions = {
                "destroy": int(destroy_action.item()),
                "repair": int(repair_action.item()),
                "severity": int(severity_action.item()),
                "temp": int(temp_action.item()),
            }
            log_probs = {
                "destroy": destroy_log_prob.squeeze(0),
                "repair": repair_log_prob.squeeze(0),
                "severity": severity_log_prob.squeeze(0),
                "temp": temp_log_prob.squeeze(0),
            }
            value = value.squeeze(0)
        else:
            actions = {
                "destroy": destroy_action,
                "repair": repair_action,
                "severity": severity_action,
                "temp": temp_action,
            }
            log_probs = {
                "destroy": destroy_log_prob,
                "repair": repair_log_prob,
                "severity": severity_log_prob,
                "temp": temp_log_prob,
            }

        return actions, log_probs, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluates specific actions for PPO policy updates.

        Args:
            states: Batch of observation tensors.
            actions: Map of integer actions harvested from a rollout.

        Returns:
            Tuple: (log_probs, state_values, entropy_map).
        """
        destroy_logits, repair_logits, severity_logits, temp_logits, values = self(states)

        destroy_dist = Categorical(logits=destroy_logits)
        repair_dist = Categorical(logits=repair_logits)
        severity_dist = Categorical(logits=severity_logits)
        temp_dist = Categorical(logits=temp_logits)

        log_probs = {
            "destroy": destroy_dist.log_prob(actions["destroy"]),
            "repair": repair_dist.log_prob(actions["repair"]),
            "severity": severity_dist.log_prob(actions["severity"]),
            "temp": temp_dist.log_prob(actions["temp"]),
        }

        entropies = {
            "destroy": destroy_dist.entropy(),
            "repair": repair_dist.entropy(),
            "severity": severity_dist.entropy(),
            "temp": temp_dist.entropy(),
        }

        return log_probs, values.squeeze(-1), entropies


class DRALNSState:
    """Transformation logic for search dynamics features.

    Maintains the 7 problem-agnostic features defined in AAAI 2024:
    [Best improved, Accepted, Improved, Is best, Cost diff, Stagnation, Budget].

    Attributes:
        best_improved (int): 1 if best solution was updated in last step.
        current_accepted (int): 1 if candidate was accepted.
        current_improved (int): 1 if candidate was objective-improving.
        is_current_best (int): 1 if incumbent is identical to best.
        cost_diff_best (float): Ratio of profit loss vs best.
        stagnation_count (int): Iterations without a global best update.
        search_budget (float): Ratio of iterations consumed.
    """

    def __init__(self) -> None:
        """Initializes empty state."""
        self.best_improved = 0
        self.current_accepted = 0
        self.current_improved = 0
        self.is_current_best = 0
        self.cost_diff_best = 0.0
        self.stagnation_count = 0
        self.search_budget = 0.0

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Serializes current features into a float tensor.

        Args:
            device: Optional target hardware device.

        Returns:
            torch.Tensor: Vector of size [7].
        """
        state_vec = torch.tensor(
            [
                float(self.best_improved),
                float(self.current_accepted),
                float(self.current_improved),
                float(self.is_current_best),
                self.cost_diff_best,
                float(self.stagnation_count),
                self.search_budget,
            ],
            dtype=torch.float32,
        )
        if device is not None:
            state_vec = state_vec.to(device)
        return state_vec

    def update(
        self,
        best_profit: float,
        current_profit: float,
        previous_profit: float,
        new_accepted: bool,
        new_best: bool,
        iteration: int,
        max_iterations: int,
        iterations_since_best: int,
    ) -> None:
        """Refreshes state features based on current iteration results.

        Args:
            best_profit: Peak profit in history.
            current_profit: Incumbent profit.
            previous_profit: profit of incumbent in previous step.
            new_accepted: outcome of acceptance check.
            new_best: outcome of best check.
            iteration: current step count.
            max_iterations: iteration limit.
            iterations_since_best: stagnation duration.
        """
        self.best_improved = 1 if new_best else 0
        self.current_accepted = 1 if new_accepted else 0
        self.current_improved = 1 if (new_accepted and current_profit > previous_profit) else 0
        self.is_current_best = 1 if abs(current_profit - best_profit) < 1e-6 else 0

        # Relative objective proximity
        if abs(current_profit) > 1e-10:
            self.cost_diff_best = (best_profit - current_profit) / abs(current_profit)
        else:
            self.cost_diff_best = -1.0

        self.stagnation_count = iterations_since_best
        self.search_budget = iteration / max_iterations if max_iterations > 0 else 0.0
