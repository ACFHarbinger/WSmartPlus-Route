"""Optimized Orchestration for HyperNetwork Weight Adjustment.

This module provides the `HyperNetworkOptimizer`, which manages the training,
buffering, and real-time generation of adaptive reward weights using a
HyperNetwork. It identifies high-performing weight configurations and trains
 the network to mimic them indexed by temporal and fleet metrics.

Attributes:
    HyperNetworkOptimizer: Controller for adaptive weight meta-heuristics.

Example:
    None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .hypernetwork import HyperNetwork


class HyperNetworkOptimizer:
    """Manager for HyperNetwork lifecycle and training.

    Maintains an experience buffer of agent performance under various weight
    configurations and performs a 'weight-mimicking' optimization to teach the
    HyperNetwork how to maximize efficiency across temporal shifts.

    Attributes:
        input_dim (int): count of input metrics (efficiency, overflows, etc.).
        output_dim (int): count of weights to generate.
        cost_weight_keys (List[str]): names of the adjustable reward components.
        constraint_value (float): constant sum required for weights.
        device (torch.device): target hardware.
        n_days (int): wrapping period for time embeddings.
        hypernetwork (HyperNetwork): the weight-generating meta-model.
        optimizer (torch.optim.Optimizer): trainer for the hypernetwork.
        buffer (List[Dict[str, Any]]): experience replay storage.
        buffer_size (int): max size for the experience buffer.
        best_performance (float): minimum objective value reached.
        best_weights (Optional[torch.Tensor]): weight set corresponding to best_performance.
    """

    def __init__(
        self,
        cost_weight_keys: List[str],
        constraint_value: float,
        device: torch.device,
        problem: Any,
        lr: float = 1e-4,
        buffer_size: int = 100,
    ) -> None:
        """Initializes the HyperNetwork optimizer.

        Args:
            cost_weight_keys: list of objective component names.
            constraint_value: total sum value for normalized weights.
            device: computation device.
            problem: environment/problem definition (context).
            lr: learning rate for hypernetwork updates.
            buffer_size: maximum history of trajectory metrics to maintain.
        """
        self.input_dim = 6  # [efficiency, overflows, kg, km, kg_lost, day_progress]
        self.output_dim = len(cost_weight_keys)
        self.cost_weight_keys = cost_weight_keys
        self.constraint_value = constraint_value
        self.device = device

        self.n_days = 365
        self.hypernetwork = HyperNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_days=self.n_days,
            hidden_dim=64,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=lr)

        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = buffer_size

        self.best_performance = float("inf")
        self.best_weights: Optional[torch.Tensor] = None

    def update_buffer(
        self,
        metrics: torch.Tensor,
        day: int,
        weights: torch.Tensor,
        performance: float,
    ) -> None:
        """Stores a trajectory outcome in the replay buffer.

        Args:
            metrics: state metrics [B].
            day: day of the year.
            weights: used weight vector.
            performance: scalar objective outcome (minimize).
        """
        self.buffer.append(
            {
                "metrics": metrics,
                "day": day,
                "weights": weights,
                "performance": performance,
            }
        )

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Track best performance for target-mimicking labels
        if performance < self.best_performance:
            self.best_performance = performance
            self.best_weights = weights.clone()

    def train(self, epochs: int = 10) -> None:
        """Executes a training phase over buffered experiences.

        Minimizes the MSE between predicted weights and the weights that
        achieved the best historical performance, weighted by relative
        trajectory quality.

        Args:
            epochs: number of gradient descent passes over the buffer.
        """
        if len(self.buffer) < 10:
            return

        self.hypernetwork.train()

        for _ in range(epochs):
            indices = torch.randperm(len(self.buffer))[: min(16, len(self.buffer))]

            metrics_batch = torch.stack([self.buffer[i]["metrics"] for i in indices]).to(self.device)
            day_batch = torch.tensor([self.buffer[i]["day"] for i in indices], dtype=torch.long).to(self.device)
            weights_batch = torch.stack([self.buffer[i]["weights"] for i in indices]).to(self.device)
            performance_batch = torch.tensor([self.buffer[i]["performance"] for i in indices]).float().to(self.device)

            pred_weights = self.hypernetwork(metrics_batch, day_batch)

            # Normalize to constraint sum
            pred_weights_sum = pred_weights.sum(dim=1, keepdim=True)
            pred_weights = pred_weights * (self.constraint_value / pred_weights_sum)

            # Target Mimicking: Loss is MSE to best weights, weighted by performance
            best_perf_idx = performance_batch.argmin()
            target_weights = weights_batch[best_perf_idx].unsqueeze(0).expand_as(pred_weights)

            perf_min = performance_batch.min()
            perf_max = performance_batch.max()
            perf_diff = (performance_batch - perf_min) / (perf_max - perf_min + 1e-8)
            perf_weights = 1.0 - perf_diff.unsqueeze(1).expand_as(pred_weights)

            loss = F.mse_loss(pred_weights, target_weights, reduction="none") * perf_weights
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_weights(
        self,
        all_costs: Dict[str, torch.Tensor],
        day: int,
        default_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Generates real-time adaptive weights for the current period.

        Args:
            all_costs: collection of raw metric tensors from the fleet.
            day: current simulation day.
            default_weights: values to return if the hypernetwork is not yet trained.

        Returns:
            Dict[str, float]: map from objective key to generated weight scalar.
        """
        if len(self.buffer) < 5:
            return default_weights

        with torch.no_grad():
            self.hypernetwork.eval()

            # Extract and normalize metrics for network input
            overflows = torch.mean(all_costs["overflows"].float()).item()
            kg = torch.mean(all_costs["kg"]).item()
            km = torch.mean(all_costs["km"]).item()

            efficiency = kg / (km + 1e-8)
            kg_lost = all_costs.get("kg_lost", torch.tensor(0.0)).mean().item()
            day_progress = day / self.n_days

            metrics = (
                torch.tensor(
                    [efficiency, overflows, kg, km, kg_lost, day_progress],
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(self.device)
            )

            day_tensor = torch.tensor([day], dtype=torch.long).to(self.device)

            # Generate weights and normalize
            weights = self.hypernetwork(metrics, day_tensor).squeeze(0)
            weights_sum = weights.sum()
            normalized_weights = weights * (self.constraint_value / weights_sum)

            # Convert to dictionary format for logic layer consumption
            weights_dict = {key: normalized_weights[i].item() for i, key in enumerate(self.cost_weight_keys)}

            return weights_dict
