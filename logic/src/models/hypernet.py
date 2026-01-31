"""
This module contains the Hypernetwork implementation for adaptive cost weight generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.src.utils.functions.problem import is_vrpp_problem, is_wc_problem

from .modules import ActivationFunction, Normalization


class Hypernetwork(nn.Module):
    """
    Hypernetwork that generates adaptive cost weights for time-variant RL tasks.
    Takes temporal and performance metrics as input and outputs cost weights.
    """

    def __init__(
        self,
        problem,
        n_days=365,
        embedding_dim=16,
        hidden_dim=64,
        normalization="layer",
        activation="relu",
        learn_affine=True,
        bias=True,
    ):
        """
        Initialize the Hypernetwork.

        Args:
            problem (object): The problem instance wrapper.
            n_days (int, optional): Number of days in the year. Defaults to 365.
            embedding_dim (int, optional): Dimension of time embedding. Defaults to 16.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
            normalization (str, optional): Normalization type. Defaults to 'layer'.
            activation (str, optional): Activation function. Defaults to 'relu'.
            learn_affine (bool, optional): Whether to learn affine parameters in norm. Defaults to True.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        """
        super(Hypernetwork, self).__init__()
        self.problem = problem
        self.is_wc = is_wc_problem(problem)
        self.is_vrpp = is_vrpp_problem(problem)
        if self.is_vrpp or self.is_wc:
            cost_dim = 3 * 2
        else:
            cost_dim = 1 * 2

        self.n_days = n_days
        self.time_embedding = nn.Embedding(n_days, embedding_dim)  # Day of year embedding

        # Combined input: metrics + time embedding
        combined_dim = cost_dim + embedding_dim

        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, cost_dim, bias=bias),
        )

        # Output activation to ensure positive weights
        self.activation = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for the linear layers using Xavier Uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, metrics, day):
        """
        Generate cost weights based on current metrics and temporal information

        Args:
            metrics: Tensor of performance metrics [batch_size, input_dim]
            day: Tensor of day indices [batch_size]

        Returns:
            cost_weights: Tensor of generated cost weights [batch_size, output_dim]
        """
        # Get time embeddings
        day_embed = self.time_embedding(day % self.n_days)

        # Concatenate metrics with time embeddings
        combined = torch.cat([metrics, day_embed], dim=1)

        # Generate raw weights
        raw_weights = self.layers(combined)

        # Apply activation and normalize to sum to constraint
        weights = self.activation(raw_weights)

        return weights


class HypernetworkOptimizer:
    """
    Manages the hypernetwork training and integration with the main RL training loop.
    """

    def __init__(
        self,
        cost_weight_keys,
        constraint_value,
        device,
        problem,
        lr=1e-4,
        buffer_size=100,
    ):
        """
        Initialize the HypernetworkOptimizer.

        Args:
            cost_weight_keys (list): List of keys for cost weights.
            constraint_value (float): Constraint value for weight normalization.
            device (torch.device): Device to run the model on.
            problem (object): The problem instance.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            buffer_size (int, optional): Size of the experience buffer. Defaults to 100.
        """
        self.input_dim = 6  # [efficiency, overflows, kg, km, kg_lost, day_progress]
        self.output_dim = len(cost_weight_keys)
        self.cost_weight_keys = cost_weight_keys
        self.constraint_value = constraint_value
        self.device = device

        # Create hypernetwork
        self.hypernetwork = Hypernetwork(problem=problem, hidden_dim=64).to(device)

        # Create optimizer for hypernetwork
        self.optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=lr)

        # Experience buffer to train hypernetwork
        self.buffer = []
        self.buffer_size = buffer_size

        # Performance tracking
        self.best_performance = float("inf")
        self.best_weights = None

    def update_buffer(self, metrics, day, weights, performance):
        """
        Add experience to buffer.

        Args:
            metrics (torch.Tensor): Performance metrics.
            day (int): Current day.
            weights (torch.Tensor): Applied weights.
            performance (float): Resulting performance value.
        """
        self.buffer.append(
            {
                "metrics": metrics,
                "day": day,
                "weights": weights,
                "performance": performance,
            }
        )

        # Keep buffer at desired size
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Track best performance
        if performance < self.best_performance:
            self.best_performance = performance
            self.best_weights = weights.clone()

    def train(self, epochs=10):
        """
        Train hypernetwork on buffered experiences.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 10.
        """
        if len(self.buffer) < 10:  # Need minimum samples to train
            return

        self.hypernetwork.train()

        for _ in range(epochs):
            # Sample minibatch
            indices = torch.randperm(len(self.buffer))[: min(16, len(self.buffer))]

            metrics_batch = torch.stack([self.buffer[i]["metrics"] for i in indices]).to(self.device)
            day_batch = torch.tensor([self.buffer[i]["day"] for i in indices], dtype=torch.long).to(self.device)
            weights_batch = torch.stack([self.buffer[i]["weights"] for i in indices]).to(self.device)
            performance_batch = torch.tensor([self.buffer[i]["performance"] for i in indices]).to(self.device)

            # Generate predictions
            pred_weights = self.hypernetwork(metrics_batch, day_batch)

            # Normalize to constraint sum
            pred_weights_sum = pred_weights.sum(dim=1, keepdim=True)
            pred_weights = pred_weights * (self.constraint_value / pred_weights_sum)

            # Calculate loss (combine performance targeting and weight mimicking)
            best_perf_idx = performance_batch.argmin()
            target_weights = weights_batch[best_perf_idx].unsqueeze(0).expand_as(pred_weights)

            # Loss is MSE to best weights, weighted by relative performance
            perf_diff = (performance_batch - performance_batch.min()) / (
                performance_batch.max() - performance_batch.min() + 1e-8
            )
            perf_weights = 1.0 - perf_diff.unsqueeze(1).expand_as(pred_weights)

            loss = F.mse_loss(pred_weights, target_weights, reduction="none") * perf_weights
            loss = loss.mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_weights(self, all_costs, day, default_weights):
        """
        Generate optimized weights based on current metrics

        Args:
            all_costs: Dictionary of current cost values
            day: Current day (int)
            default_weights: Default weights to use if hypernetwork not ready

        Returns:
            dict: Dictionary of optimized cost weights
        """
        # If not enough experience, use default weights
        if len(self.buffer) < 5:
            return default_weights

        # Prepare input metrics
        with torch.no_grad():
            self.hypernetwork.eval()

            # Extract and normalize metrics
            overflows = torch.mean(all_costs["overflows"].float()).item()
            kg = torch.mean(all_costs["kg"]).item()
            km = torch.mean(all_costs["km"]).item()

            # Calculate derived metrics
            efficiency = kg / (km + 1e-8)
            kg_lost = all_costs.get("kg_lost", torch.tensor(0.0)).mean().item()
            day_progress = day / self.n_days  # Normalized day of year

            # Combine metrics
            metrics = (
                torch.tensor(
                    [efficiency, overflows, kg, km, kg_lost, day_progress],
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(self.device)
            )

            day_tensor = torch.tensor([day], dtype=torch.long).to(self.device)

            # Generate weights
            weights = self.hypernetwork(metrics, day_tensor).squeeze(0)

            # Normalize to constraint sum
            weights_sum = weights.sum()
            normalized_weights = weights * (self.constraint_value / weights_sum)

            # Convert to dictionary
            weights_dict = {key: normalized_weights[i].item() for i, key in enumerate(self.cost_weight_keys)}

            return weights_dict
