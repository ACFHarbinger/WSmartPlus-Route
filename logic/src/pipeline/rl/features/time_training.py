"""
Time-based training utilities.
Handles temporal aspects of the WCVRP environment:
- Daily updates of bin fill levels.
- Dataset augmentation with fill history.
- Simulation of waste accumulation.
"""
from typing import Dict, List

import torch
from tensordict import TensorDict


class TimeBasedMixin:
    """Mixin for time-based/temporal training support."""

    def setup_time_training(self, opts: Dict):
        """Initialize time-based training state."""
        self.temporal_horizon = opts.get("temporal_horizon", 0)
        self.current_day = 0
        self.fill_history = []
        # Store a REFERENCE to the dataset we are modifying
        self.current_dataset = None

    def update_dataset_for_day(self, routes: List[torch.Tensor], day: int):
        """
        Update local dataset (self.train_dataset) state after a day's routes.
        1. Process collections: Reset accumulated waste for visited nodes.
        2. Simulate accumulation: Add stochastic waste to all nodes.

        Args:
            routes: List of tensors, where each tensor is a sequence of node indices visited.
            day: The current simulation day index (0-based).
        """
        if self.current_dataset is None:
            # Try to find the dataset attached to self
            if hasattr(self, "train_dataset"):
                self.current_dataset = self.train_dataset
            elif hasattr(self, "dataset"):
                self.current_dataset = self.dataset
            else:
                return  # Cannot update without dataset reference

        self.current_day = day
        self.fill_history.append(routes)

        # Access the TensorDict underlying the dataset
        if hasattr(self.current_dataset, "data"):
            td = self.current_dataset.data
        else:
            td = self.current_dataset  # Assuming it is a TensorDict if not wrapped

        if not isinstance(td, TensorDict):
            return

        # --- 1. Process Collections ---
        # visited_mask: [batch_size, num_nodes]
        batch_size = td.batch_size[0]
        num_nodes = td["locs"].shape[1] - 1  # Excluding depot? Usually locs is [B, N+1, 2]

        # Helper to convert routes to mask
        device = td.device
        visited_mask = torch.zeros((batch_size, num_nodes + 1), dtype=torch.bool, device=device)

        # routes is a list of [batch_size] tensors usually?
        # Or routes is [batch, seq_len]?
        # Typically in RL loop, 'out["actions"]' is [batch, seq_len].

        # Let's assume routes is the 'actions' tensor [batch, seq_len]
        actions = routes if isinstance(routes, torch.Tensor) else None

        if actions is not None:
            # Gather visited nodes
            # actions contains node indices. 0 is depot.
            # We want to mask nodes that were visited.
            # scatter_add is useful, or simple indexing.

            # Expand visited mask
            visited_mask.scatter_(1, actions.long(), True)
            # Depot (0) doesn't have waste, ignore it
            visited_mask[:, 0] = False

            # Current Demand/Waste
            # Usually "demand" or "current_fill"
            key = "demand" if "demand" in td.keys() else "current_fill"
            if key in td.keys():
                current_fill = td[key]  # [batch, num_nodes+1] usually? Or [batch, num_nodes]

                # Reset visited to 0
                # shape check
                if current_fill.shape == visited_mask.shape:
                    current_fill[visited_mask] = 0.0
                elif current_fill.shape[1] == num_nodes:  # maybe no depot in demand
                    # Adjust mask
                    current_fill[visited_mask[:, 1:]] = 0.0

                td[key] = current_fill

        # --- 2. Simulate Accumulation (Waste Generation) ---
        # Add random waste to all nodes (e.g. normal distribution or purely stochastic)
        # This brings us to the next day's state.

        # Get capacity to clamp
        capacity = td.get("capacity", torch.tensor([1.0], device=device))

        # Generate waste increments
        # Simple Logic: 0.1 * capacity * random
        # Real logic should interact with 'generation_rate' if present in data
        if "generation_rate" in td.keys():
            rate = td["generation_rate"]
        else:
            rate = 0.05 * capacity  # Default 5% per day

        noise = torch.randn_like(current_fill) * 0.1 + 1.0  # Multiplicative noise N(1, 0.1)
        increment = rate * noise
        increment = torch.max(increment, torch.zeros_like(increment))  # No negative waste

        # Update
        new_fill = current_fill + increment
        # Clamp to capacity? Or allow overflow (soft constraint models)?
        # WSmart-Route features overflow penalties, so we allow > capacity.
        # But usually we track 'overflow' separately?
        # For simple RL env, we just update the fill.

        td[key] = new_fill

        print(f"Time Training: Updated dataset for Day {day+1}. Mean fill: {new_fill.mean():.3f}")


def prepare_time_dataset(dataset, day, history):
    """Augment dataset with temporal history."""
    # Inject history into the TensorDict
    # This prepares the specific batch or dataset for the policy (e.g. TAM) to see past actions.
    if hasattr(dataset, "data"):
        td = dataset.data
        # Create a 'history' entry: [batch, day, num_nodes] or summarized?
        # Usually simple TAM uses embedding features.
        td.set("current_day", torch.tensor(day))
        # We might set specific history features here if the Model expects them
    return dataset
