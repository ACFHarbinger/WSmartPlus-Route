"""
Simulator day logic for Neural Agent.
"""

from __future__ import annotations

import torch

from logic.src.policies.tsp import (
    get_route_cost,
)
from logic.src.utils.functions import add_attention_hooks


class SimulationMixin:
    """
    Mixin for single-day simulation logic in NeuralAgent.
    """

    def compute_simulator_day(
        self,
        input,
        graph,
        distC,
        profit_vars=None,
        waste_history=None,
        cost_weights=None,
        must_go=None,
    ):
        """
        Execute neural routing policy for a single simulation day.

        Main entry point for simulator integration. Generates a collection route
        for the current day using the trained neural model.

        Must-Go Selection:
        - If must_go is provided and all False: Returns [0] (no routing needed)
        - If must_go has True values: Route must include those bins
        - The environment's action mask enforces must_go constraints

        HRL Integration:
        - If hrl_manager provided: Manager decides whether to route (gate)
          and which bins to mask (selective collection)
        - If gate closed: Returns empty route [0]
        - If gate open: Worker model constructs route respecting mask

        Post-processing:
        - TODO: Add post-processing logic

        Args:
            input (dict): Problem instance with 'loc', 'waste', etc.
            graph (tuple): (edges, dist_matrix) for the model
            distC (torch.Tensor or np.ndarray): Distance matrix for cost calculation
            profit_vars (dict, optional): VRPP parameters (vehicle_capacity, R, C, etc.)
            waste_history (torch.Tensor, optional): Historical bin levels (Days x N) or (N x Days)
            threshold (float): Gating probability threshold. Default: 0.5
            must_go (torch.Tensor, optional): Boolean mask (N,) or (1, N) where True = must visit.
                If all False, returns [0] immediately (no routing needed).

        Returns:
            Tuple[List[int], float, dict]: Route, cost, and attention data
                - route: List of node IDs [0, node1, ..., 0]
                - cost: Total tour distance * 100
                - output_dict: {'attention_weights', 'graph_masks'}
        """
        edges, dist_matrix = graph
        hook_data = add_attention_hooks(self.model.encoder)

        # Check must_go: if provided and empty (no bins to visit), skip routing
        if must_go is not None:
            # Ensure tensor format
            if not isinstance(must_go, torch.Tensor):
                must_go = torch.tensor(must_go, dtype=torch.bool)

            # Handle shape: (N,) -> (1, N)
            if must_go.dim() == 1:
                must_go = must_go.unsqueeze(0)

            # Check if any bins are marked as must-go (excluding depot at index 0)
            has_must_go = must_go[:, 1:].any() if must_go.size(1) > 1 else must_go.any()

            if not has_must_go:
                # No bins to visit - return immediately
                for handle in hook_data["handles"]:
                    handle.remove()
                return (
                    [0],
                    0,
                    {
                        "attention_weights": torch.tensor([]),
                        "graph_masks": [],
                        "must_go_empty": True,
                    },
                )

        mask = None
        dynamic_feat = None
        input_for_model = input.copy()  # Shallow copy
        if "edges" not in list(input_for_model.keys()) and edges is not None:
            input_for_model["edges"] = edges
        if "dist" not in list(input_for_model.keys()) and dist_matrix is not None:
            input_for_model["dist"] = dist_matrix

        # Add must_go mask to input for environment's action mask logic
        if must_go is not None:
            # Ensure proper shape (1, N) for batched processing
            if must_go.dim() == 1:
                must_go = must_go.unsqueeze(0)
            input_for_model["must_go"] = must_go

        # Populate temporal features (fill1, fill2, ...) if model has temporal_horizon > 0
        horizon = getattr(self.model, "temporal_horizon", 0)
        if horizon > 0 and waste_history is not None:
            # waste_history: (Days, N) or (N, Days). After processing/hrl: dynamic_feat (1, N, History)
            if dynamic_feat is None:
                # Get location tensor regardless of key
                loc_tensor = input.get("locs", input.get("loc"))
                if loc_tensor is None:
                    raise KeyError("Input must contain 'loc' or 'locs'")

                if loc_tensor.dim() == 2:
                    h_static = loc_tensor.unsqueeze(0)
                else:
                    h_static = loc_tensor

                if waste_history.dim() == 2:
                    h_feat = waste_history.unsqueeze(0)
                else:
                    h_feat = waste_history  # (1, Days, N)

                N_bins = h_static.size(1)
                if h_feat.size(1) != N_bins and h_feat.size(2) == N_bins:
                    h_feat = h_feat.permute(0, 2, 1)  # (1, N, Days)

                if h_feat.max() > 2.0:
                    h_feat = h_feat / 100.0
                dynamic_feat = h_feat

            for h in range(1, horizon + 1):
                # Extraction:
                if dynamic_feat.size(2) >= horizon + 1:
                    hist_slice = dynamic_feat[:, :, -horizon - 1 : -1]
                    input_for_model[f"fill{h}"] = hist_slice[:, :, h - 1]
                else:
                    # Pad with zeros if history is too short (first few days of sim)
                    # or replicate last available? Zeros is safer.
                    # dynamic_feat: (1, N, Hist)
                    # If Hist=1 (only today), we need 'horizon' zeros.
                    # We can use current waste as a fallback or just zeros.
                    input_for_model[f"fill{h}"] = torch.zeros_like(dynamic_feat[:, :, 0])

        # Expand dist_matrix for post-processing if needed (like in original code)
        if dist_matrix is not None:
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
            # We don't have embeddings size yet to expand against batch.
            # But model forward will handle batching.

        _, _, _, pi, _ = self.model(
            input_for_model,
            return_pi=True,
            mask=mask,
            profit_vars=profit_vars,
            cost_weights=cost_weights,
        )

        route = torch.cat((torch.tensor([0]).to(pi.device), pi.squeeze(0)))
        cost = get_route_cost(distC * 100, route)
        for handle in hook_data["handles"]:
            handle.remove()

        attention_weights = torch.tensor([])
        if hook_data["weights"]:
            attention_weights = torch.stack(hook_data["weights"])
        route_list = route if isinstance(route, list) else route.cpu().numpy().tolist()
        return (
            route_list,
            cost,
            {"attention_weights": attention_weights, "graph_masks": hook_data["masks"]},
        )
