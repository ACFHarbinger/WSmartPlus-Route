from typing import Any, List, Optional, Tuple

import torch
from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.neural_agent import NeuralAgent
from logic.src.utils.functions import move_to


@PolicyRegistry.register("neural")
class NeuralPolicy(IPolicy):
    """
    Neural Policy wrapper.
    Executes deep reinforcement learning models.

    Supports must-go bin selection via:
    - Explicit must_go mask passed in kwargs
    - Selection strategy name (selector_name) with threshold (selector_threshold)
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the neural policy.

        Additional kwargs for must-go selection:
            must_go (List[int] or torch.Tensor): Pre-computed list of must-go bin IDs or boolean mask.
            selector_name (str): Name of vectorized selector ('last_minute', 'lookahead', etc.)
            selector_threshold (float): Threshold for the selector (meaning depends on selector type).
        """
        model_env = kwargs["model_env"]
        model_ls = kwargs["model_ls"]
        bins = kwargs["bins"]
        device = kwargs["device"]
        fill = kwargs["fill"]
        dm_tensor = kwargs["dm_tensor"]
        run_tsp = kwargs["run_tsp"]
        hrl_manager = kwargs.get("hrl_manager")
        gate_prob_threshold = kwargs.get("gate_prob_threshold", 0.5)
        mask_prob_threshold = kwargs.get("mask_prob_threshold", 0.5)
        two_opt_max_iter = kwargs.get("two_opt_max_iter", 0)

        agent = NeuralAgent(model_env)
        model_data, graph, profit_vars = model_ls

        # Construct cost weights
        cost_weights = {
            "waste": kwargs.get("w_waste", 1.0),
            "length": kwargs.get("w_length", 1.0),
            "overflows": kwargs.get("w_overflows", 1.0),
        }

        # Data preparation
        model_data["waste"] = torch.as_tensor(bins.c, dtype=torch.float32).unsqueeze(0)
        if "fill_history" in model_data:
            model_data["current_fill"] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)
        daily_data = move_to(model_data, device)

        # Handle must_go selection
        must_go_mask = self._get_must_go_mask(kwargs, bins, profit_vars, device)

        tour, cost, output_dict = agent.compute_simulator_day(
            daily_data,
            graph,
            dm_tensor,
            profit_vars,
            run_tsp,
            hrl_manager=hrl_manager,
            waste_history=bins.get_level_history(device=device),
            threshold=gate_prob_threshold,
            mask_threshold=mask_prob_threshold,
            two_opt_max_iter=two_opt_max_iter,
            cost_weights=cost_weights,
            must_go=must_go_mask,
        )
        return tour, cost, output_dict

    def _get_must_go_mask(
        self,
        kwargs: dict,
        bins: Any,
        profit_vars: Optional[dict],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Compute must_go mask from kwargs or selector.

        Args:
            kwargs: Execution kwargs containing must_go or selector config.
            bins: Bins object with current fill levels.
            profit_vars: Profit variables (revenue_kg, bin_capacity, etc.)
            device: Target device for tensors.

        Returns:
            Optional[torch.Tensor]: Boolean mask (1, N+1) where True = must visit.
                                    Includes depot at index 0 (always False).
                                    Returns None if no selection is configured.
        """
        # Check for explicit must_go
        must_go = kwargs.get("must_go")
        if must_go is not None:
            return self._convert_must_go_to_mask(must_go, bins, device)

        # Check for selector-based selection
        selector_name = kwargs.get("selector_name")
        if selector_name is None:
            return None

        from logic.src.models.policies.selection import get_vectorized_selector

        selector_threshold = kwargs.get("selector_threshold", 0.7)
        current_day = kwargs.get("current_day", 1)

        # Get current fill levels as tensor
        fill_levels = torch.as_tensor(bins.c, dtype=torch.float32, device=device) / 100.0
        fill_levels = fill_levels.unsqueeze(0)  # (1, N)

        # Prepend depot (fill=0)
        depot_fill = torch.zeros(1, 1, dtype=torch.float32, device=device)
        fill_levels_with_depot = torch.cat([depot_fill, fill_levels], dim=1)  # (1, N+1)

        # Get selector and compute must_go
        try:
            selector = get_vectorized_selector(selector_name, threshold=selector_threshold)
        except ValueError:
            return None

        # Prepare selector kwargs
        selector_kwargs = {
            "threshold": selector_threshold,
            "current_day": torch.tensor(current_day, device=device),
        }

        # Add accumulation rates if available
        if hasattr(bins, "rate") and bins.rate is not None:
            rates = torch.as_tensor(bins.rate, dtype=torch.float32, device=device) / 100.0
            rates = rates.unsqueeze(0)
            depot_rate = torch.zeros(1, 1, dtype=torch.float32, device=device)
            selector_kwargs["accumulation_rates"] = torch.cat([depot_rate, rates], dim=1)

        # Add profit vars for revenue selector
        if profit_vars is not None:
            selector_kwargs["revenue_kg"] = profit_vars.get("revenue_kg", 1.0)
            selector_kwargs["bin_capacity"] = profit_vars.get("bin_capacity", 1.0)

        must_go_mask = selector.select(fill_levels_with_depot, **selector_kwargs)
        return must_go_mask

    def _convert_must_go_to_mask(
        self,
        must_go: Any,
        bins: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert must_go input to a boolean mask tensor.

        Args:
            must_go: List of bin IDs (1-indexed) or boolean mask tensor.
            bins: Bins object to get size.
            device: Target device.

        Returns:
            torch.Tensor: Boolean mask (1, N+1) including depot.
        """
        num_bins = len(bins.c)

        if isinstance(must_go, torch.Tensor):
            if must_go.dtype == torch.bool:
                # Already a boolean mask
                if must_go.dim() == 1:
                    must_go = must_go.unsqueeze(0)
                # Ensure depot is included
                if must_go.size(1) == num_bins:
                    depot = torch.zeros(1, 1, dtype=torch.bool, device=device)
                    must_go = torch.cat([depot, must_go.to(device)], dim=1)
                return must_go.to(device)
            else:
                # Assume list of indices
                must_go = must_go.tolist()

        # Convert list of bin IDs to mask
        if isinstance(must_go, (list, tuple)):
            mask = torch.zeros(1, num_bins + 1, dtype=torch.bool, device=device)
            for bin_id in must_go:
                if 1 <= bin_id <= num_bins:
                    mask[0, bin_id] = True
            return mask

        return torch.zeros(1, num_bins + 1, dtype=torch.bool, device=device)
