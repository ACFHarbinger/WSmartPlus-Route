"""
Legacy problem definitions for backward compatibility.
Provides the BaseProblem interface expected by legacy AttentionModel and Evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.utils.functions.beam_search import beam_search as beam_search_func

# Legacy constants for compatibility
COST_KM = 1.0
REVENUE_KG = 1.0
BIN_CAPACITY = 100.0
VEHICLE_CAPACITY = 100.0


class BaseProblem:
    """
    Legacy base class for routing problems.
    """

    NAME: str = "base"

    @staticmethod
    def validate_tours(pi: torch.Tensor) -> bool:
        """Validates tours (no duplicates except depot)."""
        if pi.size(-1) <= 1:
            return True
        sorted_pi: torch.Tensor = pi.data.sort(1)[0]
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all()
        return True

    @staticmethod
    def get_tour_length(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates tour length."""
        if pi.size(-1) <= 1:
            return torch.zeros(pi.size(0), device=pi.device)

        use_dist_matrix = dist_matrix is not None and isinstance(dist_matrix, torch.Tensor)
        if use_dist_matrix:
            # Simple distance matrix lookup
            assert dist_matrix is not None
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
            src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
            dst_mask: torch.Tensor = dst_vertices != 0
            pair_mask: torch.Tensor = (src_vertices != 0) & (dst_mask)
            dists: torch.Tensor = dist_matrix[0, src_vertices, dst_vertices] * pair_mask.float()
            last_dst: torch.Tensor = torch.max(
                dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices
            length: torch.Tensor = (
                dist_matrix[
                    0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0
                ]
                + dists.sum(dim=1)
                + dist_matrix[0, 0, pi[:, 0]]
            )
        else:
            loc = dataset.get("locs") if "locs" in dataset.keys() else dataset.get("loc")
            if loc.size(1) == dataset["depot"].size(0) + (
                dataset.get("waste").size(1) if "waste" in dataset.keys() else 0
            ):
                # already concatenated
                loc_with_depot = loc
            else:
                loc_with_depot: torch.Tensor = torch.cat((dataset["depot"][:, None, :], loc), 1)
            d: torch.Tensor = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)
                + (d[:, 0] - dataset["depot"]).norm(p=2, dim=-1)
                + (d[:, -1] - dataset["depot"]).norm(p=2, dim=-1)
            )
        return length

    @classmethod
    def beam_search(cls, input, beam_size, cost_weights, model=None, **kwargs):
        """Beam search bridge."""
        assert model is not None
        fixed = model.precompute_fixed(input, edges=input.get("edges"))

        def propose_expansions(beam):
            """Propose expansions for the current beam search state."""
            return model.propose_expansions(beam, fixed, normalize=True)

        # Note: make_state is problem-specific, must be implemented by subclasses
        state = cls.make_state(input, cost_weights=cost_weights, **kwargs)
        return beam_search_func(state, beam_size, propose_expansions)

    @classmethod
    def make_state(
        cls, input_data: Any, edges: Any = None, cost_weights: Any = None, dist_matrix: Any = None, **kwargs: Any
    ) -> Any:
        """
        Bridge to RL4CO environments.
        Initializes a TensorDict from the input and returns a state wrapper.
        """
        from tensordict import TensorDict

        from logic.src.envs import get_env
        from logic.src.models.policies.utils import TensorDictStateWrapper

        env_name = cls.NAME

        if isinstance(input_data, dict):
            # Determine batch size from typical batched tensors
            bs = 1
            device = torch.device("cpu")
            for k in ["loc", "locs", "waste", "prize", "demand"]:
                if k in input_data and torch.is_tensor(input_data[k]):
                    bs = input_data[k].size(0)
                    device = input_data[k].device
                    break

            # Initialize environment (lazy loading or re-use could be better but this is safe)
            # We pass any extra args like cost weights if needed, though get_reward handles it usually
            env = get_env(env_name, batch_size=torch.Size([bs]), device=device)

            # Create TensorDict, unsqueezing non-batched tensors if needed
            td_data = {}
            for k, v in input_data.items():
                if torch.is_tensor(v):
                    # Key mapping for simulator compatibility
                    target_key = k
                    if k == "loc":
                        target_key = "locs"
                    elif k == "waste":
                        # In new RL4CO envs: VRPP uses 'prize', WCVRP uses 'demand'
                        # But CVRPP needs both prize (reward) and demand (capacity)
                        if env_name in ["vrpp", "cvrpp"]:
                            target_key = "prize"
                            if env_name == "cvrpp":
                                # Also set demand (must match prize/locs shape later)
                                if v.dim() >= 1 and v.size(0) == bs:
                                    td_data["demand"] = v
                                elif v.dim() >= 2:
                                    td_data["demand"] = v.unsqueeze(0).expand(bs, *([-1] * v.dim()))
                                else:
                                    td_data["demand"] = (
                                        v.expand(bs, *([-1] * v.dim())) if v.dim() > 0 else v.unsqueeze(0).expand(bs)
                                    )
                                # Prepend will happen later in the locs concatenation block
                        else:
                            target_key = "demand"

                    if v.dim() >= 1 and v.size(0) == bs:
                        td_data[target_key] = v
                    elif v.dim() >= 2:  # Potentially a shared matrix like 'dist'
                        td_data[target_key] = v.unsqueeze(0).expand(bs, *([-1] * v.dim()))
                    else:
                        # Scalar or weird shape, try to expand
                        td_data[target_key] = (
                            v.expand(bs, *([-1] * v.dim())) if v.dim() > 0 else v.unsqueeze(0).expand(bs)
                        )
                else:
                    td_data[k] = v

            td = TensorDict(td_data, batch_size=[bs], device=device)
        elif isinstance(input_data, TensorDict):
            td = input_data
            bs = td.batch_size[0] if len(td.batch_size) > 0 else 1
            env = get_env(env_name, batch_size=torch.Size([bs]), device=td.device)
        else:
            # Fallback for weird cases
            td = TensorDict({}, batch_size=[1])
            env = get_env(env_name, batch_size=torch.Size([1]))

        # Ensure 'dist' and 'edges' are present and correctly shaped
        if "dist" not in td.keys() and dist_matrix is not None:
            if dist_matrix.dim() == 2:
                td["dist"] = dist_matrix.unsqueeze(0).expand(td.batch_size[0], -1, -1)
            else:
                td["dist"] = dist_matrix
        if "edges" not in td.keys() and edges is not None:
            if edges.dim() == 2:
                td["edges"] = edges.unsqueeze(0).expand(td.batch_size[0], -1, -1)
            else:
                td["edges"] = edges

        # Consolidate 'locs' logic: usually we concatenate depot and nodes
        # If we have both 'locs' (customers) and 'depot', we must concatenate them to form the full graph for the environment
        if "depot" in td.keys() and "locs" in td.keys():
            depot = td["depot"]
            locs = td["locs"]

            # Check if locs likely excludes depot (e.g., simpler dimension check or if it matches raw 'loc' size)
            # We assume if separate depot is provided, locs usually contains just customers (standard VRP lib format)
            # To be safe, we check tensor dimensions.
            # depot: (B, 2) or (B, 1, 2). locs: (B, N, 2).
            # If we align them:
            if depot.dim() == locs.dim() and depot.dim() == 3:
                pass  # shapes match (B, 1, 2) and (B, N, 2)
            elif depot.dim() == 2 and locs.dim() == 3:
                depot = depot.unsqueeze(1)

            # Now depot is (B, 1, 2). locs is (B, N, 2).
            # If N=100 (customers). We want 101.
            # If locs was 101, it might already include depot.
            # But the simulation passed 'loc' which was mapped to 'locs'. And simulation 'loc' excludes depot.
            # So we ALWAYS concatenate if we came from that path.
            # We can rely on the fact that we just created TD from input_data.

            # Update locs
            td["locs"] = torch.cat([depot, locs], dim=1)

            # Update demand/prize
            target_key = "prize" if env_name in ["vrpp", "cvrpp"] else "demand"
            if target_key in td.keys():
                dem = td[target_key]
                # Prepend 0 for depot
                zero_dem = torch.zeros(td.batch_size[0], 1, device=td.device)
                td[target_key] = torch.cat([zero_dem, dem], dim=1)

            # For CVRPP, we might have both
            if env_name == "cvrpp" and "demand" in td.keys() and td["demand"].size(1) == locs.size(1):
                dem = td["demand"]
                zero_dem = torch.zeros(td.batch_size[0], 1, device=td.device)
                td["demand"] = torch.cat([zero_dem, dem], dim=1)

            # Handle max_waste consistency
            if "max_waste" in td.keys() and torch.is_tensor(td["max_waste"]):
                mw = td["max_waste"]
                # If mw is (B, N_customers), prepend a dummy for depot (usually doesn't matter for depot)
                if mw.dim() > 1 and mw.size(1) == locs.size(1):  # locs is still customer-only here
                    zero_mw = torch.zeros(td.batch_size[0], 1, device=td.device)
                    td["max_waste"] = torch.cat([zero_mw, mw], dim=1)

        # Final check for environment-specific required keys
        if "locs" not in td.keys() and "loc" in td.keys():
            td["locs"] = td["loc"]

        # Ensure capacity is present
        if "capacity" not in td.keys():
            # Try to get from profit_vars (simulation) or kwargs
            profit_vars = kwargs.get("profit_vars")
            if profit_vars and "vehicle_capacity" in profit_vars:
                # WCVRP uses 'capacity'
                td["capacity"] = torch.full((td.batch_size[0],), profit_vars["vehicle_capacity"], device=td.device)
            elif "vehicle_capacity" in kwargs:
                td["capacity"] = torch.full((td.batch_size[0],), kwargs["vehicle_capacity"], device=td.device)
            else:
                # Default capacity if not provided (e.g. VRPP might not strictly need it for env init but WCVRP does)
                # For WCVRP, we usually normalize demand so capacity is 1.0, but simulation might use real values (e.g. 70, 100)
                # If we don't have it, we default to 1.0 (assuming normalized)
                if env_name in ["wcvrp", "cwcvrp", "sdwcvrp", "scwcvrp"]:
                    td["capacity"] = torch.ones(td.batch_size[0], device=td.device)

        td_reset = TensorDict(
            source={k: v for k, v in td.items()},
            batch_size=td.batch_size,
            device=td.device,
        )
        td = env.reset(td_reset)

        return TensorDictStateWrapper(td, env_name, env=env)


class VRPP(BaseProblem):
    """
    Vehicle Routing Problem with Profits (VRPP).
    Objective: Maximize Profit (Revenue - Cost).
    """

    NAME = "vrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute VRPP costs/rewards.

        Args:
            dataset: Problem data.
            pi: Tours [batch, nodes].
            cw_dict: Cost weights dictionary.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (negative_profit, cost_dict, None).
        """
        VRPP.validate_tours(pi)
        if pi.size(-1) == 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return (
                z,
                {"length": z, "waste": z, "overflows": z, "total": z},
                None,
            )

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        w = waste_with_depot.gather(1, pi)
        if "max_waste" in dataset.keys():
            w = w.clamp(max=dataset["max_waste"][:, None])
        waste = w.sum(dim=-1)
        length = VRPP.get_tour_length(dataset, pi, dist_matrix)

        cost_km = dataset.get("cost_km", COST_KM)
        revenue_kg = dataset.get("revenue_kg", REVENUE_KG)

        neg_profit = length * cost_km - waste * revenue_kg
        if cw_dict is not None:
            neg_profit = cw_dict.get("length", 1.0) * length * cost_km - cw_dict.get("waste", 1.0) * waste * revenue_kg

        return (
            neg_profit,
            {"length": length, "waste": waste, "overflows": torch.zeros_like(neg_profit), "total": neg_profit},
            None,
        )


class CVRPP(VRPP):
    """
    Capacitated VRPP.
    Includes vehicle capacity constraints.
    """

    NAME = "cvrpp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute CVRPP costs (same as VRPP but checks capacity).

        Args:
            dataset: Problem data.
            pi: Tours.
            cw_dict: Cost weights.
            dist_matrix: optional.

        Returns:
            Tuple of (cost, dict, None).
        """
        cost, c_dict, _ = VRPP.get_costs(dataset, pi, cw_dict, dist_matrix)

        # CVRPP specific: Check total capacity PER TRIP
        capacity = dataset.get("capacity", dataset.get("max_waste", torch.tensor(100.0)))

        # Extract trip waste
        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        # For each sequence in pi, calculate cumulative waste and reset at 0
        w = waste_with_depot.gather(1, pi)

        # Simple loop-based trip check for robustness in tests
        for b in range(pi.size(0)):
            cur_trip_waste = 0
            for i in range(pi.size(1)):
                node = pi[b, i].item()
                if node == 0:
                    cur_trip_waste = 0
                else:
                    cur_trip_waste += w[b, i].item()
                    # Use a small epsilon for float comparison
                    assert cur_trip_waste <= capacity[b].item() + 1e-6, "Used more than capacity"

        return cost, c_dict, _


class WCVRP(BaseProblem):
    """
    Waste Collection VRP (WCVRP).
    Objective: Minimize Overflow + TourLength - CollectedWaste.
    """

    NAME = "wcvrp"

    @staticmethod
    def get_costs(dataset, pi, cw_dict, dist_matrix=None):
        """
        Compute WCVRP costs.

        Args:
            dataset: Problem data.
            pi: Tours.
            cw_dict: Cost weights.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (cost, dict, None).
        """
        WCVRP.validate_tours(pi)
        if pi.size(-1) == 1:
            overflows = (dataset["waste"] >= dataset.get("max_waste", 1.0)).float().sum(-1)
            return (
                overflows,
                {
                    "overflows": overflows,
                    "length": torch.zeros_like(overflows),
                    "waste": torch.zeros_like(overflows),
                    "total": overflows,
                },
                None,
            )

        waste_with_depot = torch.cat((torch.zeros_like(dataset["waste"][:, :1]), dataset["waste"]), 1)
        visited_mask = torch.zeros_like(waste_with_depot, dtype=torch.bool)
        visited_mask.scatter_(1, pi, True)

        max_w = dataset.get("max_waste", torch.tensor(1.0, device=dataset["waste"].device))
        overflow_mask = waste_with_depot >= max_w
        overflows = torch.sum(overflow_mask[:, 1:] & ~visited_mask[:, 1:], dim=-1).float()

        length = WCVRP.get_tour_length(dataset, pi, dist_matrix)
        waste = waste_with_depot.gather(1, pi).clamp(max=max_w).sum(dim=-1)

        cost = overflows + length - waste
        if cw_dict is not None:
            cost = (
                cw_dict.get("overflows", 1.0) * overflows
                + cw_dict.get("length", 1.0) * length
                - cw_dict.get("waste", 1.0) * waste
            )

        return cost, {"overflows": overflows, "length": length, "waste": waste, "total": cost}, None


class CWCVRP(WCVRP):
    """Capacitated WCVRP."""

    NAME = "cwcvrp"


class SDWCVRP(WCVRP):
    """Stochastic Demand WCVRP."""

    NAME = "sdwcvrp"


class SCWCVRP(WCVRP):
    """Selective Capacitated WCVRP."""

    NAME = "scwcvrp"
