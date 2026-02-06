"""
Command Pattern Implementation for Simulation Day Execution.

This module implements the Command Pattern to encapsulate simulation actions
as discrete, reusable objects. Each action represents a step in the daily
simulation cycle: bin filling, policy execution, waste collection, and logging.

The Command Pattern provides:
- Separation of concerns: Each action is isolated and testable
- Flexibility: Actions can be reordered, extended, or replaced
- Composability: Complex workflows are built from simple commands

Classes:
    SimulationAction: Abstract base class for all simulation commands
    FillAction: Executes daily bin filling (stochastic or empirical)
    MustGoSelectionAction: Identifies targets for collection
    PolicyExecutionAction: Runs routing policy and computes tours
    PostProcessAction: Refines generated tours
    CollectAction: Processes waste collection from visited bins
    LogAction: Records daily metrics and outputs results
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from loguru import logger

from logic.src.constants import ROOT_DIR
from logic.src.pipeline.simulations.day import get_daily_results
from logic.src.policies.adapters import PolicyFactory
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.logging.log_utils import send_daily_output_to_gui


def _flatten_config(cfg: Any) -> dict:
    """
    Helper to flatten nested configuration structures (e.g. hgs.custom -> list of dicts).
    """
    if not cfg:
        return {}

    curr = cfg
    # Unwrap single-key nested dicts (Hydra structure often starts with policy name)
    while isinstance(curr, dict) and len(curr) == 1:
        key = next(iter(curr))
        # If we reached the target object itself, stop unwrapping
        if key in ["must_go", "policy", "post_processing"]:
            break
        curr = curr[key]

    # Handle list of dicts (common in Hydra 'custom' lists)
    if isinstance(curr, list):
        merged = {}
        for item in curr:
            if isinstance(item, dict):
                merged.update(item)
        return merged

    # Handle dict which might contain lists to be flattened (e.g. {'custom': [...], 'ortools': [...]})
    if isinstance(curr, dict):
        flat = {**curr}
        # Iterate over all keys and flatten if value is a list of dicts
        for k, v in list(flat.items()):
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        flat.update(item)
        return flat

    return {}


class SimulationAction(ABC):
    """
    Abstract base class for simulation day actions.

    Defines the interface for all simulation commands. Each action receives
    a shared context dictionary and modifies it in-place with its outputs.

    The context dictionary is the primary communication mechanism between
    actions within a single simulation day.
    """

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """
        Executes the action and updates the context in-place.

        Args:
            context: Shared dictionary containing simulation state, inputs,
                and outputs. Modified in-place to communicate results to
                subsequent actions.

        Returns:
            None. All outputs are written to the context dictionary.
        """
        pass


class FillAction(SimulationAction):
    """
    Executes daily bin filling simulation.
    ...
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute daily bin filling."""
        bins = context["bins"]
        day = context["day"]

        if bins.is_stochastic():
            new_overflows, fill, total_fill, sum_lost = bins.stochasticFilling()
        else:
            new_overflows, fill, total_fill, sum_lost = bins.loadFilling(day)
        context["new_overflows"] = new_overflows
        context["fill"] = fill
        context["total_fill"] = total_fill
        context["sum_lost"] = sum_lost

        # Accumulate overflows in context (if needed by subsequent steps or final return)
        if "overflows" in context:
            context["overflows"] += new_overflows


class MustGoSelectionAction(SimulationAction):
    """
    Identifies which bins are targets for collection based on various strategies.

    Strategies range from simple fill-threshold rules (Regular, Last-Minute)
    to sophisticated stochastic prediction (Means/Std Dev). The output is
    a list of 'must_go' bin indices stored in the context for policy execution.

    Context Inputs:
        policy_name: Policy identifier used to map to selection strategy
        bins: Bin state management object
        threshold: Global override threshold
        distance_matrix: Distance information
        paths_between_states: Shortest paths
        max_capacity: Vehicle capacity

    Context Outputs:
        must_go: List of bin indices identified for collection (List[int])
    """

    def execute(self, context: Any) -> None:
        """
        Identifies bins that MUST be collected on the current day.
        """
        import os

        import numpy as np

        from logic.src.policies.must_go_selection import MustGoSelectionFactory, SelectionContext

        # 1. Gather all strategies to run
        strategies = []

        # Check config for 'must_go' list
        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)
        config_must_go = flat_cfg.get("must_go")

        if config_must_go:
            if not isinstance(config_must_go, list):
                config_must_go = [config_must_go]

            for item in config_must_go:
                if isinstance(item, str) and (item.endswith(".xml") or item.endswith(".yaml")):
                    # Load config file
                    fpath = os.path.join(ROOT_DIR, "assets", "configs", "policies", item)
                    cfg = load_config(fpath)
                    # Extract strategy name and params
                    # Flatten if root is 'config'
                    if "config" in cfg and len(cfg) == 1:
                        cfg = cfg["config"]

                    for k, v in cfg.items():
                        strategies.append({"name": k, "params": v if isinstance(v, dict) else {}})

                elif isinstance(item, dict):
                    # Direct dict config: { "lookahead": { "days": 7 } }
                    for k, v in item.items():
                        strategies.append({"name": k, "params": v if isinstance(v, dict) else {}})
                else:
                    # It's a direct name like "means_std" (legacy or simple string)
                    strategies.append({"name": item, "params": {}})

        # 2. Execute all strategies and union results
        bins = context["bins"]
        # Use more robust access for both real Bins and mock/dict contexts
        current_fill = getattr(bins, "c", bins.get("c") if isinstance(bins, dict) else None)
        n_bins = getattr(bins, "n", None)
        if n_bins is None:
            n_bins = len(current_fill) if current_fill is not None else 0

        accumulation_rates = getattr(bins, "means", bins.get("means") if isinstance(bins, dict) else None)
        std_deviations = getattr(bins, "std", bins.get("std") if isinstance(bins, dict) else None)

        final_must_go = set()

        for strat_info in strategies:
            s_name = strat_info["name"]
            s_params = strat_info["params"]

            # Context Preparation
            # We map specific params to the context as needed, or the Strategy extracts them
            # The Strategy pattern expects a SelectionContext.

            # Determine threshold/days from params if standard keys exist
            # This logic mimics the old parameter extraction but from dict
            thresh = 0.0
            if isinstance(s_params, dict):
                val = s_params.get("threshold") or s_params.get("cf") or s_params.get("param") or s_params.get("lvl")
                thresh = float(val) if val is not None else 0.0

            sel_ctx = SelectionContext(
                bin_ids=np.arange(0, n_bins, dtype="int32"),
                current_fill=np.array(current_fill) if current_fill is not None else np.array([]),
                accumulation_rates=accumulation_rates,
                std_deviations=std_deviations,
                current_day=context.get("day", 0),
                threshold=float(thresh) if thresh is not None else 0.0,
                next_collection_day=context.get("next_collection_day"),
                distance_matrix=context.get("distance_matrix"),
                paths_between_states=context.get("paths_between_states"),
                vehicle_capacity=context.get("max_capacity", 100.0),
            )

            if s_name == "select_all":
                res = list(sel_ctx.bin_ids)
            else:
                strategy = MustGoSelectionFactory.create_strategy(str(s_name), **s_params)
                res = strategy.select_bins(sel_ctx)

            final_must_go.update(res)

            # Ensure list
            if hasattr(res, "tolist"):
                res = res.tolist()
            elif not isinstance(res, list):
                res = list(res)

            final_must_go.update(res)

        context["must_go"] = list(final_must_go)


class PolicyExecutionAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.

    Dispatches to the appropriate policy adapter (Neural, Heuristic, Exact solver)
    to compute which bins to visit and in what order. The policy uses current
    bin states, distance information, and vehicle constraints to make decisions.

    Supported Policy Types:
        - Neural (AM, DDAM, TransGCN): Deep learning models
        - Heuristic (ALNS, HGS, Look-ahead): Metaheuristics
        - Exact (Gurobi, Hexaly): Mathematical optimization
        - Baseline (Regular, Last-minute): Simple rules

    Context Inputs:
        policy_name: Policy identifier string
        bins: Bin state management object
        fill: Current fill levels from FillAction
        distpath_tup: (distance_matrix, paths, dm_tensor, distancesC)
        model_env: Loaded neural model or solver environment
        model_ls: Model configuration tuple
        config: Policy-specific configuration dict
        (Plus various simulation parameters)

    Context Outputs:
        tour: Ordered list of bin IDs to visit (List[int])
        cost: Total tour distance in km (float)
        extra_output: Policy-specific metadata (varies by policy)
        cached: Updated cache for regular policy
        output_dict: Neural model outputs (attention, embeddings, etc.)
    """

    def execute(self, context: Any) -> None:
        """Execute the selected routing policy."""
        # 1. Standardized parameters
        policy_name = str(context.get("policy_name") or context.get("policy") or "")
        engine = context.get("engine")
        threshold = context.get("threshold")
        must_go = context.get("must_go", [])

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})

        solver_key = None

        # Check for top-level keys that map to known policies
        known_policy_keys = ["vrpp", "cvrp", "tsp", "hgs", "alns", "bcp", "sans", "neural"]
        for key in known_policy_keys:
            if key in raw_cfg:
                solver_key = key
                break

        flat_cfg = _flatten_config(raw_cfg)
        policy_cfg = flat_cfg.get("policy", {})

        # 2a. Check explicit config first (if not found by top-level key)
        if not solver_key:
            if isinstance(policy_cfg, dict):
                solver_key = policy_cfg.get("type") or policy_cfg.get("solver") or policy_cfg.get("engine")
            elif isinstance(policy_cfg, str):
                solver_key = policy_cfg

        # 2b. Fallback to 'engine' context var if not in policy config
        if not solver_key and engine:
            solver_key = engine

        # 2c. Fallback for backwards compatibility (if no config provided)
        # We try to use the policy_name directly as the key
        if not solver_key:
            solver_key = policy_name

        # 3. SELECTION STRATEGY FALLBACK (Default to TSP for routing if no engine specified)
        # If the key is essentially a selection strategy, we default to TSP/CVRP
        selection_keys = ["regular", "last_minute", "select_all", "lookahead", "means_std", "revenue"]
        if any(k in str(solver_key).lower() for k in selection_keys) and not any(
            k in str(solver_key).lower() for k in ["hgs", "alns", "vrpp", "neural", "bcp"]
        ):
            solver_key = "tsp"

        # 4. Short-circuit if no targets identified (Agnostic Contract)
        if (
            not must_go
            and solver_key != "neural"
            and not (str(solver_key).startswith("am") or str(solver_key).startswith("ddam"))
        ):
            context["tour"] = [0, 0]
            context["cost"] = 0.0
            context["extra_output"] = None
            return

        # 5. Routing Phase
        try:
            # Get adapter
            # We pass the full policy_cfg as kwargs directly to factory or adapter?
            # The factory `get_adapter` signature is: get_adapter(name, engine=..., threshold=..., **kwargs)
            # We pass the identified solver_key as the name.
            adapter = PolicyFactory.get_adapter(solver_key, engine=engine, threshold=threshold)

            # Policy execution (Agnostic: receives targets)
            tour, cost, extra_output = adapter.execute(**context)

            context["tour"] = tour
            context["cost"] = cost
            context["extra_output"] = extra_output

            # Handle specific extra outputs updates (Legacy support)
            if "regular" in policy_name:
                context["cached"] = extra_output
            elif solver_key == "neural" or str(solver_key).startswith("am"):
                context["output_dict"] = extra_output

        except ValueError as e:
            # If factory fails, we might want to catch it or let it crash
            raise ValueError(f"Failed to load policy adapter for '{solver_key}': {e}")


class PostProcessAction(SimulationAction):
    """
    Refines generated collection tours using modular refinement strategies.

    This action applies optimization passes (like fast_tsp or Local Search) to the
    tour generated by a policy. It enables clean separation between the base
    routing algorithm and late-stage refinement.

    Context Inputs:
        tour: Original tour from PolicyExecutionAction
        post_process: Name of post-processing strategy
        config: Simulation configuration (containing post_process_cfg)
        distance_matrix: Required for cost re-computation

    Context Outputs:
        tour: Refined tour (replaces previous tour)
        cost: Refined cost (replaces previous cost)
    """

    def execute(self, context: Any) -> None:
        """Refine the generated collection tour."""
        tour = context.get("tour")
        if not tour or len(tour) <= 2:
            return

        # 1. Determine list of post-processors
        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)

        # Check config for 'post_processing' list
        pp_list = flat_cfg.get("post_processing") or context.get("post_process")

        if pp_list:
            if not isinstance(pp_list, list):
                if isinstance(pp_list, str) and pp_list.lower() != "none":
                    pp_list = [pp_list]
                else:
                    pp_list = []

            import os

            from logic.src.policies.post_processing import PostProcessorFactory

            for item in pp_list:
                pp_name = ""
                pp_params = {
                    k: v for k, v in context.items() if k != "tour"
                }  # Start with Context (excluding tour to avoid duplication)

                if isinstance(item, str) and (item.endswith(".xml") or item.endswith(".yaml")):
                    # Load config
                    fpath = os.path.join(ROOT_DIR, "assets", "configs", "policies", item)
                    try:
                        cfg = load_config(fpath)
                        if "config" in cfg and len(cfg) == 1:
                            cfg = cfg["config"]

                        for k, v in cfg.items():
                            pp_name = k
                            if isinstance(v, dict):
                                pp_params.update(v)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Error loading post_processing config {item}: {e}")
                        continue
                else:
                    pp_name = item

                if not pp_name or pp_name.lower() == "none":
                    continue

                try:
                    # Apply refinement
                    processor = PostProcessorFactory.create(pp_name)
                    # Merge context with params but prioritize params from config?
                    # Actually pp_params already has context + config params

                    refined_tour = processor.process(tour, **pp_params)

                    if refined_tour != tour:
                        from logic.src.policies.single_vehicle import get_route_cost

                        dist_matrix = context.get("distance_matrix")
                        new_cost = get_route_cost(dist_matrix, refined_tour)

                        tour = refined_tour
                        context["tour"] = refined_tour
                        context["cost"] = new_cost

                except Exception as e:
                    logger.warning(f"Post-processing {pp_name} skipped due to error: {e}")


class CollectAction(SimulationAction):
    """
    Processes waste collection from bins visited in the tour.

    Simulates the physical act of emptying bins along the computed route.
    Updates bin states, calculates collection statistics, and computes
    profit based on revenue and operational costs.

    The collection process:
    1. Validates tour (must have >2 nodes to be non-trivial)
    2. Empties visited bins and records collected waste
    3. Updates bin statistics (means, std dev) using Welford's method
    4. Computes profit: (collected_kg × revenue) - (distance × cost_per_km)

    Context Inputs:
        bins: Bin state manager
        tour: Ordered list of bin IDs from PolicyExecutionAction
        cost: Tour distance in km

    Context Outputs:
        collected: Per-bin collected waste array (np.ndarray)
        total_collected: Total kg collected across all bins (float)
        ncol: Number of bins collected (int)
        profit: Net profit for this day (float)
    """

    def execute(self, context: Any) -> None:
        """Execute waste collection based on the generated tour."""
        bins = context["bins"]
        tour = context["tour"]
        cost = context["cost"]

        collected, total_collected, ncol, profit = bins.collect(tour, cost)

        context["collected"] = collected
        context["total_collected"] = total_collected
        context["ncol"] = ncol
        context["profit"] = profit


class LogAction(SimulationAction):
    """
    Records daily simulation metrics and outputs results.

    Aggregates results from previous actions into a structured daily log
    and optionally streams data to the GUI for real-time visualization.

    Logged metrics include:
        - Operational: kg collected, km traveled, bins visited, efficiency (kg/km)
        - Economic: profit, total cost
        - Environmental: overflows, waste lost
        - Routing: tour sequence with real bin IDs

    Context Inputs:
        tour: Computed route
        cost: Tour distance
        total_collected: Collected waste (kg)
        ncol: Number of bins collected
        profit: Net profit
        new_overflows: Bins that overflowed today
        sum_lost: Waste lost to overflows
        day: Current simulation day
        coords: Bin coordinate DataFrame
        (Plus GUI communication parameters)

    Context Outputs:
        daily_log: Structured dictionary of today's metrics
    """

    def execute(self, context: Any) -> None:
        """Log daily results and update GUI."""
        tour = context["tour"]
        cost = context["cost"]
        total_collected = context["total_collected"]
        ncol = context["ncol"]
        profit = context["profit"]
        new_overflows = context["new_overflows"]
        coords = context["coords"]
        day = context["day"]
        sum_lost = context["sum_lost"]

        dlog = get_daily_results(
            total_collected,
            ncol,
            cost,
            tour,
            day,
            new_overflows,
            sum_lost,
            coords,
            profit,
        )

        context["daily_log"] = dlog

        send_daily_output_to_gui(
            dlog,
            context["policy_name"],
            context["sample_id"],
            context["day"],
            context["total_fill"],
            context["collected"],
            context["bins"].c,
            context["realtime_log_path"],
            tour,
            coords,
            context["lock"],
        )
