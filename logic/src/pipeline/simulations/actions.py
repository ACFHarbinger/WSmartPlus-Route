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
from typing import Any

from logic.src.pipeline.simulations.day import get_daily_results
from logic.src.policies.adapters import PolicyFactory
from logic.src.utils.logging.log_utils import send_daily_output_to_gui


class SimulationAction(ABC):
    """
    Abstract base class for simulation day actions.

    Defines the interface for all simulation commands. Each action receives
    a shared context dictionary and modifies it in-place with its outputs.

    The context dictionary is the primary communication mechanism between
    actions within a single simulation day.
    """

    @abstractmethod
    def execute(self, context: Any) -> None:
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

    Determines how much waste is added to each bin during the current day.
    Supports two filling modes:
    - Stochastic: Samples from statistical distributions (Gamma or Empirical)
    - Deterministic: Loads pre-recorded waste data from files

    Context Inputs:
        bins: Bins object managing bin state
        day: Current simulation day (int)

    Context Outputs:
        new_overflows: Number of bins that overflowed today (int)
        fill: Array of waste added to each bin today (np.ndarray)
        total_fill: Array of current bin levels after filling (np.ndarray)
        sum_lost: Total kg of waste lost due to overflows (float)
        overflows: Cumulative overflow count (updated if exists)
    """

    def execute(self, context: Any) -> None:
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
        """Identify bins targets for collection."""
        import numpy as np

        from logic.src.policies.must_go_selection import MustGoSelectionFactory, SelectionContext

        # Standardized parameters mapping
        policy_name_val = str(context.get("policy_name") or "")
        policy_val = str(context.get("policy") or "")
        full_policy_val = str(context.get("full_policy") or "")

        # Combine for parameter extraction, but use policy_name_val for strategy detection
        # Falls back to policy_val if policy_name_val is empty
        best_name = policy_name_val if policy_name_val else policy_val
        if not best_name:
            best_name = full_policy_val

        policy_name = best_name.lower()
        search_str = f"{policy_name_val} {policy_val} {full_policy_val}".lower()

        threshold = context.get("threshold") or getattr(context.get("bins"), "collectlevl", 90.0)
        if isinstance(threshold, (np.ndarray, list)):
            threshold = float(threshold[0]) if len(threshold) > 0 else 0.0

        # 1. Map policy name to selection strategy
        if "regular" in search_str:
            strategy_name = "regular"
        elif "and_path" in search_str:
            strategy_name = "last_minute_and_path"
        elif "last_minute" in search_str:
            strategy_name = "last_minute"
        elif any(k in search_str for k in ["lookahead", "look_ahead", "ahead"]):
            strategy_name = "lookahead"
        elif any(k in search_str for k in ["means_std", "means_std_dev", "meanstd"]):
            strategy_name = "means_std"
        elif any(k in search_str for k in ["revenue", "profit"]):
            strategy_name = "revenue"
        else:
            raise ValueError(f"Unknown must go selection strategy: {policy_name}")

        # 2. Determine strategy-specific parameters
        sel_threshold = threshold

        if strategy_name == "regular":
            import re

            match = re.search(r"regular(\d+)", search_str)
            if match:
                val_str = match.group(1)
                sel_threshold = float(int(val_str) - 1)
                if sel_threshold < 0:
                    raise ValueError(f"Invalid lvl value for policy_regular: {val_str}")
            else:
                sel_threshold = threshold if threshold is not None else 0.0
        elif strategy_name == "last_minute":
            import re

            match = re.search(r"last_minute(\d+)", search_str)
            sel_threshold = float(match.group(1)) if match else 90.0
        elif strategy_name == "last_minute_and_path":
            import re

            # Extract cf from something like policy_last_minute_and_path-100
            match = re.search(r"and_path(-?\d+)", search_str)
            if match:
                val = int(match.group(1))
                if val < 0:
                    raise ValueError(f"Invalid cf value for policy_last_minute_and_path: {val}")
                sel_threshold = float(val)
            else:
                sel_threshold = threshold if threshold is not None else 90.0
        elif strategy_name == "means_std":
            import re

            match = re.search(r"means_std(-?\d+)", search_str)
            if match:
                val = int(match.group(1))
                if val < 0:
                    raise ValueError(f"Invalid std parameter value for means_std: {val}")
                sel_threshold = float(val)
            else:
                sel_threshold = threshold if threshold is not None else 0.84
        elif strategy_name == "lookahead":
            if "_z" in search_str or "_invalid" in search_str:
                raise ValueError("Invalid policy_look_ahead configuration")
            sel_threshold = threshold if threshold is not None else 0.5

        if sel_threshold is None:
            sel_threshold = 0.5

        # 3. Build Selection Context
        bins = context["bins"]
        # Use more robust access for both real Bins and mock/dict contexts
        current_fill = getattr(bins, "c", bins.get("c") if isinstance(bins, dict) else None)
        n_bins = getattr(bins, "n", None)
        if n_bins is None:
            n_bins = len(current_fill) if current_fill is not None else 0

        sel_ctx = SelectionContext(
            bin_ids=np.arange(0, n_bins, dtype="int32"),
            current_fill=np.array(current_fill) if current_fill is not None else np.array([]),
            accumulation_rates=getattr(bins, "means", bins.get("means") if isinstance(bins, dict) else None),
            std_deviations=getattr(bins, "std", bins.get("std") if isinstance(bins, dict) else None),
            current_day=context.get("day", 0),
            threshold=sel_threshold,
            next_collection_day=context.get("next_collection_day"),
            distance_matrix=context.get("distance_matrix"),
            paths_between_states=context.get("paths_between_states"),
            vehicle_capacity=context.get("max_capacity", 100.0),
        )

        # 4. Use Factory to select bins
        if strategy_name == "select_all":
            must_go = list(sel_ctx.bin_ids)
        else:
            strategy = MustGoSelectionFactory.create_strategy(strategy_name)
            must_go = strategy.select_bins(sel_ctx)

        # Ensure must_go is a list (for better compatibility with policy logic and asserts)
        if hasattr(must_go, "tolist"):
            must_go = must_go.tolist()
        elif not isinstance(must_go, list):
            must_go = list(must_go)

        context["must_go"] = must_go


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

        # 2. Short-circuit if no targets identified (Agnostic Contract)
        if not must_go and "neural" not in policy_name:
            context["tour"] = [0, 0]
            context["cost"] = 0.0
            context["extra_output"] = None
            return

        # 3. Routing Phase
        # Map policy to Core Solver
        if any(k in policy_name for k in ["am", "ddam", "transgcn", "neural"]):
            solver_key = "neural"
        elif "vrpp" in policy_name:
            solver_key = "vrpp"
        elif any(k in policy_name for k in ["hgs", "alns", "sans", "lac", "lkh", "bcp"]):
            solver_key = next(k for k in ["hgs", "alns", "sans", "lac", "lkh", "bcp"] if k in policy_name)
        elif "tsp" in policy_name or "regular" in policy_name or "last_minute" in policy_name:
            solver_key = "tsp"
        elif "cvrp" in policy_name:
            solver_key = "cvrp"
        else:
            # Let PolicyFactory handle unknown policies or raise ValueError
            solver_key = policy_name

        # Get adapter
        adapter = PolicyFactory.get_adapter(solver_key, engine=engine, threshold=threshold)

        # Policy execution (Agnostic: receives targets)
        tour, cost, extra_output = adapter.execute(**context)

        context["tour"] = tour
        context["cost"] = cost
        context["extra_output"] = extra_output

        # Handle specific extra outputs updates
        if "regular" in policy_name:
            context["cached"] = extra_output
        elif solver_key == "neural":
            context["output_dict"] = extra_output


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

        # 1. Determine if post-processing is requested
        post_process_name = context.get("post_process") or context.get("config", {}).get("post_process")

        if post_process_name and post_process_name.lower() != "none":
            from logic.src.policies.post_processing import PostProcessorFactory

            # 2. Extract configuration parameters for post-processing
            config = context.get("config", {})
            post_process_config = config.get("post_process_cfg", {})
            # Merge with context for processor.process access
            proc_kwargs = {**context, **post_process_config}

            try:
                # 3. Apply refinement
                processor = PostProcessorFactory.create(post_process_name)
                refined_tour = processor.process(tour, **proc_kwargs)

                # 4. Update context with refined tour and re-compute cost
                if refined_tour != tour:
                    from logic.src.policies.single_vehicle import get_route_cost

                    dist_matrix = context.get("distance_matrix")
                    new_cost = get_route_cost(dist_matrix, refined_tour)

                    context["tour"] = refined_tour
                    context["cost"] = new_cost
            except Exception as e:
                # Log error and keep original tour
                print(f"Post-processing skipped due to error: {e}")


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
