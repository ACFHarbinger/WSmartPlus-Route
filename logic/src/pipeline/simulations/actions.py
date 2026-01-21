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
    PolicyExecutionAction: Runs routing policy and computes tours
    CollectAction: Processes waste collection from visited bins
    LogAction: Records daily metrics and outputs results
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

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

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the selected routing policy."""
        policy_name = context["policy_name"]
        adapter = PolicyFactory.get_adapter(policy_name)

        # NeuralPolicyAdapter expects 'fill' in context, which FillAction just put there.

        # Extract config
        context.get("config", {})

        tour, cost, extra_output = adapter.execute(**context)

        context["tour"] = tour
        context["cost"] = cost
        context["extra_output"] = extra_output

        # Handle specific extra outputs updates
        if "policy_regular" in policy_name:
            context["cached"] = extra_output
        elif policy_name[:2] == "am" or policy_name[:4] == "ddam" or "transgcn" in policy_name:
            context["output_dict"] = extra_output


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

    def execute(self, context: Dict[str, Any]) -> None:
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

    def execute(self, context: Dict[str, Any]) -> None:
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
