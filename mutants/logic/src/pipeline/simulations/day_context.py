"""
Simulation state encapsulation and consolidation.

This module provides the SimulationDayContext dataclass and orchestrates
the execution of a single simulation day using the Command Pattern.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from multiprocessing.synchronize import Lock
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from logic.src.constants import DAY_METRICS
from logic.src.pipeline.simulations.bins import Bins
from logic.src.utils.functions import move_to


@dataclass
class SimulationDayContext(Mapping):
    """
    Context object encapsulating the state of a simulation day.

    Attributes:
        graph_size: Total nodes in the problem graph, **including** the depot (= num_loc + 1).
        full_policy: Full string identifier of the policy (e.g., 'policy_regular3_gamma1').
        policy: Parsed policy name.
        policy_name: Name of the policy.
        bins: The Bins object managing waste levels.
        new_data: DataFrame containing new data for the day.
        coords: DataFrame containing node coordinates.
        distance_matrix: Matrix of distances between nodes.
        distpath_tup: Tuple (dist_matrix, paths, dm_tensor, distC).
        distancesC: Integer distance matrix (numpy).
        paths_between_states: Precomputed paths.
        dm_tensor: Tensor version of distance matrix.
        run_tsp: Boolean indicating if TSP should be run.
        sample_id: ID of the current sample.
        overflows: Current count of overflows.
        day: Current day index.
        model_env: The model environment object.
        model_ls: Tuple of model components.
        n_vehicles: Number of vehicles.
        area: Area identifier.
        realtime_log_path: Path for real-time logging.
        waste_type: Type of waste.
        current_collection_day: Index of the current collection day.
        cached: Cached data.
        device: Torch device.
        lock: Multiprocessing lock.
        hrl_manager: Manager for Hierarchical RL.
        gate_prob_threshold: Threshold for gating probability.
        mask_prob_threshold: Threshold for masking probability.
        two_opt_max_iter: Max iterations for 2-opt local search.
        config: Configuration dictionary.
        w_length: Weight for length.
        w_waste: Weight for waste.
        w_overflows: Weight for overflows.
        engine: Policy engine.
        threshold: Decision threshold.

        # Mutable attributes added during run_day
        daily_log: Dictionary for daily logs.
        output_dict: Dictionary for output results.
        tour: The generated tour.
        cost: The cost of the route.
        profit: The profit from collection.
        collected: IDs of collected bins.
        total_collected: Total amount collected.
        ncol: Number of collections.
        new_overflows: Number of new overflows today.
        sum_lost: Amount of waste lost.
        fill: Fill levels.
        total_fill: Total fill levels.
        extra_output: Any extra output from policy.
        must_go: List of must-go bins.
    """

    # Required/Core Fields
    graph_size: int = 0
    full_policy: str = ""
    policy: str = ""
    policy_name: str = ""
    bins: Optional[Bins] = None
    new_data: Optional[pd.DataFrame] = None
    coords: Optional[pd.DataFrame] = None
    distance_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None
    distpath_tup: Tuple[Any, ...] = (None, None, None, None)
    distancesC: Optional[np.ndarray] = None
    paths_between_states: Optional[Dict[Tuple[int, int], List[int]]] = None
    dm_tensor: Optional[torch.Tensor] = None
    run_tsp: bool = False
    sample_id: int = 0
    overflows: int = 0
    day: int = 0
    model_env: Any = None
    model_ls: Tuple[Any, ...] = (None,)
    n_vehicles: int = 1
    area: str = ""
    realtime_log_path: Optional[str] = None
    waste_type: str = ""
    current_collection_day: int = 0
    cached: Optional[List[int]] = None
    device: Optional[torch.device] = None
    lock: Optional[Lock] = None
    hrl_manager: Any = None
    gate_prob_threshold: float = 0.5
    mask_prob_threshold: float = 0.5
    two_opt_max_iter: int = 0
    config: Optional[Dict[str, Any]] = None
    w_length: float = 1.0
    w_waste: float = 1.0
    w_overflows: float = 1.0
    engine: Optional[str] = None
    threshold: Optional[float] = None

    # Optional/Mutable Fields
    daily_log: Optional[Dict[str, Any]] = None
    output_dict: Optional[Dict[str, Any]] = None
    tour: Optional[List[int]] = None
    cost: float = 0.0
    profit: float = 0.0
    collected: Optional[np.ndarray] = None
    total_collected: float = 0.0
    ncol: int = 0
    new_overflows: int = 0
    sum_lost: float = 0.0
    fill: Optional[np.ndarray] = None
    total_fill: Optional[np.ndarray] = None
    extra_output: Any = None
    must_go: Optional[List[int]] = None

    @property
    def field_names(self):
        """Returns the names of all fields in the dataclass."""
        return [f.name for f in fields(self)]

    def __post_init__(self):
        """Initialize any derived or default state if needed."""
        if self.config is None:
            self.config = {}

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access to context fields."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow setting context fields via dictionary syntax."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a value from the context with an optional default."""
        return getattr(self, key, default)

    def __iter__(self):
        """Return an iterator over field names to support Mapping interface."""
        return iter((f.name for f in fields(self)))

    def __len__(self):
        """Return the number of fields to support Mapping interface."""
        return len(fields(self))

    def __contains__(self, key: object) -> bool:
        """Check if a field exists in the context."""
        if not isinstance(key, str):
            return False
        return hasattr(self, key)


def set_daily_waste(
    model_data: Dict[str, Any],
    waste: np.ndarray,
    device: torch.device,
    fill: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Updates neural model input with current bin waste levels."""
    waste_tensor = torch.as_tensor(waste, dtype=torch.float32).unsqueeze(0).div(100.0)
    if device.type == "cuda":
        waste_tensor = waste_tensor.pin_memory()
    model_data["waste"] = waste_tensor

    if "fill_history" in model_data and fill is not None:
        fill_tensor = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0).div(100.0)
        if device.type == "cuda":
            fill_tensor = fill_tensor.pin_memory()
        model_data["current_fill"] = fill_tensor
    return move_to(model_data, device, non_blocking=True)


def get_daily_results(
    total_collected: float,
    ncol: int,
    cost: float,
    tour: List[int],
    day: int,
    new_overflows: int,
    sum_lost: float,
    coordinates: pd.DataFrame,
    profit: float,
) -> Dict[str, Union[int, float, List[Union[int, str]]]]:
    """Formats raw simulation outputs into structured daily log dictionary."""
    dlog: Dict[str, Any] = {key: 0 for key in DAY_METRICS}
    dlog["day"] = day
    dlog["overflows"] = new_overflows
    dlog["kg_lost"] = sum_lost
    if tour and len(tour) > 2:
        rl_cost = new_overflows - total_collected + cost
        dlog["kg"] = total_collected
        dlog["ncol"] = ncol
        dlog["km"] = cost
        dlog["kg/km"] = total_collected / cost if cost > 0 else 0
        dlog["cost"] = rl_cost
        dlog["profit"] = profit
        ids = np.array([x for x in tour if x != 0])
        dlog["tour"] = [0] + coordinates.loc[ids, "ID"].tolist() + [0]
    else:
        dlog["kg"] = 0
        dlog["ncol"] = 0
        dlog["km"] = 0
        dlog["kg/km"] = 0
        dlog["cost"] = new_overflows
        dlog["profit"] = 0
        dlog["tour"] = [0]
    return dlog


def run_day(context: SimulationDayContext) -> SimulationDayContext:
    """Orchestrates a single simulation day using the Command Pattern."""
    from logic.src.pipeline.simulations.actions import (
        CollectAction,
        FillAction,
        LogAction,
        MustGoSelectionAction,
        PolicyExecutionAction,
        PostProcessAction,
    )

    commands = [
        FillAction(),
        MustGoSelectionAction(),
        PolicyExecutionAction(),
        PostProcessAction(),
        CollectAction(),
        LogAction(),
    ]

    for command in commands:
        command.execute(cast(Dict[str, Any], context))

    return context
