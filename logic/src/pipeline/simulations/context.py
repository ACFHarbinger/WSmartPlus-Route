"""
Simulation state encapsulation and context management.

This module provides the SimulationDayContext dataclass which groups
all state variables for a single simulation day.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SimulationDayContext(Mapping):
    """
    Context object encapsulating the state of a simulation day.

    This object replaces the loosely typed dictionary previously used to pass state
    between `run_day` and the various `Action` classes. It provides structured
    access to simulation variables while maintaining backward compatibility via
    dictionary-like access methods (`__getitem__`, `get`).

    Attributes:
        graph_size: Size of the graph (number of nodes).
        full_policy: Full string identifier of the policy (e.g., 'policy_regular3_gamma1').
        policy: Parsed policy name.
        policy_name: Name of the policy.
        bins: The Bins object managing waste levels.
        new_data: DataFrame containing new data for the day.
        coords: DataFrame containing node coordinates.
        distance_matrix: Matrix of distances between nodes.
        distpath_tup: Tuple containing distance and path information (dist_matrix, paths, dm_tensor, distC).
        distancesC: Integer distance matrix (numpy).
        paths_between_states: precomputed paths.
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

        # Helper/Mutable attributes often added during run_day
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
    """

    # Required/Core Fields
    graph_size: int = 0
    full_policy: str = ""
    policy: str = ""
    policy_name: str = ""
    bins: Any = None
    new_data: Any = None
    coords: Any = None
    distance_matrix: Any = None
    distpath_tup: Tuple = (None, None, None, None)
    distancesC: Any = None
    paths_between_states: Any = None
    dm_tensor: Any = None
    run_tsp: bool = False
    sample_id: int = 0
    overflows: int = 0
    day: int = 0
    model_env: Any = None
    model_ls: Tuple = (None,)
    n_vehicles: int = 1
    area: str = ""
    realtime_log_path: Optional[str] = None
    waste_type: str = ""
    current_collection_day: int = 0
    cached: Any = None
    device: Any = None
    lock: Any = None
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

    # Optional/Mutable Fields (defaults to None or reasonable zero)
    daily_log: Optional[Dict] = None
    output_dict: Optional[Dict] = None
    tour: Optional[List[int]] = None
    cost: float = 0.0
    profit: float = 0.0
    collected: Any = None
    total_collected: float = 0.0
    ncol: int = 0
    new_overflows: int = 0
    sum_lost: float = 0.0
    fill: Any = None
    total_fill: Any = None
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
