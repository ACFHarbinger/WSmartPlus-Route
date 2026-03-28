"""
Branch-and-Bound Optimizer Interface.

Provides a unified dispatcher for selecting between different B&B formulations:
- MTZ (Miller-Tucker-Zemlin): Compact formulation with load variables
- DFJ (Dantzig-Fulkerson-Johnson): Lazy subtour elimination with Gurobi's B&B

This architecture mirrors the SWC-TCF dispatcher pattern, allowing seamless
switching between mathematical formulations based on problem characteristics.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .dfj import run_bb_dfj
from .mtz import run_bb_mtz


def run_bb_optimizer(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    must_go_indices: Optional[Set[int]] = None,
    env: Optional[gp.Env] = None,
    seed: Optional[int] = None,
    recorder: Optional[PolicyStateRecorder] = None,
    formulation: str = "dfj",
) -> Tuple[List[List[int]], float]:
    """
    Solve VRPP using the specified Branch-and-Bound formulation.

    This dispatcher routes the problem to the appropriate B&B implementation
    based on the formulation parameter. The choice of formulation affects:
    - Subtour elimination mechanism
    - LP relaxation tightness
    - Computational performance
    - Memory footprint

    **Formulation Comparison**:

    **MTZ (Miller-Tucker-Zemlin)**:
    - Compact formulation with O(n²) constraints
    - Load variables track accumulated waste
    - Subtours prevented implicitly in LP relaxation
    - Custom B&B tree management
    - Better for: Small to medium instances, custom branching strategies

    **DFJ (Dantzig-Fulkerson-Johnson)**:
    - Exponential constraints added lazily
    - Delegates B&B to Gurobi's internal engine
    - Subtours eliminated dynamically via callbacks
    - Leverages Gurobi's advanced heuristics
    - Better for: Large instances, when Gurobi license available

    Args:
        dist_matrix: Symmetric distance matrix (n x n).
        wastes: Mapping of customer IDs to fill levels/profits.
        capacity: Vehicle payload capacity.
        R: Revenue coefficient per unit collected.
        C: Cost coefficient per unit distance.
        values: Configuration dictionary (time_limit, mip_gap, etc.).
        must_go_indices: Set of mandatory customer nodes.
        env: Optional Gurobi environment for resource management.
        seed: Optional random seed for reproducibility.
        recorder: Optional telemetry recorder for state tracking.
        formulation: B&B formulation to use ("mtz" or "dfj"). Defaults to "dfj".

    Returns:
        Tuple of (routes, objective_value).

    Raises:
        ValueError: If formulation is not "mtz" or "dfj".

    Examples:
        >>> # Use DFJ for large instances
        >>> routes, obj = run_bb_optimizer(
        ...     dist_matrix, wastes, capacity, R, C, values,
        ...     formulation="dfj"
        ... )

        >>> # Use MTZ for custom branching
        >>> routes, obj = run_bb_optimizer(
        ...     dist_matrix, wastes, capacity, R, C, values,
        ...     formulation="mtz"
        ... )
    """
    if formulation == "mtz":
        return run_bb_mtz(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            values=values,
            must_go_indices=must_go_indices,
            env=env,
            seed=seed,
            recorder=recorder,
        )
    elif formulation == "dfj":
        return run_bb_dfj(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            values=values,
            must_go_indices=must_go_indices,
            env=env,
            seed=seed,
            recorder=recorder,
        )
    else:
        raise ValueError(
            f"Unknown B&B formulation: '{formulation}'. "
            f"Supported formulations: 'mtz', 'dfj'. "
            f"Use 'mtz' for compact MTZ formulation or 'dfj' for lazy DFJ cuts."
        )
