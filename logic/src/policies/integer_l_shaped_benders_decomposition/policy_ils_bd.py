r"""
Integer L-Shaped (Benders Decomposition) Policy Adapter.

Adapts the Benders decomposition engine to the system-agnostic routing policy
interface (BaseRoutingPolicy), handling parameter mapping, VRPPModel construction,
engine invocation, and raw-distance reporting.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import IntegerLShapedBendersConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.other.branching_solvers.vrpp_model import VRPPModel

from .ils_bd_engine import IntegerLShapedEngine
from .params import ILSBDParams


@PolicyRegistry.register("ils_bd")
class IntegerLShapedPolicy(BaseRoutingPolicy):
    r"""Integer L-Shaped (Benders Decomposition) policy for stochastic VRPP / SCWCVRP.

    Formulates the routing-under-uncertainty problem as a Two-Stage Stochastic
    Integer Program (2-SIP):

        Stage 1 (Master Problem):
            Binary routing arcs x_{ij} and node-visit flags y_i are decided
            before stochastic demand is revealed.  A surrogate variable θ
            represents the expected recourse cost.

        Stage 2 (Recourse Subproblem):
            For each SAA scenario ω, the penalty incurred by the Stage 1
            decision is evaluated against the realised bin fill level w_i(ω):
                - Overflow penalty p⁺: unvisited bins above threshold.
                - Undervisit penalty p⁻: visited bins below threshold.

    The Integer L-Shaped Method (Laporte & Louveaux, 1993) then iterates:
        1. Solve MP → integer routing + surrogate θ̂.
        2. Evaluate Q̄(ŷ̂) analytically over SAA scenarios.
        3. If Q̄(ŷ̂) > θ̂ + gap: add Benders optimality cut; repeat.
        4. Else: θ̂ accurately represents recourse → optimal.

    Because Stage 1 variables are binary, the cuts are *combinatorial* L-shaped
    cuts, valid at every binary point (not merely at the LP relaxation).

    Registry key: ``"ils_bd"``

    References:
        Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
        stochastic integer programs with complete recourse". Operations Research
        Letters, 13(3), 133-142.
    """

    def __init__(
        self,
        config: Optional[Union[IntegerLShapedBendersConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the ILS policy adapter.

        Args:
            config: IntegerLShapedBendersConfig dataclass, raw configuration dict from YAML, or
                    None to use all framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the typed configuration class for automatic parsing."""
        return IntegerLShapedBendersConfig

    def _get_config_key(self) -> str:
        """Return the YAML config key for this policy."""
        return "ils_bd"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        r"""Run the Integer L-Shaped Benders decomposition solver.

        Constructs a VRPPModel from the localised sub-problem, initialises the
        ILS engine with the merged configuration, executes the Benders outer
        loop, and returns the best feasible routing solution.

        Args:
            sub_dist_matrix: Localised N×N distance matrix (0 = depot).
            sub_wastes: Observed fill levels {local_node_idx: fill_%}.
            capacity: Vehicle payload capacity (problem units).
            revenue: Revenue per unit of waste collected (€/%-fill unit).
            cost_unit: Travel cost per distance unit (€/km).
            values: Merged configuration dict (area defaults + YAML overrides).
            mandatory_nodes: Local node indices that MUST be visited.
            **kwargs: Additional simulation context (n_vehicles, etc.).

        Returns:
            Tuple of (routes, profit, raw_distance):
                routes: List of customer-node route lists (depot excluded).
                profit: Deterministic net profit = revenue − travel_cost.
                        The recourse surrogate θ is excluded to maintain
                        compatibility with the BaseRoutingPolicy interface.
                raw_distance: Total depot-to-depot travel distance (km).
        """
        n_nodes = len(sub_dist_matrix)

        # ---- Build VRPPModel (network structure for Gurobi) ---------------
        must_go_set: Set[int] = set(mandatory_nodes)
        vrpp_model = VRPPModel(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=must_go_set,
        )

        # Override fleet size from simulation context if explicitly provided
        n_vehicles = kwargs.get("n_vehicles")
        if n_vehicles is not None:
            vehicle_limit = int(n_vehicles) if int(n_vehicles) > 0 else None
            if vehicle_limit is not None:
                vrpp_model.num_vehicles = vehicle_limit

        # ---- Build typed ILS parameters -----------------------------------
        params = ILSBDParams.from_config(values)

        # ---- Run Benders decomposition ------------------------------------
        engine = IntegerLShapedEngine(model=vrpp_model, params=params)
        routes, _y_hat, profit, _stats = engine.solve(sub_wastes=sub_wastes)

        # ---- Compute raw travel distance (km) ----------------------------
        raw_distance = 0.0
        for route in routes:
            inner = [n for n in route if n != 0]
            if not inner:
                continue
            path = [0] + inner + [0]
            for k in range(len(path) - 1):
                raw_distance += float(sub_dist_matrix[path[k]][path[k + 1]])

        return routes, profit, raw_distance
