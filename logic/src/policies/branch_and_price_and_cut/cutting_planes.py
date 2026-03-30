"""
Cutting plane separation engines for Branch-and-Price-and-Cut algorithms.

Provides modular separation algorithms for different cut families:
- RCC (Rounded Capacity Cuts): For capacitated vehicle routing
- LCI: (DEPRECATED) Lifted Cover Inequalities require complex RCSPP state-space
  modifications for dual integration and are currently disabled.

References:
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
      Mathematical Programming, 100(2), 423-445.
      (Rounded Capacity Cuts)

    - Balas, E. (1975). "Facets of the knapsack polytope."
      Mathematical Programming, 8(1), 146-164.
      (Cover inequalities)

    - Zemel, E. (1989). "Easily computable facets of the knapsack polytope."
      Mathematics of Operations Research, 14(4), 760-764.
      (Lifting procedures for cover inequalities)
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..branch_and_cut.separation import SeparationEngine
from ..branch_and_cut.vrpp_model import VRPPModel
from ..branch_and_price.master_problem import VRPPMasterProblem


class CuttingPlaneEngine(ABC):
    """
    Abstract base class for cutting plane separation engines.

    Each engine implements a specific family of valid inequalities and
    provides separation algorithms to identify violated cuts.
    """

    @abstractmethod
    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify violated cuts and add them to the master problem.

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Engine-specific parameters

        Returns:
            Number of cuts added
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the engine name for logging."""
        pass


class RoundedCapacityCutEngine(CuttingPlaneEngine):
    """
    Rounded Capacity Cut (RCC) separation engine for VRPP Set Packing.

    Mathematical Formulation:
    -------------------------
    For a subset S ⊆ N \\ {0} of customer nodes, the RCC is:

        ∑_{e ∈ δ(S)} x_e ≥ 2 * k(S)

    where:
    - δ(S) is the set of edges crossing the boundary of S
    - k(S) = ⌈demand(S) / Q⌉ is the minimum number of vehicles needed
    - x_e is the aggregated edge usage from the column generation solution

    For VRPP Set Packing, we relax this with visitation variables:

        ∑_{e ∈ δ(S)} x_e ≥ 2*k(S) - M * ∑_{i ∈ S} (1 - y_i)

    where y_i = ∑_{k: i ∈ route_k} λ_k is the fractional visitation probability.

    Integration with Pricing:
    --------------------------
    The dual variables from these cuts must be integrated into the pricing
    problem's reduced cost calculation. For each cut on set S:

        reduced_cost -= dual_S * (number of times route crosses δ(S))

    This is handled by the master problem's dual tracking and passed to
    the RCSPP solver via capacity_cut_duals parameter.

    References:
    -----------
    Lysgaard et al. (2004), Section 3.2: "Capacity cuts"
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        """
        Initialize the RCC separation engine.

        Args:
            v_model: VRPP model for edge indexing and demand tracking
            sep_engine: Separation engine from branch-and-cut module
        """
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Separate RCCs and add to master problem.

        Algorithm:
        1. Get current edge usage from master problem's LP solution
        2. Map edge usage to the separation engine's representation
        3. Run separation algorithms (heuristic + exact)
        4. Add violated cuts to master with Set Packing relaxation

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Unused (for interface consistency)

        Returns:
            Number of cuts added
        """
        edge_vars = master.get_edge_usage()
        if not edge_vars:
            return 0

        # Map edge usage to x_vals array for SeparationEngine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                idx = self.v_model.edge_to_idx[edge_tuple]
                x_vals[idx] = val

        # Separate cuts using the existing separation engine
        violated_ineqs = self.sep_engine.separate(x_vals, max_cuts=max_cuts)
        added_cuts = 0

        for ineq in violated_ineqs:
            if ineq.type == "CAPACITY":
                node_set = list(ineq.node_set)

                # For Set Packing, add cut with boundary relaxation
                # The master problem handles the y_i visitation tracking
                if master.add_set_packing_capacity_cut(node_set, ineq.rhs):
                    added_cuts += 1

        # Re-solve LP if cuts were added
        if added_cuts > 0:
            master.solve_lp_relaxation()

        return added_cuts

    def get_name(self) -> str:
        return "rcc"


# class LiftedCoverInequalityEngine(CuttingPlaneEngine):
#     """
#     Lifted Cover Inequality (LCI) separation engine for knapsack constraints.
#     ... (commented out for mathematical rigor)
#     """


def create_cutting_plane_engine(
    engine_name: str, v_model: VRPPModel, sep_engine: Optional[SeparationEngine] = None
) -> CuttingPlaneEngine:
    """
    Factory function to create cutting plane engines.

    Args:
        engine_name: Name of the engine ("rcc" or "lci")
        v_model: VRPP model for problem data
        sep_engine: Separation engine (required for RCC)

    Returns:
        Instance of the requested cutting plane engine

    Raises:
        ValueError: If engine_name is not recognized or required args missing

    Example:
        >>> sep_engine = SeparationEngine(v_model)
        >>> cut_engine = create_cutting_plane_engine("rcc", v_model, sep_engine)
        >>> cuts_added = cut_engine.separate_and_add_cuts(master, max_cuts=5)
    """
    if engine_name == "rcc":
        if sep_engine is None:
            raise ValueError("RCC engine requires sep_engine parameter")
        return RoundedCapacityCutEngine(v_model, sep_engine)
    elif engine_name == "lci":
        raise ValueError(
            "LCI engine is currently disabled. Lifted Cover Inequalities "
            "require complex RCSPP state-space modifications for dual integration "
            "to ensure valid LP bounds."
        )
    else:
        raise ValueError(f"Unknown cutting plane engine '{engine_name}'. Valid options are: ['rcc']")
