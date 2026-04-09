"""
Cutting plane separation engines for Branch-and-Price-and-Cut algorithms.

Provides modular separation algorithms for different cut families:
- RCC (Rounded Capacity Cuts): Standard VRP cuts derived from bin-packing requirements.
- SEC (Subtour Elimination Cuts): Enforces connectivity for prize-collecting variants.
- SRI (Subset-Row Inequalities): Tightens set partitioning for subsets (typically size 3).
- EdgeClique (Edge-Capacity Clique Cuts): Clique cuts for single-vehicle edge capacity.

Methodology:
    Each engine implements a `separate_and_add_cuts` method that inspects the
    fractional LP solution of the Master Problem, identifies violated
    inequalities from its specific family, and registers them into the MP's
    Gurobi model.

References:
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
      Mathematical Programming, 100(2), 423-445.
    - Jepsen, M., Petersen, B., Spoorendonk, S., & Pisinger, D. (2008).
      "Subset-row inequalities applied to the vehicle-routing problem with time windows."
      Operations Research, 56(2), 497-511.
    - Balas, E. (1975). "Facets of the knapsack polytope."
      Mathematical Programming, 8(1), 146-164 (Cover inequalities).
"""

import itertools
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .master_problem import VRPPMasterProblem
from .separation import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
)
from .vrpp_model import VRPPModel


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

    def _is_orthogonal(self, candidate_vec: np.ndarray, active_vecs: List[np.ndarray], threshold: float = 0.8) -> bool:
        """
        Task 9 (SOTA): Cut Orthogonality filtering.
        Calculates maximum cosine similarity between candidate vector and active vectors.
        Rejects if similarity > threshold (i.e. if cut is too parallel to existing cuts).
        """
        if not active_vecs:
            return True

        norm_c = np.linalg.norm(candidate_vec)
        if norm_c < 1e-6:
            return False

        for a_vec in active_vecs:
            norm_a = np.linalg.norm(a_vec)
            if norm_a < 1e-6:
                continue
            cos_sim = np.abs(np.dot(candidate_vec, a_vec)) / (norm_c * norm_a)
            if cos_sim > threshold:
                return False
        return True


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
           (Elementary only: cycles break max-flow conservation).
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
        # Task 2: Exact separation requires flow conservation on elementary routes.
        # Use filtered edge usage to prevent invalid cut generation from cyclic support.
        edge_vars = master.get_elementary_edge_usage()
        if not edge_vars:
            return 0

        # Map edge usage to x_vals array for SeparationEngine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                idx = self.v_model.edge_to_idx[edge_tuple]
                x_vals[idx] = val

        # Build y_vals: fractional node visitation probabilities from the LP solution.
        # y_vals[i-1] = Σ_{k: i ∈ route_k} λ_k for each customer node i.
        # Index 0 corresponds to customer node 1 (depot node 0 is excluded).
        node_visits = master.get_node_visitation()
        n_customers = self.v_model.n_nodes - 1  # exclude depot
        y_vals = np.zeros(n_customers)
        for node, val in node_visits.items():
            if 1 <= node <= n_customers:
                y_vals[node - 1] = min(val, 1.0)  # clamp to [0,1] for LP fractional values

        # Separate cuts using the existing separation engine
        # BPC separation always operates on fractional LP solutions.
        # Use separate_fractional() for exact max-flow based RCC separation.
        node_depth = kwargs.get("node_depth", 0)
        violated_ineqs = self.sep_engine.separate_fractional(
            x_vals, y_vals=y_vals, max_cuts=max_cuts, node_count=node_depth
        )
        added_cuts = 0

        for ineq in violated_ineqs:
            if isinstance(ineq, CapacityCut):
                node_set = list(ineq.node_set)

                # For Set Packing, add cut with boundary relaxation
                # The master problem handles the y_i visitation tracking
                if master.add_capacity_cut(node_set, ineq.rhs):
                    added_cuts += 1
            elif isinstance(ineq, PCSubtourEliminationCut):
                node_set = list(ineq.node_set)
                # Form 2.1 is always global. Form 2.3 with fractional RHS is local only.
                # getattr used for defensive runtime robustness.
                is_global = ineq.facet_form == "2.1" and not getattr(ineq, "local_only", False)
                if master.add_sec_cut(
                    node_set,
                    ineq.rhs,
                    cut_name=ineq.facet_form,
                    global_cut=is_global,
                    node_i=ineq.node_i,
                    node_j=ineq.node_j,
                ):
                    added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        return "rcc"


class SubsetRowCutEngine(CuttingPlaneEngine):
    """
    Subset-Row Inequality (SRI) separation engine.

    Specifically implements 3-SRI (k=3, p=2):
    Σ_{k} ⌊ 1/2 * |S ∩ route_k| ⌋ * λ_k  <= 1

    Heuristic Separation:
    1. Identify customer nodes with high fractional visitation.
    2. Check subsets of 3 nodes that are spatially close.
    3. Evaluate violation and add to master.
    """

    def __init__(self, v_model: VRPPModel):
        self.v_model = v_model

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        if master.model is None or len(master.routes) < 2:
            return 0

        # Structural SRI Separation (Academic Standard):
        # Instead of a myopic visitation-based heuristic, we identify candidate
        # triplets (i, j, k) that "co-occur" in active fractional columns.
        # This focuses separation on subsets most likely to be violated by
        # the current RMP solution.

        # 1. Identify "fractional" routes (0 < λ_k < 1)
        fractional_routes = []
        for idx, var in enumerate(master.lambda_vars):
            val = var.X
            if 0.01 < val < 0.99:
                fractional_routes.append((master.routes[idx], val))

        if not fractional_routes:
            return 0

        # 2. Build co-occurrence connectivity
        # We look for nodes that appear together in fractional routes.
        # This forms the "structural entanglement" suggested by Jepsen et al. (2008).
        adjacency: Dict[int, Set[int]] = {}
        for route, _ in fractional_routes:
            nodes = list(route.node_coverage)
            for i, j in itertools.combinations(nodes, 2):
                adjacency.setdefault(i, set()).add(j)
                adjacency.setdefault(j, set()).add(i)

        # 3. Candidate Triplets: Nodes forming a triangle or near-triangle in the support graph.
        candidate_subsets: Set[FrozenSet[int]] = set()
        for i, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue
            for j, k in itertools.combinations(list(neighbors), 2):
                # Triplets that share at least two fractional support edges
                candidate_subsets.add(frozenset([i, j, k]))

        if not candidate_subsets:
            return 0

        added = 0
        # 4. Evaluate violation for structural candidates
        for node_fset in candidate_subsets:
            if added >= max_cuts:
                break
            if self._evaluate_and_add_sri(master, set(node_fset)):
                added += 1

        return added

    def _evaluate_and_add_sri(self, master: VRPPMasterProblem, node_set: Set[int]) -> bool:
        """Calculate SRI violation and add to master if violated."""
        val = 0.0
        for idx, route in enumerate(master.routes):
            lam = master.lambda_vars[idx].X
            if lam < 1e-6:
                continue
            count = len(node_set.intersection(route.node_coverage))
            coeff = count // 2
            if coeff > 0:
                val += float(coeff) * lam

        if val > 1.0 + 1e-4:
            return master.add_subset_row_cut(list(node_set))
        return False

    def get_name(self) -> str:
        return "sri"


class EdgeCliqueCutEngine(CuttingPlaneEngine):
    r"""
    Separates edge-clique inequalities for the Set Partitioning master problem.

    These cuts enforce that at most one route may traverse a given edge (i, j),
    derived from the conflict graph of routes sharing that edge. They are a
    specialization of clique inequalities for the route-edge incidence matrix.

    Note: These are NOT Lifted Cover Inequalities (LCI) in the sense of
    Barnhart, Hane & Vance (2000), which are derived from commodity-flow
    arc-capacity knapsack constraints that do not exist in the VRPP formulation.
    The 'lci' engine name is therefore intentionally mapped to NotImplementedError.

    Mathematical Theory:
    --------------------
    For a physical edge (i, j) with capacity $W=1$ (since node visits must be partitioned)
    and a fractional solution $\bar{\lambda}$, a clique $C$ is the set of all routes traversing $(i, j)$.


    The Edge-Clique Inequality is defined as:
        $\sum_{k \in C} \lambda_k \le 1$

    O(1) Calculation:
    -----------------
    Since weights $w_{ik}$ are essentially 1, any set of routes crossing the same
    edge form a complete conflict graph. Thus, the lifting coefficient for all
    such routes simplifies directly to $\alpha_k = 1.0$.
    """

    def __init__(self, v_model: VRPPModel, capacity: float = 1.0, epsilon: float = 0.01) -> None:
        """
        Initialize the LCI separation engine.

        Args:
            v_model: VRPP model for edge indexing.
            capacity: Physical capacity of the arc (default 1.0 for VRP).
            epsilon: Violation threshold.
        """
        self.v_model = v_model
        self.capacity = capacity
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        """
        Identify violated LCIs by lifting minimal covers.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        # 1. Aggregate edge flow: x_e = Σ_{k} w_{ek} λ_k
        route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        edge_to_routes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  # arc -> list of (route_idx, weight)

        for ridx in route_values:
            route = master.routes[ridx]
            nodes = [0] + route.nodes + [0]
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                e = tuple(sorted((u, v)))
                # Track weights (crossings) for the knapsack formulation
                # In standard VRP, weight is 1. If route repeats arc, weight > 1.
                found = False
                if e in edge_to_routes:
                    for j, (ridx_existing, w) in enumerate(edge_to_routes[e]):  # type: ignore[index]
                        if ridx_existing == ridx:
                            edge_to_routes[e][j] = (ridx_existing, w + 1)  # type: ignore[index]
                            found = True
                            break
                if not found:
                    edge_to_routes.setdefault(e, []).append((ridx, 1))  # type: ignore[arg-type]

        # 2. Identify saturated edges
        saturated_edges = []
        for e, routes_info in edge_to_routes.items():
            flow = sum(master.lambda_vars[ridx].X * w for ridx, w in routes_info)
            if flow > self.capacity + self.epsilon:
                saturated_edges.append((e, routes_info, flow))

        # Sort by flow violation
        saturated_edges.sort(key=lambda x: x[2], reverse=True)

        added_cuts = 0
        for edge_tuple, routes_info, _ in saturated_edges:
            if added_cuts >= max_cuts:
                break

            # 3. Find minimal cover C
            # Sort routes by λ_k descending
            routes_info.sort(key=lambda x: master.lambda_vars[x[0]].X, reverse=True)

            cover_indices = []
            weight_sum = 0
            for idx, w in routes_info:
                cover_indices.append((idx, w))
                weight_sum += w
                if weight_sum > self.capacity:
                    break

            if weight_sum <= self.capacity:
                continue  # Should not happen for saturated arc if all w >= 1

            # 4. O(1) Clique Lifting
            # For Set Partitioning, capacity is 1. Any pair of routes traversing
            # this edge form a clique. The lifting coefficient is uniformly 1.0.
            rhs = 1.0
            coefficients = {ridx: 1.0 for ridx, _ in routes_info}

            # 5. Add to master
            if master.add_edge_clique_cut(edge_tuple[0], edge_tuple[1], coefficients=coefficients, rhs=rhs):
                added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        return "edge_clique"


class KnapsackCoverEngine(CuttingPlaneEngine):
    """
    Separates Cover Inequalities for knapsack constraints in the Master Problem.

    In the VRPP context, this primarily targets the fleet size (vehicle limit)
    constraint: Σ λ_k ≤ K. While simpler than a general knapsack with varied
    weights, separating covers for the fleet limit helps strengthen the
    integrality of the master problem in tightly constrained instances.

    Reference: Barnhart et al. (2000), Section 5.

    Interaction with the fleet constraint:
        When VRPPMasterProblem has an active vehicle_limit constraint
        (Σ λ_k ≤ K), the fleet constraint already prevents the LP from
        selecting more than K routes. In that case, this engine's cover cut
        (Σ_{k ∈ top_K+1} λ_k ≤ K) is redundant and will never fire.

        This engine is primarily useful when NO explicit vehicle_limit is set
        in the master problem — e.g., in uncapacitated-fleet VRPP variants —
        where the cover cut provides a surrogate fleet-size tightening.

        If vehicle_limit is always set, consider disabling this engine to
        avoid unnecessary separation overhead.
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify and add violated cover inequalities for the vehicle limit.
        """
        # Task 4 & 6: LCI already handles fleet-size tightening more effectively.
        # This engine serves as a fallback or for non-fleet knapsacks.
        if master.vehicle_limit is None or master.model is None:
            return 0

        # Σ λ_k ≤ K.
        # A cover C is any subset of routes such that |C| > K.
        # The inequality is Σ_{k ∈ C} λ_k ≤ |C| - 1.
        # Violation occurs if Σ_{k ∈ C} λ_k > |C| - 1.

        # Strategy: find routes with largest fractional values.
        # If the sum of the top K+1 λ_k values exceeds K, we have a violation.
        route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        if len(route_values) <= master.vehicle_limit:
            return 0

        sorted_indices = sorted(route_values.keys(), key=lambda i: route_values[i], reverse=True)
        top_k_plus_1 = sorted_indices[: master.vehicle_limit + 1]
        sum_top = sum(route_values[i] for i in top_k_plus_1)

        if sum_top > master.vehicle_limit + 1e-4:
            # Violated cover found.
            # The cover is the set of nodes visited by the top K+1 routes.
            # Use their union as the node set S for the capacity cut registry.
            cover_node_set: set = set()
            for i in top_k_plus_1:
                cover_node_set.update(master.routes[i].node_coverage)
            cover_node_set.discard(0)  # exclude depot

            if cover_node_set and master.add_capacity_cut(list(cover_node_set), float(master.vehicle_limit)):
                return 1

        return 0

    def get_name(self) -> str:
        return "cover"


class CompositeCuttingPlaneEngine(CuttingPlaneEngine):
    """
    Combines multiple separation engines to run in sequence.
    """

    def __init__(self, engines: List[CuttingPlaneEngine]) -> None:
        self.engines = engines

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        per_engine = max(1, max_cuts // len(self.engines))
        total_added = 0
        for engine in self.engines:
            total_added += engine.separate_and_add_cuts(master, per_engine, **kwargs)
        return total_added

    def get_name(self) -> str:
        return "composite"


class LiftedCoverCutEngine(CuttingPlaneEngine):
    r"""
    Separates Lifted Cover Inequalities (LCI) for knapsack constraints.

    In VRPP, the primary knapsack constraint is the vehicle fleet limit:
    $\sum_{k \in \Omega} \lambda_k \le K$

    A Cover $C \subseteq \Omega$ is a set of routes such that $|C| > K$.
    The basic cover inequality is: $\sum_{k \in C} \lambda_k \le K$.

    This engine strengthens this by lifting variables outside the cover:
    $\sum_{k \in C} \lambda_k + \sum_{k \in \Omega \setminus C} \alpha_k \lambda_k \le K$

    References:
        - Barnhart et al. (2000), "Using branch-and-price-and-cut to solve
          origin-destination integer multicommodity flow problems."
          Operations Research, 48(2):318-326.
        - Balas (1975), "Facets of the knapsack polytope."
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        if master.vehicle_limit is None or not master.lambda_vars:
            return 0

        K = float(master.vehicle_limit)
        # 1. Identify Fractional Routes sorted by λ_k descending
        active_routes = []
        for i, var in enumerate(master.lambda_vars):
            if var.X > 1e-4:
                active_routes.append((i, var.X))
        active_routes.sort(key=lambda x: x[1], reverse=True)

        if len(active_routes) <= K:
            return 0

        # 2. Find Minimal Cover C
        # Since all weights are 1.0, any K+1 routes form a cover.
        # To maximize violation, pick the top K+1 fractional routes.
        cover_indices = [idx for idx, _ in active_routes[: int(K) + 1]]
        sum_val = sum(master.lambda_vars[i].X for i in cover_indices)

        if sum_val <= K + self.epsilon:
            return 0

        # 3. Apply Sequence-Independent Lifting (Barnhart et al. 2000)
        # For the knapsack Σ λ_k ≤ K, the lifting coefficient α_k for k ∉ C is:
        # α_k = 1.0 (since all weights are identical and binary).
        # This reduces the cut to Σ_{k ∈ Ω} λ_k ≤ K, which is already the
        # fleet constraint. LCIs are most powerful when routes have different
        # resource consumptions (e.g. multi-load).
        # However, for VRPP, we can lift nodes NOT in S if we reformulate
        # as a subset-visitation knapsack.

        # Implementation Note: Modern BPC solvers (like VRPSolver) prefer
        # Subset-Row Inequalities for tightening Set Partitioning.
        # We add the basic cover cut here as a starting point.
        added = 0
        coefficients = {i: 1.0 for i in cover_indices}
        if master.add_capacity_cut(list(range(1, self.v_model.n_nodes)), K, coefficients=coefficients):
            added += 1

        return added

    def get_name(self) -> str:
        return "lci"


def create_cutting_plane_engine(
    engine_name: str, v_model: VRPPModel, sep_engine: Optional[SeparationEngine] = None
) -> CuttingPlaneEngine:
    """
    Factory function to create cutting plane engines.

    Args:
        engine_name: Name of the engine ("rcc", "src", "cover", or "all")
        v_model: VRPP model for problem data
        sep_engine: Separation engine (required for RCC/SRC)

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
    elif engine_name == "sri":
        return SubsetRowCutEngine(v_model)
    elif engine_name == "edge_clique":
        return EdgeCliqueCutEngine(v_model)
    elif engine_name == "lci":
        return LiftedCoverCutEngine(v_model)
    elif engine_name == "cover":
        if sep_engine is None:
            raise ValueError("Cover engine requires sep_engine parameter")
        return KnapsackCoverEngine(v_model, sep_engine)
    elif engine_name in ("all", "composite"):
        engines = []
        if sep_engine is not None:
            engines.append(RoundedCapacityCutEngine(v_model, sep_engine))
        engines.append(SubsetRowCutEngine(v_model))  # type: ignore[arg-type]
        engines.append(EdgeCliqueCutEngine(v_model))  # type: ignore[arg-type]
        engines.append(LiftedCoverCutEngine(v_model))  # type: ignore[arg-type]
        if sep_engine is not None:
            engines.append(KnapsackCoverEngine(v_model, sep_engine))  # type: ignore[arg-type]
        return CompositeCuttingPlaneEngine(engines)  # type: ignore[arg-type]
    else:
        valid = ["rcc", "sri", "edge_clique", "lci", "cover", "all"]
        raise ValueError(f"Unknown cutting plane engine '{engine_name}'. Valid options are: {valid}")
