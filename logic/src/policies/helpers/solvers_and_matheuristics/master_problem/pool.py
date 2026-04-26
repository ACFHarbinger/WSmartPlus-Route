"""
Global Cut Pool for Branch-and-Price-and-Cut.

Attributes:
    CutInfo: Metadata describing a generated valid inequality.
    GlobalCutPool: Centralized repository for globally valid inequalities.

Example:
    >>> pool = GlobalCutPool()
    >>> pool.add_cut("rcc", (frozenset({1, 2}), 2.0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Set, Tuple

if TYPE_CHECKING:
    from .problem_support import MasterProblemSupport


@dataclass
class CutInfo:
    """
    Metadata describing a generated valid inequality.

    Attributes:
        type (str): Type of cut (e.g., 'rcc', 'sec', 'sri', 'lci').
        data (Any): Cut-specific data (node-set, RHS, lifting coeffs, etc.).
        active (bool): Whether the cut is active.
        violation (float): The violation amount of the cut.
    """

    type: str  # e.g., 'rcc', 'sec', 'sri', 'lci'
    data: Any  # node-set, RHS, lifting coeffs, etc.
    active: bool = True
    violation: float = 0.0


class GlobalCutPool:
    """
    Centralized repository for globally valid inequalities across B&B nodes.

    Philosophy:
    In BPC, separation is expensive. By pooling valid inequalities (RCC, SRI, SEC 2.1)
    globally, we ensure that a cut discovered in one branch tightens the LP bound
    in sibling and child branches immediately, avoiding redundant separation and
    reducing the total number of B&B nodes explored.

    RCC storage note:
        RCC cuts are stored as (node_set, rhs) pairs so that the original RHS
        (= 2*⌈demand(S)/Q⌉, computed at discovery) is faithfully replayed when
        the cut is re-injected at descendant nodes. Storing only the node set
        and hard-coding rhs=1.0 would produce trivially weak cuts.

        Assumption: Customer demands and vehicle capacities are static throughout
        the B&B tree. If these were dynamic (e.g., stochastic demands handled at
        internal nodes), the RHS of purely node-set-based cuts could change,
        invalidating the global mathematical integrity of this archive.

    Attributes:
        rcc_cuts: Dictionary mapping node sets to their RHS values.
        sri_cuts: Set of node sets with SRI cuts.
        active_sri_vectors: Active SRI coefficient vectors.
        sec_cuts: Set of node sets with SEC Form 2.1 cuts.
        edge_clique_cuts: Set of edges with Edge Clique cuts.
        lci_cuts: Dictionary mapping node sets to LCI cut data.
        lci_arcs: Arcs associated with LCI cuts.
    """

    def __init__(self) -> None:
        """Initializes empty global cut registries.

        Sets up dictionaries and sets for RCC, SRI, SEC, Edge Clique, and LCI cuts.

        Args:
            None

        Returns:
            None
        """
        # RCC: maps node_set -> original rhs (2*ceil(demand/Q))
        self.rcc_cuts: Dict[FrozenSet[int], float] = {}
        self.sri_cuts: Set[FrozenSet[int]] = set()
        self.active_sri_vectors: Dict[FrozenSet[int], Dict[str, float]] = {}
        self.sec_cuts: Set[FrozenSet[int]] = set()  # Form 2.1 (Global)
        self.edge_clique_cuts: Set[Tuple[int, int]] = set()
        # LCI: maps node_set -> (rhs, route_coefficients, node_alphas)
        # node_alphas: per-node lifting coefficients for pricing (Barnhart et al. 2000 §4.2)
        self.lci_cuts: Dict[FrozenSet[int], Tuple[float, Dict[int, float], Dict[int, float]]] = {}
        # Optional source arc (i, j) for arc-saturation LCI (SaturatedArcLCIEngine).
        # When set, the pricing dual fires ONLY when the DP traverses that specific arc,
        # not on any visit to a node in the cover set.  None for node/capacity LCI.
        self.lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}
        # Multistar: maps node_set -> route_coefficients {route_idx: -a_k}
        # Duals γ_S are applied per-arc in the RCSPP via multistar_duals.
        # (Letchford, Eglese, Lysgaard 2002 — Generalized Multistar Inequalities)
        self.multistar_cuts: Dict[FrozenSet[int], Dict[int, float]] = {}

    def add_cut(self, cut_type: str, data: Any) -> None:
        """Archive a globally valid cut in the pool.

        Only archives cuts that are valid at EVERY node in the B&B tree.
        Node-local cuts must not be added here.

        Args:
            cut_type: Type of cut ("rcc", "sri", "sec_2.1", "edge_clique", "lci").
            data: Cut-specific data (node-sets, coefficients, etc.).

        Returns:
            None
        """

        if cut_type == "rcc":
            node_set, rhs = data
            # Only archive if better (tighter) than any existing cut on this set.
            existing = self.rcc_cuts.get(node_set, 0.0)
            if rhs > existing:
                self.rcc_cuts[node_set] = rhs
        elif cut_type == "sri":
            node_set, coeff_vec = data
            self.sri_cuts.add(node_set)
            self.active_sri_vectors[node_set] = coeff_vec
        elif cut_type == "sec_2.1":
            self.sec_cuts.add(data)
        elif cut_type == "edge_clique":
            self.edge_clique_cuts.add(data)
        elif cut_type == "lci":
            # Accept 3-tuple, 4-tuple (+ node_alphas), or 5-tuple (+ arc) variants.
            # 5-tuple: (node_set, rhs, coefficients, node_alphas, arc)
            # 4-tuple: (node_set, rhs, coefficients, node_alphas)
            # 3-tuple: (node_set, rhs, coefficients)
            if len(data) == 5:
                node_set, rhs, coefficients, node_alphas, arc = data
            elif len(data) == 4:
                node_set, rhs, coefficients, node_alphas = data
                arc = None
            else:
                node_set, rhs, coefficients = data
                node_alphas = {}
                arc = None
            self.lci_cuts[node_set] = (rhs, coefficients, node_alphas)
            self.lci_arcs[node_set] = arc
        elif cut_type == "multistar":
            # data = (node_set, coefficients) where coefficients = {route_idx: -a_k}
            node_set, coefficients = data
            # Only keep/update if this is a new or more restrictive cut for the same S.
            self.multistar_cuts[node_set] = coefficients

    def _inject_multistar_cut(self, master: MasterProblemSupport) -> bool:
        """Inject multistar cuts into the master problem.

        Args:
            master: MasterProblem instance to receive the cuts.

        Returns:
            True if the cut was successfully applied, False otherwise.
        """
        # Re-inject Multistar cuts.  Coefficients are recomputed from current route pool
        # because route indices shift across B&B nodes (column deletion/addition).
        cut_added = False
        if hasattr(master, "add_multistar_cut") and self.multistar_cuts:
            for node_set, _stale_coeffs in self.multistar_cuts.items():
                S = set(node_set)
                Q = master.capacity  # type: ignore[attr-defined]
                new_coeffs: Dict[int, float] = {}
                for idx, route in enumerate(master.routes):  # type: ignore[union-attr]
                    path = [0] + route.nodes + [0]
                    k_cross = 0
                    k_adj = 0.0
                    k_visit = sum(master.wastes.get(n, 0.0) for n in route.nodes if n in S)  # type: ignore[attr-defined]
                    for p in range(len(path) - 1):
                        u, v = path[p], path[p + 1]
                        u_in, v_in = u in S, v in S
                        if u_in != v_in:
                            k_cross += 1
                            if u_in and v != 0 and not v_in:
                                k_adj += master.wastes.get(v, 0.0)  # type: ignore[attr-defined]
                            elif v_in and u != 0 and not u_in:
                                k_adj += master.wastes.get(u, 0.0)  # type: ignore[attr-defined]
                    a_k = k_cross - (2.0 / Q) * k_visit - (2.0 / Q) * k_adj
                    if abs(a_k) > 1e-6:
                        new_coeffs[idx] = -a_k
                if new_coeffs and master.add_multistar_cut(list(S), new_coeffs):
                    cut_added = True
        return cut_added

    def apply_to_master(self, master: MasterProblemSupport) -> int:
        """Inject all pooled global cuts into a fresh Master Problem instance.

        Typically called when entering a new B&B node to tighten the root relaxation.

        Args:
            master: MasterProblem instance to receive the cuts.

        Returns:
            Number of cuts successfully applied.
        """

        added = 0
        # RCC: replay with the stored (correct) RHS, not a hard-coded value.
        for node_set, rhs in self.rcc_cuts.items():
            if master.add_capacity_cut(list(node_set), rhs=rhs, _skip_pool=True):
                added += 1
        for nodes in self.sri_cuts:
            if master.add_subset_row_cut(nodes):
                added += 1
        for nodes in self.sec_cuts:
            # Form 2.1 is always global
            if master.add_sec_cut(nodes, rhs=1.0, facet_form="2.1"):
                added += 1
        # Re-inject Edge Clique cuts from the global pool.
        for edge_tuple in self.edge_clique_cuts:
            if master.add_edge_clique_cut(edge_tuple[0], edge_tuple[1]):
                added += 1

        # Re-inject LCI cuts — recompute route coefficients from node_alphas using the
        # current master's route list.  The stale coefficients stored at discovery time
        # reference route indices from the discovery B&B node; at a descendant node the
        # master may have a different route pool, so those indices are meaningless.
        for node_set, lci_data in self.lci_cuts.items():
            if len(lci_data) == 3:
                rhs, _stale_coefficients, node_alphas = lci_data
            else:
                rhs, _stale_coefficients = lci_data  # type: ignore[misc]
                node_alphas = {}
            # Retrieve the source arc stored at discovery time (None for node-capacity LCI).
            lci_arc: Optional[Tuple[int, int]] = self.lci_arcs.get(node_set)
            new_coefficients: Dict[int, float] = {}
            for idx, route in enumerate(master.routes):  # type: ignore[union-attr]
                alpha_k = sum(node_alphas.get(n, 1.0 if n in node_set else 0.0) for n in route.node_coverage if n != 0)
                if alpha_k > 1e-6:
                    new_coefficients[idx] = alpha_k
            if master.add_lci_cut(list(node_set), rhs, new_coefficients, node_alphas=node_alphas, arc=lci_arc):
                added += 1

        # Re-inject Multistar cuts.  Coefficients are recomputed from current route pool
        # because route indices shift across B&B nodes (column deletion/addition).
        added += 1 if self._inject_multistar_cut(master) else 0
        return added
