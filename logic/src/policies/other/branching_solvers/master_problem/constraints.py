"""
Constraint addition and management mixin for VRPPMasterProblem.
"""

import hashlib
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp

if TYPE_CHECKING:
    from logic.src.policies.other.branching_solvers.common.route import Route
    from logic.src.policies.other.branching_solvers.master_problem.model import VRPPMasterProblem


class VRPPMasterProblemConstraintsMixin:
    """
    Mixin containing methods for adding cutting planes to the master problem.
    This helps reduce the size of the main model.py file.
    """

    def add_edge_clique_cut(
        self: "VRPPMasterProblem", u: int, v: int, coefficients: Optional[Dict[int, float]] = None, rhs: float = 1.0
    ) -> bool:
        """Add an Edge Clique cut specifically covering edge (u, v)."""
        if self.model is None or not self.lambda_vars:
            return False

        edge_tuple = (min(u, v), max(u, v))
        key = edge_tuple

        if key in self.active_edge_clique_cuts:
            return False

        lhs = gp.LinExpr()
        found_columns = False
        for idx, route in enumerate(self.routes):
            nodes = [0] + route.nodes + [0]
            contains_edge = False
            for i in range(len(nodes) - 1):
                if tuple(sorted((nodes[i], nodes[i + 1]))) == edge_tuple:
                    contains_edge = True
                    break

            if contains_edge:
                coeff = 1.0
                if coefficients is not None:
                    coeff = coefficients.get(idx, 1.0)
                lhs.add(self.lambda_vars[idx], coeff)
                found_columns = True

        if not found_columns:
            return False

        constr = self.model.addConstr(lhs <= rhs, name=f"Edge_Clique_{edge_tuple[0]}_{edge_tuple[1]}")
        stored_coeffs = coefficients if coefficients is not None else {}
        self.active_edge_clique_cuts[key] = (constr, stored_coeffs)
        self.global_cut_pool.add_cut("edge_clique", edge_tuple)
        self.model.update()
        return True

    def add_subset_row_cut(self: "VRPPMasterProblem", node_set: Union[List[int], Set[int], FrozenSet[int]]) -> bool:
        """Add a 3-Subset Row Inequality (3-SRI) to the master problem."""
        if self.model is None or len(node_set) != 3 or not self.lambda_vars:
            return False

        nodes = sorted(node_set)
        subset_frozenset = frozenset(nodes)
        if subset_frozenset in self.active_sri_cuts:
            return False

        cut_name = f"SRI_{nodes[0]}_{nodes[1]}_{nodes[2]}"
        lhs = gp.LinExpr()
        found_columns = False
        coeff_dict: Dict[str, float] = {}
        for idx, route in enumerate(self.routes):
            count = sum(1 for n in nodes if n in route.node_coverage)
            coeff = count // 2
            if coeff > 0:
                lhs.add(self.lambda_vars[idx], float(coeff))
                content = ",".join(map(str, route.nodes))
                route_h = hashlib.md5(content.encode()).hexdigest()
                coeff_dict[route_h] = float(coeff)
                found_columns = True

        if not found_columns:
            return False

        new_cut = self.model.addConstr(lhs <= 1.0, name=cut_name)
        self.active_sri_cuts[subset_frozenset] = new_cut
        self.global_cut_pool.add_cut("sri", (subset_frozenset, coeff_dict))
        self.model.update()
        return True

    def add_capacity_cut(
        self: "VRPPMasterProblem",
        node_list: List[int],
        rhs: float,
        coefficients: Optional[Dict[int, float]] = None,
        is_global: bool = True,
        _skip_pool: bool = False,
    ) -> bool:
        """Add a Rounded Capacity Cut (RCC) to the master problem."""
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_capacity_cuts:
            return False

        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"capacity_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_capacity_cuts[node_set] = constr
        if is_global and not _skip_pool:
            self.global_cut_pool.add_cut("rcc", (node_set, rhs))

        self.model.update()
        return True

    def add_lci_cut(
        self: "VRPPMasterProblem",
        node_list: List[int],
        rhs: float,
        coefficients: Dict[int, float],
        node_alphas: Optional[Dict[int, float]] = None,
        arc: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Add a Lifted Cover Inequality (LCI) to the master problem."""
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_lci_cuts:
            return False

        lhs = gp.LinExpr()
        for idx, coeff in coefficients.items():
            if idx < len(self.lambda_vars):
                lhs += coeff * self.lambda_vars[idx]

        if lhs.size() == 0:
            return False

        name = f"lci_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs <= rhs, name=name)
        self.active_lci_cuts[node_set] = constr
        effective_node_alphas: Dict[int, float] = node_alphas if node_alphas is not None else {}
        self.active_lci_node_alphas[node_set] = effective_node_alphas
        self.active_lci_arcs[node_set] = arc
        self.global_cut_pool.add_cut("lci", (node_set, rhs, coefficients, effective_node_alphas, arc))
        self.model.update()
        return True

    def add_sec_cut(
        self: "VRPPMasterProblem",
        node_list: Union[List[int], Set[int], FrozenSet[int]],
        rhs: float,
        cut_name: str = "",
        global_cut: bool = True,
        node_i: int = -1,
        node_j: int = -1,
        facet_form: str = "2.1",
    ) -> bool:
        """Add a Subtour Elimination Cut (SEC) or PC-SEC to the master problem."""
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        registry = self.active_sec_cuts if global_cut else self.active_sec_cuts_local

        if node_set in registry:
            return False

        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            val = float(self._count_crossings(route, node_set))
            if node_i > 0 and node_i in route.node_coverage:
                val -= 2.0
            if node_j > 0 and node_j in route.node_coverage:
                val -= 2.0

            if abs(val) > 1e-6:
                lhs += val * self.lambda_vars[idx]

        name = cut_name if cut_name else f"sec_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        registry[node_set] = constr

        if global_cut and node_i < 0 and node_j < 0:
            self.global_cut_pool.add_cut("sec_2.1", node_set)

        self.model.update()
        return True

    def _count_crossings(self: "VRPPMasterProblem", route: "Route", node_set: FrozenSet[int]) -> int:
        """Count how many times a route crosses the boundary δ(S)."""
        crossings = 0
        path_nodes = [0] + route.nodes + [0]
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if (u in node_set) != (v in node_set):
                crossings += 1
        return crossings
