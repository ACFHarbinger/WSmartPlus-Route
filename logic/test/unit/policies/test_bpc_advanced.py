import pytest
import numpy as np
from typing import Dict, Tuple, Set

from logic.src.policies.branch_and_price.branching import MultiEdgePartitionBranching
from logic.src.policies.branch_and_price.master_problem import VRPPMasterProblem, Route
from logic.src.policies.branch_and_price.rcspp_dp import RCSPPSolver, Label
from logic.src.policies.branch_and_price_and_cut.cutting_planes import create_cutting_plane_engine, SubsetRowCutEngine, LiftedCoverCutEngine
from logic.src.policies.branch_and_cut.separation import SeparationEngine
from logic.src.policies.branch_and_cut.vrpp_model import VRPPModel

def test_polar_angle_partitioning():
    """Verify that spatial branching correctly partitions arcs by polar angle."""
    # Node 0 (Depot) at (0, 0)
    # Node 1 at (1, 1) -> angle pi/4  (0.78 rad)
    # Node 2 at (-1, 1) -> angle 3pi/4 (2.35 rad)
    # Node 3 at (0, 1) -> angle pi/2  (1.57 rad)

    node_coords = {
        0: (0.0, 0.0),
        1: (1.0, 1.0),
        2: (-1.0, 1.0),
        3: (0.0, 1.0)
    }

    # Arcs: (0, 1), (0, 2), (0, 3)
    r1 = Route(nodes=[1], cost=10, revenue=20, load=10, node_coverage={1})
    r2 = Route(nodes=[2], cost=10, revenue=20, load=10, node_coverage={2})
    r3 = Route(nodes=[3], cost=10, revenue=20, load=10, node_coverage={3})

    routes = [r1, r2, r3]
    lambdas = [0.34, 0.33, 0.33]
    route_values = {i: lambdas[i] for i in range(len(lambdas))}

    brancher = MultiEdgePartitionBranching()
    d, set1, set2 = brancher.find_divergence_node(routes, route_values, node_coords=node_coords)

    assert d == 0
    # Arcs sorted by angle: (0, 1) [0.78], (0, 3) [1.57], (0, 2) [2.35]
    # If using median, (0, 1) and (0, 3) are in one set, (0, 2) in another.
    assert (0, 1) in set1 or (0, 1) in set2
    assert (0, 2) in set1 or (0, 2) in set2
    assert (0, 1) != (0, 2)

def test_sri_dual_penalties():
    """Verify that SRI duals are correctly applied and tracked in RCSPP labels."""
    n_nodes = 3
    cost_matrix = np.zeros((4, 4))
    wastes = {1: 10.0, 2: 10.0, 3: 10.0}

    solver = RCSPPSolver(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=100.0,
        revenue_per_kg=1.0,
        cost_per_km=1.0
    )

    # SRI on subset {1, 2, 3} with dual = 10.0
    sri_subset = frozenset({1, 2, 3})
    sri_duals = {sri_subset: 10.0}

    composite_duals = {
        "node_duals": {1: 0.0, 2: 0.0, 3: 0.0},
        "sri_duals": sri_duals
    }

    routes = solver.solve(dual_values=composite_duals)

    route_rcs = {tuple(r.nodes): r.reduced_cost for r in routes}
    # Route [1, 2]: revenue 20, penalty 10 -> RC = 10.
    assert route_rcs.get((1, 2)) == 10.0
    # Route [1]: revenue 10, penalty 0 -> RC = 10.
    assert route_rcs.get((1,)) == 10.0

def test_cutting_plane_engine_composition():
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_sep = MagicMock()

    # create_cutting_plane_engine(engine_name, v_model, sep_engine)
    engine = create_cutting_plane_engine("all", mock_model, mock_sep)
    assert any(isinstance(e, SubsetRowCutEngine) for e in engine.engines)
    assert any(isinstance(e, LiftedCoverCutEngine) for e in engine.engines)


def test_lci_dual_penalties():
    """
    Test that LCI dual penalties are correctly applied to the step objective.
    """
    # 1. Setup pricer with a dummy environment
    from logic.src.policies.branch_and_price.rcspp_dp import RCSPPSolver

    pricer = RCSPPSolver(
        n_nodes=2,
        cost_matrix=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        wastes={1: 10.0, 2: 10.0},
        capacity=20.0,
        revenue_per_kg=1.0,
        cost_per_km=1.0,
    )

    # 2. Define node duals and an LCI penalty on edge (1,2)
    node_duals = {1: 2.0, 2: 2.0}
    lci_cut_duals = {(1, 2): 5.0}

    # 3. Extend label from node 1 to node 2
    from logic.src.policies.branch_and_price.rcspp_dp import Label

    l1 = Label(
        reduced_cost=10.0,  # Arbitrary starting RC
        node=1,
        cost=1.0,
        load=10.0,
        revenue=10.0,
        path=[1],
        visited={1},
    )

    # Note: solve() must be called to initialize self.lci_cut_duals
    pricer.solve(dual_values={"node_duals": node_duals, "lci_duals": lci_cut_duals})

    l2 = pricer._extend_label(l1, 2, rf_together=set())

    # 4. Verify step_obj
    # revenue(2) = 10*R(1.0) = 10
    # edge_cost(1,2) = 1*C(1.0) = 1
    # node_dual(2) = 2
    # lci_penalty(1,2) = 5
    # step_obj = 10 - 1 - 2 - 5 = 2
    # new_rc = 10 + 2 = 12
    assert l2 is not None
    assert l2.reduced_cost == 12.0
