import numpy as np
from logic.src.policies.helpers.branching_solvers import MultiEdgePartitionBranching
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.cutting_planes import (
    EdgeCliqueCutEngine,
    SubsetRowCutEngine,
    create_cutting_plane_engine,
    CompositeCuttingPlaneEngine,
)
from typing import cast
from logic.src.policies.helpers.branching_solvers import Route
from logic.src.policies.helpers.branching_solvers import RCSPPSolver


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
    res = brancher.find_divergence_node(routes, route_values, node_coords=cast(np.ndarray, node_coords))
    assert res is not None
    d, set1, set2, strength = res

    assert d == 0
    # Arcs sorted by angle: (0, 1) [0.78], (0, 3) [1.57], (0, 2) [2.35]
    # If using median, (0, 1) and (0, 3) are in one set, (0, 2) in another.
    assert (0, 1) in set1 or (0, 1) in set2
    assert (0, 2) in set1 or (0, 2) in set2
    assert (0, 1) != (0, 2)

def test_sri_dual_penalties():
    """Verify that SRI duals are correctly applied and tracked in RCSPP labels.

    SRI {1,2,3} with dual=5 should:
    - Route [1,2]:   floor(2/2)=1 penalty  → RC = 20 - 5  = 15
    - Route [1,2,3]: floor(3/2)=1 penalty  → RC = 30 - 5  = 25
    - Route [1]:     floor(1/2)=0 penalty  → RC = 10      = 10
    Route [1] has lower RC than 2-node routes so won't appear in a small
    max_routes cap.  We request max_routes=50 to capture all routes.
    """
    n_nodes = 3
    cost_matrix = np.zeros((4, 4))
    wastes = {1: 10.0, 2: 10.0, 3: 10.0}

    solver = RCSPPSolver(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=100.0,
        revenue_per_kg=1.0,
        cost_per_km=1.0,
    )

    # SRI on subset {1, 2, 3} with dual = 5.0
    sri_subset = frozenset({1, 2, 3})
    composite_duals = {
        "node_duals": {1: 0.0, 2: 0.0, 3: 0.0},
        "rcc_duals": {},
        "sri_duals": {sri_subset: 5.0},
        "edge_clique_duals": {},
    }

    # Request enough routes so that dominated single-node routes are included.
    routes = solver.solve(dual_values=composite_duals, max_routes=50)
    route_rcs = {tuple(r.nodes): r.reduced_cost for r in routes}

    # 2-node route visits 2 of 3 nodes in S: floor(2/2)=1 → penalty 5.
    assert (1, 2) in route_rcs, "Route [1,2] should be generated"
    rc_12 = route_rcs[(1, 2)]
    assert rc_12 is not None
    assert abs(rc_12 - 15.0) < 1e-6, f"Route [1,2] RC should be 15, got {rc_12}"

    # Single-node route visits 1 node in S: floor(1/2)=0 → no penalty.
    assert (1,) in route_rcs, "Route [1] should be generated (max_routes=50 captures it)"
    rc_1 = route_rcs[(1,)]
    assert rc_1 is not None
    assert abs(rc_1 - 10.0) < 1e-6, f"Route [1] RC should be 10, got {rc_1}"

def test_cutting_plane_engine_composition():
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_sep = MagicMock()

    # create_cutting_plane_engine(engine_name, v_model, sep_engine)
    engine = create_cutting_plane_engine("all", mock_model, mock_sep)
    assert isinstance(engine, CompositeCuttingPlaneEngine)
    assert any(isinstance(e, SubsetRowCutEngine) for e in engine.engines)
    assert any(isinstance(e, EdgeCliqueCutEngine) for e in engine.engines)


def test_edge_clique_dual_penalties():
    """
    Verify that edge-clique duals are propagated through solve() and reduce the
    reduced cost of routes that traverse a penalised edge.

    Setup
    -----
    n_nodes = 2, zero-distance matrix (travel is free), unit revenue per kg.
    Edge-clique dual = 5.0 on edge (1, 2).

    Expected reduced costs
    ----------------------
    Route [1]:     rev=10, cost=0, no edge-clique penalty  → RC = 10
    Route [2]:     rev=10, cost=0, no edge-clique penalty  → RC = 10
    Route [1, 2]:  rev=20, cost=0, edge (1,2) penalty = 5 → RC = 15
    Route [2, 1]:  rev=20, cost=0, edge (2,1) ≡ (1,2) penalty = 5 → RC = 15

    Without the fix (edge_clique_duals never reaching _extend_label), route
    [1, 2] would show RC = 20 instead of 15, and the cut would be ignored.
    """
    from logic.src.policies.helpers.branching_solvers import RCSPPSolver

    # Zero cost matrix so travel is free; revenue dominates.
    cost_matrix = np.zeros((3, 3))
    pricer = RCSPPSolver(
        n_nodes=2,
        cost_matrix=cost_matrix,
        wastes={1: 10.0, 2: 10.0},
        capacity=30.0,
        revenue_per_kg=1.0,
        cost_per_km=1.0,
    )

    # Edge-clique dual on the canonical edge (1, 2) — should reduce RC by 5
    # for any route that traverses 1→2 or 2→1.
    dual_values = {
        "node_duals": {1: 0.0, 2: 0.0},
        "rcc_duals": {},
        "sri_duals": {},
        "edge_clique_duals": {(1, 2): 5.0},
    }

    routes = pricer.solve(dual_values=dual_values, max_routes=20)
    route_rcs = {tuple(r.nodes): r.reduced_cost for r in routes}

    # Single-node routes must NOT be penalised (they don't cross edge (1,2)).
    assert (1,) in route_rcs, "Route [1] should be generated"
    assert (2,) in route_rcs, "Route [2] should be generated"
    rc1 = route_rcs[(1,)]
    rc2 = route_rcs[(2,)]
    assert rc1 is not None and rc2 is not None
    assert abs(rc1 - 10.0) < 1e-6, f"Route [1] RC should be 10, got {rc1}"
    assert abs(rc2 - 10.0) < 1e-6, f"Route [2] RC should be 10, got {rc2}"

    # Two-node route traverses edge (1,2) once → penalty of 5.
    if (1, 2) in route_rcs:
        rc12 = route_rcs[(1, 2)]
        assert rc12 is not None
        assert abs(rc12 - 15.0) < 1e-6, (
            f"Route [1,2] RC should be 15 (20 revenue - 5 edge-clique penalty), "
            f"got {rc12}"
        )
