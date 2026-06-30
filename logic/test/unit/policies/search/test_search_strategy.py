from unittest.mock import MagicMock

import pytest
from logic.src.policies.helpers.solvers_and_matheuristics.search.search_strategy import (
    BestFirstSearch,
    DepthFirstSearch,
    HybridSearchStrategy,
    create_search_strategy,
)


def test_best_first_search() -> None:
    strategy = BestFirstSearch()
    assert strategy.get_name() == "best_first"

    # Test empty list
    assert strategy.select_node([]) is None

    # Test selection based on lp_bound
    node1 = MagicMock()
    node1.lp_bound = 10.0
    node2 = MagicMock()
    node2.lp_bound = 25.0
    node3 = MagicMock()
    node3.lp_bound = 15.0

    open_nodes = [node1, node2, node3]
    selected = strategy.select_node(open_nodes)
    assert selected == node2
    assert len(open_nodes) == 2
    assert node2 not in open_nodes

    # Test lp_bound is None (treated as -inf)
    node4 = MagicMock()
    node4.lp_bound = None
    node5 = MagicMock()
    node5.lp_bound = -5.0
    open_nodes = [node4, node5]
    selected = strategy.select_node(open_nodes)
    assert selected == node5

def test_depth_first_search() -> None:
    strategy = DepthFirstSearch()
    assert strategy.get_name() == "depth_first"

    # Test empty list
    assert strategy.select_node([]) is None

    # Test LIFO selection
    node1 = MagicMock()
    node2 = MagicMock()
    node3 = MagicMock()

    open_nodes = [node1, node2, node3]
    selected = strategy.select_node(open_nodes)
    assert selected == node3
    assert len(open_nodes) == 2
    assert node3 not in open_nodes

def test_hybrid_search_strategy() -> None:
    mock_tree = MagicMock()
    mock_tree.best_integer_solution = None

    strategy = HybridSearchStrategy(mock_tree)
    assert strategy.get_name() == "hybrid"

    # Test empty list
    assert strategy.select_node([]) is None

    node1 = MagicMock()
    node1.lp_bound = 10.0
    node2 = MagicMock()
    node2.lp_bound = 25.0
    node3 = MagicMock()
    node3.lp_bound = 15.0

    # Phase 1: No integer incumbent -> should act like DFS (LIFO)
    open_nodes = [node1, node2, node3]
    selected = strategy.select_node(open_nodes)
    assert selected == node3

    # Phase 2: Incumbent exists -> should act like BestFS (BFS)
    mock_tree.best_integer_solution = MagicMock()
    open_nodes = [node1, node2, node3]
    selected = strategy.select_node(open_nodes)
    assert selected == node2

def test_create_search_strategy() -> None:
    # Test valid creations
    strategy = create_search_strategy("best_first")
    assert isinstance(strategy, BestFirstSearch)

    strategy = create_search_strategy("depth_first")
    assert isinstance(strategy, DepthFirstSearch)

    mock_tree = MagicMock()
    strategy = create_search_strategy("hybrid", mock_tree)
    assert isinstance(strategy, HybridSearchStrategy)
    assert strategy.bb_tree == mock_tree

    # Test missing tree for hybrid
    with pytest.raises(ValueError, match="Hybrid search strategy requires a BranchAndBoundTree reference"):
        create_search_strategy("hybrid")

    # Test unknown strategy
    with pytest.raises(ValueError, match="Unknown search strategy"):
        create_search_strategy("unknown_strategy")
