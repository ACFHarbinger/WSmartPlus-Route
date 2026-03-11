"""Unit tests for HGS local search operators."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from logic.src.policies.other.operators import (
    move_relocate,
    move_swap,
    move_swap_star,
)
from logic.src.policies.other.operators import (
    move_2opt_intra,
    move_2opt_star,
)


def test_move_relocate_success(mock_ls):
    """Test successful relocation of node 3 to after node 1."""
    # Move node 3 (route 1, pos 0) to route 0 after node 1 (pos 0)
    # New R0: [1, 3, 2], New R1: [4]

    # Cost check:
    mock_ls.d[1, 3] = 0.1
    mock_ls.d[3, 2] = 0.1

    u, v = 3, 1
    r_u, p_u = 1, 0 # Route 1, index 0 is node 3
    r_v, p_v = 0, 0 # Route 0, index 0 is node 1

    res = move_relocate(mock_ls, u, v, r_u, p_u, r_v, p_v)
    assert res is True
    assert mock_ls.routes[0] == [1, 3, 2]
    assert mock_ls.routes[1] == [4]


def test_move_relocate_capacity_fail(mock_ls):
    """Test relocation failing due to capacity."""
    mock_ls.Q = 20.0
    # Routes loads are 20 each.
    # Moving 3 (dem=10) to Route 0 -> Load 30 > 20 -> Fail

    mock_ls.d[1, 3] = 0.1 # Make it cheap
    mock_ls.d[3, 2] = 0.1

    u, v = 3, 1
    r_u, p_u = 1, 0
    r_v, p_v = 0, 0

    res = move_relocate(mock_ls, u, v, r_u, p_u, r_v, p_v)
    assert res is False
    assert mock_ls.routes[0] == [1, 2] # Unchanged


def test_move_swap_success(mock_ls):
    """Test successful swap of node 2 and node 3."""
    # R0: [1, 2], R1: [3, 4]
    # Swap 2 and 3 -> R0: [1, 3], R1: [2, 4]

    # Distances for success
    # Remove 1->2(1), 2->0(5) -> Cost 6
    # Remove 0->3(5), 3->4(1) -> Cost 6
    # Add 1->3(0.1), 3->0(5) -> New 5.1
    # Add 0->2(5), 2->4(0.1) -> New 5.1

    mock_ls.d[1, 3] = 0.1
    mock_ls.d[2, 4] = 0.1

    u, v = 2, 3
    r_u, p_u = 0, 1 # Route 0, pos 1 is node 2
    r_v, p_v = 1, 0 # Route 1, pos 0 is node 3

    res = move_swap(mock_ls, u, v, r_u, p_u, r_v, p_v)
    assert res is True
    assert mock_ls.routes[0] == [1, 3]
    assert mock_ls.routes[1] == [2, 4]


def test_move_2opt_intra_success(mock_ls):
    """Test 2-opt intra-route optimization."""
    # Route 0: [1, 2, 3, 4] (Modified for this test)
    mock_ls.routes = [[1, 2, 3, 4]]

    mock_ls.d[1, 2] = 10
    mock_ls.d[3, 4] = 10
    mock_ls.d[1, 3] = 1
    mock_ls.d[2, 4] = 1

    u, v = 1, 3
    r_u, p_u, r_v, p_v = 0, 0, 0, 2

    res = move_2opt_intra(mock_ls, u, v, r_u, p_u, r_v, p_v)
    assert res is True
    assert mock_ls.routes[0] == [1, 3, 2, 4]


def test_move_2opt_star_success(mock_ls):
    """Test 2-opt star (concatenation of tails)."""
    mock_ls.d[1, 2] = 10
    mock_ls.d[3, 4] = 10
    mock_ls.d[1, 4] = 1
    mock_ls.d[3, 2] = 1

    u, v = 1, 3
    r_u, p_u = 0, 0
    r_v, p_v = 1, 0

    res = move_2opt_star(mock_ls, u, v, r_u, p_u, r_v, p_v)
    assert res is True
    assert mock_ls.routes[0] == [1, 4]
    assert mock_ls.routes[1] == [3, 2]
