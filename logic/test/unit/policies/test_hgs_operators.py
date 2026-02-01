"""Unit tests for HGS local search operators."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from logic.src.policies.hgs_aux.operators.move_operators import move_relocate, move_swap
from logic.src.policies.hgs_aux.operators.route_operators import (
    move_2opt_intra,
    move_2opt_star,
    move_swap_star
)





def test_move_relocate_success(mock_ls):
    """Test successful relocation of node 3 to after node 1."""
    # Move node 3 (route 1, pos 0) to route 0 after node 1 (pos 0)
    # New R0: [1, 3, 2], New R1: [4]

    # Cost check:
    # Old: 1->2 (1.0), 3->4 (1.0)
    # New: 1->3, 3->2, 4->...
    # We cheat distances to make it favorable
    mock_ls.d[1, 3] = 0.1
    mock_ls.d[3, 2] = 0.1
    # Removes 1->2 (1.0), adds 1->3 (0.1) + 3->2 (0.1) -> Gain 0.8
    # Also removes 0->3 (5.0) + 3->4 (1.0), adds 0->4?
    # Relocate logic:
    # delta = -d[prev_u, u] - d[u, next_u] + d[prev_u, next_u] (Removal of u)
    # delta -= d[v, v_next] (Insertion point break)
    # delta += d[v, u] + d[u, v_next] (Insertion of u)

    # u=3, v=1.
    # Removal of 3 from [3, 4] (preceded by 0, followed by 4)
    # prev_u=0, next_u=4.
    # -d[0,3](5) - d[3,4](1) + d[0,4](5) = -1

    # Insertion after 1 in [1, 2]
    # v=1, v_next=2
    # -d[1,2](1)
    # +d[1,3](0.1) + d[3,2](0.1) = +0.2 - 1 = -0.8

    # Total delta = -1.8 < 0 -> Success

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
    # Current: 1->2->3->4
    # Cross: 1->3->2->4 (Reverse 2-3 segment)

    # 1->2(10), 2->3(10), 3->4(10) -> Total 30
    # 1->3(1), 3->2(1), 2->4(1) -> Total 3

    # Nodes
    # u=1 (pos 0), v=3 (pos 2). Segment is (u_next...v) -> 2,3 ?
    # 2opt reverses segment between u and v?
    # Code: segment = route[p_u + 1 : p_v + 1] -> route[1:3] = [2, 3]
    # Reversed: [3, 2]
    # New: 1, 3, 2, 4. Matches geometry.

    # Links handled:
    # -d[u, u_next] - d[v, v_next]
    # +d[u, v] + d[u_next, v_next]
    # u=1, u_next=2. v=3, v_next=4.
    # Remove 1-2, 3-4. Add 1-3, 2-4.

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
    # R0: [1, 2], R1: [3, 4]
    # u=1, v=3
    # New R0: [1, 4] (1 + tail of R1 from 3) -> 1 -> 4
    # New R1: [3, 2] (3 + tail of R0 from 1) -> 3 -> 2

    # Remove u->u_next (1->2), v->v_next (3->4)
    # Add u->v_next (1->4), v->u_next (3->2)

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
