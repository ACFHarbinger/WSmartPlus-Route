from unittest.mock import patch

import numpy as np
import pandas as pd

from logic.src.policies.look_ahead_aux.check import (
    check_bins_overflowing_feasibility,
    check_solution_admissibility,
)
from logic.src.policies.look_ahead_aux.move import (
    move_1_route,
    move_2_routes,
    move_n_2_routes_consecutive,
    move_n_2_routes_random,
    move_n_route_consecutive,
    move_n_route_random,
)
from logic.src.policies.look_ahead_aux.select import (
    add_bin,
    add_n_bins_random,
    add_route_random,
    remove_bin,
    remove_n_bins_random,
)
from logic.src.policies.look_ahead_aux.solutions import find_initial_solution, find_solutions
from logic.src.policies.look_ahead_aux.swap import (
    swap_1_route,
    swap_2_routes,
    swap_n_2_routes_consecutive,
    swap_n_2_routes_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)


class TestPoliciesAuxCheck:
    def test_check_bins_overflowing_feasibility(self):
        # Setup data
        # E * B = capacity. Let E=1.0, B=100.0 => Cap=100.0
        # Bin 1: Stock=50, Rate=10 => 60 < 100 (Safe)
        # Bin 2: Stock=95, Rate=10 => 105 >= 100 (Overflow)
        data = pd.DataFrame({"#bin": [1, 2], "Stock": [50.0, 95.0], "Accum_Rate": [10.0, 10.0]})
        routes_list = [[0, 1, 0], [0, 2, 0]]
        # Overflowing bins: [2]
        # Bins in routes: 1 and 2.
        # intersecting: 2 is in routes.
        # total_ovf_bins_in_routes = 1

        # Check tolerance
        # number_of_bins = 2
        # perc = 0.0 -> check = 1 - 0 = 1.
        # total_ovf (1) >= check (1) -> pass
        res = check_bins_overflowing_feasibility(
            data, routes_list, number_of_bins=2, perc_bins_can_overflow=0.0, E=1.0, B=100.0
        )
        assert res == "pass"

        # Case: Overflowing bin NOT in routes
        # Remove bin 2 from routes
        routes_list_safe = [[0, 1, 0]]
        # total_ovf_in_routes = 0.
        # check = 1.
        # 0 >= 1 -> False -> fail
        res_fail = check_bins_overflowing_feasibility(
            data, routes_list_safe, number_of_bins=2, perc_bins_can_overflow=0.0, E=1.0, B=100.0
        )
        assert res_fail == "fail"

    def test_check_solution_admissibility(self):
        # routes: [0, 1, 0], [0, 2, 0]. Lengths: 3, 3. Total=6.
        # removed: []. Length=0.
        # number_of_bins = 2.
        # Equation: (6 + 0) == 2 + 2*2 => 6 == 6. True.
        routes = [[0, 1, 0], [0, 2, 0]]
        removed = []
        assert check_solution_admissibility(routes, removed, 2)

        # Inconsistent
        assert not check_solution_admissibility(routes, removed, 3)


class TestPoliciesAuxSelect:
    def test_remove_bin(self, policies_routes_setup):
        # Remove a random bin.
        # Route 1: [0, 1, 2, 3, 0]. can remove 1 or 3 (2 is mandatory).
        # Route 2: [0, 4, 5, 0]. can remove 4 or 5.

        # Just run without patch, verify structural changes

        # Run multiple times to hit different branches?
        # remove_bin modifies lists in-place.
        routes = [r[:] for r in policies_routes_setup]
        removed = []
        cannot = []

        # Should remove something.
        remove_bin(routes, removed, cannot)

        assert len(removed) == 1
        assert len(routes[0]) + len(routes[1]) == 23  # 12+12-1=23

    def test_add_bin(self):
        r_list = [[0, 1, 0]]
        removed = [2]

        add_bin(r_list, removed)

        assert len(removed) == 0
        assert len(r_list[0]) == 4  # 0, 1, 2, 0 (order varies)
        assert 2 in r_list[0]

    def test_remove_n_bins_random(self):
        # remove 2 bins
        r_list = [[0, 1, 2, 3, 4, 5, 0]]
        removed = []
        cannot = []

        # Mock rsample to return n=2
        # It gets called multiple times.
        # 1. chosen_n = rsample([2,3,4,5], 1)[0]
        # 2. chosen_route = rsample(routes_list, 1)[0]
        # 3. bin_to_rem_1 = rsample(...)
        # 4. bin_to_rem_2 = rsample(...)

        with patch("logic.src.policies.look_ahead_aux.select.rsample") as mock_rsample:

            def side_eff(pop, k):
                # Picking n
                if pop == [2, 3, 4, 5]:
                    return [2]
                # Picking route
                if len(pop) == 1 and pop[0] == r_list[0]:
                    return [pop[0]]
                # Picking bins - just pick first available
                if len(pop) > 0:
                    return [pop[0]]
                return []

            mock_rsample.side_effect = side_eff

            res_bins, chosen_n = remove_n_bins_random(r_list, removed, cannot)

        # If n=2, removes 2.
        assert chosen_n == 2
        assert len(res_bins) == 2
        assert len(removed) == 2
        # 7 items start, removes 2 -> 5 items
        assert len(r_list[0]) == 5

    def test_add_n_bins_random(self):
        r_list = [[0, 1, 0]]
        removed = [2, 3, 4, 5, 6]

        res_bins, n = add_n_bins_random(r_list, removed)

        assert len(res_bins) == n
        assert len(r_list[0]) > 3

    @patch("logic.src.policies.look_ahead_aux.select.organize_route")
    def test_add_route_random(self, mock_organize):
        mock_organize.return_value = [0, 99, 0]

        r_list = [[0, 1, 2, 3, 4, 5, 0]]
        dist_mat = np.zeros((10, 10))

        # It picks a subset from existing route to make a new route
        add_route_random(r_list, dist_mat)

        assert len(r_list) == 2  # Added a new route
        assert r_list[-1] == [0, 99, 0]
        assert len(r_list[0]) < 7  # Removed from original


class TestPoliciesAuxMove:
    def test_move_1_route(self, policies_routes_setup):
        routes = [r[:] for r in policies_routes_setup]
        move_1_route(routes)
        # Length shouldn't change, but order might
        assert len(routes[0]) == 12

    def test_move_2_routes(self, policies_routes_setup):
        # Moves boolean from one route to another
        # r1 len 12, r2 len 12
        routes = [r[:] for r in policies_routes_setup]
        move_2_routes(routes)
        lens = sorted([len(routes[0]), len(routes[1])])
        assert lens == [11, 13]

    def test_move_n_route_random(self, policies_routes_setup):
        # We need to test behavior for n=2, 3, 4, 5.
        # rsample is used to pick chosen_n from [2,3,4,5].
        # Then rsample pick bins.

        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            with patch("logic.src.policies.look_ahead_aux.move.rsample") as mock_rs:
                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff

                move_n_route_random(routes)
                # Verify length is conserved (intra-route move)
                assert len(routes[0]) == 12

    def test_move_n_route_consecutive(self, policies_routes_setup):
        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            with patch("logic.src.policies.look_ahead_aux.move.rsample") as mock_rs:
                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff

                move_n_route_consecutive(routes)
                assert len(routes[0]) == 12

    def test_move_n_2_routes_random(self, policies_routes_setup):
        # moves n bins from one route to another
        # r1 -> r2 or r2 -> r1
        # if n=2, lens become 10, 14
        from itertools import cycle

        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            with patch("logic.src.policies.look_ahead_aux.move.rsample") as mock_rs:
                # We need to cycle routes to avoid infinite loop in while chosen_2 == chosen_1
                route_iter = cycle([routes[0], routes[1]])

                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    # Check if picking from routes (list of lists)
                    if len(pop) == 2 and isinstance(pop[0], list):
                        return [next(route_iter)]
                    # Picking bins
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff

                move_n_2_routes_random(routes)
                l1 = len(routes[0])
                l2 = len(routes[1])
                diff = abs(l1 - l2)
                # Should correspond to n moved from one to other (equal size start)
                assert diff == 2 * n

    def test_move_n_2_routes_consecutive(self, policies_routes_setup):
        from itertools import cycle

        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            with patch("logic.src.policies.look_ahead_aux.move.rsample") as mock_rs:
                route_iter = cycle([routes[0], routes[1]])

                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    if len(pop) == 2 and isinstance(pop[0], list):
                        return [next(route_iter)]
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff

                move_n_2_routes_consecutive(routes)
                l1 = len(routes[0])
                l2 = len(routes[1])
                diff = abs(l1 - l2)
                assert diff == 2 * n


class TestPoliciesAuxSwap:
    def test_swap_1_route(self, policies_routes_setup):
        routes = [r[:] for r in policies_routes_setup]
        old = list(routes[0])
        # Ensure rsample picks r1
        with patch("logic.src.policies.look_ahead_aux.swap.rsample") as mock_rs:
            # side_effect: choose route, then choose 2 bins positions?
            # logic is: choose route, choose bin1, choose bin2
            # rsample(list, 1) returns [elt]
            mock_rs.side_effect = [[routes[0]], [1], [2]]
            swap_1_route(routes)
        assert routes[0] != old
        assert set(routes[0]) == set(old)

    def test_swap_2_routes(self, policies_routes_setup):
        # Swaps one bin between routes
        # Lens stay same
        routes = [r[:] for r in policies_routes_setup]
        old_set = set(routes[0] + routes[1])
        swap_2_routes(routes)
        assert len(routes[0]) == 12
        assert len(routes[1]) == 12
        new_set = set(routes[0] + routes[1])
        assert old_set == new_set

    def test_swap_n_route_random(self, policies_routes_setup):
        # Helper to force chosen_n
        def run_with_n(n):
            routes = [r[:] for r in policies_routes_setup]
            # patch rsample
            # It calls rsample(possible_n, 1) -> [n]
            # Then rsample(routes, 1) -> [r1]
            # Then many rsamples for bins.
            # We can just let bins be random, but force n.
            with patch("logic.src.policies.look_ahead_aux.swap.rsample") as mock_rs:
                # We need to passthrough for most calls but capture the first one?
                # Hard with side_effect if we don't know sequence.
                # Custom side effect
                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    if len(pop) == 2 and isinstance(pop[0], list):  # choosing between routes
                        return [pop[0]]
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff
                swap_n_route_random(routes)

            assert len(routes[0]) == 12

        for n in [2, 3, 4, 5]:
            run_with_n(n)

    def test_swap_n_route_consecutive(self, policies_routes_setup):
        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            # This one picks n=random.
            # We need to mock rsample again.
            with patch("logic.src.policies.look_ahead_aux.swap.rsample") as mock_rs:
                # Logic involves picking route, then n, then start bin
                # We force n.
                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    # standard passthrough ish
                    if len(pop) == 2 and hasattr(pop[0], "__iter__"):
                        return [pop[0]]
                    # picking bins
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff
                swap_n_route_consecutive(routes)
            assert len(routes[0]) == 12

    def test_swap_n_2_routes_random(self, policies_routes_setup):
        from itertools import cycle

        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            with patch("logic.src.policies.look_ahead_aux.swap.rsample") as mock_rs:
                route_iter = cycle([routes[0], routes[1]])
                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    if len(pop) == 2 and isinstance(pop[0], list):
                        return [next(route_iter)]
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff

                swap_n_2_routes_random(routes)
            # lens stay same
            assert len(routes[0]) == 12
            assert len(routes[1]) == 12

    def test_swap_n_2_routes_consecutive(self, policies_routes_setup):
        from itertools import cycle

        for n in [2, 3, 4, 5]:
            routes = [r[:] for r in policies_routes_setup]
            with patch("logic.src.policies.look_ahead_aux.swap.rsample") as mock_rs:
                route_iter = cycle([routes[0], routes[1]])
                import random

                def side_eff(pop, k):
                    if pop == [2, 3, 4, 5]:
                        return [n]
                    if len(pop) == 2 and isinstance(pop[0], list):
                        return [next(route_iter)]
                    return random.sample(pop, k)

                mock_rs.side_effect = side_eff

                swap_n_2_routes_consecutive(routes)
            assert len(routes[0]) == 12
            assert len(routes[1]) == 12


class TestSolutions:
    def test_find_initial_solution(self, policies_vpp_data, policies_bins_coords, policies_dist_matrix):
        # E=1, B=1. Stock+Accum = 60. Cap=100.
        # Can pick 1 bin per route? or 1. something
        # 1 bin = 60. 2 bins = 120 > 100.
        # So each route will have 1 bin.
        # 3 bins total -> 3 routes.

        routes = find_initial_solution(
            policies_vpp_data,
            policies_bins_coords,
            policies_dist_matrix,
            number_of_bins=3,
            vehicle_capacity=100,
            E=1.0,
            B=1.0,
        )

        # Expect list of routes.
        # logic: while bins != 0...
        assert isinstance(routes, list)
        assert len(routes) == 3  # Should be 3 routes if cap allows only 1
        # Check content
        # routes have depot 0 at start? Logic says "globals()[...].append(depot[0])" at start?
        # No, "bin_chosen_n = depot[0]" then append.
        # Then appends depot at end.
        pass

    @patch("logic.src.policies.look_ahead_aux.solutions.local_search")
    @patch("logic.src.policies.look_ahead_aux.solutions.uncross_arcs_in_routes")
    @patch("logic.src.policies.look_ahead_aux.solutions.local_search_2")
    @patch("logic.src.policies.look_ahead_aux.solutions.local_search_reversed")
    @patch("logic.src.policies.look_ahead_aux.solutions.remove_bins_end")
    @patch("logic.src.policies.look_ahead_aux.solutions.insert_bins")
    @patch("logic.src.policies.look_ahead_aux.solutions.compute_real_profit")
    @patch("logic.src.policies.look_ahead_aux.solutions.compute_profit")
    def test_find_solutions(
        self,
        mock_profit,
        mock_real,
        mock_insert,
        mock_remove,
        mock_ls_rev,
        mock_ls2,
        mock_uncross,
        mock_ls,
        policies_vpp_data,
        policies_bins_coords,
        policies_dist_matrix,
        policies_solution_values,
    ):
        # Mock returns
        mock_uncross.return_value = ([[0, 1, 0]], [], [])
        mock_ls.return_value = (
            "drop",
            0,
            None,
            None,
            [],
            [],
            [],
            [],
            [],
            [],  # procedure, etc
        )
        mock_ls2.return_value = ([[0, 1, 0]], [], [])
        mock_ls_rev.return_value = ([[0, 1, 0]], [], [])
        mock_remove.return_value = ([[0, 1, 0]], [], [], [])
        mock_insert.return_value = ([[0, 1, 0]], [], [], [])

        chosen_comb = (
            1,  # iterations
            100,  # T_initial
            1,  # T_param
            1,  # p_vehicle
            1,  # p_load
            1,  # p_route_diff
            1,  # p_shift
        )

        res = find_solutions(
            policies_vpp_data,
            policies_bins_coords,
            policies_dist_matrix,
            chosen_comb,
            must_go_bins=[],
            values=policies_solution_values,
            n_bins=3,
            points={},
            time_limit=10.0,
        )

        # Returns (routes, profit, removed)
        assert isinstance(res, tuple)
        assert isinstance(res[0], list)
