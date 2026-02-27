"""
Soccer League Competition (SLC) algorithm for VRPP.

Models a population of routing solutions as players organised into
hierarchical teams.  The globally best solution is the "superstar".

Dual competition structure:
  1. Intra-team: Each player locally perturbs its own solution; replaces
     itself if improved.
  2. Inter-team: Teams play probabilistic matches; the loser's weakest player
     adopts structure from the winner's best player via OX-inspired recombination.

Stagnation detection: if a team's aggregate profit does not improve for
`stagnation_limit` seasons, the entire team is regenerated from scratch.

Reference:
    Survey §"Soccer League Competition" — dual-tier competition framework.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_insertion
from .params import SLCParams


class SLCSolver(PolicyVizMixin):
    """
    Soccer League Competition solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SLCParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the SLC algorithm.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initialise teams: list of lists of (routes, profit)
        teams: List[List[Tuple[List[List[int]], float]]] = [self._new_team() for _ in range(self.params.n_teams)]
        stagnation: List[int] = [0] * self.params.n_teams
        team_best: List[float] = [max(p for _, p in team) for team in teams]

        # Global best (superstar)
        best_routes, best_profit = self._league_best(teams)
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # --- Intra-team competition: local perturbation ---
            for _t_idx, team in enumerate(teams):
                for p_idx in range(len(team)):
                    routes, profit = team[p_idx]
                    new_routes = self._perturb(routes)
                    new_profit = self._evaluate(new_routes)
                    if new_profit > profit:
                        team[p_idx] = (new_routes, new_profit)

            # --- Inter-team competition: probabilistic match ---
            team_indices = list(range(self.params.n_teams))
            random.shuffle(team_indices)
            for k in range(0, len(team_indices) - 1, 2):
                a_idx = team_indices[k]
                b_idx = team_indices[k + 1]

                fit_a: float = float(sum(p for _, p in teams[a_idx]))
                fit_b: float = float(sum(p for _, p in teams[b_idx]))

                total = abs(fit_a) + abs(fit_b) + 1e-9
                p_win_a = (fit_a - min(fit_a, fit_b) + 1e-9) / total

                if random.random() < p_win_a:
                    winner, loser = a_idx, b_idx
                else:
                    winner, loser = b_idx, a_idx

                # Weakest player in losing team adopts structure from winner's best
                winner_best_routes = max(teams[winner], key=lambda x: x[1])[0]
                loser_worst_idx = int(np.argmin([p for _, p in teams[loser]]))
                child = self._recombine(teams[loser][loser_worst_idx][0], winner_best_routes)
                child_profit = self._evaluate(child)
                teams[loser][loser_worst_idx] = (child, child_profit)

            # --- Stagnation check and team regeneration ---
            for t_idx, team in enumerate(teams):
                current_best = max(p for _, p in team)
                if current_best > team_best[t_idx] + 1e-9:
                    team_best[t_idx] = current_best
                    stagnation[t_idx] = 0
                else:
                    stagnation[t_idx] += 1
                    if stagnation[t_idx] >= self.params.stagnation_limit:
                        teams[t_idx] = self._new_team()
                        stagnation[t_idx] = 0
                        team_best[t_idx] = max(p for _, p in teams[t_idx])

            # Update superstar
            iter_best_routes, iter_best_profit = self._league_best(teams)
            if iter_best_profit > best_profit:
                best_routes = copy.deepcopy(iter_best_routes)
                best_profit = iter_best_profit
                best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_teams=self.params.n_teams,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_team(self) -> List[Tuple[List[List[int]], float]]:
        """Create a fresh team of `team_size` random players."""
        team = []
        for _ in range(self.params.team_size):
            routes = self._random_solution()
            profit = self._evaluate(routes)
            team.append((routes, profit))
        return team

    def _random_solution(self) -> List[List[int]]:
        """Generate a random feasible routing solution."""
        shuffled = random.sample(self.nodes, len(self.nodes))
        return greedy_insertion(
            [],
            shuffled,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Intra-team perturbation: remove one node and reinsert greedily.

        Args:
            routes: Current player routes.

        Returns:
            Perturbed routes.
        """
        n = max(1, self.params.n_removal)
        try:
            partial, removed = random_removal(routes, n)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        except Exception:
            return copy.deepcopy(routes)

    def _recombine(self, loser_routes: List[List[int]], winner_routes: List[List[int]]) -> List[List[int]]:
        """
        Recombine loser's route with winner's best route via OX-inspired injection.

        Extracts a random segment from the winner's flat tour and inserts
        unvisited nodes from that segment into the loser's solution.

        Args:
            loser_routes: Losing player's route set.
            winner_routes: Winner's best player route set.

        Returns:
            New child routing solution.
        """
        winner_flat = [n for r in winner_routes for n in r]
        loser_visited = {n for r in loser_routes for n in r}

        if len(winner_flat) < 2:
            return copy.deepcopy(loser_routes)

        # Extract a random segment from the winner
        a = random.randint(0, len(winner_flat) - 1)
        b = random.randint(a, min(a + max(1, len(winner_flat) // 3), len(winner_flat)))
        segment = winner_flat[a:b]
        new_nodes = [n for n in segment if n not in loser_visited]

        if not new_nodes:
            return self._perturb(loser_routes)

        child = copy.deepcopy(loser_routes)
        with contextlib.suppress(Exception):
            child = greedy_insertion(
                child,
                new_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        return child

    def _league_best(self, teams: List[List[Tuple[List[List[int]], float]]]) -> Tuple[List[List[int]], float]:
        """Return the best (routes, profit) across all teams."""
        best_p = -float("inf")
        best_r: List[List[int]] = []
        for team in teams:
            for routes, profit in team:
                if profit > best_p:
                    best_p = profit
                    best_r = routes
        return copy.deepcopy(best_r), best_p

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
