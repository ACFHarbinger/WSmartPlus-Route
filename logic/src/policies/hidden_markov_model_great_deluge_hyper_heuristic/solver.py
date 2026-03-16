"""
Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH) for VRPP.

This online-learning approach treats the sequence of applied Low-Level
Heuristics (LLHs) as a Markov chain. The system maintains a *belief*
distribution over three hidden states — improving, stagnating, escaping —
using the Forward Algorithm, rather than assigning a single deterministic
state.

At each iteration the belief vector P(S_t | O_{1:t}) is updated via the
observation likelihood of the normalised profit change Δ_norm.  The LLH
selection probabilities are then the belief-weighted mixture of the
per-state emission matrices.

The Great Deluge acceptance criterion accepts candidate solutions whose
profit exceeds a linearly rising water level, providing a deterministic
escape from local optima without requiring temperature tuning.

Reference:
    Onsem, W. V., Demoen, B., & Causmaecker, P. D. "HHaaHMM:
    A Hyper-Heuristic as a Hidden Markov Model", 2014
    McMullan, P. "An Extended Implementation of the Great
    Deluge Algorithm for Course Timetabling", 2007
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import (
    cluster_removal,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    worst_removal,
)
from .params import HMMGDHHParams

# HMM states
_STATE_IMPROVING = 0
_STATE_STAGNATING = 1
_STATE_ESCAPING = 2
_N_STATES = 3


class HMMGDHHSolver(PolicyVizMixin):
    """
    Hidden Markov Model + Great Deluge Hyper-Heuristic solver for VRPP.

    Improvements over the naive HMM-GD implementation:

    1. **Forward Algorithm state belief** — instead of assigning a single
       deterministic state, the solver maintains a probability distribution
       ``belief[s]`` = P(state=s | observations_{1:t}).  The Forward Algorithm
       updates this distribution at every iteration.

    2. **Relative improvement Δ_norm** — the observation likelihood is
       parameterised by Δ_norm = (f(S_new) − f(S_old)) / |f(S_old)|, which
       normalises for problem scale.  Each state defines a Gaussian emission
       over Δ_norm so that improvements, stagnation and escaping moves are
       weighted proportionally.

    3. **No embedded local search** — ACOLocalSearch has been removed from
       the main loop.  Each LLH application stands on its own merit, giving
       the HMM a clean signal about which operator is actually effective.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HMMGDHHParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
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
        self.n_llh = params.n_llh
        self.random = random.Random(seed) if seed is not None else random.Random()

        # LLH pool
        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
        ]

        # --- HMM parameters ---

        # Emission matrix B[state][llh]: probability of selecting each LLH
        # in a given state.  Initialised uniformly; updated via Δ_norm reward.
        self._B: np.ndarray = np.ones((_N_STATES, self.n_llh)) / self.n_llh

        # State transition matrix T[s_from][s_to]: probability of moving
        # between hidden states.  Initialised with sensible priors:
        #   - Improving tends to stay improving (0.6) or stagnate (0.3)
        #   - Stagnating tends to stay stagnating (0.5) or escape (0.3)
        #   - Escaping tends to improve (0.4) or stagnate (0.4)
        self._T: np.ndarray = np.array(
            [
                [0.6, 0.3, 0.1],  # from improving
                [0.2, 0.5, 0.3],  # from stagnating
                [0.4, 0.4, 0.2],  # from escaping
            ],
            dtype=np.float64,
        )

        # Observation emission parameters per state: Gaussian(mean, std) over Δ_norm
        # Improving: positive Δ_norm; Stagnating: near-zero; Escaping: negative
        self._obs_mean = np.array([0.05, 0.0, -0.03], dtype=np.float64)
        self._obs_std = np.array([0.05, 0.02, 0.05], dtype=np.float64)

        # State belief: P(state | observations)  —  initialised uniformly
        self._belief: np.ndarray = np.ones(_N_STATES) / _N_STATES

        # Cumulative reward accumulators for Δ_norm-weighted B updates
        self._llh_reward: np.ndarray = np.zeros((_N_STATES, self.n_llh))
        self._llh_counts: np.ndarray = np.ones((_N_STATES, self.n_llh))  # avoid /0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run HMM-GD-HH and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initial solution
        routes = self._build_random_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit
        best_cost = self._cost(best_routes)

        # Great Deluge for maximisation: water level starts *below* initial profit
        # and slowly rises.  Accept any move whose profit ≥ water level.
        water_level = (
            best_profit * (1.0 - self.params.flood_margin) if best_profit > 0 else -abs(self.params.flood_margin)
        )

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Belief-weighted LLH selection probabilities
            llh_probs = self._belief @ self._B  # shape (n_llh,)
            llh_probs_sum = llh_probs.sum()
            if llh_probs_sum > 1e-12:
                llh_probs /= llh_probs_sum
            else:
                llh_probs = np.ones(self.n_llh) / self.n_llh

            llh_idx = self._sample_llh(llh_probs)
            llh = self._llh_pool[llh_idx]

            # Apply LLH (no embedded local search — image fix #3)
            try:
                new_routes = llh(routes, self.params.n_removal)
                new_profit = self._evaluate(new_routes)
            except Exception:
                new_routes = routes
                new_profit = profit

            delta = new_profit - profit

            # --- Relative improvement Δ_norm (image fix #2) ---
            delta_norm = delta / abs(profit) if abs(profit) > 1e-12 else 0.0

            # --- Great Deluge acceptance (maximisation) ---
            accepted = new_profit >= water_level

            if accepted:
                routes = new_routes
                profit = new_profit

                if profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = profit
                    best_cost = self._cost(best_routes)

            # --- Forward Algorithm belief update (image fix #1) ---
            # Compute observation likelihood P(Δ_norm | state) for each state
            obs_likelihood = self._gaussian_pdf(delta_norm)

            # Forward step: belief'[s'] = Σ_s belief[s] * T[s][s'] * P(obs | s')
            new_belief = np.zeros(_N_STATES)
            for s_next in range(_N_STATES):
                new_belief[s_next] = obs_likelihood[s_next] * np.dot(self._belief, self._T[:, s_next])

            belief_sum = new_belief.sum()
            if belief_sum > 1e-12:
                new_belief /= belief_sum
            else:
                new_belief = np.ones(_N_STATES) / _N_STATES

            self._belief = new_belief

            # --- Online emission matrix B update with Δ_norm reward ---
            # Weight each state's contribution by the current belief
            for s in range(_N_STATES):
                self._llh_counts[s][llh_idx] += self._belief[s]
                if delta_norm > 0:
                    self._llh_reward[s][llh_idx] += self._belief[s] * delta_norm

            # Recompute B[s] from accumulated rewards
            lr = self.params.learning_rate
            for s in range(_N_STATES):
                reward_rate = self._llh_reward[s][llh_idx] / self._llh_counts[s][llh_idx]
                self._B[s][llh_idx] = (1.0 - lr) * self._B[s][llh_idx] + lr * reward_rate
                # Re-normalise row so it sums to 1
                row_sum = self._B[s].sum()
                if row_sum > 1e-12:
                    self._B[s] /= row_sum
                else:
                    self._B[s] = np.ones(self.n_llh) / self.n_llh

            # Increase water level (flood rises)
            water_level += self.params.rain_speed * abs(best_profit + 1e-9)

            # Determine most likely state for viz
            hmm_state = int(np.argmax(self._belief))

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                water_level=water_level,
                hmm_state=hmm_state,
                llh_selected=llh_idx,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Forward Algorithm helpers
    # ------------------------------------------------------------------

    def _gaussian_pdf(self, delta_norm: float) -> np.ndarray:
        """Compute Gaussian observation likelihood P(Δ_norm | state) per state."""
        diff = delta_norm - self._obs_mean
        exponent = -0.5 * (diff / self._obs_std) ** 2
        pdf = np.exp(exponent) / (self._obs_std * np.sqrt(2.0 * np.pi))
        # Clamp to avoid numerical zeros
        return np.maximum(pdf, 1e-12)

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L0: random_removal + greedy_insertion."""
        partial, removed = random_removal(routes, n, self.random)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L1: worst_removal + regret_2_insertion."""
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L2: cluster_removal + greedy_insertion."""
        partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L3: worst_removal + greedy_insertion."""
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L4: random_removal + regret_2_insertion."""
        partial, removed = random_removal(routes, n, self.random)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_llh(self, probs: np.ndarray) -> int:
        """Sample an LLH index from the probability distribution."""
        r = self.random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes

        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return optimized_routes

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
