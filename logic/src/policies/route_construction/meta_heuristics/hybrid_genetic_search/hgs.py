"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits (VRPP).

Reference:
    Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source
    implementation and SWAP* neighborhood. Computers & Operations Research,
    140, 105643.

Attributes:
    HGSSolver: Main solver class for the Hybrid Genetic Search.

Example:
    >>> solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
from logic.src.policies.helpers.operators.crossover_recombination import ordered_crossover

from .evolution import _extract_edges, evaluate, update_biased_fitness
from .individual import Individual
from .params import HGSParams
from .split import LinearSplit


class HGSSolver:
    """
    Implements Hybrid Genetic Search for the VRPP following Vidal et al. (2022).

    Key invariant maintained throughout the main loop:
        The biased fitness of every individual in both subpopulations is always
        up-to-date. Fitness is recomputed immediately after every population
        modification (insertion or deletion), never lazily before selection.
        This allows _generate_offspring / _select_parents to use fitness values
        directly without a recomputation step.

    Attributes:
        d (np.ndarray): NxN distance matrix.
        wastes (Dict[int, float]): Dictionary of node wastes.
        Q (float): Maximum vehicle capacity.
        R (float): Revenue multiplier.
        C (float): Cost multiplier.
        params (HGSParams): Detailed HGS parameters.
        mandatory_nodes (Optional[List[int]]): List of nodes that must be visited.
        random (random.Random): Seeded random number generator.
        n_nodes (int): Number of customer nodes (excluding depot).
        nodes (List[int]): Customer node indices [1, ..., n_nodes].
        split_manager (LinearSplit): Split algorithm manager.
        ls (HGSLocalSearch): Local search engine.

    Example:
        >>> solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params)
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSParams,
        mandatory_nodes: Optional[List[int]] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
    ):
        """
        Initialize the HGS solver.

        Args:
            dist_matrix: NxN distance matrix (index 0 = depot).
            wastes: Dictionary of node wastes (demands).
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier (profit per unit of waste collected).
            C: Cost multiplier (cost per unit distance).
            params: HGS configuration parameters.
            mandatory_nodes: Local node indices that MUST be visited.
            x_coords: Optional x-coordinates for SWAP* polar sector pruning.
            y_coords: Optional y-coordinates for SWAP* polar sector pruning.

        Returns:
            None.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(
            dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes, vrpp=params.vrpp
        )
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, params)
        if x_coords is not None and y_coords is not None:
            self.ls.x_coords = x_coords
            self.ls.y_coords = y_coords

        # Per-subpopulation diversity distance caches, keyed by (id(ind_a), id(ind_b)).
        # Maintained across iterations; entries are evicted precisely when an individual
        # is removed, avoiding the stale-hit problem from clearing the entire cache.
        self._feas_cache: dict = {}
        self._infeas_cache: dict = {}

        # Rolling list of offspring feasibility outcomes (True/False), used to compute
        # the recent feasibility rate for penalty adaptation.
        self._offspring_feasibility: List[bool] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_cache(self, ind: Individual, cache: dict) -> None:
        """
        Remove all diversity distance entries that involve a specific individual.

        Called immediately before removing an individual from a subpopulation
        so that the object-ID key can never produce a stale cache hit (Python
        may reuse the id() of a garbage-collected object for a new individual).

        Args:
            ind: Individual about to be removed.
            cache: Distance cache to purge (modified in-place).

        Returns:
            None.
        """
        ind_id = id(ind)
        stale_keys = [k for k in cache if ind_id in k]
        for k in stale_keys:
            del cache[k]

    def _insert_into_pop(
        self,
        ind: Individual,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
    ) -> None:
        """
        Insert an individual into its respective subpopulation and immediately
        update biased fitness for that subpopulation.

        This is the single point of population modification during the main loop,
        ensuring the fitness-current invariant is maintained after every insertion.

        Args:
            ind: Individual to insert.
            pop_feasible: Feasible subpopulation (modified in-place).
            pop_infeasible: Infeasible subpopulation (modified in-place).

        Returns:
            None.
        """
        if ind.is_feasible:
            pop_feasible.append(ind)
            update_biased_fitness(pop_feasible, self.params.nb_elite, self.params.nb_close, self._feas_cache)
        else:
            pop_infeasible.append(ind)
            update_biased_fitness(pop_infeasible, self.params.nb_elite, self.params.nb_close, self._infeas_cache)

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, penalty_capacity: float) -> Tuple[List[Individual], List[Individual]]:
        """
        Initialize dual subpopulations with 4μ random solutions improved by local search.

        Following Vidal (2022) Algorithm 1: "Initialize population with random
        solutions improved by local search." Fitness is updated once after all
        individuals are generated (bulk update is more efficient than incremental
        updates during initialization).

        Args:
            penalty_capacity: Initial penalty coefficient for capacity violations.

        Returns:
            Tuple[List[Individual], List[Individual]]: (feasible pop, infeasible pop).
        """
        pop_feasible: List[Individual] = []
        pop_infeasible: List[Individual] = []

        initial_size = 4 * self.params.mu
        for _ in range(initial_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager, penalty_capacity)
            ind = self.ls.optimize(ind)
            evaluate(ind, self.split_manager, penalty_capacity)

            if ind.is_feasible:
                pop_feasible.append(ind)
            else:
                pop_infeasible.append(ind)

        # Bulk fitness update after initialization (uses the instance caches which
        # were reset in solve() / restart logic before this call).
        update_biased_fitness(pop_feasible, self.params.nb_elite, self.params.nb_close, self._feas_cache)
        update_biased_fitness(pop_infeasible, self.params.nb_elite, self.params.nb_close, self._infeas_cache)

        return pop_feasible, pop_infeasible

    # ------------------------------------------------------------------
    # Survivor selection
    # ------------------------------------------------------------------

    def _trim_pop(self, pop: List[Individual], cache: dict) -> None:
        """
        Trim a subpopulation from μ+λ down to μ, removing clones first then
        worst-fitness individuals.

        Following Vidal (2022) Section 2: "iteratively removing a clone solution
        (i.e., identical to another one) if such a solution exists, or otherwise
        the worst solution in terms of fitness according to Eq. (1)."

        Cache entries for evicted individuals are removed precisely (not via a
        full clear), so valid pairwise distances for surviving individuals are
        reused in subsequent fitness updates.

        Args:
            pop: Subpopulation to trim (modified in-place).
            cache: Per-subpopulation distance cache (modified in-place).

        Returns:
            None.
        """
        if len(pop) < self.params.mu + self.params.n_offspring:
            return

        while len(pop) > self.params.mu:
            update_biased_fitness(pop, self.params.nb_elite, self.params.nb_close, cache)

            clone_idx = self._find_clone(pop)
            if clone_idx is not None:
                self._evict_cache(pop[clone_idx], cache)
                pop.pop(clone_idx)
            else:
                worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness)
                self._evict_cache(pop[worst_idx], cache)
                pop.pop(worst_idx)

    def _find_clone(self, pop: List[Individual]) -> Optional[int]:
        """
        Return the index of any individual whose solution is identical (by edge set)
        to an earlier individual in the population, or None if no clones exist.

        FIX (was: frozenset of route tuples): Uses edge-set hashing, which is the
        same undirected representation used in broken-pairs distance computation.
        This correctly identifies clones regardless of route traversal direction:
        route [1,2,3] and route [3,2,1] produce identical edge sets {(0,1),(1,2),
        (2,3),(3,0)} and are thus correctly detected as clones.

        Args:
            pop: Subpopulation to inspect.

        Returns:
            Optional[int]: Index of a clone, or None.
        """
        seen: set = set()
        for i, ind in enumerate(pop):
            # _extract_edges returns a set of frozenset({u, v}) undirected edges.
            key = frozenset(_extract_edges(ind))
            if key in seen:
                return i
            seen.add(key)
        return None

    # ------------------------------------------------------------------
    # Parent selection
    # ------------------------------------------------------------------

    def _select_parents(
        self,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
    ) -> Tuple[Individual, Individual]:
        """
        Binary tournament selection from both subpopulations.

        Fitness is assumed to be current (maintained by _insert_into_pop and
        _initialize_population — no recomputation is performed here).

        Args:
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.

        Returns:
            Tuple[Individual, Individual]: Two selected parents (p1, p2).
        """

        combined = pop_feasible + pop_infeasible

        def tournament(pop: List[Individual]) -> Individual:
            """
            Helper function to perform binary tournament selection.

            Args:
                pop: Population to select from.

            Returns:
                Individual: Selected parent.
            """
            i1, i2 = self.random.sample(pop, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(combined), tournament(combined)

    # ------------------------------------------------------------------
    # Offspring generation
    # ------------------------------------------------------------------

    def _generate_offspring(
        self,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
    ) -> Individual:
        """
        Generate and educate a new offspring via crossover and local search.

        FIX (was: called update_biased_fitness before every selection):
        Fitness is maintained as an invariant and is always current on entry,
        so no recomputation is needed here. The O(P²) pairwise distance
        computation now happens only when the population actually changes
        (inside _insert_into_pop and _trim_pop), not every iteration.

        FIX: crossover_rate is now applied (was declared but never used).
        When crossover is skipped, P1 is cloned so local search can still
        educate the individual — equivalent to a mutation-only step.

        Args:
            pop_feasible: Feasible subpopulation (fitness assumed current).
            pop_infeasible: Infeasible subpopulation (fitness assumed current).
            penalty_capacity: Current penalty coefficient.

        Returns:
            Individual: Educated offspring.
        """
        p1, p2 = self._select_parents(pop_feasible, pop_infeasible)

        if self.random.random() < self.params.crossover_rate:
            child = ordered_crossover(p1, p2, rng=self.random)
        else:
            # No crossover: clone p1 so local search education still occurs.
            child = Individual(p1.giant_tour[:], expand_pool=p1.expand_pool)

        evaluate(child, self.split_manager, penalty_capacity)
        child = self.ls.optimize(child)
        evaluate(child, self.split_manager, penalty_capacity)
        return child

    def _insert_and_repair(
        self,
        child: Individual,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
    ) -> None:
        """
        Insert child into its respective subpopulation, then optionally repair
        if infeasible.

        FIX (was: discarded infeasible repaired solutions): Vidal (2022)
        Algorithm 1 step 8 states "repair C (local search) and insert it into
        respective subpopulation" — the word "respective" means the subpopulation
        is determined by the repaired individual's own feasibility, and insertion
        is unconditional. A repaired solution that remains infeasible may still
        be a significantly better infeasible solution than what is in the pool,
        and discarding it wastes the repair effort.

        The child's feasibility is also recorded for penalty adaptation tracking.

        Args:
            child: Educated offspring.
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.
            penalty_capacity: Current penalty coefficient.

        Returns:
            None.
        """
        self._insert_into_pop(child, pop_feasible, pop_infeasible)
        self._offspring_feasibility.append(child.is_feasible)

        if not child.is_feasible and self.random.random() < self.params.repair_probability:
            repaired = Individual(child.giant_tour[:], expand_pool=child.expand_pool)
            # Repair = local search under 10× penalty (Vidal 2022 Section 2)
            evaluate(repaired, self.split_manager, penalty_capacity * 10.0)
            repaired = self.ls.optimize(repaired)
            evaluate(repaired, self.split_manager, penalty_capacity)
            # FIX: insert regardless of whether repair achieved feasibility.
            self._insert_into_pop(repaired, pop_feasible, pop_infeasible)

    # ------------------------------------------------------------------
    # Penalty adaptation
    # ------------------------------------------------------------------

    def _adjust_penalties(self, current_penalty: float) -> float:
        """
        Adapt the capacity penalty coefficient to maintain the target proportion
        of feasible offspring.

        FIX (was: measuring len(pop_feasible) / total_population, i.e., the
        steady-state population composition): Vidal (2022) Section 2 specifies
        "monitoring the number of feasible solutions *obtained*" — i.e., the
        feasibility rate of recently produced offspring after local-search
        education, not the accumulated population ratio. The population ratio is
        a lagging indicator that conflates the current penalty's effect with
        historical decisions. The offspring-feasibility rate directly reflects
        the current penalty's pressure on the search.

        Uses the last 100 offspring entries from self._offspring_feasibility,
        matching the every-100-iterations adaptation frequency.

        Args:
            current_penalty: Current capacity penalty coefficient.

        Returns:
            float: Updated penalty coefficient.
        """
        if not self._offspring_feasibility:
            return current_penalty

        recent = self._offspring_feasibility[-100:]
        feasible_ratio = sum(recent) / len(recent)

        if feasible_ratio > self.params.target_feasible + 0.05:
            # Too many feasible offspring: reduce penalty to allow more infeasibility
            return current_penalty * self.params.penalty_decrease
        elif feasible_ratio < self.params.target_feasible - 0.05:
            # Too few feasible offspring: increase penalty to push toward feasibility
            return current_penalty * self.params.penalty_increase
        return current_penalty

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def _get_best_solution(
        self, pop_feasible: List[Individual], pop_infeasible: List[Individual]
    ) -> Tuple[List[List[int]], float, float]:
        """
        Return the best feasible solution found, or the best infeasible solution
        if no feasible individuals remain in the population.

        Args:
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.

        Returns:
            Tuple[List[List[int]], float, float]: Routes, profit, and cost.
        """
        if pop_feasible:
            best_ind = max(pop_feasible, key=lambda x: x.profit_score)
        elif pop_infeasible:
            best_ind = max(pop_infeasible, key=lambda x: x.profit_score)
        else:
            return [], 0.0, 0.0
        return best_ind.routes, best_ind.profit_score, best_ind.cost

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm following Vidal et al. (2022)
        Algorithm 1, with VRPP adaptations.

        Summary of fixes applied in this method and the helpers it calls:
            1. Wall-clock time (perf_counter) instead of process time, matching
               T_max semantics in the paper.
            2. Fitness maintained as an invariant (updated after each population
               modification), not recomputed before every tournament.
            3. Per-subpopulation distance caches with precise entry eviction,
               replacing a shared cache that was cleared after every trim.
            4. Offspring feasibility tracking for penalty adaptation, replacing
               the population-composition ratio.
            5. Per-subpopulation parent tournament, replacing combined-pool
               tournament which made cross-subpopulation rank comparison invalid.
            6. Repaired individuals inserted unconditionally into their respective
               subpopulation (feasible or infeasible), per Algorithm 1 step 8.
            7. crossover_rate parameter wired up (was declared but never applied).
            8. Edge-set clone detection handling reversed routes correctly.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, total cost.
        """
        # FIX: use wall-clock time (perf_counter) to match T_max in Vidal (2022)
        # Section 5. perf_counter() measures CPU time consumed by this process,
        # which diverges from wall clock under I/O waits or system scheduling.
        start_time = time.perf_counter()

        penalty_capacity = self.params.initial_penalty_capacity

        # Reset all per-run state before initializing the population.
        # These are instance variables so they are also accessible from helper methods.
        self._feas_cache = {}
        self._infeas_cache = {}
        self._offspring_feasibility = []

        if self.params.acceptance_criterion is not None:
            self.params.acceptance_criterion.setup(0.0)

        pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)

        # Track the globally best feasible individual across all restarts.
        best_feasible_ind: Optional[Individual] = (
            max(pop_feasible, key=lambda x: x.profit_score) if pop_feasible else None
        )
        best_profit_so_far = best_feasible_ind.profit_score if best_feasible_ind is not None else -float("inf")

        it = 0
        it_no_improvement = 0
        use_time_limit = self.params.time_limit > 0
        restart_threshold = (
            self.params.restart_timer if self.params.restart_timer > 0 else self.params.n_iterations_no_improvement
        )

        # Main evolutionary loop — Algorithm 1 of Vidal (2022)
        while True:
            # --- Primary stopping criterion (Algorithm 1 condition) ---
            elapsed = time.perf_counter() - start_time
            if use_time_limit and elapsed >= self.params.time_limit:
                break
            if not use_time_limit and it_no_improvement >= self.params.n_iterations_no_improvement:
                break

            # --- Restart (time-limited mode, Vidal 2022 Section 4) ---
            # "The algorithm restarts after each N_it iterations without improvement
            #  and collects the best solution until the time limit."
            if use_time_limit and it_no_improvement >= restart_threshold:
                penalty_capacity = self.params.initial_penalty_capacity
                self._feas_cache = {}
                self._infeas_cache = {}
                self._offspring_feasibility = []
                pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)
                it_no_improvement = 0

            combined = pop_feasible + pop_infeasible
            if len(combined) < 2:
                break

            it += 1
            it_no_improvement += 1

            # Steps 3–6: select parents, crossover, educate with local search
            child = self._generate_offspring(pop_feasible, pop_infeasible, penalty_capacity)

            # Steps 6–8: insert child, optionally repair infeasible offspring
            self._insert_and_repair(child, pop_feasible, pop_infeasible, penalty_capacity)

            # Track globally best feasible individual (across all restarts)
            if child.is_feasible and child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                best_feasible_ind = child
                it_no_improvement = 0

            # Steps 9–10: survivor selection — triggered when subpopulation reaches μ+λ
            if len(pop_feasible) >= self.params.mu + self.params.n_offspring:
                self._trim_pop(pop_feasible, self._feas_cache)
            if len(pop_infeasible) >= self.params.mu + self.params.n_offspring:
                self._trim_pop(pop_infeasible, self._infeas_cache)

            # Step 11: penalty adaptation every 100 iterations
            if it % 100 == 0:
                penalty_capacity = self._adjust_penalties(penalty_capacity)

            # Acceptance criterion step (extension beyond Vidal 2022)
            if self.params.acceptance_criterion is not None:
                self.params.acceptance_criterion.step(
                    current_obj=best_profit_so_far,
                    candidate_obj=child.profit_score if child.is_feasible else -float("inf"),
                    accepted=child.is_feasible and child.profit_score > best_profit_so_far,
                    iteration=it,
                    max_iterations=self.params.n_iterations_no_improvement,
                )

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(pop_feasible) + len(pop_infeasible),
            )

        # Return the globally best feasible solution found, across all restarts.
        # Fall back to _get_best_solution only if no feasible individual was ever found.
        if best_feasible_ind is not None:
            return (
                best_feasible_ind.routes,
                best_feasible_ind.profit_score,
                best_feasible_ind.cost,
            )
        return self._get_best_solution(pop_feasible, pop_infeasible)
