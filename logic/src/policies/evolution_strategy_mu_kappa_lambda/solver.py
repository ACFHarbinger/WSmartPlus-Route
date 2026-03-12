"""
(μ,κ,λ) Evolution Strategy with age-based selection.

This implementation follows the classical ES formulation from Emmerich et al. (2015),
where selection occurs from μ parents who have not exceeded an age of κ and λ offspring.

Key features:
- **Mutative self-adaptation**: Step sizes (σ) evolve alongside decision variables
- **Age-based selection**: Parents exceeding age κ are discarded
- **Recombination**: Intermediate or discrete recombination of ρ parents
- **Gaussian mutation**: x'_i ← x_i + σ_i · N(0,1)

Algorithm (from paper, page 5):
    1. Initialize P_0 with μ parent individuals
    2. For each generation t:
        a. Recombine(P_{t-1}) → create λ offspring
        b. Mutate(offspring) → apply self-adaptive mutation
        c. Evaluate(offspring) → compute fitness
        d. Select(offspring ∪ eligible_parents) → choose μ best
           where eligible_parents are those with age ≤ κ
        e. Increment age of all surviving individuals
        f. Update best solution if improved
    3. Return best solution found

Complexity:
    - Time: O(T × λ × d) where T = iterations, λ = offspring, d = dimensions
    - Space: O((μ + λ) × d) for population + offspring

Reference:
    Emmerich, M., Shir, O. M., & Wang, H. (2015). Evolution Strategies.
    In: Handbook of Natural Computing (pages 1-31).
"""

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from .params import MuKappaLambdaESParams
except ImportError:
    from params import MuKappaLambdaESParams


class Individual:
    """
    Individual in the Evolution Strategy population.

    Attributes:
        x: Decision variables (solution vector).
        sigma: Step sizes for mutation (one per dimension for individual adaptation).
        fitness: Objective function value.
        age: Number of generations the individual has survived.
    """

    def __init__(self, x: np.ndarray, sigma: np.ndarray, fitness: float = -np.inf, age: int = 0):
        """
        Initialize an individual.

        Args:
            x: Decision variable vector (d-dimensional).
            sigma: Step size vector (d-dimensional).
            fitness: Fitness value (default: -∞).
            age: Age in generations (default: 0).
        """
        self.x = x.copy()
        self.sigma = sigma.copy()
        self.fitness = fitness
        self.age = age

    def copy(self) -> "Individual":
        """Create a deep copy of the individual."""
        return Individual(x=self.x.copy(), sigma=self.sigma.copy(), fitness=self.fitness, age=self.age)


class MuKappaLambdaESSolver:
    """
    (μ,κ,λ) Evolution Strategy solver for continuous optimization.

    Implements age-based selection where parents exceeding age κ are excluded
    from selection pool. Uses mutative self-adaptation of step sizes.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimension: int,
        params: MuKappaLambdaESParams,
        seed: Optional[int] = None,
        minimize: bool = True,
    ):
        """
        Initialize the (μ,κ,λ) Evolution Strategy solver.

        Args:
            objective_function: Function to optimize f(x) → ℝ.
            dimension: Dimensionality of the search space.
            params: ES configuration parameters.
            seed: Random seed for reproducibility.
            minimize: If True, minimize; if False, maximize.
        """
        self.f = objective_function
        self.d = dimension
        self.params = params
        self.minimize = minimize
        self.rng = np.random.RandomState(seed)

        # Adjust learning rates based on dimensionality (paper recommendation, page 7)
        self.params.tau_local = 1.0 / np.sqrt(2.0 * self.d)
        self.params.tau_global = 1.0 / (2.0 * np.sqrt(self.d))

        # Statistics
        self.n_evaluations = 0
        self.best_individual: Optional[Individual] = None
        self.convergence_curve: List[float] = []

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Execute the (μ,κ,λ) Evolution Strategy.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        start_time = time.time()

        # Initialize population P_0 with μ parents (Algorithm 1, page 6)
        parents = self._initialize_population()

        # Evaluate initial population
        for ind in parents:
            ind.fitness = self._evaluate(ind.x)

        # Track global best
        self._update_best(parents)

        # Evolution loop (Algorithm 1, page 6)
        generation = 0
        while generation < self.params.max_iterations:
            if self.params.time_limit > 0 and (time.time() - start_time) > self.params.time_limit:
                break

            # Step 1: Generate λ offspring via recombination and mutation
            offspring = []
            for _ in range(self.params.lambda_):
                # Select ρ parents for recombination (page 7)
                selected_parents = self._select_parents_for_recombination(parents)

                # Create offspring via recombination
                child = self._recombine(selected_parents)

                # Apply mutation with self-adaptive step sizes (Eq. 4-6, page 9)
                mutated_child = self._mutate(child)

                # Evaluate offspring
                mutated_child.fitness = self._evaluate(mutated_child.x)

                offspring.append(mutated_child)

            # Step 2: Age-based selection - select from eligible parents + offspring
            # Eligible parents are those with age ≤ κ (page 5)
            eligible_parents = [p for p in parents if p.age < self.params.kappa]

            # Combine eligible parents and offspring for selection pool
            selection_pool = eligible_parents + offspring

            # Step 3: Select μ best individuals from the pool
            # Sort by fitness (ascending for minimization, descending for maximization)
            selection_pool.sort(key=lambda ind: ind.fitness, reverse=not self.minimize)
            parents = selection_pool[: self.params.mu]

            # Step 4: Increment age of surviving parents
            for p in parents:
                if p in eligible_parents:  # Only increment if it was a parent
                    p.age += 1

            # Update global best
            self._update_best(parents)

            # Record convergence
            self.convergence_curve.append(self.best_individual.fitness)

            generation += 1

        return self.best_individual.x.copy(), self.best_individual.fitness

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_population(self) -> List[Individual]:
        """
        Initialize population with μ individuals (page 6).

        Individuals are initialized with:
        - x: Uniform random in [bounds_min, bounds_max]
        - σ: Initial step size (recommended 5% of search space, page 7)
        - age: 0

        Returns:
            List of μ initialized individuals.

        Complexity: O(μ × d)
        """
        population = []
        for _ in range(self.params.mu):
            # Initialize decision variables
            if self.params.bounds_min is not None and self.params.bounds_max is not None:
                x = self.rng.uniform(self.params.bounds_min, self.params.bounds_max, size=self.d)
            else:
                x = self.rng.randn(self.d)

            # Initialize step sizes (recommended 5% of search space, page 7)
            sigma = np.full(self.d, self.params.initial_sigma)

            ind = Individual(x=x, sigma=sigma, age=0)
            population.append(ind)

        return population

    # ------------------------------------------------------------------
    # Recombination (page 7)
    # ------------------------------------------------------------------

    def _select_parents_for_recombination(self, parents: List[Individual]) -> List[Individual]:
        """
        Select ρ parents for recombination via uniform random sampling (page 7).

        Args:
            parents: Current parent population.

        Returns:
            List of ρ randomly selected parents.

        Complexity: O(ρ)
        """
        indices = self.rng.choice(len(parents), size=self.params.rho, replace=False)
        return [parents[i] for i in indices]

    def _recombine(self, parents: List[Individual]) -> Individual:
        """
        Create offspring via recombination of ρ parents (page 7).

        Two types:
        1. **Intermediate recombination** (page 7, Eq. before Eq. 3):
           r_j = (1/ρ) Σ q_j^(i) for i=1..ρ
           Averages components across parents.

        2. **Discrete (dominant) recombination** (page 7):
           r_j = q_u_j^(j) where u_j ~ Uniform{1..ρ}
           Each component randomly chosen from one parent.

        Args:
            parents: List of ρ parent individuals.

        Returns:
            Offspring individual.

        Complexity: O(ρ × d)
        """
        if self.params.recombination_type == "intermediate":
            # Intermediate recombination: average all parents
            x_child = np.mean([p.x for p in parents], axis=0)
            sigma_child = np.mean([p.sigma for p in parents], axis=0)
        else:
            # Discrete recombination: randomly select component from parents
            x_child = np.zeros(self.d)
            sigma_child = np.zeros(self.d)
            for j in range(self.d):
                parent_idx = self.rng.randint(len(parents))
                x_child[j] = parents[parent_idx].x[j]
                sigma_child[j] = parents[parent_idx].sigma[j]

        return Individual(x=x_child, sigma=sigma_child, age=0)

    # ------------------------------------------------------------------
    # Mutation with Self-Adaptation (page 9, Eq. 4-6)
    # ------------------------------------------------------------------

    def _mutate(self, individual: Individual) -> Individual:
        """
        Apply mutative self-adaptation to offspring (page 9, Eq. 4-6).

        Step-size mutation (Eq. 5):
            N_global ~ N(0,1)  (shared across all dimensions)
            σ'_i ← σ_i · exp(τ_global · N_global + τ_local · N_i(0,1))

        Decision variable mutation (Eq. 6):
            x'_i ← x_i + σ'_i · N(0,1)

        Learning rates (page 7):
            τ_local = 1/√(2d)
            τ_global = 1/(2√d)

        Args:
            individual: Individual to mutate.

        Returns:
            Mutated individual.

        Complexity: O(d)
        """
        mutant = individual.copy()

        # Step 1: Mutate step sizes (Eq. 5)
        N_global = self.rng.randn()  # Global random factor (Eq. 4)

        for i in range(self.d):
            N_local = self.rng.randn()
            mutant.sigma[i] *= np.exp(self.params.tau_global * N_global + self.params.tau_local * N_local)

            # Clamp σ to prevent degeneration (page 8, recommended practice)
            mutant.sigma[i] = np.clip(mutant.sigma[i], self.params.min_sigma, self.params.max_sigma)

        # Step 2: Mutate decision variables (Eq. 6)
        for i in range(self.d):
            mutant.x[i] += mutant.sigma[i] * self.rng.randn()

            # Apply bounds if specified
            if self.params.bounds_min is not None and self.params.bounds_max is not None:
                mutant.x[i] = np.clip(mutant.x[i], self.params.bounds_min, self.params.bounds_max)

        return mutant

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate objective function and track evaluation count.

        Args:
            x: Decision variable vector.

        Returns:
            Fitness value.

        Complexity: O(1) + O(f)
        """
        self.n_evaluations += 1
        return self.f(x)

    def _update_best(self, population: List[Individual]):
        """
        Update global best solution.

        Args:
            population: Current population.

        Complexity: O(μ)
        """
        if self.minimize:
            current_best = min(population, key=lambda ind: ind.fitness)
            if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best.copy()
        else:
            current_best = max(population, key=lambda ind: ind.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """
        Get optimization statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "n_evaluations": self.n_evaluations,
            "best_fitness": self.best_individual.fitness if self.best_individual else None,
            "best_solution": self.best_individual.x.copy() if self.best_individual else None,
            "convergence_curve": self.convergence_curve.copy(),
        }
