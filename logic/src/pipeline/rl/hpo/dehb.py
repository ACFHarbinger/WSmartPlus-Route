"""
Differential Evolution Hyperband (DEHB) implementation.
Ported and adapted from the legacy codebase.
"""
import time
from typing import Callable, Dict, Tuple

import numpy as np


class DifferentialEvolutionHyperband:
    """
    DEHB: Differential Evolution Hyperband.
    Combines the strengths of DE (evolution) and Hyperband (multi-fidelity).
    """

    def __init__(
        self,
        cs: Dict[str, Tuple[float, float]],
        f: Callable,
        min_fidelity: int = 1,
        max_fidelity: int = 10,
        eta: int = 3,
        n_workers: int = 1,
        output_path: str = "./dehb_output",
        **kwargs,
    ):
        """Initialize DifferentialEvolutionHyperband."""
        self.cs = cs
        self.f = f
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.eta = eta
        self.n_workers = n_workers
        self.output_path = output_path

        self.parameter_names = list(cs.keys())
        self.bounds = np.array([cs[name] for name in self.parameter_names])

        # Hyperband parameters
        self.s_max = int(np.floor(np.log(self.max_fidelity / self.min_fidelity) / np.log(self.eta)))
        self.B = (self.s_max + 1) * self.max_fidelity

        self.population = []
        self.history = []

    def _sample_random_config(self) -> np.ndarray:
        return np.array([np.random.uniform(low, high) for low, high in self.bounds])

    def _init_population(self, size: int):
        self.population = [self._sample_random_config() for _ in range(size)]

    def _eval(self, config_vec: np.ndarray, fidelity: int) -> float:
        config = {name: val for name, val in zip(self.parameter_names, config_vec)}
        result = self.f(config, fidelity)
        # Returning fitness (lower is better assumed by DEHB usually, or handle maximize)
        return result.get("fitness", 1e6)

    def run(self, fevals: int = 100, **kwargs):
        """Simple sequential DEHB execution for parity."""
        start_time = time.time()
        best_fitness = float("inf")
        best_config = None

        # Initial population eval at min fidelity
        if not self.population:
            self._init_population(20)

        for i, config_vec in enumerate(self.population):
            fitness = self._eval(config_vec, self.min_fidelity)
            if fitness < best_fitness:
                best_fitness = fitness
                best_config = config_vec

        # Main evolution loop (Simplified for now)
        for i in range(fevals):
            # DE Step: Mutation + Crossover
            target_idx = np.random.randint(0, len(self.population))
            idxs = [idx for idx in range(len(self.population)) if idx != target_idx]
            a, b, c = [self.population[idx] for idx in np.random.choice(idxs, 3, replace=False)]

            # Mutation
            mutant = np.clip(a + 0.5 * (b - c), self.bounds[:, 0], self.bounds[:, 1])

            # Crossover
            trial = np.copy(self.population[target_idx])
            cross_points = np.random.rand(len(self.bounds)) < 0.7
            trial[cross_points] = mutant[cross_points]

            # Evaluate at increasing fidelity (Successive Halving logic placeholder)
            fitness = self._eval(trial, self.max_fidelity)

            if fitness < best_fitness:
                best_fitness = fitness
                best_config = trial
                self.population[target_idx] = trial

        runtime = time.time() - start_time
        return best_config, runtime, self.history

    def get_incumbents(self):
        """Get best configurations found."""
        # Return best config found
        idx = np.argmin([self._eval(p, self.max_fidelity) for p in self.population])
        best_config = {name: val for name, val in zip(self.parameter_names, self.population[idx])}
        return best_config, self._eval(self.population[idx], self.max_fidelity)
