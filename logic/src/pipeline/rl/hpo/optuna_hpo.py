"""
Optuna HPO Integration.
"""
from typing import Callable

import optuna

from logic.src.configs import Config


class OptunaHPO:
    """
    Handles Optuna-based Hyperparameter Optimization.
    Supports TPE, Random, Grid, and Hyperband pruning.
    """

    def __init__(self, cfg: Config, objective_fn: Callable):
        self.cfg = cfg
        self.objective_fn = objective_fn

    def run(self) -> float:
        """Run the optimization study."""
        sampler = self._get_sampler()
        pruner = self._get_pruner()

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            lambda trial: self.objective_fn(trial, self.cfg),
            n_trials=self.cfg.hpo.n_trials,
            n_jobs=1,
        )

        return study.best_value

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        seed = self.cfg.seed
        if self.cfg.hpo.method == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        elif self.cfg.hpo.method == "grid":
            # Grid sampler requires search space to be static
            search_space = {k: v for k, v in self.cfg.hpo.search_space.items()}
            return optuna.samplers.GridSampler(search_space)
        else:
            # Default TPE
            return optuna.samplers.TPESampler(seed=seed)

    def _get_pruner(self) -> optuna.pruners.BasePruner:
        if self.cfg.hpo.method == "hyperband":
            return optuna.pruners.HyperbandPruner()
        return optuna.pruners.MedianPruner()
