"""
Hydra dispatch module.

This module provides the unified Hydra entry point for all configuration-driven
commands in the WSmart-Route application. It handles dispatching to the
appropriate task handler based on the command-line arguments.

Attributes:
    cs: ConfigStore instance for registering configurations.
    ROOT_KEYS: List of root-level keys to filter when pretty-printing configs.
    hydra_entry_point: Unified entry point for all configuration-driven commands.
    _run_task: Dispatch to the appropriate task handler.
    _pretty_print_hydra_config: Pretty print filtered sections of the Hydra configuration.

Example:
    >>> from logic.controller.hydra_dispatch import hydra_entry_point
    >>> hydra_entry_point()
    # Runs the default task (train) with default configuration
    >>> hydra_entry_point("--task=eval --eval.model_path=path/to/model")
    # Runs the evaluation task with specified model path
"""

import os
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.constants import CONFIGS_DIR

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

ROOT_KEYS = ["load_dataset", "seed", "device", "experiment_name", "task", "output_dir", "run_name", "start", "tracking"]


def _pretty_print_hydra_config(cfg: Any, filter_keys: Any = None) -> None:
    """
    Pretty print filtered sections of the Hydra configuration.

    Args:
        cfg: The Hydra configuration object.
        filter_keys: Optional list of keys to filter the configuration by.

    Returns:
        None
    """
    print("\n" + "=" * 80)
    print("HYDRA CONFIGURATION".center(80))
    print("=" * 80)
    display_cfg = OmegaConf.masked_copy(cfg, filter_keys) if filter_keys else cfg
    print(OmegaConf.to_yaml(display_cfg, resolve=False))
    print("=" * 80 + "\n")


@hydra.main(version_base=None, config_path=os.path.join(CONFIGS_DIR, "assets", "configs"), config_name="config")
def hydra_entry_point(cfg: Config) -> float:
    """
    Unified Hydra entry point for all configuration-driven commands.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        float: The result of the executed task.
    """
    if cfg.tracking.profile:
        from logic.src.tracking.profiling import start_global_profiling, stop_global_profiling

        start_global_profiling(log_dir=cfg.tracking.log_dir)

    try:
        return _run_task(cfg)
    finally:
        if cfg.tracking.profile:
            stop_global_profiling()


def _run_task(cfg: Config) -> float:
    """
    Dispatch to the appropriate task handler.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        float: The result of the executed task.
    """
    task = cfg.task

    if task in ("train", "meta_train", "hpo"):
        from logic.src.pipeline.features.train import run_hpo, run_training

        if cfg.tracking.verbose:
            _pretty_print_hydra_config(
                cfg,
                # type: ignore[arg-type]
                filter_keys=ROOT_KEYS + ["env", "model", "train", "rl", "optim"],
            )
        if cfg.hpo.n_trials > 0:
            return run_hpo(cfg)
        return run_training(cfg)

    if task == "eval":
        from logic.src.pipeline.features.eval import run_evaluate_model

        if cfg.tracking.verbose:
            _pretty_print_hydra_config(cfg, filter_keys=ROOT_KEYS + ["eval"])  # type: ignore[arg-type]
        run_evaluate_model(cfg)
        return 0.0

    if task == "test_sim":
        from logic.src.pipeline.features.test import run_wsr_simulator_test

        if cfg.tracking.verbose:
            _pretty_print_hydra_config(cfg, filter_keys=ROOT_KEYS + ["sim"])  # type: ignore[arg-type]
        run_wsr_simulator_test(cfg)
        return 0.0

    if task == "hpo_sim":
        from logic.src.policies.helpers.hpo.hpo_handler import run_hpo_sim

        if cfg.tracking.verbose:
            _pretty_print_hydra_config(cfg, filter_keys=ROOT_KEYS + ["hpo_sim"])  # type: ignore[arg-type]
        run_hpo_sim(cfg)
        return 0.0

    if task == "gen_data":
        import logic.src.tracking as wst
        from logic.src.data.generators import generate_datasets

        if cfg.tracking.verbose:
            _pretty_print_hydra_config(cfg, filter_keys=ROOT_KEYS + ["data"])  # type: ignore[arg-type]

        experiment_name = cfg.experiment_name or f"gen_data_{cfg.data.problem}"
        wst.init(experiment_name=experiment_name)
        try:
            generate_datasets(cfg)
        finally:
            run = wst.get_active_run()
            if run is not None:
                run.set_tag("status", "completed")
                run.flush()
        return 0.0

    raise ValueError(f"Unknown task: {task}")
