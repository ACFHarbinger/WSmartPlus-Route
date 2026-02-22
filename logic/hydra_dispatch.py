import os
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.constants import ROOT_DIR

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

ROOT_KEYS = ["load_dataset", "seed", "device", "experiment_name", "task", "output_dir", "run_name", "start", "tracking"]


def _pretty_print_hydra_config(cfg: Any, filter_keys: Any = None) -> None:
    """Pretty print filtered sections of the Hydra configuration."""
    print("\n" + "=" * 80)
    print("HYDRA CONFIGURATION".center(80))
    print("=" * 80)
    display_cfg = OmegaConf.masked_copy(cfg, filter_keys) if filter_keys else cfg
    print(OmegaConf.to_yaml(display_cfg, resolve=False))
    print("=" * 80 + "\n")


@hydra.main(version_base=None, config_path=os.path.join(ROOT_DIR, "assets", "configs"), config_name="config")
def hydra_entry_point(cfg: Config) -> float:
    """Unified Hydra entry point for all configuration-driven commands."""
    if cfg.tracking.profile:
        from logic.src.tracking.profiling import start_global_profiling, stop_global_profiling

        start_global_profiling(log_dir=cfg.tracking.log_dir)

    try:
        return _run_task(cfg)
    finally:
        if cfg.tracking.profile:
            stop_global_profiling()


def _run_task(cfg: Config) -> float:
    """Dispatch to the appropriate task handler."""
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

    if task == "gen_data":
        from logic.src.data.generators import generate_datasets

        if cfg.tracking.verbose:
            _pretty_print_hydra_config(cfg, filter_keys=ROOT_KEYS + ["data"])  # type: ignore[arg-type]
        generate_datasets(cfg)
        return 0.0

    raise ValueError(f"Unknown task: {task}")
