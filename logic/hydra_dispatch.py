import os
from typing import Any, cast

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from logic.src.configs import Config
from logic.src.constants import ROOT_DIR

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def _pretty_print_hydra_config(cfg: DictConfig, filter_keys: Any = None) -> None:
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
    task = cfg.task

    if task in ("train", "meta_train", "hpo"):
        from logic.src.pipeline.features.train import run_hpo, run_training

        if cfg.verbose:
            _pretty_print_hydra_config(cfg, filter_keys=["env", "model", "train", "rl", "optim"])  # type: ignore[arg-type]
        if cfg.hpo.n_trials > 0:
            return run_hpo(cfg)
        return run_training(cfg)

    if task == "evaluation":
        from logic.src.pipeline.features.base import flatten_config_dict
        from logic.src.pipeline.features.eval import run_evaluate_model, validate_eval_args

        if cfg.verbose:
            _pretty_print_hydra_config(cfg, filter_keys="eval")  # type: ignore[arg-type]
        eval_args = cast(dict[str, Any], OmegaConf.to_container(cfg.eval, resolve=True))
        eval_args = flatten_config_dict(eval_args)
        args = validate_eval_args(eval_args)
        run_evaluate_model(args)
        return 0.0

    if task == "test_sim":
        from logic.src.pipeline.features.base import flatten_config_dict
        from logic.src.pipeline.features.test import run_wsr_simulator_test, validate_test_sim_args

        if cfg.verbose:
            _pretty_print_hydra_config(cfg, filter_keys="sim")  # type: ignore[arg-type]
        sim_args = cast(dict[str, Any], OmegaConf.to_container(cfg.sim, resolve=True))
        sim_args = flatten_config_dict(sim_args)
        args = validate_test_sim_args(sim_args)
        run_wsr_simulator_test(args)
        return 0.0

    if task == "gen_data":
        from logic.src.data.generators import generate_datasets, validate_gen_data_args
        from logic.src.pipeline.features.base import flatten_config_dict

        if cfg.verbose:
            _pretty_print_hydra_config(cfg, filter_keys="data")  # type: ignore[arg-type]
        data_args = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
        data_args = flatten_config_dict(data_args)
        args = validate_gen_data_args(data_args)
        generate_datasets(args)
        return 0.0

    raise ValueError(f"Unknown task: {task}")
