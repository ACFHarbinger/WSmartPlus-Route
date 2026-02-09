"""initializing.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import initializing
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from logic.src.constants import DAY_METRICS, ROOT_DIR
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.configs.setup_env import setup_env
from logic.src.utils.configs.setup_manager import setup_hrl_manager
from logic.src.utils.configs.setup_worker import setup_model
from logic.src.utils.logging.log_utils import setup_system_logger

from ..bins import Bins
from ..checkpoints import SimulationCheckpoint
from ..processor import (
    process_data,
    process_model_data,
    setup_basedata,
    setup_dist_path_tup,
)
from .base import SimState

if TYPE_CHECKING:
    from .base import SimulationContext


class InitializingState(SimState):
    """State handles the initialization of simulation data (graph, models, etc.)."""

    def handle(self, ctx: SimulationContext) -> None:
        """Handle.

        Args:
            ctx (SimulationContext): Description of ctx.
        """
        opts = ctx.opts

        setup_system_logger(opts.get("log_file", "logs/simulation.log"), opts.get("log_level", "INFO"))

        if not os.path.exists(ctx.results_dir):
            os.makedirs(ctx.results_dir)
            logger.info(f"Created results directory: {ctx.results_dir}")
        else:
            logger.info(f"Results directory already exists: {ctx.results_dir}")

        ctx.config = {}
        config_paths = opts.get("config_path")

        if config_paths:
            if isinstance(config_paths, dict):
                for key, path in config_paths.items():
                    try:
                        if isinstance(path, dict):
                            loaded = path
                        else:
                            loaded = load_config(path)

                        ctx.config[key] = loaded
                        print(f"[INFO] Loaded configuration for '{key}' from {path}")
                    except (OSError, ValueError) as e:
                        print(f"[Warning] Failed to load config file {path}: {e}")
            else:
                try:
                    ctx.config = load_config(config_paths)
                    print(f"[INFO] Loaded configuration from {config_paths}")
                except (OSError, ValueError):
                    pass

        # Robust neural detection: check for specific keywords as whole words or specific prefixes
        neural_keywords = ["am", "ptr", "transgcn", "neural", "ddam", "ham", "l2d"]
        should_load_neural = any(kw in ctx.pol_strip.split("_") for kw in neural_keywords) or any(
            ctx.pol_strip.startswith(kw) for kw in ["am", "ddam", "ptr"]
        )

        neural_cfg_path = os.path.join(ROOT_DIR, "assets", "configs", "policies", "policy_neural.yaml")
        if should_load_neural and os.path.exists(neural_cfg_path):
            try:
                neural_cfg = load_config(neural_cfg_path)
                if neural_cfg:
                    if "neural" in neural_cfg:
                        for pol_key, pol_val in neural_cfg["neural"].items():
                            ctx.config[pol_key] = pol_val
                            if isinstance(pol_val, list):
                                for item in pol_val:
                                    if isinstance(item, dict) and "model_path" in item:
                                        if opts.get("model_path") is None:
                                            opts["model_path"] = {}
                                        opts["model_path"][pol_key] = item["model_path"]
                    else:
                        ctx.config.update(neural_cfg)
                print(f"[INFO] Loaded configuration from {neural_cfg_path}")
            except (OSError, ValueError) as e:
                print(f"[WARNING] Failed to load neural config {neural_cfg_path}: {e}")

        data, bins_coordinates, depot = setup_basedata(opts["size"], ctx.data_dir, opts["area"], opts["waste_type"])

        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        capacities, _, _, _, _ = load_area_and_waste_type_params(opts["area"], opts["waste_type"])
        ctx.vehicle_capacity = capacities

        ctx.checkpoint = SimulationCheckpoint(ctx.results_dir, opts["checkpoint_dir"], ctx.policy, ctx.sample_id)

        saved_state, last_day = (None, 0)
        if opts["resume"]:
            saved_state, last_day = ctx.checkpoint.load_state()
            if saved_state is not None and ctx.overall_progress:
                ctx.overall_progress.update(last_day)

        configs = None
        if should_load_neural and (
            "am" in ctx.pol_strip.split("_") or "transgcn" in ctx.pol_strip.split("_") or ctx.pol_strip.startswith("am")
        ):
            # Extract decoding parameters with support for nested/namespaced decoding config
            decoding_opts = opts.get("decoding", {})
            temp = decoding_opts.get("temperature", opts.get("decoding.temperature", opts.get("temperature", 1.0)))
            strat = decoding_opts.get("strategy", opts.get("decoding.strategy", opts.get("strategy", "greedy")))

            ctx.model_env, configs = setup_model(
                ctx.policy,
                ctx.model_weights_path,
                opts["model_path"],
                ctx.device,
                ctx.lock,  # type: ignore
                temp,
                strat,
            )
            ctx.hrl_manager = setup_hrl_manager(
                opts,
                ctx.device,
                configs,
                policy=ctx.policy,
                base_path=ctx.model_weights_path,
                worker_model=ctx.model_env,
            )
        elif "vrpp" in ctx.pol_strip:
            ctx.model_env = setup_env(
                ctx.policy,
                opts["server_run"],
                opts["gplic_file"],
                opts["symkey_name"],
                opts["env_file"],
            )
            ctx.model_tup = (None, None)
        else:
            ctx.model_tup = (None, None)

        if opts["resume"] and saved_state is not None:
            (
                ctx.new_data,
                ctx.coords,
                ctx.dist_tup,
                adj_matrix,
                ctx.bins,
                ctx.model_tup,
                ctx.cached,
                ctx.overflows,
                ctx.current_collection_day,
                ctx.daily_log,
                ctx.run_time,
            ) = saved_state
            ctx.start_day = last_day + 1
        else:
            ctx.new_data, ctx.coords = process_data(data, bins_coordinates, depot, ctx.indices)

            ctx.dist_tup, adj_matrix = setup_dist_path_tup(
                ctx.coords,
                opts["size"],
                opts["distance_method"],
                opts["dm_filepath"],
                opts["env_file"],
                opts["gapik_file"],
                opts["symkey_name"],
                ctx.device,
                opts["edge_threshold"],
                opts["edge_method"],
                ctx.indices,
            )

            if should_load_neural and (
                "am" in ctx.pol_strip.split("_")
                or "transgcn" in ctx.pol_strip.split("_")
                or ctx.pol_strip.startswith("am")
            ):
                ctx.model_tup = process_model_data(
                    ctx.coords,
                    ctx.dist_tup[2],
                    ctx.device,
                    opts["vertex_method"],
                    configs,
                    opts["edge_threshold"],
                    opts["edge_method"],
                    opts["area"],
                    opts["waste_type"],
                    adj_matrix,
                )

            if "gamma" in ctx.data_dist:
                ctx.bins = Bins(
                    opts["size"],
                    ctx.data_dir,
                    ctx.data_dist[:-1],
                    area=opts["area"],
                    waste_type=opts["waste_type"],
                    waste_file=opts["waste_filepath"],
                    noise_mean=opts.get("noise_mean", 0.0),
                    noise_variance=opts.get("noise_variance", 0.0),
                )
                gamma_option = int(ctx.policy[-1]) - 1
                ctx.bins.setGammaDistribution(option=gamma_option)
            else:
                ctx.bins = Bins(
                    opts["size"],
                    ctx.data_dir,
                    ctx.data_dist,
                    area=opts["area"],
                    waste_type=opts["waste_type"],
                    waste_file=opts["waste_filepath"],
                    noise_mean=opts.get("noise_mean", 0.0),
                    noise_variance=opts.get("noise_variance", 0.0),
                )

            ctx.cached = [] if opts["cache_regular"] else None
            if opts["stats_filepath"] is not None:
                ctx.bins.set_statistics(opts["stats_filepath"])
            if opts["waste_filepath"] is not None:
                ctx.bins.set_sample_waste(ctx.sample_id)

            ctx.bins.set_indices(ctx.indices)
            ctx.daily_log = {key: [] for key in DAY_METRICS}

        logger.info(f"Initialization complete. Transitioning to RunningState for {ctx.policy} policy.")
        from .running import RunningState

        ctx.transition_to(RunningState())
