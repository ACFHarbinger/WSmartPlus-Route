"""initializing.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import initializing
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from logic.src.constants import DAY_METRICS, ROOT_DIR
from logic.src.data.processor import (
    process_data,
    process_model_data,
    setup_basedata,
    setup_dist_path_tup,
)
from logic.src.tracking.logging.log_utils import setup_system_logger
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.configs.setup_env import setup_env
from logic.src.utils.configs.setup_manager import setup_hrl_manager
from logic.src.utils.configs.setup_worker import setup_model

from ..bins import Bins
from ..checkpoints import SimulationCheckpoint
from .base import SimState

if TYPE_CHECKING:
    from .base import SimulationContext


class InitializingState(SimState):
    """State handles the initialization of simulation data (graph, models, etc.)."""

    def handle(self, ctx: SimulationContext) -> None:
        """Handle initialization of simulation state."""
        # Seeding for reproducibility (covers both sequential and parallel runs)
        seed = ctx.cfg.sim.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sim = ctx.cfg.sim

        self._setup_logging_and_dirs(ctx)
        self._load_all_configurations(ctx)

        # Load base data
        data, bins_coordinates, depot = setup_basedata(
            sim.graph.num_loc, ctx.data_dir, sim.graph.area, sim.graph.waste_type
        )
        self._setup_capacities(ctx)

        # Checkpoints
        ctx.checkpoint = SimulationCheckpoint(ctx.results_dir, sim.checkpoint_dir, ctx.pol_name, ctx.sample_id)
        saved_state, last_day = self._load_checkpoint_if_needed(ctx)

        # Setup Models
        self._setup_models(ctx)

        # Restore or Initialize State
        if sim.resume and saved_state is not None:
            self._restore_state(ctx, saved_state, last_day)
        else:
            self._initialize_new_state(ctx, data, bins_coordinates, depot)

        logger.info(f"Initialization complete. Transitioning to RunningState for {ctx.pol_name} policy.")
        from .running import RunningState

        ctx.transition_to(RunningState())

    def _setup_logging_and_dirs(self, ctx):
        log_file = ctx.cfg.tracking.log_file
        if log_file is None:
            from datetime import datetime

            log_file = Path(ctx.cfg.tracking.log_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        setup_system_logger(log_file, ctx.cfg.tracking.log_level)

        # Redirect stderr to the simulation log file for the main process
        from logic.src.tracking.logging.logger_writer import setup_logger_redirection

        setup_logger_redirection(
            log_file=log_file,
            silent=True,
            redirect_stdout=False,  # Keep stdout for terminal progress/output
            redirect_stderr=True,
            echo_to_terminal=True,  # Still echo errors to terminal
        )

        if not os.path.exists(ctx.results_dir):
            os.makedirs(ctx.results_dir)
            logger.info(f"Created results directory: {ctx.results_dir}")
        else:
            logger.info(f"Results directory already exists: {ctx.results_dir}")

    def _load_all_configurations(self, ctx):
        ctx.config = {}
        sim = ctx.cfg.sim
        config_paths = sim.config_path if sim.config_path else None

        if config_paths and hasattr(config_paths, "items") and hasattr(config_paths, "keys"):
            for key, path in config_paths.items():
                try:
                    loaded = path if hasattr(path, "items") else load_config(path)
                    ctx.config[key] = loaded
                    print(f"[INFO] Loaded configuration for '{key}' from {path}")
                except (OSError, ValueError) as e:
                    print(f"[Warning] Failed to load config file {path}: {e}")

        self._load_neural_configs(ctx)

    def _load_neural_configs(self, ctx):
        # Neural if model.name or specific neural parameters are in config
        neural_keywords = ["am", "ptr", "transgcn", "neural", "ddam", "ham", "l2d"]

        # Priority 1: policy_cfg (new structured format)
        model_name = ""
        if isinstance(ctx.pol_cfg, dict) and "model" in ctx.pol_cfg:
            model_name = ctx.pol_cfg["model"].get("name", "").lower()

        neural_cfg_path = os.path.join(ROOT_DIR, "assets", "configs", "policies", "policy_neural.yaml")
        should_load_neural = any(kw in model_name for kw in neural_keywords) or any(
            model_name.startswith(kw) for kw in ["am", "ddam", "ptr"]
        )

        if should_load_neural and os.path.exists(neural_cfg_path):
            try:
                neural_cfg = load_config(neural_cfg_path)
                if neural_cfg:
                    if "neural" in neural_cfg:
                        for pol_key, pol_val in neural_cfg["neural"].items():
                            ctx.config[pol_key] = pol_val
                    else:
                        ctx.config.update(neural_cfg)
                print(f"[INFO] Loaded configuration from {neural_cfg_path}")
            except (OSError, ValueError) as e:
                print(f"[WARNING] Failed to load neural config {neural_cfg_path}: {e}")

    def _setup_capacities(self, ctx):
        from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

        sim = ctx.cfg.sim
        capacities, _, _, _, _ = load_area_and_waste_type_params(sim.graph.area, sim.graph.waste_type)
        ctx.vehicle_capacity = capacities

    def _load_checkpoint_if_needed(self, ctx):
        saved_state, last_day = (None, 0)
        if ctx.cfg.sim.resume:
            saved_state, last_day = ctx.checkpoint.load_state()
            if saved_state is not None and ctx.overall_progress:
                ctx.overall_progress.update(last_day)
        return saved_state, last_day

    def _setup_models(self, ctx):
        model_name = ""
        if isinstance(ctx.pol_cfg, dict) and "model" in ctx.pol_cfg:
            model_name = ctx.pol_cfg["model"].get("name", "").lower()

        sim = ctx.cfg.sim
        should_load_neural = any(kw in model_name for kw in ["am", "ptr", "transgcn", "neural", "ddam"]) or any(
            model_name.startswith(kw) for kw in ["am", "ddam", "ptr"]
        )

        configs = None
        if should_load_neural and ("am" in model_name or "transgcn" in model_name or model_name.startswith("am")):
            # Extract decoding parameters from policy config
            decoding = ctx.pol_cfg.get("decoding", {}) if isinstance(ctx.pol_cfg, dict) else {}
            temp = decoding.get("temperature", 1.0)
            strat = str(decoding.get("strategy", "greedy"))

            model_path_raw = ctx.pol_cfg.get("model_path") if isinstance(ctx.pol_cfg, dict) else None
            model_paths: dict[str, str] = model_path_raw if isinstance(model_path_raw, dict) else {}

            ctx.model_env, configs = setup_model(
                ctx.pol_name,
                ctx.model_weights_path,
                model_paths,
                ctx.device,
                ctx.lock,
                temp,
                strat,
            )
            ctx.hrl_manager = setup_hrl_manager(
                ctx.cfg.sim,
                ctx.device,
                configs,
                policy=ctx.pol_name,
                base_path=ctx.model_weights_path,
                worker_model=ctx.model_env,
            )
            self.configs = configs
        elif "vrpp" in model_name:
            ctx.model_env = setup_env(
                ctx.pol_name,
                sim.server_run,
                sim.gplic_file,
                sim.symkey_name,
                sim.env_file,
            )
            ctx.model_tup = (None, None)
            self.configs = None  # type: ignore[assignment]
        else:
            ctx.model_tup = (None, None)
            self.configs = None  # type: ignore[assignment]

    def _restore_state(self, ctx, saved_state, last_day):
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

    def _initialize_new_state(self, ctx, data, bins_coordinates, depot):
        sim = ctx.cfg.sim

        ctx.new_data, ctx.coords = process_data(data, bins_coordinates, depot, ctx.indices)

        ctx.dist_tup, adj_matrix = setup_dist_path_tup(
            ctx.coords,
            sim.graph.num_loc,
            sim.graph.distance_method,
            sim.graph.dm_filepath,
            sim.env_file,
            sim.gapik_file,
            sim.symkey_name,
            ctx.device,
            sim.graph.edge_threshold,
            sim.graph.edge_method,
            ctx.indices,
        )

        model_name = ""
        if isinstance(ctx.pol_cfg, dict) and "model" in ctx.pol_cfg:
            model_name = ctx.pol_cfg["model"].get("name", "").lower()

        should_load_neural = any(kw in model_name for kw in ["am", "ptr", "transgcn", "neural", "ddam"]) or any(
            model_name.startswith(kw) for kw in ["am", "ddam", "ptr"]
        )

        if should_load_neural and ("am" in model_name or "transgcn" in model_name or model_name.startswith("am")):
            ctx.model_tup = process_model_data(
                ctx.coords,
                ctx.dist_tup[2],
                ctx.device,
                sim.graph.vertex_method,
                ctx.config,
                sim.graph.edge_threshold,
                sim.graph.edge_method,
                sim.graph.area,
                sim.graph.waste_type,
                adj_matrix,
            )

        self._initialize_bins(ctx)

        ctx.cached = [] if sim.cache_regular else None
        if sim.stats_filepath is not None:
            ctx.bins.set_statistics(sim.stats_filepath)
        if ctx.bins.waste_dataset is not None:
            ctx.bins.set_sample_waste(ctx.sample_id)

        ctx.bins.set_indices(ctx.indices)
        ctx.daily_log = {key: [] for key in DAY_METRICS}

    def _initialize_bins(self, ctx):
        sim = ctx.cfg.sim
        data_dist = sim.data_distribution
        if "gamma" in data_dist:
            ctx.bins = Bins(
                sim.graph.num_loc,
                ctx.data_dir,
                data_dist[:-1],
                area=sim.graph.area,
                waste_type=sim.graph.waste_type,
                waste_file=getattr(ctx.cfg, "load_dataset", None),
                noise_mean=sim.noise_mean,
                noise_variance=sim.noise_variance,
                n_days=sim.days,
                n_samples=sim.n_samples,
                seed=ctx.cfg.sim.seed + ctx.sample_id,
            )
            # Try to get gamma option from config (e.g., sim.data_distribution="gamma1" -> alpha=1)
            try:
                gamma_option = int(data_dist[-1]) - 1
            except (ValueError, IndexError):
                gamma_option = 0
            ctx.bins.set_gamma_distribution(option=gamma_option)  # type: ignore[attr-defined]
        else:
            ctx.bins = Bins(
                sim.graph.num_loc,
                ctx.data_dir,
                data_dist,
                area=sim.graph.area,
                waste_type=sim.graph.waste_type,
                waste_file=getattr(ctx.cfg, "load_dataset", None),
                noise_mean=sim.noise_mean,
                noise_variance=sim.noise_variance,
                n_days=sim.days,
                n_samples=sim.n_samples,
                seed=ctx.cfg.sim.seed + ctx.sample_id,
            )
