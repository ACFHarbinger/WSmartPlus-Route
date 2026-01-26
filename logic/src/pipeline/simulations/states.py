"""
State Pattern Implementation for Simulation Lifecycle Management.

This module implements the State Pattern to manage the three phases of
simulation execution: Initialization, Running, and Finishing. Each phase
is encapsulated in a separate state class with well-defined transitions.

Architecture:
    - SimulationContext: Maintains simulation state and manages transitions
    - SimState: Abstract base class for states
    - InitializingState: Setup (data loading, model loading, checkpoints)
    - RunningState: Day-by-day execution loop
    - FinishingState: Result aggregation and persistence

State Transitions:
    InitializingState → RunningState → FinishingState → None (end)

The State Pattern provides:
    - Clear separation of lifecycle phases
    - Explicit state transitions
    - Centralized state management
    - Resumption from checkpoints (re-enter RunningState)

Classes:
    SimulationContext: Context object holding simulation state
    SimState: Abstract state interface
    InitializingState: Initialization phase
    RunningState: Execution phase
    FinishingState: Finalization phase
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from logic.src.constants import DAY_METRICS, ROOT_DIR, SIM_METRICS, TQDM_COLOURS
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.configs.setup_utils import setup_env, setup_hrl_manager, setup_model
from logic.src.utils.logging.log_utils import (
    final_simulation_summary,
    log_to_json,
    setup_system_logger,
)

from .bins import Bins
from .checkpoints import CheckpointError, SimulationCheckpoint, checkpoint_manager
from .day import run_day
from .processor import (
    process_data,
    process_model_data,
    save_matrix_to_excel,
    setup_basedata,
    setup_dist_path_tup,
)


class SimulationContext:
    """
    Context object for the Simulation State Machine.

    Manages the full lifecycle of a single simulation run, coordinating
    transitions between initialization, execution, and finalization states.

    The context holds all simulation state:
        - Configuration (opts, device, policy)
        - Data (bins, coordinates, distance matrices)
        - Models (neural networks, solvers)
        - Runtime (checkpoints, progress bars, locks)
        - Results (logs, execution times)

    State transitions are triggered by calling transition_to(new_state),
    which updates current_state and delegates control to the new state's
    handle() method.

    Attributes:
        opts: Simulation configuration dictionary
        device: torch.device for neural models
        policy: Policy identifier string
        bins: Bins state manager
        model_env: Loaded neural model or solver
        checkpoint: SimulationCheckpoint for persistence
        current_state: Active state object
        result: Final result dictionary (populated on completion)
    """

    def __init__(
        self,
        opts: Dict[str, Any],
        device: torch.device,
        indices: List[int],
        sample_id: int,
        pol_id: int,
        model_weights_path: str,
        variables_dict: Dict[str, Any],
    ):
        """
        Initializes the simulation context with configuration and shared variables.

        Args:
            opts: Dictionary of simulation options.
            device: torch.device for tensor computations.
            indices: List of bin indices for the current sample.
            sample_id: Identifier for the current data sample.
            pol_id: Index of the current policy in opts['policies'].
            model_weights_path: Path to neural model weights.
            variables_dict: Shared variables (locks, counters, progress bars).
        """
        self.opts = opts
        self.device = device
        self.indices = indices
        self.sample_id = sample_id
        self.pol_id = pol_id
        self.model_weights_path = model_weights_path

        # Shared Variables (Global Lock/Counter passed from outside)
        self.lock = variables_dict.get("lock")
        self.counter = variables_dict.get("counter")
        self.overall_progress = variables_dict.get("overall_progress")

        # State Variables
        self.current_state: Optional["SimState"] = None
        self.result: Optional[Dict[str, Any]] = None

        # Simulation Data
        self.data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        self.results_dir = os.path.join(
            ROOT_DIR,
            "assets",
            opts["output_dir"],
            str(opts["days"]) + "_days",
            str(opts["area"]) + "_" + str(opts["size"]),
        )
        self.policy = opts["policies"][pol_id]
        self.pol_strip, self.data_dist = self.policy.rsplit("_", 1)

        # Runtime Data
        self.start_day: int = 1
        self.checkpoint: Optional[SimulationCheckpoint] = None
        self.bins: Optional[Bins] = None
        self.new_data: Optional[Dict[str, Any]] = None
        self.coords: Optional[pd.DataFrame] = None
        self.dist_tup: Optional[Tuple[np.ndarray, Any, Any, Any]] = None
        self.model_tup: Optional[Tuple[Any, ...]] = None
        self.model_env: Optional[Any] = None
        self.hrl_manager: Optional[Any] = None
        self.cached: Optional[List[Any]] = None
        self.run_time: float = 0
        self.overflows: int = 0
        self.current_collection_day: int = 0
        self.daily_log: Optional[Dict[str, List[Any]]] = None
        self.attention_dict: Dict[str, List[Any]] = {}
        self.execution_time: float = 0
        self.tic: float = 0
        self.config: Optional[Dict[str, Any]] = None

        # Progress Bar items
        self.tqdm_pos: int = variables_dict.get("tqdm_pos", 0)
        self.colour: str = TQDM_COLOURS[pol_id % len(TQDM_COLOURS)]

        self.transition_to(InitializingState())

    def transition_to(self, state: Optional["SimState"]) -> None:
        """
        Transitions the context to a new state.

        Args:
            state: The new SimState object to transition to.
        """
        self.current_state = state
        if self.current_state is not None:
            self.current_state.context = self

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Main execution loop that runs the state machine until completion.

        Returns:
            The simulation result dictionary.
        """
        while self.current_state is not None:
            self.current_state.handle(self)
        return self.result

    def get_current_state_tuple(self) -> Tuple[Any, ...]:
        """
        Retrieves a tuple of current simulation variables for checkpointing.

        Returns:
            A tuple containing simulation state data.
        """
        # Helper for checkpoints
        return (
            self.new_data,
            self.coords,
            self.dist_tup,
            None,  # adj_matrix not strictly stored in context unless added
            self.bins,
            self.model_tup,
            self.cached,
            self.overflows,
            self.current_collection_day,
            self.daily_log,
            self.execution_time,
        )


class SimState(ABC):
    """Abstract base class for simulation states."""

    context: SimulationContext

    @abstractmethod
    def handle(self, ctx: SimulationContext) -> None:
        """Handles the current state and returns the next state."""
        pass


class InitializingState(SimState):
    """State handles the initialization of simulation data (graph, models, etc.)."""

    def handle(self, ctx: SimulationContext) -> None:
        """
        Handles the initialization phase of the simulation.

        Loads data, setups models, and initializes indices and distributions.
        Transitions to RunningState upon completion.
        """
        opts = ctx.opts

        # Setup system logger
        setup_system_logger(opts.get("log_file", "logs/simulation.log"), opts.get("log_level", "INFO"))

        # Ensure results directory exists
        if not os.path.exists(ctx.results_dir):
            os.makedirs(ctx.results_dir)
            logger.info(f"Created results directory: {ctx.results_dir}")
        else:
            logger.info(f"Results directory already exists: {ctx.results_dir}")

        # Load Configuration
        # config_path is now expected to be a dict {key: path} or None
        ctx.config = {}
        config_paths = opts.get("config_path")

        if config_paths:
            if isinstance(config_paths, dict):
                for key, path in config_paths.items():
                    try:
                        loaded = load_config(path)
                        # Store in ctx.config under the key, OR merge if we want a unified config?
                        # Strategy: Store all under 'configs' and also try to resolve 'active' config?
                        # For simplicity, let's merge them into ctx.config?
                        # But conflicts might arise.
                        # Better strategy: ctx.config will be the dictionary of {key: loaded_content}
                        # And run_day will pick the right one?
                        # Or checking strict match?

                        # Let's merge everything into ctx.config so lookup is easy (e.g. ctx.config['lookahead']['hgs'])
                        # But wait, if we load two lookahead files, they might clash.
                        # If I have keys 'lookahead' and 'hgs' and policy is 'policy_look_ahead_hgs', merge both.
                        # For now, let's just store the full loaded dict under the cli key if the cli key is explicit.
                        # User said "similar to model_path" where keys are policies.
                        # So ctx.config[key] = load_config(path) is reasonable.

                        ctx.config[key] = loaded
                        print(f"Loaded configuration for '{key}' from {path}")
                    except Exception as e:
                        print(f"Warning: Failed to load config file {path}: {e}")
            else:
                # Legacy fall back if it's a string (though arg_parser should prevent this)
                try:
                    ctx.config = load_config(config_paths)  # Assuming single path
                    print(f"Loaded configuration from {config_paths}")
                except Exception:
                    pass

        # Setup Data
        data, bins_coordinates, depot = setup_basedata(opts["size"], ctx.data_dir, opts["area"], opts["waste_type"])
        ctx.checkpoint = SimulationCheckpoint(ctx.results_dir, opts["checkpoint_dir"], ctx.policy, ctx.sample_id)

        # Resume Logic
        saved_state, last_day = (None, 0)
        if opts["resume"]:
            saved_state, last_day = ctx.checkpoint.load_state()

            # If we resumed and have an overall progress bar, we need to account for skipped days
            # logic/src/pipeline/simulator/simulation.py:289 -> overall_progress.update(completed_days)
            if saved_state is not None and ctx.overall_progress:
                ctx.overall_progress.update(last_day)

        # Model Setup
        configs = None
        if "am" in ctx.pol_strip or "transgcn" in ctx.pol_strip:
            ctx.model_env, configs = setup_model(
                ctx.policy,
                ctx.model_weights_path,
                opts["model_path"],
                ctx.device,
                ctx.lock,  # type: ignore
                opts["temperature"],
                opts["decode_type"],
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

        # Restore or Init State
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

            # Recalculate Distances
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

            if "am" in ctx.pol_strip or "transgcn" in ctx.pol_strip:
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

            # Setup Bins
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
        ctx.transition_to(RunningState())


class RunningState(SimState):
    """State handles the day-by-day simulation loop."""

    def handle(self, ctx: SimulationContext) -> None:
        """
        Handles the day-by-day simulation execution.

        Runs the daily simulation loop, manages checkpoints, and updates progress.
        Transitions to FinishingState after all days are processed.
        """
        opts = ctx.opts

        desc = f"{ctx.policy} #{ctx.sample_id}"
        realtime_log_path = os.path.join(
            ctx.results_dir,
            f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl",
        )

        ctx.tic = time.process_time() + ctx.run_time

        try:
            assert ctx.checkpoint is not None
            with checkpoint_manager(ctx.checkpoint, opts["checkpoint_days"], ctx.get_current_state_tuple) as hook:
                hook.set_timer(ctx.tic)

                iterator = range(ctx.start_day, opts["days"] + 1)
                if not opts["no_progress_bar"]:
                    iterator = tqdm(
                        iterator,
                        desc=desc,
                        position=ctx.tqdm_pos + 1,
                        dynamic_ncols=True,
                        leave=False,
                        colour=ctx.colour,
                    )

                for day in iterator:
                    hook.before_day(day)

                    # Select specific config for this policy if available
                    # Strategy: Check if any key in ctx.config is a substring of ctx.policy
                    current_policy_config = {}

                    # 1. Base config (if any unkeyed config exists or common sections)
                    # For now ctx.config is {key: dict}.

                    # 2. Merge matching configs
                    # If I have keys 'lookahead' and 'hgs' and policy is 'policy_look_ahead_hgs', merge both.
                    for key, cfg in (ctx.config or {}).items():
                        if key in ctx.policy:
                            # Merge cfg into current_policy_config
                            # Simple update for now (shallow merge of top keys)
                            # Deep merge would be better but keeping it simple.
                            current_policy_config.update(cfg)

                    # Prepare SimulationDayContext
                    assert ctx.dist_tup is not None
                    (
                        distance_matrix,
                        paths_between_states,
                        dm_tensor,
                        distancesC,
                    ) = ctx.dist_tup

                    policy_stripped = ctx.policy.rsplit("_", 1)[0]

                    # Robust mapping of stripped policy to canonical PolicyFactory names
                    p_name = policy_stripped
                    if "lookahead" in policy_stripped or "look_ahead" in policy_stripped:
                        p_name = "policy_look_ahead"
                    elif policy_stripped.startswith("am") or "_am" in policy_stripped:
                        p_name = "am_policy"
                    elif "gurobi" in policy_stripped:
                        p_name = "gurobi_vrpp"
                    elif "hexaly" in policy_stripped:
                        p_name = "hexaly_vrpp"
                    elif "last_minute" in policy_stripped:
                        p_name = "policy_last_minute"
                    elif policy_stripped == "regular":
                        p_name = "policy_regular"

                    from logic.src.pipeline.simulations.context import (
                        SimulationDayContext,
                    )

                    day_context = SimulationDayContext(
                        graph_size=opts["size"],
                        full_policy=ctx.policy,
                        policy=policy_stripped,
                        policy_name=p_name,
                        bins=ctx.bins,
                        new_data=ctx.new_data,
                        coords=ctx.coords,
                        run_tsp=opts["run_tsp"],
                        sample_id=ctx.sample_id,
                        overflows=ctx.overflows,
                        day=day,
                        model_env=ctx.model_env,
                        model_ls=ctx.model_tup or (None, None),
                        n_vehicles=opts["n_vehicles"],
                        area=opts["area"],
                        realtime_log_path=realtime_log_path,
                        waste_type=opts["waste_type"],
                        distpath_tup=ctx.dist_tup,
                        distance_matrix=distance_matrix,
                        distancesC=distancesC,
                        paths_between_states=paths_between_states,
                        dm_tensor=dm_tensor,
                        current_collection_day=ctx.current_collection_day,
                        cached=ctx.cached,
                        device=ctx.device,
                        lock=ctx.lock,
                        hrl_manager=ctx.hrl_manager,
                        gate_prob_threshold=opts["gate_prob_threshold"],
                        mask_prob_threshold=opts["mask_prob_threshold"],
                        two_opt_max_iter=opts["two_opt_max_iter"],
                        config=current_policy_config,
                        w_length=opts.get("w_length", 1.0),
                        w_waste=opts.get("w_waste", 1.0),
                        w_overflows=opts.get("w_overflows", 1.0),
                    )

                    day_context = run_day(day_context)

                    ctx.execution_time = time.process_time() - ctx.tic

                    # Update state from context results
                    ctx.new_data = day_context.new_data
                    ctx.coords = day_context.coords
                    ctx.bins = day_context.bins
                    ctx.overflows = day_context.overflows
                    dlog = day_context.daily_log
                    output_dict = day_context.output_dict
                    ctx.cached = day_context.cached

                    if ctx.counter:
                        with ctx.counter.get_lock():
                            ctx.counter.value += 1

                    if "am" in ctx.pol_strip or "transgcn" in ctx.pol_strip:
                        if ctx.pol_strip not in ctx.attention_dict:
                            ctx.attention_dict[ctx.pol_strip] = []
                        ctx.attention_dict[ctx.pol_strip].append(output_dict)

                    assert ctx.daily_log is not None
                    if dlog is not None:
                        for key, val in dlog.items():
                            ctx.daily_log[key].append(val)

                    hook.after_day(ctx.execution_time)

                    if ctx.overall_progress:
                        ctx.overall_progress.update(1)

            logger.info(f"Simulation loop complete. Processed {opts['days']} days.")
            ctx.transition_to(FinishingState())

        except CheckpointError as e:
            ctx.result = e.error_result
            if opts.get("print_output") and ctx.result:
                final_simulation_summary(ctx.result, ctx.policy, opts["n_samples"])
            ctx.transition_to(None)  # End
        except Exception as e:
            raise e


class FinishingState(SimState):
    """State handles final result aggregation and persistence."""

    def handle(self, ctx: SimulationContext) -> None:
        """
        Handles the finalization phase of the simulation.

        Aggregates results, saves logs to disk, and performs cleanup.
        Sets the final result in the context and ends the state machine.
        """
        ctx = self.context
        opts = ctx.opts
        assert ctx.bins is not None

        ctx.execution_time = time.process_time() - ctx.tic

        lg = [
            np.sum(ctx.bins.inoverflow),
            np.sum(ctx.bins.collected),
            np.sum(ctx.bins.ncollections),
            np.sum(ctx.bins.lost),
            ctx.bins.travel,
            (np.sum(ctx.bins.collected) / ctx.bins.travel if ctx.bins.travel > 0 else 0.0),
            np.sum(ctx.bins.inoverflow) - np.sum(ctx.bins.collected) + ctx.bins.travel,
            ctx.bins.profit,
            ctx.bins.ndays,
            ctx.execution_time,
        ]

        daily_log_path = os.path.join(
            ctx.results_dir,
            f"daily_{opts['data_distribution']}_{opts['n_samples']}N.json",
        )

        if opts["n_samples"] > 1:
            log_path = os.path.join(ctx.results_dir, f"log_full_{opts['n_samples']}N.json")
            log_to_json(
                log_path,
                SIM_METRICS,
                {ctx.policy: lg},
                sample_id=ctx.sample_id,
                lock=ctx.lock,
            )
            assert ctx.daily_log is not None
            log_to_json(
                daily_log_path,
                DAY_METRICS,
                {f"{ctx.pol_strip} #{ctx.sample_id}": ctx.daily_log.values()},
                lock=ctx.lock,
            )
        else:
            log_path = os.path.join(ctx.results_dir, f"log_mean_{opts['n_samples']}N.json")
            log_to_json(log_path, SIM_METRICS, {ctx.policy: lg}, lock=ctx.lock)
            assert ctx.daily_log is not None
            log_to_json(
                daily_log_path,
                DAY_METRICS,
                {ctx.pol_strip: ctx.daily_log.values()},
                lock=ctx.lock,
            )

        # Save fill history
        save_matrix_to_excel(
            ctx.bins.get_fill_history(),
            ctx.results_dir,
            opts["seed"],
            opts["data_distribution"],
            ctx.policy,
            ctx.sample_id,
        )

        if ctx.checkpoint:
            ctx.checkpoint.clear()

        ctx.result = {ctx.policy: lg, "success": True}

        if opts.get("print_output"):
            from logic.src.utils.logging.log_utils import final_simulation_summary

            final_simulation_summary({ctx.policy: lg}, ctx.policy, opts["n_samples"])

        self.context.transition_to(None)  # End
