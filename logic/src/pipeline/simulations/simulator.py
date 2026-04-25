"""
High-Level Simulation Orchestration and Parallelization.

This module provides the top-level entry points for running simulations:
single runs, sequential batches, and parallel execution via multiprocessing.

Responsibilities:
    - Worker process initialization (shared locks/counters)
    - Sequential simulation loops with progress tracking
    - Aggregate statistics computation (mean, std dev)
    - JSON logging and result persistence

Architecture:
    - single_simulation: Worker function for one (policy, sample) pair
    - sequential_simulations: Runs multiple samples sequentially
    - init_single_sim_worker: Multiprocessing initializer
    - display_log_metrics: Pretty-prints results to terminal

The simulation execution delegates to SimulationContext (states.py)
which manages the state machine for each run.

Functions:
    init_single_sim_worker: Initialize globals for parallel workers
    single_simulation: Execute one simulation run
    sequential_simulations: Execute batch of runs sequentially
    display_log_metrics: Format and display aggregated results

Attributes:
    _lock: Multiprocessing Lock for file I/O synchronization.
    _counter: Multiprocessing Value for overall progress tracking.
    _shared_metrics: Multiprocessing Manager.dict for real-time stats.

Example:
    >>> from logic.src.pipeline.simulations.simulator import sequential_simulations
    >>> # Run a sequential batch (pseudo-code)
    >>> # log, log_std, failed = sequential_simulations(cfg, device, indices, samples, weights, lock)
"""

from __future__ import annotations

import contextlib
import os
import random
import statistics
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

from logic.src.constants import ROOT_DIR, SIM_METRICS
from logic.src.pipeline.simulations.repository import set_repository_from_path
from logic.src.tracking.logging.log_utils import log_to_json, output_stats
from logic.src.tracking.logging.logger_writer import setup_logger_redirection
from logic.src.utils.configs.setup_utils import get_pol_name

try:
    import logic.src.tracking as wst
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    wst = None  # type: ignore[assignment]
    get_active_run = None  # type: ignore[assignment]

from .checkpoints import CheckpointError
from .states import SimulationContext

if TYPE_CHECKING:
    from logic.src.configs import Config


# Create global variables/placeholders for parallel workers
_lock: Optional[Any] = None
_counter: Optional[Any] = None
_shared_metrics: Optional[Any] = None


def init_single_sim_worker(
    lock_from_main: Any,
    counter_from_main: Any,
    shared_metrics_from_main: Any = None,
    log_file: Optional[str] = None,
    cfg: Optional[Config] = None,
    tracking_uri: Optional[str] = None,
    tracking_run_id: Optional[str] = None,
) -> None:
    """
    Initializes global shared resources for parallel simulation workers.

    This function is called by the multiprocessing Pool during its worker
    initialization phase. It sets process-global variables to ensure all
    simulations can access the shared resources.

    Args:
        lock_from_main: Multiprocessing Lock for file I/O synchronization.
        counter_from_main: Multiprocessing Value for overall progress tracking.
        shared_metrics_from_main: Multiprocessing Manager.dict for real-time stats.
        log_file: Path to the log file for redirection.
        cfg: Root configuration (used to initialize the repository).
        tracking_uri: WSTracker database directory from the parent process.
        tracking_run_id: UUID of the parent run to attach to.
    """
    global _lock
    global _counter
    global _shared_metrics
    _lock = lock_from_main
    _counter = counter_from_main
    _shared_metrics = shared_metrics_from_main
    # Seeding for reproducibility in parallel workers
    if cfg is not None:
        seed = cfg.sim.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Initialize simulation repository in the worker process
    if cfg is not None:
        _initialize_worker_repository(cfg)

    # Attach to the parent tracking run so get_active_run() works in this process
    if tracking_uri and tracking_run_id and wst is not None:
        wst.init_worker(tracking_uri=tracking_uri, run_id=tracking_run_id)

    # Setup logger redirection in the worker process (silent=True to avoid garbling dashboard)
    if log_file:
        setup_logger_redirection(log_file, silent=True)


def _initialize_worker_repository(cfg: Config) -> None:
    """Initialize the singleton repository instance in a worker process.

    Args:
        cfg: The configuration object.
    """
    load_ds = cfg.load_dataset
    if load_ds is not None and set_repository_from_path(str(load_ds)):
        return

    set_repository_from_path(str(ROOT_DIR))


def display_log_metrics(
    output_dir: str,
    size: int,
    n_samples: int,
    days: int,
    area: str,
    policies: List[str],
    log: Dict[str, Union[List[float], Dict[str, float]]],
    log_std: Optional[Dict[str, Union[List[float], Dict[str, float]]]] = None,
    lock: Optional[Any] = None,
) -> None:
    """
    Pretty-prints aggregated simulation results to the console.

    Args:
        output_dir: Base directory for results.
        size: Graph size (number of bins).
        n_samples: Number of samples/seeds processed.
        days: Simulation duration in days.
        area: Geographic area name.
        policies: List of policy names.
        log: Dict mapping policy name to mean metrics list.
        log_std: Dict mapping policy name to std dev metrics list (optional).
        lock: Thread/process lock for safe printing/logging.
    """
    if n_samples > 1 and log_std is not None:
        abs_output_dir = os.path.join(
            ROOT_DIR,
            "assets",
            output_dir,
            str(days) + "_days",
            str(area) + "_" + str(size),
        )
        log_to_json(
            os.path.join(abs_output_dir, f"log_mean_{n_samples}N.json"),
            SIM_METRICS,
            log,
            sample_id=None,
            lock=lock,
        )
        log_to_json(
            os.path.join(abs_output_dir, f"log_std_{n_samples}N.json"),
            SIM_METRICS,
            log_std,
            sample_id=None,
            lock=lock,
        )
        for lg, lg_std, pol in zip(log.values(), log_std.values(), log.keys(), strict=False):
            logm = lg.values() if isinstance(lg, dict) else lg
            logs = lg_std.values() if isinstance(lg_std, dict) else lg_std
            tmp_lg = [(str(x), str(y)) for x, y in zip(logm, logs, strict=False)]
            print(f"\n{pol} log:")
            for (x, y), key in zip(tmp_lg, SIM_METRICS, strict=False):
                print(f"- {key} value: {x[: x.find('.') + 3]} +- {y[: y.find('.') + 5]}")
    else:
        for pol, lg in log.items():
            print(f"\n{pol} log:")
            lg_vals = lg.values() if isinstance(lg, dict) else lg
            for key, val in zip(SIM_METRICS, lg_vals, strict=False):
                print(f"- {key}: {val}")


def single_simulation(
    cfg: Union[Config, DictConfig],
    device: torch.device,
    indices: Any,
    sample_id: int,
    pol_id: int,
    model_weights_path: Optional[str],
    n_cores: int,
) -> Dict[str, Any]:
    """
    Executes a single simulation run for one (policy, sample) pair.

    Worker function for parallel execution via multiprocessing. Retrieves
    shared locks/counters from globals and delegates to SimulationContext.

    Args:
        cfg: Root configuration object.
        device: torch.device for neural models.
        indices: Bin subset indices for this sample.
        sample_id: Sample/seed identifier.
        pol_id: Policy index in cfg.sim.full_policies.
        model_weights_path: Path to pretrained model weights.
        n_cores: Number of CPU cores (for progress bar positioning).

    Returns:
        Result dictionary from SimulationContext.run()
    """
    # Retrieve the shared objects via the global variables initialized by init_worker
    global _lock
    global _counter
    global _shared_metrics

    sim = cfg.sim
    policies = sim.full_policies

    try:
        sys.stdout.flush()
        variables_dict: Dict[str, Any] = {
            "lock": _lock,
            "counter": _counter,
            "shared_metrics": _shared_metrics,
            "tqdm_pos": os.getpid() % n_cores,
        }

        context = SimulationContext(cfg, device, indices, sample_id, pol_id, model_weights_path, variables_dict)
        res: Optional[Dict[str, Any]] = context.run()
        # Aggregate execution result
        if res and "success" in res and res["success"]:
            # If successful, extract the policy name to return the correct dictionary key
            pol_name = get_pol_name(policies[pol_id])
            if pol_name in res:
                return {pol_name: res[pol_name], "success": True, "sample_id": sample_id}

        print(f"\n[INFO] Finished simulation for policy {pol_id} and sample {sample_id}")
        return res or {"error": "Unknown error", "policy": "unknown", "sample_id": sample_id, "success": False}
    except BaseException as e:
        # Report to redirected stderr so it's captured in simulation log files
        print(
            f"\n[CRITICAL ERROR] in single_simulation (Sample {sample_id}, Policy {pol_id}): {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)

        # Also print a pointer to the log file on the REAL stderr so it's visible even during failure
        if hasattr(sys.stderr, "filename"):
            print(f"Detailed traceback available in: {sys.stderr.filename}", file=sys.__stderr__ or sys.stderr)

        return {"policy": "unknown", "sample_id": sample_id, "error": str(e), "success": False}


# Internal counter for display
class SimpleCounter:
    """
    Simple counter for display.

    Attributes:
        value: The current value of the counter.
    """

    def __init__(self, val=0):
        """
        Initializes the simple counter.

        Args:
            val: Initial value. Defaults to 0.
        """
        self.value = val

    def update(self, n=1):
        """
        Updates the counter by adding n.

        Args:
            n: Value to add.
        """
        self.value += n


# Progress bar wrapper to maintain compatibility with RunningState which calls .update(1)
class ProgressUpdater:
    """
    Wrapper for the progress bar to maintain compatibility with RunningState.

    Calls the process_display_updates function to update the progress bar.

    Attributes:
        display: The display object to update (SimulationMonitor).
        shared_metrics: The shared metrics dictionary to update.
        log_tmp: The temporary log dictionary to update.
        last_reported_days: The last reported days dictionary to update.
        policy_names: The list of policy names.
        loop_tic: The loop start timestamp.
        counter: The counter object (SimpleCounter).
        process_display_updates: The function to update the display.
    """

    def __init__(self, display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter):
        """
        Initializes the progress updater wrapper.

        Args:
            display: The display object to update.
            shared_metrics: The shared metrics dictionary to update.
            log_tmp: The temporary log dictionary to update.
            last_reported_days: The last reported days dictionary to update.
            policy_names: The list of policy names.
            loop_tic: The loop tic value.
            counter: The counter object.
        """
        self.display = display
        self.shared_metrics = shared_metrics
        self.log_tmp = log_tmp
        self.last_reported_days = last_reported_days
        self.policy_names = policy_names
        self.loop_tic = loop_tic
        self.counter = counter

        from logic.src.pipeline.features.test.orchestrator.monitor import process_display_updates

        self.process_display_updates = process_display_updates

    def set_values(self, display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter) -> None:
        """
        Set the values for the progress bar.

        Args:
            display: Display object to update.
            shared_metrics: Shared metrics dictionary to update.
            log_tmp: Temporary log dictionary to update.
            last_reported_days: Last reported days dictionary to update.
            policy_names: List of policy names.
            loop_tic: Loop tic value.
            counter: Counter object.
        """
        self.display = display
        self.shared_metrics = shared_metrics
        self.log_tmp = log_tmp
        self.last_reported_days = last_reported_days
        self.policy_names = policy_names
        self.loop_tic = loop_tic
        self.counter = counter

    def update(self, n=1) -> None:
        """
        Update the progress bar.

        Args:
            n: Number of steps to update the progress bar.
        """
        if self.display:
            self.process_display_updates(
                self.display,
                self.shared_metrics,
                self.log_tmp,
                self.last_reported_days,
                self.policy_names,
                self.loop_tic,
                self.counter,
            )


def sequential_simulations(  # noqa: C901
    cfg: Config,
    device: torch.device,
    indices_ls: List[Any],
    sample_idx_ls: List[List[int]],
    model_weights_path: str,
    lock: Optional[Any],
    shared_metrics: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes multiple simulation runs sequentially with aggregation.

    Args:
        cfg: Root configuration object.
        device: torch.device for neural models.
        indices_ls: List of bin index arrays (one per sample).
        sample_idx_ls: List of sample IDs for each policy.
        model_weights_path: Path to pretrained model weights.
        lock: Threading lock for file I/O synchronization.
        shared_metrics: Optional shared dict for real-time metrics.

    Returns:
        Tuple containing:
            - log: Dict[policy] -> List[mean_metrics]
            - log_std: Dict[policy] -> List[std_metrics] (None if n_samples=1)
            - failed_log: List of error result dictionaries
    """
    sim = cfg.sim
    policies = sim.full_policies

    log: Dict[str, Any] = {}
    failed_log: List[Dict[str, Any]] = []
    log_std: Optional[Dict[str, Any]] = {}
    log_full: Dict[str, List[List[float]]] = {get_pol_name(policy): [] for policy in policies}

    counter = SimpleCounter()
    loop_tic = time.time()
    last_reported_days = {p: 0 for p in policies}
    log_tmp: Dict[str, Any] = {p: {} for p in policies}
    policy_names = [get_pol_name(p) for p in policies]

    from logic.src.pipeline.features.test.orchestrator.monitor import initialize_simulation_display

    display = (
        initialize_simulation_display(policy_names, sim.n_samples, sim.days)
        if not cfg.tracking.no_progress_bar
        else None
    )

    if shared_metrics is None:
        shared_metrics = {}

    overall_progress = ProgressUpdater(
        display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter
    )
    results_dir = os.path.join(
        ROOT_DIR,
        "assets",
        sim.output_dir,
        f"{sim.days}_days",
        f"{sim.graph.area}_{sim.graph.num_loc}",
    )

    for pol_id, policy in enumerate(policies):
        pol_name = get_pol_name(policy)
        for sample_id in sample_idx_ls[pol_id]:
            try:
                variables_dict: Dict[str, Any] = {
                    "lock": lock,
                    "overall_progress": overall_progress,
                    "shared_metrics": shared_metrics,
                }

                context = SimulationContext(
                    cfg,
                    device,
                    indices_ls[sample_id],
                    sample_id,
                    pol_id,
                    model_weights_path,
                    variables_dict,
                )
                result_dict = context.run()

                print(f"\n[INFO] Finished simulation for policy {pol_name} and sample {sample_id}")

                # Aggregate execution result
                if result_dict and "success" in result_dict and result_dict["success"]:
                    lg = result_dict[pol_name]

                    # Always append to log_full to support uniform aggregation logic
                    log_full[pol_name].append(lg)
                    log[pol_name] = lg

                # Final update for this sample
                if display:
                    overall_progress.set_values(
                        display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter
                    )
                    overall_progress.update()

            except BaseException as e:
                # Report to redirected stderr so it's captured in simulation log files
                print(
                    f"\n[CRITICAL ERROR] in sequential_simulations (Sample {sample_id}, Policy {pol_id}): {e}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)

                # Also print a pointer to the log file on the REAL stderr
                if hasattr(sys.stderr, "filename"):
                    print(f"Detailed traceback available in: {sys.stderr.filename}", file=sys.__stderr__ or sys.stderr)
            except CheckpointError:
                # Skip broken checkpoints
                pass

        if sim.n_samples >= 1:
            if sim.resume:
                res_log, res_std = output_stats(
                    results_dir,
                    sim.n_samples,
                    [pol_name],
                    SIM_METRICS,
                    lock=lock,
                )
                if res_log:
                    log.update(res_log)
                if res_std and log_std is not None:
                    log_std.update(res_std)
            elif pol_name in log_full and log_full[pol_name]:
                log[pol_name] = [*map(statistics.mean, zip(*log_full[pol_name], strict=False))]
                if len(log_full[pol_name]) > 1:
                    if log_std is not None:
                        log_std[pol_name] = [*map(statistics.stdev, zip(*log_full[pol_name], strict=False))]
                elif log_std is not None:
                    log_std[pol_name] = [0.0] * len(log[pol_name])

                log_to_json(
                    os.path.join(results_dir, f"log_mean_{sim.n_samples}N.json"),
                    SIM_METRICS,
                    {pol_name: log[pol_name]},
                    lock=lock,
                )
                if log_std is not None:
                    log_to_json(
                        os.path.join(results_dir, f"log_std_{sim.n_samples}N.json"),
                        SIM_METRICS,
                        {pol_name: log_std[pol_name]},
                        lock=lock,
                    )

    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_metric("sim/n_failed_runs", float(len(failed_log)))
            for pol_name_k, metrics in log.items():
                if isinstance(metrics, (list, tuple)):
                    for metric_name, val in zip(SIM_METRICS, metrics, strict=False):
                        run.log_metric(f"sim/{pol_name_k}/{metric_name}", float(val))
                if log_std is not None and pol_name_k in log_std:
                    std_metrics = log_std[pol_name_k]
                    if isinstance(std_metrics, (list, tuple)):
                        for metric_name, val in zip(SIM_METRICS, std_metrics, strict=False):
                            run.log_metric(f"sim/{pol_name_k}/{metric_name}_std", float(val))

    return log, log_std, failed_log
