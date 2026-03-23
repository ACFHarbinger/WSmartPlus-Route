"""
Simulation Orchestrator.
Refactored into modular components.
"""

import multiprocessing as mp
import os
import signal
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from omegaconf import OmegaConf

import logic.src.constants as udef
import logic.src.tracking as wst
from logic.src.configs import Config
from logic.src.pipeline.callbacks import PolicySummaryCallback

# Local imports from modular components
from logic.src.pipeline.features.test.orchestrator.parallel_runner import run_parallel_simulations
from logic.src.pipeline.simulations.repository import load_indices
from logic.src.pipeline.simulations.simulator import (
    display_log_metrics,
    sequential_simulations,
)
from logic.src.tracking.logging.log_utils import (
    output_stats,
    runs_per_policy,
    send_final_output_to_gui,
)
from logic.src.tracking.logging.logger_writer import LoggerWriter, setup_logger_redirection


def simulator_testing(cfg: Config, data_size: int, device: Any) -> None:
    """
    Orchestrates the parallel execution of multiple simulation runs.

    Args:
        cfg: Root configuration with ``cfg.sim`` containing simulation params.
        data_size: Number of available data points for the area.
        device: Torch device for neural models.
    """
    sim = cfg.sim
    log_file = setup_logger_redirection()
    if OmegaConf.is_config(cfg):
        OmegaConf.set_struct(cfg, False)  # type: ignore[arg-type]
        cfg.tracking.log_file = log_file
        OmegaConf.set_struct(cfg, True)  # type: ignore[arg-type]
    else:
        cfg.tracking.log_file = log_file

    # Capture original stderr for shutdown messages if redirected
    original_stderr = sys.stderr
    if isinstance(original_stderr, LoggerWriter):
        original_stderr = original_stderr.terminal

    # Display policy summary
    if not cfg.tracking.no_progress_bar:
        PolicySummaryCallback().display(cfg)

    # Register immediate shutdown handler for CTRL+C
    def _shutdown_handler(sig: int, frame: Any) -> None:
        original_stderr.write("\n\n[WARNING] Caught CTRL+C (SIGINT). Forcing immediate shutdown...\n")
        original_stderr.flush()
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            os._exit(1)

    signal.signal(signal.SIGINT, _shutdown_handler)

    manager = mp.Manager()
    lock = manager.Lock()
    shared_metrics = manager.dict()

    policies = sim.full_policies
    sample_idx_dict: Dict[str, List[int]] = {pol: list(range(sim.n_samples)) for pol in policies}
    if sim.resume:
        runs_per_policy_any: Any = runs_per_policy
        to_remove = runs_per_policy_any(
            home_dir=str(udef.ROOT_DIR),
            ndays=sim.days,
            nbins=[sim.graph.num_loc],
            output_dir=sim.output_dir,
            area=sim.graph.area,
            nsamples=[sim.n_samples],
            policies=policies,
            lock=lock,
        )[0]
        for pol in policies:
            if len(to_remove[pol]) > 0:
                sample_idx_dict[pol] = [x for x in sample_idx_dict[pol] if x not in to_remove[pol]]

    sample_idx_ls = [list(val) for val in sample_idx_dict.values()]
    task_count = sum(len(sample_idx) for sample_idx in sample_idx_ls)

    n_cores = sim.cpu_cores
    n_cores = min(task_count, n_cores) if n_cores >= 1 else min(task_count, max(1, mp.cpu_count() - 1))
    if data_size != sim.graph.num_loc:
        indices = load_indices(sim.graph.focus_graph, sim.n_samples, sim.graph.num_loc, data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * sim.n_samples
    else:
        indices: List[Optional[Any]] = [None] * sim.n_samples  # type: ignore[no-redef]

    weights_path = os.path.join(udef.ROOT_DIR, "assets", "model_weights")

    # Extract active tracking context so parallel workers can attach to the
    # parent run.  Sequential workers run in the same process and already
    # inherit get_active_run() directly.
    _active_run = wst.get_active_run()
    _tracker = wst.get_tracker()
    tracking_uri: Optional[str] = _tracker.tracking_uri if _tracker is not None else None
    tracking_run_id: Optional[str] = _active_run.run_id if _active_run is not None else None

    if n_cores > 1:
        log, log_std, _failed_log = run_parallel_simulations(
            cfg,
            device,
            indices,
            sample_idx_ls,
            weights_path,
            lock,
            manager,
            n_cores,
            task_count,
            log_file,
            original_stderr,
            shared_metrics,
            tracking_uri=tracking_uri,
            tracking_run_id=tracking_run_id,
        )
    else:
        log, log_std, _failed_log = sequential_simulations(
            cfg, device, indices, sample_idx_ls, weights_path, lock, shared_metrics
        )

    realtime_log_path = os.path.join(
        udef.ROOT_DIR,
        "assets",
        sim.output_dir,
        f"{sim.days}_days",
        f"{sim.graph.area}_{sim.graph.num_loc}",
        f"log_realtime_{sim.data_distribution}_{sim.n_samples}N.jsonl",
    )
    send_final_output_to_gui(log, log_std, sim.n_samples, policies, realtime_log_path)

    if not cfg.tracking.no_progress_bar:
        display_log_metrics(
            sim.output_dir,
            sim.graph.num_loc,
            sim.n_samples,
            sim.days,
            sim.graph.area,
            policies,
            log,
            log_std,
            lock,
        )
