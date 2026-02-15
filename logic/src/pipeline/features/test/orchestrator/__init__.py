"""
Simulation Orchestrator.
Refactored into modular components.
"""

import multiprocessing as mp
import os
import signal
import sys

from loguru import logger

import logic.src.constants as udef
from logic.src.pipeline.callbacks.policy_summary import PolicySummaryCallback

# Local imports from modular components
from logic.src.pipeline.features.test.orchestrator.parallel_runner import run_parallel_simulations
from logic.src.pipeline.simulations.repository import load_indices
from logic.src.pipeline.simulations.simulator import (
    display_log_metrics,
    sequential_simulations,
)
from logic.src.utils.logging.log_utils import (
    runs_per_policy,
    send_final_output_to_gui,
)
from logic.src.utils.logging.logger_writer import LoggerWriter, setup_logger_redirection


def simulator_testing(opts, data_size, device):
    """
    Orchestrates the parallel execution of multiple simulation runs.
    """
    log_file = setup_logger_redirection()

    # Capture original stderr for shutdown messages if redirected
    original_stderr = sys.stderr
    if isinstance(original_stderr, LoggerWriter):
        original_stderr = original_stderr.terminal

    # Display policy summary
    if not opts.get("no_progress_bar"):
        PolicySummaryCallback().display(opts)

    # Register immediate shutdown handler for CTRL+C
    def _shutdown_handler(sig, frame):
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
    opts["shared_metrics"] = shared_metrics

    sample_idx_dict = {pol: list(range(opts["n_samples"])) for pol in opts["policies"]}
    if opts["resume"]:
        to_remove = runs_per_policy(  # type: ignore[call-arg, misc]
            udef.ROOT_DIR,  # type: ignore[arg-type]
            opts["days"],
            [opts["size"]],
            opts["output_dir"],
            opts["area"],
            [opts["n_samples"]],
            opts["policies"],
            lock=lock,
        )[0]
        for pol in opts["policies"]:
            if len(to_remove[pol]) > 0:
                sample_idx_dict[pol] = [x for x in sample_idx_dict[pol] if x not in to_remove[pol]]

    sample_idx_ls = [list(val) for val in sample_idx_dict.values()]
    task_count = sum(len(sample_idx) for sample_idx in sample_idx_ls)

    n_cores = opts.get("cpu_cores", 0)
    n_cores = min(task_count, n_cores) if n_cores >= 1 else min(task_count, max(1, mp.cpu_count() - 1))
    if data_size != opts["size"]:
        indices = load_indices(opts["focus_graph"], opts["n_samples"], opts["size"], data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * opts["n_samples"]
    else:
        indices = [None] * opts["n_samples"]

    weights_path = os.path.join(udef.ROOT_DIR, "assets", "model_weights")

    if n_cores > 1:
        log, log_std, _failed_log = run_parallel_simulations(
            opts,
            device,
            indices,
            sample_idx_ls,
            weights_path,
            lock,
            manager,
            n_cores,
            task_count,
            log_file,  # loader_file not used directly here anymore? or we pass it
            original_stderr,
        )
    else:
        print(f"Launching {task_count} WSmart Route simulations on a single CPU core...")
        log, log_std, _failed_log = sequential_simulations(opts, device, indices, sample_idx_ls, weights_path, lock)

    realtime_log_path = os.path.join(
        udef.ROOT_DIR,
        "assets",
        opts["output_dir"],
        str(opts["days"]) + "_days",
        str(opts["area"]) + "_" + str(opts["size"]),
        f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl",
    )
    send_final_output_to_gui(log, log_std, opts["n_samples"], opts["policies"], realtime_log_path)

    if not opts.get("no_progress_bar"):
        display_log_metrics(
            opts["output_dir"],
            opts["size"],
            opts["n_samples"],
            opts["days"],
            opts["area"],
            opts["policies"],
            log,
            log_std,
            lock,
        )
