"""Analysis utilities for experiment and simulation logs.

This module provides tools for aggregating, summarizing, and reporting
on experimental results. It handles loading log directories, computing
mean and standard deviation statistics across multiple runs, and generating
human-readable summaries for different routing policies and graph sizes.

Attributes:
    load_log_dict: Loads file paths for log JSONs grouped by problem size.
    output_stats: Computes and persists statistical summaries (mean/std).
    runs_per_policy: Detects and counts valid samples per solver policy.
    final_simulation_summary: Prints a formatted metric report to the logger.

Example:
    >>> from logic.src.tracking.logging.modules.analysis import output_stats
    >>> output_stats("logs/run1", 100, ["greedy", "alns"], ["profit", "km"])
"""

import json
import os
import statistics
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, cast

from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

import logic.src.constants as udef
from logic.src.utils.io.files import compose_dirpath, read_json


@compose_dirpath
def load_log_dict(
    dir_paths: List[str],
    nsamples: List[int],
    show_incomplete: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Dict[str, str]:
    """Load log file paths from directories keyed by graph size.

    Args:
        dir_paths: List of directory paths to scan.
        nsamples: List of sample counts corresponding to each path.
        show_incomplete: Whether to print warnings for missing samples. Defaults to False.
        lock: Optional thread lock for safe file reading. Defaults to None.

    Returns:
        Dict[str, str]: Map of graph sizes to their corresponding mean log path.
    """
    logs: Dict[str, str] = {}
    for path, ns in zip(dir_paths, nsamples, strict=False):
        gsize = int(os.path.basename(path).split("_")[1])
        logs[f"{gsize}"] = os.path.join(path, f"log_mean_{ns}N.json")
        if show_incomplete and ns > 1:
            log_full = cast(List[Dict[str, Any]], read_json(os.path.join(path, f"log_full_{ns}N.json"), lock))
            counter: Counter[str] = Counter()
            for run in log_full:
                counter.update(run.keys())
            for key, val in dict(counter).items():
                if ns - val > 0:
                    print(f"graph {gsize} incomplete runs: - {key} - {ns - val}")
    return logs


@compose_dirpath
def output_stats(
    dir_path: str,
    nsamples: int,
    policies: List[str],
    keys: List[str],
    sort_log_func: Optional[Any] = None,
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute mean and std statistics for policies and write to JSON.

    Args:
        dir_path: Directory contains the 'log_full' results.
        nsamples: Number of simulation samples in the full log.
        policies: List of policy names to include in statistics.
        keys: Metric names (e.g. 'profit', 'km') to aggregate.
        sort_log_func: Optional function to reorder entries. Defaults to None.
        print_output: Whether to print results to stdout. Defaults to False.
        lock: Optional thread lock for concurrent file access. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Mean dict and Std-dev dict.
    """
    mean_filename = os.path.join(dir_path, f"log_mean_{nsamples}N.json")
    std_filename = os.path.join(dir_path, f"log_std_{nsamples}N.json")
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}, {}
    try:
        if os.path.isfile(mean_filename):
            mean_dit = cast(Dict[str, Any], read_json(mean_filename, lock=None))
            std_dit = cast(Dict[str, Any], read_json(std_filename, lock=None))
        else:
            mean_dit, std_dit = {}, {}
        data = cast(List[Dict[str, Any]], read_json(os.path.join(dir_path, f"log_full_{nsamples}N.json"), lock=None))
        for pol in policies:
            tmp = [list(data[n_id][pol].values()) for n_id in range(nsamples)]
            mean_dit[pol] = {
                key: val for key, val in zip(keys, [*map(statistics.mean, zip(*tmp, strict=False))], strict=False)
            }
            if nsamples > 1:
                std_dit[pol] = {
                    key: val for key, val in zip(keys, [*map(statistics.stdev, zip(*tmp, strict=False))], strict=False)
                }
            else:
                std_dit[pol] = {key: 0.0 for key in keys}

        if sort_log_func:
            mean_dit = sort_log_func(mean_dit)
            std_dit = sort_log_func(std_dit)

        if print_output:
            for pol in mean_dit:
                lg_obj = mean_dit[pol]
                lg_std_obj = std_dit[pol]
                logm = lg_obj.values() if hasattr(lg_obj, "values") else lg_obj
                logs = lg_std_obj.values() if hasattr(lg_std_obj, "values") else lg_std_obj
                print(f"{pol}:")
                for key, m, s in zip(keys, logm, logs, strict=False):
                    print(f"- {key}: {m:.2f} +- {s:.4f}")
        with open(mean_filename, "w") as fp:
            json.dump(mean_dit, fp, indent=True)
        with open(std_filename, "w") as fp:
            json.dump(std_dit, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return mean_dit, std_dit


@compose_dirpath
def runs_per_policy(
    dir_paths: List[str],
    nsamples: List[int],
    policies: List[str],
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> List[Dict[str, List[int]]]:
    """Count runs per policy from full log files.

    Args:
        dir_paths: List of output directories to inspect.
        nsamples: Expected sample counts for each directory.
        policies: List of policy names to search for.
        print_output: Whether to print counts to stdout. Defaults to False.
        lock: Optional thread lock for path scanning. Defaults to None.

    Returns:
        List[Dict[str, List[int]]]: For each path, a map of policy to sample IDs found.
    """
    runs_ls = []
    for path, ns in zip(dir_paths, nsamples, strict=False):
        dit: Dict[str, List[int]] = {pol: [] for pol in policies}
        full_log_path = os.path.join(path, f"log_full_{ns}N.json")
        if not os.path.exists(full_log_path):
            runs_ls.append(dit)
            continue
        data = cast(List[Dict[str, Any]], read_json(full_log_path, lock))
        for id, run_data in enumerate(data):
            for key in dit:
                if key in run_data:
                    dit[key].append(id)
        runs_ls.append(dit)
        if print_output:
            gsize = int(os.path.basename(path).rsplit("_", 1)[1])
            print(f"graph {gsize} #runs per policy:")
            for key, val in dit.items():
                print(f"- {key}: {len(val)} samples: {val}")
    return runs_ls


def final_simulation_summary(log: Dict[str, Any], policy: str, n_samples: int) -> None:
    """Log a final summary of simulation statistics for a policy.

    Args:
        log: The summary log dictionary containing policy keys.
        policy: The specific policy name to summarize.
        n_samples: Total number of samples involved in the run.
    """
    if policy not in log:
        logger.warning(f"Policy {policy} not found in log for summary.")
        return

    # Use the pretty table for a single policy if that's all we have,
    # or if we want to maintain the single-policy logging behavior.
    display_simulation_summary_table(
        {policy: log[policy]},
        title=f"Simulation Summary: [bold cyan]{policy}[/] ({n_samples} samples)",
    )


def display_simulation_summary_table(
    log: Dict[str, Any],
    title: str = "Simulation Summary",
    lock: Optional[Any] = None,
) -> None:
    """Display a pretty comparative table of simulation results for multiple policies.

    Args:
        log: Dictionary mapping policy names to metric dictionaries or lists.
        title: Title for the table. Defaults to "Simulation Summary".
        lock: Optional lock for thread-safe printing. Defaults to None.
    """
    if not log:
        return

    console = Console()
    table = Table(
        title=title,
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        border_style="blue",
        title_style="bold white on blue",
        show_header=True,
        expand=False,
    )

    # Core metrics to display in the table
    display_metrics = [
        ("Profit", "profit"),
        ("Collected", "kg"),
        ("Lost", "kg_lost"),
        ("Col", "ncol"),
        ("Dist", "km"),
        ("Eff", "kg/km"),
        ("Over", "overflows"),
        ("Days", "days"),
        ("Time", "time"),
    ]

    table.add_column("Policy", style="cyan", no_wrap=True)
    for label, _ in display_metrics:
        table.add_column(label, justify="right")

    for pol, stats in log.items():
        row = [pol]
        for _, key in display_metrics:
            if isinstance(stats, dict):
                val = stats.get(key, 0.0)
            elif isinstance(stats, (list, tuple)):
                try:
                    idx = udef.SIM_METRICS.index(key)
                    val = stats[idx]
                except (ValueError, IndexError):
                    val = 0.0
            else:
                val = 0.0

            if key == "profit":
                row.append(f"${val:,.2f}")
            elif key in ["kg", "kg_lost"]:
                row.append(f"{val:,.1f}kg")
            elif key == "km":
                row.append(f"{val:,.1f}km")
            elif key == "kg/km":
                row.append(f"{val:.2f}")
            elif key in ["overflows", "ncol", "days"]:
                color = "red" if key == "overflows" and val > 0 else "white"
                if key == "days":
                    color = "cyan"
                row.append(f"[{color}]{int(val)}[/]")
            elif key == "time":
                row.append(f"{val:.2f}s")
            else:
                row.append(f"{val:.2f}")
        table.add_row(*row)

    if lock:
        with lock:
            console.print(table)
    else:
        console.print(table)
