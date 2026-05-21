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

import contextlib
import json
import os
import statistics
import threading
import traceback
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
        # Return the directory path so callers can locate per-policy log files
        logs[f"{gsize}"] = path
        if show_incomplete and ns > 1:
            import glob as _glob

            pol_files = _glob.glob(os.path.join(path, f"log_*_{ns}N.json"))
            for pol_file in pol_files:
                pol_data = cast(Dict[str, Any], read_json(pol_file, lock))
                if isinstance(pol_data, dict) and "samples" in pol_data:
                    n_recorded = len(pol_data["samples"])
                    if ns - n_recorded > 0:
                        pol_name = os.path.basename(pol_file).replace(f"_{ns}N.json", "")[4:]
                        print(f"graph {gsize} incomplete runs: - {pol_name} - {ns - n_recorded}")
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

    Reads per-sample data from ``log_{pol}_{nsamples}N.json`` files (``samples``
    section) and writes back the computed ``mean`` and ``std`` sections.

    Args:
        dir_path: Directory containing per-policy log JSON files.
        nsamples: Number of simulation samples in the full log.
        policies: List of policy names to include in statistics.
        keys: Metric names (e.g. 'profit', 'km') to aggregate.
        sort_log_func: Optional function to reorder entries. Defaults to None.
        print_output: Whether to print results to stdout. Defaults to False.
        lock: Optional thread lock for concurrent file access. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Mean dict and Std-dev dict.
    """
    from logic.src.tracking.logging.modules.storage import update_policy_log_section

    mean_dit: Dict[str, Any] = {}
    std_dit: Dict[str, Any] = {}

    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}, {}
    try:
        for pol in policies:
            pol_log_path = os.path.join(dir_path, f"log_{pol}_{nsamples}N.json")
            if not os.path.isfile(pol_log_path):
                continue
            try:
                pol_data = cast(Dict[str, Any], read_json(pol_log_path, lock=None))
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(pol_data, dict) or "samples" not in pol_data:
                continue
            samples_section = pol_data["samples"]
            # samples_section is {str(sample_id): {metric: val}}
            sample_values = [list(v.values()) for v in samples_section.values() if isinstance(v, dict)]
            if not sample_values:
                continue
            mean_vals = [*map(statistics.mean, zip(*sample_values, strict=False))]
            mean_dit[pol] = dict(zip(keys, mean_vals, strict=False))
            if len(sample_values) > 1:
                std_vals = [*map(statistics.stdev, zip(*sample_values, strict=False))]
                std_dit[pol] = dict(zip(keys, std_vals, strict=False))
            else:
                std_dit[pol] = {key: 0.0 for key in keys}

            update_policy_log_section(pol_log_path, "mean", mean_dit[pol], lock=None)
            update_policy_log_section(pol_log_path, "std", std_dit[pol], lock=None)

        if sort_log_func:
            mean_dit = sort_log_func(mean_dit)
            std_dit = sort_log_func(std_dit)

        if print_output:
            for pol in mean_dit:
                print(f"{pol}:")
                for key in keys:
                    m = mean_dit[pol].get(key, 0.0)
                    s = std_dit[pol].get(key, 0.0) if pol in std_dit else 0.0
                    print(f"- {key}: {m:.2f} +- {s:.4f}")
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
        for pol in policies:
            pol_log_path = os.path.join(path, f"log_{pol}_{ns}N.json")
            if not os.path.exists(pol_log_path):
                continue
            try:
                pol_data = cast(Dict[str, Any], read_json(pol_log_path, lock))
            except Exception:
                continue
            if isinstance(pol_data, dict) and "samples" in pol_data:
                for sample_id_str in pol_data["samples"]:
                    with contextlib.suppress(ValueError, TypeError):
                        dit[pol].append(int(sample_id_str))
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


def display_simulation_summary_table(  # noqa: C901
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


def display_per_policy_simulation_summary(  # noqa: C901
    pol_name: str,
    sample_id: int,
    aggregate_metrics: List[float],
    daily_log: Dict[str, List[Any]],
    title_prefix: str = "Results for",
    lock: Optional[Any] = None,
) -> None:
    """Display detailed results for a single policy simulation run.

    Args:
        pol_name: Name of the policy.
        sample_id: ID of the sample/seed.
        aggregate_metrics: List of aggregate metrics for the entire run.
        daily_log: Dictionary of daily metrics.
        title_prefix: Prefix for the table titles.
        lock: Optional lock for thread-safe printing.
    """
    console = Console()

    # 1. STATISTICS SUMMARY TABLE
    summary_title = f"{title_prefix} [bold cyan]{pol_name}[/] (Sample #{sample_id}) - [yellow]Stats Summary[/]"

    def _print_tables():  # noqa: C901
        try:
            # Using rule for better separation
            console.rule(f"[bold white on blue] {summary_title} [/]")
            display_simulation_summary_table({pol_name: aggregate_metrics}, title=None, lock=None)

            # 2. DAILY PERFORMANCE TABLE (Filtered)
            daily_title = f"{title_prefix} [bold cyan]{pol_name}[/] (Sample #{sample_id}) - [green]Daily Routes[/]"

            table = Table(
                box=box.MINIMAL_DOUBLE_HEAD,
                header_style="bold magenta",
                border_style="green",
                show_header=True,
                expand=False,
            )

            # Define columns — 'day' is now derived from index (1-based), not stored
            columns = [
                ("Day", "day", "cyan"),
                ("Mandatory", "mandatory_nodes", "yellow"),
                ("Tour", "tour", "white"),
                ("Profit", "profit", "green"),
                ("KG", "kg", "white"),
                ("Lost", "kg_lost", "red"),
                ("Col", "ncol", "white"),
                ("Dist", "km", "white"),
                ("Eff", "kg/km", "yellow"),
                ("Over", "overflows", "red"),
            ]

            for label, _, style in columns:
                table.add_column(label, style=style, justify="right" if label not in ("Tour", "Mandatory") else "left")

            # Use km list length as the authoritative iteration count
            kms = daily_log.get("km", [])

            has_active_days = False
            for i, km_val in enumerate(kms):
                # Only show days where a route was performed (km > 0)
                if km_val > 0:
                    has_active_days = True
                    row = []
                    for _, key, _ in columns:
                        if key == "day" or key is None:
                            # "Day" column — synthetic 1-based index (not stored in daily_log)
                            row.append(str(i + 1))
                            continue
                        vals = daily_log.get(key, [])
                        val = vals[i] if i < len(vals) else None

                        if key == "mandatory_nodes":
                            mand_str = str(val) if val else "[]"
                            if len(mand_str) > 40:
                                mand_str = mand_str[:37] + "..."
                            row.append(mand_str)
                        elif key == "tour":
                            tour_str = str(val) if val is not None else "[]"
                            if len(tour_str) > 50:
                                tour_str = tour_str[:47] + "..."
                            row.append(tour_str)
                        elif key == "profit":
                            if val is not None:
                                color = "green" if val > 0 else "red"
                                row.append(f"[{color}]${val:,.2f}[/]")
                            else:
                                row.append("-")
                        elif key in ["kg", "kg_lost"]:
                            row.append(f"{val:,.1f}kg" if val is not None else "0.0kg")
                        elif key == "km":
                            row.append(f"{val:,.1f}km" if val is not None else "0.0km")
                        elif key == "kg/km":
                            row.append(f"{val:.2f}" if val is not None else "0.00")
                        elif key == "overflows":
                            if val is not None:
                                color = "red" if val > 0 else "white"
                                row.append(f"[{color}]{int(val)}[/]")
                            else:
                                row.append("0")
                        elif key == "ncol":
                            row.append(f"{int(val)}" if val is not None else "0")
                        else:
                            row.append(str(val))
                    table.add_row(*row)

            if has_active_days:
                console.print("\n")
                console.rule(f"[bold white on green] {daily_title} [/]")
                console.print(table)
            else:
                console.print("\n[yellow]No routes performed during this simulation run (all KM=0).[/]")

            console.print("\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to display simulation summary: {e}")
            traceback.print_exc()

    if lock:
        with lock:
            _print_tables()
    else:
        _print_tables()
