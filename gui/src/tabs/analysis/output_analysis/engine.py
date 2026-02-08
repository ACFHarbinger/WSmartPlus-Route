"""
Data processing engine for Output Analysis.
Handles JSON pivoting, TensorBoard event parsing, and Pareto front calculations.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_num_bins_from_path(fpath: str) -> int:
    """
    Extract Num Bins from parent directory name.
    Example: /path/to/areaname_50/log.json -> 50
    """
    parent_dir = os.path.basename(os.path.dirname(fpath))
    bin_match = re.search(r"_(\d+)$", parent_dir)
    return int(bin_match.group(1)) if bin_match else 0


def pivot_json_data(data: Dict[str, Any], filename_prefix: str = "", file_id: Any = None) -> Dict[str, List[Any]]:
    """
    Pivots nested JSON data into a flat metrics dictionary.

    Args:
        data (dict): The nested results dictionary from a JSON log.
        filename_prefix (str): Prefix to use for policy names if not present.
        file_id (any): Identifier for the source file.

    Returns:
        dict: Flattened dictionary with __Policy_Names__, __Distributions__, and __File_IDs__.
    """
    metrics = defaultdict(list)
    policy_names = []
    distributions = []
    file_ids = []

    # Regex to find common distribution patterns.
    DIST_PATTERN = r"_(emp|gamma\d+|uniform)\b"

    for policy, results in data.items():
        if not isinstance(results, dict):
            continue

        # 1. Determine Distribution
        match = re.search(DIST_PATTERN, policy, re.IGNORECASE)

        if match:
            dist = match.group(1).lower()
            base_name = policy[: match.start()] + policy[match.end() :]
            base_name = base_name.rstrip("_")
        else:
            dist = "unknown"
            base_name = policy

        distributions.append(dist)
        policy_names.append(base_name)
        file_ids.append(file_id)

        # 3. Append Metrics
        for metric, value in results.items():
            metrics[metric].append(value)

    metrics["__Policy_Names__"] = policy_names
    metrics["__Distributions__"] = distributions
    metrics["__File_IDs__"] = file_ids
    return dict(metrics)


def process_tensorboard_file(fpath: str) -> Dict[str, List[Any]]:
    """
    Extracts scalar data from a TensorBoard event file.

    Args:
        fpath (str): Path to the tfevents file.

    Returns:
        dict: Dictionary of metrics indexed by tag name.
    """
    ea = EventAccumulator(fpath)
    ea.Reload()

    all_tags = ea.Tags()
    tags_list = all_tags.get("scalars", [])
    tags = list(tags_list) if isinstance(tags_list, list) else []
    if not tags:
        return {}

    metrics = defaultdict(list)
    data_by_step = defaultdict(dict)
    filename = os.path.basename(fpath)

    for tag in tags:
        events = ea.Scalars(tag)
        for e in events:
            data_by_step[e.step][tag] = e.value
            data_by_step[e.step]["wall_time"] = e.wall_time

    sorted_steps = sorted(data_by_step.keys())
    count = len(sorted_steps)

    metrics["step"] = sorted_steps
    metrics["wall_time"] = [data_by_step[s].get("wall_time", 0) for s in sorted_steps]

    for tag in tags:
        metrics[tag] = [data_by_step[s].get(tag, float("nan")) for s in sorted_steps]

    metrics["__Policy_Names__"] = [f"{filename}" for _ in range(count)]
    metrics["__Distributions__"] = ["tensorboard"] * count
    metrics["__File_IDs__"] = [fpath] * count

    return dict(metrics)


def calculate_pareto_front(x_values: List[float], y_values: List[float]) -> List[int]:
    """
    Calculates indices of non-dominated points (Min X, Max Y).

    Args:
        x_values (list): List of X coordinates.
        y_values (list): List of Y coordinates.

    Returns:
        list: List of indices for points on the Pareto front.
    """
    points = []
    for i, (xv, yv) in enumerate(zip(x_values, y_values)):
        points.append({"idx": i, "x": xv, "y": yv})

    pareto_indices = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            # Logic: Min X and Max Y
            if (
                other["x"] <= point["x"]
                and other["y"] >= point["y"]
                and (other["x"] < point["x"] or other["y"] > point["y"])
            ):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(point["idx"])
    return pareto_indices
