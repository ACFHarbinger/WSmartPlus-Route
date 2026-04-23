"""GUI communication bridge for simulation results.

This module provides functions for serializing real-time and end-of-run simulation
data into a format recognized by the PySide6 desktop application. It handles
coordinate transformation, popup HTML rendering via Jinja2, and safe file
appending for multi-threaded simulation environments.

Attributes:
    send_daily_output_to_gui: Serializes daily simulation metrics and state.
    send_final_output_to_gui: Serializes aggregated experiment statistics.

Example:
    >>> from logic.src.tracking.logging.modules import gui
    >>> gui.send_daily_output_to_gui(log, "greedy", 0, 1, ..., "app.log")
"""

import json
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jinja2
import pandas as pd
from omegaconf import OmegaConf

import logic.src.constants as udef
from logic.src.utils.configs.setup_utils import deep_sanitize

# Set up Jinja environment and pre-load the template for performance
template_dir = os.path.dirname(os.path.abspath(__file__))
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
POPUP_TEMPLATE = jinja_env.get_template("popup.html")


def send_daily_output_to_gui(
    daily_log: Dict[str, Any],
    policy: str,
    sample_idx: int,
    day: int,
    bins_c: Sequence[float],
    collected: Sequence[float],
    bins_real_c_after: Sequence[float],
    log_path: str,
    tour: Sequence[int],
    coordinates: Union[pd.DataFrame, List[Any]],
    lock: Optional[threading.Lock] = None,
    mandatory: Optional[Sequence[int]] = None,
) -> None:
    """Write daily simulation output to a log file for GUI consumption.

    Args:
        daily_log: Dictionary containing day-to-day metrics.
        policy: Name of the policy/solver being evaluated.
        sample_idx: Index of the current simulation sample.
        day: Current simulation day (1-indexed).
        bins_c: Sequence of bin capacities/levels.
        collected: Sequence of waste amounts collected.
        bins_real_c_after: Real bin levels after collection (for stochasticity).
        log_path: Path to the GUI log file.
        tour: Sequence of nodes visited in the route.
        coordinates: Node coordinate data (DataFrame or list).
        lock: Optional thread lock for safe file appending. Defaults to None.
        mandatory: Optional sequence of nodes that MUST be visited. Defaults to None.
    """
    # Preserve the entire daily_log, including its pre-mapped Dataset ID "tour" list
    full_payload = {k: v for k, v in daily_log.items() if k in udef.DAY_METRICS}

    coords_lookup = None
    if isinstance(coordinates, pd.DataFrame):
        coords_lookup = coordinates.copy()
        coords_lookup.columns = [str(c).upper().strip() for c in coords_lookup.columns]

    # Output full coordinates map only on day 1 to act as a structural lookup in the UI
    if day == 1:
        route_coords = []

        node_count = len(coords_lookup) if coords_lookup is not None else len(coordinates)
        for i in range(node_count):
            point_data = _process_tour_point(i, coords_lookup)
            route_coords.append(point_data)

        full_payload.update({"all_bin_coords": route_coords})
    full_payload.update(
        {
            "bin_state_c": list(bins_c),
            "bin_state_collected": list(collected),
            "bins_state_real_c_after": list(bins_real_c_after),
        }
    )

    # Tour bin indices: 0-indexed bin IDs visited in route order (excluding depot)
    tour_indices = [int(idx) - 1 for idx in tour if int(idx) > 0]
    full_payload["tour_indices"] = tour_indices

    if mandatory is not None:
        # Map 1-based bin IDs (logic) to 0-based GUI IDs (indices)
        # Bins are i=1..N, Depot is i=0 (skipped)
        mapped_mandatory = [int(i) - 1 for i in mandatory if int(i) > 0]
        full_payload.update({"mandatory": mapped_mandatory})

    full_payload = deep_sanitize(full_payload)
    full_payload = OmegaConf.to_container(OmegaConf.create(full_payload), resolve=True)
    log_msg = f"GUI_DAY_LOG_START:{policy},{sample_idx},{day},{json.dumps(full_payload)}"
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()


def send_final_output_to_gui(
    log: Dict[str, Any],
    log_std: Optional[Dict[str, Any]],
    n_samples: int,
    policies: List[str],
    log_path: str,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Write final simulation summary to a log file for GUI consumption.

    Args:
        log: Dictionary of mean results per policy.
        log_std: Optional dictionary of standard deviations.
        n_samples: Total number of samples in the simulation.
        policies: List of policy names included in the run.
        log_path: Path to the GUI log file.
        lock: Optional thread lock for safe file writing. Defaults to None.
    """
    lgsd = (
        {k: [[0] * len(v) if isinstance(v, (tuple, list)) else 0 for v in pol_data] for k, pol_data in log.items()}
        if log_std is None
        else {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log_std.items()}
    )
    summary_data = {
        "log": {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log.items()},
        "log_std": lgsd,
        "n_samples": n_samples,
        "policies": list(policies),
    }

    # Deep sanitize summary_data to handle any remaining ListConfigs/DictConfigs and NumPy types
    summary_data = deep_sanitize(summary_data)
    summary_data = OmegaConf.to_container(OmegaConf.create(summary_data), resolve=True)

    summary_message = f"GUI_SUMMARY_LOG_START: {json.dumps(summary_data)}"
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(summary_message + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write summary to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()


def _get_lat_lon(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Extract lat/lon from a row with flexible column naming.

    Args:
        row: Pandas series representing a node's data.

    Returns:
        Tuple[Optional[float], Optional[float]]: Latitude and longitude.
    """
    lat = row.get("LATITUDE") or row.get("LAT") or row.get("Y")
    lon = row.get("LONGITUDE") or row.get("LNG") or row.get("LON") or row.get("X")
    return lat, lon


def _process_tour_point(node_idx: int, coords_lookup: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Build a tour point dict from a node index.

    Standard mapping:
    - Node 0 (Depot) -> Payload ID -1
    - Nodes 1..N (Bins) -> Payload ID 0..N-1

    Args:
        node_idx: The 1-based index of the node in the logic layer.
        coords_lookup: Optional coordinate metadata from a CSV/DataFrame.

    Returns:
        Dict[str, Any]: Formatted dictionary for GUI map rendering.
    """
    try:
        node_idx_int = int(node_idx)
    except (ValueError, TypeError):
        return {"id": node_idx}

    gui_id = node_idx_int - 1
    dataset_id = gui_id
    if coords_lookup is not None and 0 <= node_idx_int < len(coords_lookup):
        row = coords_lookup.iloc[node_idx_int]
        dataset_id = int(row.get("ID", gui_id))

    if node_idx_int == 0:
        gui_id = -1
        p_type = "depot"
        p_label = "Depot"
    else:
        gui_id = node_idx_int - 1
        p_type = "bin"
        p_label = f"Bin {gui_id}"

    point_data: Dict[str, Any] = {"id": gui_id, "type": p_type, "dataset_id": dataset_id}

    if coords_lookup is not None and 0 <= node_idx_int < len(coords_lookup):
        row = coords_lookup.iloc[node_idx_int]
        lat, lon = _get_lat_lon(row)
        if lat is not None and lon is not None:
            point_data["lat"] = float(lat)
            point_data["lng"] = float(lon)

            # 2. Render the popup using the pre-loaded template
            # This removes the hardcoded HTML tags from the Python file
            point_data["popup"] = POPUP_TEMPLATE.render(p_label=p_label, dataset_id=dataset_id)

    return point_data
