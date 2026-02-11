"""
GUI communication for simulation logs and daily stats.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

import logic.src.constants as udef


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
    must_go: Optional[Sequence[int]] = None,
) -> None:
    """Write daily simulation output to a log file for GUI consumption."""
    full_payload = {k: v for k, v in daily_log.items() if k in udef.DAY_METRICS[:-1]}
    route_coords = []
    coords_lookup = None
    if isinstance(coordinates, pd.DataFrame):
        coords_lookup = coordinates.copy()
        coords_lookup.columns = [str(c).upper().strip() for c in coords_lookup.columns]

    # Tour contains node indices (0=depot, 1..N=bins)
    for idx in tour:
        point_data = _process_tour_point(idx, coords_lookup)
        route_coords.append(point_data)

    full_payload.update({udef.DAY_METRICS[-1]: route_coords})
    full_payload.update(
        {
            "bin_state_c": list(bins_c),
            "bin_state_collected": list(collected),
            "bins_state_real_c_after": list(bins_real_c_after),
        }
    )

    # Include coordinates for ALL nodes (depot + bins)
    if coords_lookup is not None:
        all_bin_coords = _build_all_bin_coords(coords_lookup, len(bins_c))
        full_payload["all_bin_coords"] = all_bin_coords

    if must_go is not None:
        # Standardize must_go to 0-indexed bin IDs (0..N-1)
        # Selection strategies return Node IDs (1..N)
        shifted_must_go = [i - 1 for i in must_go if i > 0]
        full_payload.update({"must_go": shifted_must_go})

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
    """Write final simulation summary to a log file for GUI consumption."""
    lgsd = (
        {k: [[0] * len(v) if isinstance(v, (tuple, list)) else 0 for v in pol_data] for k, pol_data in log.items()}
        if log_std is None
        else {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log_std.items()}
    )
    summary_data = {
        "log": {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log.items()},
        "log_std": lgsd,
        "n_samples": n_samples,
        "policies": policies,
    }
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


def _get_lat_lon(row: pd.Series) -> tuple:
    """Extract lat/lon from a row with flexible column naming."""
    lat = row.get("LATITUDE") or row.get("LAT") or row.get("Y")
    lon = row.get("LONGITUDE") or row.get("LNG") or row.get("LON") or row.get("X")
    return lat, lon


def _process_tour_point(node_idx: int, coords_lookup: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Build a tour point dict from a node index.

    Standard mapping:
    - Node 0 (Depot) -> Payload ID -1
    - Nodes 1..N (Bins) -> Payload ID 0..N-1
    """
    try:
        node_idx_int = int(node_idx)
    except (ValueError, TypeError):
        return {"id": node_idx}

    if node_idx_int == 0:
        gui_id = -1
        row_idx = 0
        p_type = "depot"
        p_label = "Depot"
    else:
        gui_id = node_idx_int - 1
        row_idx = node_idx_int
        p_type = "bin"
        p_label = f"Bin {gui_id}"

    point_data: Dict[str, Any] = {"id": gui_id, "type": p_type}

    if coords_lookup is not None and 0 <= row_idx < len(coords_lookup):
        row = coords_lookup.iloc[row_idx]
        lat, lon = _get_lat_lon(row)
        if lat is not None and lon is not None:
            point_data["lat"] = float(lat)
            point_data["lng"] = float(lon)
            # Add Dataset ID for detailed reference
            dataset_id = row.get("ID", row_idx)
            point_data["popup"] = f"<b>{p_label}</b><br>Dataset ID: {dataset_id}"

    return point_data


def _build_all_bin_coords(coords_lookup: pd.DataFrame, n_bins: int) -> List[Dict[str, Any]]:
    """Build coordinate entries for ALL nodes using the standarized ID mapping."""
    all_bin_coords: List[Dict[str, Any]] = []
    # Total nodes in coords_lookup should be n_bins + 1 (depot)
    for row_idx in range(len(coords_lookup)):
        row = coords_lookup.iloc[row_idx]
        lat, lon = _get_lat_lon(row)
        if lat is not None and lon is not None:
            if row_idx == 0:
                gui_id = -1
                p_type = "depot"
                p_label = "Depot"
            else:
                gui_id = row_idx - 1
                p_type = "bin"
                p_label = f"Bin {gui_id}"

            dataset_id = row.get("ID", row_idx)
            all_bin_coords.append(
                {
                    "id": gui_id,
                    "lat": float(lat),
                    "lng": float(lon),
                    "type": p_type,
                    "popup": f"<b>{p_label}</b><br>Dataset ID: {dataset_id}",
                }
            )
    return all_bin_coords
