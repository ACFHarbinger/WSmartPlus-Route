"""
GUI communication for simulation logs and daily stats.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import logic.src.constants as udef
import numpy as np
import pandas as pd


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

    for idx in tour:
        point_data: Dict[str, Any] = {"id": str(idx), "type": "bin"}
        if idx == 0:
            point_data["type"] = "depot"
            point_data["popup"] = "Depot"
        if coords_lookup is not None:
            lookup_idx: Union[int, str] = idx
            if lookup_idx not in coords_lookup.index:
                if str(idx) in coords_lookup.index:
                    lookup_idx = str(idx)
                elif isinstance(idx, (int, np.integer)) and int(idx) in coords_lookup.index:
                    lookup_idx = int(idx)
            if lookup_idx in coords_lookup.index:
                try:
                    row = coords_lookup.loc[lookup_idx]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    lat_val = row.get("LAT", row.get("LATITUDE"))
                    lng_val = row.get("LNG", row.get("LONGITUDE", row.get("LONG")))
                    if lat_val is not None and lng_val is not None:
                        point_data["lat"] = float(str(lat_val).replace(",", "."))
                        point_data["lng"] = float(str(lng_val).replace(",", "."))
                        if idx != 0:
                            point_data["popup"] = f"ID {row.get('ID', idx)}"
                except Exception:
                    pass
        route_coords.append(point_data)

    full_payload.update({udef.DAY_METRICS[-1]: route_coords})
    full_payload.update(
        {
            "bin_state_c": list(bins_c),
            "bin_state_collected": list(collected),
            "bins_state_real_c_after": list(bins_real_c_after),
        }
    )
    if must_go is not None:
        full_payload.update({"must_go": list(must_go)})
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
