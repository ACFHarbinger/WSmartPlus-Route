"""
PyDeck-based animated map for multi-period VRP simulation visualisation.

Renders two layers driven by a time-scrubbing slider:

* **ColumnLayer** — Each waste bin rendered as a 3-D column whose height and
  colour represent the reconstructed fill level at the selected simulation day.
  Columns are green when empty, red when near-full, and reset to zero on the
  day the bin is collected.

* **ArcLayer** — Vehicle travel arcs between consecutive stops in the tour
  recorded for the selected day, coloured source-blue → target-orange.

Expected log format (.jsonl, one record per day):

.. code-block:: json

    {"day": 1, "policy": "gurobi", "tour": [0, 5, 12, 8, 0],
     "kg": 125.5, "km": 45.2, "cost": 22.6, "profit": 50.2, "overflows": 0}

Attributes:
    PYDECK_TOOLTIP_HTML: HTML string for the PyDeck tooltip.
    PYDECK_TOOLTIP_STYLE: Dictionary for the PyDeck tooltip style.
    _load_jsonl: Loads and parses a .jsonl simulation log.
    _reconstruct_fill_levels: Reconstructs fill levels from the simulation log.
    _build_column_data: Builds column data for the PyDeck map.
    _build_arc_data: Builds arc data for the PyDeck map.
    _render_day_kpis: Renders the day KPIs.
    _render_static_bins: Renders the static bins.
    render_pydeck_animated_map: Main entry point for the animated map.

Example:
    from logic.src.ui.components.maps.pydeck_animated import render_pydeck_animated_map

    render_pydeck_animated_map(
        bin_locations=bins,
        simulation_log_path="assets/results/sim_log.jsonl",
        policy_name="gurobi",
        map_center_lat=51.22,
        map_center_lon=4.41,
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None

# 1. Load the raw HTML tooltip string
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "templates")
tooltip_path = os.path.join(template_dir, "pydeck_tooltip.html")
try:
    with open(tooltip_path, "r", encoding="utf-8") as f:
        PYDECK_TOOLTIP_HTML = f.read().strip()
except FileNotFoundError:
    PYDECK_TOOLTIP_HTML = "Bin Tooltip Missing"

# 2. Load the Tooltip CSS Dictionary
json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "json")
style_path = os.path.join(json_dir, "pydeck_style.json")
try:
    with open(style_path, "r", encoding="utf-8") as f:
        PYDECK_TOOLTIP_STYLE = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Fallback just in case
    PYDECK_TOOLTIP_STYLE = {}

# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_pydeck_animated_map(
    bin_locations: List[Dict[str, float]],
    simulation_log_path: str,
    policy_name: str = "gurobi",
    map_center_lat: float = 51.0,
    map_center_lon: float = 4.0,
    initial_zoom: int = 12,
    map_style: str = "road",
    column_radius: int = 50,
    column_elevation_scale: int = 10,
    arc_width: int = 3,
    daily_fill_increment: float = 5.0,
    height: int = 600,
    title: str = "Simulation Animation",
) -> None:
    """
    Render an animated PyDeck map showing vehicle routes and bin fill levels
    over a multi-period simulation horizon.

    Args:
        bin_locations: List of dicts, each with ``"lat"``, ``"lon"`` (or ``"lng"``),
            and optionally ``"bin_id"`` keys for every waste bin.
        simulation_log_path: Path to a ``.jsonl`` log file produced by the
            simulator's ``logging.py`` action module.
        policy_name: Filter log entries to this policy name. Default ``"gurobi"``.
        map_center_lat: Initial map latitude. Default 51.0.
        map_center_lon: Initial map longitude. Default 4.0.
        initial_zoom: Mapbox zoom level. Default 12.
        map_style: PyDeck map style string. Use ``"road"`` or ``"satellite"``
            for open-access styles, or a ``"mapbox://styles/..."`` URL when a
            Mapbox token is configured. Default ``"road"``.
        column_radius: Bin column radius in metres. Default 50.
        column_elevation_scale: Multiplier applied to fill-level → column height.
            Default 10.
        arc_width: Vehicle arc stroke width in pixels. Default 3.
        daily_fill_increment: Percentage fill added to each bin per simulated day
            (used to reconstruct fill levels from the log). Default 5.0.
        height: Map widget height in pixels. Default 600.
        title: Section heading rendered above the map.
    """
    if pdk is None:
        st.error(
            "**pydeck** is required for the animated map. Run: `pip install pydeck`  (also bundled with Streamlit)."
        )
        return

    st.subheader(title)

    # ── Load simulation log ──────────────────────────────────────────────────
    log_path = Path(simulation_log_path)
    if not log_path.exists():
        st.warning(f"Simulation log not found: `{simulation_log_path}`")
        _render_static_bins(bin_locations, map_center_lat, map_center_lon, initial_zoom, pdk)
        return

    records = _load_jsonl(log_path, policy_filter=policy_name)
    if not records:
        st.warning(f"No log entries found for policy **{policy_name}** in `{simulation_log_path}`.")
        return

    days = sorted({r["day"] for r in records})

    # ── Time slider ──────────────────────────────────────────────────────────
    selected_day = st.slider(
        "Simulation Day",
        min_value=days[0],
        max_value=days[-1],
        value=days[0],
        format="Day %d",
        key=f"pydeck_day_{title}",
    )

    # ── Build layer data ─────────────────────────────────────────────────────
    fill_levels = _reconstruct_fill_levels(bin_locations, records, selected_day, daily_fill_increment)
    column_data = _build_column_data(bin_locations, fill_levels)
    arc_data = _build_arc_data(bin_locations, records, selected_day)

    # ── PyDeck layers ────────────────────────────────────────────────────────
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=column_data,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=column_elevation_scale,
        radius=column_radius,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["tgt_lon", "tgt_lat"],
        get_source_color=[0, 153, 255, 200],
        get_target_color=[255, 80, 0, 200],
        get_width=arc_width,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=map_center_lat,
        longitude=map_center_lon,
        zoom=initial_zoom,
        pitch=45,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[column_layer, arc_layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={
            "html": PYDECK_TOOLTIP_HTML,
            "style": PYDECK_TOOLTIP_STYLE,
        },
    )

    st.pydeck_chart(deck, height=height)

    # ── Day KPIs ─────────────────────────────────────────────────────────────
    day_rec = next((r for r in records if r["day"] == selected_day), None)
    if day_rec:
        _render_day_kpis(day_rec)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_jsonl(
    path: Path,
    policy_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load and parse a .jsonl simulation log, optionally filtering by policy.

    Args:
        path: Path to the .jsonl log file.
        policy_filter: Optional policy name to filter log entries by.

    Returns:
        List[Dict[str, Any]]: List of parsed log records.
    """
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if policy_filter and rec.get("policy") != policy_filter:
                        continue
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return records


def _reconstruct_fill_levels(
    bins: List[Dict[str, float]],
    records: List[Dict[str, Any]],
    up_to_day: int,
    daily_increment: float,
) -> List[float]:
    """
    Replay simulation days up to ``up_to_day`` to reconstruct bin fill levels.

    Each bin accumulates ``daily_increment`` percent per day and resets to 0.0
    when it appears in the collected tour (indices are 1-based, depot = 0).

    Args:
        bins: List of bin coordinate dictionaries.
        records: Simulation log records.
        up_to_day: The simulation day to reconstruct fill levels for.
        daily_increment: Percentage added per bin per day.

    Returns:
        List[float]: Reconstructed fill levels (0-100) for each bin.
    """
    n = len(bins)
    fill = [0.0] * n
    processed_days = sorted({r["day"] for r in records if r["day"] <= up_to_day})

    for day in processed_days:
        # Increment all bins
        fill = [min(100.0, f + daily_increment) for f in fill]

        # Reset collected bins
        rec = next((r for r in records if r["day"] == day), None)
        if rec is None:
            continue
        tour = rec.get("tour", [])
        for node_idx in set(tour) - {0}:  # depot is always 0
            bin_idx = node_idx - 1  # bins are 1-indexed in tour
            if 0 <= bin_idx < n:
                fill[bin_idx] = 0.0

    return fill


def _build_column_data(
    bins: List[Dict[str, float]],
    fill_levels: List[float],
) -> List[Dict[str, Any]]:
    """
    Build ColumnLayer records with dynamic colour (green→red gradient).

    Args:
        bins: Bin coordinate list.
        fill_levels: Reconstructed fill percentages.

    Returns:
        List[Dict[str, Any]]: Records formatted for Pydeck ColumnLayer.
    """
    data: List[Dict[str, Any]] = []
    for i, b in enumerate(bins):
        pct = fill_levels[i] if i < len(fill_levels) else 0.0
        r = int(min(255, pct * 2.55))
        g = int(max(0, 255 - pct * 2.55))
        data.append(
            {
                "lat": float(b["lat"]),
                "lon": float(b.get("lon", b.get("lng", 0.0))),
                "bin_id": str(b.get("bin_id", i)),
                "fill_pct": round(pct, 1),
                "elevation": max(1.0, pct),
                "color": [r, g, 0, 210],
            }
        )
    return data


def _build_arc_data(
    bins: List[Dict[str, float]],
    records: List[Dict[str, Any]],
    day: int,
) -> List[Dict[str, Any]]:
    """
    Build ArcLayer records for vehicle movements on a specific day.

    Depot is treated as node index 0, bins as indices 1 … N.

    Args:
        bins: Master list of bin coordinates.
        records: Simulation log records.
        day: The day to extract route arcs for.

    Returns:
        List[Dict[str, Any]]: Records formatted for Pydeck ArcLayer.
    """
    rec = next((r for r in records if r["day"] == day), None)
    if rec is None:
        return []

    tour: List[int] = rec.get("tour", [])
    if len(tour) < 2:
        return []

    # Coordinate lookup: index 0 = depot (centroid), 1..N = bins
    depot_lat = float(np.mean([b["lat"] for b in bins])) if bins else 0.0
    depot_lon = float(np.mean([b.get("lon", b.get("lng", 0.0)) for b in bins])) if bins else 0.0

    coords: List[Tuple[float, float]] = [(depot_lat, depot_lon)]
    for b in bins:
        coords.append((float(b["lat"]), float(b.get("lon", b.get("lng", 0.0)))))

    arcs: List[Dict[str, Any]] = []
    for k in range(len(tour) - 1):
        src, tgt = tour[k], tour[k + 1]
        if src >= len(coords) or tgt >= len(coords):
            continue
        arcs.append(
            {
                "src_lat": coords[src][0],
                "src_lon": coords[src][1],
                "tgt_lat": coords[tgt][0],
                "tgt_lon": coords[tgt][1],
            }
        )
    return arcs


def _render_day_kpis(record: Dict[str, Any]) -> None:
    """
    Display per-day KPI metrics below the map.

    Args:
        record: The log record for the selected day.
    """
    kpi_defs: List[Tuple[str, str, str]] = [
        ("kg", "Waste (kg)", ".1f"),
        ("km", "Distance (km)", ".1f"),
        ("cost", "Cost (€)", ".2f"),
        ("profit", "Profit (€)", ".2f"),
        ("overflows", "Overflows", "d"),
    ]
    cols = st.columns(len(kpi_defs))
    for col, (key, label, fmt) in zip(cols, kpi_defs, strict=False):
        val = record.get(key, "N/A")
        display = f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)
        col.metric(label, display)


def _render_static_bins(
    bins: List[Dict[str, float]],
    lat: float,
    lon: float,
    zoom: int,
    pdk: Any,
) -> None:
    """
    Fallback: render bin locations as scatter dots when no log is available.

    Args:
        bins: List of bin coordinate dictionaries.
        lat: Latitude for view initialization.
        lon: Longitude for view initialization.
        zoom: Initial zoom level.
        pdk: The PyDeck module instance.
    """
    if not bins:
        st.info("No bin locations provided.")
        return
    data = [{"lat": float(b["lat"]), "lon": float(b.get("lon", b.get("lng", 0.0)))} for b in bins]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["lon", "lat"],
        get_radius=60,
        get_color=[0, 153, 255, 160],
        pickable=True,
    )
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=30)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
