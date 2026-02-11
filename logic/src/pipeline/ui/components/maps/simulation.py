"""
Folium map renderer for simulation tours.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, cast

import folium
import pandas as pd

from logic.src.constants.dashboard import BIN_COLORS, ROUTE_COLORS
from logic.src.utils.ui.maps_utils import get_map_center


def _add_bin_marker(
    m: folium.Map,
    point: Dict[str, Any],
    bin_id: int,
    bin_states: Optional[List[float]],
    collected_set: set,
    must_go_set: set,
    toured_ids: set,
) -> None:
    """Add a single bin marker to the map with appropriate styling.

    Args:
        m: The folium map to add the marker to.
        point: Dict with 'lat', 'lng' (or 'lon'), and optionally 'popup'.
        bin_id: The 0-indexed bin ID (matches indices in state arrays).
        bin_states: List of per-bin fill levels (0-100).
        collected_set: Set of bin IDs that were collected today.
        must_go_set: Set of bin IDs selected for collection today.
        toured_ids: Set of bin IDs that appear in the tour.
    """
    lat = point.get("lat")
    lng = point.get("lng") or point.get("lon")
    if lat is None or lng is None:
        return

    is_toured = bin_id in toured_ids
    is_served = bin_id in collected_set
    is_must_go = bin_id in must_go_set

    # Get fill level
    fill_level = 50.0
    if bin_states is not None and 0 <= bin_id < len(bin_states):
        fill_level = bin_states[bin_id]

    # 3-tier color system:
    #   Green  = served (collected today)
    #   Orange = must-go (selected but not yet served)
    #   Red    = pending (not selected, not served)
    if is_served:
        color = BIN_COLORS["served"]
    elif is_must_go:
        color = BIN_COLORS["must_go"]
    else:
        color = BIN_COLORS["pending"]

    # Non-toured bins are smaller and more transparent
    if is_toured:
        radius = 5 + (fill_level / 100.0) * 10
        fill_opacity = 0.7
    else:
        radius = 4 + (fill_level / 100.0) * 4
        fill_opacity = 0.35

    border_weight = 4 if is_must_go else 2

    # Build popup
    popup_text = point.get("popup", f"Bin {bin_id}")
    enhanced_popup = f"{popup_text}<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}<br>Fill: {fill_level:.1f}%"
    if is_must_go:
        enhanced_popup += "<br><b style='color: #fd7e14;'>Must-Go</b>"
    if is_served:
        enhanced_popup += "<br><b style='color: #28a745;'>Served</b>"
    if not is_toured:
        enhanced_popup += "<br><i>Not in route</i>"

    folium.CircleMarker(
        location=[lat, lng],
        radius=radius,
        popup=enhanced_popup,
        color=color,
        fill=True,
        fillColor=BIN_COLORS["must_go"] if is_must_go and is_served else color,
        fillOpacity=fill_opacity,
        weight=border_weight,
    ).add_to(m)

    # For must-go bins that were served, add an outer dashed ring
    if is_must_go and is_served:
        folium.CircleMarker(
            location=[lat, lng],
            radius=radius + 4,
            popup=None,
            color=BIN_COLORS["must_go"],
            fill=False,
            weight=2,
            dashArray="5 3",
        ).add_to(m)


def create_simulation_map(  # noqa: C901
    tour: List[Dict[str, Any]],
    bin_states: Optional[List[float]] = None,
    served_indices: Optional[List[int]] = None,
    must_go: Optional[List[int]] = None,
    all_bin_coords: Optional[List[Dict[str, Any]]] = None,
    collected: Optional[List[float]] = None,
    vehicle_id: int = 0,
    show_route: bool = True,
    zoom_start: int = 12,
    distance_matrix: Optional[pd.DataFrame] = None,
    dist_strategy: str = "hsd",
) -> folium.Map:
    """
    Create a Folium map visualizing a simulation tour.

    Args:
        tour: List of tour points with id, type, lat, lng, popup.
              Bins use IDs 0..N-1, Depot uses ID -1.
        bin_states: Optional list of bin fill levels (0-100).
        served_indices: Optional list of indices that were served.
        must_go: Optional list of 0-indexed bin IDs.
        all_bin_coords: Optional list of coordinate dicts for ALL bins.
        collected: Optional list of per-bin collected amounts (kg).
    """
    center = get_map_center(tour)
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")
    must_go_set = set(must_go) if must_go else set()

    # Build set of bin IDs that appear in the tour
    toured_ids: set = set()
    for point in tour:
        with contextlib.suppress(ValueError, TypeError):
            id_val = int(point.get("id", -100))
            if id_val != -100:
                toured_ids.add(id_val)

    # Build a collected set: bin IDs (0-indexed) where collected > 0
    # matches bin_id 0..N-1 in the markers
    collected_set: set = set()
    if collected:
        for i, amt in enumerate(collected):
            if amt > 0:
                collected_set.add(i)

    # Collect coordinates for route line
    route_coords: List[Tuple[float, float]] = []
    route_ids: List[int] = []

    # --- Pass 1: Render the depot from tour ---
    for point in tour:
        if point.get("type") != "depot":
            continue
        if "lat" not in point or "lng" not in point:
            continue
        lat, lng = point["lat"], point["lng"]
        popup_text = point.get("popup", "Depot")
        folium.Marker(
            location=[lat, lng],
            popup=f"{popup_text}<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}",
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(m)

    # --- Pass 2: Render ALL bins (toured + non-toured) ---
    all_bins = all_bin_coords if all_bin_coords else []
    rendered_ids: set = set()

    for bin_point in all_bins:
        if "lat" not in bin_point or "lng" not in bin_point:
            continue
        try:
            bin_id = int(bin_point["id"])
        except (ValueError, TypeError, KeyError):
            continue
        if bin_id == -1:
            continue  # Skip depot (handled in Pass 1)

        rendered_ids.add(bin_id)
        _add_bin_marker(
            m,
            bin_point,
            bin_id,
            bin_states,
            collected_set,
            must_go_set,
            toured_ids,
        )

    # Fallback for bins in tour not in all_bin_coords
    for point in tour:
        if point.get("type") == "depot":
            continue
        if "lat" not in point or "lng" not in point:
            continue
        try:
            bin_id = int(point.get("id", -100))
        except (ValueError, TypeError):
            continue
        if bin_id in rendered_ids or bin_id == -1:
            continue

        rendered_ids.add(bin_id)
        _add_bin_marker(
            m,
            point,
            bin_id,
            bin_states,
            collected_set,
            must_go_set,
            toured_ids,
        )

    # --- Pass 3: Build route coordinates for polylines ---
    for point in tour:
        if "lat" not in point or "lng" not in point:
            continue
        route_coords.append((point["lat"], point["lng"]))
        try:
            route_ids.append(int(point.get("id", -100)))
        except (ValueError, TypeError):
            route_ids.append(-100)

    # Draw route polyline segments
    if show_route and len(route_coords) > 1:
        from logic.src.pipeline.simulations.network import (
            EuclideanStrategy,
            GeodesicStrategy,
            HaversineStrategy,
            haversine_distance,
        )

        strategy_calc = None
        strategy_label = "Haversine"

        if dist_strategy == "gdsc":
            strategy_calc = cast(Optional[GeodesicStrategy], GeodesicStrategy())
            strategy_label = "Geodesic"
        elif dist_strategy == "ogd":
            strategy_calc = cast(Optional[GeodesicStrategy], EuclideanStrategy())
            strategy_label = "Euclidean"
        elif dist_strategy == "hsd":
            strategy_calc = cast(Optional[GeodesicStrategy], HaversineStrategy())
            strategy_label = "Haversine"
        elif dist_strategy == "gmaps":
            strategy_label = "Google Maps"
        elif dist_strategy == "osm":
            strategy_label = "OSM"
        elif dist_strategy == "gpd":
            strategy_label = "GeoPandas"
        elif dist_strategy == "load_matrix":
            strategy_label = "Custom Matrix"

        # Depot ID is now -1
        depot_id = -1
        trip_num = 1

        for i in range(len(route_coords) - 1):
            start = route_coords[i]
            end = route_coords[i + 1]

            if i > 0 and route_ids[i] == depot_id and route_ids[i + 1] != depot_id:
                trip_num += 1

            trip_color = ROUTE_COLORS[(trip_num - 1) % len(ROUTE_COLORS)]
            dist_km = 0.0
            dist_source = strategy_label

            if distance_matrix is not None:
                try:
                    # distance_matrix handles raw node indices (0..N)
                    # We need to map our GUI IDs back to node indices
                    idx_from = route_ids[i] + 1 if route_ids[i] >= 0 else 0
                    idx_to = route_ids[i + 1] + 1 if route_ids[i + 1] >= 0 else 0

                    if 0 <= idx_from < len(distance_matrix) and 0 <= idx_to < len(distance_matrix):
                        dist_km = distance_matrix.iloc[idx_from, idx_to]
                        dist_source = f"{strategy_label} (Matrix)"
                except Exception:
                    pass

            if dist_km == 0.0 and strategy_calc is not None:
                try:
                    dist_km = strategy_calc.calculate_pair(start, end)
                    dist_source = f"{strategy_label} (Calc)"
                except Exception:
                    pass

            if dist_km == 0.0:
                dist_km = haversine_distance(start[0], start[1], end[0], end[1])  # type: ignore[assignment]
                if strategy_label != "Haversine" and "Matrix" not in dist_source:
                    dist_source += " (Fallback)"

            folium.PolyLine(
                locations=[start, end],
                color=trip_color,
                weight=3,
                opacity=0.8,
                tooltip=f"Trip {trip_num}, Leg {i + 1}: {dist_km:.2f} km ({dist_source})",
            ).add_to(m)

    return m
