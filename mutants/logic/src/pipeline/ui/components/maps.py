# Copyright (c) WSmart-Route. All rights reserved.
"""
Folium map renderers for simulation visualization.

Provides interactive map components for routes and bin states.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import folium
import pandas as pd

# Color palette for different vehicles/routes
ROUTE_COLORS = [
    "#e41a1c",  # Red
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#984ea3",  # Purple
    "#ff7f00",  # Orange
    "#ffff33",  # Yellow
    "#a65628",  # Brown
    "#f781bf",  # Pink
]

# Bin status colors
BIN_COLORS = {
    "served": "#28a745",  # Green
    "pending": "#dc3545",  # Red
    "depot": "#007bff",  # Blue
}


def get_map_center(tour: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculate the center point of a tour.

    Args:
        tour: List of tour points with lat/lng.

    Returns:
        (latitude, longitude) tuple for center.
    """
    lats = [p.get("lat", 0) for p in tour if "lat" in p]
    lngs = [p.get("lng", 0) for p in tour if "lng" in p]

    if not lats or not lngs:
        return (39.33, -8.94)  # Default to Portugal area

    return (sum(lats) / len(lats), sum(lngs) / len(lngs))


def load_distance_matrix(instance_name: str = "riomaior") -> Optional[pd.DataFrame]:
    """
    Load the distance matrix for a given problem instance.

    Args:
        instance_name: Name of the instance (e.g., 'riomaior').

    Returns:
        DataFrame containing the distance matrix, or None if not found.
    """
    from pathlib import Path

    import pandas as pd

    # Try to find a matching file in data/wsr_simulator/distance_matrix
    # Common pattern seems to be gmaps_distmat_plastic[{instance_name}].csv
    base_path = Path("data/wsr_simulator/distance_matrix")
    if not base_path.exists():
        return None

    # Search for files containing the instance name
    candidates = list(base_path.glob(f"*{instance_name}*.csv"))

    if not candidates:
        return None

    # Prefer the 'plastic' one if multiple, or just take the first
    # Example: gmaps_distmat_plastic[riomaior].csv
    selected_file = candidates[0]
    for cand in candidates:
        if f"plastic[{instance_name}]" in cand.name:
            selected_file = cand
            break

    try:
        # Load matrix, assuming first row/col are headers/indices if it's a named matrix
        # Based on file inspection, it might be a raw matrix or have headers.
        # Usually these matrices are square.
        df = pd.read_csv(selected_file, header=None)
        return df
    except Exception:
        return None


def create_simulation_map(
    tour: List[Dict[str, Any]],
    bin_states: Optional[List[float]] = None,
    served_indices: Optional[List[int]] = None,
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
        bin_states: Optional list of bin fill levels (0-100).
        served_indices: Optional list of indices that were served (collected).
        vehicle_id: Vehicle ID for route coloring.
        show_route: Whether to draw route polylines.
        zoom_start: Initial zoom level.
        distance_matrix: Optional distance matrix for exact edge weights.
        dist_strategy: Strategy key ('gmaps', 'osm', 'gdsc', 'hsd', 'ogd').

    Returns:
        Folium Map object.
    """
    center = get_map_center(tour)
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    route_color = ROUTE_COLORS[vehicle_id % len(ROUTE_COLORS)]

    # Collect coordinates for route line
    route_coords: List[Tuple[float, float]] = []
    route_ids: List[int] = []

    # Add markers for each point
    for i, point in enumerate(tour):
        if "lat" not in point or "lng" not in point:
            continue

        lat = point["lat"]
        lng = point["lng"]
        point_id = point.get("id", str(i))
        point_type = point.get("type", "bin")
        popup_text = point.get("popup", f"Point {point_id}")

        route_coords.append((lat, lng))
        # Try to parse ID as int for matrix lookup
        try:
            route_ids.append(int(point_id))
        except ValueError:
            route_ids.append(-1)

        # Enhanced popup with coordinates
        coord_info = f"<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}"

        # Determine marker properties
        if point_type == "depot":
            # Depot marker - larger, distinct
            folium.Marker(
                location=[lat, lng],
                popup=f"{popup_text}{coord_info}",
                icon=folium.Icon(color="blue", icon="home", prefix="fa"),
            ).add_to(m)
        else:
            # Bin marker
            # Determine if served
            is_served = False
            if served_indices is not None:
                try:
                    idx = int(point_id)
                    is_served = idx in served_indices
                except (ValueError, TypeError):
                    is_served = False

            # Get fill level for color intensity
            fill_level = 50.0  # Default
            if bin_states is not None:
                try:
                    idx = int(point_id)
                    if 0 <= idx < len(bin_states):
                        fill_level = bin_states[idx]
                except (ValueError, TypeError, IndexError):
                    pass

            # Color based on served status
            color = BIN_COLORS["served"] if is_served else BIN_COLORS["pending"]

            # Size based on fill level (min 5, max 15)
            radius = 5 + (fill_level / 100.0) * 10

            # Enhanced popup
            enhanced_popup = f"{popup_text}{coord_info}<br>Fill: {fill_level:.1f}%"
            if is_served:
                enhanced_popup += "<br><b>Served</b>"

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                popup=enhanced_popup,
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2,
            ).add_to(m)

    # Draw route polyline segments with info
    if show_route and len(route_coords) > 1:
        # Import strategies from network
        from logic.src.pipeline.simulations.network import (
            EuclideanStrategy,
            GeodesicStrategy,
            HaversineStrategy,
            haversine_distance,
        )

        # Instantiate strategy if applicable
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

        for i in range(len(route_coords) - 1):
            start = route_coords[i]
            end = route_coords[i + 1]

            # Distance calculation
            dist_km = 0.0
            dist_source = strategy_label

            # Priority 1: Distance Matrix (if passed)
            if distance_matrix is not None:
                try:
                    id_from = route_ids[i]
                    id_to = route_ids[i + 1]
                    if id_from >= 0 and id_to >= 0:
                        if id_from < len(distance_matrix) and id_to < len(distance_matrix):
                            dist_km = distance_matrix.iloc[id_from, id_to]
                            dist_source = f"{strategy_label} (Matrix)"
                except Exception:
                    pass

            # Priority 2: On-the-fly Calculation (Iterative Strategies)
            if dist_km == 0.0 and strategy_calc is not None:
                try:
                    dist_km = strategy_calc.calculate_pair(start, end)
                    dist_source = f"{strategy_label} (Calc)"
                except Exception:
                    pass

            # Fallback
            if dist_km == 0.0:
                dist_km = haversine_distance(start[0], start[1], end[0], end[1])
                if strategy_label != "Haversine" and "Matrix" not in dist_source:
                    dist_source += " (Fallback)"

            folium.PolyLine(
                locations=[start, end],
                color=route_color,
                weight=3,
                opacity=0.8,
                tooltip=f"Leg {i + 1}: {dist_km:.2f} km ({dist_source})",
            ).add_to(m)

    return m


def create_multi_route_map(
    routes: List[List[Dict[str, Any]]],
    bin_states: Optional[List[float]] = None,
    zoom_start: int = 12,
) -> folium.Map:
    """
    Create a map with multiple vehicle routes.

    Args:
        routes: List of tours, one per vehicle.
        bin_states: Optional bin fill levels.
        zoom_start: Initial zoom level.

    Returns:
        Folium Map object.
    """
    # Find center from all routes
    all_points: List[Dict[str, Any]] = []
    for route in routes:
        all_points.extend(route)

    center = get_map_center(all_points)
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    # Add each route with different colors
    for vehicle_id, tour in enumerate(routes):
        route_color = ROUTE_COLORS[vehicle_id % len(ROUTE_COLORS)]
        route_coords: List[Tuple[float, float]] = []

        for point in tour:
            if "lat" not in point or "lng" not in point:
                continue

            lat = point["lat"]
            lng = point["lng"]
            route_coords.append((lat, lng))

        if len(route_coords) > 1:
            folium.PolyLine(
                locations=route_coords,
                color=route_color,
                weight=3,
                opacity=0.8,
                tooltip=f"Vehicle {vehicle_id}",
            ).add_to(m)

    # Add bin markers (avoiding duplicates)
    added_bins: set = set()
    for route in routes:
        for point in route:
            if "lat" not in point or "lng" not in point:
                continue

            point_id = point.get("id", "")
            if point_id in added_bins:
                continue
            added_bins.add(point_id)

            lat = point["lat"]
            lng = point["lng"]
            point_type = point.get("type", "bin")
            popup_text = point.get("popup", f"Point {point_id}")

            coord_info = f"<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}"

            if point_type == "depot":
                folium.Marker(
                    location=[lat, lng],
                    popup=f"{popup_text}{coord_info}",
                    icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=6,
                    popup=f"{popup_text}{coord_info}",
                    color="#6c757d",
                    fill=True,
                    fillOpacity=0.5,
                ).add_to(m)

    return m


def create_bin_heatmap(
    bin_locations: List[Dict[str, Any]],
    bin_states: List[float],
) -> folium.Map:
    """
    Create a heatmap-style map showing bin fill levels.

    Args:
        bin_locations: List of points with lat/lng.
        bin_states: Fill levels (0-100) for each bin.

    Returns:
        Folium Map object.
    """
    center = get_map_center(bin_locations)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    for i, point in enumerate(bin_locations):
        if "lat" not in point or "lng" not in point:
            continue

        lat = point["lat"]
        lng = point["lng"]

        fill_level = bin_states[i] if i < len(bin_states) else 50.0

        # Color gradient from green (empty) to red (full)
        if fill_level < 33:
            color = "#28a745"  # Green
        elif fill_level < 66:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red

        coord_info = f"<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}"

        folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            popup=f"Fill: {fill_level:.1f}%{coord_info}",
            color=color,
            fill=True,
            fillOpacity=0.8,
        ).add_to(m)

    return m
