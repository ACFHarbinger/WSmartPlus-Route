"""
Folium map renderer for simulation tours.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import folium
import pandas as pd

from logic.src.constants.dashboard import BIN_COLORS, ROUTE_COLORS
from logic.src.utils.ui.maps_utils import get_map_center


def create_simulation_map(  # noqa: C901
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
                    if id_from >= 0 and id_to >= 0 and id_from < len(distance_matrix) and id_to < len(distance_matrix):
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
                dist_km = haversine_distance(start[0], start[1], end[0], end[1])  # type: ignore[assignment]
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
