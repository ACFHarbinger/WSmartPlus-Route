"""
Folium map renderer for multiple vehicle routes.
"""

from typing import Any, Dict, List, Optional, Tuple

import folium
from logic.src.constants.dashboard import ROUTE_COLORS
from logic.src.utils.ui.maps_utils import get_map_center


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
