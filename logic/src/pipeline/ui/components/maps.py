# Copyright (c) WSmart-Route. All rights reserved.
"""
Folium map renderers for simulation visualization.

Provides interactive map components for routes and bin states.
"""

from typing import Any, Dict, List, Optional, Tuple

import folium

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


def create_simulation_map(
    tour: List[Dict[str, Any]],
    bin_states: Optional[List[float]] = None,
    served_indices: Optional[List[int]] = None,
    vehicle_id: int = 0,
    show_route: bool = True,
    zoom_start: int = 12,
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

    Returns:
        Folium Map object.
    """
    center = get_map_center(tour)
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    route_color = ROUTE_COLORS[vehicle_id % len(ROUTE_COLORS)]

    # Collect coordinates for route line
    route_coords: List[Tuple[float, float]] = []

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

        # Determine marker properties
        if point_type == "depot":
            # Depot marker - larger, distinct
            folium.Marker(
                location=[lat, lng],
                popup=popup_text,
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
            enhanced_popup = f"{popup_text}<br>Fill: {fill_level:.1f}%"
            if is_served:
                enhanced_popup += "<br><b>Served</b>"

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                popup=enhanced_popup,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2,
            ).add_to(m)

    # Draw route polyline
    if show_route and len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords,
            color=route_color,
            weight=3,
            opacity=0.8,
            tooltip=f"Vehicle {vehicle_id}",
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

            if point_type == "depot":
                folium.Marker(
                    location=[lat, lng],
                    popup=popup_text,
                    icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=6,
                    popup=popup_text,
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

        folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            popup=f"Fill: {fill_level:.1f}%",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
        ).add_to(m)

    return m
