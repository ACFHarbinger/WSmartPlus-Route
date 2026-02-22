"""
Folium map renderer for showing bin fill levels as a heatmap.
"""

from typing import Any, Dict, List

import folium

from logic.src.utils.ui.maps_utils import get_map_center


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
