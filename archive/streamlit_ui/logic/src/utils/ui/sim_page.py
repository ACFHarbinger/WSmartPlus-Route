"""
Utility functions for the simulation page.

Attributes:
    normalize_tour_points: Normalize tour point keys to map 'lon' -> 'lng' for consistency with map components.
    filter_simulation_data: Filter simulation entries based on controls.

Example:
    >>> from logic.src.ui.services.log_parser import filter_entries
    >>> from logic.src.utils.ui.sim_page import normalize_tour_points, filter_simulation_data
    >>> normalize_tour_points([{"lon": 1, "lat": 2}])
    [{'lon': 1, 'lat': 2, 'lng': 1}]
    >>> filter_simulation_data([{"policy": "policy1", "sample_id": 1, "day": 1}], {"selected_policy": "policy1", "selected_sample": 1, "selected_day": 1, "is_live": False}, (1, 1))
    [{'policy': 'policy1', 'sample_id': 1, 'day': 1}]
"""

from typing import Any, Dict, List, Tuple

from logic.src.ui.services.log_parser import filter_entries


def normalize_tour_points(tour: List[Any]) -> List[Any]:
    """
    Normalize tour point keys to map 'lon' -> 'lng' for consistency with map components.

    Args:
        tour: List of tour points.

    Returns:
        List of normalized tour points.
    """
    for point in tour:
        if isinstance(point, int):
            return tour  # Skip if items are unresolved ID indices
        if isinstance(point, dict) and "lon" in point and "lng" not in point:
            point["lng"] = point["lon"]
    return tour


def filter_simulation_data(entries: List[Any], controls: Dict[str, Any], day_range: Tuple[int, int]) -> List[Any]:
    """
    Filter simulation entries based on controls.

    Args:
        entries: List of simulation entries.
        controls: Dictionary containing filter controls.
        day_range: Tuple containing the day range.

    Returns:
        List of filtered simulation entries.
    """
    filtered = filter_entries(
        entries,
        policy=controls["selected_policy"],
        sample_id=controls["selected_sample"],
        day=controls["selected_day"] if not controls["is_live"] else None,
    )

    if controls["is_live"]:
        # Get latest day
        latest_day = max(e.day for e in filtered) if filtered else day_range[1]
        filtered = filter_entries(filtered, day=latest_day)

    return filtered
