from typing import Any, Dict, List, Tuple

from logic.src.pipeline.ui.services.log_parser import filter_entries


def normalize_tour_points(tour: List[Any]) -> List[Any]:
    """Normalize tour point keys: map 'lon' -> 'lng' for consistency with map components."""
    for point in tour:
        if isinstance(point, int):
            return tour  # Skip if items are unresolved ID indices
        if isinstance(point, dict) and "lon" in point and "lng" not in point:
            point["lng"] = point["lon"]
    return tour


def filter_simulation_data(entries: List[Any], controls: Dict[str, Any], day_range: Tuple[int, int]) -> List[Any]:
    """Filter simulation entries based on controls."""
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
