"""
UI utilities module.

Attributes:
    normalize_tour_points: Normalize tour point keys to map 'lon' -> 'lng' for consistency with map components.
    filter_simulation_data: Filter simulation entries based on controls.
    get_map_center: Calculate the center point of a tour.
    load_distance_matrix: Load the distance matrix for a given problem instance.

Example:
    >>> from logic.src.utils.ui.sim_page import normalize_tour_points, filter_simulation_data
    >>> from logic.src.utils.ui.maps_utils import get_map_center, load_distance_matrix
    >>> normalize_tour_points([{"lon": 1, "lat": 2}])
    [{'lon': 1, 'lat': 2, 'lng': 1}]
    >>> filter_simulation_data([{"policy": "policy1", "sample_id": 1, "day": 1}], {"selected_policy": "policy1", "selected_sample": 1, "selected_day": 1, "is_live": False}, (1, 1))
    [{'policy': 'policy1', 'sample_id': 1, 'day': 1}]
    >>> get_map_center([{"lat": 1, "lng": 2}, {"lat": 3, "lng": 4}])
    (2.0, 3.0)
    >>> load_distance_matrix("riomaior")
    DataFrame containing the distance matrix for riomaior.
"""
