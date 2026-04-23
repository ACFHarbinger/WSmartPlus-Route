"""Geospatial mapping component facade.

This package exports high-level map renderers for waste collection routes,
bin heatmaps, and multi-day simulation playback.

Example:
    from logic.src.ui.components.maps import create_simulation_map
    m = create_simulation_map(entry)

Attributes:
    create_bin_heatmap: Visualizes spatial distribution of bin fill levels.
    create_multi_route_map: Renders multiple routes with temporal overlays.
    create_simulation_map: Main Digital Twin interactive map renderer.
"""

from logic.src.utils.ui.maps_utils import get_map_center as get_map_center
from logic.src.utils.ui.maps_utils import load_distance_matrix as load_distance_matrix

from .heatmap import create_bin_heatmap as create_bin_heatmap
from .multi_route import create_multi_route_map as create_multi_route_map
from .simulation import create_simulation_map as create_simulation_map
