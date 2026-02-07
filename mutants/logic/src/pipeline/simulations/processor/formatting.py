"""
Coordinate transformation and normalization logic for simulation data.
"""

from typing import Any, Optional, Tuple

import numpy as np
from logic.src.constants import EARTH_RADIUS, EARTH_WMP_RADIUS

from ..network import haversine_distance


def format_coordinates(coords: Any, method: str, col_names: Optional[list[str]] = None) -> Tuple[Any, Any]:
    """
    Normalizes and formats coordinates based on the specified method.
    Supported methods: 'mmn', 'mun', 'smsd', 'ecp', 'utmp', 'wmp', 'hdp', 'c3d', 's4d'.
    """
    assert method in [
        "mmn",
        "mun",
        "smsd",
        "ecp",
        "utmp",
        "wmp",
        "hdp",
        "c3d",
        "s4d",
    ]

    if col_names is None:
        col_names = ["Lat", "Lng"]

    IS_PANDAS = hasattr(coords, "columns")
    lat = coords[col_names[0]] if IS_PANDAS else coords[:, :, 0]
    lng = coords[col_names[1]] if IS_PANDAS else coords[:, :, 1]

    depot, loc = None, None

    if method == "c3d":  # Conversion to 3D Cartesian coordinates
        latr = np.radians(lat)
        lngr = np.radians(lng)
        x_axis = EARTH_RADIUS * np.cos(latr) * np.cos(lngr)
        y_axis = EARTH_RADIUS * np.cos(latr) * np.sin(lngr)
        z_axis = EARTH_RADIUS * np.sin(latr)
        if IS_PANDAS:
            x_axis = (x_axis - x_axis.min()) / (x_axis.max() - x_axis.min())
            y_axis = (y_axis - y_axis.min()) / (y_axis.max() - y_axis.min())
            z_axis = (z_axis - z_axis.min()) / (z_axis.max() - z_axis.min())
            depot = np.array([x_axis.iloc[0], y_axis.iloc[0], z_axis.iloc[0]])
            loc = np.array([[x, y, z] for x, y, z in zip(x_axis.iloc[1:], y_axis.iloc[1:], z_axis.iloc[1:])])
        else:
            coords_3d = np.stack((x_axis, y_axis, z_axis), axis=-1)
            min_arr = np.min(coords_3d, axis=1, keepdims=True)
            max_arr = np.max(coords_3d, axis=1, keepdims=True)
            coords_3d = (coords_3d - min_arr) / (max_arr - min_arr)
            depot = coords_3d[:, 0, :]
            loc = coords_3d[:, 1:, :]

    elif method == "s4d":  # Conversion to 4D spherical coordinates
        latr = np.radians(lat)
        lngr = np.radians(lng)
        lats, latc = np.sin(latr), np.cos(latr)
        lngs, lngc = np.sin(lngr), np.cos(lngr)
        if IS_PANDAS:
            lats = (lats - lats.min()) / (lats.max() - lats.min())
            latc = (latc - latc.min()) / (latc.max() - latc.min())
            lngs = (lngs - lngs.min()) / (lngs.max() - lngs.min())
            lngc = (lngc - lngc.min()) / (lngc.max() - lngc.min())
            depot = np.array([lats.iloc[0], latc.iloc[0], lngs.iloc[0], lngc.iloc[0]])
            loc = np.array(
                [[x, y, z, w] for x, y, z, w in zip(lats.iloc[1:], latc.iloc[1:], lngs.iloc[1:], lngc.iloc[1:])]
            )
        else:
            coords4d = np.stack([lats, latc, lngs, lngc], axis=-1)
            min_arr = np.min(coords4d, axis=1, keepdims=True)
            max_arr = np.max(coords4d, axis=1, keepdims=True)
            coords4d = (coords4d - min_arr) / (max_arr - min_arr)
            depot = coords4d[:, 0, :]
            loc = coords4d[:, 1:, :]

    else:
        if method == "mun":  # Mean (μ) normalization
            if IS_PANDAS:
                lat = (lat - lat.mean()) / (lat.max() - lat.min())
                lng = (lng - lng.mean()) / (lng.max() - lng.min())
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                mean_arr = np.mean(coords, axis=1, keepdims=True)
                coords = (coords - mean_arr) / (max_arr - min_arr)
        elif method == "smsd":  # Standardization (using μ and σ)
            if IS_PANDAS:
                lat = (lat - lat.mean()) / lat.std()
                lng = (lng - lng.mean()) / lng.std()
            else:
                coords = coords[:, :, [1, 0]]
                mean_arr = np.mean(coords, axis=1, keepdims=True)
                std_arr = np.std(coords, axis=1, keepdims=True)
                coords = (coords - mean_arr) / std_arr
        elif method == "ecp":  # Equidistant cylindrical

            def per_func(arr, percent):
                """Calculate the percentile for a given array (pandas or numpy)."""
                if IS_PANDAS:
                    return np.percentile(arr, percent)
                return np.percentile(arr, percent, axis=1, keepdims=True)

            if IS_PANDAS:
                center_meridian = (lng.max() + lng.min()) / 2
                lat_lower, lat_upper = per_func(lat, 10), per_func(lat, 90)
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                center_meridian = (max_arr + min_arr) / 2
                lat_lower, lat_upper = per_func(coords[:, :, 1], 10), per_func(coords[:, :, 1], 90)

            center_parallel = (lat_upper + lat_lower) / 2
            offset = (lat_upper - lat_lower) / 2
            pscale = (np.cos(np.radians(center_parallel - offset)) + np.cos(np.radians(center_parallel + offset))) / 2
            lat = EARTH_RADIUS * (np.radians(lat) - np.radians(center_parallel))
            lng = EARTH_RADIUS * (np.radians(lng) - np.radians(center_meridian)) * pscale
        elif method == "utmp":
            raise NotImplementedError("UTM Projection not implemented")
        elif method == "wmp":  # World Mercator projection
            lng = EARTH_WMP_RADIUS * np.radians(lng)
            lat = EARTH_WMP_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
            if not IS_PANDAS:
                coords = np.stack((lat, lng), axis=-1)
        elif method == "hdp":  # Haversine distance projection

            def max_func(h1, h2):
                """Return the maximum of two values or element-wise maximum."""
                if IS_PANDAS:
                    return max(h1, h2)
                return np.max(np.concatenate((h1, h2)), axis=0, keepdims=True)

            if IS_PANDAS:
                lat_max, lat_min = lat.max(), lat.min()
                lng_max, lng_min = lng.max(), lng.min()
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                lng_min, lat_min = np.split(min_arr, 2, axis=-1)
                lng_max, lat_max = np.split(max_arr, 2, axis=-1)

            mid_lat = (lat_max + lat_min) / 2
            mid_lng = (lng_max + lng_min) / 2
            max_distance = (
                EARTH_RADIUS
                * np.pi
                / 180
                * max_func(
                    haversine_distance(lat_min, lng_min, lat_max, lng_max),
                    haversine_distance(lat_min, lng_max, lat_max, lng_min),
                )
            )
            lat = (lat - mid_lat) / max_distance
            lng = (lng - mid_lng) / max_distance
        else:
            assert method == "mmn"  # Min-Max normalization
            if IS_PANDAS:
                lat = (lat - lat.min()) / (lat.max() - lat.min()) if lat.max() != lat.min() else lat
                lng = (lng - lng.min()) / (lng.max() - lng.min()) if lng.max() != lng.min() else lng
            else:
                coords = coords[:, :, [1, 0]]
                min_arr = np.min(coords, axis=1, keepdims=True)
                max_arr = np.max(coords, axis=1, keepdims=True)
                # Avoid division by zero
                diff = max_arr - min_arr
                diff[diff == 0] = 1.0
                coords = (coords - min_arr) / diff

        if IS_PANDAS:
            depot = np.array([lng.iloc[0], lat.iloc[0]])
            loc = np.array([[x, y] for x, y in zip(lng.iloc[1:], lat.iloc[1:])])
        else:
            depot = coords[:, 0, :]
            loc = coords[:, 1:, :]

    return depot, loc
