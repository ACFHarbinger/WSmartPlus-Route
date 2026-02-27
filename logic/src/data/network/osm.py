"""
Strategy for computing distances using OpenStreetMap.
"""

from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd
from networkx import MultiDiGraph
from tqdm import tqdm

from .base import DistanceStrategy


class OSMStrategy(DistanceStrategy):
    """Strategy for computing distances using OpenStreetMap road network."""

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes road network distances using OpenStreetMap (OSMnx).
        """
        if "graph" in kwargs and isinstance(kwargs["graph"], MultiDiGraph):
            GG = kwargs["graph"]
        elif self._eval_kwarg("download_method", kwargs) and kwargs["download_method"] == "bbox":
            bounding_box = (
                coords["Lat"].max(),
                coords["Lat"].min(),
                coords["Lng"].max(),
                coords["Lng"].min(),
            )
            GG = ox.graph_from_bbox(bounding_box, network_type="drive")
        else:
            GG = None

        size = len(coords)
        distance_matrix = np.zeros((size, size))

        # Iterative calculation
        for id_i, row_i in tqdm(
            coords.iterrows(),
            total=size,
            desc="Outer Loop",
            disable=not kwargs.get("verbose", False),
        ):
            for id_j, row_j in tqdm(
                coords.iterrows(),
                total=size,
                desc="Inner Loop",
                leave=False,
                disable=not kwargs.get("verbose", False),
            ):
                if id_i != id_j:
                    coords_i = (row_i["Lat"], row_i["Lng"])
                    coords_j = (row_j["Lat"], row_j["Lng"])

                    # This part looks slow if GG is None, but keeping original logic
                    G = GG if GG is not None else ox.graph_from_point(coords_i, dist=10000, network_type="drive")
                    bin_i = ox.distance.nearest_nodes(G, coords_i[1], coords_i[0])
                    bin_j = ox.distance.nearest_nodes(G, coords_j[1], coords_j[0])
                    try:
                        length = ox.shortest_path(G, bin_i, bin_j, weight="length")
                        if length:
                            distance_matrix[id_i, id_j] = sum(length) / 10_000_000_000
                        else:
                            distance_matrix[id_i, id_j] = float("inf")
                    except Exception:
                        distance_matrix[id_i, id_j] = float("inf")

        return distance_matrix
