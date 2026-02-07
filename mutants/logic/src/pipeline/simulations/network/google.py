"""
Strategy for computing distances using Google Maps API.
"""

import os
from typing import Any

import googlemaps
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from logic.src.constants import ROOT_DIR
from logic.src.utils.security import decrypt_file_data, load_key

from .base import DistanceStrategy


class GoogleMapsStrategy(DistanceStrategy):
    """Strategy for computing distances using Google Maps API."""

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes distances using the Google Maps Distance Matrix API.
        """
        assert self._eval_kwarg("env_filename", kwargs)
        env_path = os.path.join(ROOT_DIR, "env", kwargs["env_filename"])
        config = dotenv_values(env_path)
        api_key = config.get("GOOGLE_API_KEY", "")

        if api_key == "" and self._eval_kwarg("symkey_name", kwargs):
            assert self._eval_kwarg("gapik_file", kwargs)
            sym_key = load_key(kwargs["symkey_name"], kwargs["env_filename"])
            api_key = decrypt_file_data(sym_key, kwargs["gapik_file"])
        elif api_key == "" and self._eval_kwarg("gapik_file", kwargs):
            with open(kwargs["gapik_file"], "r") as gapik_file:
                api_key = gapik_file.read()
        else:
            assert api_key is not None and api_key != "", "Google API key not found."

        gmaps = googlemaps.Client(key=api_key)
        size = len(coords)
        distance_matrix = np.zeros((size, size))

        FREE_SIZE = 10
        src = dst = coords[["Lat", "Lng"]].values.tolist()

        # Batch processing
        for id_i in range(0, size, FREE_SIZE):
            for id_j in range(0, size, FREE_SIZE):
                origins = src[id_i : id_i + FREE_SIZE]
                dests = dst[id_j : id_j + FREE_SIZE]
                response = gmaps.distance_matrix(origins, dests, mode="driving", units="metric")
                for row_id, row in enumerate(response["rows"]):
                    for col_id, elem in enumerate(row["elements"]):
                        if "distance" in elem:
                            distance_matrix[id_i + row_id, id_j + col_id] = elem["distance"]["value"] / 1000

        return distance_matrix
