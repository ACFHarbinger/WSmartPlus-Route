"""
Data Processing and Transformation Pipeline for Simulation Setup.

This module implements data normalization, coordinate transformation, and
model input preparation for the WSmart-Route simulator. It bridges raw
data (CSV files) with runtime representations (PyTorch tensors, numpy arrays).

Key Responsibilities:
    - Geographic coordinate normalization (9 projection methods)
    - DataFrame preprocessing (sorting, indexing, type casting)
    - Neural model input preparation (tensors, edges, distance matrices)
    - Result persistence (Excel exports)

The SimulationDataMapper class centralizes all data transformations,
following the Data Mapper pattern to decouple data structures from
business logic.

Coordinate Normalization Methods:
    - mmn: Min-Max normalization [0, 1]
    - mun: Mean normalization
    - smsd: Standardization (z-score)
    - ecp: Equidistant Cylindrical Projection
    - wmp: Web Mercator Projection
    - hdp: Haversine Distance Projection
    - c3d: 3D Cartesian coordinates
    - s4d: 4D Spherical coordinates

Classes:
    SimulationDataMapper: Central data transformation hub
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch

from logic.src.constants import EARTH_RADIUS, EARTH_WMP_RADIUS, MAX_WASTE
from logic.src.utils.functions.graph_utils import (
    adj_to_idx,
    get_adj_knn,
    get_edge_idx_dist,
)

from .loader import load_area_and_waste_type_params, load_depot, load_simulator_data
from .network import (
    apply_edges,
    compute_distance_matrix,
    get_paths_between_states,
    haversine_distance,
)


class SimulationDataMapper:
    """
    Data Mapper for the WSmart+ Route simulator.

    Centralizes all data transformations between raw data sources (CSV/Excel)
    and runtime representations (PyTorch tensors, NumPy arrays). Implements
    the Data Mapper pattern to isolate data structure knowledge.

    Key Methods:
        - format_coordinates: Normalize geographic coords (9 projection modes)
        - process_model_input: Prepare neural model inputs (tensors, graphs)
        - setup_df: Merge depot and bin data into unified DataFrames
        - save_results: Export simulation outputs to Excel

    This class is stateless and thread-safe. All methods operate on
    input parameters without modifying internal state.
    """

    def sort_dataframe(self, df: pd.DataFrame, metric_tosort: str, ascending_order: bool = True) -> pd.DataFrame:
        """
        Sorts a DataFrame by a metric and ensures that metric is the first column.
        """
        df = df.sort_values(by=metric_tosort, ascending=ascending_order)
        columns = [metric_tosort] + [col for col in df.columns if col != metric_tosort]
        return cast(pd.DataFrame, df[columns])

    def get_df_types(self, df: pd.DataFrame, prec: str = "32") -> Dict[str, str]:
        """
        Infers and maps column data types to specific precisions (e.g., float32).
        """
        df_types = dict(df.dtypes)
        for key, val in df_types.items():
            if key == "ID":
                new_type = f"int{prec}"
            elif "obj" in str(val):
                new_type = "string"
            elif "float" in str(val) or "int" in str(val):
                new_type = str(val)[:-2] + prec
            else:
                new_type = str(val)  # Fallback
            df_types[key] = new_type
        return df_types

    def setup_df(
        self,
        depot: pd.DataFrame,
        df: pd.DataFrame,
        col_names: List[str],
        index_name: Optional[str] = "#bin",
    ) -> pd.DataFrame:
        """
        Merges depot data with bin data and sets up a unified indexing scheme.
        """
        df = df.loc[:, col_names].copy()  # Ensure copy to avoid setting on slice
        df.loc[-1] = depot.loc[0, col_names].values
        df.index = df.index + 1
        df = df.sort_index()
        if index_name is None:
            df = df.sort_values(by="ID").reset_index(drop=True).astype(self.get_df_types(df))
        else:
            df = df.sort_values(by="ID").reset_index().astype(self.get_df_types(df))
            df = df.rename(columns={"index": index_name})
            df[index_name] = df[index_name].astype(df["ID"].dtype)
        return df

    def sample_df(
        self,
        df: pd.DataFrame,
        n_elems: int,
        depot: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Samples a subset of rows from a DataFrame and optionally adds the depot.
        """
        df = df.sample(n=n_elems)
        df_types = self.get_df_types(df)
        if depot is not None:
            df.loc[0] = depot
        if output_path is not None:
            if os.path.isfile(output_path):
                with open(output_path) as fp:
                    data = json.load(fp)
                data.append(df.sort_index().index.tolist())
            else:
                data = [df.sort_index().index.tolist()]
            with open(output_path, "w") as fp:
                json.dump(data, fp)
        df = df.sort_values(by="ID").reset_index(drop=True).astype(df_types)
        return df

    def process_indices(self, df: pd.DataFrame, indices: Optional[List[int]]) -> pd.DataFrame:
        """
        Extracts a subset of rows or columns from a DataFrame based on indices.
        """
        if indices is None:
            df = df.copy()
        else:
            if "index" in df.columns or "ID" in df.columns:
                df = df.iloc[indices]
                df = df.sort_values(by="ID").reset_index(drop=True).astype(self.get_df_types(df))
            else:
                df = df.iloc[:, indices]
        return df

    def process_raw_data(
        self,
        data: pd.DataFrame,
        bins_coordinates: pd.DataFrame,
        depot: pd.DataFrame,
        indices: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Higher-level method to filter and prepare both statistic and coordinate data.
        """
        new_data = self.process_indices(data, indices)
        coords = self.process_indices(bins_coordinates, indices)
        coords = self.setup_df(depot, coords, ["ID", "Lat", "Lng"])
        new_data = self.setup_df(depot, new_data, ["ID", "Stock", "Accum_Rate"])
        return new_data, coords

    def format_coordinates(self, coords: Any, method: str, col_names: Optional[List[str]] = None) -> Tuple[Any, Any]:
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

        # Type guard: col_names is now definitely not None
        assert col_names is not None

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
                else:
                    coords = coords[:, :, [1, 0]]
                    min_arr = np.min(coords, axis=1, keepdims=True)
                    max_arr = np.max(coords, axis=1, keepdims=True)
                    center_meridian = (max_arr + min_arr) / 2

                lat_lower, lat_upper = per_func(lat, 10), per_func(lat, 90)
                center_parallel = (lat_upper + lat_lower) / 2
                offset = (lat_upper - lat_lower) / 2
                pscale = (
                    np.cos(np.radians(center_parallel - offset)) + np.cos(np.radians(center_parallel + offset))
                ) / 2
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
                    lat = (
                        (lat - lat.mean()) / (lat.max() - lat.min()) if lat.max() != lat.min() else lat
                    )  # Careful with div by zero?
                    # Revert to original code logic:
                    lat = (lat - lat.min()) / (lat.max() - lat.min())
                    lng = (lng - lng.min()) / (lng.max() - lng.min())
                else:
                    coords = coords[:, :, [1, 0]]
                    min_arr = np.min(coords, axis=1, keepdims=True)
                    max_arr = np.max(coords, axis=1, keepdims=True)
                    coords = (coords - min_arr) / (max_arr - min_arr)

            if IS_PANDAS:
                depot = np.array([lng.iloc[0], lat.iloc[0]])
                loc = np.array([[x, y] for x, y in zip(lng.iloc[1:], lat.iloc[1:])])
            else:
                depot = coords[:, 0, :]
                loc = coords[:, 1:, :]

        return depot, loc

    def process_model_input(
        self,
        coordinates,
        dist_matrix,
        device,
        method,
        configs,
        edge_threshold,
        edge_method,
        area,
        waste_type,
        adj_matrix=None,
    ):
        """
        Prepares and normalizes input data for neural model consumption.

        Converts raw coordinates and distance matrices into PyTorch tensors,
        applies graph sparsification, and loads problem parameters.
        """
        problem_size = len(dist_matrix) - 1
        depot, loc = self.format_coordinates(coordinates, method)
        model_data = {
            "locs": torch.as_tensor(loc, dtype=torch.float32),
            "depot": torch.as_tensor(depot, dtype=torch.float32),
            "waste": torch.zeros(problem_size),
        }

        if configs.get("problem") in ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp"]:
            model_data["max_waste"] = torch.as_tensor(MAX_WASTE, dtype=torch.float32)
        else:
            # Default fallback or error if problem strictly required? Original code raised ValueError
            if "problem" in configs:
                raise ValueError(f"Unknown problem: {configs['problem']}")

        if "model" in configs and configs["model"] in ["tam"]:
            model_data["fill_history"] = torch.zeros((1, configs["graph_size"], configs["temporal_horizon"]))

        if edge_threshold > 0 and edge_threshold < 1:
            if edge_method == "dist":
                edges = (
                    torch.tensor(adj_to_idx(adj_matrix, negative=False))
                    if adj_matrix is not None
                    else torch.tensor(get_edge_idx_dist(dist_matrix[1:, 1:], edge_threshold))
                )
            else:
                assert edge_method == "knn"
                edges = (
                    torch.from_numpy(adj_matrix)
                    if adj_matrix is not None
                    else torch.from_numpy(get_adj_knn(dist_matrix[1:, 1:], edge_threshold, negative=False))
                )

            dtype = torch.float32 if "encoder" in configs and configs["encoder"] in ["gac", "tgc"] else torch.bool
            edges = edges.unsqueeze(0).to(device, dtype=dtype)
        else:
            edges = None

        (
            VEHICLE_CAPACITY,
            REVENUE_KG,
            DENSITY,
            COST_KM,
            VOLUME,
        ) = load_area_and_waste_type_params(area, waste_type)
        BIN_CAPACITY = VOLUME * DENSITY
        VEHICLE_CAPACITY = VEHICLE_CAPACITY / 100
        profit_vars = {
            "cost_km": COST_KM,
            "revenue_kg": REVENUE_KG,
            "bin_capacity": BIN_CAPACITY,
            "vehicle_capacity": VEHICLE_CAPACITY,
        }

        if isinstance(dist_matrix, torch.Tensor):
            dm_tensor = dist_matrix.float().to(device)
        else:
            dm_tensor = torch.from_numpy(dist_matrix).float().to(device)

        return (
            {key: val.unsqueeze(0) for key, val in model_data.items()},
            (edges, dm_tensor),
            profit_vars,
        )

    def save_results(self, matrix, results_dir, seed, data_dist, policy, sample_id):
        """
        Exports simulation fill history to Excel files.
        """
        parent_dir = os.path.join(results_dir, "fill_history", data_dist)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        fills_filepath = os.path.join(parent_dir, f"enchimentos_seed{seed}_sample{sample_id}.xlsx")
        if os.path.exists(fills_filepath) and os.path.isfile(fills_filepath):
            return

        df = pd.DataFrame(matrix).transpose()
        filepath = os.path.join(parent_dir, f"{policy}{seed}_sample{sample_id}.xlsx")
        df.to_excel(filepath, index=False, header=False)


_mapper = SimulationDataMapper()


def sort_dataframe(df, metric_tosort, ascending_order=True):
    """Wrapper for SimulationDataMapper.sort_dataframe."""
    return _mapper.sort_dataframe(df, metric_tosort, ascending_order)


def get_df_types(df, prec="32"):
    """Wrapper for SimulationDataMapper.get_df_types."""
    return _mapper.get_df_types(df, prec)


def setup_df(depot, df, col_names, index_name="#bin"):
    """Wrapper for SimulationDataMapper.setup_df."""
    return _mapper.setup_df(depot, df, col_names, index_name)


def sample_df(df, n_elems, depot=None, output_path=None):
    """Wrapper for SimulationDataMapper.sample_df."""
    return _mapper.sample_df(df, n_elems, depot, output_path)


def process_indices(df, indices):
    """Wrapper for SimulationDataMapper.process_indices."""
    return _mapper.process_indices(df, indices)


def process_data(data, bins_coordinates, depot, indices=None):
    """Wrapper for SimulationDataMapper.process_raw_data."""
    return _mapper.process_raw_data(data, bins_coordinates, depot, indices)


def haversine_distance_old(lat1, lng1, lat2, lng2):
    """
    Deprecated: use network.haversine_distance.
    Included for backward compatibility with older simulation scripts.
    """
    return haversine_distance(lat1, lng1, lat2, lng2)


def process_coordinates(coords, method, col_names=["Lat", "Lng"]):
    """Wrapper for SimulationDataMapper.format_coordinates."""
    return _mapper.format_coordinates(coords, method, col_names)


def process_model_data(
    coordinates,
    dist_matrix,
    device,
    method,
    configs,
    edge_threshold,
    edge_method,
    area,
    waste_type,
    adj_matrix=None,
):
    """Wrapper for SimulationDataMapper.process_model_input."""
    return _mapper.process_model_input(
        coordinates,
        dist_matrix,
        device,
        method,
        configs,
        edge_threshold,
        edge_method,
        area,
        waste_type,
        adj_matrix,
    )


def create_dataframe_from_matrix(matrix):
    """
    Converts a simulation result matrix into a formatted DataFrame for reporting.
    """
    enchimentos = [row[-1] for row in matrix]
    ids_rota = np.arange(len(matrix))
    data = pd.DataFrame({"#bin": ids_rota, "Stock": enchimentos, "Accum_Rate": np.zeros(len(ids_rota))})
    return data


def convert_to_dict(bins_coordinates):
    """
    Converts a coordinates DataFrame into a dictionary indexed by bin ID.
    Used for mapping-related visualizations.
    """
    coordinates_dict = {}
    for _, row in bins_coordinates.iterrows():
        bin_id = row["ID"]
        lat = np.float64(row["Lat"])
        lng = np.float64(row["Lng"])
        coordinates_dict[bin_id] = {"lat": lat, "lng": lng}
    return coordinates_dict


def save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id):
    """Wrapper for SimulationDataMapper.save_results."""
    return _mapper.save_results(matrix, results_dir, seed, data_dist, policy, sample_id)


def setup_basedata(n_bins, data_dir, area, waste_type):
    """
    High-level initialization sequence to load all required simulation data.
    """
    depot = load_depot(data_dir, area)
    data, bins_coordinates = load_simulator_data(data_dir, n_bins, area, waste_type)
    assert data.shape == bins_coordinates.shape
    return data, bins_coordinates, depot


def setup_dist_path_tup(
    bins_coordinates,
    size,
    dist_method,
    dm_filepath,
    env_filename,
    gapik_file,
    symkey_name,
    device,
    edge_thresh,
    edge_method,
    focus_idx=None,
):
    """
    Combined setup for distance matrix, shortest paths, and sparsification.
    """
    dist_matrix = compute_distance_matrix(
        bins_coordinates,
        dist_method,
        dm_filepath=dm_filepath,
        env_filename=env_filename,
        gapik_file=gapik_file,
        symkey_name=symkey_name,
        focus_idx=focus_idx,
    )
    dist_matrix_edges, shortest_paths, adj_matrix = apply_edges(dist_matrix, edge_thresh, edge_method)
    paths = get_paths_between_states(size + 1, shortest_paths)
    dm_tensor = torch.from_numpy(dist_matrix_edges / 100.0).to(device)
    distC = np.round(dist_matrix_edges * 10).astype("int32")
    return (dist_matrix_edges, paths, dm_tensor, distC), adj_matrix
