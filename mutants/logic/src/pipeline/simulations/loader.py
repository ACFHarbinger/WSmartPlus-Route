"""
Data Loading and Repository Pattern for Simulation Initialization.

This module implements the Repository Pattern to abstract data access for
simulation setup. It loads geographic coordinates, waste statistics, and
area-specific parameters from the filesystem.

Architecture:
    - SimulationRepository: Abstract interface for data access
    - FileSystemRepository: Concrete implementation using CSV/Excel files
    - Module-level functions: Convenience wrappers for backward compatibility
    - Historical waste data (fill rates, accumulation)
    - Area parameters (vehicle capacity, revenue, density, expenses)
    - Depot locations (waste processing facilities)

Geographic Areas:
    - Rio Maior (Portugal): 317 bins, real sensor data
    - Figueira da Foz: 1094 bins
    - Mix RM/BAC: Virtual test instances (20-225 bins)

Classes:
    SimulationRepository: Abstract data access interface
    FileSystemRepository: File-based data loader
"""

import json
import os
from abc import ABC, abstractmethod

import pandas as pd

import logic.src.constants as udef

# from logic.src.utils.data.data_utils import load_area_and_waste_type_params as _load_area_params


class SimulationRepository(ABC):
    """
    Abstract interface for simulation data access.

    Defines the contract for loading geographic, waste, and configuration
    data required to initialize simulations. Implementations can source
    data from files, databases, APIs, or other backends.

    Methods:
        get_indices: Load or generate bin subset indices
        get_depot: Retrieve depot (facility) coordinates
        get_simulator_data: Load waste statistics and bin coordinates
        get_area_params: Get area-specific parameters (capacity, revenue, etc.)
    """

    @abstractmethod
    def get_indices(self, filename, n_samples, n_nodes, data_size, lock=None):
        """
        Loads or generates a list of bin indices for simulation samples.

        Args:
            filename: JSON file containing pre-selected indices.
            n_samples: Number of random samples to generate if file missing.
            n_nodes: Number of bins per sample.
            data_size: Total number of bins available in the dataset.
            lock: Optional multiprocessing Lock for file I/O.

        Returns:
            List[List[int]]: Nested list of bin indices.
        """
        pass

    @abstractmethod
    def get_depot(self, area, data_dir=None):
        """
        Retrieves the depot coordinates for a given area.

        Args:
            area: Name of the geographic area.
            data_dir: Optional override for the data directory.

        Returns:
            pd.DataFrame: DataFrame containing depot ID, Lat, Lng, and metadata.
        """
        pass

    @abstractmethod
    def get_simulator_data(
        self,
        number_of_bins,
        area="Rio Maior",
        waste_type=None,
        lock=None,
        data_dir=None,
    ):
        """
        Loads waste statistics and coordinate data for the simulator.

        Args:
            number_of_bins: Target number of nodes.
            area: Geographic area name.
            waste_type: Waste stream type.
            lock: Optional multiprocessing Lock.
            data_dir: Optional override for the data directory.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (Waste statistics, Bin coordinates).
        """
        pass

    @abstractmethod
    def get_area_params(self, area, waste_type):
        """
        Retrieves area and waste-type specific simulation parameters.

        Args:
            area: Geographic area name.
            waste_type: Waste stream type.

        Returns:
            Tuple with capacity, revenue, density, expenses, volume.
        """
        pass


class FileSystemRepository(SimulationRepository):
    """
    File-based implementation of SimulationRepository.

    Loads data from CSV, Excel, and JSON files stored in the project's
    data/wsr_simulator directory. Supports multiple geographic areas
    and waste types with area-specific file naming conventions.

    Attributes:
        default_data_dir: Root directory for simulation data files
    """

    def __init__(self, data_root_dir):
        """
        Initialize the repository.

        Args:
            data_root_dir: Root directory path for data resolution.
        """
        self.default_data_dir = os.path.join(data_root_dir, "data", "wsr_simulator")

    def _get_data_dir(self, override_dir=None):
        return override_dir if override_dir else self.default_data_dir

    def get_indices(self, filename, n_samples, n_nodes, data_size, lock=None):
        """
        Implementation of get_indices that persists generated indices to JSON.
        """
        graphs_file_path = os.path.join(self.default_data_dir, "bins_selection", filename)
        if os.path.isfile(graphs_file_path):
            if lock is not None:
                lock.acquire(timeout=udef.LOCK_TIMEOUT)
            try:
                with open(graphs_file_path) as fp:
                    indices = json.load(fp)
            finally:
                if lock is not None:
                    lock.release()
                if len(indices) == 1 and n_samples > 1:
                    indices *= n_samples
        else:
            df = pd.Series(range(data_size))
            indices = []
            for _ in range(n_samples):
                data = df.sample(n=n_nodes).to_list()
                data.sort()
                while len(indices) > 0 and data in indices:
                    data = df.sample(n=n_nodes).to_list()
                    data.sort()
                indices.append(data)

            if lock is not None:
                lock.acquire(timeout=udef.LOCK_TIMEOUT)
            try:
                with open(graphs_file_path, "w") as fp:
                    fp.write("[\n")
                    fp.write(",\n".join(json.dumps(idx) for idx in indices))
                    fp.write("\n]")
            finally:
                if lock is not None:
                    lock.release()
        return indices

    def get_depot(self, area, data_dir=None):
        """
        Implementation of get_depot that reads from Facilities.csv.
        """
        src_area = area.translate(str.maketrans("", "", "-_ ")).lower()
        d_dir = self._get_data_dir(data_dir)
        facilities = pd.read_csv(os.path.join(d_dir, "coordinates", "Facilities.csv"))
        depot_df = (
            facilities[facilities["Sigla"] == udef.MAP_DEPOTS[src_area]].loc[:, ["Lat", "Lng"]].reset_index(drop=True)
        )
        depot_df.insert(0, "ID", [0])
        new_cols = pd.DataFrame({"Stock": [0], "Accum_Rate": [0]})
        return pd.concat([depot_df, new_cols], axis=1)

    def get_simulator_data(
        self,
        number_of_bins,
        area="Rio Maior",
        waste_type=None,
        lock=None,
        data_dir=None,
    ):
        """
        Implementation of get_simulator_data that handles area-specific file logic.
        """
        d_dir = self._get_data_dir(data_dir)

        def _preprocess_county_date(data, date_str="Date"):
            data[date_str] = pd.to_datetime(data[date_str], format="%Y-%m-%d")
            data = data.set_index(date_str)
            data = data.round()
            data.columns = data.columns.astype(int)
            return data

        def _preprocess_county_data(data):
            def __get_stock(col):
                """Helper to extract initial stock from time series column."""
                positive_values = col[col >= 1e-32].dropna()
                if not positive_values.empty:
                    return positive_values.iloc[0]
                return 0

            accum_rate = data.clip(lower=0).fillna(0).mean()
            stock = data.apply(__get_stock, axis=0)
            new_data = pd.DataFrame({"ID": data.columns})
            new_data["Stock"] = new_data["ID"].map(stock)
            new_data["Accum_Rate"] = new_data["ID"].map(accum_rate)
            new_data[["Stock", "Accum_Rate"]] = (
                new_data[["Stock", "Accum_Rate"]] - new_data[["Stock", "Accum_Rate"]].min()
            ) / (new_data[["Stock", "Accum_Rate"]].max() - new_data[["Stock", "Accum_Rate"]].min())
            return new_data

        src_area = area.translate(str.maketrans("", "", "-_ ")).lower()
        wtype = waste_type if waste_type is None else udef.WASTE_TYPES[waste_type]
        if lock is not None:
            lock.acquire(timeout=udef.LOCK_TIMEOUT)
        try:
            if src_area == "mixrmbac":
                if number_of_bins <= 20:
                    data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate - small.xlsx"))
                    bins_coordinates = pd.read_excel(os.path.join(d_dir, "coordinates", "Coordinates - small.xlsx"))
                elif number_of_bins <= 50 and number_of_bins > 20:
                    data = pd.read_excel(
                        os.path.join(
                            d_dir,
                            "bins_waste",
                            "StockAndAccumulationRate - 50bins.xlsx",
                        )
                    )
                    bins_coordinates = pd.read_excel(os.path.join(d_dir, "coordinates", "Coordinates - 50bins.xlsx"))
                else:
                    assert number_of_bins <= 225, f"Number of bins for area {src_area} must be <= 225"
                    data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate.xlsx"))
                    bins_coordinates = pd.read_excel(os.path.join(d_dir, "coordinates", "Coordinates.xlsx"))
            elif src_area == "riomaior":
                assert number_of_bins <= 317, f"Number of bins for area {src_area} must be <= 317"
                if number_of_bins == 104:
                    df = pd.read_csv(
                        os.path.join(
                            d_dir,
                            "bins_waste",
                            "Rio_Maior_Sensores_2021_2024_cleaned_104.csv",
                        )
                    )
                    df["Data da leitura"] = pd.to_datetime(df["Data da leitura"], format="%Y-%m-%d")
                    data = df.pivot_table(
                        index="Data da leitura",
                        columns="idcontentor",
                        values="Enchimento",
                        aggfunc="mean",
                    )
                    data = data.round()
                    data.columns = data.columns.astype(int)
                    coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "coordinates104.csv"))
                    coords_tmp = coords_tmp.rename(columns={"Lon": "Lng", "Longitude": "Lng", "Latitude": "Lat"})
                else:
                    data = _preprocess_county_date(
                        pd.read_csv(
                            os.path.join(
                                d_dir,
                                "bins_waste",
                                f"old_out_crude_rate[{src_area}].csv",
                            )
                        )
                    )
                    coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", f"old_out_info[{src_area}].csv"))
                    coords_tmp = coords_tmp.rename(columns={"Latitude": "Lat", "Longitude": "Lng"})
                    if wtype:
                        coords_tmp = coords_tmp[coords_tmp["Tipo de Residuos"] == wtype]
                bins_coordinates = coords_tmp[["ID", "Lat", "Lng"]]
                data = _preprocess_county_data(data)
                data = data[data["ID"].isin(bins_coordinates["ID"])]
            elif src_area == "figueiradafoz":
                data = _preprocess_county_date(pd.read_csv(os.path.join(d_dir, "out_crude_rate[figdafoz].csv")))
                assert number_of_bins <= 1094, f"Number of bins for area {src_area} must be <= 1094"
                coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "out_info[figdafoz].csv"))
                coords_tmp = coords_tmp.rename(columns={"Latitude": "Lat", "Longitude": "Lng"})
                if wtype:
                    coords_tmp = coords_tmp[coords_tmp["Tipo de Residuos"] == wtype]
                bins_coordinates = coords_tmp[["ID", "Lat", "Lng"]]
                data = _preprocess_county_data(data)
                data = data[data["ID"].isin(bins_coordinates["ID"])]
            else:
                assert src_area == "both", f"Invalid area: {src_area}"
                wsrs_data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate.xlsx"))
                wsba_data = _preprocess_county_data(
                    _preprocess_county_date(
                        pd.read_csv(
                            os.path.join(
                                d_dir,
                                "bins_waste",
                                f"old_out_crude_rate[{src_area}].csv",
                            )
                        )
                    )
                )
                if number_of_bins <= 57:
                    coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "intersection.csv"))
                    bins_coordinates = coords_tmp.drop("ID317", axis=1).rename(columns={"ID225": "ID"})
                    data = wsrs_data[wsrs_data["ID"].isin(bins_coordinates["ID"])]
                elif number_of_bins <= 371:
                    coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "merged.csv"))
                    data = wsrs_data[wsrs_data["ID"].isin(coords_tmp["ID225"])]
                    data_tmp = wsba_data[wsba_data["ID"].isin(coords_tmp.loc[coords_tmp["ID225"].isna(), "ID317"])]
                    data = pd.concat([data, data_tmp], axis=0)
                    bins_coordinates = pd.DataFrame(
                        {
                            "ID": coords_tmp["ID225"].fillna(coords_tmp["ID317"]),
                            "Lat": coords_tmp["Lat"],
                            "Lng": coords_tmp["Lng"],
                        }
                    )
                elif number_of_bins <= 485:
                    coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "union.csv"))
                    data = wsba_data[wsba_data["ID"].isin(coords_tmp["ID317"])]
                    data_tmp = wsrs_data[wsrs_data["ID"].isin(coords_tmp.loc[coords_tmp["ID317"].isna(), "ID225"])]
                    data = pd.concat([data, data_tmp], axis=0)
                    bins_coordinates = pd.DataFrame(
                        {
                            "ID": coords_tmp["ID317"].fillna(coords_tmp["ID225"]),
                            "Lat": coords_tmp["Lat"],
                            "Lng": coords_tmp["Lng"],
                        }
                    )
                else:
                    assert number_of_bins <= 542, f"Number of bins for {src_area} must be <= 542"
                    bins_coordinates = pd.read_csv(os.path.join(d_dir, "coordinates", f"old_out_info[{src_area}].csv"))
                    bins_coordinates = bins_coordinates.rename(columns={"Latitude": "Lat", "Longitude": "Lng"})
                    bins_coordinates = bins_coordinates[["ID", "Lat", "Lng"]]
                    coords_tmp = pd.read_excel(os.path.join(d_dir, "Coordinates.xlsx"))

                    data = wsba_data[wsba_data["ID"].isin(bins_coordinates["ID"])]
                    data_tmp = wsrs_data[wsrs_data["ID"].isin(coords_tmp["ID"])]

                    # Change ID since bin with same ID (but diff coords) exists in out_INFO.csv
                    coords_tmp.loc[coords_tmp["ID"] == 1610, "ID"] = coords_tmp.iloc[-1]["ID"] + 1
                    data_tmp.loc[data_tmp["ID"] == 1610, "ID"] = data_tmp.iloc[-1]["ID"] + 1
                    bins_coordinates = pd.concat([bins_coordinates, coords_tmp])
                    data = pd.concat([data, data_tmp])
                data = data[data["ID"].isin(bins_coordinates["ID"])]
            bins_coordinates = bins_coordinates[bins_coordinates["ID"].isin(data["ID"])]
        finally:
            if lock is not None:
                lock.release()
        return data.sort_values(by="ID").reset_index(drop=True), bins_coordinates.sort_values(by="ID").reset_index(
            drop=True
        )

    def get_area_params(self, area, waste_type):
        """
        Retrieves area and waste-type specific simulation parameters.
        Delegates to data_utils.load_area_and_waste_type_params.
        """
        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        return load_area_and_waste_type_params(area, waste_type)


# Singleton repository instance
_repository = FileSystemRepository(udef.ROOT_DIR)


def load_indices(filename, n_samples, n_nodes, data_size, lock=None):
    """
    Convenience wrapper to load indices from the singleton repository.
    """
    return _repository.get_indices(filename, n_samples, n_nodes, data_size, lock)


def load_depot(data_dir, area="Rio Maior"):
    """
    Convenience wrapper to load depot coords from the singleton repository.
    """
    return _repository.get_depot(area, data_dir=data_dir)


def load_simulator_data(data_dir, number_of_bins, area="Rio Maior", waste_type=None, lock=None):
    """
    Convenience wrapper to load simulator data from the singleton repository.
    """
    return _repository.get_simulator_data(number_of_bins, area, waste_type, lock, data_dir=data_dir)


def load_area_and_waste_type_params(area, waste_type):
    """
    Convenience wrapper to load area params.
    """
    from logic.src.utils.data.data_utils import load_area_and_waste_type_params

    return load_area_and_waste_type_params(area, waste_type)
