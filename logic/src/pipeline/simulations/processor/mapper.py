"""
Data Mapper for transformation between raw data and simulation representations.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, cast

import pandas as pd
import torch

from logic.src.constants import MAX_WASTE
from logic.src.utils.graph import (
    adj_to_idx,
    get_adj_knn,
    get_edge_idx_dist,
)

from .formatting import format_coordinates


class SimulationDataMapper:
    """
    Data Mapper for the WSmart+ Route simulator.
    """

    def sort_dataframe(self, df: pd.DataFrame, metric_tosort: str, ascending_order: bool = True) -> pd.DataFrame:
        """Sorts a DataFrame by a metric."""
        df = df.sort_values(by=metric_tosort, ascending=ascending_order)
        columns = [metric_tosort] + [col for col in df.columns if col != metric_tosort]
        return cast(pd.DataFrame, df[columns])

    def get_df_types(self, df: pd.DataFrame, prec: str = "32") -> Dict[str, str]:
        """Infers and maps column data types to specific precisions."""
        df_types = dict(df.dtypes)
        for key, val in df_types.items():
            if key == "ID":
                new_type = f"int{prec}"
            elif "obj" in str(val):
                new_type = "string"
            elif "float" in str(val) or "int" in str(val):
                new_type = str(val)[:-2] + prec
            else:
                new_type = str(val)
            df_types[key] = new_type
        return df_types

    def setup_df(
        self,
        depot: pd.DataFrame,
        df: pd.DataFrame,
        col_names: List[str],
        index_name: Optional[str] = "#bin",
    ) -> pd.DataFrame:
        """Merges depot and bin data."""
        df = df.loc[:, col_names].copy()
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
        """Samples a subset of bins."""
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
        return df.sort_values(by="ID").reset_index(drop=True).astype(df_types)

    def process_indices(self, df: pd.DataFrame, indices: Optional[List[int]]) -> pd.DataFrame:
        """Extracts subset based on indices."""
        if indices is None:
            return df.copy()
        if "index" in df.columns or "ID" in df.columns:
            sampled_df = df.iloc[indices]
            if isinstance(sampled_df, pd.Series):
                sampled_df = sampled_df.to_frame().T
            return sampled_df.sort_values(by="ID").reset_index(drop=True).astype(self.get_df_types(sampled_df))
        sampled_df = df.iloc[:, indices]
        if isinstance(sampled_df, pd.Series):
            sampled_df = sampled_df.to_frame().T
        return sampled_df

    def process_raw_data(
        self,
        data: pd.DataFrame,
        bins_coordinates: pd.DataFrame,
        depot: pd.DataFrame,
        indices: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares coordinate and statistic data."""
        new_data = self.process_indices(data, indices)
        coords = self.process_indices(bins_coordinates, indices)
        coords = self.setup_df(depot, coords, ["ID", "Lat", "Lng"])
        new_data = self.setup_df(depot, new_data, ["ID", "Stock", "Accum_Rate"])
        return new_data, coords

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
        """Prepares data for neural models."""
        problem_size = len(dist_matrix) - 1
        depot, loc = format_coordinates(coordinates, method)
        model_data = {
            "locs": torch.as_tensor(loc, dtype=torch.float32),
            "depot": torch.as_tensor(depot, dtype=torch.float32),
            "waste": torch.zeros(problem_size),
        }

        if configs.get("problem") in ["vrpp", "cvrpp", "wcvrp", "cwcvrp", "sdwcvrp"]:
            model_data["max_waste"] = torch.as_tensor(MAX_WASTE, dtype=torch.float32)
        elif "problem" in configs:
            raise ValueError(f"Unknown problem: {configs['problem']}")

        if configs.get("model") == "tam":
            model_data["fill_history"] = torch.zeros((1, configs["graph_size"], configs["temporal_horizon"]))

        if 0 < edge_threshold < 1:
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

            dtype = torch.float32 if configs.get("encoder") in ["gac", "tgc"] else torch.bool
            edges = edges.unsqueeze(0).to(device, dtype=dtype)
        else:
            edges = None

        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        VEHICLE_CAPACITY, REVENUE_KG, DENSITY, COST_KM, VOLUME = load_area_and_waste_type_params(area, waste_type)

        profit_vars = {
            "cost_km": COST_KM,
            "revenue_kg": REVENUE_KG,
            "bin_capacity": VOLUME * DENSITY,
            "vehicle_capacity": VEHICLE_CAPACITY / 100,
        }

        dm_tensor = torch.as_tensor(dist_matrix, dtype=torch.float32, device=device)
        return ({k: v.unsqueeze(0) for k, v in model_data.items()}, (edges, dm_tensor), profit_vars)

    def save_results(self, matrix, results_dir, seed, data_dist, policy, sample_id):
        """Exports history to Excel."""
        path = os.path.join(results_dir, "fill_history", data_dist)
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{policy}{seed}_sample{sample_id}.xlsx")
        if not os.path.exists(filepath):
            pd.DataFrame(matrix).transpose().to_excel(filepath, index=False, header=False)
