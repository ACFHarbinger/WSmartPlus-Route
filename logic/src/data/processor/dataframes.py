"""DataFrame manipulation utilities for simulation data."""

import numpy as np
import pandas as pd

from ._logging import _log_processor_event
from .mapper import SimulationDataMapper

_mapper = SimulationDataMapper()


def sort_dataframe(df, metric_tosort, ascending_order=True):
    """Sort dataframe by a given metric column.

    Args:
        df: DataFrame to sort.
        metric_tosort: Column name to sort by.
        ascending_order: Sort ascending if True.

    Returns:
        Sorted DataFrame with the metric column first.
    """
    result = _mapper.sort_dataframe(df, metric_tosort, ascending_order)
    _log_processor_event("sort_dataframe", variable_name="dataframe", metric=metric_tosort, ascending=ascending_order)
    return result


def get_df_types(df, prec="32"):
    """Infer and map column data types to specific precisions.

    Args:
        df: DataFrame to inspect.
        prec: Numeric precision suffix (e.g. '32', '64').

    Returns:
        Dict mapping column names to type strings.
    """
    result = _mapper.get_df_types(df, prec)
    _log_processor_event("get_df_types", variable_name="df_types", prec=prec)
    return result


def setup_df(depot, df, col_names, index_name="#bin"):
    """Merge depot and bin data into a single DataFrame.

    Args:
        depot: Depot DataFrame.
        df: Bins DataFrame.
        col_names: Columns to keep.
        index_name: Name for the index column.

    Returns:
        Combined DataFrame sorted by ID.
    """
    result = _mapper.setup_df(depot, df, col_names, index_name)
    _log_processor_event("setup_df", variable_name="dataframe", col_names=col_names, index_name=index_name)
    return result


def sample_df(df, n_elems, depot=None, output_path=None):
    """Sample a subset of bins from a DataFrame.

    Args:
        df: Source DataFrame.
        n_elems: Number of elements to sample.
        depot: Optional depot row to include.
        output_path: Optional path to persist sampled indices.

    Returns:
        Sampled DataFrame.
    """
    result = _mapper.sample_df(df, n_elems, depot, output_path)
    _log_processor_event("sample_df", variable_name="dataframe", n_elems=n_elems, output_path=output_path)
    return result


def process_indices(df, indices):
    """Extract a subset of rows based on indices.

    Args:
        df: Source DataFrame.
        indices: List of row indices.

    Returns:
        Filtered DataFrame.
    """
    result = _mapper.process_indices(df, indices)
    _log_processor_event(
        "process_indices", variable_name="indices", n_indices=len(indices) if indices is not None else 0
    )
    return result


def create_dataframe_from_matrix(matrix):
    """Create a bin-level DataFrame from a fill-history matrix.

    Args:
        matrix: List of rows; last element per row is current stock.

    Returns:
        DataFrame with columns ``#bin``, ``Stock``, ``Accum_Rate``.
    """
    enchimentos = [row[-1] for row in matrix]
    ids_rota = np.arange(len(matrix))
    result = pd.DataFrame({"#bin": ids_rota, "Stock": enchimentos, "Accum_Rate": np.zeros(len(ids_rota))})
    _log_processor_event("create_dataframe_from_matrix", variable_name="dataframe", num_rows=len(matrix))
    return result


def convert_to_dict(bins_coordinates):
    """Convert a coordinates DataFrame to a ``{ID: {lat, lng}}`` dict.

    Args:
        bins_coordinates: DataFrame with ``ID``, ``Lat``, ``Lng`` columns.

    Returns:
        Dict mapping bin ID to coordinate dict.
    """
    coordinates_dict = {}
    for _, row in bins_coordinates.iterrows():
        coordinates_dict[row["ID"]] = {"lat": np.float64(row["Lat"]), "lng": np.float64(row["Lng"])}
    _log_processor_event("convert_to_dict", variable_name="coordinates_dict", num_bins=len(bins_coordinates))
    return coordinates_dict


def save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id):
    """Export a fill-history matrix to an Excel file.

    Args:
        matrix: Fill-history data.
        results_dir: Base output directory.
        seed: Random seed used.
        data_dist: Distribution name.
        policy: Policy name.
        sample_id: Sample identifier.
    """
    result = _mapper.save_results(matrix, results_dir, seed, data_dist, policy, sample_id)
    _log_processor_event(
        "save_matrix_to_excel",
        variable_name="excel_file",
        event_type="save",
        seed=seed,
        data_dist=data_dist,
        policy=policy,
        sample_id=sample_id,
    )
    return result
