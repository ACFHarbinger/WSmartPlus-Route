"""
Data Processing and Transformation Pipeline.

This package provides coordinate formatting, DataFrame manipulation,
and model-input preparation for the WSmart+ Route simulator.

Submodules
----------
- ``formatting``:  Coordinate normalization methods (mmn, wmp, hdp, …)
- ``mapper``:      ``SimulationDataMapper`` class
- ``dataframes``:  DataFrame helpers (sort, sample, convert, export)
- ``processing``:  Data transform wrappers (process_data, process_coordinates, process_model_data)
- ``setup``:       Simulation bootstrap (setup_basedata, setup_dist_path_tup)

Attributes:
    dataframes: DataFrame helpers (sort, sample, convert, export)
    formatting: Coordinate normalization methods (mmn, wmp, hdp, …)
    mapper:      ``SimulationDataMapper`` class
    processing:  Data transform wrappers (process_data, process_coordinates, process_model_data)
    setup:       Simulation bootstrap (setup_basedata, setup_dist_path_tup)

Example:
    >>> from logic.src.data.processor.dataframes import convert_to_dict
    >>> convert_to_dict(pd.DataFrame({"Lng": [0, 1], "Lat": [0, 1]}))
    {'Lng': [0, 1], 'Lat': [0, 1]}
"""

from .dataframes import (
    convert_to_dict,
    create_dataframe_from_matrix,
    get_df_types,
    process_indices,
    sample_df,
    save_matrix_to_excel,
    setup_df,
    sort_dataframe,
)
from .formatting import format_coordinates
from .mapper import SimulationDataMapper
from .processing import process_coordinates, process_data, process_model_data
from .setup import setup_basedata, setup_dist_path_tup

__all__ = [
    "SimulationDataMapper",
    "format_coordinates",
    "sort_dataframe",
    "get_df_types",
    "setup_df",
    "sample_df",
    "process_indices",
    "process_data",
    "process_coordinates",
    "process_model_data",
    "create_dataframe_from_matrix",
    "convert_to_dict",
    "save_matrix_to_excel",
    "setup_basedata",
    "setup_dist_path_tup",
]
