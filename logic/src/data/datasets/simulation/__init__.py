"""
Simulation dataset classes for WSmart-Route.

Attributes:
    GenerativeDataset: Generative dataset for simulation.
    NumpyDictDataset: NumpyDict dataset for simulation.
    NumpyPickleDataset: NumpyPickle dataset for simulation.
    PandasCsvDataset: PandasCsv dataset for simulation.
    PandasExcelDataset: PandasExcel dataset for simulation.
    SimulationDataset: Simulation dataset for simulation.

Example:
    >>> from logic.src.data.datasets.simulation import NumpyPickleDataset, GenerativeDataset
    >>> # Load pre-recorded simulation dataset
    >>> dataset = NumpyPickleDataset.load(
    ...     "path/to/data.pkl",
    ...     area="candaval",
    ...     waste_type="residuo"
    ... )
    >>> # Or generate waste simulation data on-the-fly
    >>> gen_dataset = GenerativeDataset(
    ...     data_dir="data",
    ...     n_samples=100,
    ...     n_days=31,
    ...     n_bins=50
    ... )
"""

from .gen_dataset import GenerativeDataset
from .np_pkl_dataset import NumpyPickleDataset
from .npz_dataset import NumpyDictDataset
from .pd_csv_dataset import PandasCsvDataset
from .pd_xlsx_dataset import PandasExcelDataset
from .sim_dataset import SimulationDataset

__all__ = [
    "GenerativeDataset",
    "NumpyDictDataset",
    "NumpyPickleDataset",
    "PandasCsvDataset",
    "PandasExcelDataset",
    "SimulationDataset",
]
