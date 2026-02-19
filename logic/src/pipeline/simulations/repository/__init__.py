"""
Repository Pattern for Simulation data access.
"""

from typing import Union

from .base import SimulationRepository
from .filesystem import FileSystemRepository
from .npz import NumpyDictRepository
from .xlsx import PandasExcelRepository

# Singleton repository instance
_REPOSITORY = None


def set_repository(repo: Union[FileSystemRepository, NumpyDictRepository, PandasExcelRepository]) -> None:
    """
    Replace the singleton repository instance.

    Call this before simulation initialization to switch between
    filesystem-based and npz-based data loading.
    """
    global _REPOSITORY
    _REPOSITORY = repo


def load_indices(filename, n_samples, n_nodes, data_size, lock=None):
    """
    Convenience wrapper to load indices from the singleton repository.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    return _REPOSITORY.get_indices(filename, n_samples, n_nodes, data_size, lock)


def load_depot(data_dir, area="Rio Maior"):
    """
    Convenience wrapper to load depot coords from the singleton repository.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    return _REPOSITORY.get_depot(area, data_dir=data_dir)


def load_simulator_data(data_dir, number_of_bins, area="Rio Maior", waste_type=None, lock=None):
    """
    Convenience wrapper to load simulator data from the singleton repository.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    return _REPOSITORY.get_simulator_data(number_of_bins, area, waste_type, lock, data_dir=data_dir)


def load_area_and_waste_type_params(area, waste_type):
    """
    Convenience wrapper to load area params.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    return _REPOSITORY.get_area_params(area, waste_type)


__all__ = [
    "SimulationRepository",
    "FileSystemRepository",
    "NumpyDictRepository",
    "PandasExcelRepository",
    "set_repository",
    "load_indices",
    "load_depot",
    "load_simulator_data",
    "load_area_and_waste_type_params",
]
