"""
Repository Pattern for Simulation data access.
"""

from logic.src.constants import ROOT_DIR

from .base import SimulationRepository
from .filesystem import FileSystemRepository

# Singleton repository instance
_repository = FileSystemRepository(ROOT_DIR)


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
    return _repository.get_area_params(area, waste_type)


__all__ = [
    "SimulationRepository",
    "FileSystemRepository",
    "load_indices",
    "load_depot",
    "load_simulator_data",
    "load_area_and_waste_type_params",
]
