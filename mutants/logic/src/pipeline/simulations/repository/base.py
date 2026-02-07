"""
Abstract interface for simulation data access.
"""

from abc import ABC, abstractmethod


class SimulationRepository(ABC):
    """
    Abstract interface for simulation data access.

    Defines the contract for loading geographic, waste, and configuration
    data required to initialize simulations. Implementations can source
    data from files, databases, APIs, or other backends.
    """

    @abstractmethod
    def get_indices(self, filename, n_samples, n_nodes, data_size, lock=None):
        """
        Loads or generates a list of bin indices for simulation samples.
        """
        pass

    @abstractmethod
    def get_depot(self, area, data_dir=None):
        """
        Retrieves the depot coordinates for a given area.
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
        """
        pass

    @abstractmethod
    def get_area_params(self, area, waste_type):
        """
        Retrieves area and waste-type specific simulation parameters.
        """
        pass
