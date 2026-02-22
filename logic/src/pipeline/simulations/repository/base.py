"""
Abstract interface for simulation data access.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import pandas as pd


class SimulationRepository(ABC):
    """
    Abstract interface for simulation data access.

    Defines the contract for loading geographic, waste, and configuration
    data required to initialize simulations. Implementations can source
    data from files, databases, APIs, or other backends.
    """

    @abstractmethod
    def get_indices(
        self, filename: Any, n_samples: int, n_nodes: int, data_size: int, lock: Optional[Any] = None
    ) -> List[List[int]]:
        """
        Loads or generates a list of bin indices for simulation samples.
        """
        pass

    @abstractmethod
    def get_depot(self, area: Any, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves the depot coordinates for a given area.
        """
        pass

    @abstractmethod
    def get_simulator_data(
        self,
        number_of_bins: int,
        area: str = "Rio Maior",
        waste_type: Optional[str] = None,
        lock: Optional[Any] = None,
        data_dir: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads waste statistics and coordinate data for the simulator.
        """
        pass

    @abstractmethod
    def get_area_params(self, area: Any, waste_type: Any) -> Tuple[float, float, float, float, float]:
        """
        Retrieves area and waste-type specific simulation parameters.
        """
        pass
