"""
Abstract base class for simulation datasets.

All simulation dataset formats (npz, pkl, xlsx) implement this interface
so that consumers (e.g. Bins) can depend on a single type.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class SimulationDataset(ABC):
    """
    Base class for simulation datasets.

    Every implementation must support indexed access that returns a sample
    dict with at least the keys: 'depot', 'locs', 'waste', 'noisy_waste',
    'max_waste'.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]: ...
