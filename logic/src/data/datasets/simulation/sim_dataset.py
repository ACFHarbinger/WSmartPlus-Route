"""
Abstract base class for simulation datasets.

All simulation dataset formats (npz, pkl, xlsx) implement this interface
so that consumers (e.g. Bins) can depend on a single type.

Attributes:
    SimulationDataset: Base class for simulation datasets.

Example:
    >>> from logic.src.data.datasets import SimulationDataset
    >>> dataset = SimulationDataset()
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
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

    Attributes:
        None
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return the sample at the given index.

        Args:
            index: Description of index.
        """
        ...
