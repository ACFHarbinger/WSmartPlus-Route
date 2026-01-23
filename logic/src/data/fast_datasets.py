"""
Optimized dataset implementations for TensorDicts.
"""

from typing import Union

import tensordict
import torch
from packaging import version
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset

# Version check for tensordict
if version.parse(tensordict.__version__) <= version.parse("0.4.0"):
    td_kwargs = {"_run_checks": False}
else:
    td_kwargs = {}


class FastTdDataset(Dataset):
    """
    Optimized Dataset for TensorDicts.
    """

    def __init__(self, td: TensorDict):
        """
        Initialize the FastTdDataset.

        Args:
            td: A TensorDict containing the dataset tensors.
        """
        self.data_len = td.batch_size[0]
        self.data = td

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data_len

    def __getitems__(self, idx):
        """
        Retrieve samples by indices (batch getter).

        Args:
            idx: Indices of the samples.

        Returns:
            TensorDict: The samples at the specified indices.
        """
        return self.data[idx]

    @staticmethod
    def collate_fn(batch: Union[dict, TensorDict]):
        """
        Collate function that passes batch through unchanged.

        Args:
            batch: A batch of data.

        Returns:
            The batch unchanged.
        """
        return batch


class TensorDictDatasetFastGeneration(Dataset):
    """
    Dataset optimized for fast instantiation (avoiding list comp).
    """

    def __init__(self, td: TensorDict):
        """
        Initialize the TensorDictDatasetFastGeneration.

        Args:
            td: A TensorDict containing the dataset tensors.
        """
        self.data = td

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitems__(self, index):
        """
        Retrieve samples by indices, returning a new TensorDict.

        Args:
            index: Indices of the samples.

        Returns:
            TensorDict: A new TensorDict with the selected samples.
        """
        return TensorDict(
            {key: item[index] for key, item in self.data.items()},
            batch_size=torch.Size([len(index)]),
            **td_kwargs,
        )

    @staticmethod
    def collate_fn(batch: Union[dict, TensorDict]):
        """
        Collate function that passes batch through unchanged.

        Args:
            batch: A batch of data.

        Returns:
            The batch unchanged.
        """
        return batch
