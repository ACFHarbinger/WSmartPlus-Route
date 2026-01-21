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
        self.data_len = td.batch_size[0]
        self.data = td

    def __len__(self):
        return self.data_len

    def __getitems__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch: Union[dict, TensorDict]):
        return batch


class TensorDictDatasetFastGeneration(Dataset):
    """
    Dataset optimized for fast instantiation (avoiding list comp).
    """

    def __init__(self, td: TensorDict):
        self.data = td

    def __len__(self):
        return len(self.data)

    def __getitems__(self, index):
        return TensorDict(
            {key: item[index] for key, item in self.data.items()},
            batch_size=torch.Size([len(index)]),
            **td_kwargs,
        )

    @staticmethod
    def collate_fn(batch: Union[dict, TensorDict]):
        return batch
