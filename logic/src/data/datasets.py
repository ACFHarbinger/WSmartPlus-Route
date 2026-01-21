"""
Dataset classes for WSmart-Route.
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


class TensorDictDataset(Dataset):
    """
    Dataset compatible with TensorDicts.
    """

    def __init__(self, td: TensorDict):
        self.data = td

    def __len__(self):
        return self.data.batch_size[0]

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load(path: str):
        return TensorDictDataset(torch.load(path))

    def save(self, path: str):
        torch.save(self.data, path)


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


class GeneratorDataset(Dataset):
    """
    Dataset that generates instances on-the-fly.
    """

    def __init__(self, generator, size: int):
        self.generator = generator
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TensorDict:
        return self.generator(batch_size=1)[0]


class BaselineDataset(Dataset):
    """
    Dataset wrapping baseline values for training.
    """

    def __init__(self, dataset: Dataset, baseline: torch.Tensor):
        super().__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item: int) -> dict:
        return {"data": self.dataset[item], "baseline": self.baseline[item]}

    def __len__(self) -> int:
        return len(self.dataset)


def tensordict_collate_fn(batch: list[TensorDict]) -> TensorDict:
    """Collate list of TensorDicts into batched TensorDict."""
    return torch.stack(batch)  # type: ignore
