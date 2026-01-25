"""
Dataset classes for WSmart-Route.
"""

from typing import Sized, Union, cast

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
        """
        Initialize the TensorDictDataset.

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
        return self.data.batch_size[0]

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx: Index of the sample.

        Returns:
            TensorDict: The sample at the specified index.
        """
        return self.data[idx]

    @staticmethod
    def load(path: str):
        """
        Load a TensorDictDataset from disk.

        Args:
            path: File path to load from.

        Returns:
            TensorDictDataset: The loaded dataset.
        """
        return TensorDictDataset(torch.load(path))

    def save(self, path: str):
        """
        Save the dataset to disk.

        Args:
            path: File path to save to.
        """
        torch.save(self.data, path)


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


class GeneratorDataset(Dataset):
    """
    Dataset that generates instances on-the-fly.
    """

    def __init__(self, generator, size: int):
        """
        Initialize the GeneratorDataset.

        Args:
            generator: A callable that generates TensorDicts when invoked.
            size: The virtual size of the dataset.
        """
        self.generator = generator
        self.size = size

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.size

    def __getitem__(self, idx: int) -> TensorDict:
        """
        Generate a sample on-the-fly.

        Args:
            idx: Index of the sample (not used, provided for API compatibility).

        Returns:
            TensorDict: A freshly generated sample.
        """
        return self.generator(batch_size=1)[0]


class ExtraKeyDataset(Dataset):
    """
    Dataset that includes an extra key/value pair (e.g., for baseline rewards).
    """

    def __init__(self, dataset: Dataset, extra: dict[str, torch.Tensor]):
        """
        Initialize the ExtraKeyDataset.

        Args:
            dataset: The underlying dataset.
            extra: Dictionary of extra tensors to include (e.g., {'baseline': ...}).
        """
        super().__init__()
        self.dataset = dataset
        self.extra = extra
        # Validate lengths
        for k, v in extra.items():
            assert len(cast(Sized, dataset)) == len(
                v
            ), f"Length mismatch for key {k}: {len(cast(Sized, dataset))} vs {len(v)}"

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a sample with extra keys.

        Args:
            idx: Index of the sample.

        Returns:
            dict: Dictionary with 'data' and extra keys.
        """
        item = {"data": self.dataset[idx]}
        for k, v in self.extra.items():
            item[k] = v[idx]
        return item

    def __len__(self) -> int:
        """
        Return the number of samples.

        Returns:
            int: Number of samples.
        """
        return len(cast(Sized, self.dataset))


class BaselineDataset(Dataset):
    """
    Dataset wrapping baseline values for training.
    """

    def __init__(self, dataset: Dataset, baseline: torch.Tensor):
        """
        Initialize the BaselineDataset.

        Args:
            dataset: The underlying dataset.
            baseline: Tensor of precomputed baseline values.
        """
        super().__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(cast(Sized, self.dataset)) == len(self.baseline)

    def __getitem__(self, item: int) -> dict:
        """
        Retrieve a sample with its associated baseline value.

        Args:
            item: Index of the sample.

        Returns:
            dict: Dictionary with 'data' and 'baseline' keys.
        """
        return {"data": self.dataset[item], "baseline": self.baseline[item]}

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(cast(Sized, self.dataset))


def tensordict_collate_fn(
    batch: list[Union[dict, TensorDict]],
) -> Union[dict, TensorDict]:
    """Collate list of TensorDicts or dicts into batched TensorDict or dict."""
    if isinstance(batch[0], dict):
        # We recursively collate the values
        return {key: tensordict_collate_fn([d[key] for d in batch]) for key in batch[0].keys()}

    if isinstance(batch[0], TensorDict):
        # We stack and convert to dict for pin_memory compatibility.
        # TensorDict's __init__ is called in the pin_memory thread without batch_size,
        # which causes errors. Standard dicts of tensors are safe.
        res = torch.stack(batch)  # type: ignore[arg-type]
        if hasattr(res, "to_dict"):
            return res.to_dict()
        return res

    return torch.stack(batch)  # type: ignore[arg-type]
