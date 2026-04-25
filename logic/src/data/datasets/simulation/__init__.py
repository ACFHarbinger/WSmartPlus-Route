"""
Simulation dataset classes for WSmart-Route.

Attributes:
    GenerativeDataset: Dataset that generates instances on-the-fly.

Example:
    >>> from logic.src.data.datasets import GenerativeDataset
    >>> dataset = GenerativeDataset(generator, size)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

from .gen_dataset import GenerativeDataset

__all__ = ["GenerativeDataset"]
