"""
RL-specific utility functions.
"""

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.interfaces import ITensorDictLike


def ensure_tensordict(batch: Union[Dict[str, Any], TensorDict], device: Optional[torch.device] = None) -> TensorDict:
    """
    Ensure the input batch is a TensorDict and resides on the correct device.
    Handles wrapped batches from BaselineDataset (which have a 'data' key).

    Args:
        batch: Input batch as a dict or TensorDict.
        device: Target device.

    Returns:
        TensorDict: The processed batch.
    """
    # 1. Already a TensorDict? Just handle device.
    if isinstance(batch, TensorDict):
        td = batch

    # 2. Handle standard dicts
    elif isinstance(batch, dict):
        if "data" in batch.keys() and isinstance(batch["data"], (dict, TensorDict)):
            # Handling BaselineDataset wrapped output: {"data": {...}}
            data = batch["data"]
            if isinstance(data, TensorDict):
                td = data
            else:
                # Dict with tensors, infer batch size from first value
                if data:
                    first_val = next(iter(data.values()))
                    if hasattr(first_val, "ndim"):
                        bs = [first_val.shape[0]] if first_val.ndim > 0 else []
                    else:
                        bs = [len(first_val)] if hasattr(first_val, "__len__") else []
                    td = TensorDict(data, batch_size=bs)
                else:
                    td = TensorDict({}, batch_size=[0])
        elif any(isinstance(v, (torch.Tensor, TensorDict)) for v in batch.values()):
            # Direct dict of tensors
            first_val = next(iter(batch.values()))
            if hasattr(first_val, "ndim"):
                bs = [first_val.shape[0]] if first_val.ndim > 0 else []
            else:
                bs = [len(first_val)] if hasattr(first_val, "__len__") else []
            td = TensorDict(batch, batch_size=bs)
        else:
            # Non-tensor dict, wrap it
            bs = [len(batch)] if hasattr(batch, "__len__") else []
            td = TensorDict({"data": batch}, batch_size=bs)

    # 3. Handle ITensorDictLike protocol (e.g. state wrappers)
    elif isinstance(batch, ITensorDictLike):
        # Prefer to_tensordict or internal td if available
        if hasattr(batch, "to_tensordict"):
            td = batch.to_tensordict()
        elif hasattr(batch, "td") and isinstance(batch.td, TensorDict):
            td = batch.td
        else:
            # Generic construction from items() and batch_size
            td = TensorDict(dict(batch.items()), batch_size=batch.batch_size)

    # 4. Fallback for other types
    else:
        bs = [len(batch)] if hasattr(batch, "__len__") else []
        td = TensorDict({"data": batch}, batch_size=bs)

    if device is not None:
        td = td.to(device)

    return td
