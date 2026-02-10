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
    if isinstance(batch, ITensorDictLike):
        if "data" in batch:
            # Handling BaselineDataset wrapped output
            td_data = batch["data"]
            if not isinstance(td_data, TensorDict):
                # Infer batch size from first value
                first_val = next(iter(td_data.values()))
                if hasattr(first_val, "ndim") and first_val.ndim == 0:
                    batch_size = []
                else:
                    batch_size = [len(first_val)] if hasattr(first_val, "__len__") else []
                td_data = TensorDict(td_data, batch_size=batch_size)
            td = td_data
        # Direct dict or TensorDict
        elif not isinstance(batch, TensorDict):
            first_val = next(iter(batch.values()))
            if hasattr(first_val, "ndim") and first_val.ndim == 0:
                batch_size = []
            else:
                batch_size = [len(first_val)] if hasattr(first_val, "__len__") else []
            td = TensorDict(batch, batch_size=batch_size)
        else:
            td = batch
    else:
        # Fallback for unexpected types (try to wrap)
        td = TensorDict({"data": batch}, batch_size=[len(batch)] if hasattr(batch, "__len__") else [])

    if device is not None:
        td = td.to(device)

    return td
