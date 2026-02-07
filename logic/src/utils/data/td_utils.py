"""
Utilities for TensorDict datasets.
"""

from typing import Union, cast

import tensordict
import torch
from packaging import version
from tensordict.tensordict import TensorDict

# Version check for tensordict
if version.parse(str(tensordict.__version__)) <= version.parse("0.4.0"):
    td_kwargs = {"_run_checks": False}
else:
    td_kwargs = {}


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
        return cast(Union[dict, TensorDict], res)

    return cast(Union[dict, TensorDict], torch.stack(batch))  # type: ignore[arg-type]
