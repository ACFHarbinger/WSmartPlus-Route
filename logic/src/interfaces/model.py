from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class IModel(Protocol):
    """
    Protocol for Neural Models.
    """

    def forward(self, td: TensorDict, **kwargs: Any) -> Any:
        ...

    def to(self, device: torch.device) -> "IModel":
        ...

    def eval(self) -> "IModel":
        ...

    def train(self) -> "IModel":
        ...
