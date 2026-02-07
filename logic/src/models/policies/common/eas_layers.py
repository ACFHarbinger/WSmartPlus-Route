from typing import Any, Dict, Optional

import torch.nn as nn

from .eas import EAS


class EASLay(EAS):
    """
    EAS variant that optimizes specific layers (e.g., the last few layers or added layers).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 20,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            optimizer_kwargs=optimizer_kwargs,
            n_search_steps=n_search_steps,
            search_param_names=["init_proj", "layers.2"],  # Example: last layer of 3
            **kwargs,
        )
