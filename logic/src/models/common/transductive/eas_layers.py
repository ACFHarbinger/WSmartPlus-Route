"""EAS Layer-specific variant module.

This module provides the `EASLay` class, which is a version of EAS that
refines the model by optimizing only specific structural layers (e.g., terminal
projections or late-stage graph layers).

Attributes:
    EASLay: EAS variant targeting specific architectural layers.

Example:
    >>> from logic.src.models.common.transductive.eas_layers import EASLay
    >>> search = EASLay(model, n_search_steps=25)
    >>> results = search(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from .eas import EAS


class EASLay(EAS):
    """EAS variant targeting specific architectural layers.

    Refines the policy by adapting parameters within designated layers
    (typically the projection head or final attention blocks) identified
    by substring matching.

    Attributes:
        model (nn.Module): The logic policy instance being adapted.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initializes the Layer-targeted EAS model.

        Args:
            model: The base neural strategy to refine.
            optimizer_kwargs: Configuration for the Adams optimizer.
            n_search_steps: Total refinement iterations at test-time.
            **kwargs: Additional parameters for the EAS base class.
        """
        super().__init__(
            model=model,
            optimizer_kwargs=optimizer_kwargs,
            n_search_steps=n_search_steps,
            search_param_names=["init_proj", "layers.2"],  # Example: last project/layer
            **kwargs,
        )
