"""EAS Embedding-only variant module.

This module provides the `EASEmb` class, which is a specialized version of EAS
that restricts test-time adaptation exclusively to the initial graph embeddings.

Attributes:
    EASEmb: EAS variant focusing on latent representation optimization.

Example:
    >>> from logic.src.models.common.transductive.eas_embeddings import EASEmb
    >>> search = EASEmb(model, n_search_steps=30)
    >>> results = search(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from .eas import EAS


class EASEmb(EAS):
    """EAS variant focusing on latent embedding optimization.

    Restricts transductive refinement to parameters containing "init_embedding",
    making it a lightweight alternative to full Active Search.

    Attributes:
        model (nn.Module): The base policy instance.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initializes the Embedding-focused EAS model.

        Args:
            model: The base policy model to be adapted.
            optimizer_kwargs: Optimizer settings (e.g., learning rate).
            n_search_steps: Number of test-time optimization iterations.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            model=model,
            optimizer_kwargs=optimizer_kwargs,
            n_search_steps=n_search_steps,
            search_param_names=["init_embedding"],
            **kwargs,
        )
