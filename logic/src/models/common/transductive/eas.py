"""Efficient Active Search (EAS) implementation.

This module implements EAS (Hottung et al. 2022), a parameter-efficient
transductive inference method that optimizes only a small subset of
model parameters (e.g., initial embeddings) during test-time refinement.

Attributes:
    EAS: Transductive model that optimizes selective parameter subsets.

Example:
    >>> from logic.src.models.common.transductive.eas import EAS
    >>> search = EAS(policy, search_param_names=["init_embedding"])
    >>> results = search(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch import nn

from .base import TransductiveModel


class EAS(TransductiveModel):
    """Efficient Active Search (Hottung et al. 2022).

    EAS improves upon full Active Search by restricting optimization to specific
    components of the network (like the initial node embeddings), reducing
    computational overhead while maintaining search efficacy.

    Attributes:
        search_param_names (List[str]): Substrings used to identify parameters
            for optimization via name matching.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 20,
        search_param_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the EAS transductive model.

        Args:
            model: The base neural strategy to refine.
            optimizer_kwargs: Setup parameters for the refinement optimizer.
            n_search_steps: Iteration count for test-time adaptation.
            search_param_names: Identifier tokens for parameters to adapt.
                Defaults to ["init_embedding", "init_proj"].
            **kwargs: Configuration properties for the underlying TransductiveModel.
        """
        super().__init__(
            model=model,
            optimizer_kwargs=optimizer_kwargs,
            n_search_steps=n_search_steps,
            **kwargs,
        )
        self.search_param_names = search_param_names or ["init_embedding", "init_proj"]

    def _get_search_params(self) -> Any:
        """Identifies parameters by name for selective optimization.

        Filters the model's named parameters, including only those whose names
        match the specified `search_param_names`.

        Returns:
            Any: A list of tensors to be updated during the search process.
        """
        params = []
        for name, param in self.model.named_parameters():
            if any(p_name in name for p_name in self.search_param_names):
                params.append(param)

        # If no targeted params found, fallback to all (safeguard)
        if not params:
            return self.model.parameters()

        return params
