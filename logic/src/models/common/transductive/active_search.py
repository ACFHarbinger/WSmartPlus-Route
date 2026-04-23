"""Active Search implementation.

This module implements Active Search (Bello et al. 2016), a transductive
inference method that fine-tunes all model parameters on a per-instance
basis using the REINFORCE algorithm.

Attributes:
    ActiveSearch: Transductive model that optimizes all parameters.

Example:
    >>> from logic.src.models.common.transductive.active_search import ActiveSearch
    >>> search = ActiveSearch(policy, n_search_steps=50)
    >>> results = search(td, env)
"""

from __future__ import annotations

from typing import Any

from .base import TransductiveModel


class ActiveSearch(TransductiveModel):
    """Active Search (Bello et al. 2016).

    Optimizes the entirety of the wrapped model's parameters on individual
    test instances to find better solutions than the zero-shot policy.

    Attributes:
        model (nn.Module): The policy to be optimized.
    """

    def _get_search_params(self) -> Any:
        """Defines parameters for optimization.

        In Active Search, all parameters of the underlying model are updated.

        Returns:
            Any: An iterable of all model parameters.
        """
        return self.model.parameters()
