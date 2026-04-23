"""SymNCO Policy with projection head.

This module implements the policy for SymNCO (Kim et al. 2022), which extends
the standard Attention Model with a projection head to support symbolic
invariance and augmentation-based training objectives.

Attributes:
    SymNCOPolicy: Specialized policy with embedding projection support.

Example:
    >>> from logic.src.models.core.attention_model.symnco_policy import SymNCOPolicy
    >>> policy = SymNCOPolicy(embed_dim=128, use_projection_head=True)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.core.attention_model.policy import AttentionModelPolicy


class SymNCOPolicy(AttentionModelPolicy):
    """SymNCO Policy variant for contrastive/invariant RL.

    Enhances the Attention Model with a projection MLP that maps initial
    embeddings into a space suitable for invariance losses, while maintaining
    standard autoregressive construction capabilities.

    Attributes:
        use_projection_head (bool): Flag to enable the invariance projection.
        projection_head (nn.Module): 3-layer MLP for embedding transformation.
    """

    def __init__(self, embed_dim: int = 128, use_projection_head: bool = True, **kwargs: Any) -> None:
        """Initializes the SymNCOPolicy.

        Args:
            embed_dim: Feature vector size.
            use_projection_head: Whether to instantiate the auxiliary projection head.
            **kwargs: Additional parameters for the underlying AttentionModelPolicy.
        """
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.use_projection_head = use_projection_head

        if self.use_projection_head:
            # RL4CO standard 3-layer MLP architecture
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(
        self,
        td: TensorDict,
        env: Any,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes decoding and computes projection embeddings.

        Args:
            td: Problem state container.
            env: Environment managing rewards and dynamics.
            strategy: constructive decoding strategy.
            num_starts: Multi-start construction count.
            actions: Expert/target paths for comparison.
            **kwargs: Configuration flags for base policy call.

        Returns:
            Dict[str, Any]: Policy outputs extended with `proj_embeddings`.
        """
        # Ensure initial embeddings are returned for the projection head
        kwargs["return_init_embeds"] = True
        out = super().forward(td, env, strategy=strategy, num_starts=num_starts, actions=actions, **kwargs)

        if self.use_projection_head:
            # Project the raw features [batch, nodes, dim]
            init_embeds = out["init_embeds"]
            out["proj_embeddings"] = self.projection_head(init_embeds)

        return out
