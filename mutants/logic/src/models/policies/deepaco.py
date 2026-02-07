"""
DeepACO Policy: Combines encoder and decoder for end-to-end inference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.common.nonautoregressive import NonAutoregressivePolicy
from logic.src.models.subnets.deepaco_decoder import ACODecoder
from logic.src.models.subnets.deepaco_encoder import DeepACOEncoder
from tensordict import TensorDict


class DeepACOPolicy(NonAutoregressivePolicy):
    """
    DeepACO Policy.

    Combines DeepACOEncoder (heatmap prediction) with ACODecoder
    (ant colony solution construction) for end-to-end inference.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        env_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize DeepACOPolicy.

        Args:
            embed_dim: Embedding dimension for encoder.
            num_encoder_layers: Number of GNN layers.
            num_heads: Number of attention heads.
            n_ants: Number of ants for ACO.
            n_iterations: Number of ACO iterations.
            alpha: Pheromone weight.
            beta: Heuristic weight.
            rho: Evaporation rate.
            use_local_search: Whether to apply 2-opt.
            env_name: Environment name.
        """
        encoder = DeepACOEncoder(
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            **kwargs,
        )
        decoder = ACODecoder(
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_local_search=use_local_search,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass: encode heatmap + ACO decode.

        Args:
            td: TensorDict with problem instance.
            env: Environment.
            num_starts: Number of ACO runs (default 1).

        Returns:
            Dictionary with reward, actions, log_likelihood, heatmap.
        """
        return super().forward(td, env, num_starts=num_starts, **kwargs)
