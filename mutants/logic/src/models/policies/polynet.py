"""
PolyNet Policy.

Multi-strategy policy combining encoder with PolyNet decoder.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.embeddings import get_init_embedding
from logic.src.models.policies.common.autoregressive import AutoregressivePolicy
from logic.src.models.subnets.decoders.polynet_decoder import PolyNetDecoder
from logic.src.models.subnets.encoders.gat_encoder import GraphAttentionEncoder
from tensordict import TensorDict


class PolyNetPolicy(AutoregressivePolicy):
    """
    PolyNet Policy for learning K diverse solution strategies.

    Combines AttentionModel or MatNet encoder with PolyNet decoder.
    Uses static binary vectors to condition attention and learn diverse strategies.

    Reference:
        Hottung et al. "PolyNet: Learning Diverse Solution Strategies for
        Neural Combinatorial Optimization" (2024)
    """

    def __init__(
        self,
        k: int,
        encoder: Optional[nn.Module] = None,
        encoder_type: str = "AM",
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: str = "vrpp",
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
        **kwargs,
    ) -> None:
        """
        Initialize PolyNet policy.

        Args:
            k: Number of strategies to learn.
            encoder: Pre-built encoder or None to create default.
            encoder_type: Type of encoder ("AM" or "MatNet").
            embed_dim: Embedding dimension.
            num_encoder_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            normalization: Normalization type.
            feedforward_hidden: Feed-forward hidden dimension.
            env_name: Environment name.
            temperature: Temperature for softmax.
            tanh_clipping: Tanh clipping value.
            mask_logits: Whether to mask logits.
            train_decode_type: Decoding type during training.
            val_decode_type: Decoding type during validation.
            test_decode_type: Decoding type during testing.
        """
        # Create encoder if not provided
        if encoder is None:
            encoder = GraphAttentionEncoder(
                n_heads=num_heads,
                embed_dim=embed_dim,
                n_layers=num_encoder_layers,
                normalization=normalization,
                feed_forward_hidden=feedforward_hidden,
            )

        # Create decoder
        decoder = PolyNetDecoder(
            k=k,
            encoder_type=encoder_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            **kwargs,
        )

        super().__init__(
            env_name=env_name,
            encoder=encoder,  # type: ignore[arg-type]
            decoder=decoder,  # type: ignore[arg-type]
        )

        self.k = k
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits

        # Initialize with problem-specific embeddings
        self.init_embedding = get_init_embedding(env_name, embed_dim)

        # Store decode types
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = True,
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through policy.

        Args:
            td: TensorDict containing problem state.
            env: Environment for state transitions.
            phase: Current phase ('train', 'val', 'test').
            return_actions: Whether to return actions.
            num_starts: Number of starting points.

        Returns:
            Dictionary with reward, log_likelihood, and actions.
        """
        # Get initial embeddings
        embedding = self.init_embedding(td)

        # Encode
        embeddings = self.encoder(embedding)  # type: ignore[attr-defined]

        # Determine decode type
        decode_type = kwargs.pop("decode_type", None)
        if decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Decode
        log_likelihood, actions = self.decoder(  # type: ignore[attr-defined]
            td,
            embeddings,
            env,
            decode_type=decode_type,
            num_starts=num_starts,
            **kwargs,
        )

        # Calculate reward
        reward = env.get_reward(td, actions)  # type: ignore[attr-defined]

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": actions if return_actions else None,
        }
