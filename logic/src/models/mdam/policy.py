"""
MDAM Policy.

Multi-Decoder Attention Model policy combining encoder and multi-path decoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.common.autoregressive_policy import AutoregressivePolicy
from logic.src.models.subnets.embeddings import get_init_embedding

if TYPE_CHECKING:
    from logic.src.models.subnets.decoders.mdam import MDAMDecoder
    from logic.src.models.subnets.encoders.mdam.encoder import MDAMGraphAttentionEncoder


class MDAMPolicy(AutoregressivePolicy):
    """
    Multi-Decoder Attention Model (MDAM) Policy.

    Combines a shared encoder with multiple decoder paths to generate
    diverse solutions. KL divergence between paths encourages exploration
    of different solution strategies.

    Reference:
        Xin et al. "Multi-Decoder Attention Model with Embedding Glimpse for
        Solving Vehicle Routing Problems" (AAAI 2021)
    """

    def __init__(
        self,
        encoder: Optional[MDAMGraphAttentionEncoder] = None,
        decoder: Optional[MDAMDecoder] = None,
        embed_dim: int = 128,
        env_name: str = "vrpp",
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        num_paths: int = 5,
        normalization: str = "batch",
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **decoder_kwargs,
    ) -> None:
        """
        Initialize MDAM policy.

        Args:
            encoder: Pre-built encoder or None to create default.
            decoder: Pre-built decoder or None to create default.
            embed_dim: Embedding dimension.
            env_name: Environment name for context embedding.
            num_encoder_layers: Number of encoder transformer layers.
            num_heads: Number of attention heads.
            num_paths: Number of parallel decoder paths.
            normalization: Normalization type ('batch', 'layer', 'instance').
            train_decode_type: Decoding type during training.
            val_decode_type: Decoding type during validation.
            test_decode_type: Decoding type during testing.
            **decoder_kwargs: Additional decoder arguments.
        """
        from logic.src.models.subnets.decoders.mdam import MDAMDecoder
        from logic.src.models.subnets.encoders.mdam.encoder import MDAMGraphAttentionEncoder

        # Create encoder if not provided
        if encoder is None:
            encoder = MDAMGraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
            )

        # Create decoder if not provided
        if decoder is None:
            decoder = MDAMDecoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_paths=num_paths,
                **decoder_kwargs,
            )

        # MDAMDecoder doesn't inherit from AutoregressiveDecoder, but has compatible interface
        super().__init__(env_name=env_name, encoder=encoder, decoder=decoder)  # type: ignore[arg-type]

        self.embed_dim = embed_dim
        self.env_name = env_name
        self.num_paths = num_paths

        # Initialize with problem-specific embeddings
        self.init_embedding = get_init_embedding(env_name, embed_dim)

        # Store decode types for phase switching
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "sampling",
        num_starts: int = 1,
        phase: str = "train",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Policy forward pass.

        Args:
            td: TensorDict containing problem state.
            env: Environment for state transitions.
            phase: Current phase ('train', 'val', 'test').
            return_actions: Whether to return actions.
            **decoder_kwargs: Additional decoder arguments.

        Returns:
            Dictionary with:
                - reward: (batch_size, num_paths)
                - log_likelihood: (batch_size, num_paths)
                - entropy: KL divergence loss for diversity
                - actions: Action sequence (if return_actions)
        """
        # Initial embeddings
        embedding = self.init_embedding(td)

        # Encode inputs (encoder is guaranteed to be MDAMGraphAttentionEncoder)
        encoded_inputs, graph_embed, attn, V, h_old = self.encoder(td, x=embedding)  # type: ignore[misc]

        # Determine decode type based on phase
        # decode_type = decoder_kwargs.pop("decode_type", None) # Removed as decode_type is now a direct arg
        # if decode_type is None: # Removed as decode_type is now a direct arg
        #     decode_type = getattr(self, f"{phase}_decode_type") # Removed as decode_type is now a direct arg

        # Decode via multi-path decoder (decoder is guaranteed to be MDAMDecoder)
        log_p, actions, reward, kl_divergence = self.decoder(  # type: ignore[misc]
            td,
            (encoded_inputs, graph_embed, attn, V, h_old),
            env,
            self.encoder,  # Pass encoder for change() method
            decode_type=decode_type,
            num_starts=num_starts,
            **kwargs,
        )

        return {
            "reward": reward,
            "log_likelihood": log_p,
            "kl_divergence": kl_divergence,  # Diversity loss
            "actions": actions,
        }
