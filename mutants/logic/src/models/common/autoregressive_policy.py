from typing import Any, Dict, Optional

from logic.src.envs.base import RL4COEnvBase
from tensordict import TensorDict

from .autoregressive_decoder import AutoregressiveDecoder
from .autoregressive_encoder import AutoregressiveEncoder
from .constructive import ConstructivePolicy


class AutoregressivePolicy(ConstructivePolicy):
    """
    Base class for autoregressive policies.

    Combines an AR encoder with an AR decoder to form a complete policy.
    Inherits from ConstructivePolicy to leverage standardized decoding strategies.
    """

    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs,
    ):
        """Initialize AutoregressivePolicy."""
        # Use placeholders for base initialization if not provided
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
        decode_type: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full forward pass: encode + decode.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions.
            decode_type: Decoding strategy.
            num_starts: Number of solution starts.

        Returns:
            Dictionary containing reward, log_likelihood, and actions.
        """
        # Encode
        embeddings = self.encoder(td, **kwargs) if self.encoder is not None else None

        # Decode
        if self.decoder is not None:
            # Note: Many decoders in WSmart-Route implement their own loop.
            # We assume the decoder handles the AR process.
            log_p, actions = self.decoder(td, embeddings, env, decode_type=decode_type, num_starts=num_starts, **kwargs)
        else:
            raise ValueError("AutoregressivePolicy requires a decoder.")

        # Calculate reward
        reward = env.get_reward(td, actions)

        return {
            "reward": reward,
            "log_likelihood": log_p,
            "actions": actions,
        }
