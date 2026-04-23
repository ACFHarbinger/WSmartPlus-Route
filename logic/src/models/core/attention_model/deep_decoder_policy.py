"""Deep Decoder Policy for constructive routing.

This module provides a policy that utilizes a multi-layer (deep) attention
decoder for constructing routing solutions. This architecture allows for
more complex decision-making during decoding compared to single-layer variants.

Attributes:
    DeepDecoderPolicy: Constructive policy with a multi-layer graph decoder.

Example:
    >>> from logic.src.models.core.attention_model.deep_decoder_policy import DeepDecoderPolicy
    >>> policy = DeepDecoderPolicy(env_name="vrpp", n_decode_layers=3)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.subnets.decoders.gat import DeepGATDecoder
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder


class DeepDecoderPolicy(AutoregressivePolicy):
    """Routing policy with multi-layer attention construction.

    Uses a standard GAT encoder but employs a DeepGATDecoder, which applies
    multiple layers of cross-attention between the current state and node
    embeddings at each decoding step.

    Attributes:
        encoder (GraphAttentionEncoder): Graph feature extractor.
        decoder (DeepGATDecoder): Multi-layer sequential constructor.
        init_embedding (nn.Module): Problem-specific latent projection.
    """

    encoder: GraphAttentionEncoder
    decoder: DeepGATDecoder

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_decode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes the DeepDecoderPolicy.

        Args:
            env_name: Name of the target environment.
            embed_dim: Latent vector dimensionality.
            hidden_dim: Hidden size for FFN and attention sublayers.
            n_encode_layers: Number of transformer encoder blocks.
            n_decode_layers: Number of transformer decoder blocks.
            n_heads: Parallel attention head count.
            normalization: Type of feature normalization to apply.
            dropout_rate: Probability for internal dropout layers.
            **kwargs: Additional parameters for components.
        """
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=hidden_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.decoder = DeepGATDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decode_layers,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the deep constructive decoding pass.

        Args:
            td: Environment state container.
            env: RL environment object.
            strategy: Action selection tactic ('greedy', 'sampling').
            num_starts: Construction start count.
            actions: Tour sequence for teacher forcing/evaluation.
            **kwargs: Additional control arguments.

        Returns:
            Dict[str, Any]: Policy outputs including rewards, log-probs, and actions.

        Raises:
            AssertionError: If sub-components or env_name are uninitialized.
        """
        # 1. Initialize latent node representations
        init_embeds = self.init_embedding(td)

        # 2. Extract global graph features
        assert self.encoder is not None, "Encoder is not initialized"
        embeddings = self.encoder(init_embeds)

        # 3. Cache constant encoding for step-wise construction
        assert self.decoder is not None, "Decoder is not initialized"
        fixed = self.decoder._precompute(embeddings)

        # 4. Sequential construction loop
        log_likelihood: float | torch.Tensor = 0.0
        entropy: float | torch.Tensor = 0.0
        output_actions = []
        step_idx = 0

        while not td["done"].all():
            assert self.env_name is not None, "env_name must be set"
            from logic.src.utils.data import TensorDictStateWrapper

            state_wrapper = TensorDictStateWrapper(td, self.env_name)

            # Masked attention over nodes
            logits, mask = self.decoder._get_log_p(fixed, state_wrapper)

            # Flatten head dimension if present
            if logits.dim() == 3:
                logits = logits[:, 0, :] if logits.size(1) > 1 else logits.squeeze(1)

            # Invert validity mask (DeepDecoder returns invalid=True)
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            valid_mask = ~mask

            if actions is not None:
                # Teachers forcing for evaluation/distillation
                action = actions[:, step_idx]
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            else:
                # Stochastic or deterministic selection
                action, log_p, entropy_step = self._select_action(logits, valid_mask, strategy)
                if isinstance(entropy_step, torch.Tensor):
                    entropy = entropy + entropy_step

            # Transition to next step
            td["action"] = action
            td = env.step(td)["next"].clone()

            log_likelihood = log_likelihood + log_p
            output_actions.append(action)
            step_idx += 1

        # Calculate final solution reward
        constructed_actions = torch.stack(output_actions, dim=1)
        reward = env.get_reward(td, constructed_actions)

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": constructed_actions,
            "entropy": entropy,
            "td": td,
        }
