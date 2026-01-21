"""
Attention Model Policy for RL4CO.

Adapts the existing GATEncoder and AttentionDecoder to the RL4CO architecture
using TensorDict for state management.
"""
from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from logic.src.data.transforms import batchify
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.embeddings import get_init_embedding
from logic.src.models.policies.base import ConstructivePolicy
from logic.src.models.policies.utils import DummyProblem, TensorDictStateWrapper
from logic.src.models.subnets.attention_decoder import AttentionDecoder
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder


class AttentionModelPolicy(ConstructivePolicy):
    """
    RL4CO-style Policy using existing Attention Model components.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        **kwargs,
    ):
        """Initialize AttentionModelPolicy."""
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=hidden_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
            **kwargs,
        )

        self.decoder = AttentionDecoder(
            embedding_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=DummyProblem(env_name),
            n_heads=n_heads,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass executing the constructive solution generation.
        """
        # 1. Initialize embeddings
        init_embeds = self.init_embedding(td)

        # 2. Encoder
        edges = td.get("edges", None)
        assert self.encoder is not None, "Encoder is not initialized"
        embeddings = self.encoder(init_embeds, edges)

        # 3. Multi-start expansion if needed
        if num_starts > 1:
            td = batchify(td, num_starts)
            embeddings = embeddings.unsqueeze(1).repeat(1, num_starts, 1, 1).reshape(-1, *embeddings.shape[1:])

        # 4. Decoder Precomputation
        assert self.decoder is not None, "Decoder is not initialized"
        fixed = self.decoder._precompute(embeddings)

        # 5. Decoding Loop
        log_likelihood = 0
        output_actions = []
        step_idx = 0

        # Assuming environment is already reset
        while not td["done"].all():
            # Wrap state for legacy compatibility
            assert self.env_name is not None, "env_name must be set"
            state_wrapper = TensorDictStateWrapper(td, self.env_name)

            # Get logits from decoder
            logits, mask = self.decoder._get_log_p(fixed, state_wrapper)
            # mask returned by _get_log_p is the INVALID mask (True=masked)

            # AttentionDecoder returns (batch, n_heads, n_nodes). We take head 0.
            logits = logits[:, 0, :]

            # Invert mask for _select_action (expects True=VALID)
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            valid_mask = ~mask

            if actions is not None:
                # Teacher forcing
                action = actions[:, step_idx]
                # Compute log_prob of this action
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            else:
                # Select action
                action, log_p = self._select_action(logits, valid_mask, decode_type)

            # Update state
            td["action"] = action
            td = env.step(td)

            # update log likelihood
            log_likelihood = log_likelihood + log_p
            output_actions.append(action)
            step_idx += 1

        # Collect reward
        reward = env.get_reward(td, torch.stack(output_actions, dim=1))

        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": torch.stack(output_actions, dim=1),
        }

        if kwargs.get("return_init_embeds", False):
            out["init_embeds"] = init_embeds

        return out
