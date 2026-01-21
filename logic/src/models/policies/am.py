"""
Attention Model Policy for RL4CO.

Adapts the existing GATEncoder and AttentionDecoder to the RL4CO architecture
using TensorDict for state management.
"""
from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.embeddings import get_init_embedding
from logic.src.models.policies.base import ConstructivePolicy
from logic.src.models.subnets.attention_decoder import AttentionDecoder
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder


class TensorDictStateWrapper:
    """
    Wraps a TensorDict to expose methods expected by legacy AttentionDecoder.
    """

    def __init__(self, td: TensorDict, problem_name: str = "vrpp"):
        self.td = td
        self.problem_name = problem_name

        # Expose common properties directly
        self.dist_matrix = td.get("dist", None)
        # Handle 'demands_with_depot' for WCVRP partial updates
        if "demand" in td.keys():
            # In WCVRPEnv, demand includes all nodes.
            # Encoder/Decoder expects [batch, steps, 1] usually?
            # Actually legacy code expects [batch, n_nodes].
            self.demands_with_depot = td["demand"]

    def get_mask(self) -> Optional[torch.Tensor]:
        """Get action mask from TensorDict."""
        # RL4CO envs provide "action_mask" where True means VALID.
        # AttentionDecoder expects mask where True means INVALID (masked out).
        # So we invert it.
        if "action_mask" in self.td.keys():
            mask = ~self.td["action_mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            return mask
        return None

    def get_edges_mask(self) -> Optional[torch.Tensor]:
        return self.td.get("graph_mask", None)

    def get_current_node(self) -> torch.Tensor:
        """Get current node (last visited)."""
        return self.td["current_node"].long()

    def get_current_profit(self) -> torch.Tensor:
        """For VRPP: get cumulative collected prize."""
        # This is used for context embedding.
        val = self.td.get("collected_prize", torch.zeros(self.td.batch_size, device=self.td.device))
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        return val

    def get_current_efficiency(self) -> torch.Tensor:
        """For WCVRP: get current efficiency."""
        # Legacy placeholder
        val = torch.zeros(self.td.batch_size, device=self.td.device)
        return val.unsqueeze(-1)

    def get_remaining_overflows(self) -> torch.Tensor:
        """For WCVRP: get remaining overflows."""
        # Legacy placeholder
        val = torch.zeros(self.td.batch_size, device=self.td.device)
        return val.unsqueeze(-1)


class DummyProblem:
    """Minimal problem wrapper for AttentionDecoder init."""

    def __init__(self, name: str):
        self.NAME = name


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
        **kwargs,
    ) -> dict:
        """
        Forward pass executing the constructive solution generation.
        """
        # 1. Initialize embeddings
        init_embeds = self.init_embedding(td)

        # 2. Encoder
        # Pass graph structure if available (e.g., k-NN edges)
        edges = td.get("edges", None)
        embeddings = self.encoder(init_embeds, edges)

        # 3. Decoder Precomputation
        fixed = self.decoder._precompute(embeddings)

        # 4. Decoding Loop
        log_likelihood = 0
        actions = []

        # Assuming environment is already reset
        while not td["done"].all():
            # Wrap state for legacy compatibility
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

            # Select action
            action, log_p = self._select_action(logits, valid_mask, decode_type)

            # Update state
            td["action"] = action
            td = env.step(td)

            # update log likelihood
            log_likelihood = log_likelihood + log_p
            actions.append(action)

        # Collect reward
        reward = env.get_reward(td, torch.stack(actions, dim=1))

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": torch.stack(actions, dim=1),
        }
