"""
Critic Network for RL4CO.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.models.embeddings import get_init_embedding
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder


class CriticNetwork(nn.Module):
    """
    Critic Network for estimating Value(State).
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        dropout_rate: float = 0.0,
        aggregation: str = "avg",
        encoder: nn.Module = None,
        **kwargs,
    ):
        """Initialize CriticNetwork."""
        super().__init__()
        self.aggregation = aggregation

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embed_dim,
                feed_forward_hidden=hidden_dim,
                n_layers=n_layers,
                normalization=normalization,
                dropout_rate=dropout_rate,
                **kwargs,
            )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Estimate value of state.
        Output: (batch_size, 1)
        """
        init_embeds = self.init_embedding(td)
        edges = td.get("edges", None)

        embeddings = self.encoder(init_embeds, edges)

        # Aggregation
        if self.aggregation == "avg":
            graph_embed = embeddings.mean(1)
        elif self.aggregation == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation == "max":
            graph_embed = embeddings.max(1)[0]
        else:
            # Default to avg
            graph_embed = embeddings.mean(1)

        value = self.value_head(graph_embed)
        return value


def create_critic_from_actor(policy: nn.Module, backbone_name: str = "encoder", **critic_kwargs):
    """
    Create a critic network re-using the encoder from the policy.
    """
    encoder = getattr(policy, backbone_name, None)
    if encoder is None:
        raise ValueError(f"Critic requires a backbone in the policy network: {backbone_name}")

    # Deepcopy the encoder to ensure independent weights initially
    critic = CriticNetwork(
        env_name=policy.env_name,  # Assuming policy has env_name
        embed_dim=policy.embed_dim,
        encoder=copy.deepcopy(encoder),
        **critic_kwargs,
    ).to(next(policy.parameters()).device)

    return critic
