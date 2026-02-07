"""
Unified context embedding modules for VRP variants.

This module combines initial node projection logic (ContextEmbedder)
and decoder step-context embedding logic (EnvContext).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from logic.src.constants.models import (
    DEPOT_DIM,
    NODE_DIM,
    VRPP_STEP_CONTEXT_OFFSET,
    WC_STEP_CONTEXT_OFFSET,
)

# =============================================================================
# INITIAL NODE EMBEDDINGS (Formerly context_embedder.py)
# =============================================================================


class ContextEmbedder(nn.Module, ABC):
    """
    Abstract base class for problem-specific context embeddings.
    Responsible for initializing node embeddings and determining step context dimensions.
    """

    def __init__(self, embed_dim: int, node_dim: int, temporal_horizon: int):
        """
        Initialize the ContextEmbedder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon

        # Depot embedding is usually just x,y (DEPOT_DIM)
        self.init_embed_depot = nn.Linear(DEPOT_DIM, embed_dim)
        self.init_embed = None

    @abstractmethod
    def init_node_embeddings(self, nodes: dict[str, Any]) -> torch.Tensor:
        """Initialize node embeddings from input data."""
        raise NotImplementedError()

    @abstractmethod
    def step_context_dim(self) -> int:
        """Get the dimension of the step context."""
        raise NotImplementedError()

    def forward(self, input: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.init_node_embeddings(input), None


class WCContextEmbedder(ContextEmbedder):
    """Context Embedder for Waste Collection (WC) problems."""

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        super().__init__(embed_dim, node_dim, temporal_horizon)
        input_dim = node_dim + temporal_horizon
        self.init_embed = nn.Linear(input_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        waste_key = "waste"
        keys = nodes.keys()
        if "waste" not in keys:
            for k in ["demand", "noisy_waste", "real_waste"]:
                if k in keys:
                    waste_key = k
                    break

        if temporal_features:
            features = [waste_key] + [f"fill{day}" for day in range(1, self.temporal_horizon + 1)]
        else:
            features = [waste_key]

        locs_key = "locs" if "locs" in nodes else "loc"
        node_features = torch.cat((nodes[locs_key], *(nodes[feat][:, :, None] for feat in features)), -1)

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self) -> int:
        return self.embed_dim + WC_STEP_CONTEXT_OFFSET


class VRPPContextEmbedder(ContextEmbedder):
    """Context Embedder for VRP with Profits (VRPP) families."""

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        super().__init__(embed_dim, node_dim, temporal_horizon)
        input_dim = node_dim + temporal_horizon
        self.init_embed = nn.Linear(input_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        primary_key = None
        keys = nodes.keys()
        for key in ["waste", "demand", "noisy_waste", "real_waste", "prize"]:
            if key in keys:
                primary_key = key
                break

        if primary_key is None or self.node_dim == 2:
            locs_key = "locs" if "locs" in keys else "loc"
            node_features = nodes[locs_key]
        else:
            primary_feature = nodes[primary_key][:, :, None]
            locs_key = "locs" if "locs" in keys else "loc"
            if temporal_features:
                feature_list = [primary_feature]
                for day in range(1, self.temporal_horizon + 1):
                    feat = f"fill{day}"
                    if feat in keys:
                        feature_list.append(nodes[feat][:, :, None])
                    else:
                        feature_list.append(torch.zeros_like(primary_feature))
                node_features = torch.cat((nodes[locs_key], *feature_list), -1)
            else:
                node_features = torch.cat((nodes[locs_key], primary_feature), -1)

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self) -> int:
        if self.node_dim == 2:
            return self.embed_dim
        return self.embed_dim + VRPP_STEP_CONTEXT_OFFSET


# =============================================================================
# DECODER STEP CONTEXT EMBEDDINGS (Formerly context_embedding.py)
# =============================================================================


class EnvContext(nn.Module):
    """Base class for environment context embeddings."""

    def __init__(self, embed_dim: int, step_context_dim: int = 0, node_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.step_context_dim = step_context_dim
        self.node_dim = node_dim

        if step_context_dim > 0:
            self.project_context = nn.Linear(step_context_dim, embed_dim, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        td: Any,
    ) -> torch.Tensor:
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)

        if state_embedding is not None:
            return cur_node_embedding + state_embedding
        return cur_node_embedding

    def _cur_node_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        cur_node = td["current_node"]
        batch_size = embeddings.size(0)
        if cur_node.dim() == 1:
            cur_node = cur_node.unsqueeze(-1)
        return torch.gather(embeddings, 1, cur_node.unsqueeze(-1).expand(batch_size, 1, self.embed_dim))

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor | None:
        return None


class VRPPContext(EnvContext):
    """Context embedding for VRPP."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim, step_context_dim=1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        keys = td.keys()
        if "remaining_capacity" in keys:
            feat = td["remaining_capacity"]
        elif "current_load" in keys:
            feat = td["current_load"]
        else:
            return torch.zeros(embeddings.size(0), 1, self.embed_dim, device=embeddings.device)

        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)

        out = self.project_context(feat)
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out


class CVRPContext(VRPPContext):
    """Context embedding for CVRP."""

    pass


class WCVRPContext(VRPPContext):
    """Context embedding for WCVRP."""

    pass


class SWCVRPContext(WCVRPContext):
    """Context embedding for SWCVRP (Stochastic WCVRP)."""

    pass


class GenericContextEmbedder(ContextEmbedder):
    """
    Generic context embedder for problems without specific implementations.
    Embeds depot (2D) and nodes (node_dim) simply.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        super().__init__(embed_dim, node_dim, temporal_horizon)
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any]) -> torch.Tensor:
        # Fallback to 'loc' or 'locs'
        locs_key = "locs" if "locs" in nodes.keys() else "loc"
        node_features = nodes[locs_key]

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self) -> int:
        return self.embed_dim
