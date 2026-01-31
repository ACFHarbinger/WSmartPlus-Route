"""
This module contains the Context Embedder implementations for various VRP variants.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from logic.src.constants.models import (
    DEPOT_DIM,
    NODE_DIM,
    VRPP_STEP_CONTEXT_OFFSET,
    WC_STEP_CONTEXT_OFFSET,
)


class ContextEmbedder(nn.Module, ABC):
    """
    Abstract base class for problem-specific context embeddings.
    Responsible for initializing node embeddings and determining step context dimensions.
    """

    def __init__(self, embedding_dim, node_dim, temporal_horizon):
        """
        Initialize the ContextEmbedder.

        Args:
            embedding_dim (int): Dimension of the embedding.
            node_dim (int): Dimension of node features.
            temporal_horizon (int): Temporal horizon for features.
        """
        super(ContextEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon

        # Common layers or definitions can go here or in concrete classes
        # Depot embedding is usually just x,y (DEPOT_DIM)
        self.init_embed_depot = nn.Linear(DEPOT_DIM, embedding_dim)

        # Node embedding input dimension depends on features
        # We'll let subclasses define the exact input dimension or layer
        self.init_embed = None

    @abstractmethod
    def init_node_embeddings(self, input):
        """
        Initialize node embeddings from input data.

        Args:
            input (dict): Input data dictionary.

        Returns:
            torch.Tensor: Initial node embeddings [batch_size, num_nodes + 1, embedding_dim].
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def step_context_dim(self):
        """
        Get the dimension of the step context.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()


class WCContextEmbedder(ContextEmbedder):
    """
    Context Embedder for Waste Collection (WC) problems.
    """

    def __init__(self, embedding_dim, node_dim=NODE_DIM, temporal_horizon=0):
        """
        Initialize the WCContextEmbedder.

        Args:
            embedding_dim (int): Dimension of the embedding.
            node_dim (int, optional): Dimension of node features. Defaults to 3.
            temporal_horizon (int, optional): Temporal horizon for features. Defaults to 0.
        """
        super(WCContextEmbedder, self).__init__(embedding_dim, node_dim, temporal_horizon)

        # Input: loc(2) + waste(1) + temporal_horizon + capacity/etc?
        # Actually in WC we usually have loc(2) + current_fill(1) + history...
        input_dim = node_dim + temporal_horizon
        self.init_embed = nn.Linear(input_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # Depot is just loc(2)

    def init_node_embeddings(self, nodes, temporal_features=True):
        """
        Initialize node embeddings for WC problems.

        Args:
            nodes (dict): Dictionary of node features.
            temporal_features (bool, optional): Whether to include temporal features. Defaults to True.

        Returns:
            torch.Tensor: Combined embeddings for depot and nodes.
        """
        waste_key = (
            "waste"
            if "waste" in list(nodes.keys())
            else (
                "demand"
                if "demand" in list(nodes.keys())
                else ("noisy_waste" if "noisy_waste" in list(nodes.keys()) else "real_waste")
            )
        )
        if temporal_features:
            features = tuple([waste_key] + ["fill{}".format(day) for day in range(1, self.temporal_horizon + 1)])
        else:
            features = (waste_key,)

        # Concatenate features
        # nodes['loc']: [batch, graph_size, 2]
        # nodes[feat]: [batch, graph_size]

        locs_key = "locs" if "locs" in nodes.keys() else "loc"
        node_features = torch.cat((nodes[locs_key], *(nodes[feat][:, :, None] for feat in features)), -1)

        # Embed depot and nodes
        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self):
        """
        Get the dimension of the step context for WC.

        Returns:
            int: Step context dimension (embedding_dim + 2).
        """
        # WC uses embedding_dim + WC_STEP_CONTEXT_OFFSET
        return self.embedding_dim + WC_STEP_CONTEXT_OFFSET


class VRPPContextEmbedder(ContextEmbedder):
    """
    Context Embedder for VRP with Profits (VRPP) families (vrpp, cvrpp).
    """

    def __init__(self, embedding_dim, node_dim=NODE_DIM, temporal_horizon=0):
        """
        Initialize the VRPPContextEmbedder.

        Args:
            embedding_dim (int): Dimension of the embedding.
            node_dim (int, optional): Dimension of node features. Defaults to 3.
            temporal_horizon (int, optional): Temporal horizon for features. Defaults to 0.
        """
        super(VRPPContextEmbedder, self).__init__(embedding_dim, node_dim, temporal_horizon)

        # Input: loc(2) + waste(1) + temporal_horizon
        # Note: VRPP usually has prize/demand structure but code in AttentionModel treated it
        # similarly to WC regarding waste/fill for embedding?
        # Re-checking _init_embed logic: it applies same logic for is_wc and is_vrpp regarding
        # features list. "vrpp has waste, wc has waste" comment in AttentionModel.

        input_dim = node_dim + temporal_horizon
        self.init_embed = nn.Linear(input_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(
            2, embedding_dim
        )  # Added this line as it was missing from the original and needed for init_node_embeddings

    def init_node_embeddings(self, nodes, temporal_features=True):
        """
        Initialize node embeddings for VRPP problems.

        Args:
            nodes (dict): Dictionary of node features.
            temporal_features (bool, optional): Whether to include temporal features. Defaults to True.

        Returns:
            torch.Tensor: Combined embeddings for depot and nodes.
        """
        waste_key = (
            "waste"
            if "waste" in list(nodes.keys())
            else (
                "demand"
                if "demand" in list(nodes.keys())
                else ("noisy_waste" if "noisy_waste" in list(nodes.keys()) else "real_waste")
            )
        )
        # Logic identical to WC in original code, reused here
        if temporal_features:
            features = tuple([waste_key] + ["fill{}".format(day) for day in range(1, self.temporal_horizon + 1)])
        else:
            features = (waste_key,)

        locs_key = "locs" if "locs" in nodes.keys() else "loc"
        node_features = torch.cat((nodes[locs_key], *(nodes[feat][:, :, None] for feat in features)), -1)

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self):
        """
        Get the dimension of the step context for VRPP.

        Returns:
            int: Step context dimension (embedding_dim + 1).
        """
        # VRPP uses embedding_dim + VRPP_STEP_CONTEXT_OFFSET
        return self.embedding_dim + VRPP_STEP_CONTEXT_OFFSET
