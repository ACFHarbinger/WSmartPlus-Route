import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ContextEmbedder(nn.Module, ABC):
    """
    Abstract base class for problem-specific context embeddings.
    Responsible for initializing node embeddings and determining step context dimensions.
    """
    def __init__(self, embedding_dim, node_dim, temporal_horizon):
        super(ContextEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon

        # Common layers or definitions can go here or in concrete classes
        # Depot embedding is usually just x,y (2 dims)
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        
        # Node embedding input dimension depends on features
        # We'll let subclasses define the exact input dimension or layer
        self.init_embed = None 

    @abstractmethod
    def init_node_embeddings(self, nodes, temporal_features=True):
        """
        Compute initial embeddings for all nodes.
        
        Args:
            nodes: Dictionary of node features (loc, diff, etc.)
            temporal_features: Boolean to include temporal features
            
        Returns:
            Embeddings tensor [batch_size, graph_size+1, embedding_dim]
        """
        pass

    @property
    @abstractmethod
    def step_context_dim(self):
        """
        Return the dimension of the step context vector.
        """
        pass


class WCContextEmbedder(ContextEmbedder):
    """
    Context Embedder for Waste Collection (WC) families (wcvrp, cwcvrp, sdwcvrp).
    """
    def __init__(self, embedding_dim, node_dim=3, temporal_horizon=0):
        super(WCContextEmbedder, self).__init__(embedding_dim, node_dim, temporal_horizon)
        
        # Input: loc(2) + waste(1) + temporal_horizon
        input_dim = node_dim + temporal_horizon
        self.init_embed = nn.Linear(input_dim, embedding_dim)

    def init_node_embeddings(self, nodes, temporal_features=True):
        if temporal_features:
            features = tuple(['waste'] + ["fill{}".format(day) for day in range(1, self.temporal_horizon + 1)])
        else:
            features = ('waste',)
            
        # Concatenate features
        # nodes['loc']: [batch, graph_size, 2]
        # nodes[feat]: [batch, graph_size]
        
        node_features = torch.cat((
            nodes['loc'],
            *(nodes[feat][:, :, None] for feat in features)
        ), -1)
        
        # Embed depot and nodes
        return torch.cat(
            (
                self.init_embed_depot(nodes['depot'])[:, None, :],
                self.init_embed(node_features)
            ),
            1
        )

    @property
    def step_context_dim(self):
        # WC uses embedding_dim + 2 (usually related to capacity/remaining length/etc)
        return self.embedding_dim + 2


class VRPPContextEmbedder(ContextEmbedder):
    """
    Context Embedder for VRP with Profits (VRPP) families (vrpp, cvrpp).
    """
    def __init__(self, embedding_dim, node_dim=3, temporal_horizon=0):
        super(VRPPContextEmbedder, self).__init__(embedding_dim, node_dim, temporal_horizon)
        
        # Input: loc(2) + waste(1) + temporal_horizon
        # Note: VRPP usually has prize/demand structure but code in AttentionModel treated it similarly to WC regarding waste/fill for embedding?
        # Re-checking _init_embed logic: it applies same logic for is_wc and is_vrpp regarding features list.
        # "vrpp has waste, wc has waste" comment in AttentionModel.
        
        input_dim = node_dim + temporal_horizon
        self.init_embed = nn.Linear(input_dim, embedding_dim)

    def init_node_embeddings(self, nodes, temporal_features=True):
        # Logic identical to WC in original code, reused here
        if temporal_features:
            features = tuple(['waste'] + ["fill{}".format(day) for day in range(1, self.temporal_horizon + 1)])
        else:
            features = ('waste',)
            
        node_features = torch.cat((
            nodes['loc'],
            *(nodes[feat][:, :, None] for feat in features)
        ), -1)
        
        return torch.cat(
            (
                self.init_embed_depot(nodes['depot'])[:, None, :],
                self.init_embed(node_features)
            ),
            1
        )

    @property
    def step_context_dim(self):
        # VRPP uses embedding_dim + 1
        return self.embedding_dim + 1
