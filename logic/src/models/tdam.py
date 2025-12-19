import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter


class DistanceAwareGraphConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        distance_influence: str = "inverse",
        aggregation: str = "sum",
        bias: bool = True
    ):
        super(DistanceAwareGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation
        self.distance_influence = distance_influence
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        
        # Distance weighting parameters
        if distance_influence == "learnable":
            self.distance_transform = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        elif distance_influence == "exponential":
            self.temp = nn.Parameter(torch.tensor(1.0))
        elif distance_influence == "inverse":
            self.alpha = nn.Parameter(torch.tensor(1.0))
            
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.init_parameters()
        
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            
    def get_distance_weights(self, dist_matrix):
        eps = 1e-8
        
        if self.distance_influence == "inverse":
            weights = 1.0 / (dist_matrix.pow(self.alpha) + eps)
        elif self.distance_influence == "exponential":
            weights = torch.exp(-dist_matrix / self.temp.clamp(min=eps))
        elif self.distance_influence == "learnable":
            flat_dist = dist_matrix.view(-1, 1)
            flat_weights = self.distance_transform(flat_dist)
            weights = flat_weights.view(dist_matrix.shape)
        else:
            weights = (dist_matrix > 0).float()
            
        return weights
    
    def forward(self, h, adj, dist_matrix=None):
        batch_size, num_nodes = h.size(0), h.size(1)
        support = self.linear(h)
        
        if dist_matrix is None:
            dist_matrix = adj
            
        edge_weights = self.get_distance_weights(dist_matrix)
        weighted_adj = adj * edge_weights
        
        if self.aggregation == "max":
            source_idx, target_idx = adj.squeeze(0).nonzero(as_tuple=True)
            edge_weights_flat = edge_weights.view(-1)[adj.view(-1).nonzero().squeeze()]
            
            num_edges = source_idx.size(0)
            batch_indices = torch.arange(batch_size, device=h.device)
            expanded_source = source_idx.repeat(batch_size)
            expanded_target = target_idx.repeat(batch_size)
            batch_id = batch_indices.repeat_interleave(num_edges)
            
            messages = support[batch_id, expanded_source]
            expanded_weights = edge_weights_flat.repeat(batch_size)
            weighted_messages = messages * expanded_weights.unsqueeze(1)
            
            batch_offset = batch_id * num_nodes
            unique_targets = batch_offset + expanded_target
            
            flat_output = scatter(weighted_messages, unique_targets, dim=0,
                                dim_size=batch_size * num_nodes, reduce="max")
            out = flat_output.view(batch_size, num_nodes, self.out_channels)
            
        elif self.aggregation == "mean":
            degrees = weighted_adj.squeeze(0).sum(dim=0).clamp(min=1e-8)
            out = torch.bmm(weighted_adj.expand(batch_size, -1, -1), support)
            out = out / degrees.view(1, -1, 1).expand(batch_size, -1, self.out_channels)
            
        else:  # sum aggregation
            out = torch.bmm(weighted_adj.expand(batch_size, -1, -1), support)
        
        if self.bias is not None:
            out += self.bias
            
        return out


class GatedRecurrentFillPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(GatedRecurrentFillPredictor, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.projection = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            predictions: Predicted next values (batch_size, 1)
            hidden: Updated hidden state
        """
        output, hidden = self.gru(x, hidden)
        predictions = self.projection(output[:, -1])
        return predictions, hidden


class TemporalDistanceAttentionModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        encoder_class,
        n_encode_layers=2,
        n_encode_sublayers=None,
        n_decode_layers=None,
        dropout_rate=0.1,
        aggregation="sum",
        aggregation_graph="avg",
        tanh_clipping=10.,
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        normalization='batch',
        learn_affine=True,
        activation='gelu',
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        temporal_horizon=5,  # Number of time steps to consider
        predictor_layers=2,
        distance_influence="inverse"  # Method for distance weighting
    ):
        super(TemporalDistanceAttentionModel, self).__init__()
        
        # Store model parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads
        self.temporal_horizon = temporal_horizon
        
        # Problem type identification (from parent class)
        self.problem = problem
        self.is_vrp = problem.NAME == 'vrp'
        self.is_orienteering = problem.NAME == 'orienteering'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_wc = problem.NAME == 'wcvrp' or problem.NAME == 'cwcvrp' or problem.NAME == 'sdwcvrp'
        self.is_vrpp = problem.NAME == 'vrpp' or problem.NAME == 'cvrpp'
        
        # Determine if we should predict future values
        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
            self.predict_future = True
        else:
            self.predict_future = False
        
        # Initialize encoder (from parent)
        self.encoder = encoder_class(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            n_sublayers=n_encode_sublayers,
            normalization=normalization,
            learn_affine=learn_affine,
            activation=activation,
            dropout=dropout_rate
        )
        
        # Replace standard graph convolution with distance-aware version
        self.distance_gcn = DistanceAwareGraphConvolution(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            distance_influence=distance_influence,
            aggregation=aggregation_graph,
            bias=True
        )
        
        # Temporal components
        self.fill_predictor = GatedRecurrentFillPredictor(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout_rate
        )
        
        self.temporal_embed = nn.Linear(1, embedding_dim)
        
        # For combining spatial and temporal embeddings
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            self.get_activation_fn(activation),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Components from parent class (simplified for brevity)
        self.project_node_embeddings = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Attention mechanism for decoder
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        
        # Additional projection layers
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
    def get_activation_fn(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError("Unknown activation function: {}".format(activation))
    
    def forward(self, x, dist_matrix, return_pi=False, state=None, temporal_data=None):
        """
        Args:
            x: Input data
            dist_matrix: Distance matrix between nodes
            return_pi: Whether to return policy
            state: Current state (for RL)
            temporal_data: Historical data of shape [batch_size, num_nodes, temporal_horizon]
        """
        # Embed inputs
        embeddings, _ = self.encoder(x)
        
        # Apply distance-aware graph convolution
        adj = self.get_adjacency_matrix(x)  # Function to create adjacency matrix from input
        graph_embeddings = self.distance_gcn(embeddings, adj, dist_matrix)
        
        # Process temporal data if available
        if temporal_data is not None and self.predict_future:
            # Shape: [batch_size, num_nodes, temporal_horizon, 1]
            temporal_features = temporal_data.unsqueeze(-1)
            batch_size, num_nodes = temporal_features.shape[0], temporal_features.shape[1]
            
            # Reshape for GRU processing
            reshaped_temporal = temporal_features.view(batch_size * num_nodes, self.temporal_horizon, 1)
            
            # Predict next values
            predictions, _ = self.fill_predictor(reshaped_temporal)
            predictions = predictions.view(batch_size, num_nodes, 1)
            
            # Embed predictions
            temporal_embeddings = self.temporal_embed(predictions)
            
            # Combine spatial (graph) and temporal embeddings
            combined_embeddings = self.combine_embeddings(
                torch.cat([graph_embeddings, temporal_embeddings], dim=-1)
            )
            
            node_embeddings = combined_embeddings
        else:
            # Use only spatial embeddings if no temporal data
            node_embeddings = graph_embeddings
        
        # Project embeddings for attention mechanism
        query = self.project_node_embeddings(node_embeddings)
        key = self.project_fixed_context(node_embeddings)
        
        # Standard attention decoder would be here
        # This is a simplified placeholder for decoder logic
        logits = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.embedding_dim)
        
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
            
        if self.mask_logits:
            # Apply masking if needed
            mask = self.get_mask(x)  # Function to create mask from input
            logits[mask] = -1e8
        
        # Apply log softmax for log probabilities
        log_p = F.log_softmax(logits, dim=-1)
        
        # For RL training, often need to return policy
        if return_pi:
            return node_embeddings, log_p
        
        return log_p
        
    def get_adjacency_matrix(self, x):
        """
        Create adjacency matrix from input data.
        This is a placeholder - implement based on your problem structure.
        """
        # Example implementation - connect all nodes
        batch_size, num_nodes = x.size(0), x.size(1)
        adj = torch.ones(1, num_nodes, num_nodes, device=x.device)
        
        # Typically you'd create sparse connections based on problem structure
        # For example, in VRP you might connect nodes within a certain distance
        
        return adj
        
    def get_mask(self, x):
        """
        Create mask for invalid moves.
        This is a placeholder - implement based on your problem structure.
        """
        # Example implementation
        batch_size, num_nodes = x.size(0), x.size(1)
        mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=x.device)
        
        # Set diagonal to True (can't select self)
        diag_idx = torch.arange(num_nodes, device=x.device)
        mask[:, diag_idx, diag_idx] = True
        
        return mask