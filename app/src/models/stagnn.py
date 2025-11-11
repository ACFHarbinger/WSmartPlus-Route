import torch
import torch.nn as nn

from .modules import GraphConvolution


class SpatioTemporalAttentionGNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_time_steps: int,
                 num_nodes: int,
                 temporal_horizon: int = 5,
                 n_heads: int = 4,
                 dropout_rate: float = 0.1,
                 aggregation: str = "sum",
                 normalization: str = 'batch'):
        """
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features
            out_channels: Number of output features per node
            num_time_steps: Number of time steps in the sequence
            num_nodes: Number of nodes in the graph
            temporal_horizon: Number of historical time steps for attention
            n_heads: Number of attention heads
            dropout_rate: Dropout probability
            aggregation: Graph aggregation method ("sum", "mean", "max")
            normalization: Type of normalization ('batch' or 'layer')
        """
        super(SpatioTemporalAttentionGNN, self).__init__()
        
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.temporal_horizon = temporal_horizon
        
        # Spatial Graph Convolution
        self.spatial_gcn = GraphConvolution(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggregation=aggregation
        )
        
        # Temporal Attention
        self.temporal_embed = nn.Linear(hidden_channels, hidden_channels)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=n_heads,
            batch_first=True
        )
        
        # History Predictor (inspired by TemporalAttentionModel)
        self.history_predictor = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, adj, history=None):
        """
        Args:
            x: Input tensor (batch_size, num_time_steps, num_nodes, in_channels)
            adj: Adjacency matrix (num_nodes, num_nodes)
            history: Historical data (batch_size, num_nodes, temporal_horizon, in_channels) or None
            
        Returns:
            Output tensor (batch_size, num_time_steps, num_nodes, out_channels)
        """
        batch_size, T, N, _ = x.shape
        
        # Spatial processing
        x_spatial = x.view(batch_size * T, N, -1)
        spatial_features = self.spatial_gcn(x_spatial, adj)
        spatial_features = torch.relu(spatial_features)
        
        # Reshape: (batch_size, num_time_steps, num_nodes, hidden_channels)
        spatial_features = spatial_features.view(batch_size, T, N, -1)
        
        # Temporal attention processing
        temporal_embed = self.temporal_embed(spatial_features)
        # Reshape for attention: (batch_size * num_nodes, num_time_steps, hidden_channels)
        temporal_embed = temporal_embed.transpose(1, 2).reshape(batch_size * N, T, -1)
        
        # Apply temporal attention
        attn_output, _ = self.temporal_attention(temporal_embed, temporal_embed, temporal_embed)
        attn_output = attn_output.view(batch_size, N, T, -1).transpose(1, 2)
        
        # Process historical data if provided
        if history is not None:
            # (batch_size, num_nodes, temporal_horizon, in_channels)
            history_flat = history.view(batch_size * N, self.temporal_horizon, -1)
            history_embed = self.spatial_gcn(history_flat, adj)
            # Predict next step from history
            pred_history, _ = self.history_predictor(history_embed)
            pred_history = pred_history[:, -1, :].view(batch_size, N, -1)
            
            # Combine with attention output
            combined = torch.cat([attn_output, pred_history.unsqueeze(1).expand(-1, T, -1, -1)], dim=-1)
        else:
            combined = torch.cat([attn_output, attn_output], dim=-1)  # Simple concatenation if no history
            
        # Final processing
        combined = self.norm(combined.transpose(2, 3)).transpose(2, 3)
        output = self.output_layer(combined)
        
        return output
    
    def update_history(self, history, new_data):
        """
        Update temporal history with new observations
        """
        if history is None:
            return new_data.unsqueeze(2)
        updated = torch.cat([history[:, :, 1:], new_data.unsqueeze(2)], dim=2)
        return updated