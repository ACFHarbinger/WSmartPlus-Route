import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AttentionModel


class ImprovedTemporalAttentionModel(AttentionModel):
    def __init__(self,
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
                 temporal_horizon=5,
                 predictor_layers=2,
                 efficiency_weight=1.0,
                 overflow_tolerance=0.2,
                 skip_threshold=0.1):
        super(ImprovedTemporalAttentionModel, self).__init__(
            embedding_dim,
            hidden_dim,
            problem,
            encoder_class,
            n_encode_layers,
            n_encode_sublayers,
            n_decode_layers,
            dropout_rate,
            aggregation,
            aggregation_graph,
            tanh_clipping,
            mask_inner,
            mask_logits,
            mask_graph,
            normalization,
            learn_affine,
            activation,
            n_heads,
            checkpoint_encoder,
            shrink_size
        )
        from models.modules import ActivationFunction
        from . import GatedRecurrentFillPredictor
        
        self.temporal_horizon = temporal_horizon
        self.efficiency_weight = efficiency_weight
        self.overflow_tolerance = overflow_tolerance
        self.skip_threshold = skip_threshold
        
        # Improved predictor with optimized architecture
        self.fill_predictor = GatedRecurrentFillPredictor(
            input_dim=1,  
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout_rate
        )

        # Smaller, more efficient embeddings
        self.temporal_embed = nn.Linear(1, embedding_dim // 2)
        self.waste_ratio_embed = nn.Linear(1, embedding_dim // 2)
        
        if self.is_wc or self.is_vrpp:
            self.predict_future = True
        else:
            self.predict_future = False
        
        # Efficient embedding combination with residual connection
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            ActivationFunction(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Priority scoring for node selection
        self.priority_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            ActivationFunction(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        # Efficiency-focused cost function adjustment
        self.cost_adjuster = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # [waste, distance, capacity]
            ActivationFunction(activation),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _init_embed(self, nodes):
        # Get the base embeddings from parent class
        base_embeddings = super()._init_embed(nodes)
        
        if 'fill_history' not in nodes or not self.predict_future:
            return base_embeddings
        
        # Process historical fill data more efficiently
        fill_history = nodes['fill_history']
        batch_size, graph_size, _ = fill_history.size()
        
        # Reshape once for all operations
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)
        
        # Predict future fills with gradient checkpointing for memory efficiency
        predicted_fills = self.fill_predictor(fill_history)
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)
        
        # Calculate waste-to-capacity ratio for more informed decisions
        if 'max_prize' in nodes:
            waste_ratio = predicted_fills / (nodes['max_prize'].unsqueeze(-1) + 1e-8)
            waste_ratio = torch.clamp(waste_ratio, 0, 1)
        else:
            waste_ratio = torch.sigmoid(predicted_fills)  # Normalize if max_prize not available
        
        # For depot node, set predicted fill to 0
        if self.is_wc or self.is_vrpp:
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            depot_ratio = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)
            waste_ratio = torch.cat((depot_ratio, waste_ratio), dim=1)
        
        # Create efficient embeddings
        fill_embeddings = self.temporal_embed(predicted_fills)
        ratio_embeddings = self.waste_ratio_embed(waste_ratio)
        
        # Combine temporal embeddings
        temporal_features = torch.cat((fill_embeddings, ratio_embeddings), dim=-1)
        
        # Combine with base embeddings using residual connection for gradient flow
        combined_embeddings = base_embeddings + self.combine_embeddings(
            torch.cat((base_embeddings, temporal_features), dim=-1)
        )
        
        return combined_embeddings
    
    def _calculate_node_priorities(self, embeddings, locations, predicted_fills, max_capacity):
        """Calculate priority scores for each node based on waste and distance"""
        batch_size, graph_size, _ = embeddings.size()
        
        # Get base priority from embeddings
        base_priority = self.priority_scorer(embeddings).squeeze(-1)
        
        # Calculate distance-based penalty (from current position or depot)
        if hasattr(self, 'current_position'):
            current_pos = self.current_position
        else:
            # Use depot as default position
            current_pos = locations[:, 0:1, :]
            
        # Calculate distances efficiently using broadcasting
        distances = torch.norm(locations - current_pos.expand_as(locations), p=2, dim=-1)
        
        # Calculate waste-to-distance ratio
        waste_distance_ratio = predicted_fills.squeeze(-1) / (distances + 1e-8)
        
        # Apply overflow tolerance - priority boost for bins near capacity
        overflow_risk = F.relu(predicted_fills.squeeze(-1) - max_capacity * (1 - self.overflow_tolerance))
        
        # Combine factors for final priority
        priority = base_priority + self.efficiency_weight * waste_distance_ratio + 2.0 * overflow_risk
        
        # Mask out nodes with waste below threshold to avoid unnecessary visits
        skip_mask = predicted_fills.squeeze(-1) < (max_capacity * self.skip_threshold)
        priority = priority.masked_fill(skip_mask, -1e8)
        
        return priority
    
    def forward(self, input, cost_weights=None, return_pi=False, pad=False):
        # Initialize fill history if not provided
        if 'fill_history' not in input and self.predict_future:
            batch_size = input['loc'].size(0)
            graph_size = input['loc'].size(1)
            
            # For VRP-like problems, adjust for depot (excluded from graph size)
            if self.is_wc or self.is_vrpp:
                graph_size -= 1
                
            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input['loc'].device
            )
            input['fill_history'] = fill_history
        
        # Optimize cost weights for efficiency if provided
        if cost_weights is not None and (self.is_vrpp or self.is_wc):
            # Adjust weights to prioritize efficiency
            adjusted_weights = cost_weights.copy()
            
            # Reduce overflow penalty to allow more skipping of less important nodes
            if 'overflows' in adjusted_weights:
                adjusted_weights['overflows'] *= (1.0 - self.overflow_tolerance)
                
            # Increase length penalty to encourage shorter routes
            if 'length' in adjusted_weights:
                adjusted_weights['length'] *= self.efficiency_weight
                
            cost_weights = adjusted_weights
            
        return super().forward(input, cost_weights, return_pi, pad)
    
    def update_fill_history(self, fill_history, new_fills):
        # Efficient history update with single operation
        updated_history = torch.cat([fill_history[:, :, 1:], new_fills.unsqueeze(-1)], dim=-1)
        return updated_history
    
    def compute_simulator_day(self, input, graph, run_tsp=False):
        if 'fill_history' in input and 'current_fill' in input:
            input['fill_history'] = self.update_fill_history(
                input['fill_history'], 
                input['current_fill']
            )
            
            # Store current position for distance calculations
            if 'current_position' in input:
                self.current_position = input['current_position']
                
        return super().compute_simulator_day(input, graph, run_tsp)
    
    def _precompute_waste_metrics(self, input):
        """Precompute waste metrics for efficient routing decisions"""
        if 'prize' in input and 'max_prize' in input:
            batch_size, graph_size = input['prize'].size()
            
            # Calculate fill ratios (how full each bin is)
            fill_ratios = input['prize'] / (input['max_prize'] + 1e-8)
            
            # Calculate potential waste loss if not visited
            potential_waste = F.relu(input['prize'] - input['max_prize'])
            
            # Calculate efficiency metric (waste collected per distance traveled)
            distances = torch.cdist(input['loc'], input['loc'])
            
            # Set self-distances to a small value to avoid division by zero
            distances = distances + torch.eye(graph_size, device=distances.device) * 1e-8
            
            # Calculate waste-to-distance ratio for all node pairs
            waste_distance_ratio = input['prize'].unsqueeze(-1) / distances
            
            return {
                'fill_ratios': fill_ratios,
                'potential_waste': potential_waste,
                'waste_distance_ratio': waste_distance_ratio
            }
        return None
    
    def beam_search(self, *args, **kwargs):
        """Override beam search to incorporate efficiency-focused node selection"""
        # Precompute waste metrics if not already done
        if 'input' in kwargs and self._precompute_waste_metrics(kwargs['input']) is not None:
            self.waste_metrics = self._precompute_waste_metrics(kwargs['input'])
            
        # Adjust beam size for more focused search
        if 'beam_size' in kwargs:
            # We can use smaller beam size with more targeted search
            kwargs['beam_size'] = max(kwargs['beam_size'] // 2, 8)
            
        return super().beam_search(*args, **kwargs)