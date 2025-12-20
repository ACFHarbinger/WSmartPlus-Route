import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AttentionModel


class TemporalAttentionModel(AttentionModel):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 encoder_class,
                 n_encode_layers=2,
                 n_encode_sublayers=None,
                 n_decode_layers=None,
                 dropout_rate=0.1,
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
                 temporal_horizon=5,  # Number of time steps to consider for temporal patterns
                 predictor_layers=2,
                 temporal_attention_heads=4,
                 use_transformer_predictor=True,
                 use_multi_head_attention=True,
                 multi_scale_prediction=True,
                 seasonal_patterns=False,
                 use_residual_connections=True):
        super(TemporalAttentionModel, self).__init__(
            embedding_dim,
            hidden_dim,
            problem,
            encoder_class,
            n_encode_layers,
            n_encode_sublayers,
            n_decode_layers,
            dropout_rate,
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
        
        self.temporal_horizon = temporal_horizon
        self.use_transformer_predictor = use_transformer_predictor
        self.multi_scale_prediction = multi_scale_prediction
        self.seasonal_patterns = seasonal_patterns
        self.use_residual_connections = use_residual_connections
        
        # Create advanced temporal prediction module
        if use_transformer_predictor:
            from . import TransformerFillPredictor
            self.fill_predictor = TransformerFillPredictor(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                num_heads=temporal_attention_heads,
                dropout=dropout_rate
            )
        else:
            from . import GatedRecurrentFillPredictor
            self.fill_predictor = GatedRecurrentFillPredictor(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate
            )
        
        # Multi-scale prediction for different time horizons
        if multi_scale_prediction:
            self.short_term_predictor = nn.Linear(temporal_horizon, 1)
            self.medium_term_predictor = nn.Linear(temporal_horizon // 2, 1)
            self.long_term_predictor = nn.Linear(temporal_horizon // 4 + 1, 1)
            self.scale_combiner = nn.Linear(3, 1)
        
        # Seasonal pattern detection (if enabled)
        if seasonal_patterns:
            self.seasonal_detector = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
            self.seasonal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced temporal embedding with optional multi-head attention
        self.temporal_embed = nn.Linear(1, embedding_dim)
        
        if use_multi_head_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=temporal_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
        else:
            self.temporal_attention = None
        
        # Improved embedding combination with residual connections
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            ActivationFunction(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Optional normalization layer for combined embeddings
        self.embedding_norm = nn.LayerNorm(embedding_dim) if use_residual_connections else None
        
        # Determine which problems need future prediction
        if self.is_vrpp or self.is_wc:
            self.predict_future = True
        else:
            self.predict_future = False
        
        # Confidence estimator for predictions
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ActivationFunction(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _multi_scale_predict(self, fill_history):
        batch_size, seq_len, _ = fill_history.size()
        
        # Short-term prediction (using full history)
        short_term = self.short_term_predictor(fill_history.squeeze(-1))
        
        # Medium-term prediction (using downsampled history)
        medium_sample_size = self.temporal_horizon // 2
        medium_indices = torch.linspace(0, self.temporal_horizon-1, medium_sample_size, dtype=torch.long)
        medium_history = fill_history[:, medium_indices, :]
        medium_term = self.medium_term_predictor(medium_history.squeeze(-1))
        
        # Long-term prediction (using more downsampled history)
        long_sample_size = self.temporal_horizon // 4 + 1
        long_indices = torch.linspace(0, self.temporal_horizon-1, long_sample_size, dtype=torch.long)
        long_history = fill_history[:, long_indices, :]
        long_term = self.long_term_predictor(long_history.squeeze(-1))
        
        # Combine predictions from different scales
        scales = torch.cat([short_term, medium_term, long_term], dim=-1)
        combined = self.scale_combiner(scales).unsqueeze(-1)
        
        return combined
    
    def _detect_seasonal_patterns(self, fill_history):
        # Reshape for 1D convolution: [batch*nodes, 1, horizon]
        batch_size, seq_len, _ = fill_history.size()
        history_reshaped = fill_history.transpose(1, 2)
        
        # Apply convolution to detect patterns
        seasonal_features = self.seasonal_detector(history_reshaped)
        seasonal_features = F.relu(seasonal_features)
        
        # Pool features to get a fixed-size representation
        seasonal_summary = self.seasonal_pooling(seasonal_features).view(batch_size, -1)
        
        return seasonal_summary
    
    def _init_embed(self, nodes):
        # Get the base embeddings from parent class
        base_embeddings = super()._init_embed(nodes)
        
        if 'fill_history' not in nodes or not self.predict_future:
            return base_embeddings
        
        fill_history = nodes['fill_history']
        batch_size, graph_size, _ = fill_history.size()
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)
        
        # Get predicted fills using the selected predictor
        predicted_fills = self.fill_predictor(fill_history)
        
        # Optional multi-scale prediction
        if self.multi_scale_prediction:
            multi_scale_prediction = self._multi_scale_predict(fill_history)
            # Blend with the main prediction based on confidence
            confidence = self.confidence_estimator(predicted_fills)
            predicted_fills = confidence * predicted_fills + (1 - confidence) * multi_scale_prediction
        
        # Reshape back to [batch_size, graph_size, 1]
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)
        
        # For depot node, set predicted fill to 0
        if self.is_vrpp or self.is_wc:
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)
        
        # Convert predicted fills to embeddings
        fill_embeddings = self.temporal_embed(predicted_fills)
        
        # Apply multi-head attention if enabled
        if self.temporal_attention is not None:
            fill_embeddings, _ = self.temporal_attention(
                fill_embeddings, fill_embeddings, fill_embeddings
            )
        
        # Combine base and fill embeddings
        combined_embeddings = self.combine_embeddings(
            torch.cat((base_embeddings, fill_embeddings), dim=-1)
        )
        
        # Apply residual connection if enabled
        if self.use_residual_connections:
            combined_embeddings = self.embedding_norm(combined_embeddings + base_embeddings)
        
        return combined_embeddings
    
    def forward(self, input, cost_weights=None, return_pi=False, pad=False, fill_history=None):
        # If fill_history is provided as a parameter, use it
        if fill_history is not None:
            input['fill_history'] = fill_history
        
        # Initialize fill_history if not present
        if 'fill_history' not in input and self.predict_future:
            batch_size = input['loc'].size(0)
            graph_size = input['loc'].size(1)
            
            # For VRP-like problems, adjust for depot (excluded from graph size)
            if self.is_vrpp or self.is_wc:
                graph_size -= 1
                
            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input['loc'].device
            )
            input['fill_history'] = fill_history
            
        return super().forward(input, cost_weights, return_pi, pad)
    
    def update_fill_history(self, fill_history, new_fills):
        updated_history = fill_history.clone()
        updated_history[:, :, :-1] = fill_history[:, :, 1:]
        updated_history[:, :, -1] = new_fills
        return updated_history
    
    def compute_simulator_day(self, input, graph, run_tsp=False):
        if 'fill_history' in input and 'current_fill' in input:
            input['fill_history'] = self.update_fill_history(
                input['fill_history'], 
                input['current_fill']
            )
        return super().compute_simulator_day(input, graph, run_tsp)
    
    def get_fill_predictions(self, fill_history):
        """
        Get explicit fill predictions without running the full model.
        Useful for analysis and visualization.
        
        Args:
            fill_history: Tensor of shape [batch_size, graph_size, temporal_horizon]
            
        Returns:
            Tensor of predicted fills of shape [batch_size, graph_size, 1]
        """
        batch_size, graph_size, _ = fill_history.size()
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)
        
        # Get base predictions
        predictions = self.fill_predictor(fill_history)
        
        # Apply multi-scale prediction if enabled
        if self.multi_scale_prediction:
            multi_scale_prediction = self._multi_scale_predict(fill_history)
            confidence = self.confidence_estimator(predictions)
            predictions = confidence * predictions + (1 - confidence) * multi_scale_prediction
        
        return predictions.view(batch_size, graph_size, 1)