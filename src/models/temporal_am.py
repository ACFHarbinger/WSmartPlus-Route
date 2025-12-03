import torch
import torch.nn as nn

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
                 aggregation="sum",
                 aggregation_graph="mean",
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 mask_graph=False,
                 normalization='batch',
                 norm_learn_affine=True,
                 norm_track_stats=False,
                 norm_eps_alpha=1e-05,
                 norm_momentum_beta=0.1,
                 lrnorm_k=1.0,
                 gnorm_groups=3,
                 activation_function='gelu',
                 af_param=1.0,
                 af_threshold=6.0,
                 af_replacement_value=6.0,
                 af_num_params=3,
                 af_uniform_range=[0.125, 1/3],
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 temporal_horizon=5,  
                 predictor_layers=2):  
        super(TemporalAttentionModel, self).__init__(
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
            norm_learn_affine,
            norm_track_stats,
            norm_eps_alpha,
            norm_momentum_beta,
            lrnorm_k,
            gnorm_groups,
            activation_function,
            af_param,
            af_threshold,
            af_replacement_value,
            af_num_params,
            af_uniform_range,
            n_heads,
            checkpoint_encoder,
            shrink_size,
            temporal_horizon
        )
        from models.modules import ActivationFunction
        from . import GatedRecurrentFillPredictor
        self.fill_predictor = GatedRecurrentFillPredictor(
            input_dim=1,  
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout_rate
        )

        self.temporal_embed = nn.Linear(1, embedding_dim)
        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
            self.predict_future = True
        else:
            self.predict_future = False
        
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            ActivationFunction(self.activation, self.af_param, self.threshold, 
                            self.replacement_value, self.n_params, self.uniform_range),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def _init_embed(self, nodes):
        # Get the base embeddings from parent class
        base_embeddings = super()._init_embed(nodes, temporal_features=False)
        
        if 'fill_history' not in nodes or not self.predict_future:
            return base_embeddings
        
        fill_history = nodes['fill_history']
        batch_size, graph_size, _ = fill_history.size()
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)
        
        predicted_fills = self.fill_predictor(fill_history)
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)
        
        # For depot node, set predicted fill to 0
        if self.is_vrpp or self.is_wc or self.is_vrp or self.is_orienteering or self.is_pctsp:
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)
        
        fill_embeddings = self.temporal_embed(predicted_fills)
        combined_embeddings = self.combine_embeddings(
            torch.cat((base_embeddings, fill_embeddings), dim=-1)
        )
        return combined_embeddings
    
    def forward(self, input, cost_weights=None, return_pi=False, pad=False):
        if 'fill_history' not in input and self.predict_future:
            batch_size = input['loc'].size(0)
            graph_size = input['loc'].size(1)
            
            # For VRP-like problems, adjust for depot (excluded from graph size)
            if self.is_vrpp or self.is_wc or self.is_vrp or self.is_orienteering or self.is_pctsp:
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
