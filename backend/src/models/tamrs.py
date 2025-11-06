import torch
import torch.nn as nn

from . import AttentionModel
from .route_scheduling import RouteSchedulingModule


class TemporalAttentionModelWithScheduling(AttentionModel):
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
                 temporal_horizon=5,  # Number of time steps to consider for temporal patterns  
                 predictor_layers=2,
                 min_fullness_threshold=0.3,
                 critical_threshold=0.8,
                 route_cost_estimate=1.0,
                 skip_penalty_factor=0.5):
        super(TemporalAttentionModelWithScheduling, self).__init__(
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
        self.route_cost_estimate = route_cost_estimate
        
        # Fill predictor
        self.fill_predictor = GatedRecurrentFillPredictor(
            input_dim=1,  
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout_rate
        )

        # Temporal embedding
        self.temporal_embed = nn.Linear(1, embedding_dim)
        
        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
            self.predict_future = True
        else:
            self.predict_future = False
        
        # Combining embeddings
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            ActivationFunction(activation),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Add routing scheduling module
        self.route_scheduler = RouteSchedulingModule(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads // 2,  # Use fewer heads for efficiency
            dropout=dropout_rate,
            activation=activation,
            min_fullness_threshold=min_fullness_threshold,
            critical_threshold=critical_threshold,
            skip_penalty_factor=skip_penalty_factor
        )
        
        # Adaptive decision threshold
        self.decision_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Keep track of collection history to improve scheduling
        self.register_buffer('collection_history', torch.zeros(temporal_horizon))
        self.day_counter = 0
    
    def _init_embed(self, nodes):
        # Get the base embeddings from parent class
        base_embeddings = super()._init_embed(nodes)
        
        if 'fill_history' not in nodes or not self.predict_future:
            return base_embeddings
        
        fill_history = nodes['fill_history']
        batch_size, graph_size, _ = fill_history.size()
        fill_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)
        
        # Get predicted fills
        predicted_fills = self.fill_predictor(fill_history)
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)
        
        # For depot node, set predicted fill to 0
        if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)
        
        # Store for route scheduling decision
        self.predicted_node_fills = predicted_fills.squeeze(-1)
        
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
            if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
                graph_size -= 1
                
            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input['loc'].device
            )
            input['fill_history'] = fill_history
            
        # Get day features if available
        day_features = input.get('day_features', None)
            
        # Call the parent forward to get embeddings and other outputs
        outputs = super().forward(input, cost_weights, return_pi, pad)
        
        # After generating the embeddings, decide whether to execute route
        if hasattr(self, 'predicted_node_fills') and 'max_prize' in input:
            # Get the embeddings
            if hasattr(self, '_embeddings'):
                node_embeddings = self._embeddings
            else:
                # If embeddings not directly accessible, approximate from decoder
                node_embeddings = self.project_node_embeddings(input)
                
            # Max capacity from input
            max_capacity = input['max_prize']
            
            # Make route execution decision
            decision_prob, metrics = self.route_scheduler(
                node_embeddings=node_embeddings,
                fill_levels=self.predicted_node_fills,
                max_capacity=max_capacity,
                fill_history=input.get('fill_history', None),
                day_features=day_features
            )
            
            # Store decision metrics for later use
            self.scheduling_metrics = metrics
            self.route_decision = decision_prob
            
            # Modify cost based on route execution decision
            if cost_weights is not None and self.training:
                # During training, allow partial execution to learn the trade-offs
                cost_adjustment = 1.0 - (1.0 - decision_prob) * self.route_cost_estimate
                
                # Apply cost adjustment to all weights except overflow penalties
                for key in cost_weights:
                    if key != 'overflows':
                        cost_weights[key] = cost_weights[key] * cost_adjustment
                
                # We can also use this to adjust the returned cost
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    cost = outputs[0]
                    pi = outputs[1]
                    outputs = (cost * cost_adjustment, pi)
        
        return outputs
    
    def update_fill_history(self, fill_history, new_fills):
        updated_history = fill_history.clone()
        updated_history[:, :, :-1] = fill_history[:, :, 1:]
        updated_history[:, :, -1] = new_fills
        return updated_history
    
    def compute_simulator_day(self, input, graph, run_tsp=False):
        # Update fill history if available
        if 'fill_history' in input and 'current_fill' in input:
            input['fill_history'] = self.update_fill_history(
                input['fill_history'], 
                input['current_fill']
            )
        
        # Get day features or create default
        day_of_week = self.day_counter % 7
        week_of_month = (self.day_counter // 7) % 4
        is_holiday = 1.0 if day_of_week >= 5 else 0.0  # Weekends as holidays
        weather_factor = torch.rand(1).item()  # Random weather (could be deterministic)
        staff_availability = 1.0 - 0.2 * is_holiday  # Less staff on holidays
        
        day_features = torch.tensor(
            [day_of_week/6, week_of_month/3, is_holiday, weather_factor, staff_availability],
            device=input['loc'].device
        ).unsqueeze(0)
        
        input['day_features'] = day_features
        
        # First run the model to get the route decision
        with torch.no_grad():
            _ = self.forward(input)
            
            # Get binary decision on whether to execute route
            execute_route = self.route_scheduler.get_decision(
                self.route_decision,
                self.scheduling_metrics,
                hard_threshold=self.decision_threshold
            )
            
            # Update collection history
            self.collection_history = torch.roll(self.collection_history, -1)
            self.collection_history[-1] = execute_route.item()
            
            # Update day counter
            self.day_counter += 1
        
        # If decision is to skip, return empty route
        if execute_route.item() < 0.5:
            # Create empty route (just depot)
            if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc or self.is_vrpp:
                empty_route = torch.zeros(
                    (input['loc'].size(0), 1), 
                    dtype=torch.long,
                    device=input['loc'].device
                )
                
                # Return with execution flag
                return empty_route, 0, {'executed': False, 'metrics': self.scheduling_metrics}
        
        # Execute normal route computation
        route, cost, attention_dict = super().compute_simulator_day(input, graph, run_tsp)
        
        # If route is a tuple with additional info
        if isinstance(route, tuple):
            route_data = route[0]
            additional_info = route[1]
            additional_info['executed'] = True
            additional_info['metrics'] = self.scheduling_metrics
            return route_data, additional_info
            
        return route, cost, {'executed': True, 'metrics': self.scheduling_metrics, **attention_dict}
    
    def project_node_embeddings(self, input):
        """Get approximate node embeddings if not directly accessible"""
        batch_size = input['loc'].size(0)
        graph_size = input['loc'].size(1)
        
        # Simplified embedding projection when actual embeddings not accessible
        loc_embedding = self.init_embed_bias(input['loc'])
        
        if 'prize' in input:
            prize_embedding = self.init_embed_bias(input['prize'].unsqueeze(-1))
            embeddings = loc_embedding + prize_embedding
        else:
            embeddings = loc_embedding
            
        return embeddings
        
    def init_embed_bias(self, x):
        """Simple embedding projection"""
        return x @ self.init_embed.weight.t() + self.init_embed.bias
    
    def get_scheduling_stats(self):
        """Return statistics about routing decisions for monitoring"""
        if hasattr(self, 'collection_history'):
            execution_rate = self.collection_history.mean().item()
            last_7_days = self.collection_history[-7:].mean().item() if self.day_counter >= 7 else float('nan')
            
            return {
                'lifetime_execution_rate': execution_rate,
                'recent_execution_rate': last_7_days,
                'days_since_last_collection': (1 - self.collection_history).flip(0).argmax().item() 
                                              if self.collection_history[-1] < 0.5 else 0
            }
        return {}