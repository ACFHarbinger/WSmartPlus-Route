import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.src.utils.functions import sample_many
from logic.src.utils.beam_search import CachedLookup
from logic.src.models.context_embedder import WCContextEmbedder, VRPPContextEmbedder
from logic.src.models.model_factory import NeuralComponentFactory


class AttentionModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 component_factory,
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
                 pomo_size=0,
                 temporal_horizon=0,
                 spatial_bias=False,
                 spatial_bias_scale=1.0,
                 entropy_weight=0.0,
                 predictor_layers=None):
        super(AttentionModel, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.problem = problem
        self.pomo_size = pomo_size
        self.checkpoint_encoder = checkpoint_encoder
        self.aggregation_graph = aggregation_graph
        self.temporal_horizon = temporal_horizon
        
        # Initialize Context Embedder Strategy
        self.is_wc = problem.NAME == 'wcvrp' or problem.NAME == 'cwcvrp' or problem.NAME == 'sdwcvrp'
        self.is_vrpp = problem.NAME == 'vrpp' or problem.NAME == 'cvrpp'
        node_dim = 3
        if self.is_wc:
            self.context_embedder = WCContextEmbedder(embedding_dim, node_dim=node_dim, temporal_horizon=temporal_horizon)
        else:
            self.context_embedder = VRPPContextEmbedder(embedding_dim, node_dim=node_dim, temporal_horizon=temporal_horizon)

        step_context_dim = self.context_embedder.step_context_dim
        
        # Use Factory to create components
        if not isinstance(component_factory, NeuralComponentFactory):
            pass

        self.embedder = component_factory.create_encoder(
            n_heads=self.n_heads,
            embed_dim=self.embedding_dim,
            n_layers=n_encode_layers,
            n_sublayers=n_encode_sublayers,
            feed_forward_hidden=self.hidden_dim,
            normalization=normalization,
            epsilon_alpha=norm_eps_alpha,
            learn_affine=norm_learn_affine,
            track_stats=norm_track_stats,
            momentum_beta=norm_momentum_beta,
            locresp_k=lrnorm_k,
            n_groups=gnorm_groups,
            activation=activation_function,
            af_param=af_param,
            threshold=af_threshold,
            replacement_value=af_replacement_value,
            n_params=af_num_params,
            uniform_range=af_uniform_range,
            dropout_rate=dropout_rate,
            agg=aggregation
        )

        self.decoder = component_factory.create_decoder(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            problem=problem,
            n_heads=self.n_heads,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            mask_graph=mask_graph,
            shrink_size=shrink_size,
            pomo_size=pomo_size,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale
        )
        
        # Configure decoder step context
        if hasattr(self.decoder, 'set_step_context_dim'):
             self.decoder.set_step_context_dim(step_context_dim)

    def set_decode_type(self, decode_type, temp=None):
        self.decoder.set_decode_type(decode_type, temp)

    def _get_initial_embeddings(self, input):
        return self.context_embedder.init_node_embeddings(input)

    def forward(self, input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None, **kwargs):
        edges = input.get('edges', None)
        dist_matrix = input.get('dist', None) # Using 'dist' key consistent with original
        
        node_embeddings = self._get_initial_embeddings(input)

        if self.checkpoint_encoder and self.training:
            embeddings = torch.utils.checkpoint.checkpoint(self.embedder, node_embeddings, edges, use_reentrant=False)
        else:
            if getattr(self.embedder, 'init_edge_embed', None) is not None:
                embeddings = self.embedder(node_embeddings, edges, dist=dist_matrix)
            else:
                embeddings = self.embedder(node_embeddings, edges)
        
        if dist_matrix is not None:
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
            
            if self.pomo_size > 0:
                def expand(t):
                    if t is None: return None
                    if isinstance(t, torch.Tensor): return t.repeat_interleave(self.pomo_size, dim=0)
                    if isinstance(t, dict): return {k: expand(v) for k, v in t.items()}
                    return t
                
                expanded_input = expand(input)
                expanded_embeddings = expand(embeddings)
                
                if dist_matrix.size(0) == 1:
                    expanded_dist_matrix = dist_matrix.expand(expanded_embeddings.size(0), -1, -1)
                else:
                    expanded_dist_matrix = dist_matrix.repeat_interleave(self.pomo_size, dim=0)
                
                expanded_mask = expand(mask)
                
                log_p, pi = self.decoder(expanded_input, expanded_embeddings, cost_weights, expanded_dist_matrix, mask=expanded_mask, expert_pi=expert_pi)
                cost, cost_dict, mask = self.problem.get_costs(expanded_input, pi, cost_weights, expanded_dist_matrix)
            else:
                if dist_matrix.size(0) == 1 and embeddings.size(0) > 1:
                    dist_matrix = dist_matrix.expand(embeddings.size(0), -1, -1)
                log_p, pi = self.decoder(input, embeddings, cost_weights, dist_matrix, mask=mask, expert_pi=expert_pi)
                cost, cost_dict, mask = self.problem.get_costs(input, pi, cost_weights, dist_matrix)
        else:
            log_p, pi = self.decoder(input, embeddings, cost_weights, None, mask=mask, expert_pi=expert_pi)
            cost, cost_dict, mask = self.problem.get_costs(input, pi, cost_weights, None)

        
        use_kl = kwargs.get('kl_loss', False) and expert_pi is not None
        if expert_pi is not None and use_kl:
            res = self.decoder._calc_log_likelihood(log_p, expert_pi, mask, return_entropy=self.training, kl_loss=True)
        else:
            res = self.decoder._calc_log_likelihood(log_p, pi, mask, return_entropy=self.training, kl_loss=False)
             
        if self.training:
            ll, entropy = res
        else:
            ll = res
            entropy = None

        if return_pi:
            if pad:
                pad_dim = input['loc'].size(1) + 1
                pi = F.pad(pi, (0, (pad_dim) - pi.size(-1)), value=0)
            return cost, ll, cost_dict, pi, entropy
        return cost, ll, cost_dict, None, entropy

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input, edges):
        embeddings = self.embedder(self.context_embedder.init_node_embeddings(input), edges)
        return CachedLookup(self.decoder._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        return self.decoder.propose_expansions(beam, fixed, expand_size, normalize, max_calc_batch_size)

    def sample_many(self, input, cost_weights=None, batch_rep=1, iter_rep=1):
        edges = input.pop('edges') if 'edges' in input.keys() else None
        # This requires decoder delegation
        return sample_many(
            lambda input: self.decoder._inner(*input[:3], cost_weights, input[3]), 
            lambda input, pi: self.problem.get_costs(input[0], pi, cost_weights)[:2],
            (input, edges, self.embedder(self.context_embedder.init_node_embeddings(input), edges), input.get('dist_matrix', None)), 
            batch_rep, iter_rep
        )