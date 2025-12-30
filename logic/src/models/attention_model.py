import math
import torch
import typing
import torch.nn as nn
import torch.nn.functional as F

from logic.src.utils.beam_search import CachedLookup
from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params
from logic.src.utils.functions import compute_in_batches, add_attention_hooks, sample_many
from logic.src.policies import find_route, get_route_cost, get_multi_tour, local_search_2opt_vectorized


class AttentionModelFixed(typing.NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return self[key]


class AttentionModel(nn.Module):
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
        self.dropout_rate = dropout_rate
        self.n_encode_layers = n_encode_layers
        self.n_encode_sublayers = n_encode_sublayers

        self.normalization = normalization
        self.epsilon_alpha = norm_eps_alpha
        self.learn_affine = norm_learn_affine
        self.track_stats = norm_track_stats
        self.momentum_beta = norm_momentum_beta
        self.locresp_k = lrnorm_k
        self.n_groups = gnorm_groups

        self.activation = activation_function
        self.af_param = af_param
        self.threshold = af_threshold
        self.replacement_value = af_replacement_value
        self.n_params = af_num_params
        self.uniform_range = af_uniform_range

        self.temp = 1.0
        self.decode_type = None
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph

        self.temporal_horizon = temporal_horizon
        self.shrink_size = shrink_size
        self.checkpoint_encoder = checkpoint_encoder

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.pomo_size = pomo_size
        self.spatial_bias = spatial_bias
        self.spatial_bias_scale = spatial_bias_scale
        self.entropy_weight = entropy_weight

        self.problem = problem
        self.allow_partial = problem.NAME == 'sdwcvrp'
        self.is_wc = problem.NAME == 'wcvrp' or problem.NAME == 'cwcvrp' or problem.NAME == 'sdwcvrp'
        self.is_vrpp = problem.NAME == 'vrpp' or problem.NAME == 'cvrpp'

        assert self.is_wc or self.is_vrpp, "Unsupported problem: {}".format(problem.NAME)

        # Problem specific context parameters (placeholder and step context dimension)
        # Embedding of last node + remaining_capacity / remaining length / current profit / current overflows + length
        if self.is_wc:
            step_context_dim = embedding_dim + 2
        else:
            # vrpp
            step_context_dim = embedding_dim + 1
        
        node_dim = 3  # x, y, demand / prize / waste (vrpp has waste, wc has waste)

        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        self.init_embed = nn.Linear(node_dim + temporal_horizon, embedding_dim)
        self.embedder = encoder_class(
            n_heads=self.n_heads,
            embed_dim=self.embedding_dim,
            n_layers=self.n_encode_layers,
            n_sublayers=self.n_encode_sublayers,
            feed_forward_hidden=self.hidden_dim,
            normalization=self.normalization,
            epsilon_alpha=self.epsilon_alpha,
            learn_affine=self.learn_affine,
            track_stats=self.track_stats,
            momentum_beta=self.momentum_beta,
            locresp_k=self.locresp_k,
            n_groups=self.n_groups,
            activation=self.activation,
            af_param=self.af_param,
            threshold=self.threshold,
            replacement_value=self.replacement_value,
            n_params=self.n_params,
            uniform_range=self.uniform_range,
            dropout_rate=self.dropout_rate,
            agg=self.aggregation
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param cost_weights: dictionary with weights for each term of the cost function
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :param pad: whether to pad the output sequences in order to use return_pi with multiple GPUs
        :param mask: (batch_size, graph_size) boolean mask where True indicates nodes to be masked (skipped)
        :return:
        """
        edges = input.get('edges', None)
        dist_matrix = input.get('dist', None)
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings = torch.utils.checkpoint.checkpoint(self.embedder, self._init_embed(input), edges, use_reentrant=False)
        else:
            # Check if embedder accepts 'dist' (e.g. GatedGraphAttConvEncoder)
            # We can check via inspection or just try/except, but clean way is check attribute or name
            if getattr(self.embedder, 'init_edge_embed', None) is not None:
                embeddings = self.embedder(self._init_embed(input), edges, dist=dist_matrix)
            else:
                embeddings = self.embedder(self._init_embed(input), edges)

        # profit_vars is not a direct argument to forward, it's derived or passed through input
        # Assuming it's handled within _inner or not needed here based on original code
        
        # If POMO is enabled, we expand the batch
        if self.pomo_size > 0:
            def expand(t):
                if t is None:
                    return None
                if isinstance(t, torch.Tensor):
                    # (B, ...) -> (B * pomo_size, ...)
                    return t.repeat_interleave(self.pomo_size, dim=0)
                if isinstance(t, dict):
                    return {k: expand(v) for k, v in t.items()}
                return t
            
            expanded_input = expand(input)
            expanded_embeddings = expand(embeddings)
            expanded_edges = expand(edges)
            
            # Efficiently expand dist_matrix for POMO
            if dist_matrix is None:
                expanded_dist_matrix = None
            elif dist_matrix.size(0) == 1:
                # Shared dist_matrix: expanded to (B*P, N, N) using view
                expanded_dist_matrix = dist_matrix.expand(expanded_embeddings.size(0), -1, -1)
            else:
                # Per-instance: repeat interleave to (B*P, N, N)
                expanded_dist_matrix = dist_matrix.repeat_interleave(self.pomo_size, dim=0)
            
            expanded_mask = expand(mask)
            
            _log_p, pi = self._inner(expanded_input, expanded_edges, expanded_embeddings, cost_weights, expanded_dist_matrix, profit_vars=None, mask=expanded_mask, expert_pi=expert_pi)
            cost, cost_dict, mask = self.problem.get_costs(expanded_input, pi, cost_weights, expanded_dist_matrix)
        else:
            if dist_matrix is not None and dist_matrix.size(0) == 1 and embeddings.size(0) > 1:
                dist_matrix = dist_matrix.expand(embeddings.size(0), -1, -1)
            _log_p, pi = self._inner(input, edges, embeddings, cost_weights, dist_matrix, profit_vars=None, mask=mask, expert_pi=expert_pi)
            cost, cost_dict, mask = self.problem.get_costs(input, pi, cost_weights, dist_matrix)
        
        # Log likelyhood and entropy
        # Use return_entropy if training or specifically requested
        res = self._calc_log_likelihood(_log_p, pi, mask, return_entropy=self.training)
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
    
    def compute_batch_sim(self, input, dist_matrix, hrl_manager=None, waste_history=None, threshold=0.5, mask_threshold=0.5):
        hook_data = add_attention_hooks(self.embedder)
        edges = input.get('edges', None)
        
        mask = None
        if hrl_manager is not None and waste_history is not None:
            # Static: Customer Locations (Batch, N, 2)
            static_feat = input['loc']
            # Dynamic: Waste History (Batch, N, History)
            dynamic_feat = waste_history
            
            # Compute Global Features
            current_waste = dynamic_feat[:, :, -1]

            # 1. Critical Ratio Now
            critical_mask = (current_waste > hrl_manager.critical_threshold).float()
            critical_ratio = critical_mask.mean(dim=1, keepdim=True) # (B, 1)
            
            # 2. Max Current Waste
            max_current_waste = current_waste.max(dim=1, keepdim=True)[0] # (B, 1)
            
            # Combine: (B, 2)
            global_features = torch.cat([
                critical_ratio, 
                max_current_waste, 
            ], dim=1)
            
            # Get Action (Deterministic)
            mask_action, gate_action, _ = hrl_manager.select_action(
                static_feat, dynamic_feat, global_features, 
                deterministic=True, threshold=threshold, mask_threshold=mask_threshold
            )
            
            # Construct Mask
            # mask_action: 1=Visit, 0=Skip. AM Mask: True=Masked(Skip), False=Keep.
            mask = (mask_action == 0)
            
            # Apply Gate: If Gate=0, Mask ALL
            gate_mask = (gate_action == 0).unsqueeze(1).expand_as(mask)
            mask = mask | gate_mask
        
        embeddings = self.embedder(self._init_embed(input), edges, dist=dist_matrix)
        _, pi = self._inner(input, edges, embeddings, cost_weights=None, dist_matrix=dist_matrix, mask=mask)
        ucost, cost_dict, _ = self.problem.get_costs(input, pi, cw_dict=None)
        src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
        dst_mask = dst_vertices != 0
        pair_mask = (src_vertices != 0) & (dst_mask)
        last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
        travelled = dist_matrix[src_vertices, dst_vertices] * pair_mask.float()
        ret_dict = {}
        ret_dict['overflows'] = cost_dict['overflows']
        ret_dict['kg'] = cost_dict['waste'] * 100
        ret_dict['waste'] = cost_dict['waste']
        if dist_matrix.dim() == 2:
            ret_dict['km'] = travelled.sum(dim=1) + dist_matrix[0, src_vertices[:, 0]] + \
                dist_matrix[dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0]
        else:
            ret_dict['km'] = travelled.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]] + \
                dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0]
        attention_weights = torch.tensor([])
        if hook_data['weights']:
            attention_weights = torch.stack(hook_data['weights'])
        return ucost, ret_dict, {'attention_weights': attention_weights, 'graph_masks': hook_data['masks']}
    
    def compute_simulator_day(self, input, graph, distC, profit_vars=None, run_tsp=False, hrl_manager=None, 
                              waste_history=None, threshold=0.5, mask_threshold=0.5, two_opt_max_iter=0):
        edges, dist_matrix = graph
        hook_data = add_attention_hooks(self.embedder)
        
        mask = None
        if hrl_manager is not None and waste_history is not None:
            # Static: Customer Locations (Batch, N, 2)
            # Should be shape (1, N, 2) if single instance
            if input['loc'].dim() == 2:
                static_feat = input['loc'].unsqueeze(0)
            else:
                static_feat = input['loc']
            
            # Dynamic: Waste History (Batch, N, History)
            # waste_history likely (N, History) if single instance
            if waste_history.dim() == 2:
                dynamic_feat = waste_history.unsqueeze(0)
            else:
                dynamic_feat = waste_history
            
            # Ensure shapes align:
            # static_feat: (Batch, N, 2)
            # dynamic_feat: (Batch, N, History)
            
            # If dynamic_feat came from (Days, N), it is now (1, Days, N)
            # We check if dim 1 matches N (from static). If not, and dim 2 does, we permute.
            N = static_feat.size(1)
            if dynamic_feat.size(1) != N and dynamic_feat.size(2) == N:
                dynamic_feat = dynamic_feat.permute(0, 2, 1) # (B, N, Days)

            # Feature Normalization Check
            # History should be [0, 1]. If we see values >> 1 (e.g. 0-100), normalize.
            if dynamic_feat.max() > 2.0: # Safe threshold, assuming valid fills roughly <= 100% (1.0)
                dynamic_feat = dynamic_feat / 100.0
            
            # Truncate to Window Size if passed full history (e.g. from Notebook)
            # dynamic_feat: (Batch, N, History)
            if hasattr(hrl_manager, 'input_dim_dynamic'):
                window_size = hrl_manager.input_dim_dynamic
                if dynamic_feat.size(2) > window_size:
                    # Take last 'window_size' steps
                    dynamic_feat = dynamic_feat[:, :, -window_size:]
            
            # Compute Global Features
            # dynamic_feat: (Batch, N, History)
            # current_waste: (Batch, N) - assuming last step is current
            current_waste = dynamic_feat[:, :, -1] 

            # 1. Critical Ratio Now
            critical_mask = (current_waste > hrl_manager.critical_threshold).float()
            critical_ratio = critical_mask.mean(dim=1, keepdim=True) # (B, 1)
            
            # 2. Max Current Waste
            max_current_waste = current_waste.max(dim=1, keepdim=True)[0] # (B, 1)
            
            # Combine: (B, 2)
            global_features = torch.cat([
                critical_ratio, 
                max_current_waste, 
            ], dim=1) # (B, 2)

            mask_action, gate_action, _ = hrl_manager.select_action(static_feat, dynamic_feat, global_features, deterministic=True, threshold=threshold, mask_threshold=mask_threshold)
            

            # If Gate is closed (0), return empty immediately
            if gate_action.item() == 0:
                for handle in hook_data['handles']:
                    handle.remove()
                return [0], 0, {'attention_weights': torch.tensor([]), 'graph_masks': hook_data['masks']}
            
            # Construct Mask
            mask = (mask_action == 0)
            # Clear hooks after manager decision as we only want worker's attention weights
            hook_data['weights'].clear()
            hook_data['masks'].clear()
        
        else:
            embeddings = self.embedder(self._init_embed(input), edges)

        # Ensure dist_matrix is expanded to batch size if present
        if dist_matrix is not None:
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
            if dist_matrix.size(0) == 1 and embeddings.size(0) > 1:
                dist_matrix = dist_matrix.expand(embeddings.size(0), -1, -1)

        _, pi = self._inner(input, edges, embeddings, None, dist_matrix, profit_vars=None, mask=mask)
        if run_tsp:
            try:
                pi_nodes = pi[pi != 0].cpu().numpy()
                if len(pi_nodes) > 0:
                    route = find_route(np.round(dist_matrix.cpu().numpy() * 1000), pi_nodes)
                    # distC might be a CUDA tensor, ensure it is numpy for get_route_cost
                    distC_np = distC.cpu().numpy() if torch.is_tensor(distC) else distC

                    # Respect Capacity in Simulator Evaluation
                    if profit_vars is not None and 'vehicle_capacity' in profit_vars:
                        raw_wastes = input['waste'].squeeze(0).cpu().numpy()
                        route = get_multi_tour(route, raw_wastes, profit_vars['vehicle_capacity'], dist_matrix.cpu().numpy())
                    
                    cost = get_route_cost(distC_np * 100, route)
                else:
                    route = [0]
                    cost = 0
            except:
                route = []
                cost = 0
        else:
            route = torch.cat((torch.tensor([0]).to(pi.device), pi.squeeze(0)))

            # Apply 2-opt refinement (GPU accelerated)
            if two_opt_max_iter > 0:
                route = local_search_2opt_vectorized(route, distC, two_opt_max_iter)
            
            cost = get_route_cost(distC * 100, route)
        
        for handle in hook_data['handles']:
            handle.remove()

        attention_weights = torch.tensor([])
        if hook_data['weights']:
            attention_weights = torch.stack(hook_data['weights'])
        route_list = route if isinstance(route, list) else route.cpu().numpy().tolist()
        return route_list, cost, {'attention_weights': attention_weights, 'graph_masks': hook_data['masks']}

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input, edges):
        embeddings = self.embedder(self._init_embed(input), edges)
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask, return_entropy=False):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
            # Check for -inf
        # Check for -inf
        # assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        ll = log_p.sum(1)

        if return_entropy:
            # _log_p: (Batch, SeqLen, GraphSize)
            probs = _log_p.exp()
            # Entropy: -sum(p * log_p)
            # Mask out -inf values for log_p calculation
            log_p_safe = _log_p.clamp(min=-20)
            entropy = -(probs * log_p_safe).sum(-1).mean(1) # Average over sequence
            return ll, entropy

        return ll

    def _init_embed(self, nodes, temporal_features=True):
        if self.is_vrpp or self.is_wc:
            if temporal_features:
                features = tuple(['waste'] + ["fill{}".format(day) for day in range(1, self.temporal_horizon + 1)])
            else:
                features = ('waste',)
            return torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(nodes['depot'])[:, None, :],
                    self.init_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        nodes['loc'],  # [batch_size, graph_size, 2]
                        *(nodes[feat][:, :, None] for feat in features)  # [batch_size, graph_size]
                    ), -1))  # [batch_size, graph_size, node_dim]
                ),
                1
            )
        assert False, "Unsupported problem"

    def _inner(self, nodes, edges, embeddings, cost_weights, dist_matrix, profit_vars=None, mask=None, expert_pi=None):
        outputs = []
        sequences = []
        state = self.problem.make_state(nodes, edges, cost_weights, dist_matrix, profit_vars=profit_vars, hrl_mask=mask)
        
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Perform decoding steps
        i = 0
        batch_size = state.ids.size(0)
        while not (self.shrink_size is None and (state.all_finished() if expert_pi is None else i >= expert_pi.size(1))):
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, step_mask = self._get_log_p(fixed, state)

            # Apply HRL Mask if provided
            if mask is not None:
                # Mask from HRL is typically (Batch, N) for customers
                # step_mask is likely (Batch, 1, N+1) including depot
                if mask.size(1) == step_mask.size(2) - 1:
                    # Pad mask with False for depot at index 0
                    # (Batch, N) -> (Batch, N+1)
                    depot_mask = torch.zeros((mask.size(0), 1), dtype=torch.bool, device=mask.device)
                    mask_padded = torch.cat((depot_mask, mask), dim=1)
                else:
                    mask_padded = mask

                if step_mask.dim() == 3:
                    # (Batch, 1, N)
                    step_mask = step_mask | mask_padded.unsqueeze(1)
                else:
                    step_mask = step_mask | mask_padded
                
                # Mask logits (log_p) for HRL-masked nodes
                # Ensure they have -inf log_prob so they have 0 probability
                # log_p is (Batch, 1, GraphSize)
                # Expand step_mask to broadcast if necessary
                if step_mask.dim() == 2:
                    current_mask = step_mask.unsqueeze(1)
                else:
                    current_mask = step_mask
                    
                log_p = log_p.masked_fill(current_mask, -math.inf)
                     
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            if expert_pi is not None:
                selected = expert_pi[:, i]
            else:
                selected = self._select_node(log_p.exp()[:, 0, :], step_mask[:, 0, :])  # Squeeze out steps dimension
                
                # POMO: For the first customer choice, we force the selection
                if self.pomo_size > 0 and i == 0:
                    # We assume the first choice after depot is a customer.
                    # In POMO, we want trajectory j for instance b to start with customer j+1
                    # selected is (B * pomo_size)
                    # We need to find B
                    current_batch_size = selected.size(0)
                    B = current_batch_size // self.pomo_size
                    # Forced actions: [1, 2, ..., pomo_size] repeated B times
                    forced_selected = torch.arange(1, self.pomo_size + 1, device=selected.device).repeat(B)
                    selected = forced_selected

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, cost_weights=None, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :param cost_weights: dictionary with weights for each term of the cost function
        :return:
        """
        edges = input.pop('edges') if 'edges' in input.keys() else None
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input[:3], cost_weights, input[3]),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi, cost_weights)[:2],  # Don't need embeddings as input to get_costs
            (input, edges, self.embedder(self._init_embed(input), edges), input.get('dist_matrix', None)),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            print(f"DEBUG: decode_type is {repr(self.decode_type)}")
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if self.aggregation_graph == "avg":
            graph_embed = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        else:  # Default: disable graph embedding
            graph_embed = embeddings.sum(1) * 0.0

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        mask = state.get_mask()

        graph_mask = None
        if self.mask_graph:
            # Compute the graph mask, for masking next action based on graph structure 
            graph_mask = state.get_edges_mask()

        # Spatial Bias: Penalize nodes that are physically far from the current location
        dist_bias = None
        if self.spatial_bias and state.dist_matrix is not None:
            dist_matrix = state.dist_matrix 
            current_node = state.get_current_node() # (B*P, 1)
            
            # dist_matrix is already (B*P, N, N) from forward() expansion
            # index shape: (Batch * POMO, 1, N)
            index = current_node.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
            dist_bias = -self.spatial_bias_scale * dist_matrix.gather(1, index).detach()
            # Shape: (Batch * POMO, 1, GraphSize)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask, dist_bias=dist_bias)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_vrpp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    state.get_current_profit()[:, :, None]
                ),
                -1
            )
        elif self.is_wc:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    state.get_current_efficiency()[:, :, None],
                    state.get_remaining_overflows()[:, :, None]
                ),
                -1
            )
        else:
            assert False, "Unsupported problem"

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None, dist_bias=None):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if dist_bias is not None:
            # compatibility: (n_heads, batch, num_steps, 1, graph_size)
            # dist_bias: (Batch, 1, GraphSize)
            # We must explicitly align Batch to Batch
            compatibility += dist_bias.unsqueeze(0).unsqueeze(2)
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
            if self.mask_graph:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if dist_bias is not None:
            # logits: (Batch, num_steps, GraphSize)
            # dist_bias: (Batch, 1, GraphSize)
            logits += dist_bias.unsqueeze(1) if logits.size(1) > 1 else dist_bias

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -math.inf
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        if self.is_wc and self.allow_partial:
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )