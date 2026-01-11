"""
Neural Agent wrapper for deep reinforcement learning models.

This module provides the interface between the simulation environment and
neural routing models (Attention Models, GCN-based models, etc.).

The NeuralAgent handles:
- Batched inference for simulation evaluation
- Hierarchical Reinforcement Learning (HRL) integration
- Attention weight extraction for interpretability
- Post-processing (TSP refinement, 2-opt optimization)
- Masking and gating decisions from manager networks

Key Components:
- compute_batch_sim: Batch processing for training/evaluation
- compute_simulator_day: Single-day routing for simulation
- HRL support: Manager-worker architecture with gating and masking

The agent serves as an adapter between the problem-agnostic neural models
and the domain-specific waste collection simulator.
"""
import torch
import numpy as np

from logic.src.utils.functions import add_attention_hooks
from logic.src.policies import find_route, get_route_cost, get_multi_tour
from logic.src.pipeline.reinforcement_learning.core.post_processing import local_search_2opt_vectorized


class NeuralAgent:
    """
    Agent interface between simulator/environment and neural routing models.

    Handles model inference, hierarchical decision-making (HRL), and
    post-processing for waste collection routing.

    Attributes:
        model: The neural routing model (AttentionModel, etc.)
        problem: Problem instance (VRPP, WCVRP, etc.) for cost calculation
    """
    def __init__(self, model):
        """
        Initializes the NeuralAgent.

        Args:
            model: The neural routing model (e.g., AttentionModel).
        """
        self.model = model
        self.problem = model.problem

    def compute_batch_sim(self, input, dist_matrix, hrl_manager=None, waste_history=None, threshold=0.5, mask_threshold=0.5):
        """
        Compute simulation step for a batch of problem instances.

        Used during training and batch evaluation. Optionally integrates with
        HRL manager for gating (decide whether to route) and masking (which bins to visit).

        Args:
            input (dict): Batch of problem data with 'loc', 'waste', etc.
            dist_matrix (torch.Tensor): Distance matrix (B x N x N) or (N x N)
            hrl_manager (optional): HRL manager network for gating decisions
            waste_history (torch.Tensor, optional): Historical waste levels (B x N x T)
            threshold (float): Gating probability threshold. Default: 0.5
            mask_threshold (float): Masking probability threshold. Default: 0.5

        Returns:
            Tuple[torch.Tensor, dict, dict]: Costs, result metrics, and attention data
                - ucost: Unweighted costs tensor
                - ret_dict: Dictionary with 'overflows', 'kg', 'waste', 'km'
                - output_dict: Dictionary with 'attention_weights', 'graph_masks'
        """
        hook_data = add_attention_hooks(self.model.embedder)
        
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
        
        # Use model forward pass
        # model call returns: cost, log_likelihood, c_dict, pi, entropy
        # We need ucost, ret_dict, etc.
        # We execute with return_pi=True
        cost, _, cost_dict, pi, _ = self.model(
            input, cost_weights=None, return_pi=True, pad=False, mask=mask
        )
        
        # Calculate ucost (unweighted cost)?
        # cost_dict comes from model.problem.get_costs which returns unweighted components
        # cost returned by model is scalar (weighted sum if cost_weights provided, or sum or something)
        # We can re-calculate ucost if needed or trust cost_dict.
        # Original code: ucost, cost_dict, _ = self.problem.get_costs(input, pi, cw_dict=None)
        
        # If we use model output, we rely on model using get_costs correctly.
        # If we want exact parity, let's call problem.get_costs ourselves if we have 'pi'.
        # self.model(..., return_pi=True) returns pi.
        
        ucost, cost_dict, _ = self.problem.get_costs(input, pi, cw_dict=None)
        
        src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
        dst_mask = dst_vertices != 0
        pair_mask = (src_vertices != 0) & (dst_mask)
        # To avoid index error on dst_vertices if size is 0 or something?
        if dst_vertices.size(1) > 0:
            last_dst = torch.max(dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device), dim=1).indices
        else:
            last_dst = torch.zeros(dst_vertices.size(0), dtype=torch.long, device=dst_vertices.device)

        travelled = dist_matrix[src_vertices, dst_vertices] * pair_mask.float()
        
        ret_dict = {}
        ret_dict['overflows'] = cost_dict['overflows']
        ret_dict['kg'] = cost_dict['waste'] * 100
        ret_dict['waste'] = cost_dict['waste']
        
        if dist_matrix.dim() == 2:
            # Assuming batch dim 0 is implied? 
            # dist_matrix: (N, N). src_vertices: (B, L).
            # This logic mimics original code but might need careful checking if dist_matrix is (B, N, N)
            ret_dict['km'] = travelled.sum(dim=1) + dist_matrix[0, src_vertices[:, 0]] + \
                dist_matrix[dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0]
        else:
             ret_dict['km'] = travelled.sum(dim=1) + dist_matrix[0, 0, src_vertices[:, 0]] + \
                dist_matrix[0, dst_vertices[torch.arange(dst_vertices.size(0), device=dst_vertices.device), last_dst], 0]

        attention_weights = torch.tensor([])
        if hook_data['weights']:
            attention_weights = torch.stack(hook_data['weights'])

        graph_masks = torch.tensor([])
        if hook_data['masks']:
            graph_masks = torch.stack(hook_data['masks'])
            
        return ucost, ret_dict, {'attention_weights': attention_weights, 'graph_masks': graph_masks}

    def compute_simulator_day(self, input, graph, distC, profit_vars=None, run_tsp=False, hrl_manager=None,
                          waste_history=None, threshold=0.5, mask_threshold=0.5, two_opt_max_iter=0):
        """
        Execute neural routing policy for a single simulation day.

        Main entry point for simulator integration. Generates a collection route
        for the current day using the trained neural model.

        HRL Integration:
        - If hrl_manager provided: Manager decides whether to route (gate)
          and which bins to mask (selective collection)
        - If gate closed: Returns empty route [0]
        - If gate open: Worker model constructs route respecting mask

        Post-processing:
        - run_tsp=True: Refine route using fast_tsp
        - two_opt_max_iter>0: Apply 2-opt local search (GPU accelerated)

        Args:
            input (dict): Problem instance with 'loc', 'waste', etc.
            graph (tuple): (edges, dist_matrix) for the model
            distC (torch.Tensor or np.ndarray): Distance matrix for cost calculation
            profit_vars (dict, optional): VRPP parameters (vehicle_capacity, R, C, etc.)
            run_tsp (bool): Whether to refine route with TSP solver. Default: False
            hrl_manager (optional): HRL manager network
            waste_history (torch.Tensor, optional): Historical bin levels (Days x N) or (N x Days)
            threshold (float): Gating probability threshold. Default: 0.5
            mask_threshold (float): Masking probability threshold. Default: 0.5
            two_opt_max_iter (int): 2-opt iterations. 0 disables. Default: 0

        Returns:
            Tuple[List[int], float, dict]: Route, cost, and attention data
                - route: List of node IDs [0, node1, ..., 0]
                - cost: Total tour distance * 100
                - output_dict: {'attention_weights', 'graph_masks'}
        """
        edges, dist_matrix = graph
        hook_data = add_attention_hooks(self.model.embedder)
        
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
        
        # Ensure dist_matrix is expanded to batch size if present
        # Note: model forward typically handles this, but explicit expansion helps with some operations
        # However, model forward expects dist_matrix in input args or we rely on model doing it.
        # The 'input' dict likely contains 'dist' if passed.
        # But 'graph' arg passed dist_matrix separately.
        
        # We need to make sure 'input' has 'dist' or 'edges' if model expects it.
        # Original code used 'edges' from graph arg.
        
        # Calling model forward
        # AttentionModel forward signature: forward(self, input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None, **kwargs)
        # It expects 'edges' and 'dist' in 'input' dict OR we rely on model finding it.
        # Looking at AM.forward:
        # edges = input.get('edges', None)
        # dist_matrix = input.get('dist', None)
        
        # The 'input' passed to compute_simulator_day might NOT have edges/dist if they were passed separately in 'graph'.
        # We should update 'input' or pass them?
        # AM.forward uses `input.get` so we should ensure they are in input.
        
        # Safe copy?
        # input is a dict.
        
        # Original code called `self._inner(input, edges, embeddings, ... dist_matrix=dist_matrix ...)`
        # It didn't call forward.
        
        # If we use `model.forward`, we must ensure input dict has what it needs.
        
        input_for_model = input.copy() # Shallow copy
        if 'edges' not in input_for_model and edges is not None:
             input_for_model['edges'] = edges
        if 'dist' not in input_for_model and dist_matrix is not None:
             input_for_model['dist'] = dist_matrix

        # Expand dist_matrix for post-processing if needed (like in original code)
        if dist_matrix is not None:
            if dist_matrix.dim() == 2:
                dist_matrix = dist_matrix.unsqueeze(0)
            # We don't have embeddings size yet to expand against batch.
            # But model forward will handle batching.
        
        _, _, _, pi, _ = self.model(input_for_model, return_pi=True, mask=mask)
        
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
