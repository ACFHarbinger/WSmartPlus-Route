import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TemporalAttentionModel


class HierarchicalTemporalAttentionModel(TemporalAttentionModel):
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
                 temporal_horizon=5,
                 predictor_layers=2,
                 num_clusters=5,
                 clustering_dim=64,
                 cluster_method='kmeans',
                 use_overflow_priority=True,
                 route_cluster_separately=False):
        
        super(HierarchicalTemporalAttentionModel, self).__init__(
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
            shrink_size,
            temporal_horizon,
            predictor_layers
        )
        
        from models.modules import ActivationFunction
        
        self.num_clusters = num_clusters
        self.clustering_dim = clustering_dim
        self.cluster_method = cluster_method
        self.use_overflow_priority = use_overflow_priority
        self.route_cluster_separately = route_cluster_separately
        
        # Clustering network
        self.clustering_network = nn.Sequential(
            nn.Linear(embedding_dim, clustering_dim),
            ActivationFunction(activation),
            nn.Linear(clustering_dim, clustering_dim)
        )
        
        # Cluster centroids (learnable)
        self.cluster_centroids = nn.Parameter(torch.Tensor(num_clusters, clustering_dim))
        nn.init.xavier_uniform_(self.cluster_centroids)
        
        # Cluster embedding
        self.cluster_embedding = nn.Linear(num_clusters, embedding_dim)
        
        # Inter-cluster routing encoder (learns relationships between clusters)
        self.inter_cluster_encoder = encoder_class(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
            learn_affine=learn_affine,
            activation=activation
        )
        
        # Overflow priority module
        if use_overflow_priority:
            self.overflow_priority = nn.Sequential(
                nn.Linear(1, hidden_dim),
                ActivationFunction(activation),
                nn.Linear(hidden_dim, 1)
            )
    
    def _assign_clusters(self, node_embeddings, node_features=None):
        """Assign nodes to clusters based on embeddings and features"""
        batch_size, graph_size, _ = node_embeddings.size()
        
        # Get clustering features
        clustering_features = self.clustering_network(node_embeddings)
        
        # For WCVRP/VRPP, incorporate waste and max_waste into clustering decision
        if (self.is_vrpp or self.is_wc) and node_features is not None:
            # Extract waste and max_waste
            waste = node_features.get('waste', None)
            max_waste = node_features.get('max_waste', None)
            
            if waste is not None and max_waste is not None:
                # Calculate fill level as waste / max_waste
                # Handle depot separately (typically at index 0)
                if waste.size(1) != graph_size:  # If waste doesn't include depot
                    fill_level = waste / max_waste.unsqueeze(1)
                    # For depot, set fill_level to 0
                    depot_fill = torch.zeros((batch_size, 1, 1), device=fill_level.device)
                    fill_level = torch.cat([depot_fill, fill_level], dim=1)
                else:
                    # If waste includes depot (at index 0), set fill level for it to 0
                    fill_level = waste.clone()
                    fill_level[:, 0] = 0
                    fill_level = fill_level / max_waste.unsqueeze(1)
                
                # Calculate overflow risk (how close to overflowing)
                overflow_risk = (fill_level / max_waste.unsqueeze(1)).clamp(0, 1)
                
                if self.use_overflow_priority:
                    # Get priority score from overflow risk
                    priority = self.overflow_priority(overflow_risk.unsqueeze(-1)).squeeze(-1)
                    
                    # Apply softmax to get priority weights
                    priority_weights = F.softmax(priority, dim=1).unsqueeze(-1)
                    
                    # Weight clustering features by overflow priority
                    clustering_features = clustering_features * priority_weights
        
        # Calculate distance to each centroid
        expanded_centroids = self.cluster_centroids.unsqueeze(0).unsqueeze(0)
        expanded_features = clustering_features.unsqueeze(2)
        
        # Calculate Euclidean distance
        distances = torch.sum((expanded_features - expanded_centroids) ** 2, dim=-1)
        
        # Get cluster assignments (hard assignment)
        cluster_assignments = torch.argmin(distances, dim=2)
        
        # Create one-hot encoding of cluster assignments
        cluster_one_hot = F.one_hot(cluster_assignments, num_classes=self.num_clusters).float()
        
        return cluster_assignments, cluster_one_hot
    
    def _init_embed(self, nodes):
        # Get temporal embeddings from parent class
        base_embeddings = super()._init_embed(nodes)
        
        # Skip clustering for small problems
        if (self.is_vrpp or self.is_wc) and base_embeddings.size(1) <= self.num_clusters + 3:
            return base_embeddings
        
        # Assign nodes to clusters
        _, cluster_one_hot = self._assign_clusters(base_embeddings, nodes)
        
        # Get cluster embeddings
        cluster_embeddings = self.cluster_embedding(cluster_one_hot)
        
        # Combine with base embeddings
        combined_embeddings = base_embeddings + cluster_embeddings
        
        return combined_embeddings

    def _calc_node_critical_score(self, input):
        """Calculate critical score for nodes based on overflow risk"""
        if not (self.is_vrpp or self.is_wc) or 'waste' not in input:
            return None
            
        # Get waste and max_waste
        waste = input['waste']
        max_waste = input['max_waste']
        
        # Calculate fill level and overflow risk
        overflow_risk = ((waste / max_waste.unsqueeze(1)) - 0.7).clamp(0, 1)
        
        return overflow_risk
    
    def _hierarchical_route(self, input, clustered_nodes, cost_weights=None):
        """Route within clusters and then between clusters"""
        batch_size = input['loc'].size(0)
        graph_size = input['loc'].size(1)
        
        # If graph is small, just use normal routing
        if graph_size <= self.num_clusters + 3:
            return super().forward(input, cost_weights, return_pi=True)
        
        # Get cluster assignments
        cluster_assignments, _ = self._assign_clusters(clustered_nodes, input)
        
        # Compute cluster centers
        cluster_centers = []
        for c in range(self.num_clusters):
            # Get mask for this cluster
            cluster_mask = (cluster_assignments == c)
            
            # Skip empty clusters
            if not cluster_mask.any():
                # Use a default position (e.g., depot)
                cluster_centers.append(input['depot'].unsqueeze(1))
                continue
                
            # Get locations for this cluster
            if self.is_vrp or self.is_orienteering or self.is_pctsp or self.is_wc:
                # Include depot in locations
                locs = torch.cat([input['depot'].unsqueeze(1), input['loc']], dim=1)
            else:
                locs = input['loc']
                
            # Mask out non-cluster nodes and compute center
            masked_locs = locs * cluster_mask.unsqueeze(-1).float()
            center = masked_locs.sum(dim=1) / cluster_mask.sum(dim=1, keepdim=True).clamp(min=1)
            cluster_centers.append(center.unsqueeze(1))
        
        # Concatenate centers
        cluster_centers = torch.cat(cluster_centers, dim=1)
        
        # Create inter-cluster problem
        inter_cluster_problem = {
            'loc': cluster_centers,
            'depot': input['depot']
        }
        
        # Get inter-cluster route
        _, inter_cluster_pi = super().forward(inter_cluster_problem, cost_weights, return_pi=True)
        
        if self.route_cluster_separately:
            # Route each cluster separately and combine
            final_routes = []
            for batch_idx in range(batch_size):
                batch_route = [0]  # Start at depot
                
                for cluster_idx in inter_cluster_pi[batch_idx]:
                    if cluster_idx == 0:  # Skip depot visits in inter-cluster route
                        continue
                        
                    # Get nodes in this cluster
                    cluster_nodes = torch.nonzero(cluster_assignments[batch_idx] == (cluster_idx - 1)).squeeze(-1)
                    
                    if cluster_nodes.size(0) == 0:
                        continue
                        
                    # Create intra-cluster problem
                    intra_cluster_problem = {
                        'loc': input['loc'][batch_idx:batch_idx+1, cluster_nodes],
                        'depot': input['depot'][batch_idx:batch_idx+1]
                    }
                    
                    # Route within cluster
                    _, intra_cluster_pi = super().forward(
                        intra_cluster_problem, 
                        cost_weights, 
                        return_pi=True
                    )
                    
                    # Map back to original indices
                    for node_idx in intra_cluster_pi[0]:
                        if node_idx > 0:  # Skip depot in intra-cluster route
                            orig_idx = cluster_nodes[node_idx-1].item() + 1  # +1 to account for depot
                            batch_route.append(orig_idx)
                
                batch_route.append(0)  # End at depot
                final_routes.append(torch.tensor(batch_route, device=inter_cluster_pi.device))
            
            # Pad routes to same length
            max_len = max(route.size(0) for route in final_routes)
            padded_routes = []
            for route in final_routes:
                padding = torch.zeros(max_len - route.size(0), dtype=route.dtype, device=route.device)
                padded_routes.append(torch.cat([route, padding]))
            
            final_pi = torch.stack(padded_routes)
            
        else:
            # Use normal routing with critical nodes prioritized
            critical_score = self._calc_node_critical_score(input)
            
            if critical_score is not None:
                # Modify cost by adding critical score as priority
                if 'waste' in input:
                    input = input.copy()
                    input['waste'] = input['waste'] * (1 + critical_score)
            
            # Get route normally
            _, final_pi = super().forward(input, cost_weights, return_pi=True)
        
        return None, final_pi
    
    def forward(self, input, cost_weights=None, return_pi=False, pad=False):
        embeddings = self._init_embed(input)
        
        if self.route_cluster_separately or (self.is_wc and input['loc'].size(1) > self.num_clusters + 3):
            # Use hierarchical routing for larger problems
            cost, pi = self._hierarchical_route(input, embeddings, cost_weights)
            if return_pi:
                return cost, pi
            return cost
        else:
            # Use normal routing for smaller problems
            return super().forward(input, cost_weights, return_pi, pad)