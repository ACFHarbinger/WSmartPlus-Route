import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


class EfficiencyOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(EfficiencyOptimizer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        # Add residual connection so the model learns adjustments
        return x + self.network(x)


def calculate_efficiency(routes, waste, distance_matrix):
    """Calculate kg/km efficiency and overflow metrics for routes"""
    total_waste = 0
    total_distance = 0
    overflow_count = 0
    for route in routes:
        # Calculate total waste collected on this route
        route_waste = sum(waste[i] for i in route)
        total_waste += route_waste
        
        # Calculate distance traveled (including return to depot)
        route_distance = 0
        for i in range(len(route)-1):
            route_distance += distance_matrix[route[i], route[i+1]]
        route_distance += distance_matrix[route[-1], 0]  # Return to depot
        total_distance += route_distance
    efficiency = total_waste / max(1, total_distance)  # kg/km
    return efficiency, overflow_count


def post_processing_optimization(main_model, dataset, epochs=100, lr=0.001, efficiency_weight=0.8, overflow_weight=0.2):
    """
    Apply post-processing optimization to improve efficiency
    
    Args:
        main_model: Your trained main routing model
        distance_matrix: Matrix of distances between nodes
        epochs: Number of epochs for post-processing
        lr: Learning rate
        efficiency_weight: Weight for efficiency in loss function
        overflow_weight: Weight for overflow penalty
    """
    # Extract model outputs as starting point
    main_model.eval()
    
    # Determine input dimension based on model output
    with torch.no_grad():
        sample_output = main_model(dataset[0])
        input_dim = sample_output.size(-1)
    
    # Create post-processing model and train it
    post_processor = EfficiencyOptimizer(input_dim)
    optimizer = optim.Adam(post_processor.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for batch in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            with torch.no_grad():
                main_outputs = main_model(batch)
            
            # Perform post-processing and compute metrics
            adjusted_outputs = post_processor(main_outputs)
            routes = decode_routes(adjusted_outputs)
            efficiency, overflows = calculate_efficiency(routes, batch['waste'], batch['dist'])
            
            # Custom loss that prioritizes efficiency with some overflow tolerance
            loss = -(efficiency_weight * efficiency) + (overflow_weight * overflows)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return post_processor


def decode_routes(model_outputs):
    """
    Convert model outputs to concrete routes
    This function depends on your specific routing representation
    """
    routes = []
    for output in model_outputs:
        route = output[output != 0]
        routes.append([0] + route.cpu().numpy().tolist() + [0])
    return routes


def apply_post_processing(main_model, post_processor, dataset):
    """Apply the post-processor to new instances"""
    main_model.eval()
    post_processor.eval()
    
    results = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            main_outputs = main_model(batch)
            optimized_outputs = post_processor(main_outputs)
            results.append(optimized_outputs)
    return torch.cat(results, dim=0)


def local_search_2opt_vectorized(tours, distance_matrix, max_iterations=200):
    """
    Vectorized 2-opt local search across a batch of tours using PyTorch.
    Optimized to perform edge swaps for all batch instances in parallel.
    """
    device = distance_matrix.device
    
    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)
    
    # Handle distance_matrix expansion
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0)
    
    B, N = tours.shape
    if N < 4:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)
        
    batch_indices = torch.arange(B, device=device).view(B, 1)
    
    for _ in range(max_iterations):
        # Generate indices for all possible edge swaps (i, j)
        indices = torch.arange(N, device=device)
        i = indices[1:-2]
        j = indices[2:-1]
        
        I, J = torch.meshgrid(i, j, indexing='ij')
        mask = J > I
        if not mask.any():
            break
            
        I_vals = I[mask]
        J_vals = J[mask]
        K = I_vals.size(0)
        
        # Tour nodes at relevant indices: (B, K)
        t_prev_i = tours[:, I_vals - 1]
        t_curr_i = tours[:, I_vals]
        t_curr_j = tours[:, J_vals]
        t_next_j = tours[:, J_vals + 1]
        
        # Gain calculation: (B, K)
        # Use advanced indexing for batch
        b_idx_exp = batch_indices.expand(B, K)
        d_curr = distance_matrix[b_idx_exp, t_prev_i, t_curr_i] + distance_matrix[b_idx_exp, t_curr_j, t_next_j]
        d_next = distance_matrix[b_idx_exp, t_prev_i, t_curr_j] + distance_matrix[b_idx_exp, t_curr_i, t_next_j]
        gains = d_curr - d_next
        
        # Find best gain for each instance in the batch
        best_gain, best_idx = torch.max(gains, dim=1)
        
        # Determine which instances actually improved
        improved = best_gain > 1e-5
        if not improved.any():
            break
            
        # Parallel segment reversal
        # Construct transform indices (B, N)
        target_i = I_vals[best_idx]
        target_j = J_vals[best_idx]
        
        k = torch.arange(N, device=device).view(1, N).expand(B, N)
        idx_map = torch.arange(N, device=device).view(1, N).expand(B, N).clone()
        
        # For instances that improved, reverse the [target_i, target_j] range
        # reversal_mask: (B, N)
        reversal_range_mask = (k >= target_i.view(B, 1)) & (k <= target_j.view(B, 1))
        reversal_mask = reversal_range_mask & improved.view(B, 1)
        
        # idx[b, k] = target_i[b] + target_j[b] - k
        rev_idx = target_i.view(B, 1) + target_j.view(B, 1) - k
        idx_map[reversal_mask] = rev_idx[reversal_mask]
        
        # Apply the best edge swap for all batch elements simultaneously
        tours = torch.gather(tours, 1, idx_map)
        
    return tours if is_batch else tours.squeeze(0)