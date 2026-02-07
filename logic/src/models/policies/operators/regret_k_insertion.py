import torch
from torch import Tensor


def vectorized_regret_k_insertion(tours: Tensor, removed_nodes: Tensor, dist_matrix: Tensor, k: int = 2) -> Tensor:
    """
    Vectorized Regret-K insertion.
    """
    B, N_curr = tours.shape
    B_rem, N_rem = removed_nodes.shape
    device = tours.device

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)

    # Set of pending nodes (mask)
    pending_mask = torch.ones((B, N_rem), dtype=torch.bool, device=device)

    # We iterate until all inserted (N_rem times)
    for _ in range(N_rem):
        B, N_curr = tours.shape
        # 1. Compute insertion costs for all pending nodes at all positions
        tours_prev = tours.unsqueeze(1)  # (B, 1, N_curr)
        tours_next = torch.roll(tours, -1, dims=1).unsqueeze(1)  # (B, 1, N_curr)

        nodes_exp = removed_nodes.unsqueeze(2)  # (B, N_rem, 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1)

        tp = tours_prev.expand(-1, N_rem, -1)  # (B, N_rem, N_curr)
        tn = tours_next.expand(-1, N_rem, -1)
        nd = nodes_exp.expand(-1, -1, N_curr)

        d_pn = dist_matrix[batch_idx, tp, nd]  # (B, N_rem, N_curr)
        d_nn = dist_matrix[batch_idx, nd, tn]
        d_pn_exist = dist_matrix[batch_idx, tp, tn]

        costs = d_pn + d_nn - d_pn_exist  # (B, N_rem, N_curr)

        # 2. Find Best and K-th Best for each node
        topk_vals, topk_indices = torch.topk(costs, k=k, dim=2, largest=False)

        best_costs = topk_vals[:, :, 0]  # (B, N_rem)
        kth_costs = topk_vals[:, :, -1]  # (B, N_rem)

        regret = kth_costs - best_costs  # (B, N_rem)

        # Mask out already inserted nodes
        regret[~pending_mask] = -float("inf")

        # 3. Select node with Max Regret
        max_regret_val, node_idx_in_rem = torch.max(regret, dim=1)  # (B,)

        # 4. Insert that node at its best position
        node_to_insert = torch.gather(removed_nodes, 1, node_idx_in_rem.unsqueeze(1))  # (B, 1)
        all_best_pos = topk_indices[:, :, 0]
        insert_pos = torch.gather(all_best_pos, 1, node_idx_in_rem.unsqueeze(1))  # (B, 1)

        if (~pending_mask).all():
            break

        # 5. Apply Insertion
        N_curr = tours.shape[1]
        N_new = N_curr + 1
        new_tours = torch.zeros((B, N_new), dtype=tours.dtype, device=device)

        seq = torch.arange(N_curr, device=device).unsqueeze(0).expand(B, N_curr)

        mask_left = seq <= insert_pos
        mask_right = ~mask_left

        write_indices = torch.zeros_like(seq)
        write_indices[mask_left] = seq[mask_left]
        write_indices[mask_right] = seq[mask_right] + 1

        new_tours.scatter_(1, write_indices, tours)
        new_tours.scatter_(1, insert_pos + 1, node_to_insert)

        tours = new_tours
        pending_mask.scatter_(1, node_idx_in_rem.unsqueeze(1), False)

    return tours
