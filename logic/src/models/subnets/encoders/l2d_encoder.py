import torch
import torch.nn as nn
from tensordict import TensorDict


class L2DGraphConv(nn.Module):
    """
    Graph Convolution Layer for L2D.

    Processes two types of edges:
    1. Precedence/Conjunctive (Job sequence)
    2. Machine/Disjunctive (Same machine)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, aggregation: str = "mean"):
        super().__init__()
        self.aggregation = aggregation

        # Message passing layers
        self.proj_prec = nn.Linear(embed_dim, embed_dim)
        self.proj_mach = nn.Linear(embed_dim, embed_dim)

        # Update layer
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  # concat(h, agg_prec, agg_mach)
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h: torch.Tensor, adj_prec: torch.Tensor, adj_mach: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings (batch, num_nodes, embed_dim)
            adj_prec: Precedence adjacency (batch, num_nodes, num_nodes) - normalized row-wise
            adj_mach: Machine adjacency (batch, num_nodes, num_nodes) - normalized row-wise
        """
        # Message passing
        # 1. Precedence neighbors
        # msg_prec = h @ W_prec
        # agg_prec = adj_prec @ msg_prec
        msg_prec = self.proj_prec(h)
        agg_prec = torch.bmm(adj_prec, msg_prec)

        # 2. Machine neighbors
        msg_mach = self.proj_mach(h)
        agg_mach = torch.bmm(adj_mach, msg_mach)

        # 3. Update
        combined = torch.cat([h, agg_prec, agg_mach], dim=-1)
        out = self.mlp(combined)

        return self.norm(h + out)  # Residual connection


class L2DEncoder(nn.Module):
    """
    L2D Encoder using Disjunctive Graph GNN.

    Encodes the JSSP state into embeddings for each operation.
    """

    def __init__(self, embed_dim: int = 128, num_layers: int = 3, feedforward_hidden: int = 512, **kwargs):
        super().__init__()

        # Initial features:
        # 1. Processing time (continuous)
        # 2. Machine ID (embedding? or just implied by graph?)
        # 3. Status (e.g., is next op, is finished? - usually dynamic L2D uses dynamic features)
        # For simplistic L2D, typically just duration embedding + graph structure is enough?
        # Let's use duration + machine embedding + simple status.

        self.embed_dim = embed_dim

        self.init_embed = nn.Linear(1, embed_dim)  # Project duration

        self.layers = nn.ModuleList([L2DGraphConv(embed_dim, feedforward_hidden) for _ in range(num_layers)])

    def forward(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            td: JSSP state.

        Returns:
            embeddings: (batch, num_jobs, embed_dim) - embeddings of the *next* operation for each job.
                        Wait, Policy needs to select a JOB. So we usually pool or select the embedding
                        of the *next operation* of that job.
                        Return all node embeddings (batch, total_ops, embed_dim) or (batch, J, M, dim)?
            init_embed: Dummy or full.

        """
        batch_size = td.batch_size[0] if td.batch_size else 1
        num_jobs = td["proc_time"].shape[1]
        num_machines = td["proc_time"].shape[2]
        total_ops = num_jobs * num_machines

        # Flatten state to nodes
        # proc_time: (B, J, M) -> (B, J*M, 1)
        durations = td["proc_time"].view(batch_size, total_ops, 1)
        machine_order = td["machine_order"].view(batch_size, total_ops)  # (B, J*M)

        # Initial embedding from duration
        h = self.init_embed(durations)  # (B, total_ops, embed_dim)

        # Build Adjacency Matrices
        # These are expensive to build every step fully dynamically.
        # Precedence is fixed per job structure.
        # Machine adjacency depends on machine_order (fixed per instance).

        # 1. Precedence Adjacency: (B, J*M, J*M)
        # Op (j, m) -> Op (j, m+1)
        # Ops are ordered in flat list by Job then Sequence usually?
        # Yes, proc_time is (J, M) implying sequence 0..M-1.
        # So node (j, i) connects to (j, i+1).
        # Indices: j*M + i -> j*M + i + 1.

        # Build base precedence matrix (sparse preferably, but for batch matmul dense is easier for now)
        # Optimization: Build once and expand
        adj_prec = torch.zeros(batch_size, total_ops, total_ops, device=td.device)

        # Mask for valid next ops (exclude last op of each job)
        # Indices: 0..M-2, M..2M-2, ...
        # i in 0..total_ops-1 where (i+1) % M != 0

        # We can construct indices
        rows = []
        cols = []
        for j in range(num_jobs):
            for i in range(num_machines - 1):
                curr_idx = j * num_machines + i
                next_idx = curr_idx + 1
                rows.append(curr_idx)
                cols.append(next_idx)
                # Also inverse edge? Or directed? disjunctive graph usually directed conjunctive.
                # GNN often undirected or bidirectional?
                # Let's assume directed forward.

        # Vectorized assignment
        # This is same for all batches if structure is fixed JxM
        # But wait, logic is (B, J, M).
        # We can create a base adjacency and expand.
        if not hasattr(self, "_base_adj_prec") or self._base_adj_prec.shape[0] != total_ops:
            base_adj = torch.zeros(total_ops, total_ops, device=td.device)
            base_rows = torch.tensor(rows, device=td.device)
            base_cols = torch.tensor(cols, device=td.device)
            base_adj[base_rows, base_cols] = 1.0
            self._base_adj_prec = base_adj

        adj_prec = self._base_adj_prec.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Machine Adjacency
        # Connect all ops that share the same machine.
        # machine_order: (B, total_ops) contains machine ID 0..M-1 for each op.
        # adj_mach[b, i, k] = 1 if machine_order[b, i] == machine_order[b, k] and i != k

        # (B, N, 1) == (B, 1, N) -> (B, N, N) boolean
        mach_match = machine_order.unsqueeze(2) == machine_order.unsqueeze(1)
        diag_mask = torch.eye(total_ops, device=td.device, dtype=torch.bool).unsqueeze(0)
        adj_mach = mach_match & (~diag_mask)
        adj_mach = adj_mach.float()

        # Normalize adjacency (row-wise sum)
        # A_prec is mostly 1s (except last ops), so mean is trivial.
        # A_mach depends on how many ops on same machine (usually J-1 neighbors).

        # Apply GNN
        for layer in self.layers:
            h = layer(h, adj_prec, adj_mach)

        # Output: Embeddings for the NEXT operation of each job.
        # We need to gather the embeddings corresponding to td["next_op_idx"] for each job.

        # next_op_idx: (B, J) - values 0..M-1
        # Flattened index for job j: j * M + next_op_idx[j]
        # But if job is finished (next_op_idx == M), what then?
        # We should clamp or mask.
        # Since usage is typically masked by valid actions, we can just clamp to M-1 to avoid index error
        # and rely on action mask to prevent selection.

        next_indices = td["next_op_idx"].clone()
        # mask finished
        finished = td["finished_jobs"]  # (B, J)
        next_indices[finished] = num_machines - 1  # Point to last op (dummy safe)

        # Calculate flat indices
        # job_offsets: [0, M, 2M, ...]
        job_offsets = torch.arange(0, num_jobs * num_machines, num_machines, device=td.device).unsqueeze(0)  # (1, J)
        flat_indices = job_offsets + next_indices  # (B, J)

        # Gather embeddings: (B, total_ops, dim) -> (B, J, dim)
        job_embeddings = h.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Also return the full graph embedding if needed? Usually for global context.
        # For now, return job_embeddings corresponding to candidate actions.

        return job_embeddings, h
