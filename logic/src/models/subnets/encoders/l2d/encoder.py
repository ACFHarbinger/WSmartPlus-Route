"""L2D Encoder using Disjunctive Graph GNN."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from .l2d_graph_conv import L2DGraphConv


class L2DEncoder(nn.Module):
    """
    L2D Encoder using Disjunctive Graph GNN.

    Encodes the JSSP state into embeddings for each operation.
    """

    def __init__(self, embed_dim: int = 128, num_layers: int = 3, feedforward_hidden: int = 512, **kwargs):
        """
        Initialize L2DEncoder.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.init_embed = nn.Linear(1, embed_dim)  # Project duration
        self.layers = nn.ModuleList([L2DGraphConv(embed_dim, feedforward_hidden) for _ in range(num_layers)])

    def forward(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """
        batch_size = td.batch_size[0] if td.batch_size else 1
        num_jobs = td["proc_time"].shape[1]
        num_machines = td["proc_time"].shape[2]
        total_ops = num_jobs * num_machines

        # Flatten state to nodes
        durations = td["proc_time"].view(batch_size, total_ops, 1)
        machine_order = td["machine_order"].view(batch_size, total_ops)  # (B, J*M)

        # Initial embedding from duration
        h = self.init_embed(durations)  # (B, total_ops, embed_dim)

        # 1. Precedence Adjacency: (B, J*M, J*M)
        rows = []
        cols = []
        for j in range(num_jobs):
            for i in range(num_machines - 1):
                curr_idx = j * num_machines + i
                next_idx = curr_idx + 1
                rows.append(curr_idx)
                cols.append(next_idx)

        if not hasattr(self, "_base_adj_prec") or self._base_adj_prec.shape[0] != total_ops:
            base_adj = torch.zeros(total_ops, total_ops, device=td.device)
            base_rows = torch.tensor(rows, device=td.device)
            base_cols = torch.tensor(cols, device=td.device)
            base_adj[base_rows, base_cols] = 1.0
            self._base_adj_prec = base_adj

        adj_prec = self._base_adj_prec.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Machine Adjacency
        mach_match = machine_order.unsqueeze(2) == machine_order.unsqueeze(1)
        diag_mask = torch.eye(total_ops, device=td.device, dtype=torch.bool).unsqueeze(0)
        adj_mach = mach_match & (~diag_mask)
        adj_mach = adj_mach.float()

        # Apply GNN
        for layer in self.layers:
            h = layer(h, adj_prec, adj_mach)

        # Output: Embeddings for the NEXT operation of each job.
        next_indices = td["next_op_idx"].clone()
        finished = td["finished_jobs"]  # (B, J)
        next_indices[finished] = num_machines - 1  # Point to last op

        # Calculate flat indices
        job_offsets = torch.arange(0, num_jobs * num_machines, num_machines, device=td.device).unsqueeze(0)  # (1, J)
        flat_indices = job_offsets + next_indices  # (B, J)

        # Gather embeddings: (B, total_ops, dim) -> (B, J, dim)
        job_embeddings = h.gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        return job_embeddings, h
