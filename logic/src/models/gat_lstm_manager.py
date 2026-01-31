"""
This module contains the GAT-LSTM Manager agent implementation for Hierarchical RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.src.constants.models import (
    DEFAULT_TEMPORAL_HORIZON,
    FEED_FORWARD_EXPANSION,
    STATIC_DIM,
)
from logic.src.constants.waste import CRITICAL_FILL_THRESHOLD
from logic.src.utils.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


class GATLSTManager(nn.Module):
    """
    Hierarchical Manager Agent for PCVRP.
    Uses LSTM to encode temporal waste patterns and GAT (Transformer) to encode spatial structure.
    Outputs:
        1. Node Mask: Which nodes to visit today.
        2. Route Gate: Whether to dispatch vehicles today.
    """

    def __init__(
        self,
        input_dim_static=STATIC_DIM,  # x, y
        input_dim_dynamic=DEFAULT_TEMPORAL_HORIZON,  # waste history length
        global_input_dim=2,  # avg_waste, max_waste (Only current)
        critical_threshold=CRITICAL_FILL_THRESHOLD,
        batch_size=1024,
        hidden_dim=128,
        lstm_hidden=64,
        num_layers_gat=3,
        n_heads=8,
        dropout=0.1,
        device="cuda",
        shared_encoder=None,
    ):
        """
        Initialize the GATLSTManager.

        Args:
            input_dim_static (int, optional): Static input dimension (e.g., coordinates). Defaults to 2.
            input_dim_dynamic (int, optional): Dynamic input dimension (e.g., waste history). Defaults to 10.
            global_input_dim (int, optional): Global input dimension (e.g., avg waste). Defaults to 2.
            critical_threshold (float, optional): Threshold for critical waste levels. Defaults to 0.9.
            batch_size (int, optional): Batch size. Defaults to 1024.
            hidden_dim (int, optional): Hidden dimension size. Defaults to 128.
            lstm_hidden (int, optional): LSTM hidden dimension. Defaults to 64.
            num_layers_gat (int, optional): Number of GAT layers. Defaults to 3.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            device (str, optional): Computation device. Defaults to 'cuda'.
            shared_encoder (nn.Module, optional): Shared encoder instance. Defaults to None.
        """
        super(GATLSTManager, self).__init__()
        self.device = device
        self.batch_size = batch_size

        # If shared_encoder is provided, we use its embed_dim if possible
        if shared_encoder is not None:
            # Most of our encoders (GAT, GGAC, etc.) have 'embed_dim' or nested attributes.
            try:
                val = None
                if hasattr(shared_encoder, "embed_dim"):
                    val = getattr(shared_encoder, "embed_dim")
                elif hasattr(shared_encoder, "layers") and len(getattr(shared_encoder, "layers")) > 0:
                    # Attempt to find embed_dim in deep layers
                    first_layer = getattr(shared_encoder, "layers")[0]
                    # Check for TransformerEncoderLayer style
                    if (
                        hasattr(first_layer, "att")
                        and hasattr(first_layer.att, "module")
                        and hasattr(first_layer.att.module, "embed_dim")
                    ):
                        val = first_layer.att.module.embed_dim
                    # Check for standard GRU/LSTM/Linear style
                    elif hasattr(first_layer, "embedding_dim"):
                        val = first_layer.embedding_dim

                if isinstance(val, int):
                    hidden_dim = val
                else:
                    logger.debug(f"Could not infer embed_dim from shared_encoder: {shared_encoder}")
            except (AttributeError, IndexError, TypeError) as e:
                logger.warning(
                    f"Error inferring embed_dim from shared_encoder: {e}. Using default hidden_dim={hidden_dim}"
                )

        self.hidden_dim = hidden_dim
        self.input_dim_dynamic = input_dim_dynamic
        self.global_input_dim = global_input_dim
        self.critical_threshold = critical_threshold

        # 1. Temporal Encoder (Long Short-Term Memory)
        # Inputs: (Batch, N, History)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        # 2. Feature Fusion
        # Static (2) + Temporal Embedding (lstm_hidden) -> Hidden
        self.feature_embedding = nn.Linear(input_dim_static + lstm_hidden, hidden_dim)

        # 3. Spatial Encoder (GAT / Transformer)
        # Using TransformerEncoder as standard "GAT" on fully connected graph
        if shared_encoder is not None:
            self.gat_encoder = shared_encoder
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * FEED_FORWARD_EXPANSION,
                dropout=dropout,
                batch_first=True,
            )
            self.gat_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_gat)

        # 4. Heads

        # Node Mask Head (Actor 1)
        # Output: (Batch, N, 2) -> Logits for [Skip, Visit]
        # Input: Hidden + Skip Connection (lstm_hidden)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim + lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Route Gate Head (Actor 2)
        # Global Pooling -> MLP -> (Batch, 2) -> Logits for [No Route, Route]
        # Input: Hidden + Global Features
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim + global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Critic (Value Function)
        # Global Pooling -> MLP -> (Batch, 1)
        # Input: Hidden + Global Features
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(device)

        # Buffers for PPO
        self.clear_memory()

        # Force the Manager to prefer Routing (1) over Skipping (0) at the start
        with torch.no_grad():
            # Assuming gate_head's last layer is a Linear layer
            self.gate_head[-1].bias.fill_(0)

            # Revert to Neutral Bias
            self.mask_head[-1].bias.fill_(0)

    def clear_memory(self):
        """
        Clear the memory buffers used for PPO.
        """
        self.states_static = []
        self.states_dynamic = []
        self.states_global = []
        self.actions_mask = []
        self.actions_gate = []
        self.log_probs_mask = []
        self.log_probs_gate = []
        self.rewards = []
        self.values = []
        self.target_masks = []  # New buffer for target masks
        self.dones = []

    def feature_processing(self, static, dynamic):
        """
        Process static and dynamic features using LSTM and Linear layers.

        Args:
            static (torch.Tensor): Static features (Batch, N, 2).
            dynamic (torch.Tensor): Dynamic features (Batch, N, History).

        Returns:
            tuple: (x, temporal_embed)
                   - x: Combined features projected to hidden dimension (Batch, N, Hidden).
                   - temporal_embed: Temporal embeddings from LSTM (Batch, N, lstm_hidden).
        """
        B, N, H_len = dynamic.size()

        # LSTM Temporal Encoding
        # Reshape dynamic for LSTM: (Batch*N, History, 1)
        dyn_flat = dynamic.view(B * N, H_len, 1)
        _, (h_n, _) = self.lstm(dyn_flat)
        temporal_embed = h_n.squeeze(0).view(B, N, -1)

        # Concatenate with static
        # static: (Batch, N, 2)
        combined = torch.cat([static, temporal_embed], dim=2)

        # Project to hidden
        x = self.feature_embedding(combined)

        return x, temporal_embed

    def forward(self, static, dynamic, global_features):
        """
        Forward pass of the Manager Agent.

        Args:
            static (torch.Tensor): Static features.
            dynamic (torch.Tensor): Dynamic features.
            global_features (torch.Tensor): Global features (Batch, global_input_dim).

        Returns:
            tuple: (mask_logits, gate_logits, value)
                   - mask_logits: Logits for node masking.
                   - gate_logits: Logits for route gating.
                   - value: Estimated state value.
        """
        # Encode
        x, temporal_embed = self.feature_processing(static, dynamic)  # (B, N, H)

        # Spatial Context
        # Transformer expects (Batch, Seq, F) with batch_first=True
        h = self.gat_encoder(x)  # (B, N, H)

        # SKIP CONNECTION: Concatenate temporal_embed (LSTM output) to h
        # h_skip: (B, N, H + lstm_hidden)
        h_skip = torch.cat([h, temporal_embed], dim=-1)

        # Node Mask Logits
        mask_logits = self.mask_head(h_skip)  # (B, N, 2)

        # Global Pooling for Gate/Critic
        # Combine Mean (for general load) and Max (for urgency/overflows)
        h_mean = torch.mean(h, dim=1)  # (B, H)
        h_max = torch.max(h, dim=1)[0]  # (B, H)
        h_global = h_mean + h_max  # Simple addition or concatenation

        # Concatenate Global Features
        # global_features: (B, G)
        h_combined = torch.cat([h_global, global_features], dim=1)  # (B, H+G)

        # Gate Logits
        gate_logits = self.gate_head(h_combined)  # (B, 2)

        # Value
        value = self.critic(h_combined)  # (B, 1)

        return mask_logits, gate_logits, value

    def select_action(
        self,
        static,
        dynamic,
        global_features=None,
        deterministic=False,
        threshold=0.5,
        mask_threshold=0.5,
        target_mask=None,
    ):
        """
        Select actions (gate and mask) based on the current state.
        """
        mask_logits, gate_logits, value = self.forward(static, dynamic, global_features)

        # 1. Gate Decision (0=Skip, 1=Route)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_dist = torch.distributions.Categorical(gate_probs)

        if deterministic:
            if threshold < 0:
                gate_action = torch.ones_like(gate_probs[..., 1]).long()
            else:
                gate_action = (gate_probs[..., 1] > threshold).long()
        else:
            gate_action = gate_dist.sample()

        # 2. Mask Decision (Which nodes to mask/unmask)
        mask_probs = F.softmax(mask_logits, dim=-1)
        mask_dist = torch.distributions.Categorical(mask_probs)

        if deterministic:
            mask_action = (mask_probs[..., 1] > mask_threshold).long()
        else:
            mask_action = mask_dist.sample()  # (Batch, N)

        # Store for PPO
        if not deterministic:
            log_prob_mask = mask_dist.log_prob(mask_action).sum(dim=1)
            log_prob_gate = gate_dist.log_prob(gate_action)

            self.states_static.append(static.detach().cpu())
            self.states_dynamic.append(dynamic.detach().cpu())
            self.states_global.append(global_features.detach().cpu() if global_features is not None else None)
            self.actions_gate.append(gate_action.detach().cpu())
            self.actions_mask.append(mask_action.detach().cpu())
            self.log_probs_gate.append(log_prob_gate.detach().cpu())
            self.log_probs_mask.append(log_prob_mask.detach().cpu())
            self.values.append(value.detach().cpu())

            # Store target mask for auxiliary loss
            if target_mask is not None:
                self.target_masks.append(target_mask.detach().cpu())
            else:
                # Fallback based on waste level
                self.target_masks.append((dynamic[:, :, -1] > self.critical_threshold).float().detach().cpu())

        return mask_action, gate_action, value
