from typing import Any, Dict, Optional

from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.common.autoregressive import (
    AutoregressivePolicy,
)
from logic.src.models.subnets.decoders.matnet import MatNetDecoder
from logic.src.models.subnets.embeddings.matnet import MatNetInitEmbedding
from logic.src.models.subnets.encoders.matnet_encoder import MatNetEncoder


class MatNetPolicy(AutoregressivePolicy):
    """
    MatNet Policy for matrix-based Combinatorial Optimization.
    Unifies MatNetEncoder and MatNetDecoder with proper initialization.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        num_layers: int = 5,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        normalization: str = "instance",
        **kwargs,
    ):
        super(MatNetPolicy, self).__init__(env_name=None)
        self.problem = problem
        self.embed_dim = embed_dim

        self.init_embedding = MatNetInitEmbedding(embed_dim, normalization)

        self.encoder = MatNetEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            feed_forward_hidden=hidden_dim,
            normalization=normalization,
        )

        self.decoder = MatNetDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            **kwargs,
        )

    def set_decode_type(self, decode_type: str, temp: Optional[float] = None):
        if self.decoder is not None and hasattr(self.decoder, "set_decode_type"):
            self.decoder.set_decode_type(decode_type, temp)

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        decode_type: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Policy forward pass.
        Args:
            input_data: Dict containing 'dist' or 'cost_matrix' of shape [batch, row_size, col_size]
        """
        # Get matrix (e.g. distance or cost matrix)
        matrix = td.get("dist")
        if matrix is None:
            matrix = td.get("cost_matrix")

        if matrix is None:
            raise ValueError("MatNetPolicy requires a cost matrix in 'dist' or 'cost_matrix' key.")

        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)

        # Initial row and column embeddings from matrix stats
        row_emb, col_emb = self.init_embedding(matrix)

        # Encoding using mixed-score attention
        assert self.encoder is not None, "Encoder is not initialized"
        row_emb, col_emb = self.encoder(row_emb, col_emb, matrix)

        # Decoding
        # We pass row_emb as embeddings and col_emb via kwargs
        assert self.decoder is not None, "Decoder is not initialized"
        cost_weights = kwargs.get("cost_weights")
        log_p, actions = self.decoder(td, row_emb, cost_weights=cost_weights, col_embeddings=col_emb, **kwargs)

        return {"log_p": log_p, "actions": actions}
