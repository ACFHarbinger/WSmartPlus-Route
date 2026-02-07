from logic.src.models.policies.am import AttentionModelPolicy
from logic.src.models.subnets.encoders.ham_encoder import HAMEncoder


class HAMPolicy(AttentionModelPolicy):
    """
    Heterogeneous Attention Model Policy for PDP.

    Uses HAMEncoder to encode the heterogeneous graph (Depot, Pickups, Deliveries)
    and standard Attention Model decoder (with masking for PDP constraints).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "layer",
        feedforward_hidden: int = 512,
        env_name: str = "pdp",
        **kwargs,
    ):
        # Initialize encoder
        encoder = HAMEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
        )

        super().__init__(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            encoder=encoder,
            **kwargs,
        )
