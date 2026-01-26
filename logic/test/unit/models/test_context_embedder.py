import torch
from logic.src.models.context_embedder import VRPPContextEmbedder, WCContextEmbedder


class TestContextEmbedder:
    def test_wc_embedder_shapes(self):
        """Test Waste Collection embedder output shapes."""
        embed_dim = 128
        model = WCContextEmbedder(embedding_dim=embed_dim, node_dim=3, temporal_horizon=2)

        # Batch=2, Graph=10
        input_data = {
            "locs": torch.rand(2, 10, 2),
            "depot": torch.rand(2, 2),
            "real_waste": torch.rand(2, 10),
            "fill1": torch.rand(2, 10),
            "fill2": torch.rand(2, 10),
        }

        embeddings = model.init_node_embeddings(input_data)

        # Expected shape: [Batch, Graph+1 (Depot), EmbedDim]
        assert embeddings.shape == (2, 11, embed_dim)

        # Check step context dim
        # WC: embed_dim + 2
        assert model.step_context_dim == embed_dim + 2

    def test_vrpp_embedder_shapes(self):
        """Test VRPP embedder output shapes."""
        embed_dim = 64
        model = VRPPContextEmbedder(embedding_dim=embed_dim, node_dim=3, temporal_horizon=0)

        input_data = {
            "loc": torch.rand(1, 5, 2),  # Test 'loc' vs 'locs' key fallback
            "depot": torch.rand(1, 2),
            "demand": torch.rand(1, 5),  # Test 'demand' vs 'waste' fallback
        }

        embeddings = model.init_node_embeddings(input_data, temporal_features=False)

        # Batch=1, Graph=5 -> Total 6
        assert embeddings.shape == (1, 6, embed_dim)

        # Check step context dim
        # VRPP: embed_dim + 1
        assert model.step_context_dim == embed_dim + 1

    def test_wc_key_fallback(self):
        """Test fallback key logic for Waste."""
        model = WCContextEmbedder(embedding_dim=16)

        # No 'waste' or 'demand', but 'noisy_waste'
        input_data = {"locs": torch.rand(1, 5, 2), "depot": torch.rand(1, 2), "noisy_waste": torch.rand(1, 5)}

        # Should not raise key error
        embeddings = model.init_node_embeddings(input_data, temporal_features=False)
        assert embeddings.shape == (1, 6, 16)
