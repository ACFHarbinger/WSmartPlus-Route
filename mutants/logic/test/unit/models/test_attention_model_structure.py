from unittest.mock import MagicMock

import pytest
import torch
from logic.src.models.attention_model import AttentionModel
from logic.src.models.model_factory import NeuralComponentFactory


class TestAttentionModelStructure:
    @pytest.fixture
    def mock_problem(self):
        problem = MagicMock()
        problem.NAME = "wcvrp"
        return problem

    @pytest.fixture
    def mock_factory(self):
        factory = MagicMock(spec=NeuralComponentFactory)
        # Mock components - strict spec causes issues with custom methods like _calc_log_likelihood
        factory.create_encoder.return_value = MagicMock()
        factory.create_decoder.return_value = MagicMock()
        return factory

    def test_initialization_wcvrp(self, mock_problem, mock_factory):
        """Test initialization for WCVRP context."""
        mock_problem.NAME = "wcvrp"

        model = AttentionModel(
            embed_dim=128, hidden_dim=64, problem=mock_problem, component_factory=mock_factory, n_encode_layers=2
        )

        # Verify Context Embedder Strategy
        assert model.is_wc
        assert not model.is_vrpp
        # Verify components creation
        assert mock_factory.create_encoder.called
        assert mock_factory.create_decoder.called

    def test_initialization_vrpp(self, mock_problem, mock_factory):
        """Test initialization for VRPP context."""
        mock_problem.NAME = "vrpp"

        model = AttentionModel(embed_dim=128, hidden_dim=64, problem=mock_problem, component_factory=mock_factory)

        assert model.is_vrpp
        assert not model.is_wc

    def test_forward_pass_structure(self, mock_problem, mock_factory):
        """Test basic forward pass flow (mocked)."""
        model = AttentionModel(embed_dim=128, hidden_dim=64, problem=mock_problem, component_factory=mock_factory)

        # Mock embedder output
        # embedder(node_embeddings, edges, ...) -> embeddings
        model.embedder.return_value = torch.rand(1, 10, 128)

        # Mock decoder output
        # decoder(input, embeddings, ...) -> (log_p, pi)
        log_p = torch.rand(1, 10)
        pi = torch.randint(0, 10, (1, 10))
        model.decoder.return_value = (log_p, pi)
        # Mock calc_log_likelihood attached to decoder
        model.decoder._calc_log_likelihood.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # Mock problem cost calculation
        # get_costs(input, pi, ...) -> (cost, cost_dict, mask)
        mock_problem.get_costs.return_value = (torch.tensor([10.0]), {}, None)

        # Fake input
        dummy_input = {
            "locs": torch.rand(1, 10, 2),
            "depot": torch.rand(1, 2),
            "demand": torch.rand(1, 10),
            "dist": torch.rand(1, 10, 10),
        }

        # Run forward
        cost, ll, cost_dict, pi_out, entropy = model(dummy_input)

        assert cost.item() == 10.0
        assert mock_problem.get_costs.called
        assert model.embedder.called
        assert model.decoder.called
