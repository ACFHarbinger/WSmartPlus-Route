from unittest.mock import MagicMock, patch

import pytest
import torch

from logic.src.models.embeddings.matnet import MatNetInitEmbedding
from logic.src.models.modules.matnet_mha import MixedScoreMHA
from logic.src.models.policies.matnet import MatNetPolicy
from logic.src.models.subnets.matnet_encoder import MatNetEncoder


class TestMatNetParity:
    @pytest.fixture
    def matrix_input(self):
        batch_size, row_size, col_size = 2, 5, 5
        return torch.rand(batch_size, row_size, col_size)

    def test_init_embedding(self, matrix_input):
        embed_dim = 128
        init_emb = MatNetInitEmbedding(embed_dim)
        row_emb, col_emb = init_emb(matrix_input)

        assert row_emb.shape == (2, 5, 128)
        assert col_emb.shape == (2, 5, 128)

    def test_mixed_score_mha(self, matrix_input):
        batch_size, row_size, col_size = matrix_input.shape
        embed_dim = 128
        row_emb = torch.rand(batch_size, row_size, embed_dim)
        col_emb = torch.rand(batch_size, col_size, embed_dim)

        mha = MixedScoreMHA(n_heads=8, embed_dim=embed_dim)
        r_out, c_out = mha(row_emb, col_emb, matrix_input)

        assert r_out.shape == (2, 5, 128)
        assert c_out.shape == (2, 5, 128)

    def test_matnet_encoder_layer(self, matrix_input):
        from logic.src.models.subnets.matnet_encoder import MatNetEncoderLayer
        embed_dim = 128
        row_emb = torch.rand(2, 5, 128)
        col_emb = torch.rand(2, 5, 128)

        layer = MatNetEncoderLayer(embed_dim=128, n_heads=8)
        r_out, c_out = layer(row_emb, col_emb, matrix_input)

        assert r_out.shape == (2, 5, 128)
        assert c_out.shape == (2, 5, 128)

    def test_matnet_policy_forward(self, matrix_input):
        # Mock problem
        problem = MagicMock()
        problem.NAME = "atsp"

        # Mock state for decoder
        state = MagicMock()
        state.ids = torch.arange(2)[:, None]
        # all_finished should return a tensor for .all() check
        state.all_finished.side_effect = [
            torch.tensor([False, False]),
            torch.tensor([True, True])
        ]

        # get_mask should return [batch, 1, nodes] or similar
        state.get_mask.return_value = torch.zeros(2, 1, 5, dtype=torch.bool)
        state.update.return_value = state

        problem.make_state.return_value = state

        policy = MatNetPolicy(
            embed_dim=128,
            hidden_dim=256,
            problem=problem,
            num_layers=2
        )

        # Mock _get_log_p to avoid GlimpseDecoder internals failing on mocks
        log_p_mock = torch.rand(2, 1, 5)
        mask_mock = torch.zeros(2, 1, 5, dtype=torch.bool)
        policy.decoder._get_log_p = MagicMock(return_value=(log_p_mock, mask_mock))

        input_data = {"dist": matrix_input}
        log_p, actions = policy(input_data)

        # log_p: [batch, seq_len, num_nodes]
        # actions: [batch, seq_len]
        assert actions.shape[0] == 2
        assert log_p.shape[0] == 2
        assert log_p.shape[2] == 5
        assert actions.dtype == torch.long
        assert policy.decoder._get_log_p.called
