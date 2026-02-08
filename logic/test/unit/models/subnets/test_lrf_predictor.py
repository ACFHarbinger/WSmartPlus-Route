
import pytest
import torch
from logic.src.models.subnets.other.lstm_fill_predictor import LongShortTermMemoryFillPredictor

class TestLRFPredictor:
    @pytest.fixture
    def predictor(self):
        return LongShortTermMemoryFillPredictor(
            input_dim=1,
            hidden_dim=32,
            num_layers=1,
            bidirectional=False
        )

    def test_initialization(self, predictor):
        assert isinstance(predictor, LongShortTermMemoryFillPredictor)
        assert predictor.lstm.hidden_size == 32
        assert predictor.lstm.num_layers == 1

    def test_forward(self, predictor):
        batch_size = 4
        seq_len = 10
        # Input shape: (Batch, Seq, InputDim)
        x = torch.randn(batch_size, seq_len, 1)

        output = predictor(x)

        # Output should be (Batch, 1) - predicted fill level
        assert output.shape == (batch_size, 1)

    def test_get_embedding(self, predictor):
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 1)

        embedding = predictor.get_embedding(x)

        # Embedding should be (Batch, HiddenDim)
        assert embedding.shape == (batch_size, 32)

    def test_bidirectional(self):
        predictor = LongShortTermMemoryFillPredictor(
            input_dim=1,
            hidden_dim=32,
            bidirectional=True
        )
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 1)

        output = predictor(x)
        assert output.shape == (batch_size, 1)

        embedding = predictor.get_embedding(x)
        # Embedding should be (Batch, NumDirections * HiddenDim)
        # In current implementation it returns output[:, -1, :], which is (Batch, NumDirections * Hidden)
        assert embedding.shape == (batch_size, 64)
