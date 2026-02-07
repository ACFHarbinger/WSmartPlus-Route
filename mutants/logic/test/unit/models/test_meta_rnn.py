"""Tests for WeightAdjustmentRNN."""

import pytest
import torch
from logic.src.models.meta_rnn import WeightAdjustmentRNN


class TestMetaRNN:
    """Tests for the WeightAdjustmentRNN module."""

    @pytest.fixture
    def rnn(self):
        """Create RNN instance for testing."""
        return WeightAdjustmentRNN(
            input_size=10,
            hidden_size=32,
            output_size=3,
            num_layers=1
        )

    def test_initialization(self, rnn):
        """Test model initialization and architecture."""
        assert rnn.hidden_size == 32
        assert rnn.num_layers == 1
        assert isinstance(rnn.lstm, torch.nn.LSTM)
        assert len(rnn.fc) == 3  # Linear, ReLU, Linear

    def test_forward_pass_single(self, rnn):
        """Test forward pass with single sequence."""
        batch_size = 4
        seq_len = 5
        input_size = 10

        x = torch.rand(batch_size, seq_len, input_size)
        adjustments, hidden = rnn(x)

        # adjustments should be [batch_size, output_size]
        assert adjustments.shape == (batch_size, 3)
        # hidden should be (h, c) for LSTM
        assert hidden[0].shape == (1, batch_size, 32)
        assert hidden[1].shape == (1, batch_size, 32)

    def test_forward_pass_with_hidden(self, rnn):
        """Test forward pass with provided hidden state."""
        batch_size = 1
        seq_len = 1
        input_size = 10

        x = torch.rand(batch_size, seq_len, input_size)
        h0 = torch.zeros(1, batch_size, 32)
        c0 = torch.zeros(1, batch_size, 32)

        adjustments, (h_n, c_n) = rnn(x, (h0, c0))

        assert adjustments is not None
        assert not torch.equal(h0, h_n)

    def test_gradients(self, rnn):
        """Verify backpropagation works."""
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
        x = torch.rand(2, 3, 10)
        target = torch.rand(2, 3)

        adjustments, _ = rnn(x)
        loss = torch.nn.functional.mse_loss(adjustments, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
