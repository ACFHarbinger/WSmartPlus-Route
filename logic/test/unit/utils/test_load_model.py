import json
import os
from unittest.mock import MagicMock, patch
import torch
from logic.src.utils.model.loader import load_model

class TestLoadModel:
    """Class for load_model tests."""

    @patch("logic.src.utils.model.problem_factory.load_problem")
    @patch("logic.src.utils.model.loader.load_args")
    @patch("torch.load")
    def test_load_model_basic(self, mock_torch_load, mock_load_args, mock_load_problem, tmp_path):
        """Test load_model by mocking internal dependencies and file structure."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        hparams = {
            "problem": "vrpp",
            "model": "am",
            "encoder": "gat",
            "embed_dim": 128,
            "hidden_dim": 512,
            "n_encode_layers": 3,
            "n_encode_sublayers": 1,
            "n_heads": 8,
            "normalization": "batch",
            "tanh_clipping": 10.0,
        }
        mock_load_args.return_value = hparams

        # Create dummy epoch file and args.json with content
        (model_dir / "epoch-10.pt").touch()

        with open(model_dir / "args.json", "w") as f:
            json.dump(hparams, f)

        mock_problem = MagicMock()
        mock_load_problem.return_value = mock_problem
        mock_torch_load.return_value = {"model": {}}

        with patch("os.listdir", return_value=["epoch-10.pt"]):
            model, args = load_model(str(model_dir))

        assert model is not None
        assert args == mock_load_args.return_value
        assert mock_torch_load.called
