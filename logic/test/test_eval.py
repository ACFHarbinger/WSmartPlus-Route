from unittest.mock import MagicMock, patch

import numpy as np
import torch

from logic.src.pipeline.eval import eval_dataset, get_best, run_evaluate_model


class TestEval:
    def test_get_best(self):
        # seqs: (batch, seq_len)
        seqs = np.array([[1, 2], [3, 4], [5, 6]])
        costs = np.array([10.0, 5.0, 20.0])
        ids = np.array([0, 0, 1])  # 0 has [1,2](10) and [3,4](5). 1 has [5,6](20).

        # for id 0, min cost is 5 ([3,4]). for id 1, min cost is 20 ([5,6]).
        best_seqs, best_costs = get_best(seqs, costs, ids)

        assert len(best_seqs) == 2
        # Check first result (id 0)
        np.testing.assert_array_equal(best_seqs[0], [3, 4])
        assert best_costs[0] == 5.0
        # Check second result (id 1)
        np.testing.assert_array_equal(best_seqs[1], [5, 6])
        assert best_costs[1] == 20.0

    @patch("logic.src.pipeline.eval.load_model")
    @patch("logic.src.pipeline.eval.save_dataset")
    @patch("logic.src.pipeline.eval.setup_cost_weights")
    @patch("logic.src.pipeline.eval.torch.utils.data.DataLoader")
    def test_eval_dataset_logic(self, mock_loader, mock_setup_weights, mock_save, mock_load_model, eval_opts):
        # Mocks
        mock_model = MagicMock()
        mock_load_model.return_value = (mock_model, {})
        mock_model.problem.NAME = "cvrpp"
        mock_model.problem.make_dataset.return_value = MagicMock()

        # Mock dataloader to return 1 batch
        batch = {"data": torch.randn(2, 5)}  # dummy batch
        mock_loader.return_value = [batch]  # 1 iteration

        # Mock model.sample_many
        # returns (seqs, costs)
        # batch size=2.
        mock_model.sample_many.return_value = (
            torch.tensor([[1, 2], [3, 4]]),  # seqs
            torch.tensor([10.0, 20.0]),  # costs
        )

        # Mock problem.get_costs
        mock_model.problem.get_costs.return_value = (
            None,
            {"length": torch.tensor([1.0]), "waste": torch.tensor([0.5]), "overflows": torch.tensor([0])},
            None,
        )

        # Run
        costs, tours, durations = eval_dataset("data.pkl", 1, 1.0, eval_opts)

        # Assertions
        assert len(costs) == 2
        mock_save.assert_called()
        mock_model.sample_many.assert_called()

    @patch("logic.src.pipeline.eval.load_model")
    @patch("logic.src.pipeline.eval.save_dataset")
    @patch("logic.src.pipeline.eval.setup_cost_weights")
    @patch("logic.src.pipeline.eval.torch.utils.data.DataLoader")
    def test_eval_dataset_beam_search(self, mock_loader, mock_setup_weights, mock_save, mock_load_model, eval_opts):
        # Mocks
        mock_model = MagicMock()
        mock_load_model.return_value = (mock_model, {})
        mock_model.problem.NAME = "cvrpp"
        mock_model.problem.make_dataset.return_value = MagicMock()

        batch = {"data": torch.randn(2, 5)}
        mock_loader.return_value = [batch]

        # Mock beam_search
        # cum_log_p, sequences, costs, ids, batch_size
        # Using batch_size=2
        mock_model.beam_search.return_value = (
            None,
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([10.0, 20.0]),
            torch.tensor([0, 1]),  # ids
            2,
        )

        mock_model.problem.get_costs.return_value = (
            None,
            {"length": torch.tensor([1.0]), "waste": torch.tensor([0.5]), "overflows": torch.tensor([0])},
            None,
        )

        eval_opts["decode_strategy"] = "bs"
        eval_dataset("data.pkl", 1, 1.0, eval_opts)
        mock_model.beam_search.assert_called()

    @patch("logic.src.pipeline.eval.eval_dataset")
    def test_run_evaluate_model(self, mock_eval):
        # Just calls eval_dataset in loop
        opts = {"seed": 1234, "width": [1], "datasets": ["d1"], "softmax_temperature": 1.0}
        run_evaluate_model(opts)
        mock_eval.assert_called()
