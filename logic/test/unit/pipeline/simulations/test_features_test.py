"""Tests for pipeline/features/test.py."""

from unittest.mock import MagicMock, patch

import torch
from logic.src.pipeline.features.test import simulator_testing


def test_simulator_testing_single_core():
    """Verify simulation testing orchestration on a single core."""
    opts = {
        "policies": ["policy1"],
        "n_samples": 1,
        "resume": False,
        "cpu_cores": 1,
        "days": 1,
        "size": 20,
        "area": "mixrmbac",
        "output_dir": "test_out",
        "data_distribution": "unif",
        "bin_idx_file": None,
        "waste_type": "residually",
    }

    device = torch.device("cpu")

    # Mock dependencies
    with patch("logic.src.pipeline.features.test.orchestrator.sequential_simulations") as mock_seq, patch(
        "logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui"
    ), patch("logic.src.pipeline.features.test.orchestrator.display_log_metrics"):
        mock_seq.return_value = ({"policy1": [0.5]}, None, [])

        simulator_testing(opts, 20, device)

        assert mock_seq.called


def test_simulator_testing_multi_core():
    """Verify multi-core simulation testing flow with process pool mocking."""
    opts = {
        "policies": ["policy1"],
        "n_samples": 2,
        "resume": False,
        "cpu_cores": 2,
        "days": 1,
        "size": 20,
        "area": "mixrmbac",
        "output_dir": "test_out",
        "data_distribution": "unif",
        "bin_idx_file": None,
        "no_progress_bar": True,
    }

    device = torch.device("cpu")

    # Mock Manager, Pool, and results
    with patch("multiprocessing.Manager") as mock_manager, patch(
        "logic.src.pipeline.features.test.orchestrator.ThreadPool"
    ) as mock_pool_cls, patch("logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui"), patch(
        "logic.src.pipeline.features.test.orchestrator.display_log_metrics"
    ), patch("logic.src.pipeline.features.test.orchestrator.output_stats"):
        mock_instance = mock_manager.return_value

        # Mock DictProxy behavior
        from multiprocessing.managers import DictProxy

        mock_dict_proxy = MagicMock(spec=DictProxy)
        # Provide at least 2 samples to avoid StatisticsError in stdev
        mock_dict_proxy.items.return_value = [("policy1", [[1.0], [2.0]])]
        mock_instance.dict.return_value = mock_dict_proxy
        mock_instance.list.return_value = []

        mock_pool = mock_pool_cls.return_value
        mock_task = MagicMock()
        mock_task.ready.return_value = True
        mock_pool.apply_async.return_value = mock_task

        # Patch isinstance to return True for our mock_dict_proxy
        with patch(
            "logic.src.pipeline.features.test.orchestrator.isinstance",
            side_effect=lambda x, y: True if x is mock_dict_proxy and y is DictProxy else isinstance(x, y),
        ):
            simulator_testing(opts, 20, device)

        assert mock_pool.apply_async.called
