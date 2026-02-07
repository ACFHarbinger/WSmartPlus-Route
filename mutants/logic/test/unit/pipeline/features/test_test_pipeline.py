from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.pipeline.features.test import run_wsr_simulator_test as run_sim_test_func
from logic.src.pipeline.features.test import simulator_testing


class TestPipelineFeaturesTest:
    @pytest.fixture
    def opts(self):
        return {
            "n_samples": 2,
            "policies": ["policy1"],
            "resume": False,
            "days": 5,
            "size": 10,
            "output_dir": "test_output",
            "area": "test_area",
            "bin_idx_file": "test_bins.json",
            "cpu_cores": 1,
            "no_progress_bar": True,
            "data_distribution": "test_dist",
            "seed": 1234,
            "waste_type": "glass",
            "checkpoint_dir": "checkpoints",
            "plastminute_cf": [1.0],
            "pregular_level": [1],
            "gurobi_param": [1],
            "hexaly_param": [1],
            "lookahead_configs": [1],
        }

    @patch("logic.src.pipeline.features.test.udef")
    @patch("logic.src.pipeline.features.test.load_indices")
    @patch("logic.src.pipeline.features.test.sequential_simulations")
    @patch("logic.src.pipeline.features.test.send_final_output_to_gui")
    @patch("logic.src.pipeline.features.test.display_log_metrics")
    def test_simulator_testing_sequential(self, mock_display, mock_send, mock_seq, mock_load, mock_udef, opts):
        # Configure mocks
        # Bottom-Up Mapping:
        # display_log_metrics -> mock_display
        # send_final_output_to_gui -> mock_send
        # sequential_simulations -> mock_seq
        # load_indices -> mock_load
        # udef -> mock_udef

        mock_udef.ROOT_DIR = "/tmp/test"
        mock_udef.SIM_METRICS = ["profit"]
        mock_udef.LOCK_TIMEOUT = 10

        mock_load.return_value = [0, 1]
        mock_seq.return_value = ({"policy1": [1.0]}, {"policy1": [0.1]}, [])

        device = torch.device("cpu")
        data_size = 20

        print("Calling simulator_testing...")
        try:
            simulator_testing(opts, data_size, device)
            print("simulator_testing returned.")
        except Exception as e:
            print(f"simulator_testing failed: {e}")
            raise

        print(f"mock_seq called: {mock_seq.called}")
        print(f"mock_display called: {mock_display.called}")

        assert mock_load.called
        assert mock_seq.called
        assert mock_send.called
        assert mock_display.called

    @patch("logic.src.pipeline.features.test.udef")
    @patch("logic.src.pipeline.features.test.load_indices")
    @patch("logic.src.pipeline.features.test.ThreadPool")
    @patch("logic.src.pipeline.features.test.output_stats")
    @patch("logic.src.pipeline.features.test.send_final_output_to_gui")
    @patch("logic.src.pipeline.features.test.display_log_metrics")
    def test_simulator_testing_parallel(
        self, mock_display, mock_send, mock_out_stats, mock_pool, mock_load, mock_udef, opts
    ):
        mock_udef.ROOT_DIR = "/tmp/test"
        mock_udef.LOCK_TIMEOUT = 100
        mock_udef.PBAR_WAIT_TIME = 0.1

        opts["cpu_cores"] = 2
        opts["n_samples"] = 2

        mock_load.return_value = [0, 1]

        # Mock Pool
        pool_instance = MagicMock()
        mock_pool.return_value = pool_instance

        # Mock results
        task1 = MagicMock()
        task1.ready.return_value = True
        task1.get.return_value = {"success": True, "policy1": 1.0}

        pool_instance.apply_async.return_value = task1

        device = torch.device("cpu")
        data_size = 10

        with patch("multiprocessing.Manager") as mock_manager:
            mock_dict = MagicMock()
            mock_list = MagicMock()
            mock_manager.return_value.dict.return_value = mock_dict
            mock_manager.return_value.list.return_value = mock_list

            def side_effect_apply(func, args, callback):
                callback({"success": True, "policy1": 1.0})
                return task1

            pool_instance.apply_async.side_effect = side_effect_apply

            mock_dict.items.return_value = [("policy1", [1.0])]

            simulator_testing(opts, data_size, device)

            assert pool_instance.apply_async.called

    @patch("logic.src.pipeline.features.test.load_simulator_data")
    @patch("logic.src.pipeline.features.test.simulator_testing")
    @patch("os.makedirs")
    def test_run_wsr_simulator_test(self, mock_makedirs, mock_sim_test, mock_load_data, opts):
        mock_load_data.return_value = ([1] * 10, None)  # 10 bins

        opts["data_dir"] = None

        run_sim_test_func(opts)

        assert mock_sim_test.called
        assert mock_makedirs.called

        # Check policy expansion
        # opts['policies'] passed to sim_test should be expanded
        call_args = mock_sim_test.call_args[0][0]
        assert "policy1_test_dist" in call_args["policies"]

    def test_run_wsr_simulator_test_fallback(self, opts):
        # Test fallback logic if load_simulator_data fails
        with patch("logic.src.pipeline.features.test.load_simulator_data", side_effect=Exception("Fail")):
            with patch("logic.src.pipeline.features.test.simulator_testing") as mock_sim_test:
                with patch("os.makedirs"):
                    opts["area"] = "mixrmbac"
                    opts["size"] = 20
                    run_sim_test_func(opts)

                    # Should use fallback data_size logic
                    # For mixrmbac 20 -> data_size 20 (or default)
                    # mock_sim_test(opts, data_size, device)
                    # check data_size argument
                    assert mock_sim_test.call_args[0][1] == 20
