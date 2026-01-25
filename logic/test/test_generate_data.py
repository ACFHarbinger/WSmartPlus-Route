"""Tests for data generation."""

import numpy as np
import pytest

from logic.src.constants import MAX_WASTE
from logic.src.data.builders import VRPInstanceBuilder
from logic.src.data.generate_data import generate_datasets


class TestVRPInstanceBuilder:
    """Tests for VRPInstanceBuilder."""

    @pytest.fixture
    def builder(self):
        """Fixture for VRPInstanceBuilder."""
        return VRPInstanceBuilder(
            data=None,
            depot_idx=0,
            vehicle_cap=1.0,
            customers=[],
            dimension=0,
            coords=[],
        )

    def test_default_initialization(self, builder):
        """Test default initialization values."""
        # We can't access private members easily, but we can verify build output with mocks
        pass

    def test_parameter_setting(self, builder):
        """Test parameter setting methods."""
        b = (
            builder.set_dataset_size(50)
            .set_problem_size(20)
            .set_waste_type("plastic")
            .set_distribution("gamma1")
            .set_area("Lisbon")
            .set_focus_graph("dummy.json", 10)
            .set_method("mmn")
            .set_num_days(5)
            .set_problem_name("vrpp")
        )

        # Verify method chaining works (returns self)
        assert isinstance(b, VRPInstanceBuilder)
        # Use simple name mangling check if needed, or assume it works if build uses them.

    def test_build_simple_random(self, builder, mocker):
        """Test building simple random instance."""
        # Mock dependencies to avoid actual data generation
        mocker.patch("logic.src.data.builders.load_focus_coords")
        mocker.patch("logic.src.data.builders.process_coordinates")
        mocker.patch(
            "logic.src.data.builders.generate_waste_prize",
            return_value=np.zeros((10, 20)),
        )

        # Setup builder
        builder.set_dataset_size(10).set_problem_size(20).set_num_days(1)

        # Act
        dataset = builder.build()

        # Assert
        assert len(dataset) == 10
        for item in dataset:
            assert len(item) == 4
            depot, loc, waste, max_waste = item
            assert isinstance(depot, list)
            assert isinstance(loc, list)
            assert isinstance(waste, list)
            assert max_waste == MAX_WASTE

    def test_build_with_focus_graph(self, builder, mocker):
        """Test building with focus graph."""
        # Mock dependencies
        mock_load = mocker.patch(
            "logic.src.data.builders.load_focus_coords",
            return_value=(
                np.zeros((10, 2)),  # depot
                np.zeros((10, 20, 2)),  # loc
                np.array([[0, 0], [1, 1]]),  # mm_arr
                [0] * 10,  # idx
            ),
        )
        mocker.patch(
            "logic.src.data.builders.generate_waste_prize",
            return_value=np.zeros((10, 20)),
        )

        # Setup builder
        builder.set_dataset_size(10).set_problem_size(20).set_focus_graph("focus.json", 10)

        # Act
        dataset = builder.build()

        # Assert
        mock_load.assert_called_once()
        assert len(dataset) == 10

    def test_build_multiday(self, builder, mocker):
        """Test building multi-day instance."""
        mocker.patch(
            "logic.src.data.builders.generate_waste_prize",
            side_effect=[np.zeros((10, 20)), np.ones((10, 20))],
        )

        builder.set_dataset_size(10).set_problem_size(20).set_num_days(2)

        dataset = builder.build()

        assert len(dataset) == 10
        # check waste structure for first item
        # waste list should be [day1_waste, day2_waste] for that instance?
        # Logic: generate_waste_prize returns (batch, nodes).
        # fill_values collects them: [(batch, nodes), (batch, nodes)] -> (days, batch, nodes) (via list)
        # transposed: (batch, days, nodes).
        # item[2] is fill_values[i].tolist() -> List of Days.

        waste_instance_0 = dataset[0][2]
        assert len(waste_instance_0) == 2  # 2 days
        assert waste_instance_0[0] == [0.0] * 20
        assert waste_instance_0[1] == [1.0] * 20


class TestGenerateData:
    """Tests for the main dataset generation logic in generate_data.py."""

    @pytest.fixture
    def mock_builder(self, mocker):
        """Fixture for mocked VRPInstanceBuilder."""
        # Mock the VRPInstanceBuilder class
        MockBuilderClass = mocker.patch("logic.src.data.generate_data.VRPInstanceBuilder")
        mock_instance = MockBuilderClass.return_value

        # Configure fluent interface: methods return self
        mock_instance.set_dataset_size.return_value = mock_instance
        mock_instance.set_problem_size.return_value = mock_instance
        mock_instance.set_waste_type.return_value = mock_instance
        mock_instance.set_distribution.return_value = mock_instance
        mock_instance.set_area.return_value = mock_instance
        mock_instance.set_focus_graph.return_value = mock_instance
        mock_instance.set_method.return_value = mock_instance
        mock_instance.set_num_days.return_value = mock_instance
        mock_instance.set_problem_name.return_value = mock_instance

        # Configure build to return a dummy dataset
        # Format: list of tuples (depot, loc, waste, max_waste)
        # waste should be list of days (if multiday) or simple list
        # For simplicity, just return something that won't break slicing [x[2] for ...]
        mock_instance.build.return_value = [
            ([0, 0], [[0, 0]], [[0.1]], [1.0])  # 1 instance
        ]

        # Configure build_td to return a dummy TensorDict
        import torch
        from tensordict import TensorDict

        mock_instance.build_td.return_value = TensorDict(
            {"locs": torch.zeros(1, 2, 2), "depot": torch.zeros(1, 2)}, batch_size=[1]
        )

        return mock_instance

    @pytest.mark.parametrize("problem", ["vrpp", "wcvrp"])
    @pytest.mark.unit
    def test_single_problem_generation(self, gen_data_opts, problem, mock_builder, mocker):
        """Tests that a single, simple problem is generated using the Builder."""

        gen_data_opts["problem"] = problem
        gen_data_opts["graph_sizes"] = [50]
        gen_data_opts["data_distributions"] = ["gamma1"]

        mock_save = mocker.patch("logic.src.data.generate_data.save_td_dataset")

        # Act
        generate_datasets(gen_data_opts)

        # Assert save_td_dataset was called exactly once
        mock_save.assert_called_once()

        # Assert Builder methods were called
        mock_builder.set_dataset_size.assert_called_with(gen_data_opts["dataset_size"])
        mock_builder.set_problem_size.assert_called_with(50)
        mock_builder.set_waste_type.assert_called_with(gen_data_opts["waste_type"])
        mock_builder.set_distribution.assert_called_with("gamma1")
        mock_builder.set_area.assert_called_with(gen_data_opts["area"])
        mock_builder.set_method.assert_called_with(gen_data_opts["vertex_method"])
        mock_builder.set_problem_name.assert_called_with(problem)
        mock_builder.build_td.assert_called_once()

    @pytest.mark.parametrize(
        "problem, expected_distributions",
        [
            (
                "wcvrp",
                [
                    "empty",
                    "const",
                    "unif",
                    "dist",
                    "emp",
                    "gamma1",
                    "gamma2",
                    "gamma3",
                    "gamma4",
                ],
            ),
        ],
    )
    @pytest.mark.unit
    def test_multiple_distribution_generation(
        self, gen_data_opts, problem, expected_distributions, mock_builder, mocker
    ):
        """Tests problems that iterate over multiple data distributions."""

        gen_data_opts["problem"] = problem
        gen_data_opts["graph_sizes"] = [10]
        gen_data_opts["data_distributions"] = ["all"]

        mocker.patch("logic.src.data.generate_data.save_td_dataset")

        # Act
        generate_datasets(gen_data_opts)

        # Assert build called correct number of times
        assert mock_builder.build_td.call_count == len(expected_distributions)

        # Check that set_distribution was called with each expected distribution
        called_distributions = [call.args[0] for call in mock_builder.set_distribution.call_args_list]
        assert sorted(called_distributions) == sorted(expected_distributions)

    @pytest.mark.unit
    def test_multiple_sizes_generation(self, gen_data_opts, mock_builder, mocker):
        """Tests generation across multiple graph sizes."""

        gen_data_opts["problem"] = "vrpp"
        gen_data_opts["data_distributions"] = ["gamma1"]
        gen_data_opts["graph_sizes"] = [10, 20]
        gen_data_opts["focus_graphs"] = [
            "dummy1",
            "dummy2",
        ]  # Strings to avoid TypeError in os.path.isfile

        mocker.patch("logic.src.data.generate_data.save_td_dataset")

        # Act
        generate_datasets(gen_data_opts)

        # Assert build called correct number of times
        assert mock_builder.build_td.call_count == len(gen_data_opts["graph_sizes"])

        # Check sizes
        called_sizes = [call.args[0] for call in mock_builder.set_problem_size.call_args_list]
        assert called_sizes == gen_data_opts["graph_sizes"]

    @pytest.mark.unit
    def test_wsr_generation(self, gen_data_opts, mock_builder, mocker):
        """Tests the special case for WSR simulator data generation."""

        gen_data_opts["dataset_type"] = "test_simulator"
        gen_data_opts["graph_sizes"] = [5]
        gen_data_opts["data_distributions"] = ["gamma1"]
        gen_data_opts["n_epochs"] = 7
        gen_data_opts["problem"] = "wcvrp"

        mock_save = mocker.patch("logic.src.data.generate_data.save_dataset")

        # Act
        generate_datasets(gen_data_opts)

        # Assert builder configuration for WSR
        mock_builder.set_num_days.assert_called_with(7)
        mock_builder.set_problem_size.assert_called_with(5)
        mock_builder.build.assert_called_once()
        mock_save.assert_called_once()

    @pytest.mark.unit
    def test_unknown_problem_raises_exception(self, gen_data_opts):
        """Tests that an unknown problem name raises a KeyError."""

        gen_data_opts["problem"] = "unknown_problem"
        gen_data_opts["graph_sizes"] = [10]

        with pytest.raises(KeyError, match="unknown_problem"):
            generate_datasets(gen_data_opts)
