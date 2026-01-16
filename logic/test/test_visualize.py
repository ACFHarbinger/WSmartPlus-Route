import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from logic.src.utils.plot_utils import (
    plot_linechart,
    plot_tsp,
    plot_vehicle_routes,
)
from logic.src.utils.visualize_utils import (
    get_batch,
    log_weight_distributions,
    plot_attention_heatmaps,
    plot_weight_trajectories,
    project_node_embeddings,
)


class TestVisualizeUtils(unittest.TestCase):
    def test_get_batch(self):
        device = torch.device("cpu")
        batch = get_batch(device, size=10, batch_size=2)
        self.assertTrue("depot" in batch)
        self.assertEqual(batch["depot"].shape, (2, 2))

    @patch("logic.src.utils.visualize_utils.plt")
    @patch("logic.src.utils.visualize_utils.os.listdir")
    @patch("logic.src.utils.visualize_utils.torch.load")
    def test_plot_weight_trajectories(self, mock_load, mock_listdir, mock_plt):
        mock_listdir.return_value = ["epoch-1.pt", "epoch-2.pt"]
        # Mock checkpoint
        mock_load.return_value = {"model": {"l1.weight": torch.randn(10, 10)}}

        plot_weight_trajectories("dummy_dir", "dummy_dir/out.png")
        mock_plt.savefig.assert_called()

    @patch("logic.src.utils.visualize_utils.SummaryWriter")
    def test_log_weight_distributions(self, mock_writer_cls):
        model = MagicMock()
        model.named_parameters.return_value = [("p1", torch.randn(5))]
        mock_writer = mock_writer_cls.return_value

        log_weight_distributions(model, 1, "logs")
        mock_writer.add_histogram.assert_called()
        mock_writer.close.assert_called()

    @patch("logic.src.utils.visualize_utils.SummaryWriter")
    def test_project_node_embeddings(self, mock_writer_cls):
        model = MagicMock()
        model._get_initial_embeddings.return_value = torch.randn(2, 5, 10)  # B, N, D
        model.embedder.return_value = torch.randn(2, 5, 10)
        batch = {"edges": None, "dist": None}
        mock_writer = mock_writer_cls.return_value

        project_node_embeddings(model, batch, "logs")
        mock_writer.add_embedding.assert_called()

    @patch("logic.src.utils.visualize_utils.plt")
    @patch("logic.src.utils.visualize_utils.sns")
    def test_plot_attention_heatmaps(self, mock_sns, mock_plt):
        model = MagicMock()
        # Create a mock layer
        layer = MagicMock()
        # Create a mock for the module (mha)
        mha = MagicMock()
        # Mock W_query
        w_query = MagicMock()
        # Set up the chain so .numpy() returns a real array
        w_query.weight.data.cpu.return_value.numpy.return_value = np.zeros((8, 10, 10))

        # Attach W_query to mha
        mha.W_query = w_query
        # Mock others to avoid auto-creation issues or just point to same w_query
        mha.W_key = w_query
        mha.W_val = w_query

        # Attach mha to layer
        layer.att.module = mha

        model.embedder.layers = [layer]

        plot_attention_heatmaps(model, "out_dir")
        mock_sns.heatmap.assert_called()
        mock_plt.savefig.assert_called()


@patch("logic.src.utils.plot_utils.plt")
class TestPlotUtils(unittest.TestCase):
    def test_plot_linechart(self, mock_plt):
        # plot_linechart expects strange shape, look at code
        # if len=2, iterates
        # Let's try simple usage
        # Actually plot_linechart is complex.
        # graph_log shape (N, 6) usually? "list(zip(*lg))[5]"
        # Let's mock small 2D array
        # It slices extensively.
        # graph_log is typically list of logs?
        msg = np.zeros((1, 2, 6))  # 1 policy, 2 points, 6 metrics
        plot_linechart("out.png", msg, mock_plt.plot, ["pol1"])
        mock_plt.savefig.assert_called()

    def test_plot_tsp(self, mock_plt):
        xy = np.random.rand(5, 2)
        tour = np.array([0, 1, 2, 3, 4, 0])
        ax = MagicMock()
        plot_tsp(xy, tour, ax)
        ax.plot.assert_called()
        ax.scatter.assert_called()

    def test_plot_vehicle_routes(self, mock_plt):
        data = {
            "depot": torch.tensor([0.5, 0.5]),
            "loc": torch.rand(5, 2),
            "demand": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
        }
        route = torch.tensor([0, 1, 2, 0, 3, 4, 5, 0])
        ax = MagicMock()
        # Default capacity is 1, total demand of route 1 is 0.2 <= 1.
        plot_vehicle_routes(data, route, ax)
        ax.plot.assert_called()
        # It creates PatchCollection
        ax.add_collection.assert_not_called()  # visualize_demands=False default
