"""
Unit tests for the Predictor class in the wsmart_bin_analysis module.
"""

from unittest.mock import patch

from ..Deliverables.predictors import Predictor


class TestPredictors:
    """
    Test suite for the Predictor class, focusing on initialization and error metrics.
    """

    def test_predictor_init(self, mock_subprocess, predictor_data):
        """Test that the Predictor initializes and manages its cache correctly."""
        train, test, dates = predictor_data

        with (
            patch.object(Predictor, "save_cache") as mock_save,
            patch.object(Predictor, "load_cache") as mock_load,
            patch.object(Predictor, "deleate_cache") as mock_del,
        ):
            Predictor(train, test)

            mock_save.assert_called_once()
            mock_subprocess.assert_called_once()
            mock_load.assert_called_once()
            mock_del.assert_called_once()

    def test_fit_39mean(self, predictor_data, mock_subprocess):
        """Test that mean39error is correctly calculated during Predictor initialization."""
        train, test, dates = predictor_data
        # Mock IO
        with (
            patch.object(Predictor, "save_cache"),
            patch.object(Predictor, "load_cache"),
            patch.object(Predictor, "deleate_cache"),
        ):
            p = Predictor(train, test)
            # Check if mean39error is calculated
            assert p.mean39error is not None
            assert p.mean39error.shape == test.shape

    def test_metrics(self, predictor_data, mock_subprocess):
        """Test calculation of MSE and prediction value retrieval."""
        train, test, dates = predictor_data

        with (
            patch.object(Predictor, "save_cache"),
            patch.object(Predictor, "load_cache"),
            patch.object(Predictor, "deleate_cache"),
        ):
            p = Predictor(train, test)
            # Populate dummy data
            p.prediction = test
            p.pred_error = test * 0.1
            p.real_error = test * 0.05
            # Index aligns with test
            p.prediction.index = test.index
            p.pred_error.index = test.index
            p.real_error.index = test.index

            mse = p.get_MSE()
            assert len(mse) == 5  # 5 bins

            vals, err = p.get_pred_values(test.index[0])
            assert len(vals) == 5
            assert len(err) == 5
