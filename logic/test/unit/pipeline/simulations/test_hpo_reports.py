"""Unit tests for Optuna HPO report export (§A.5)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import optuna
import pytest
from logic.src.pipeline.simulations.hpo.hpo_reports import (
    export_optuna_study_from_storage,
    export_optuna_study_reports,
)


class TestExportOptunaStudyReports:
    @pytest.mark.unit
    def test_skips_when_few_completed_trials(self, tmp_path: Any) -> None:
        study = MagicMock()
        study.study_name = "test_study"
        study.trials = [
            MagicMock(state=optuna.trial.TrialState.COMPLETE),
        ]

        result = export_optuna_study_reports(study, output_dir=str(tmp_path), min_completed=2)

        assert result is None
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.hpo.hpo_reports._write_plotly_figure")
    @patch("logic.src.pipeline.simulations.hpo.hpo_reports.ov", create=True)
    def test_exports_manifest_and_plots(
        self,
        mock_ov: Any,
        mock_write: Any,
        tmp_path: Any,
    ) -> None:
        mock_ov.plot_parallel_coordinate.return_value = MagicMock()
        mock_ov.plot_param_importances.return_value = MagicMock()
        mock_ov.plot_optimization_history.return_value = MagicMock()
        mock_write.side_effect = lambda _fig, stem, report_dir: [
            f"{report_dir}/{stem}.html",
        ]

        study = MagicMock()
        study.study_name = "alns_seed42"
        study.trials = [
            MagicMock(state=optuna.trial.TrialState.COMPLETE),
            MagicMock(state=optuna.trial.TrialState.COMPLETE),
        ]
        study.best_value = 12.5
        study.best_params = {"alpha": 0.1}

        report_dir = export_optuna_study_reports(study, output_dir=str(tmp_path))

        assert report_dir is not None
        assert report_dir.startswith(str(tmp_path))
        manifest_path = f"{report_dir}/manifest.json"
        with open(manifest_path, encoding="utf-8") as handle:
            import json

            manifest = json.load(handle)

        assert manifest["study_name"] == "alns_seed42"
        assert manifest["n_complete"] == 2
        assert manifest["best_value"] == 12.5
        assert "parallel_coordinate" in manifest["artefacts"]
        assert mock_write.call_count == 3

    @pytest.mark.unit
    @patch(
        "logic.src.pipeline.simulations.hpo.hpo_reports.export_optuna_study_reports",
        return_value="/tmp/reports/alns",
    )
    @patch("optuna.load_study")
    def test_export_from_storage(
        self,
        mock_load: Any,
        mock_export: Any,
        tmp_path: Any,
    ) -> None:
        mock_load.return_value = MagicMock()

        result = export_optuna_study_from_storage(
            "sqlite:///study.db",
            "alns_seed42",
            output_dir=str(tmp_path),
        )

        assert result == "/tmp/reports/alns"
        mock_load.assert_called_once_with(study_name="alns_seed42", storage="sqlite:///study.db")
        mock_export.assert_called_once()
