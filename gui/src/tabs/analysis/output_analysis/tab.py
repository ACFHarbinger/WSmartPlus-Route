"""
Dashboard and visualization for simulation results.
"""

from __future__ import annotations

import json
import os
import subprocess
import webbrowser
from collections import defaultdict

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .data.state import OutputDataState
from .engine import (
    extract_num_bins_from_path,
    pivot_json_data,
    process_tensorboard_file,
)
from .plotting import generate_plot
from .widgets.controls import OutputControlsWidget
from .widgets.visualization import VisualizationWidget


class OutputAnalysisTab(QWidget):
    """
    Main tab for analyzing and visualizing simulation results from JSON/TensorBoard files.
    """

    def __init__(self):
        """
        Initialize the OutputAnalysisTab and setup sub-widgets.
        """
        super().__init__()
        self.state = OutputDataState()

        layout = QVBoxLayout(self)

        # Controls
        self.controls = OutputControlsWidget()
        self.controls.load_btn.clicked.connect(self.load_files)
        self.controls.plot_btn.clicked.connect(self.show_plot_dialog)
        layout.addWidget(self.controls)

        # Visualization
        self.visualization = VisualizationWidget()
        layout.addWidget(self.visualization)

    def _clear_data_state_only(self):
        """Resets the data state and input controls, but keeps the current plot on the figure."""
        self.state.clear()

        self.visualization.text_view.setText("Input data cleared. Load new files to continue.")
        self.controls.y_key_combo.clear()
        self.controls.x_key_combo.clear()
        self.controls.dist_combo.clear()
        self.controls.plot_btn.setEnabled(False)

        QMessageBox.information(
            self,
            "Data Cleared",
            "All merged data, file paths, and TensorBoard sessions have been cleared.",
        )

    def clear_data(self):
        """Resets the entire state including the figure."""
        self._clear_data_state_only()
        self.visualization.clear()

    def show_plot_dialog(self):
        """Displays a dialog to ask the user if they want to clear merged data."""
        if not self.state.json_data:
            self.plot_json_key()
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Plotting Options")
        msg.setText("You have existing data loaded. Do you want to clear all previously merged data after plotting?")
        msg.setIcon(QMessageBox.Icon.Question)

        plot_only_btn = msg.addButton("Plot Current Data", QMessageBox.ButtonRole.AcceptRole)
        clear_and_plot_btn = msg.addButton("Plot and Clear Data", QMessageBox.ButtonRole.DestructiveRole)
        _ = msg.addButton(QMessageBox.StandardButton.Cancel)

        msg.exec()

        if msg.clickedButton() == plot_only_btn:
            self.plot_json_key()
        elif msg.clickedButton() == clear_and_plot_btn:
            self.plot_json_key()
            QTimer.singleShot(50, self._clear_data_state_only)

    def _launch_tensorboard(self, logdir: str):
        """Launches TensorBoard in a subprocess and opens the browser."""
        if self.state.tb_process:
            self.state.tb_process.terminate()
            self.state.tb_process = None

        try:
            port = 6006
            cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
            self.state.tb_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            url = f"http://localhost:{port}"
            webbrowser.open(url)
            QMessageBox.information(
                self,
                "TensorBoard Launched",
                f"TensorBoard running at {url}\nLogdir: {logdir}",
            )
        except Exception as e:
            QMessageBox.warning(self, "TensorBoard Error", f"Failed to launch TensorBoard: {e}")

    def load_files(self):
        """Open a file dialog to load output files and trigger data processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Output File(s)", "", "Output Files (*.json *.jsonl *.tfevents*)"
        )
        if not file_paths:
            return

        json_files, jsonl_files, tb_files = self._categorize_files(file_paths)

        for fpath in jsonl_files:
            from ...windows import SimulationResultsWindow

            win = SimulationResultsWindow(policy_names=["External_Log"], log_path=fpath)
            win.show()
            self.state.sim_windows.append(win)

        if tb_files:
            self._launch_tensorboard(os.path.dirname(tb_files[0]))

        if not json_files and not tb_files:
            if jsonl_files:
                self.visualization.text_view.setText(f"Opened {len(jsonl_files)} JSONL file(s).")
            return

        try:
            merged = self._merge_json_data(json_files + tb_files)
            self.state.json_data = merged

            final_keys = sorted(
                [k for k in merged if k not in ["__Policy_Names__", "__Distributions__", "__File_IDs__"]]
            )
            self._update_controls_with_metrics(final_keys, merged["__Distributions__"], merged["__Policy_Names__"])

            summary_text = "--- Loaded/Merged Files ---\n" + "\n".join(
                [f"- {p}" for p in self.state.get_loaded_paths()]
            )
            summary_text += f"\n\nTotal Policies/Datapoints: {len(merged['__Policy_Names__'])}\n"
            summary_text += f"Available Metrics: {', '.join(final_keys)}"
            self.visualization.text_view.setText(summary_text)
            self.visualization.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process files: {e}")

    def _categorize_files(self, file_paths):
        """Groups file paths by their extension/type."""
        json_f = [f for f in file_paths if f.endswith(".json")]
        jsonl_f = [f for f in file_paths if f.endswith(".jsonl")]
        tb_f = [f for f in file_paths if "tfevents" in f]
        return json_f, jsonl_f, tb_f

    def _merge_json_data(self, files_to_process):
        """Processes and merges data from JSON and TensorBoard files."""
        if self.state.json_data:
            all_names = self.state.json_data.get("__Policy_Names__", []).copy()
            all_dists = self.state.json_data.get("__Distributions__", []).copy()
            all_ids = self.state.json_data.get("__File_IDs__", []).copy()
            all_bins = self.state.json_data.get("Num Bins", []).copy()
            merged_metrics = defaultdict(
                list,
                {
                    k: v
                    for k, v in self.state.json_data.items()
                    if k not in ["__Policy_Names__", "__Distributions__", "__File_IDs__", "Num Bins"]
                },
            )
        else:
            merged_metrics = defaultdict(list)
            all_names, all_dists, all_ids, all_bins = [], [], [], []

        for fpath in files_to_process:
            self.state.add_loaded_path(fpath)
            n_bins = extract_num_bins_from_path(fpath)

            if "tfevents" in fpath:
                pivoted = process_tensorboard_file(fpath)
            else:
                with open(fpath, "r") as f:
                    raw = json.load(f)
                pivoted = (
                    pivot_json_data(raw, os.path.basename(fpath), fpath)
                    if isinstance(raw, dict) and raw and isinstance(next(iter(raw.values())), dict)
                    else raw
                )

            names = pivoted.get("__Policy_Names__", [])
            count = len(names)
            all_names.extend(names)
            all_dists.extend(pivoted.get("__Distributions__", ["unknown"] * count))
            all_ids.extend(pivoted.get("__File_IDs__", [fpath] * count))
            all_bins.extend([n_bins] * count)

            for k, v in pivoted.items():
                if k not in ["__Policy_Names__", "__Distributions__", "__File_IDs__"] and isinstance(v, list):
                    merged_metrics[k].extend(v)

        result = dict(merged_metrics)
        result.update(
            {
                "__Policy_Names__": all_names,
                "__Distributions__": all_dists,
                "__File_IDs__": all_ids,
                "Num Bins": all_bins,
            }
        )
        return result

    def _update_controls_with_metrics(self, final_keys, distributions, names):
        """Updates UI combo boxes with available metrics and distributions."""
        self.controls.y_key_combo.clear()
        self.controls.y_key_combo.addItems(final_keys)

        self.controls.x_key_combo.clear()
        self.controls.x_key_combo.addItem("Policy Names")
        self.controls.x_key_combo.addItems(final_keys)

        if "step" in final_keys:
            idx = self.controls.x_key_combo.findText("step")
            if idx >= 0:
                self.controls.x_key_combo.setCurrentIndex(idx)

        self.controls.dist_combo.clear()
        self.controls.dist_combo.addItem("All")
        unique_dists = sorted(list(set(d for d in distributions if d != "unknown")))
        if unique_dists:
            self.controls.dist_combo.addItems(unique_dists)

        self.controls.plot_btn.setEnabled(len(final_keys) > 0)

    def plot_json_key(self):
        """
        Generate a plot based on the current selections in the control widget.
        """
        if not self.state.json_data:
            return

        generate_plot(
            figure=self.visualization.figure,
            json_data=self.state.json_data,
            y_key=self.controls.y_key_combo.currentText(),
            x_selection=self.controls.x_key_combo.currentText(),
            chart_type=self.controls.chart_type_combo.currentText(),
            selected_dist=self.controls.dist_combo.currentText(),
            pareto_enabled=self.controls.pareto_check.isChecked(),
        )
        self.visualization.canvas.draw()
        self.visualization.setCurrentIndex(1)

    def shutdown(self):
        """
        Clean up resources (processes and data state) before closing.
        """
        self.state.clear()
