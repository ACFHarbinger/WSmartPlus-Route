"""
Output Analysis Tab UI component.
"""

from __future__ import annotations

import json
import os
import subprocess
import webbrowser
from collections import defaultdict

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .engine import (
    extract_num_bins_from_path,
    pivot_json_data,
    process_tensorboard_file,
)
from .plotting import generate_plot


class OutputAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.json_data = None
        self.sim_windows = []
        self._all_loaded_json_paths = []
        self.tb_process = None

        layout = QVBoxLayout(self)

        # Controls
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Output File(s) (JSON/JSONL/TBL)")
        self.load_btn.clicked.connect(self.load_files)
        control_layout.addWidget(self.load_btn)

        self.dist_combo = QComboBox()
        self.dist_combo.setToolTip("Filter plots by data distribution (emp, gammaX, etc.)")
        control_layout.addWidget(QLabel("Distribution:"))
        control_layout.addWidget(self.dist_combo)

        self.x_key_combo = QComboBox()
        self.x_key_combo.setPlaceholderText("X-Axis")
        control_layout.addWidget(QLabel("X:"))
        control_layout.addWidget(self.x_key_combo)

        self.y_key_combo = QComboBox()
        self.y_key_combo.setPlaceholderText("Y-Axis")
        control_layout.addWidget(QLabel("Y:"))
        control_layout.addWidget(self.y_key_combo)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"])
        control_layout.addWidget(QLabel("Type:"))
        control_layout.addWidget(self.chart_type_combo)

        self.pareto_check = QCheckBox("Pareto Front")
        self.pareto_check.setToolTip("Highlight non-dominated solutions (Min X, Min Y)")
        control_layout.addWidget(self.pareto_check)

        self.plot_btn = QPushButton("Plot Chart")
        self.plot_btn.clicked.connect(self.show_plot_dialog)
        self.plot_btn.setEnabled(False)
        control_layout.addWidget(self.plot_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Content
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.tabs.addTab(self.text_view, "Merged Data Summary")

        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_widget)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.chart_layout.addWidget(self.canvas)
        self.tabs.addTab(self.chart_widget, "Visualization")

    def _clear_data_state_only(self):
        """Resets the data state and input controls, but keeps the current plot on the figure."""
        self.json_data = None
        self._all_loaded_json_paths = []
        self.text_view.setText("Input data cleared. Load new files to continue.")
        self.y_key_combo.clear()
        self.x_key_combo.clear()
        self.dist_combo.clear()
        self.plot_btn.setEnabled(False)

        for win in self.sim_windows:
            if win is not None:
                win.close()
        self.sim_windows = []

        if self.tb_process:
            self.tb_process.terminate()
            self.tb_process = None

        QMessageBox.information(
            self,
            "Data Cleared",
            "All merged data, file paths, and TensorBoard sessions have been cleared.",
        )

    def clear_data(self):
        """Resets the entire state including the figure."""
        self._clear_data_state_only()
        self.figure.clear()
        self.canvas.draw()

    def show_plot_dialog(self):
        """Displays a dialog to ask the user if they want to clear merged data."""
        if not self.json_data:
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
        if self.tb_process:
            self.tb_process.terminate()
            self.tb_process = None

        try:
            port = 6006
            cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
            self.tb_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
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
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Output File(s)", "", "Output Files (*.json *.jsonl *.tfevents*)"
        )
        if not file_paths:
            return

        json_files = [f for f in file_paths if f.endswith(".json")]
        jsonl_files = [f for f in file_paths if f.endswith(".jsonl")]
        tb_files = [f for f in file_paths if "tfevents" in f]

        for fpath in jsonl_files:
            from ...windows import SimulationResultsWindow

            win = SimulationResultsWindow(policy_names=["External_Log"], log_path=fpath)
            win.show()
            self.sim_windows.append(win)

        if tb_files:
            tb_logdir = os.path.dirname(tb_files[0])
            self._launch_tensorboard(tb_logdir)

        if not json_files and not tb_files:
            if jsonl_files:
                self.text_view.setText(f"Opened {len(jsonl_files)} JSONL file(s) in external windows.")
            return

        try:
            if self.json_data:
                all_policy_names = self.json_data.pop("__Policy_Names__", [])
                all_distributions = self.json_data.pop("__Distributions__", [])
                all_file_ids = self.json_data.pop("__File_IDs__", [])
                all_n_bins = self.json_data.pop("Num Bins", [])
                merged_metrics = defaultdict(list, self.json_data)
                valid_keys_set = set(merged_metrics.keys())
            else:
                merged_metrics = defaultdict(list)
                all_policy_names = []
                all_distributions = []
                all_file_ids = []
                all_n_bins = []
                valid_keys_set = set()

            files_to_process = json_files + tb_files
            for fpath in files_to_process:
                self._all_loaded_json_paths.append(fpath)

            summary_text = "--- Loaded/Merged Files ---\n"
            for fpath in sorted(list(set(self._all_loaded_json_paths))):
                summary_text += f"- {fpath}\n"

            for fpath in files_to_process:
                fname_prefix = os.path.basename(fpath)
                n_bins_val = extract_num_bins_from_path(fpath)
                file_unique_id = fpath

                if "tfevents" in fpath:
                    pivoted_data = process_tensorboard_file(fpath)
                else:
                    with open(fpath, "r") as f:
                        raw_data = json.load(f)

                    if isinstance(raw_data, dict) and raw_data and isinstance(next(iter(raw_data.values())), dict):
                        pivoted_data = pivot_json_data(
                            raw_data,
                            filename_prefix=fname_prefix,
                            file_id=file_unique_id,
                        )
                    else:
                        pivoted_data = raw_data

                current_names = pivoted_data.get("__Policy_Names__", [])
                count = len(current_names)

                all_policy_names.extend(current_names)
                all_distributions.extend(pivoted_data.get("__Distributions__", ["unknown"] * count))
                all_file_ids.extend(pivoted_data.get("__File_IDs__", [file_unique_id] * count))
                all_n_bins.extend([n_bins_val] * count)

                for k, v in pivoted_data.items():
                    if k in ["__Policy_Names__", "__Distributions__", "__File_IDs__"]:
                        continue
                    if isinstance(v, list):
                        merged_metrics[k].extend(v)
                        valid_keys_set.add(k)

            self.json_data = dict(merged_metrics)
            self.json_data["__Policy_Names__"] = all_policy_names
            self.json_data["__Distributions__"] = all_distributions
            self.json_data["__File_IDs__"] = all_file_ids
            self.json_data["Num Bins"] = all_n_bins

            valid_keys_set.add("Num Bins")
            if "step" in valid_keys_set:
                valid_keys_set.add("step")

            final_keys = sorted(list(valid_keys_set))

            self.y_key_combo.clear()
            self.y_key_combo.addItems(final_keys)

            self.x_key_combo.clear()
            self.x_key_combo.addItem("Policy Names")
            self.x_key_combo.addItems(final_keys)

            if any("tfevents" in f for f in files_to_process) and "step" in final_keys:
                index = self.x_key_combo.findText("step")
                if index >= 0:
                    self.x_key_combo.setCurrentIndex(index)

            self.dist_combo.clear()
            self.dist_combo.addItem("All")
            unique_dists_found = sorted(list(set(d for d in all_distributions if d != "unknown")))
            if unique_dists_found:
                self.dist_combo.addItems(unique_dists_found)

            self.plot_btn.setEnabled(len(final_keys) > 0)

            summary_text += f"\nTotal Policies/Datapoints: {len(all_policy_names)}\n"
            summary_text += f"Available Metrics: {', '.join(final_keys)}"
            self.text_view.setText(summary_text)
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process files: {e}")

    def plot_json_key(self):
        if not self.json_data:
            return

        generate_plot(
            figure=self.figure,
            json_data=self.json_data,
            y_key=self.y_key_combo.currentText(),
            x_selection=self.x_key_combo.currentText(),
            chart_type=self.chart_type_combo.currentText(),
            selected_dist=self.dist_combo.currentText(),
            pareto_enabled=self.pareto_check.isChecked(),
        )
        self.canvas.draw()
        self.tabs.setCurrentIndex(1)

    def shutdown(self):
        for win in self.sim_windows:
            if win is not None:
                win.close()
        self.sim_windows = []
        if self.tb_process:
            self.tb_process.terminate()
            self.tb_process = None
