import json
import os
import re
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class OutputAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.json_data = None
        self.sim_windows = []
        self._all_loaded_json_paths = []
        self.tb_process = None  # Handle to the TensorBoard subprocess

        layout = QVBoxLayout(self)

        # --- Controls (UNCHANGED) ---
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Output File(s) (JSON/JSONL/TBL)")
        self.load_btn.clicked.connect(self.load_files)
        control_layout.addWidget(self.load_btn)

        # --- Distribution Selector (UNCHANGED) ---
        self.dist_combo = QComboBox()
        self.dist_combo.setToolTip("Filter plots by data distribution (emp, gammaX, etc.)")
        control_layout.addWidget(QLabel("Distribution:"))
        control_layout.addWidget(self.dist_combo)
        # ------------------------------------------

        # X-Axis Selector (UNCHANGED)
        self.x_key_combo = QComboBox()
        self.x_key_combo.setPlaceholderText("X-Axis")
        control_layout.addWidget(QLabel("X:"))
        control_layout.addWidget(self.x_key_combo)

        # Y-Axis Selector (UNCHANGED)
        self.y_key_combo = QComboBox()
        self.y_key_combo.setPlaceholderText("Y-Axis")
        control_layout.addWidget(QLabel("Y:"))
        control_layout.addWidget(self.y_key_combo)

        # Chart Type (UNCHANGED)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"])
        control_layout.addWidget(QLabel("Type:"))
        control_layout.addWidget(self.chart_type_combo)

        # Pareto Toggle (UNCHANGED)
        self.pareto_check = QCheckBox("Pareto Front")
        self.pareto_check.setToolTip("Highlight non-dominated solutions (Min X, Min Y)")
        control_layout.addWidget(self.pareto_check)

        self.plot_btn = QPushButton("Plot Chart")
        self.plot_btn.clicked.connect(self.show_plot_dialog)
        self.plot_btn.setEnabled(False)
        control_layout.addWidget(self.plot_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # --- Content (UNCHANGED) ---
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Raw Text View (UNCHANGED)
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.tabs.addTab(self.text_view, "Merged Data Summary")

        # Chart View (UNCHANGED)
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_widget)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.chart_layout.addWidget(self.canvas)
        self.tabs.addTab(self.chart_widget, "Visualization")

    # --- NEW HELPER METHOD: Clears data state without touching the figure ---
    def _clear_data_state_only(self):
        """Resets the data state and input controls, but keeps the current plot on the figure."""
        self.json_data = None
        self._all_loaded_json_paths = []
        self.text_view.setText("Input data cleared. Load new files to continue.")
        self.y_key_combo.clear()
        self.x_key_combo.clear()
        self.dist_combo.clear()
        self.plot_btn.setEnabled(False)  # Disable plotting until new data is loaded

        # Close external windows if any are open
        for win in self.sim_windows:
            if win is not None:
                win.close()
        self.sim_windows = []

        # Terminate TensorBoard if running
        if self.tb_process:
            self.tb_process.terminate()
            self.tb_process = None

        QMessageBox.information(
            self,
            "Data Cleared",
            "All merged data, file paths, and TensorBoard sessions have been cleared.",
        )

    # --- MODIFIED METHOD: Clears all loaded data/files and the figure ---
    def clear_data(self):
        """Resets the entire state including the figure."""
        self._clear_data_state_only()
        # Ensure the figure is cleared when clear_data is called explicitly (full reset)
        self.figure.clear()
        self.canvas.draw()

    # --- MODIFIED METHOD: Shows the dialog before plotting ---
    def show_plot_dialog(self):
        """Displays a dialog to ask the user if they want to clear merged data."""

        if not self.json_data:
            self.plot_json_key()
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Plotting Options")
        msg.setText("You have existing data loaded. Do you want to clear all previously merged data after plotting?")
        msg.setIcon(QMessageBox.Question)

        # Define the buttons
        plot_only_btn = msg.addButton("Plot Current Data", QMessageBox.AcceptRole)
        clear_and_plot_btn = msg.addButton("Plot and Clear Data", QMessageBox.DestructiveRole)
        _ = msg.addButton(QMessageBox.Cancel)

        msg.exec()

        # Handle the result
        if msg.clickedButton() == plot_only_btn:
            self.plot_json_key()

        elif msg.clickedButton() == clear_and_plot_btn:
            # 1. Call plot function immediately
            self.plot_json_key()

            # 2. Use QTimer.singleShot to delay state clearing (50ms), guaranteeing the plot renders first
            # We call the state-only clear to preserve the figure.
            QTimer.singleShot(50, self._clear_data_state_only)

    def _pivot_json_data(self, data, filename_prefix="", file_id=None):
        # ... (UNCHANGED) ...
        metrics = defaultdict(list)
        policy_names = []
        distributions = []
        file_ids = []

        # Regex to find common distribution patterns.
        # \b ensures it matches end of word/string (e.g. 'am_emp')
        DIST_PATTERN = r"_(emp|gamma\d+|uniform)\b"

        for policy, results in data.items():
            if not isinstance(results, dict):
                continue

            # 1. Determine Distribution
            match = re.search(DIST_PATTERN, policy, re.IGNORECASE)

            if match:
                dist = match.group(1).lower()
                # Remove the distribution suffix to get the "Base" policy name
                base_name = policy[: match.start()] + policy[match.end() :]
                base_name = base_name.rstrip("_")  # Clean trailing underscores
            else:
                dist = "unknown"
                base_name = policy

            distributions.append(dist)

            # 2. Set Policy Name
            policy_names.append(base_name)

            # Add File ID for every point
            file_ids.append(file_id)

            # 3. Append Metrics
            for metric, value in results.items():
                metrics[metric].append(value)

        metrics["__Policy_Names__"] = policy_names
        metrics["__Distributions__"] = distributions
        metrics["__File_IDs__"] = file_ids
        return metrics

    def _process_tensorboard_file(self, fpath):
        """Extracts scalar data from a TensorBoard event file."""
        ea = EventAccumulator(fpath)
        ea.Reload()

        tags = ea.Tags()["scalars"]
        if not tags:
            return {}

        metrics = defaultdict(list)

        data_by_step = defaultdict(dict)

        filename = os.path.basename(fpath)

        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                data_by_step[e.step][tag] = e.value
                data_by_step[e.step]["wall_time"] = e.wall_time

        # Flatten into lists
        sorted_steps = sorted(data_by_step.keys())

        count = len(sorted_steps)
        metrics["step"] = sorted_steps
        metrics["wall_time"] = [data_by_step[s].get("wall_time", 0) for s in sorted_steps]

        for tag in tags:
            metrics[tag] = [data_by_step[s].get(tag, float("nan")) for s in sorted_steps]

        metrics["__Policy_Names__"] = [f"{filename}" for _ in range(count)]
        metrics["__Distributions__"] = ["tensorboard"] * count
        metrics["__File_IDs__"] = [fpath] * count

        return metrics

    def _launch_tensorboard(self, logdir):
        """Launches TensorBoard in a subprocess and opens the browser."""
        # 1. Stop existing process if any
        if self.tb_process:
            self.tb_process.terminate()
            self.tb_process = None

        try:
            # 2. Launch new process
            port = 6006  # Default port
            cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
            self.tb_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            # 3. Open Browser
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

        # Launch TensorBoard if TB files are present
        if tb_files:
            # Find common parent directory or just use the directory of the first file
            # Usually strict usage would be the logdir provided to TB.
            # Using the parent directory of the first selected file is a safe bet.
            tb_logdir = os.path.dirname(tb_files[0])
            self._launch_tensorboard(tb_logdir)

        if not json_files and not tb_files:
            if jsonl_files:
                self.text_view.setText(f"Opened {len(jsonl_files)} JSONL file(s) in external windows.")
            return

        try:
            summary_text = ""

            if self.json_data:
                all_policy_names = self.json_data.pop("__Policy_Names__", [])
                all_distributions = self.json_data.pop("__Distributions__", [])
                all_file_ids = self.json_data.pop("__File_IDs__", [])
                # Extract existing Num Bins if merging
                all_n_bins = self.json_data.pop("Num Bins", [])
                merged_metrics = defaultdict(list, self.json_data)
                valid_keys_set = set(merged_metrics.keys())
            else:
                merged_metrics = defaultdict(list)
                all_policy_names = []
                all_distributions = []
                all_file_ids = []
                all_n_bins = []  # New list for bin counts
                valid_keys_set = set()

            # 1. Collect all loaded file paths first
            files_to_process = json_files + tb_files
            for fpath in files_to_process:
                self._all_loaded_json_paths.append(fpath)

            # 2. Build the list of files
            summary_text = "--- Loaded/Merged Files ---\n"
            for fpath in sorted(list(set(self._all_loaded_json_paths))):
                summary_text += f"- {fpath}\n"

            # 3. Process each file
            for fpath in files_to_process:
                fname_prefix = os.path.basename(fpath)

                # --- NEW LOGIC: Extract Num Bins from parent directory ---
                # Example: /path/to/areaname_50/log.json -> parent is areaname_50 -> extracts 50
                parent_dir = os.path.basename(os.path.dirname(fpath))
                bin_match = re.search(r"_(\d+)$", parent_dir)
                n_bins_val = int(bin_match.group(1)) if bin_match else 0
                # -----------------------------------------------------

                file_unique_id = fpath
                pivoted_data = {}

                if "tfevents" in fpath:
                    pivoted_data = self._process_tensorboard_file(fpath)
                else:
                    with open(fpath, "r") as f:
                        raw_data = json.load(f)

                    if isinstance(raw_data, dict) and raw_data and isinstance(next(iter(raw_data.values())), dict):
                        pivoted_data = self._pivot_json_data(
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

                # --- NEW LOGIC: Extend Num Bins list ---
                all_n_bins.extend([n_bins_val] * count)
                # -------------------------------------

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
            self.json_data["Num Bins"] = all_n_bins  # Store in main dict

            # Add "Num Bins" to valid keys so it appears in the dropdown
            valid_keys_set.add("Num Bins")
            if "step" in valid_keys_set:
                valid_keys_set.add("step")  # Make sure it's valid

            final_keys = sorted(list(valid_keys_set))

            self.y_key_combo.clear()
            self.y_key_combo.addItems(final_keys)

            self.x_key_combo.clear()
            self.x_key_combo.addItem("Policy Names")
            self.x_key_combo.addItems(final_keys)

            # If we loaded tensorboard data, maybe default X to 'step'
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

    def _calculate_pareto_front(self, x_values, y_values):
        # ... (UNCHANGED) ...
        points = []
        for i, (xv, yv) in enumerate(zip(x_values, y_values)):
            points.append({"idx": i, "x": xv, "y": yv})

        pareto_indices = []
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i == j:
                    continue
                # Corrected dominance check for Min X and Max Y
                if (
                    other["x"] <= point["x"]
                    and other["y"] >= point["y"]
                    and (other["x"] < point["x"] or other["y"] > point["y"])
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(point["idx"])
        return pareto_indices

    def plot_json_key(self):
        # ... (UNCHANGED) ...
        y_key = self.y_key_combo.currentText()
        x_selection = self.x_key_combo.currentText()
        chart_type = self.chart_type_combo.currentText()
        selected_dist = self.dist_combo.currentText()

        if not y_key or not self.json_data:
            return

        # --- DATA FILTERING (UNCHANGED) ---
        indices_to_plot = []
        all_dists = self.json_data.get("__Distributions__", [])

        # Logic to handle "All" vs specific selection
        for i, dist in enumerate(all_dists):
            if selected_dist == "All":
                indices_to_plot.append(i)
            elif dist == selected_dist:
                indices_to_plot.append(i)

        if not indices_to_plot:
            QMessageBox.warning(self, "Plot Error", f"No data found for distribution: {selected_dist}")
            return

        def filter_data(key, indices):
            full_list = self.json_data.get(key, [])
            return [full_list[i] for i in indices]

        y_data_filtered = filter_data(y_key, indices_to_plot)
        policy_names_filtered = filter_data("__Policy_Names__", indices_to_plot)
        dists_filtered = filter_data("__Distributions__", indices_to_plot)
        file_ids_filtered = filter_data("__File_IDs__", indices_to_plot)

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            is_2d_metric_plot = (x_selection != "Policy Names") and (x_selection in self.json_data)

            if is_2d_metric_plot:
                # --- SCATTER / METRIC vs METRIC (UNCHANGED) ---
                x_plot = filter_data(x_selection, indices_to_plot)

                # Group data by Policy Name for PLOTTING THE LINES
                policy_groups = defaultdict(list)
                for i in range(len(policy_names_filtered)):
                    name = policy_names_filtered[i]
                    policy_groups[name].append(
                        {
                            "x": x_plot[i],
                            "y": y_data_filtered[i],
                            "dist": dists_filtered[i],
                        }
                    )

                # Group data by File ID for PARETO CALCULATION
                file_groups = defaultdict(list)
                for i in range(len(file_ids_filtered)):
                    file_id = file_ids_filtered[i]
                    file_groups[file_id].append({"x": x_plot[i], "y": y_data_filtered[i]})

                # Plot each Policy Group (the lines and markers)
                for name, points in policy_groups.items():
                    points.sort(key=lambda p: p["x"])
                    xs = [p["x"] for p in points]
                    ys = [p["y"] for p in points]

                    # Plot Line + Markers
                    ax.plot(xs, ys, marker="o", label=name, markersize=6, alpha=0.8)

                ax.set_xlabel(x_selection)
                ax.set_ylabel(y_key)
                ax.set_title(f"{y_key} vs {x_selection} (Grouped by Policy)")

                # Add Legend (Might need adjustment if too many policies)
                if len(policy_groups) < 15:
                    ax.legend(fontsize="small")

                # PARETO LOGIC (Applied Per File Group) (UNCHANGED)
                if self.pareto_check.isChecked():
                    all_pareto_points_found = False

                    # 1. Iterate through each FILE GROUP
                    for file_id, points in file_groups.items():
                        group_xs = [p["x"] for p in points]
                        group_ys = [p["y"] for p in points]

                        # 2. Calculate Pareto Front for THIS FILE GROUP
                        pareto_indices = self._calculate_pareto_front(group_xs, group_ys)

                        if pareto_indices:
                            all_pareto_points_found = True
                            # 3. Extract and Sort Pareto Points
                            pareto_pts = sorted(
                                [(group_xs[i], group_ys[i]) for i in pareto_indices],
                                key=lambda p: p[0],
                            )

                            # 4. Plot the Pareto Line for THIS FILE GROUP
                            px, py = zip(*pareto_pts)
                            ax.plot(px, py, "--", color="black", linewidth=1.5, zorder=1)

                    # Add a single entry for the Pareto Front to the legend outside the loop
                    if all_pareto_points_found:
                        # Draw a transparent line just for the legend entry (only one dashed line label)
                        ax.plot(
                            [],
                            [],
                            "--",
                            color="black",
                            linewidth=1.5,
                            label="Pareto Front",
                            zorder=1,
                        )
                        # Re-add legend to include the new Pareto label
                        ax.legend(fontsize="small")

            else:
                # --- STANDARD: Metric vs Policy Name (UNCHANGED) ---
                # Since we consolidated names, duplicates may appear on X-axis if not careful.
                # Here we just treat them as individual items or grouped bars.

                x_indices = range(len(policy_names_filtered))

                # Construct label: Name + [Dist] to differentiate on X-axis
                x_labels = [f"{n} [{d}]" for n, d in zip(policy_names_filtered, dists_filtered)]

                if chart_type == "Line Chart":
                    ax.plot(x_indices, y_data_filtered, marker="o")
                elif chart_type == "Bar Chart":
                    ax.bar(x_indices, y_data_filtered)
                elif chart_type == "Scatter Plot":
                    ax.scatter(x_indices, y_data_filtered)
                elif chart_type == "Area Chart":
                    ax.fill_between(x_indices, y_data_filtered, alpha=0.5)

                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("Policy Name [Distribution]")
                ax.set_ylabel(y_key)
                ax.set_title(f"{y_key} Across Policies")

            ax.grid(True, linestyle="--", alpha=0.6)
            self.figure.tight_layout()
            self.canvas.draw()
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Could not generate plot: {str(e)}")

    def shutdown(self):
        for win in self.sim_windows:
            if win is not None:
                win.close()
        self.sim_windows = []
