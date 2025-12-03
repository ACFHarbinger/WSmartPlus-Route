import json
import os
import re 
from collections import defaultdict

from PySide6.QtWidgets import (
    QPushButton, QLabel, QFileDialog, 
    QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QTabWidget, QTextEdit, QMessageBox, QCheckBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ...windows import SimulationResultsWindow


class OutputAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.json_data = None
        self.sim_windows = []
        self._all_loaded_json_paths = []
        
        layout = QVBoxLayout(self)
        
        # --- Controls ---
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Output File(s) (JSON/JSONL)")
        self.load_btn.clicked.connect(self.load_files)
        control_layout.addWidget(self.load_btn)

        # --- Distribution Selector ---
        self.dist_combo = QComboBox()
        self.dist_combo.setToolTip("Filter plots by data distribution (emp, gammaX, etc.)")
        control_layout.addWidget(QLabel("Distribution:"))
        control_layout.addWidget(self.dist_combo)
        # ------------------------------------------

        # X-Axis Selector
        self.x_key_combo = QComboBox()
        self.x_key_combo.setPlaceholderText("X-Axis")
        control_layout.addWidget(QLabel("X:"))
        control_layout.addWidget(self.x_key_combo)

        # Y-Axis Selector
        self.y_key_combo = QComboBox()
        self.y_key_combo.setPlaceholderText("Y-Axis")
        control_layout.addWidget(QLabel("Y:"))
        control_layout.addWidget(self.y_key_combo)
        
        # Chart Type
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"])
        control_layout.addWidget(QLabel("Type:"))
        control_layout.addWidget(self.chart_type_combo)

        # Pareto Toggle
        self.pareto_check = QCheckBox("Pareto Front")
        self.pareto_check.setToolTip("Highlight non-dominated solutions (Min X, Min Y)")
        control_layout.addWidget(self.pareto_check)
        
        self.plot_btn = QPushButton("Plot Chart")
        self.plot_btn.clicked.connect(self.plot_json_key)
        self.plot_btn.setEnabled(False)
        control_layout.addWidget(self.plot_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # --- Content ---
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Raw Text View
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.tabs.addTab(self.text_view, "Merged Data Summary")
        
        # Chart View
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_widget)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.chart_layout.addWidget(self.canvas)
        self.tabs.addTab(self.chart_widget, "Visualization")

    def _pivot_json_data(self, data, filename_prefix=""):
        """
        Transforms {policy: {metric: value}} -> {metric: [val], __Policy_Names__: [name]}
        Also extracts distribution from policy name.
        """
        metrics = defaultdict(list)
        policy_names = []
        distributions = []
        
        # Regex to find common distribution patterns
        # Looks for '_emp_', '_gammaX_', etc. where X is a number
        DIST_PATTERN = r'_(emp|gamma\d+|uniform)' 

        for policy, results in data.items():
            if not isinstance(results, dict): continue
            
            # 1. Determine Distribution
            match = re.search(DIST_PATTERN, policy, re.IGNORECASE)
            # Ensure the distribution name is stored as lowercase for matching
            dist = match.group(1).lower() if match else "unknown"
            distributions.append(dist)
            
            # 2. Set Policy Name
            if filename_prefix:
                clean_name = f"[{filename_prefix}] {policy}"
            else:
                clean_name = policy
            
            policy_names.append(clean_name)
            
            # 3. Append Metrics
            for metric, value in results.items():
                metrics[metric].append(value)
                
        metrics['__Policy_Names__'] = policy_names
        metrics['__Distributions__'] = distributions
        return metrics

    def load_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Output File(s)", "", "Output Files (*.json *.jsonl)"
        )
        if not file_paths: return

        json_files = [f for f in file_paths if f.endswith('.json')]
        jsonl_files = [f for f in file_paths if f.endswith('.jsonl')]

        # 1. Handle JSONL files (Open separate windows)
        for fpath in jsonl_files:
            win = SimulationResultsWindow(policy_names=['External_Log'], log_path=fpath)
            win.show()
            self.sim_windows.append(win)

        if not json_files:
            if jsonl_files:
                self.text_view.setText(f"Opened {len(jsonl_files)} JSONL file(s) in external windows.")
            return

        try:
            # --- MODIFICATION START ---
            
            # --- Merge Initialization (Keeping current data) ---
            if self.json_data:
                all_policy_names = self.json_data.pop('__Policy_Names__', [])
                all_distributions = self.json_data.pop('__Distributions__', [])
                merged_metrics = defaultdict(list, self.json_data)
                valid_keys_set = set(merged_metrics.keys())
            else:
                merged_metrics = defaultdict(list)
                all_policy_names = []
                all_distributions = []
                valid_keys_set = set()
            
            # 1. Update the persistent path set with the current batch
            for fpath in json_files:
                self._all_loaded_json_paths.append(fpath) # Ensure tracking unique paths

            # 2. Build the summary text from the persistent, unique list
            summary_text = "--- Loaded/Merged Files ---\n"
            for fpath in sorted(list(self._all_loaded_json_paths)):
                summary_text += f"- {fpath}\n"
            
            # 3. Process data for merging
            for fpath in json_files:
                # Only use basename for the policy prefix
                fname_prefix = os.path.basename(fpath)
                
                with open(fpath, 'r') as f:
                    raw_data = json.load(f)

                if isinstance(raw_data, dict) and raw_data and isinstance(next(iter(raw_data.values())), dict):
                    pivoted_data = self._pivot_json_data(raw_data, filename_prefix=fname_prefix)
                else:
                    pivoted_data = raw_data

                # Extend the ALL lists with the new data from the current file
                current_names = pivoted_data.get('__Policy_Names__', [])
                all_policy_names.extend(current_names)
                all_distributions.extend(pivoted_data.get('__Distributions__', ['unknown'] * len(current_names)))
                
                for k, v in pivoted_data.items():
                    if k in ['__Policy_Names__', '__Distributions__']: continue
                    if isinstance(v, list):
                        merged_metrics[k].extend(v)
                        valid_keys_set.add(k)
                        
            # Reconstruct final dataset, including control lists
            self.json_data = dict(merged_metrics)
            self.json_data['__Policy_Names__'] = all_policy_names
            self.json_data['__Distributions__'] = all_distributions
            
            # 4. Final UI updates (Population logic remains the same)
            final_keys = sorted(list(valid_keys_set))
            
            self.y_key_combo.clear()
            self.y_key_combo.addItems(final_keys)

            self.x_key_combo.clear()
            self.x_key_combo.addItem("Policy Names (Default)") 
            self.x_key_combo.addItems(final_keys)

            self.dist_combo.clear()
            unique_dists_found = sorted(list(set(d for d in all_distributions if d != 'unknown')))
            
            # Add all unique distributions found (excluding 'unknown')
            if unique_dists_found:
                self.dist_combo.addItems(unique_dists_found)
            
            self.plot_btn.setEnabled(len(final_keys) > 0)
            
            # Final summary update
            summary_text += f"\nTotal Policies/Datapoints: {len(all_policy_names)}\n"
            summary_text += f"Available Metrics: {', '.join(final_keys)}"
            self.text_view.setText(summary_text)
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process JSON files: {e}")

    def _calculate_pareto_front(self, x_values, y_values):
        """
        Identifies indices of non-dominated solutions (Minimizing X and Minimizing Y).
        """
        points = []
        for i, (xv, yv) in enumerate(zip(x_values, y_values)):
            points.append({'idx': i, 'x': xv, 'y': yv})
            
        pareto_indices = []
        
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i == j: continue
                # Minimization Logic:
                if (other['x'] <= point['x'] and other['y'] <= point['y'] and 
                    (other['x'] < point['x'] or other['y'] < point['y'])):
                    dominated = True
                    break
            
            if not dominated:
                pareto_indices.append(point['idx'])
        
        return pareto_indices

    def plot_json_key(self):
        y_key = self.y_key_combo.currentText()
        x_selection = self.x_key_combo.currentText()
        chart_type = self.chart_type_combo.currentText()
        selected_dist = self.dist_combo.currentText() # Get selected distribution (e.g., 'emp')

        if not y_key or not self.json_data: return
        
        # --- DATA FILTERING ---
        indices_to_plot = []
        all_dists = self.json_data.get('__Distributions__', [])
        
        # The selected_dist will be one of the specific distributions (emp, gamma1, etc.)
        for i, dist in enumerate(all_dists):
            if dist == selected_dist: # Match the specific distribution
                indices_to_plot.append(i)

        if not indices_to_plot:
            QMessageBox.warning(self, "Plot Error", f"No data found for distribution: {selected_dist}")
            return
        
        # Helper function to extract and filter data
        def filter_data(key, indices):
            full_list = self.json_data.get(key, [])
            return [full_list[i] for i in indices]

        y_data_filtered = filter_data(y_key, indices_to_plot)
        policy_names_filtered = filter_data('__Policy_Names__', indices_to_plot)
        
        # ... (rest of the plotting logic)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            is_2d_metric_plot = (x_selection != "Policy Names (Default)") and (x_selection in self.json_data)
            
            if is_2d_metric_plot:
                # --- SCATTER: Metric vs Metric (Filtered) ---
                x_plot = filter_data(x_selection, indices_to_plot)
                y_plot = y_data_filtered
                names_plot = policy_names_filtered

                # Force Scatter plot for Metric vs Metric
                ax.scatter(x_plot, y_plot, c='#3498db', s=80, alpha=0.8, edgecolor='k', zorder=2)
                
                # Annotate
                for i, txt in enumerate(names_plot):
                    ax.annotate(txt, (x_plot[i], y_plot[i]), 
                                xytext=(5, 5), textcoords='offset points', 
                                fontsize=8, alpha=0.7)
                
                ax.set_xlabel(x_selection)
                ax.set_ylabel(y_key)
                ax.set_title(f"{y_key} vs {x_selection} (Dist: {selected_dist})")

                # Pareto Logic
                if self.pareto_check.isChecked():
                    pareto_indices = self._calculate_pareto_front(x_plot, y_plot)
                    if pareto_indices:
                        pareto_pts = sorted([(x_plot[i], y_plot[i]) for i in pareto_indices], key=lambda p: p[0])
                        px, py = zip(*pareto_pts)
                        
                        ax.plot(px, py, '--', color='red', linewidth=2, label='Pareto Front', zorder=1)
                        ax.scatter(px, py, color='red', s=100, facecolors='none', edgecolors='red', zorder=3)
                        ax.legend()

            else:
                # --- STANDARD: Metric vs Policy (Filtered) ---
                y_plot = y_data_filtered
                names_plot = policy_names_filtered
                x_indices = range(len(names_plot))

                if chart_type == "Line Chart":
                    ax.plot(x_indices, y_plot, marker='o', linestyle='-')
                elif chart_type == "Bar Chart":
                    ax.bar(x_indices, y_plot)
                elif chart_type == "Scatter Plot":
                    ax.scatter(x_indices, y_plot)
                elif chart_type == "Area Chart":
                    ax.fill_between(x_indices, y_plot, alpha=0.5)
                    ax.plot(x_indices, y_plot, alpha=0.8)

                ax.set_xticks(x_indices)
                ax.set_xticklabels(names_plot, rotation=45, ha='right', fontsize=8)
                ax.set_xlabel("Policy Name / File")
                ax.set_ylabel(y_key)
                ax.set_title(f"{y_key} Across Policies (Dist: {selected_dist})")

            ax.grid(True, linestyle='--', alpha=0.6)
            self.figure.tight_layout()
            self.canvas.draw()
            self.tabs.setCurrentIndex(1)
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Could not generate plot: {str(e)}")

    def shutdown(self):
        """
        Safely shuts down all managed external windows.
        Each SimulationResultsWindow has its own thread cleanup.
        """
        for win in self.sim_windows:
            if win is not None:
                # Assuming SimulationResultsWindow has a closeEvent/shutdown method that
                # stops its internal workers (chart_thread and file_thread).
                win.close() 
        self.sim_windows = []