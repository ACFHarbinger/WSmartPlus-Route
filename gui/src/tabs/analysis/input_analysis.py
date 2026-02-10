"""Input analysis and dataset exploration tab."""

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...helpers import DataLoadWorker
from .pandas_model import PandasModel


class InputAnalysisTab(QWidget):
    """Tab for analyzing input data and visualizing distributions."""

    # Signal to request the worker to start loading
    load_request = Signal(str)

    def __init__(self):
        """Initialize the InputAnalysisTab."""
        super().__init__()
        # self.df = None # OLD: single DF
        self.dfs = {}  # NEW: Store multiple DFs indexed by slice name
        self.current_slice_key = None  # NEW: key of the currently viewed DF

        # --- Thread Setup ---
        self.worker_thread = QThread()
        self.data_worker = DataLoadWorker()
        self.data_worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        # Connect signals for loading request and worker response
        self.load_request.connect(self.data_worker.load_data_file)
        self.data_worker.data_loaded.connect(self._handle_successful_load)
        self.data_worker.error_occurred.connect(self._handle_load_error)

        layout = QVBoxLayout(self)

        # Controls
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Data File (CSV/XLSX/PKL)")
        self.load_btn.clicked.connect(self.load_file)
        control_layout.addWidget(self.load_btn)

        # --- NEW CONTROL: Slice/Table Selector ---
        self.slice_selector = QComboBox()
        self.slice_selector.setMinimumWidth(150)
        self.slice_selector.setPlaceholderText("Select Table Slice")
        self.slice_selector.currentIndexChanged.connect(self._switch_current_df)
        control_layout.addWidget(QLabel("Table:"))
        control_layout.addWidget(self.slice_selector)

        # --- NEW CONTROL: Chart Type Selector ---
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart", "Heatmap"])
        self.chart_type_combo.currentIndexChanged.connect(self._update_chart_controls)
        control_layout.addWidget(QLabel("Type:"))
        control_layout.addWidget(self.chart_type_combo)

        self.x_axis_combo = QComboBox()
        self.x_axis_combo.setPlaceholderText("Select X Axis")
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.setPlaceholderText("Select Y Axis")
        self.plot_btn = QPushButton("Plot Chart")
        self.plot_btn.clicked.connect(self.plot_data)
        self.plot_btn.setEnabled(False)

        control_layout.addWidget(QLabel("X:"))
        control_layout.addWidget(self.x_axis_combo)
        control_layout.addWidget(QLabel("Y:"))
        control_layout.addWidget(self.y_axis_combo)
        control_layout.addWidget(self.plot_btn)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Content Area (Table vs Chart)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Table Tab
        self.table_view = QTableView()
        self.tabs.addTab(self.table_view, "Raw Data")

        # Chart Tab
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_widget)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.chart_layout.addWidget(self.canvas)
        self.tabs.addTab(self.chart_widget, "Visualization")

    def load_file(self):
        """Open a file dialog to load data for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.csv *.xlsx *.pkl)")
        if not file_path:
            return

        self.load_btn.setEnabled(False)
        self.slice_selector.clear()

        # Emit signal to trigger worker thread loading
        self.load_request.emit(file_path)

    @Slot(object)  # object is a list of (name, thread_safe_dict) OR just thread_safe_dicts (backward compat)
    def _handle_successful_load(self, thread_safe_list):
        """
        Processes successfully loaded data from the worker thread.

        Args:
            thread_safe_list (list): List of (name, data_dict) tuples from DataLoadWorker.
        """
        self.load_btn.setEnabled(True)
        self.dfs = {}
        self.slice_selector.blockSignals(True)
        self.slice_selector.clear()

        for i, item in enumerate(thread_safe_list):
            # Check if item is (name, data) tuple or just data
            if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
                name, data_dict = item
            else:
                name = None
                data_dict = item

            df = pd.DataFrame(data_dict)

            # Construct Display Key
            key = f"{name} ({df.shape[0]}x{df.shape[1]})" if name else f"Table {i + 1} ({df.shape[0]}x{df.shape[1]})"

            self.dfs[key] = df
            self.slice_selector.addItem(key)

        self.plot_btn.setEnabled(len(thread_safe_list) > 0)

        self.slice_selector.blockSignals(False)

        if self.dfs:
            self.slice_selector.setCurrentIndex(0)
            self._switch_current_df()
        else:
            self.table_view.setModel(None)
            self.x_axis_combo.clear()
            self.y_axis_combo.clear()

    @Slot()
    def _switch_current_df(self):
        """
        Switches the current DataFrame being displayed in the table and used for plotting.
        Triggered when the slice_selector index changes.
        """
        key = self.slice_selector.currentText()
        if not key or key not in self.dfs:
            self.current_slice_key = None
            self.table_view.setModel(None)
            self.x_axis_combo.clear()
            self.y_axis_combo.clear()
            return

        self.current_slice_key = key
        df = self.dfs[key]

        # Populate Table
        model = PandasModel(df)
        self.table_view.setModel(model)

        # Populate Combos
        columns = df.columns.tolist()
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()

        # --- CRITICAL FIX: Explicitly convert columns to strings ---
        # Integer column names (from arrays) fail to add to QComboBox otherwise
        str_columns = [str(c) for c in columns]
        self.x_axis_combo.addItems(str_columns)
        self.y_axis_combo.addItems(str_columns)
        # -----------------------------------------------------------

        self.tabs.setCurrentIndex(0)

    @Slot()
    def _update_chart_controls(self):
        """Disables X/Y combos if Heatmap is selected."""
        chart_type = self.chart_type_combo.currentText()
        is_heatmap = chart_type == "Heatmap"
        self.x_axis_combo.setDisabled(is_heatmap)
        self.y_axis_combo.setDisabled(is_heatmap)

    @Slot(str)
    def _handle_load_error(self, message):
        """
        Handles data loading errors by re-enabling controls and showing a message box.

        Args:
            message (str): The error message.
        """
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error Loading File", message)

    def plot_data(self):
        """Plot the selected data using the chosen visualization."""
        # Use the currently selected DataFrame
        if self.current_slice_key is None:
            return
        df = self.dfs.get(self.current_slice_key)
        if df is None:
            return

        chart_type = self.chart_type_combo.currentText()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            # --- Logic for Heatmap (Visualizes the entire numeric matrix) ---
            if chart_type == "Heatmap":
                numeric_df = df.select_dtypes(include=["number"])

                if numeric_df.empty:
                    raise ValueError("Current table has no numeric data to display as Heatmap.")

                cax = ax.imshow(numeric_df.values, aspect="auto", cmap="viridis", origin="lower")
                self.figure.colorbar(cax, ax=ax)

                ax.set_title(f"Heatmap (Slice: {self.current_slice_key})")
                ax.set_ylabel("Row Index")
                ax.set_xlabel("Columns")

                if len(numeric_df.columns) < 50:
                    ax.set_xticks(range(len(numeric_df.columns)))
                    ax.set_xticklabels(numeric_df.columns, rotation=90)

            else:
                x_col = self.x_axis_combo.currentText()
                y_col = self.y_axis_combo.currentText()

                if not x_col or not y_col:
                    # --- ADDED: Error message if axes are missing (fixes silent failure) ---
                    QMessageBox.warning(
                        self,
                        "Plot Error",
                        "Please select both X and Y axes for this chart type.",
                    )
                    return

                # Check if columns exist (sanity check)
                if x_col not in df.columns.astype(str) or y_col not in df.columns.astype(str):
                    # Handle mixed types gracefully by checking against stringified columns
                    pass

                # Prepare data
                # We need to access columns using the original types (e.g. int) if the DF has them,
                # but our combo box has strings.
                try:
                    # Try direct access (works if columns are strings)
                    x_data = df[x_col]
                    y_data = df[y_col]
                except KeyError:
                    # Fallback: Try converting combo text to int (if columns are integers)
                    try:
                        x_data = df[int(x_col)]
                        y_data = df[int(y_col)]
                    except (ValueError, KeyError):
                        raise ValueError(f"Could not find columns '{x_col}' or '{y_col}' in data.")

                if chart_type in ["Line Chart", "Area Chart"]:
                    # Create a temporary sorted DF for plotting line charts properly
                    temp_df = pd.DataFrame({"x": x_data, "y": y_data}).sort_values(by="x")
                    x_plot = temp_df["x"]
                    y_plot = temp_df["y"]
                else:
                    x_plot = x_data
                    y_plot = y_data

                # Plot based on selection
                if chart_type == "Line Chart":
                    ax.plot(x_plot, y_plot, marker="o", linestyle="-")
                elif chart_type == "Bar Chart":
                    ax.bar(x_plot, y_plot)
                elif chart_type == "Scatter Plot":
                    ax.scatter(x_plot, y_plot)
                elif chart_type == "Area Chart":
                    ax.fill_between(x_plot, y_plot, alpha=0.4)
                    ax.plot(x_plot, y_plot, alpha=0.9)

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{chart_type}: {y_col} vs {x_col}")
                ax.grid(True)

            self.canvas.draw()
            self.tabs.setCurrentIndex(1)  # Switch to chart

        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Could not plot data: {str(e)}")

    def get_params(self):
        """Get the parameters for this tab."""
        return {}  # No CLI params needed

    # Add a cleanup method for the thread
    def closeEvent(self, event):
        """Safely stops the worker thread when the widget is closed."""
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)

    # You might also want a dedicated cleanup method if closeEvent isn't reliably called:
    def shutdown(self):
        """Shutdown the background worker threads."""
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
