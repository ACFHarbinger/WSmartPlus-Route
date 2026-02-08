"""
Visualization widgets for Output Analysis.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class VisualizationWidget(QTabWidget):
    """
    Tabbed widget for displaying data summaries and Matplotlib visualizations.
    """

    def __init__(self, parent=None):
        """
        Initialize the visualization widget with summary and chart tabs.
        """
        super().__init__(parent)

        # Tab 1: Text Summary
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.addTab(self.text_view, "Merged Data Summary")

        # Tab 2: Visualization
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_widget)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.chart_layout.addWidget(self.canvas)
        self.addTab(self.chart_widget, "Visualization")

    def clear(self):
        """
        Clear the text summary and reset the Matplotlib figure.
        """
        self.text_view.clear()
        self.figure.clear()
        self.canvas.draw()
