from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class VisualizationWidget(QTabWidget):
    def __init__(self, parent=None):
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
        self.text_view.clear()
        self.figure.clear()
        self.canvas.draw()
