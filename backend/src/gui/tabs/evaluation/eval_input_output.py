from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFormLayout, QVBoxLayout,
    QSizePolicy, QPushButton,
    QScrollArea, QLabel, QWidget,
    QLineEdit, QCheckBox, QHBoxLayout,
)
from ..components import ClickableHeaderWidget


class EvalIOTab(QWidget):
    """
    Tab for input datasets, output files, and model path.
    """
    def __init__(self):
        super().__init__()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)

        form_layout.addRow(QLabel("<b>Input Files</b>"))

        # --datasets
        self.datasets_input = QLineEdit()
        self.datasets_input.setPlaceholderText("e.g., dataset_a.pkl dataset_b.pkl (space separated)")
        form_layout.addRow("Dataset Files:", self.datasets_input)
        
        # --model
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Path to trained model checkpoint")
        form_layout.addRow("Model Checkpoint:", self.model_input)
        
        form_layout.addRow(QLabel("<b>Output Settings</b>"))

        # --results_dir
        self.results_dir_input = QLineEdit('results')
        form_layout.addRow("Results Directory:", self.results_dir_input)

        # --- Output File (Custom Header) ---
        # Create a container widget for the header using the custom clickable class
        self.output_file_header_widget = ClickableHeaderWidget(self._toggle_output_file)
        self.output_file_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )
        
        of_header_layout = QHBoxLayout(self.output_file_header_widget)
        of_header_layout.setContentsMargins(0, 0, 0, 0)
        of_header_layout.setSpacing(5)

        # The main text (Standard QLabel)
        self.output_file_label = QLabel("<b>Output File</b>")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.output_file_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.output_file_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # The clickable toggle button (only the +/- sign)
        self.output_file_toggle_button = QPushButton("+")
        self.output_file_toggle_button.setFlat(True)
        self.output_file_toggle_button.setFixedSize(QSize(20, 20))
        self.output_file_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        # The button's signal still fires and calls the toggle function
        self.output_file_toggle_button.clicked.connect(self._toggle_output_file)
        
        # Add components to the header layout
        of_header_layout.addWidget(self.output_file_label)
        of_header_layout.addStretch()
        of_header_layout.addWidget(self.output_file_toggle_button)
        
        # Add the header widget to the main layout, making it span the row
        form_layout.addRow(self.output_file_header_widget)

        # Create a container for the collapsible content
        self.output_file_container = QWidget()
        output_file_layout = QFormLayout(self.output_file_container)
        output_file_layout.setContentsMargins(0, 0, 0, 0)

        # Add widgets to the container's layout
        # # -o (output file name)
        self.output_file_input = QLineEdit()
        self.output_file_input.setPlaceholderText("Results file name")
        output_file_layout.addRow("Output File Name:", self.output_file_input)

        # -f (overwrite flag)
        self.overwrite_check = QCheckBox("Overwrite existing results file")
        self.overwrite_check.setChecked(False)
        output_file_layout.addRow(self.overwrite_check)

        form_layout.addWidget(self.output_file_container)

        # Initialize state: hidden
        self.is_output_file_visible = False
        self.output_file_container.hide()

        scroll_area.setWidget(content)
        QVBoxLayout(self).addWidget(scroll_area)
    
    def _toggle_output_file(self):
        """Toggles the visibility of the Output File input fields and updates the +/- sign."""
        if self.is_output_file_visible: 
            self.output_file_container.hide()
            self.output_file_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.output_file_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.output_file_container.show()
            self.output_file_toggle_button.setText("-")

            # Remove the border from the QLabel when expanded.
            self.output_file_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_output_file_visible = not self.is_output_file_visible

    def get_params(self):
        params = {
            "datasets": self.datasets_input.text().strip() or None,
            "model": self.model_input.text().strip() or None,
            "o": self.output_file_input.text().strip() or None,
            "results_dir": self.results_dir_input.text().strip(),
            "f": self.overwrite_check.isChecked(),
        }
        return {k: v for k, v in params.items() if v is not None}
