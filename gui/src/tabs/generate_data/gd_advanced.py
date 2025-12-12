import os
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QSpinBox, QComboBox, QLabel,
    QLineEdit, QFormLayout, QWidget, 
    QPushButton, QSizePolicy, QHBoxLayout,
    QFileDialog # <-- Added
)
from gui.src.utils.app_definitions import COUNTY_AREAS, VERTEX_METHODS
from ...components import ClickableHeaderWidget


class GenDataAdvancedTab(QWidget):        
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        
        # --- Epoch Settings ---
        layout.addRow(QLabel("<b>Epoch Settings</b>"))

        # 1. --n_epochs
        self.n_epochs_input = QSpinBox()
        self.n_epochs_input.setRange(1, 1000)
        self.n_epochs_input.setValue(1)
        layout.addRow("Number of Epochs:", self.n_epochs_input)

        # 2. --epoch_start
        self.epoch_start_input = QSpinBox()
        self.epoch_start_input.setRange(0, 1000)
        self.epoch_start_input.setValue(0)
        layout.addRow("Start Epoch:", self.epoch_start_input)
        
        # 3. --vertex_method
        self.vertex_method_combo = QComboBox()
        self.vertex_method_combo.addItems(VERTEX_METHODS.keys())
        layout.addRow("Vertex Method:", self.vertex_method_combo)

        # --- Area Specific (Custom Header) --- 
        # 1. Create a container widget for the header using the custom clickable class
        self.area_specific_header_widget = ClickableHeaderWidget(self._toggle_area_specific)
        self.area_specific_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        as_header_layout = QHBoxLayout(self.area_specific_header_widget)
        as_header_layout.setContentsMargins(0, 0, 0, 0)
        as_header_layout.setSpacing(5)

        # 2. The main text (Standard QLabel)
        self.area_specific_label = QLabel("<b>Area Specific</b>")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.area_specific_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.area_specific_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button (only the +/- sign)
        self.area_specific_toggle_button = QPushButton("+")
        self.area_specific_toggle_button.setFlat(True)
        self.area_specific_toggle_button.setFixedSize(QSize(20, 20))
        self.area_specific_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        # The button's signal still fires and calls the toggle function
        self.area_specific_toggle_button.clicked.connect(self._toggle_area_specific)
        
        # 4. Add components to the header layout
        as_header_layout.addWidget(self.area_specific_label)
        as_header_layout.addStretch()
        as_header_layout.addWidget(self.area_specific_toggle_button)
        
        # 5. Add the header widget to the main layout, making it span the row
        layout.addRow(self.area_specific_header_widget)

        # 6. Create a container for the collapsible content
        self.area_specific_container = QWidget()
        area_specific_layout = QFormLayout(self.area_specific_container)
        area_specific_layout.setContentsMargins(0, 0, 0, 0)

        # 7. Add widgets to the container's layout
        # 6. --area
        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("Rio Maior")
        area_specific_layout.addRow("County Area:", self.area_input)
        
        # 7. --waste_type
        self.waste_type_input = QLineEdit()
        self.waste_type_input.setPlaceholderText("Plastic")
        area_specific_layout.addRow("Waste Type:", self.waste_type_input)

        # 8. Add the content container to the main layout
        layout.addWidget(self.area_specific_container)

        # 9. Initialize state: hidden
        self.is_area_visible = False
        self.area_specific_container.hide()

        # --- Focus Graphs (Custom Header) ---
        # 1. Create a container widget for the header using the custom clickable class
        self.focus_graphs_header_widget = ClickableHeaderWidget(self._toggle_focus_graphs)
        self.focus_graphs_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        fg_header_layout = QHBoxLayout(self.focus_graphs_header_widget)
        fg_header_layout.setContentsMargins(0, 0, 0, 0)
        fg_header_layout.setSpacing(5)

        # 2. The main text (Standard QLabel)
        self.focus_graphs_label = QLabel("<b>Focus Graphs</b>")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.focus_graphs_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.focus_graphs_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button (only the +/- sign)
        self.focus_graphs_toggle_button = QPushButton("+")
        self.focus_graphs_toggle_button.setFlat(True)
        self.focus_graphs_toggle_button.setFixedSize(QSize(20, 20))
        self.focus_graphs_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.focus_graphs_toggle_button.clicked.connect(self._toggle_focus_graphs)

        # 4. Add components to the header layout
        fg_header_layout.addWidget(self.focus_graphs_label)
        fg_header_layout.addStretch()
        fg_header_layout.addWidget(self.focus_graphs_toggle_button)
        
        # 5. Add the header widget to the main layout, making it span the row
        layout.addRow(self.focus_graphs_header_widget)

        # 6. Create a container for the collapsible content
        self.focus_graphs_container = QWidget()
        focus_graphs_layout = QFormLayout(self.focus_graphs_container)
        focus_graphs_layout.setContentsMargins(0, 0, 0, 0)

        # 7. Add widgets to the container's layout
        # 6. --focus_graph
        self.focus_graphs_input = QLineEdit()
        self.focus_graphs_input.setPlaceholderText("Paths to focus graph files")
        focus_graphs_layout.addRow("Focus Graph Paths:", self._create_browser_layout(self.focus_graphs_input))
        
        # 7. --focus_size
        self.focus_size_input = QSpinBox()
        self.focus_size_input.setRange(0, 100000)
        self.focus_size_input.setValue(0)
        focus_graphs_layout.addRow("Number per Focus Graph:", self.focus_size_input)
        
        # 8. Add the content container to the main layout
        layout.addWidget(self.focus_graphs_container)

        # 9. Initialize state: hidden
        self.is_graphs_visible = False
        self.focus_graphs_container.hide()

    def _create_browser_layout(self, line_edit, is_dir=False):
        """Helper to create a layout with a browse button."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_path(line_edit, is_dir))
        layout.addWidget(btn)
        return layout

    def _browse_path(self, line_edit, is_dir):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd())
        
        if path:
            line_edit.setText(path)

    def _toggle_area_specific(self):
        """Toggles the visibility of the Area Specific input fields and updates the +/- sign."""
        if self.is_area_visible: 
            self.area_specific_container.hide()
            self.area_specific_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.area_specific_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.area_specific_container.show()
            self.area_specific_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.area_specific_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_area_visible = not self.is_area_visible
    
    def _toggle_focus_graphs(self):
        """Toggles the visibility of the Focus Graphs input fields and updates the +/- sign."""
        if self.is_graphs_visible:
            self.focus_graphs_container.hide()
            self.focus_graphs_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.focus_graphs_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.focus_graphs_container.show()
            self.focus_graphs_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.focus_graphs_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_graphs_visible = not self.is_graphs_visible

    def get_params(self):
        params = {}
        # Mandatory fields
        params["n_epochs"] = self.n_epochs_input.value()
        params["epoch_start"] = self.epoch_start_input.value()
        params["vertex_method"] = VERTEX_METHODS.get(self.vertex_method_combo.currentText(), "")
        # Optional fields        
        if self.area_input.text():
            params["area"] = COUNTY_AREAS.get(self.area_input.text(), "")
            params["waste_type"] = self.waste_type_input.text().strip().lower()

        if self.focus_graphs_input.text().strip():
            params["focus_size"] = self.focus_size_input.value()
            params["focus_graph"] = self.focus_graphs_input.text().strip()
        return params