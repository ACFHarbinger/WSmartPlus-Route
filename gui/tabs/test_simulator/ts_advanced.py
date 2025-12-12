import multiprocessing as mp
import os

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QHBoxLayout, QDoubleSpinBox,
    QComboBox, QLineEdit, QFormLayout,
    QScrollArea, QVBoxLayout, QPushButton,
    QSpinBox, QLabel, QWidget, QSizePolicy,
    QFileDialog
)
from gui.utils.app_definitions import (
    DISTANCE_MATRIX_METHODS,
    VERTEX_METHODS, EDGE_METHODS
)
from ...styles.globals import START_RED_STYLE, START_GREEN_STYLE
from ...components import ClickableHeaderWidget


class TestSimAdvancedTab(QWidget):
    def __init__(self):
        super().__init__()
        # 1. Setup the main layout for the tab (to hold the scroll area)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 2. Create the inner widget that will contain all the form fields
        content_widget = QWidget()
        
        # 3. Apply the QFormLayout to the inner widget
        form_layout = QFormLayout(content_widget)

        # 4. Create the QScrollArea and configure it
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        
        # Add the scroll area to the main layout of the TestSimAdvancedTab
        main_layout.addWidget(scroll_area)

        # --- System Settings ---
        # 1. Add the section header directly to the inner content's QFormLayout
        form_layout.addRow(QLabel("<b>System Settings</b>"))
        
        # 2. Maximum CPU Cores and Env Vars File (Add directly to QFormLayout)
        self.cpu_cores_input = QSpinBox(value=1, minimum=1, maximum=mp.cpu_count()-1)
        form_layout.addRow("Maximum CPU Cores:", self.cpu_cores_input)

        self.env_file_input = QLineEdit("vars.env")
        # Add browse button
        form_layout.addRow("Environment Variables File:", self._create_browser_layout(self.env_file_input))
        
        # 3. Boolean flags
        flags_container = QWidget()
        system_flags_layout = QHBoxLayout(flags_container) # Use flags_container as parent
        system_flags_layout.setContentsMargins(0, 0, 0, 0) # Tidy up spacing

        self.server_run_check = QPushButton("Remote Server Execution")
        self.server_run_check.setCheckable(True)
        self.server_run_check.setChecked(False)
        self.server_run_check.setStyleSheet(START_RED_STYLE)

        self.no_progress_check = QPushButton("Progress Bar")
        self.no_progress_check.setCheckable(True)
        self.no_progress_check.setChecked(True)
        self.no_progress_check.setStyleSheet(START_GREEN_STYLE)

        self.resume_check = QPushButton("Resume Testing")
        self.resume_check.setCheckable(True)
        self.resume_check.setChecked(False)
        self.resume_check.setStyleSheet(START_RED_STYLE)
        
        # Add widgets to the horizontal layout (using addWidget)
        system_flags_layout.addWidget(self.server_run_check)
        system_flags_layout.addWidget(self.no_progress_check)
        system_flags_layout.addWidget(self.resume_check)
        
        # 4. Add the entire horizontal layout as a single row to the QFormLayout
        form_layout.addRow(QLabel("Flags:"), flags_container)

        # --- Edge/Vertex Args ---
        form_layout.addRow(QLabel("<b>Edge/Vertex Setup</b>"))
        self.vertex_method_combo = QComboBox(currentText="Min-Max Normalization")
        self.vertex_method_combo.addItems(VERTEX_METHODS.keys())
        form_layout.addRow("Vertex Method:", self.vertex_method_combo)
        self.distance_method_combo = QComboBox(currentText='Google Maps (GMaps)')
        self.distance_method_combo.addItems(DISTANCE_MATRIX_METHODS.keys())
        form_layout.addRow("Distance Method:", self.distance_method_combo)
        
        # Edge Threshold and Method
        self.edge_threshold_input = QDoubleSpinBox()
        self.edge_threshold_input.setRange(0.0, 1.0)
        self.edge_threshold_input.setSingleStep(0.01) # Step by 0.01 (1%)
        self.edge_threshold_input.setDecimals(2)      # Show 2 decimal places
        self.edge_threshold_input.setValue(1.0)      # Set a default value
        form_layout.addRow("Edge Threshold (0.0 to 1.0):", self.edge_threshold_input)
        
        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItems(EDGE_METHODS.keys())
        self.edge_method_combo.setCurrentIndex(0)
        form_layout.addRow("Edge Method:", self.edge_method_combo)

        # --- Key/License Files (Custom Header) ---
        # 1. Create a container widget for the header using the custom clickable class
        self.key_license_files_header_widget = ClickableHeaderWidget(self._toggle_key_license_files)
        self.key_license_files_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        klf_header_layout = QHBoxLayout(self.key_license_files_header_widget)
        klf_header_layout.setContentsMargins(0, 0, 0, 0)
        klf_header_layout.setSpacing(5)

        # 2. The main text (Standard QLabel)
        self.key_license_files_label = QLabel("<b>Key/License Files</b>")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.key_license_files_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.key_license_files_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button (only the +/- sign)
        self.key_license_files_toggle_button = QPushButton("+")
        self.key_license_files_toggle_button.setFlat(True)
        self.key_license_files_toggle_button.setFixedSize(QSize(20, 20))
        self.key_license_files_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.key_license_files_toggle_button.clicked.connect(self._toggle_key_license_files)

        # 4. Add components to the header layout
        klf_header_layout.addWidget(self.key_license_files_label)
        klf_header_layout.addStretch()
        klf_header_layout.addWidget(self.key_license_files_toggle_button)
        
        # 5. Add the header widget to the main form layout, making it span the row
        form_layout.addRow(self.key_license_files_header_widget)

        # 6. Create a container for the collapsible content
        self.key_license_files_container = QWidget()
        key_license_files_layout = QFormLayout(self.key_license_files_container)
        key_license_files_layout.setContentsMargins(0, 0, 0, 0)

        # 7. Add widgets to the container's layout
        self.gplic_file_input = QLineEdit()
        key_license_files_layout.addRow("Gurobi License File:", self._create_browser_layout(self.gplic_file_input))
        
        self.hexlic_file_input = QLineEdit()
        key_license_files_layout.addRow("Hexaly License File:", self._create_browser_layout(self.hexlic_file_input))
        
        self.gapik_file_input = QLineEdit()
        key_license_files_layout.addRow("Google API Key File:", self._create_browser_layout(self.gapik_file_input))
        
        self.symkey_name_input = QLineEdit()
        key_license_files_layout.addRow("Cryptographic Key Name:", self.symkey_name_input)
        
        # 8. Add the content container to the main form layout
        form_layout.addWidget(self.key_license_files_container)

        # 9. Initialize state: hidden
        self.is_key_license_files_visible = False
        self.key_license_files_container.hide()
    
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

    def _toggle_key_license_files(self):
        """Toggles the visibility of the Key/License Files input fields and updates the +/- sign."""
        if self.is_key_license_files_visible: 
            self.key_license_files_container.hide()
            self.key_license_files_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.key_license_files_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.key_license_files_container.show()
            self.key_license_files_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.key_license_files_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_key_license_files_visible = not self.is_key_license_files_visible

    def get_params(self):
        params = {
            # Advanced
            "cpu_cores": self.cpu_cores_input.value(),
            "vertex_method": VERTEX_METHODS.get(self.vertex_method_combo.currentText(), ""),
            "distance_method": DISTANCE_MATRIX_METHODS.get(self.distance_method_combo.currentText(), ""),
            
            # *** Retrieve value() from QDoubleSpinBox (returns a float) ***
            "edge_threshold": self.edge_threshold_input.value(),
            # **********************************************************************
            
            "env_file": self.env_file_input.text().strip(),
            
            # Boolean Flags
            "no_progress_bar": self.no_progress_check.isChecked(),
            "server_run": self.server_run_check.isChecked(), 
            "resume": self.resume_check.isChecked(), 
            
            # Key/License Files (Direct text inputs)
            "hexlic_file": self.hexlic_file_input.text().strip(),
        }
        
        # Optional file paths
        if self.edge_method_combo.currentText().strip(): params["edge_method"] = EDGE_METHODS.get(self.edge_method_combo.currentText().strip(), "")
        if self.gplic_file_input.text().strip(): params["gplic_file"] = self.gplic_file_input.text().strip()
        if self.symkey_name_input.text().strip(): params["symkey_name"] = self.symkey_name_input.text().strip()
        if self.gapik_file_input.text().strip(): params["gapik_file"] = self.gapik_file_input.text().strip()

        return params