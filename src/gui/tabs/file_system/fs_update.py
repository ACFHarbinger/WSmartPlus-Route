import os
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QSizePolicy, QFormLayout,
    QWidget, QLineEdit, QLabel,
    QVBoxLayout, QGroupBox, QComboBox,
    QPushButton, QHBoxLayout, QScrollArea,
    QFileDialog # <-- Added
)
from ...styles import START_RED_STYLE
from ...app_definitions import OPERATION_MAP, FUNCTION_MAP
from ...components import ClickableHeaderWidget


class FileSystemUpdateTab(QWidget):
    """
    GUI tab for the 'update' command, focused on modifying contents 
    or attributes of file system entries. The contents are wrapped in a 
    QScrollArea for scrollability.
    """
    def __init__(self):
        super().__init__()   
        # 1. Create a container widget for all content
        scrollable_content = QWidget()
        # 2. Create the layout that will hold all the settings and be scrollable
        content_layout = QVBoxLayout(scrollable_content)
        
        # Add the title
        content_layout.addWidget(QLabel("<h2>Update File System Entries</h2>"))

        # --- File Targeting Group (target_entry, filename_pattern, output_key) ---
        target_group = QGroupBox()
        target_layout = QFormLayout(target_group)
        
        # --target_entry (str)
        self.target_entry_input = QLineEdit()
        self.target_entry_input.setPlaceholderText("e.g., /path/to/directory or /path/to/single_file.json")
        target_layout.addRow("Target Entry Path:", self._create_browser_layout(self.target_entry_input, is_dir=True))

        # --output_key (str, default=None) - This is the OUTPUT key or the SINGLE input key
        self.output_key_input = QLineEdit()
        self.output_key_input.setPlaceholderText("e.g., 'overflows' (or the calculated output key)")
        target_layout.addRow("Output Field Key:", self.output_key_input)

        # --filename_pattern (str, default=None)
        target_layout.addRow(QLabel('<span style="font-weight: 600;">Target Directory Settings</span>'))
        self.filename_pattern_input = QLineEdit()
        self.filename_pattern_input.setPlaceholderText("e.g., 'log_*.json'")
        target_layout.addRow("Glob Filename Pattern:", self.filename_pattern_input)
        
        # --- Separator and Preview Checkbox ---
        target_layout.addRow(QLabel("<hr>"))
        
        # --update_preview (action='store_true' -> default False)
        self.preview_check = QPushButton("Preview Update")
        self.preview_check.setCheckable(True)
        self.preview_check.setChecked(False)
        self.preview_check.setStyleSheet(START_RED_STYLE)
        target_layout.addRow("Verify changes before updating:", self.preview_check)
        
        content_layout.addWidget(target_group)

        # --- Update Logic Group (update_operation, update_value, update_preview) ---
        logic_group = QGroupBox()
        logic_layout = QFormLayout(logic_group)
        logic_layout.addRow(QLabel('<span style="font-weight: 600;">Update Logic Settings</span>'))

        # --- Inplace Update Parameters (Custom Header) ---
        # 2.1 Create a container widget for the header using the custom clickable class
        self.inplace_update_header_widget = ClickableHeaderWidget(self._toggle_inplace_update)
        self.inplace_update_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        iu_header_layout = QHBoxLayout(self.inplace_update_header_widget)
        iu_header_layout.setContentsMargins(0, 0, 0, 0)
        iu_header_layout.setSpacing(5)

        # 2.2 The main text (Standard QLabel)
        self.inplace_update_label = QLabel('<span style="font-weight: 600;">Inplace Update Parameters</span>')

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.inplace_update_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.inplace_update_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 2.3 The clickable toggle button (only the +/- sign)
        self.inplace_update_toggle_button = QPushButton("+")
        self.inplace_update_toggle_button.setFlat(True)
        self.inplace_update_toggle_button.setFixedSize(QSize(20, 20))
        self.inplace_update_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.inplace_update_toggle_button.clicked.connect(self._toggle_inplace_update)

        # 2.4 Add components to the header layout
        iu_header_layout.addWidget(self.inplace_update_label)
        iu_header_layout.addStretch()
        iu_header_layout.addWidget(self.inplace_update_toggle_button)
        
        # 2.5 Add the header widget to the logic layout
        logic_layout.addRow(self.inplace_update_header_widget)

        # 2.6 Create a container for the collapsible content
        self.inplace_update_container = QWidget()
        inplace_update_layout = QFormLayout(self.inplace_update_container)
        inplace_update_layout.setContentsMargins(0, 0, 0, 0)

        # 2.7 Add widgets to the container's layout
        # --update_operation (default=None)
        self.update_operation_combo = QComboBox()
        self.update_operation_combo.addItems(OPERATION_MAP.keys())
        self.update_operation_combo.setCurrentIndex(-1) # Start with no selection
        self.update_operation_combo.setPlaceholderText("Select Operation")
        inplace_update_layout.addRow("Update Operation:", self.update_operation_combo)

        # --update_value (float, default=0.0)
        self.update_value_input = QLineEdit("0.0")
        self.update_value_input.setPlaceholderText("Enter a float or string value")
        inplace_update_layout.addRow("Update Value (for single-input):", self.update_value_input)
        
        # --- Input Keys for Two-Input Calculation ---
        # Container for the two horizontal input boxes
        input_keys_h_layout = QHBoxLayout()
        
        self.input_key_1_input = QLineEdit()
        self.input_key_1_input.setPlaceholderText("Key 1 (e.g., 'total_miles')")
        input_keys_h_layout.addWidget(self.input_key_1_input)

        self.input_key_2_input = QLineEdit()
        self.input_key_2_input.setPlaceholderText("Key 2 (e.g., 'cost_per_mile')")
        input_keys_h_layout.addWidget(self.input_key_2_input)

        # Add the horizontal layout to the form layout with a label
        inplace_update_layout.addRow("Input Keys (for two-input calculation):", input_keys_h_layout)

        # 2.8 Add the content container to the logic layout
        logic_layout.addRow(self.inplace_update_container)

        # 2.9 Initialize state: hidden
        self.is_inplace_update_visible = False
        self.inplace_update_container.hide()

        # --- Statistics Update Parameters (Custom Header) ---
        # 3.1 Create a container widget for the header using the custom clickable class
        self.stats_update_header_widget = ClickableHeaderWidget(self._toggle_stats_update)
        self.stats_update_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        su_header_layout = QHBoxLayout(self.stats_update_header_widget)
        su_header_layout.setContentsMargins(0, 0, 0, 0)
        su_header_layout.setSpacing(5)

        # 3.2 The main text (Standard QLabel)
        self.stats_update_label = QLabel('<span style="font-weight: 600;">Statistics Update Parameters</span>')

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.stats_update_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.stats_update_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3.3 The clickable toggle button (only the +/- sign)
        self.stats_update_toggle_button = QPushButton("+")
        self.stats_update_toggle_button.setFlat(True)
        self.stats_update_toggle_button.setFixedSize(QSize(20, 20))
        self.stats_update_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.stats_update_toggle_button.clicked.connect(self._toggle_stats_update)

        # 3.4 Add components to the header layout
        su_header_layout.addWidget(self.stats_update_label)
        su_header_layout.addStretch()
        su_header_layout.addWidget(self.stats_update_toggle_button)
        
        # 3.5 Add the header widget to the logic layout
        logic_layout.addRow(self.stats_update_header_widget)

        # 3.6 Create a container for the collapsible content
        self.stats_update_container = QWidget()
        stats_update_layout = QFormLayout(self.stats_update_container)
        stats_update_layout.setContentsMargins(0, 0, 0, 0)

        # 3.7 Add widgets to the container's layout
        # --stats_function (default=None)
        self.update_function_combo = QComboBox()
        self.update_function_combo.addItems(FUNCTION_MAP.keys())
        self.update_function_combo.addItem('')
        self.update_function_combo.setCurrentIndex(-1) # Start with no selection
        self.update_function_combo.setPlaceholderText("Select Statistics Function")
        stats_update_layout.addRow("Update Function:", self.update_function_combo)

        # --output_filename (str, default=None)
        self.output_filename_input = QLineEdit()
        self.output_filename_input.setPlaceholderText("Name of output file (e.g., log_mean.json)")
        stats_update_layout.addRow("Output Filename:", self.output_filename_input)

        # 3.8 Add the content container to the logic layout
        logic_layout.addRow(self.stats_update_container)

        # 3.9 Initialize state: hidden
        self.is_stats_update_visible = False
        self.stats_update_container.hide()

        content_layout.addWidget(logic_group)
        content_layout.addStretch() # Push content to the top
        
        # 4. Create the QScrollArea and set the content widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        scroll_area.setWidget(scrollable_content)

        # 5. Set the final layout of the tab (self)
        tab_layout = QVBoxLayout(self)
        tab_layout.addWidget(scroll_area)

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

    def _toggle_inplace_update(self):
        """Toggles the visibility of the Inplace Update input fields and updates the +/- sign."""
        if self.is_inplace_update_visible: 
            self.inplace_update_container.hide()
            self.inplace_update_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.inplace_update_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.inplace_update_container.show()
            self.inplace_update_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.inplace_update_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_inplace_update_visible = not self.is_inplace_update_visible
    
    def _toggle_stats_update(self):
        """Toggles the visibility of the Statistics Update input fields and updates the +/- sign."""
        if self.is_stats_update_visible: 
            self.stats_update_container.hide()
            self.stats_update_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.stats_update_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.stats_update_container.show()
            self.stats_update_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.stats_update_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_stats_update_visible = not self.is_stats_update_visible

    def get_params(self):
        """Extracts settings into a dictionary mimicking argparse output."""
        # 1. Collect required and optional string parameters
        params = {
            "target_entry": self.target_entry_input.text().strip(),
            "filename_pattern": self.filename_pattern_input.text().strip() or None,
            "output_key": self.output_key_input.text().strip() or None,
        }

        # 2. Handle --update_preview (action='store_false')
        # If the button is UNCHECKED (False), the user wants to PERFORM the update (update_preview=False).
        if not self.preview_check.isChecked():
            params['update_preview'] = False

        # 3. Handle Inplace Update parameters
        if self.is_inplace_update_visible and self.update_operation_combo.currentText().strip():
            # 3.1a Get update operation (must be one of the choices)
            selected_function = self.update_operation_combo.currentText()
            params["update_operation"] = OPERATION_MAP.get(selected_function, '')

            # 3.2a Get update value (Attempt to cast to float as requested by type=float)
            update_value_str = self.update_value_input.text().strip()
            try:
                # Try to convert to float, otherwise keep as string/default
                params["update_value"] = float(update_value_str)
            except ValueError:
                # If it's not a valid float, keep it as the string, as it might be used 
                # for 'replace_string' or similar functions.
                params["update_value"] = update_value_str if update_value_str else 0.0
                
            # 3.3a Handle --input_keys (nargs='*', default=(None, None))
            key1 = self.input_key_1_input.text().strip() or False
            key2 = self.input_key_2_input.text().strip() or False
            params["input_keys"] = "{}{}".format(key1 if key1 else "", f" {key2}" if key2 else "")
        
        # 4. Handle Statistics Update parameters
        if self.is_stats_update_visible and self.update_function_combo.currentText().strip():
            # 4.1 Get update function (must be one of the choices)
            selected_function = self.update_function_combo.currentText()
            params["stats_function"] = FUNCTION_MAP.get(selected_function, '')

            # 4.2 Get name of output file
            params['output_filename'] = self.output_filename_input.text().strip()

        # Final cleanup for empty strings that should be None
        for key in ["target_entry", "filename_pattern", "output_key"]:
            if not params[key]:
                params[key] = None
                
        return params