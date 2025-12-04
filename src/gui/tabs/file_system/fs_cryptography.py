import os
from PySide6.QtCore import QSize
from PySide6.QtGui import QFont, QFont
from PySide6.QtWidgets import (
    QVBoxLayout, QGroupBox,
    QPushButton, QSizePolicy,
    QSpinBox, QLabel, QHBoxLayout,
    QWidget, QLineEdit, QFormLayout,
    QFileDialog  # <-- Added
)
from ...components import ClickableHeaderWidget


class FileSystemCryptographyTab(QWidget):
    """
    GUI tab for Cryptography settings based on crypto_parser arguments.
    """
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("<h2>Cryptography Tools</h2>"))
        
        # --- Common Font for QGroupBox Titles ---
        bold_font = QFont()
        bold_font.setBold(True)
        
        # --- File Paths ---
        file_group = QGroupBox("File Paths")
        file_group.setFont(bold_font) # Apply the bold font directly
        file_group.setStyleSheet("QGroupBox::title { font-weight: bold; }") # Keep QSS as fallback
        file_layout = QFormLayout(file_group)
        
        self.env_file_input = QLineEdit("vars.env")
        file_layout.addRow(QLabel("Environment File:"), self._create_browser_layout(self.env_file_input))

        # --------------------------------------------------------------------
        # --- Input Output (Custom Header) ---
        # --------------------------------------------------------------------
        # Create a container widget for the header using the custom clickable class
        self.input_output_header_widget = ClickableHeaderWidget(self._toggle_input_output)
        
        # NOTE: The header widget itself has NO border
        self.input_output_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        io_header_layout = QHBoxLayout(self.input_output_header_widget)
        io_header_layout.setContentsMargins(0, 0, 0, 0)
        io_header_layout.setSpacing(5)

        # The main text (Standard QLabel)
        self.input_output_label = QLabel("<b>Input-Output Settings</b>")
        
        # Ensure the label shrinks to fit its content
        self.input_output_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel (border around text only)
        self.input_output_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # The clickable toggle button (only the +/- sign)
        self.input_output_toggle_button = QPushButton("+")
        self.input_output_toggle_button.setFlat(True)
        self.input_output_toggle_button.setFixedSize(QSize(20, 20))
        self.input_output_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.input_output_toggle_button.clicked.connect(self._toggle_input_output)
        
        # Add components to the header layout
        io_header_layout.addWidget(self.input_output_label)
        io_header_layout.addStretch()
        io_header_layout.addWidget(self.input_output_toggle_button)
        
        # QVBoxLayout uses addWidget()
        main_layout.addWidget(self.input_output_header_widget)

        # Create a container for the collapsible content
        self.input_output_container = QWidget()
        input_output_layout = QFormLayout(self.input_output_container)
        input_output_layout.setContentsMargins(0, 5, 0, 0) 

        # Add widgets to the container's layout
        # --symkey_name
        self.symkey_name_input = QLineEdit()
        self.symkey_name_input.setPlaceholderText("Name of the key")
        input_output_layout.addRow(QLabel("Symmetric Key Name:"), self.symkey_name_input)

        # --input_path
        self.input_path_input = QLineEdit()
        self.input_path_input.setPlaceholderText("Input file path")
        input_output_layout.addRow(QLabel("Input Path:"), self._create_browser_layout(self.input_path_input))
        
        # --output_path
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Output file path")
        input_output_layout.addRow(QLabel("Output Path:"), self._create_browser_layout(self.output_path_input))

        # The collapsible content container is added here
        main_layout.addWidget(self.input_output_container)

        # Initialize state: hidden (must match the default style above)
        self.is_input_output_visible = False
        self.input_output_container.hide()

        # --------------------------------------------------------------------
        main_layout.addWidget(file_group)

        # --- Key Generation Parameters ---
        param_group = QGroupBox("Key Generation Parameters")
        param_group.setFont(bold_font) # Apply the bold font directly
        param_group.setStyleSheet("QGroupBox::title { font-weight: bold; }") # Keep QSS as fallback
        param_layout = QFormLayout(param_group)

        # --salt_size (default=16)
        self.salt_size_input = QSpinBox(minimum=1, maximum=256, value=16)
        param_layout.addRow(QLabel("Salt Size (bytes):"), self.salt_size_input)

        # --key_length (default=32)
        self.key_length_input = QSpinBox(minimum=1, maximum=1024, value=32)
        param_layout.addRow(QLabel("Key Length (bytes):"), self.key_length_input)
        
        # --hash_iterations (default=100_000)
        self.hash_iterations_input = QSpinBox(minimum=1000, maximum=1_000_000, value=100_000)
        self.hash_iterations_input.setSingleStep(10000)
        param_layout.addRow(QLabel("Hash Iterations:"), self.hash_iterations_input)
        
        main_layout.addWidget(param_group)
        main_layout.addStretch() # Push everything to the top
    
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

    def _toggle_input_output(self):
        """Toggles the visibility of the Input-Output Settings input fields and updates the +/- sign."""
        if self.is_input_output_visible: 
            self.input_output_container.hide()
            self.input_output_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.input_output_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.input_output_container.show()
            self.input_output_toggle_button.setText("-")

            # Remove the border from the QLabel when expanded.
            self.input_output_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_input_output_visible = not self.is_input_output_visible

    def get_params(self):
        """Extracts settings into a dictionary mimicking argparse output."""
        params = {
            "salt_size": self.salt_size_input.value(),
            "key_length": self.key_length_input.value(),
            "hash_iterations": self.hash_iterations_input.value(),
            "env_file": self.env_file_input.text().strip(),
            # Optional parameters are only included if they have text
            "symkey_name": self.symkey_name_input.text().strip() or None,
            "input_path": self.input_path_input.text().strip() or None,
            "output_path": self.output_path_input.text().strip() or None,
        }
        return params