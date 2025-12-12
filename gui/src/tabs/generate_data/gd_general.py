import os
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QSpinBox, QComboBox, QCheckBox,
    QHBoxLayout, QSizePolicy, QPushButton,
    QLineEdit, QFormLayout, QWidget, QLabel,
    QFileDialog # <-- Added
)
from ...components import ClickableHeaderWidget


class GenDataGeneralTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        
        # 1. --name
        self.name_input = QLineEdit("default_dataset")
        layout.addRow("Dataset Name:", self.name_input)

        # --- Filename (Custom Header) ---
        # 1. Create a container widget for the header using the custom clickable class
        self.filename_header_widget = ClickableHeaderWidget(self._toggle_filename)
        self.filename_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        fn_header_layout = QHBoxLayout(self.filename_header_widget)
        fn_header_layout.setContentsMargins(0, 0, 0, 0)
        fn_header_layout.setSpacing(5)

        # 2. The main text (Standard QLabel)
        self.filename_label = QLabel("Filename")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.filename_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.filename_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button (only the +/- sign)
        self.filename_toggle_button = QPushButton("+")
        self.filename_toggle_button.setFlat(True)
        self.filename_toggle_button.setFixedSize(QSize(20, 20))
        self.filename_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.filename_toggle_button.clicked.connect(self._toggle_filename)

        # 4. Add components to the header layout
        fn_header_layout.addWidget(self.filename_label)
        fn_header_layout.addStretch()
        fn_header_layout.addWidget(self.filename_toggle_button)
        
        # 5. Add the header widget to the main layout, making it span the row
        layout.addRow(self.filename_header_widget)

        # 6. Create a container for the collapsible content
        self.filename_container = QWidget()
        filename_layout = QFormLayout(self.filename_container)
        filename_layout.setContentsMargins(0, 0, 0, 0)

        # 7. Add widgets to the container's layout
        # 2. --filename
        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("e.g., my_data.pkl (ignores data_dir)")
        filename_layout.addRow(QLabel("Specific Filename:"), self._create_browser_layout(self.filename_input))
        
        # 8. Add the content container to the main layout
        layout.addWidget(self.filename_container)

        # 9. Initialize state: hidden
        self.is_filename_visible = False
        self.filename_container.hide()
        
        # 3. --data_dir
        self.data_dir_input = QLineEdit("datasets")
        layout.addRow("Data Directory:", self._create_browser_layout(self.data_dir_input, is_dir=True))
        
        # 4. --dataset_size
        self.dataset_size_input = QSpinBox()
        self.dataset_size_input.setRange(1000, 10_000_000)
        self.dataset_size_input.setSingleStep(1000)
        self.dataset_size_input.setValue(128000)
        layout.addRow("Dataset Size:", self.dataset_size_input)

        # 5. --dataset_type
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.addItems(['train', 'train_time', 'test_simulator'])
        layout.addRow("Dataset Type:", self.dataset_type_combo)

        # 6. --seed
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 100000)
        self.seed_input.setValue(42)
        layout.addRow("Random Seed:", self.seed_input)
        
        # 7. -f (Overwrite)
        self.overwrite_check = QCheckBox()
        self.overwrite_check.setText("Overwrite existing file")
        layout.addRow(self.overwrite_check)
    
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

    def _toggle_filename(self):
        """Toggles the visibility of the Filename input field and updates the +/- sign."""
        if self.is_filename_visible: 
            self.filename_container.hide()
            self.filename_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.filename_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.filename_container.show()
            self.filename_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.filename_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_filename_visible = not self.is_filename_visible
        
    def get_params(self):
        params = {}
        # Mandatory fields
        params["name"] = self.name_input.text().strip()
        params["data_dir"] = self.data_dir_input.text().strip()
        params["dataset_size"] = self.dataset_size_input.value()
        params["dataset_type"] = self.dataset_type_combo.currentText()
        params["seed"] = self.seed_input.value()

        # Optional fields
        if self.filename_input.text().strip():
            params["filename"] = self.filename_input.text().strip()
        if self.overwrite_check.isChecked():
            params["f"] = True
        
        return params