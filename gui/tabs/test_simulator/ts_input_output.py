import os
from PySide6.QtWidgets import (
    QLabel, QSpinBox, QComboBox,
    QLineEdit, QFormLayout, QWidget,
    QPushButton, QHBoxLayout, QFileDialog # <-- Added
)
from gui.utils.app_definitions import WASTE_TYPES, COUNTY_AREAS, DATA_DISTRIBUTIONS


class TestSimIOTab(QWidget):
    def __init__(self, settings_tab): # <-- Accept settings_tab
        super().__init__()
        self.settings_tab = settings_tab # <-- Store reference to settings tab

        # Use QFormLayout directly as the main layout for the tab
        form_layout = QFormLayout(self)
        
        # I/O Paths and Checkpoints
        form_layout.addRow(QLabel("<b>Input-Output Paths</b>"))

        self.output_dir_input = QLineEdit("output")
        form_layout.addRow("Output Directory:", self._create_browser_layout(self.output_dir_input, is_dir=True))
        
        self.checkpoint_dir_input = QLineEdit("temp")
        form_layout.addRow("Checkpoint Directory:", self._create_browser_layout(self.checkpoint_dir_input, is_dir=True))
        
        self.checkpoint_days_input = QSpinBox(value=5, minimum=0, maximum=365)
        form_layout.addRow("Checkpoint Save Days:", self.checkpoint_days_input)
        
        # Input Files
        form_layout.addRow(QLabel("<b>Input Files</b>"))
        # Remove hardcoded defaults; they will be set by update_default_paths
        self.waste_filepath_input = QLineEdit() 
        form_layout.addRow("Waste Fill File:", self._create_browser_layout(self.waste_filepath_input))
        
        self.dm_filepath_input = QLineEdit()
        form_layout.addRow("Distance Matrix File:", self._create_browser_layout(self.dm_filepath_input))
        
        self.bin_idx_file_input = QLineEdit()
        form_layout.addRow("Bin Index File:", self._create_browser_layout(self.bin_idx_file_input))
        
        # Simulator Data Context
        form_layout.addRow(QLabel("<b>Simulator Data Context</b>"))
        
        # 1. County Area (Now a QComboBox)
        self.area_input = QComboBox()
        self.area_input.addItems(COUNTY_AREAS.keys())
        self.area_input.setCurrentText("Rio Maior") # Set a default value
        form_layout.addRow("County Area:", self.area_input)
        
        # 2. Waste Type (Now a QComboBox)
        self.waste_type_input = QComboBox()
        self.waste_type_input.addItems(WASTE_TYPES)
        self.waste_type_input.setCurrentText("Plastic") # Set a default value
        form_layout.addRow("Waste Type:", self.waste_type_input)

        # --- Connect signals to update default paths ---
        self.connect_signals()

        # --- Set initial default paths ---
        self.update_default_paths()

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

    def connect_signals(self):
        """Connect all relevant signals to the update method."""
        # Signals from Settings Tab
        if self.settings_tab:
            self.settings_tab.data_dist_input.currentTextChanged.connect(self.update_default_paths)
            self.settings_tab.size_input.valueChanged.connect(self.update_default_paths)
            self.settings_tab.days_input.valueChanged.connect(self.update_default_paths)
            self.settings_tab.n_samples_input.valueChanged.connect(self.update_default_paths)
            self.settings_tab.seed_input.valueChanged.connect(self.update_default_paths)
            self.settings_tab.n_vehicles_input.valueChanged.connect(self.update_default_paths)

        # Signals from this (IO) Tab
        self.area_input.currentTextChanged.connect(self.update_default_paths)
        self.waste_type_input.currentTextChanged.connect(self.update_default_paths)

    def update_default_paths(self):
        """(Re)generates and sets the default file path QLineEdits."""
        if not self.settings_tab:
            return # Safety check in case settings_tab isn't passed

        try:
            # 1. Get values from Settings Tab
            size = self.settings_tab.size_input.value()
            days = self.settings_tab.days_input.value()
            n_samples = self.settings_tab.n_samples_input.value()
            seed = self.settings_tab.seed_input.value()
            n_vehicles = self.settings_tab.n_vehicles_input.value()
            
            # Get the file-safe key (e.g., 'gamma1') from the display name (e.g., 'Gamma 1')
            data_dist_display = self.settings_tab.data_dist_input.currentText()
            data_dist_key = DATA_DISTRIBUTIONS.get(data_dist_display, "unknown")

            # 2. Get values from this (IO) Tab
            area_display = self.area_input.currentText()
            area_key = COUNTY_AREAS.get(area_display, "unknown") # e.g., 'riomaior'
            waste_type_key = self.waste_type_input.currentText().strip().lower() # e.g., 'plastic'

            # 3. Construct file paths based on original default patterns
            
            # Pattern: "daily_waste/riomaior50_gamma1_wsr31_N10_seed42.pkl"
            waste_file = f"daily_waste/{area_key}{size}_{data_dist_key}_wsr{days}_N{n_samples}_seed{seed}.pkl"
            
            # Pattern: "data/wsr_simulator/distance_matrix/gmaps_distmat_plastic[riomaior].csv"
            dm_file = f"data/wsr_simulator/distance_matrix/gmaps_distmat_{waste_type_key}[{area_key}].csv"
            
            # Pattern: "graphs_50V_1N_plastic.json"
            bin_idx_file = f"graphs_{size}V_{n_vehicles}N_{waste_type_key}.json"

            # 4. Set the text
            self.waste_filepath_input.setText(waste_file)
            self.dm_filepath_input.setText(dm_file)
            self.bin_idx_file_input.setText(bin_idx_file)

        except Exception as e:
            # Avoid crashing the GUI if a widget isn't fully initialized
            print(f"Warning: Could not update default paths. Error: {e}")

    def get_params(self):
        params = {
            # I/O
            "output_dir": self.output_dir_input.text().strip(),
            "checkpoint_dir": self.checkpoint_dir_input.text().strip(),
            "checkpoint_days": self.checkpoint_days_input.value(),
            
            # Context items now retrieve text from QComboBox
            "area": COUNTY_AREAS.get(self.area_input.currentText().strip(), ""), 
            "waste_type": self.waste_type_input.currentText().strip().lower(),
        }
        
        # Optional file paths
        if self.waste_filepath_input.text().strip(): params["waste_filepath"] = self.waste_filepath_input.text().strip()
        if self.dm_filepath_input.text().strip(): params["dm_filepath"] = self.dm_filepath_input.text().strip()
        if self.bin_idx_file_input.text().strip(): params["bin_idx_file"] = self.bin_idx_file_input.text().strip()

        return params