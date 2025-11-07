from PySide6.QtWidgets import (
    QLabel, QSpinBox, QComboBox,
    QLineEdit, QFormLayout, QWidget,
)
from backend.src.gui.app_definitions import WASTE_TYPES, COUNTY_AREAS


class TestSimIOTab(QWidget):
    def __init__(self):
        super().__init__()
        # Use QFormLayout directly as the main layout for the tab
        form_layout = QFormLayout(self)
        
        # I/O Paths and Checkpoints
        form_layout.addRow(QLabel("<b>Input-Output Paths</b>"))

        self.output_dir_input = QLineEdit("output")
        form_layout.addRow("Output Directory:", self.output_dir_input)
        
        self.checkpoint_dir_input = QLineEdit("temp")
        form_layout.addRow("Checkpoint Directory:", self.checkpoint_dir_input)
        
        self.checkpoint_days_input = QSpinBox(value=5, minimum=0, maximum=365)
        form_layout.addRow("Checkpoint Save Days:", self.checkpoint_days_input)
        
        # Input Files
        form_layout.addRow(QLabel("<b>Input Files</b>"))
        self.waste_filepath_input = QLineEdit("daily_waste/riomaior50_gamma1_wsr31_N10_seed42.pkl")
        form_layout.addRow("Waste Fill File:", self.waste_filepath_input)
        
        self.dm_filepath_input = QLineEdit("data/wsr_simulator/distance_matrix/gmaps_distmat_plastic[riomaior].csv")
        form_layout.addRow("Distance Matrix File:", self.dm_filepath_input)
        
        self.bin_idx_file_input = QLineEdit("graphs_50V_1N_plastic.json")
        form_layout.addRow("Bin Index File:", self.bin_idx_file_input)
        
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
