from PySide6.QtWidgets import (
    QLineEdit, QDoubleSpinBox,
    QLabel, QWidget, QFormLayout,
    QComboBox, QSpinBox, QCheckBox,
    QVBoxLayout, QScrollArea, QHBoxLayout,
)
from backend.src.gui.app_definitions import (
    COUNTY_AREAS, WASTE_TYPES,
    VERTEX_METHODS, EDGE_METHODS,
    DATA_DISTRIBUTIONS, DATA_DIST_PROBLEMS,
    PROBLEM_TYPES, DISTANCE_MATRIX_METHODS, 
) 
from .rl_base import BaseReinforcementLearningTab


class RLDataTab(BaseReinforcementLearningTab):
    """Data parameters for Reinforcement Learning"""
    def __init__(self):
        super().__init__()
        self.widgets = {}
        # Container for the field widget
        self.data_dist_row = None 
        # Container for the label widget (NEW)
        self.data_dist_label = None 
        self.init_ui()
        
    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        
        layout = QFormLayout()
        
        # Problem
        self.widgets['problem'] = QComboBox()
        self.widgets['problem'].addItems(PROBLEM_TYPES)
        # Set default to 'TSP'
        self.widgets['problem'].setCurrentText('TSP') 
        layout.addRow(QLabel("Problem:"), self.widgets['problem'])

        # --- Data Distribution Dropdown (Conditional) ---
        self.widgets['data_distribution'] = QComboBox()
        self.widgets['data_distribution'].addItems(DATA_DISTRIBUTIONS.keys())
        self.widgets['data_distribution'].setCurrentText("Gamma 1")
        
        # 1. Create and store the label
        self.data_dist_label = QLabel("Data Distribution:")
        
        # 2. Wrap the QComboBox in a simple widget (the field container)
        self.data_dist_row = QWidget()
        row_layout = QHBoxLayout(self.data_dist_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.widgets['data_distribution'])
        
        # 3. Add both the stored label and the field container to the row
        layout.addRow(self.data_dist_label, self.data_dist_row)
        
        # MANUALLY HIDE BOTH WIDGETS ON INITIAL LOAD
        self.data_dist_row.setVisible(False)
        self.data_dist_label.setVisible(False)

        # Connect the signal for conditional visibility
        self.widgets['problem'].currentTextChanged.connect(self.update_distribution_visibility)
        
        # Graph size
        self.widgets['graph_size'] = QSpinBox(value=20, minimum=1, maximum=1000)
        layout.addRow(QLabel("Graph Size:"), self.widgets['graph_size'])
        
        # Edge threshold
        self.widgets['edge_threshold'] = QDoubleSpinBox()
        self.widgets['edge_threshold'].setRange(0.0, 1.0)
        self.widgets['edge_threshold'].setSingleStep(0.01) # Step by 0.01 (1%)
        self.widgets['edge_threshold'].setDecimals(2)      # Show 2 decimal places
        self.widgets['edge_threshold'].setValue(1.0)      # Set a default value
        layout.addRow(QLabel("Edge Threshold (0.0 to 1.0):"), self.widgets['edge_threshold'])
        
        # Edge method
        self.widgets['edge_method'] = QComboBox()
        self.widgets['edge_method'].addItems(EDGE_METHODS.keys())
        self.widgets['edge_method'].setCurrentIndex(0)
        layout.addRow(QLabel("Edge Method:"), self.widgets['edge_method'])
        
        # Batch size
        self.widgets['batch_size'] = QSpinBox(value=256, minimum=1, maximum=10000)
        layout.addRow(QLabel("Batch Size:"), self.widgets['batch_size'])
        
        # Epoch size
        self.widgets['epoch_size'] = QSpinBox(value=128000, minimum=1, maximum=1000000)
        layout.addRow(QLabel("Epoch Size:"), self.widgets['epoch_size'])
        
        # Val size
        self.widgets['val_size'] = QSpinBox(value=0, minimum=0, maximum=100000)
        layout.addRow(QLabel("Validation Size:"), self.widgets['val_size'])
        
        # Val dataset
        self.widgets['val_dataset'] = QLineEdit()
        layout.addRow(QLabel("Validation Dataset:"), self.widgets['val_dataset'])
        
        # Eval batch size
        self.widgets['eval_batch_size'] = QSpinBox(value=256, minimum=1, maximum=10000)
        layout.addRow(QLabel("Eval Batch Size:"), self.widgets['eval_batch_size'])
        
        # Train dataset
        self.widgets['train_dataset'] = QLineEdit()
        layout.addRow(QLabel("Train Dataset:"), self.widgets['train_dataset'])
        
        # Area
        self.widgets['area'] = QComboBox()
        self.widgets['area'].addItems(COUNTY_AREAS.keys())
        self.widgets['area'].setCurrentText("Rio Maior") # Set a default value
        layout.addRow(QLabel("County Area:"), self.widgets['area'])
        
        # Waste type
        self.widgets['waste_type'] = QComboBox()
        self.widgets['waste_type'].addItems(WASTE_TYPES)
        self.widgets['waste_type'].setCurrentText("Plastic") # Set a default value
        layout.addRow(QLabel("Waste Type:"), self.widgets['waste_type'])
        
        # Distance method
        self.widgets['distance_method'] = QComboBox()
        self.widgets['distance_method'].addItems(DISTANCE_MATRIX_METHODS.keys())
        layout.addRow(QLabel("Distance Method:"), self.widgets['distance_method'])
        
        # Vertex method
        self.widgets['vertex_method'] = QComboBox()
        self.widgets['vertex_method'].addItems(VERTEX_METHODS.keys())
        layout.addRow(QLabel("Vertex Method:"), self.widgets['vertex_method'])
        
        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    
    def update_distribution_visibility(self, problem_type):
        """Shows or hides the data distribution dropdown based on the selected problem."""
        is_visible = problem_type.upper() in DATA_DIST_PROBLEMS
        
        # Hide or show the field container
        if self.data_dist_row:
            self.data_dist_row.setVisible(is_visible)
        
        # Hide or show the label (NEW CRITICAL STEP)
        if self.data_dist_label:
            self.data_dist_label.setVisible(is_visible)

    def get_params(self):
        """Returns parameters from the TrainDataTab widgets."""
        params = {}
        for key, widget in self.widgets.items():
            
            value = None
            if isinstance(widget, QSpinBox):
                value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
                continue
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
                
                # Special handling for data_distribution: map display name to command-line argument
                if key == 'data_distribution':
                    # Only include data_distribution if its container is visible
                    if self.data_dist_row and self.data_dist_row.isVisible():
                        # Use the map to get the correct command-line argument string
                        value = DATA_DISTRIBUTIONS.get(value, "") 
                    else:
                        continue # Skip parameter if not relevant/visible
                elif key in 'vertex_method':
                    value = VERTEX_METHODS.get(value, "")
                elif key == 'edge_method':
                    value = EDGE_METHODS.get(value, "")
                elif key == 'area':
                    value = COUNTY_AREAS.get(value, "")
                elif key == 'waste_type':
                    value = value.lower()
                elif key == 'distance_method':
                    value = DISTANCE_MATRIX_METHODS.get(value, "")
            elif isinstance(widget, QCheckBox):
                if widget.isChecked():
                    params[key] = True
                    continue
            
            if value is not None:
                # Handle empty strings (i.e., optional fields left blank), except for '0' in edge_threshold
                if isinstance(value, str) and not value and key != 'edge_threshold':
                    continue
                
                params[key] = value

        return params
