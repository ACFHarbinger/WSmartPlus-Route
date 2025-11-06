from PySide6.QtWidgets import (
    QDoubleSpinBox, QComboBox,
    QLineEdit, QFormLayout, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QWidget
)
from backend.src.gui.app_definitions import DATA_DISTRIBUTIONS, PROBLEM_TYPES


class GenDataProblemTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        
        # 1. --problem
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(PROBLEM_TYPES + ['All'])
        self.problem_combo.setCurrentText('All')
        layout.addRow("Problem Type:", self.problem_combo)
        
        # 2. --graph_sizes
        self.graph_sizes_input = QLineEdit("20 50 100")
        self.graph_sizes_input.setPlaceholderText("Space-separated list of sizes (e.g., 20 50 100)")
        layout.addRow("Graph Sizes:", self.graph_sizes_input)

        # 3. --data_distributions
        layout.addRow(QLabel("<b>Data Distributions:</b>"))
        
        distributions_container = QWidget()
        distributions_layout = QVBoxLayout(distributions_container)
        distributions_layout.setContentsMargins(0, 0, 0, 0)
        
        self.dist_buttons = {}
        row_layout = None
        
        # Determine how many distributions per row (e.g., 3)
        NUM_COLS = 4
        
        # Create buttons for each distribution
        for i, dist_name in enumerate(DATA_DISTRIBUTIONS.keys()):
            if i % NUM_COLS == 0:
                row_layout = QHBoxLayout()
                distributions_layout.addLayout(row_layout)
            
            btn = QPushButton(dist_name.title()) # Use .title() for display text
            btn.setCheckable(True)
            
            # Set default styling (similar to your policy buttons)
            btn.setStyleSheet("""
                QPushButton:checked {
                    background-color: #3320b5; /* Blue when checked */
                    color: white;
                    border: 1px solid #27ae60;
                }
                QPushButton:hover:!checked {
                    background-color: #3498db; /* Light hover color for unchecked */
                }
                QPushButton:hover:checked {
                    background-color: #00838a;
                }
            """)
            
            # Automatically check 'all' or 'uniform' if you have a default
            if dist_name.lower() == 'all':
                btn.setChecked(True)
                
            row_layout.addWidget(btn)
            self.dist_buttons[dist_name] = btn

        # Add stretch to incomplete rows
        if row_layout is not None and len(DATA_DISTRIBUTIONS.keys()) % NUM_COLS != 0:
            for _ in range(NUM_COLS - (len(DATA_DISTRIBUTIONS.keys()) % NUM_COLS)):
                row_layout.addStretch(1)

        # Select All / Deselect All Buttons
        all_btn_layout = QHBoxLayout()
        self.btn_select_all_dists = QPushButton("Select All")
        self.btn_select_all_dists.setStyleSheet("background-color: green; color: white;")
        self.btn_select_all_dists.clicked.connect(self.select_all_distributions)
        self.btn_deselect_all_dists = QPushButton("Deselect All")
        self.btn_deselect_all_dists.setStyleSheet("background-color: red; color: white;")
        self.btn_deselect_all_dists.clicked.connect(self.deselect_all_distributions)

        all_btn_layout.addWidget(self.btn_select_all_dists)
        all_btn_layout.addWidget(self.btn_deselect_all_dists)
        distributions_layout.addLayout(all_btn_layout)

        layout.addRow(distributions_container)

        layout.addRow(QLabel('<span style="font-weight: 600;">PDP Parameters</span>'))
        # 4. --is_gaussian
        start_red_style = """
            QPushButton:checked {
                background-color: #06402B;
                color: white;
            }
            QPushButton {
                background-color: #8B0000;
                color: white;
            }
        """
        self.is_gaussian_check = QPushButton("Use Gaussian Distribution")
        self.is_gaussian_check.setCheckable(True)
        self.is_gaussian_check.setChecked(False)
        self.is_gaussian_check.setStyleSheet(start_red_style)
        layout.addRow(QLabel("Gaussian:"), self.is_gaussian_check)

        # 5. --sigma
        self.sigma_input = QDoubleSpinBox()
        self.sigma_input.setRange(0, 1)
        self.sigma_input.setSingleStep(0.1)
        self.sigma_input.setValue(0.6)
        layout.addRow("Sigma Value:", self.sigma_input)

        # 6. --penalty_factor
        layout.addRow(QLabel('<span style="font-weight: 600;">PCTSP Parameters</span>'))
        self.penalty_factor_input = QDoubleSpinBox()
        self.penalty_factor_input.setRange(0.1, 10.0)
        self.penalty_factor_input.setSingleStep(0.1)
        self.penalty_factor_input.setValue(3.0)
        layout.addRow("Penalty Factor:", self.penalty_factor_input)

    def select_all_distributions(self):
        """Sets all distribution buttons to checked."""
        for btn in self.dist_buttons.values():
            btn.setChecked(True)

    def deselect_all_distributions(self):
        """Sets all distribution buttons to unchecked."""
        for btn in self.dist_buttons.values():
            btn.setChecked(False)

    # --- Modified get_params ---

    def get_params(self):
        params = {}
        # Mandatory fields
        params["problem"] = self.problem_combo.currentText()
        params["penalty_factor"] = self.penalty_factor_input.value()
        params["is_gaussian"] = 1 if self.is_gaussian_check.isChecked() else 0

        # List fields (only include if non-empty)
        if self.graph_sizes_input.text().strip():
            params["graph_sizes"] = self.graph_sizes_input.text().strip()

        # 4. --data_distributions (Collect from buttons) 🌟
        selected_dists = [
            dist_name 
            for dist_name, btn in self.dist_buttons.items() 
            if btn.isChecked()
        ]
        
        if selected_dists:
            # Join the list of selected distribution names with a space
            params["data_distributions"] = " ".join([DATA_DISTRIBUTIONS[sd] for sd in selected_dists])
            
        if self.is_gaussian_check.isChecked():
            params["sigma"] = self.sigma_input.value()
        
        return params
