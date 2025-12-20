from PySide6.QtWidgets import (
    QComboBox,
    QLineEdit, QFormLayout, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QWidget
)
from gui.src.utils.app_definitions import DATA_DISTRIBUTIONS, PROBLEM_TYPES
from ...styles.globals import (
    SECTION_HEADER_STYLE, SUCCESS_BUTTON_STYLE,
    SECONDARY_BUTTON_STYLE, SUB_HEADER_STYLE
)


class GenDataProblemTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)
        
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
        dist_header = QLabel("Data Distributions")
        dist_header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addRow(dist_header)
        
        distributions_container = QWidget()
        distributions_layout = QVBoxLayout(distributions_container)
        distributions_layout.setContentsMargins(0, 0, 0, 0)
        distributions_layout.setSpacing(6)
        
        self.dist_buttons = {}
        row_layout = None
        
        NUM_COLS = 4
        
        for i, dist_name in enumerate(DATA_DISTRIBUTIONS.keys()):
            if i % NUM_COLS == 0:
                row_layout = QHBoxLayout()
                distributions_layout.addLayout(row_layout)
            
            btn = QPushButton(dist_name.title())
            btn.setCheckable(True)
            btn.setObjectName("toggleStyleButton") # Apply new toggle style ID
            
            if dist_name.lower() == 'all':
                btn.setChecked(True)
                
            row_layout.addWidget(btn)
            self.dist_buttons[dist_name] = btn

        if row_layout is not None and len(DATA_DISTRIBUTIONS.keys()) % NUM_COLS != 0:
            for _ in range(NUM_COLS - (len(DATA_DISTRIBUTIONS.keys()) % NUM_COLS)):
                row_layout.addStretch(1)

        # Select All / Deselect All Buttons
        all_btn_layout = QHBoxLayout()
        self.btn_select_all_dists = QPushButton("Select All")
        self.btn_select_all_dists.setStyleSheet(SUCCESS_BUTTON_STYLE) # Apply new green style
        self.btn_select_all_dists.clicked.connect(self.select_all_distributions)
        
        self.btn_deselect_all_dists = QPushButton("Deselect All")
        self.btn_deselect_all_dists.setStyleSheet(SECONDARY_BUTTON_STYLE) # Apply new secondary style
        self.btn_deselect_all_dists.clicked.connect(self.deselect_all_distributions)

        all_btn_layout.addWidget(self.btn_select_all_dists)
        all_btn_layout.addWidget(self.btn_deselect_all_dists)
        distributions_layout.addLayout(all_btn_layout)

        layout.addRow(distributions_container)

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

        # List fields (only include if non-empty)
        if self.graph_sizes_input.text().strip():
            params["graph_sizes"] = self.graph_sizes_input.text().strip()

        # 4. --data_distributions (Collect from buttons)
        selected_dists = [
            dist_name 
            for dist_name, btn in self.dist_buttons.items() 
            if btn.isChecked()
        ]
        
        if selected_dists:
            params["data_distributions"] = " ".join([DATA_DISTRIBUTIONS[sd] for sd in selected_dists])
            
        return params
