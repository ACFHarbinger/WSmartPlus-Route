from PySide6.QtWidgets import (
    QWidget, QSpinBox, QComboBox, QSizePolicy,
    QFormLayout, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QScrollArea,
)
from backend.src.gui.app_definitions import (
    SIMULATOR_TEST_POLICIES,
    PROBLEM_TYPES, DATA_DISTRIBUTIONS, 
)


class TestSimSettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # 1. Create the content widget to hold all elements
        self.content_widget = QWidget()
        
        # 2. Use the QFormLayout for the content widget
        content_layout = QFormLayout(self.content_widget)
        
        self.selected_policies = set()
        content_layout.addRow(QLabel("<b>Test Policies</b>"))

        # --- Policy Selection Setup (Remains within content_layout) ---
        policies_container = QWidget()
        policies_layout = QVBoxLayout(policies_container)
        policies_layout.setContentsMargins(0, 0, 0, 0)
        
        # Policy Buttons
        self.policy_buttons = {}
        row_layout = None
        for i, policy_name in enumerate(SIMULATOR_TEST_POLICIES.keys()):
            if i % 3 == 0:
                row_layout = QHBoxLayout()
                policies_layout.addLayout(row_layout)
            
            btn = QPushButton(policy_name)
            btn.setCheckable(True)
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
            
            btn.clicked.connect(lambda checked, p=policy_name: self.toggle_policy(p, checked))
            row_layout.addWidget(btn)
            self.policy_buttons[policy_name] = btn

        if row_layout is not None and len(SIMULATOR_TEST_POLICIES.keys()) % 3 != 0:
            for _ in range(3 - (len(SIMULATOR_TEST_POLICIES.keys()) % 3)):
                row_layout.addStretch(1)

        # Select All / Deselect All Buttons
        all_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.setStyleSheet("background-color: green; color: white;")
        self.btn_select_all.clicked.connect(self.select_all_policies)
        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.setStyleSheet("background-color: red; color: white;")
        self.btn_deselect_all.clicked.connect(self.deselect_all_policies)

        all_btn_layout.addWidget(self.btn_select_all)
        all_btn_layout.addWidget(self.btn_deselect_all)
        policies_layout.addLayout(all_btn_layout)

        content_layout.addRow(policies_container)
        
        # --- Test Environment --- 
        content_layout.addRow(QLabel("<hr>")) # Use a horizontal line for separation
        content_layout.addRow(QLabel("<b>Test Environment</b>"))
        
        # 2. --data_distribution
        self.data_dist_input = QComboBox()
        self.data_dist_input.addItems(DATA_DISTRIBUTIONS.keys())
        self.data_dist_input.setCurrentText("Gamma 1")
        content_layout.addRow("Waste Fill Data Distribution:", self.data_dist_input)
         
        # 3. --problem
        self.problem_input = QComboBox()
        self.problem_input.addItems(PROBLEM_TYPES)
        self.problem_input.setCurrentText("VRPP")
        content_layout.addRow("Problem Type:", self.problem_input)

        # 4. --size
        self.size_input = QSpinBox(value=50, minimum=5, maximum=500)
        content_layout.addRow("Graph Size:", self.size_input)

        # 5. --days
        self.days_input = QSpinBox(value=31, minimum=1, maximum=365)
        content_layout.addRow("Simulation Days:", self.days_input)

        # 6. --n_samples
        self.n_samples_input = QSpinBox(value=10, minimum=1, maximum=100)
        content_layout.addRow("Number of Samples:", self.n_samples_input)

        # 7. --n_vehicles
        self.n_vehicles_input = QSpinBox(value=1, minimum=1, maximum=10)
        content_layout.addRow("Number of Vehicles:", self.n_vehicles_input)

        # 8. --seed
        self.seed_input = QSpinBox(value=42, minimum=0, maximum=100000)
        content_layout.addRow("Random Seed:", self.seed_input)
        
        # --- Make Tab Scrollable ---
        
        # 3. Create the QScrollArea
        scroll_area = QScrollArea()
        
        # Set the content_widget as the QScrollArea's widget
        scroll_area.setWidgetResizable(True) # Allows the widget to be resized when the scroll area is resized
        scroll_area.setWidget(self.content_widget)
        
        # Ensure the content widget takes minimum space vertically
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, 
            QSizePolicy.Policy.MinimumExpanding
        )
        
        # 4. Set the QScrollArea as the main layout of TestSimSettingsTab
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
    
    # Policy Management Methods (Unchanged)
    def toggle_policy(self, policy_name, checked):
        if checked:
            self.selected_policies.add(policy_name)
        else:
            self.selected_policies.discard(policy_name)
    
    def select_all_policies(self):
        for policy_name in SIMULATOR_TEST_POLICIES.keys():
            self.selected_policies.add(policy_name)
            self.policy_buttons[policy_name].setChecked(True)
            
    def deselect_all_policies(self):
        self.selected_policies.clear()
        for policy_name in SIMULATOR_TEST_POLICIES.keys():
            self.policy_buttons[policy_name].setChecked(False)

    # Parameter Retrieval Method (Unchanged)
    def get_params(self):
        params = {
            "data_distribution": DATA_DISTRIBUTIONS[self.data_dist_input.currentText().strip()],
            "problem": self.problem_input.currentText().strip().lower(),
            "size": self.size_input.value(),
            "days": self.days_input.value(),
            "n_samples": self.n_samples_input.value(),
            "n_vehicles": self.n_vehicles_input.value(),
            "seed": self.seed_input.value(),
        }
        
        policies_str = " ".join([SIMULATOR_TEST_POLICIES[pol] for pol in self.selected_policies])
        if policies_str: 
            params["policies"] = policies_str
            
        return params
