import sys
import multiprocessing as mp

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QVBoxLayout,
    QWidget, QSpinBox, QComboBox, QSizePolicy,
    QPushButton, QLabel, QScrollArea, QButtonGroup,
)
from ..components import ClickableHeaderWidget
from ..styles.globals import (
    SECONDARY_BUTTON_STYLE,
    SECTION_HEADER_STYLE, SCRIPT_HEADER_STYLE
)


class RunScriptsTab(QWidget):
    def __init__(self):
        super().__init__()    
        # Define available scripts
        self.SCRIPTS = {
            "setup_env": "Setup Environment",
            "gen_data": "Generate Data",
            "train": "Train Model",
            "hyperparam_optim": "Hyperparameter Optimization", 
            "evaluation": "Model Evaluation",
            "test_sim": "Test Simulation",
            # The Slurm scripts are conditionally added/used below
            "slurm": "Slurm",
            "slim_slurm": "Slim Slurm",
        }
        
        # 1. Create the content widget
        self.content_widget = QWidget()
        
        # 2. Use QFormLayout for the content
        content_layout = QFormLayout(self.content_widget)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(5, 5, 5, 5)
        
        self.selected_script = None
        script_header = QLabel("Script Selection")
        script_header.setStyleSheet(SECTION_HEADER_STYLE)
        content_layout.addRow(script_header)

        # --- Script Selection Setup ---
        scripts_container = QWidget()
        scripts_layout = QVBoxLayout(scripts_container)
        scripts_layout.setContentsMargins(0, 0, 0, 0)
        scripts_layout.setSpacing(6)
        
        self.script_buttons = {}
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        
        all_script_names = list(self.SCRIPTS.keys())
        if sys.platform.startswith('linux'):
            scripts_to_display = all_script_names
            linux_only_scripts = []
        else:
            scripts_to_display = all_script_names[:-2]
            linux_only_scripts = all_script_names[-2:]

        # --- Dynamic 2x2 Layout Generation ---
        i = 0
        while i < len(scripts_to_display):
            row_layout = QHBoxLayout()
            for j in range(2):
                if i + j < len(scripts_to_display):
                    script_name = scripts_to_display[i + j]
                    btn = self.create_script_button(script_name)
                    row_layout.addWidget(btn)
                else:
                    row_layout.addStretch(1) 
                    break
            scripts_layout.addLayout(row_layout)
            i += 2

        # Clear Selection Button
        clear_btn_layout = QHBoxLayout()
        self.btn_clear_selection = QPushButton("Clear Selection")
        self.btn_clear_selection.setStyleSheet(SECONDARY_BUTTON_STYLE) # Apply new style
        self.btn_clear_selection.clicked.connect(self.clear_script_selection)
        clear_btn_layout.addStretch() # Push button to the right
        clear_btn_layout.addWidget(self.btn_clear_selection)
        scripts_layout.addLayout(clear_btn_layout)

        # =================================================================
        # ⚠️ MODIFICATION: Non-Linux system (Hidden/Disabled Slurm scripts)
        # =================================================================
        if not sys.platform.startswith('linux'):
            self.linux_only_scripts_header_widget = ClickableHeaderWidget(self._toggle_linux_only_scripts)
            self.linux_only_scripts_header_widget.setStyleSheet(
                "QWidget { border: none; padding: 0; margin-top: 10px; }"
            )

            los_header_layout = QHBoxLayout(self.linux_only_scripts_header_widget)
            los_header_layout.setContentsMargins(0, 0, 0, 0)
            los_header_layout.setSpacing(5)

            self.linux_only_scripts_label = QLabel("Linux Only Scripts (Disabled)")
            self.linux_only_scripts_label.setStyleSheet(SCRIPT_HEADER_STYLE)
            
            self.linux_only_scripts_toggle_button = QPushButton("+")
            self.linux_only_scripts_toggle_button.setFlat(True)
            self.linux_only_scripts_toggle_button.setFixedSize(QSize(20, 20))
            self.linux_only_scripts_toggle_button.setStyleSheet(
                """
                QPushButton { 
                    font-weight: 600; 
                    padding: 0; 
                    border: none; 
                    background: transparent; 
                    color: #7F8C8D; /* Muted text */
                }
                """
            )
            self.linux_only_scripts_toggle_button.clicked.connect(self._toggle_linux_only_scripts)
            
            los_header_layout.addWidget(self.linux_only_scripts_label)
            los_header_layout.addStretch()
            los_header_layout.addWidget(self.linux_only_scripts_toggle_button)
            
            scripts_layout.addWidget(self.linux_only_scripts_header_widget) 

            self.linux_only_scripts_container = QWidget()
            linux_only_scripts_layout = QVBoxLayout(self.linux_only_scripts_container)
            linux_only_scripts_layout.setContentsMargins(0, 5, 0, 0)
            
            self.linux_buttons_container = QWidget()
            row_slurm_layout = QHBoxLayout(self.linux_buttons_container)
            row_slurm_layout.setContentsMargins(0, 0, 0, 0)
            
            self.linux_buttons = []

            for script_name in linux_only_scripts:
                btn = self.create_script_button(script_name)
                
                # New "warning" disabled style
                # This inline style is fine as it's a specific disabled state,
                # but we must *append* to the existing stylesheet.
                current_style = btn.styleSheet() if btn.styleSheet() else ""
                btn.setStyleSheet(
                    current_style + 
                    """
                    QPushButton:disabled { 
                        color: #B87A00; /* Dark Yellow/Orange */
                        background-color: #FFF9E6; /* Light Yellow */
                        border: 1px solid #FFEBC6;
                    }
                    """
                )
                btn.setEnabled(False)
                
                self.linux_buttons.append(btn)
                row_slurm_layout.addWidget(btn)
            
            if len(linux_only_scripts) == 1:
                row_slurm_layout.addStretch(1)
            
            linux_only_scripts_layout.addWidget(self.linux_buttons_container)
            scripts_layout.addWidget(self.linux_only_scripts_container) 
            self.is_area_visible = False
            self.linux_only_scripts_container.hide()
        # =================================================================

        content_layout.addRow(scripts_container)
        
        # --- Script Parameters Section --- 
        content_layout.addRow(self.create_separator())
        params_header = QLabel("Script Parameters")
        params_header.setStyleSheet(SECTION_HEADER_STYLE)
        content_layout.addRow(params_header)
        
        # Selected Script Display
        self.selected_script_label = QLabel("No script selected")
        # Modern "pill" style for selection
        self.selected_script_label.setStyleSheet("""
            font-weight: 600; 
            color: #007AFF; /* Primary Accent */ 
            background-color: #E6F2FF; /* Light blue */
            border-radius: 4px; 
            padding: 5px 8px;
        """)
        content_layout.addRow("Selected Script:", self.selected_script_label)
        
        # Common parameters
        self.verbose_checkbox = QPushButton("Verbose Mode")
        self.verbose_checkbox.setCheckable(True)
        self.verbose_checkbox.setObjectName("toggleStyleButton") # Use standard toggle ID
        content_layout.addRow("Verbose Output:", self.verbose_checkbox)
        
        self.cores_input = QSpinBox(value=22, minimum=1, maximum=mp.cpu_count())
        content_layout.addRow("CPU Cores:", self.cores_input)
        
        # Script-specific parameters
        specific_params_header = QLabel("Script-Specific Parameters")
        specific_params_header.setStyleSheet(SCRIPT_HEADER_STYLE)
        content_layout.addRow(specific_params_header)

        self.script_params_container = QWidget()
        self.script_params_layout = QFormLayout(self.script_params_container)
        self.script_params_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addRow(self.script_params_container)
        
        self.script_params_container.setVisible(False)
        
        # --- Make Tab Scrollable ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.content_widget)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, 
            QSizePolicy.Policy.MinimumExpanding
        )
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        
        self.init_script_parameters()
    
    def create_script_button(self, script_name):
        """Helper method to create script buttons"""
        display_name = self.SCRIPTS[script_name]
        btn = QPushButton(display_name)
        btn.setCheckable(True)
        btn.setObjectName("toggleStyleButton") # Apply standard toggle style ID
        
        btn.clicked.connect(lambda checked, s=script_name: self.select_script(s, checked))
        self.button_group.addButton(btn)
        self.script_buttons[script_name] = btn
        return btn
    
    def create_separator(self):
        """Creates a modern, thin horizontal separator."""
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #DDE3E8;") # BORDER_COLOR
        return separator
    
    def _toggle_linux_only_scripts(self):
        """Toggles the visibility of the Linux Only Scripts input fields."""
        if self.is_area_visible: 
            self.linux_only_scripts_container.hide()
            self.linux_only_scripts_toggle_button.setText("+")
            self.linux_only_scripts_label.setText("Linux Only Scripts (Disabled)")
        else:
            self.linux_only_scripts_container.show()
            self.linux_only_scripts_toggle_button.setText("–") # Use en-dash
            self.linux_only_scripts_label.setText("Linux Only Scripts (Disabled)")
        self.is_area_visible = not self.is_area_visible

    def init_script_parameters(self):
        """Initialize parameter widgets for each script type"""
        self.param_widgets = {}
        
        # Environment setup parameters
        setup_env_params = QWidget(self.script_params_container)
        setup_env_layout = QFormLayout(setup_env_params)
        setup_env_layout.setContentsMargins(0, 0, 0, 0)
        
        self.env_manager_input = QComboBox()
        self.env_manager_input.addItems(["uv", "conda", "venv"])
        setup_env_layout.addRow("Package Manager:", self.env_manager_input)
        
        self.param_widgets["setup_env"] = setup_env_params
        self.script_params_layout.addWidget(setup_env_params) 
        
        # Default placeholder
        self.default_params_label = QLabel("No specific parameters for this script.")
        self.default_params_label.setStyleSheet("color: #7F8C8D;") # Muted text
        self.script_params_layout.addWidget(self.default_params_label)

        setup_env_params.hide()
        self.default_params_label.hide()

    def select_script(self, script_name, checked):
        """Handle script selection"""
        if checked:
            self.selected_script = script_name
            display_name = self.SCRIPTS[script_name]
            self.selected_script_label.setText(display_name)
            self.show_script_parameters(script_name)
        else:
            # This case is handled by the QButtonGroup, 
            # but we reset if "Clear" is hit.
            self.selected_script = None
            self.selected_script_label.setText("No script selected")
            self.hide_script_parameters()

    def clear_script_selection(self):
        """Clear the current script selection"""
        current_button = self.button_group.checkedButton()
        if current_button:
            self.button_group.setExclusive(False)
            current_button.setChecked(False)
            self.button_group.setExclusive(True)
        
        self.selected_script = None
        self.selected_script_label.setText("No script selected")
        self.hide_script_parameters()

    def show_script_parameters(self, script_name):
        """Show parameters specific to the selected script."""
        self.script_params_container.setVisible(True)
        
        for widget in self.param_widgets.values():
            widget.hide()
        self.default_params_label.hide()
        
        if script_name in self.param_widgets:
            self.param_widgets[script_name].show()
        else:
            self.default_params_label.setText(f"No specific parameters for {self.SCRIPTS[script_name]}")
            self.default_params_label.show()

    def hide_script_parameters(self):
        """Hide the script parameters section."""
        self.script_params_container.setVisible(False)
        for widget in self.param_widgets.values():
            widget.hide()
        self.default_params_label.hide()

    # get_params and get_command logic remains unchanged
    def get_params(self):
        """Get all parameters for the selected script"""
        if not self.selected_script:
            return {}
        
        base_params = {
            "script": self.selected_script,
            "verbose": self.verbose_checkbox.isChecked(),
            "cores": self.cores_input.value(),
        }
        
        if self.selected_script == "setup_env":
            base_params.update({
                "manager": self.env_manager_input.currentText(),
            })
        
        return base_params

    def get_command(self):
        """Generate the command to run the selected script"""
        params = self.get_params()
        if not params:
            return ""
        
        script_name = params["script"]
        command_parts = []
        
        if script_name in ["test_sim", "train", "hyperparam_optim", "gen_data"]:
            command_parts.append(f"python main.py {script_name.replace('_', ' ')}")
        elif script_name == "setup_env":
            if sys.platform.startswith('linux'):
                command_parts.append(f"scripts/setup_env.sh {params.get('manager', 'uv')}")
            else:
                command_parts.append(f"scripts\\setup_env.bat {params.get('manager', 'uv')}")
        elif script_name in ["slim_slurm", "slurm"]:
            command_parts.append(f"bash {script_name}.sh")
        
        if params.get("verbose"):
            command_parts.append("--verbose")
        
        if params.get("cores") and script_name in ["test_sim", "slurm"]:
            command_parts.append(f"-nc {params['cores']}")
        
        return " ".join(command_parts)
