import sys
import multiprocessing as mp

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QVBoxLayout,
    QWidget, QSpinBox, QComboBox, QSizePolicy,
    QPushButton, QLabel, QScrollArea, QButtonGroup,
)
from ..components import ClickableHeaderWidget


class FileSystemScriptsTab(QWidget):
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
        
        # 1. Create the content widget to hold all elements
        self.content_widget = QWidget()
        
        # 2. Use the QFormLayout for the content widget
        content_layout = QFormLayout(self.content_widget)
        
        self.selected_script = None
        content_layout.addRow(QLabel("<b>Script Selection</b>"))

        # --- Script Selection Setup ---
        scripts_container = QWidget()
        scripts_layout = QVBoxLayout(scripts_container)
        scripts_layout.setContentsMargins(0, 0, 0, 0)
        
        # Script Buttons with mutual exclusion
        self.script_buttons = {}
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)  # Only one can be selected at a time
        
        # Get all script keys
        all_script_names = list(self.SCRIPTS.keys())

        # Determine which scripts to display based on the platform
        if sys.platform.startswith('linux'):
            # Include all 8 scripts for Linux
            scripts_to_display = all_script_names
            linux_only_scripts = [] # None left for the collapsible section
        else:
            # Exclude the last two (Slurm and Slim Slurm) for non-Linux platforms (6 scripts)
            scripts_to_display = all_script_names[:-2]
            linux_only_scripts = all_script_names[-2:] # Slurm and Slim Slurm

        # --- Dynamic 2x2 Layout Generation for Non-Slurm Scripts ---
        i = 0
        # Iterate over the list in steps of 2
        while i < len(scripts_to_display):
            row_layout = QHBoxLayout()
            
            # Add up to 2 buttons to the current row
            for j in range(2):
                if i + j < len(scripts_to_display):
                    script_name = scripts_to_display[i + j]
                    btn = self.create_script_button(script_name)
                    row_layout.addWidget(btn)
                else:
                    # If the last row only has one button, add a stretch widget
                    row_layout.addStretch(1) 
                    break
            
            # Add the completed row layout to the main scripts layout
            scripts_layout.addLayout(row_layout)
            i += 2
            
        # Add the scripts container to the main content layout
        # content_layout.addRow(scripts_container) # This is handled later

        # Clear Selection Button
        clear_btn_layout = QHBoxLayout()
        self.btn_clear_selection = QPushButton("Clear Selection")
        self.btn_clear_selection.setStyleSheet("background-color: #e74c3c; color: white;")
        self.btn_clear_selection.clicked.connect(self.clear_script_selection)
        clear_btn_layout.addWidget(self.btn_clear_selection)
        scripts_layout.addLayout(clear_btn_layout)

        # =================================================================
        # ⚠️ MODIFICATION: Non-Linux system (Hidden/Disabled Slurm scripts)
        # =================================================================
        if not sys.platform.startswith('linux'):
            # --- Linux Only Scripts (Custom Header) --- 
            self.linux_only_scripts_header_widget = ClickableHeaderWidget(self._toggle_linux_only_scripts)
            self.linux_only_scripts_header_widget.setStyleSheet(
                "QWidget { border: none; padding: 0; margin-top: 5px; }"
            )

            los_header_layout = QHBoxLayout(self.linux_only_scripts_header_widget)
            los_header_layout.setContentsMargins(0, 0, 0, 0)
            los_header_layout.setSpacing(5)

            self.linux_only_scripts_label = QLabel("<b>Linux Only Scripts</b>")
            self.linux_only_scripts_label.setSizePolicy(
                QSizePolicy.Policy.Fixed, 
                QSizePolicy.Policy.Preferred
            )
            self.linux_only_scripts_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )

            self.linux_only_scripts_toggle_button = QPushButton("+")
            self.linux_only_scripts_toggle_button.setFlat(True)
            self.linux_only_scripts_toggle_button.setFixedSize(QSize(20, 20))
            self.linux_only_scripts_toggle_button.setStyleSheet(
                "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
            )
            self.linux_only_scripts_toggle_button.clicked.connect(self._toggle_linux_only_scripts)
            
            los_header_layout.addWidget(self.linux_only_scripts_label)
            los_header_layout.addStretch()
            los_header_layout.addWidget(self.linux_only_scripts_toggle_button)
            
            scripts_layout.addWidget(self.linux_only_scripts_header_widget) 

            # 6. Create a container for the collapsible content
            self.linux_only_scripts_container = QWidget()
            linux_only_scripts_layout = QVBoxLayout(self.linux_only_scripts_container)
            linux_only_scripts_layout.setContentsMargins(0, 0, 0, 0)
            
            # 7. Create a separate container for the Slurm/Slim Slurm buttons
            self.linux_buttons_container = QWidget()
            row_slurm_layout = QHBoxLayout(self.linux_buttons_container) # Renamed to row_slurm_layout
            row_slurm_layout.setContentsMargins(0, 0, 0, 0)
            
            self.linux_buttons = []

            # Row: 2 scripts (Slurm, Slim Slurm) - Indices 6 and 7
            for script_name in linux_only_scripts: # Use the determined list: ['slurm', 'slim_slurm']
                btn = self.create_script_button(script_name)
                
                # Set initial style: Dark red text, warm yellow background, and disabled
                btn.setStyleSheet(
                    btn.styleSheet() + 
                    """
                    QPushButton:disabled { 
                        color: #8B0000; 
                        background-color: #FCCB06;
                    }
                    """
                )
                btn.setEnabled(False)
                
                self.linux_buttons.append(btn)
                row_slurm_layout.addWidget(btn)
            
            # Ensure the 2x2 structure is maintained by adding a stretch if necessary
            if len(linux_only_scripts) == 1:
                row_slurm_layout.addStretch(1)
            
            linux_only_scripts_layout.addWidget(self.linux_buttons_container)

            # 8. Add the content container to the script list (scripts_layout)
            scripts_layout.addWidget(self.linux_only_scripts_container) 

            # 9. Initialize state: hidden
            self.is_area_visible = False
            self.linux_only_scripts_container.hide()
        # =================================================================

        content_layout.addRow(scripts_container) # This holds all buttons and the collapsible header
        
        # --- Script Parameters Section --- 
        content_layout.addRow(QLabel("<hr>"))
        content_layout.addRow(QLabel("<b>Script Parameters</b>"))
        
        # Selected Script Display
        self.selected_script_label = QLabel("No script selected")
        self.selected_script_label.setStyleSheet("font-weight: bold; color: #2c3e50; padding: 5px;")
        content_layout.addRow("Selected Script:", self.selected_script_label)
        
        # Common parameters that might be used by multiple scripts
        self.verbose_checkbox = QPushButton("Verbose Mode")
        self.verbose_checkbox.setCheckable(True)
        self.verbose_checkbox.setStyleSheet("""
            QPushButton:checked {
                background-color: #06402B;
                color: white;
            }
            QPushButton {
                background-color: #8B0000;
                color: white;
            }
        """)
        content_layout.addRow("Verbose Output:", self.verbose_checkbox)
        
        # CPU cores parameter (used by test_sim, slurm, etc.)
        self.cores_input = QSpinBox(value=22, minimum=1, maximum=mp.cpu_count())
        content_layout.addRow("CPU Cores:", self.cores_input)
        
        # Script-specific parameters
        content_layout.addRow(QLabel('<span style="font-weight: 600;">Script-Specific Parameters</span>'))

        # Script-specific parameters container
        self.script_params_container = QWidget()
        self.script_params_layout = QFormLayout(self.script_params_container)
        content_layout.addRow(self.script_params_container)
        
        # Initially hide parameters container
        self.script_params_container.setVisible(False)
        
        # --- Make Tab Scrollable ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.content_widget)
        
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, 
            QSizePolicy.Policy.MinimumExpanding
        )
        
        # Set the QScrollArea as the main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        
        # Initialize script-specific parameter widgets
        self.init_script_parameters()
    
    def create_script_button(self, script_name):
        """Helper method to create script buttons"""
        display_name = self.SCRIPTS[script_name]
        btn = QPushButton(display_name)
        btn.setCheckable(True)
        btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #3320b5;
                color: white;
                border: 1px solid #27ae60;
            }
            QPushButton:hover:!checked {
                background-color: #3498db;
            }
            QPushButton:hover:checked {
                background-color: #00838a;
            }
            QPushButton {
                min-height: 30px;
            }
        """)
        
        btn.clicked.connect(lambda checked, s=script_name: self.select_script(s, checked))
        self.button_group.addButton(btn)
        self.script_buttons[script_name] = btn
        return btn
    
    def _toggle_linux_only_scripts(self):
        """Toggles the visibility of the Linux Only Scripts input fields and updates the +/- sign."""
        if self.is_area_visible: 
            self.linux_only_scripts_container.hide()
            self.linux_only_scripts_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.linux_only_scripts_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.linux_only_scripts_container.show()
            self.linux_only_scripts_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.linux_only_scripts_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_area_visible = not self.is_area_visible

    def init_script_parameters(self):
        """Initialize parameter widgets for each script type"""
        self.param_widgets = {}
        
        # Environment setup parameters
        setup_env_params = QWidget(self.script_params_container) # Parent to container
        setup_env_layout = QFormLayout(setup_env_params)
        
        self.env_manager_input = QComboBox()
        self.env_manager_input.addItems(["uv", "conda", "venv"])
        setup_env_layout.addRow("Package Manager:", self.env_manager_input)
        
        self.param_widgets["setup_env"] = setup_env_params
        
        # Add ALL parameter widgets to the main script_params_container's layout.
        # This ensures they are part of the widget tree and won't be deleted.
        self.script_params_layout.addWidget(setup_env_params) 
        
        # Create and pre-add the default/placeholder label
        self.default_params_label = QLabel("No specific parameters for this script.")
        self.script_params_layout.addWidget(self.default_params_label)

        # Initially hide all of them
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
            self.selected_script = None
            self.selected_script_label.setText("No script selected")
            self.hide_script_parameters()

    def clear_script_selection(self):
        """Clear the current script selection"""
        self.button_group.setExclusive(False)
        for button in self.script_buttons.values():
            button.setChecked(False)
        self.button_group.setExclusive(True)
        self.selected_script = None
        self.selected_script_label.setText("No script selected")
        self.hide_script_parameters()

    def show_script_parameters(self, script_name):
        """Show parameters specific to the selected script by toggling visibility."""
        self.script_params_container.setVisible(True)
        
        # 1. Hide ALL custom parameter widgets and the default label
        for widget in self.param_widgets.values():
            widget.hide()
        self.default_params_label.hide()
        
        # 2. Show the relevant widget
        if script_name in self.param_widgets:
            # Show the custom parameter widget for the selected script
            self.param_widgets[script_name].show()
        else:
            # Show the default label for scripts without custom parameters
            self.default_params_label.setText(f"No specific parameters for {self.SCRIPTS[script_name]}")
            self.default_params_label.show()

    def hide_script_parameters(self):
        """Hide the script parameters section and all parameter widgets."""
        self.script_params_container.setVisible(False)
        
        # Hide all children widgets as well to ensure a clean state
        for widget in self.param_widgets.values():
            widget.hide()
        self.default_params_label.hide()

    def get_params(self):
        """Get all parameters for the selected script"""
        if not self.selected_script:
            return {}
        
        base_params = {
            "script": self.selected_script,
            "verbose": self.verbose_checkbox.isChecked(),
            "cores": self.cores_input.value(),
        }
        
        # Add script-specific parameters
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
        
        # Base command
        if script_name in ["test_sim", "train", "hyperparam_optim", "gen_data"]:
            command_parts.append(f"python main.py {script_name.replace('_', ' ')}")
        elif script_name == "setup_env":
            # Use 'setup_env.sh' for Linux, 'setup_env.bat' for others (like Windows)
            if sys.platform.startswith('linux'):
                command_parts.append(f"scripts/setup_env.sh {params.get('manager', 'uv')}")
            else:
                command_parts.append(f"scripts\\setup_env.bat {params.get('manager', 'uv')}")
        elif script_name in ["slim_slurm", "slurm"]:
            command_parts.append(f"bash {script_name}.sh")
        
        # Add common parameters
        if params.get("verbose"):
            command_parts.append("--verbose")
        
        if params.get("cores") and script_name in ["test_sim", "slurm"]:
            command_parts.append(f"-nc {params['cores']}")
        
        return " ".join(command_parts)
