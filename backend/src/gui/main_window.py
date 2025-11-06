import re
import sys
import subprocess

from PySide6.QtWidgets import (
    QComboBox, QTextEdit, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QApplication,
    QTabWidget, QPushButton, QWidget, QLabel, QMessageBox 
)
from .tabs import (
    RLCostsTab, RLDataTab, RLModelTab, FileSystemScriptsTab,
    GenDataGeneralTab, GenDataProblemTab, GenDataAdvancedTab,
    RLOptimizerTab, RLOutputTab, RLTrainingTab, TestSuiteTab,
    TestSimAdvancedTab, TestSimIOTab, FileSystemCryptographyTab,
    TestSimSettingsTab, TestSimPolicyParamsTab, FileSystemDeleteTab,
    EvalIOTab, EvalDataBatchingTab, EvalDecodingTab, EvalProblemTab,
    MetaRLTrainParserTab, HyperParamOptimParserTab, FileSystemUpdateTab,
)


class MainWindow(QWidget):
    def __init__(self, test_only=False, initial_window='Train Model', restart_callback=None, initial_tab_index=0):
        super().__init__()
        self.test_only = test_only
        self.restart_callback = restart_callback
        self.setWindowTitle("Neural Combinatorial Optimization — Configuration GUI")
        self.resize(900, 700)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Neural Combinatorial Optimization")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title)
        
        # Command selection
        command_layout = QHBoxLayout()
        command_layout.addWidget(QLabel("Command:"))
        self.command_combo = QComboBox()
        self.command_combo.addItems(['Train Model', 'Generate Data', 'Evaluate', 'Test Simulator', 'File System Tools', 'Other Tools'])
        self.command_combo.currentTextChanged.connect(self.on_command_changed)
        command_layout.addWidget(self.command_combo)
        command_layout.addStretch()
        main_layout.addLayout(command_layout)
        
        # Initialize ALL tab sets
        self.train_tabs_map = {
            "Data": RLDataTab(),
            "Model": RLModelTab(),
            "Training": RLTrainingTab(),
            "Optimizer": RLOptimizerTab(),
            "Cost Weights": RLCostsTab(),
            "Output": RLOutputTab(),
            "Hyper-Parameter Optimization": HyperParamOptimParserTab(),
            "Meta-Learning": MetaRLTrainParserTab()
        }
        self.gen_data_tabs_map = {
            "General Output": GenDataGeneralTab(),
            "Problem Definition": GenDataProblemTab(),
            "Advanced Settings": GenDataAdvancedTab()
        }
        self.test_sim_tabs_map = {
            "Simulator Settings": TestSimSettingsTab(),
            "Policy Parameters": TestSimPolicyParamsTab(),
            "IO Settings": TestSimIOTab(),
            "Advanced Settings": TestSimAdvancedTab()
        }
        self.eval_tabs_map = {
            'IO Settings': EvalIOTab(), 
            'Data Configurations': EvalDataBatchingTab(), 
            'Decoding Strategy': EvalDecodingTab(), 
            'Problem Definition': EvalProblemTab()
        }
        self.file_system_tabs_map = {
            "Update Settings": FileSystemUpdateTab(),
            "Delete Settings": FileSystemDeleteTab(),
            "Cryptography Settings": FileSystemCryptographyTab(),
        }
        self.other_tabs_map = {
            "Execute Script": FileSystemScriptsTab(),
            "Program Test Suite": TestSuiteTab()
        }
        self.all_tabs = {
            'Train Model': self.train_tabs_map,
            'Generate Data': self.gen_data_tabs_map,
            'Evaluate': self.eval_tabs_map,
            'Test Simulator': self.test_sim_tabs_map,
            'File System Tools': self.file_system_tabs_map,
            'Other Tools': self.other_tabs_map
        }

        # Tabs container
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Preview
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("Generated Command:"))
        self.preview = QTextEdit()  # <--- WIDGET CREATED HERE NOW
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(150)
        preview_layout.addWidget(self.preview)

        # Restore logic starts here 
        # -----------------------------------------------------

        # 1. Restore the command combo (This triggers on_command_changed -> update_preview)
        # We temporarily block signals to prevent redundant calls during initialization.
        self.command_combo.blockSignals(True)
        self.command_combo.setCurrentText(initial_window) 
        self.command_combo.blockSignals(False)
        
        # Initial tab setup based on restored command (to load the correct tabs)
        self.setup_tabs(initial_window)
        
        # 2. Restore the specific tab index 
        if initial_tab_index is not None:
            self.tabs.setCurrentIndex(initial_tab_index)

        # -----------------------------------------------------
        # Now that self.preview exists, you can call update_preview for the first time
        # This initial update is crucial for the restoration.
        self.update_preview() 
        
        # Connect the tab change signal to refresh the preview
        self.tabs.currentChanged.connect(self.update_preview)
        
        # Preview and controls (now placing the created widgets into the layout)
        lower_layout = QHBoxLayout()
        # Preview
        lower_layout.addLayout(preview_layout, 3) # Use the layout we created above

        # Controls
        control_layout = QVBoxLayout()

        # Add a new button to the control layout
        self.reopen_button = QPushButton("Close and Reopen GUI")
        self.reopen_button.clicked.connect(self.close_and_reopen)
        control_layout.insertWidget(0, self.reopen_button)

        # Style for Reopen Button (Warning/Action Color - Bright Red/Orange)
        self.reopen_button.setStyleSheet("""
            QPushButton {
                background-color: #E91E63; /* Deep Pink */
                color: white; 
                font-weight: bold;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #C2185B; /* Darker Pink */
            }
        """)

        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.clicked.connect(self.update_preview)

        # Style for Refresh Button (Deep Teal)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #1ABC9C; /* Turquoise */
                color: white;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #16A085; /* Darker Turquoise */
            }
        """)
        control_layout.addWidget(self.refresh_button)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)

        # Style for Copy Button (Muted Indigo/Periwinkle)
        self.copy_button.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6; /* Amethyst (lighter purple/indigo) */
                color: white;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #8E44AD; /* Darker Amethyst */
            }
        """)
        control_layout.addWidget(self.copy_button)
        
        self.run_button = QPushButton("Run Command (simulated)" if self.test_only else "Run Command")
        self.run_button.clicked.connect(self.run_command)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50; /* Deep Slate Blue / Wet Asphalt */
                color: white; 
                font-weight: bold;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #34495E; /* Darker Slate Blue */
            }
            QPushButton:disabled {
                background-color: #FFFFCC; /* Very Pale Yellow */
                color: #7D0000; /* Dark Scarlet Red */
                font-weight: bold;
            }
        """)
        control_layout.addWidget(self.run_button)
        control_layout.addStretch()
        
        suffix_notes = "\n• Run is simulated here." if self.test_only else ""
        notes_label = QLabel("Notes:\n• Leave fields empty to use defaults\n• Cost weights of 0 are ignored\n• Use Refresh to update preview" + suffix_notes)
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet("font-size: 10px; color: gray;")
        control_layout.addWidget(notes_label)
        
        lower_layout.addLayout(control_layout, 1)
        
        main_layout.addLayout(lower_layout)
        
        self.setLayout(main_layout) 
        
        # Initial preview
        self.update_preview()

        # Connect the tab change signal to refresh the preview
        self.tabs.currentChanged.connect(self.update_preview)
    
    def close_and_reopen(self):
        """Hides the current window and triggers the external restart."""
        # 1. Save the current state before closing
        current_tab_index = self.tabs.currentIndex()
        
        self.hide() # Hide the window immediately
        if self.restart_callback:
            # 2. Pass the saved state back to the launch function
            # The signature of restart_callback (launch_gui) needs to match this.
            self.restart_callback(
                test_only=self.test_only, 
                tab_index=current_tab_index
            )
    
    def get_actual_command(self, main_command_display):
        """Maps the display name to the command-line argument dynamically."""
        command_mapping = {
            'Train Model': 'train',
            'Generate Data': 'gen_data',
            'Evaluate': 'eval',
            'Test Simulator': 'test_sim',
        }
        actual_command = command_mapping.get(main_command_display)
        if main_command_display == 'Train Model':
            current_tab_title = self.tabs.tabText(self.tabs.currentIndex())
            if current_tab_title == "Hyper-Parameter Optimization":
                actual_command = 'hp_optim'
            elif current_tab_title == "Meta-Learning":
                actual_command = 'mrl_train'
            # Otherwise, actual_command remains 'train'
        elif main_command_display == 'File System Tools':
            current_tab_title = self.tabs.tabText(self.tabs.currentIndex())
            actual_command = 'file_system '
            if current_tab_title == "Cryptography Settings":
                actual_command += 'cryptography'
            elif current_tab_title == "Update Settings":
                actual_command += 'update'
            elif current_tab_title == "Delete Settings":
                actual_command += 'delete'
        elif main_command_display == 'Other Tools':
            current_tab_title = self.tabs.tabText(self.tabs.currentIndex())
            if current_tab_title == 'Execute Script':
                actual_command = 'scripts'
            elif current_tab_title == 'Program Test Suite':
                actual_command = 'test_suite'

        if actual_command == 'scripts':
            # Get parameters from the active ScriptsTab widget
            script_tab_widget = self.other_tabs_map['Execute Script']
            if hasattr(script_tab_widget, 'get_params'):
                # Call get_params to fetch all arguments for the script execution
                script_params = script_tab_widget.get_params()
                
                # Pop the script name (which should be stored under the key 'script')
                script_name = script_params.pop('script', None) 
                
                # Crucially, the MainWindow.update_preview logic must now rely on 
                # collecting parameters from the active tab in the "File System Tools" section.
                if script_name:
                    # Replace 'scripts' with 'scripts/script_name.sh'
                    if sys.platform.startswith('linux'):
                        actual_command = f'scripts/{script_name}'
                        if not actual_command.endswith('.sh'):
                            actual_command += '.sh' # Assuming the script requires .sh extension
                    elif sys.platform.startswith('win'):
                        actual_command = f'scripts\{script_name}'
                        if not actual_command.endswith('.bat'):
                            actual_command += '.bat' # Assuming the script requires .bat extension         
                
        # Default to the mapped command or raise an error if needed
        return actual_command if actual_command else main_command_display

    def setup_tabs(self, command):
        """Dynamically loads the correct set of tabs based on the command."""
        # Remove existing tabs
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)

        # Add new tabs
        if command in self.all_tabs:
            tab_set = self.all_tabs[command]
            for title, tab_widget in tab_set.items():
                self.tabs.addTab(tab_widget, title)
        else:
            # Placeholder for other commands
            placeholder = QWidget()
            placeholder.setLayout(QVBoxLayout())
            placeholder.layout().addWidget(QLabel(f"GUI for '{command}' coming soon."))
            self.tabs.addTab(placeholder, "Info")

    def on_command_changed(self, command):
        """Handle command selection change and update UI."""
        self.setup_tabs(command)
        self.update_preview()

    def update_preview(self):
        """Update the command preview by collecting parameters from current tabs."""
        # Get the actual command based on the main selection and current tab
        main_command_display = self.command_combo.currentText()
        actual_command = self.get_actual_command(main_command_display)

        # Collect parameters based on the main command 🚀
        all_params = {}
        if main_command_display in ['File System Tools', 'Other Tools']:
            # Only get parameters from the CURRENT tab
            current_tab_widget = self.tabs.currentWidget()
            if hasattr(current_tab_widget, 'get_params'):
                all_params.update(current_tab_widget.get_params())
        elif main_command_display == 'Train Model':
            # For Train Model: Base parameters are always included, but HPO/Meta-Learning are conditional
            current_tab_title = self.tabs.tabText(self.tabs.currentIndex())
            for title, tab_widget in self.train_tabs_map.items():
                if hasattr(tab_widget, 'get_params'):
                    # Always include params from the base tabs (Data, Model, Training, etc.)
                    is_base_tab = title not in ["Hyper-Parameter Optimization", "Meta-Learning"]
                    
                    # Include HPO or Meta-Learning params ONLY if that tab is currently active
                    is_active_special_tab = (
                        title == current_tab_title and 
                        title in ["Hyper-Parameter Optimization", "Meta-Learning"]
                    )
                    
                    if is_base_tab or is_active_special_tab:
                        all_params.update(tab_widget.get_params())
        else:
            # For all other main commands, we iterate over all loaded tabs for that section
            for i in range(self.tabs.count()):
                tab_widget = self.tabs.widget(i)
                if hasattr(tab_widget, 'get_params'):
                    all_params.update(tab_widget.get_params())

        # Build command string        
        if actual_command.startswith('scripts/') and 'script' in all_params:
            # Pop the argument so it is not added as a --script flag in the final command string
            all_params.pop('script')
            cmd_parts = [f"bash {actual_command}"] if sys.platform.startswith('linux') else [actual_command]
        else:
            cmd_parts = [f"python main.py {actual_command}"] # Use actual_command here
        
        for key, value in all_params.items():
            if value is None or value == "":
                continue

            # Special case for boolean flags (True/False values)
            if isinstance(value, bool):
                if key in ['mask_inner', 'mask_logits'] and value is False:
                    # Specific "no" flag handling
                    cmd_parts.append(f"--no_{key.replace('_', '-')}")
                elif value is True:
                    # Standard flag when True
                    cmd_parts.append(f"--{key.replace('_', '-')}")
                # Ignore False boolean values unless it's a specific 'no-' flag
            
            # Numeric values
            elif isinstance(value, (int, float)):
                # Handle is_gaussian 0/1 explicitly if needed, but QSpinBox handles this fine
                cmd_parts.append(f"--{key.replace('_', '-')} {value}")
            
            # String values (including space-separated lists like graph_sizes)
            elif isinstance(value, str):
                # Check for spaces or quotes, which indicates a complex string or list
                if ' ' in value or '"' in value or "'" in value:
                    # These keys contain space-separated arguments that should NOT be quoted.
                    list_keys = [
                        'focus_graph', 'input_keys',
                        'graph_sizes', 'data_distributions',
                        'policies', 'pregular_level', 'plastminute_cf', 
                        'lookahead_configs', 'gurobi_param', 'hexaly_param',
                    ]
                    if key in list_keys:
                        # Append list elements directly: --key 2 3 6
                        parts = value.split()
                        cmd_parts.append(f"--{key}")
                        cmd_parts.extend(parts)
                    else:
                        # Standard string argument (e.g., a path or a complex string that requires quotes)
                        cmd_parts.append(f"--{key} '{value}'") 
                elif key == 'update_operation':
                    cmd_parts.append(f"--{key} '{value}'")
                else:
                    # Simple strings/numbers without spaces
                    cmd_parts.append(f"--{key} {value}")
        command_str = " \\\n  ".join(cmd_parts)
        self.preview.setPlainText(command_str)
        
    def copy_to_clipboard(self):
        """Copy command to clipboard"""
        self.update_preview()
        clipboard = QApplication.clipboard()
        clipboard.setText(self.preview.toPlainText())
        # Use a non-blocking message box instead of alert/confirm
        QMessageBox.information(self, "Copied:", self.preview.toPlainText())
        
    def run_command(self):
        """Simulate command execution (actual execution is environment-dependent)"""
        self.run_button.setDisabled(True)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #FFFFCC;
                color: #7D0000;
                font-weight: bold;
                border: none;
                padding: 6px;
            }
        """)
        self.update_preview()

        regex = r"(?<!-)-(?!-)"
        command_str = self.preview.toPlainText()
        command_str = re.sub(regex, '_', command_str)
        if self.test_only:
            QMessageBox.information(
                self, 
                "Command Simulation", 
                f"The following command would be executed:\n\n{command_str}\n\n(Execution is simulated in this environment)."
            )
        else:
            print(f"Executing: {command_str}")
            try:
                # Replace the continuation markers for shell execution
                shell_command = command_str.replace(" \\\n  ", " ")
                # Use subprocess.run for simple execution and capturing output
                result = subprocess.run(
                    shell_command, 
                    shell=True, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                print("\n--- Command Output ---")
                print(result.stdout)
                if result.stderr:
                    print("\n--- Command Error Output (if any) ---")
                    print(result.stderr)
                print("------------------------\nCommand execution finished successfully.")
            except subprocess.CalledProcessError as e:
                print(f"\nCommand failed with exit code {e.returncode}")
                print("\n--- Command STDOUT ---")
                print(e.stdout)
                print("\n--- Command STDERR ---")
                print(e.stderr)
        self.run_button.setDisabled(False)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50; /* Deep Slate Blue */
                color: white; 
                font-weight: bold;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
