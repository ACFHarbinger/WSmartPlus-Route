import re
import sys
import subprocess

from PySide6.QtWidgets import (
    QComboBox, QTextEdit, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QApplication,
    QTabWidget, QPushButton, QWidget, QLabel, QMessageBox
)
# Assuming these imports exist and are correct for your project
from .tabs import (
    RLCostsTab, RLDataTab, RLModelTab, RunScriptsTab,
    GenDataGeneralTab, GenDataProblemTab, GenDataAdvancedTab,
    RLOptimizerTab, RLOutputTab, RLTrainingTab, TestSuiteTab,
    TestSimAdvancedTab, TestSimIOTab, FileSystemCryptographyTab,
    TestSimSettingsTab, TestSimPolicyParamsTab, FileSystemDeleteTab,
    EvalIOTab, EvalDataBatchingTab, EvalDecodingTab, EvalProblemTab,
    MetaRLTrainParserTab, HyperParamOptimParserTab, FileSystemUpdateTab,
)
from .styles import (
    TEXT_COLOR, CONTAINER_BG_COLOR, PRIMARY_ACCENT_COLOR,
    BACKGROUND_COLOR, MUTED_TEXT_COLOR, PRIMARY_HOVER_COLOR,
    SECONDARY_ACCENT_COLOR, SECONDARY_HOVER_COLOR, BORDER_COLOR
)


class MainWindow(QWidget):
    def __init__(self, test_only=False, initial_window='Train Model', restart_callback=None, initial_tab_index=0):
        super().__init__()
        self.test_only = test_only
        self.restart_callback = restart_callback
        self.setWindowTitle("Machine Learning Models and Operations Research Solvers for Combinatorial Optimization Problems")
        self.setMinimumSize(1080, 900)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Apply Global Stylesheet for a Modern Light Theme
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 13px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
                padding-bottom: 2px;
            }}
            QComboBox, QTextEdit {{
                border: 1px solid {BORDER_COLOR};
                padding: 6px 8px;
                background-color: {CONTAINER_BG_COLOR};
                selection-background-color: {PRIMARY_ACCENT_COLOR};
                color: {TEXT_COLOR};
                border-radius: 5px; /* Rounded corners */
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: {BORDER_COLOR};
                border-left-style: solid; 
            }}
            QTabWidget::pane {{ /* The tab content area */
                border: 1px solid {BORDER_COLOR};
                background-color: {CONTAINER_BG_COLOR};
                border-radius: 5px;
                border-top-left-radius: 0; /* Align with tab */
            }}
            QTabBar::tab {{
                background: {BACKGROUND_COLOR};
                color: {MUTED_TEXT_COLOR};
                font-weight: 500;
                padding: 10px 18px;
                border: 1px solid {BORDER_COLOR};
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }}
            QTabBar::tab:selected {{
                background: {CONTAINER_BG_COLOR};
                color: {PRIMARY_ACCENT_COLOR};
                font-weight: 600;
                /* Trick to make tab connect to pane */
                border-bottom: 2px solid {CONTAINER_BG_COLOR}; 
                margin-bottom: -2px;
            }}
            QTabBar::tab:hover {{
                color: {TEXT_COLOR};
            }}
            
            /* General Button Styling */
            QPushButton {{
                color: white;
                font-weight: 600;
                border: none;
                padding: 10px 12px;
                border-radius: 5px;
            }}
            QPushButton:pressed {{
                /* Add a subtle press effect */
                padding-top: 11px;
                padding-bottom: 9px;
            }}
            QPushButton:disabled {{
                background-color: #BDC3C7; /* Muted gray */
                color: #7F8C8D;
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12) # Add space between main sections
        main_layout.setContentsMargins(15, 15, 15, 15) # Add padding to window

        # Title
        title = QLabel("Machine Learning and Operations Research for Combinatorial Optimization")
        title.setStyleSheet(f"""
            font-size: 26px; 
            font-weight: 700; 
            padding-bottom: 5px;
            color: {TEXT_COLOR}; 
        """)
        main_layout.addWidget(title)

        # Command selection
        command_layout = QHBoxLayout()
        command_label = QLabel("Select Command:")
        command_label.setStyleSheet("font-weight: 600;")
        command_layout.addWidget(command_label)
        
        self.command_combo = QComboBox()
        self.command_combo.addItems(['Train Model', 'Generate Data', 'Evaluate', 'Test Simulator', 'File System Tools', 'Other Tools'])
        self.command_combo.currentTextChanged.connect(self.on_command_changed)
        self.command_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        command_layout.addWidget(self.command_combo)
        command_layout.addStretch()
        main_layout.addLayout(command_layout)

        # --- Tab Initialization (No visual change needed here, keeping original structure) ---
        self.train_tabs_map = {
            "Data": RLDataTab(), "Model": RLModelTab(), "Training": RLTrainingTab(),
            "Optimizer": RLOptimizerTab(), "Cost Weights": RLCostsTab(), "Output": RLOutputTab(),
            "Hyper-Parameter Optimization": HyperParamOptimParserTab(), "Meta-Learning": MetaRLTrainParserTab()
        }
        self.gen_data_tabs_map = {
            "General Output": GenDataGeneralTab(), "Problem Definition": GenDataProblemTab(), "Advanced Settings": GenDataAdvancedTab()
        }
        self.test_sim_tabs_map = {
            "Simulator Settings": TestSimSettingsTab(), "Policy Parameters": TestSimPolicyParamsTab(),
            "IO Settings": TestSimIOTab(), "Advanced Settings": TestSimAdvancedTab()
        }
        self.eval_tabs_map = {
            'IO Settings': EvalIOTab(), 'Data Configurations': EvalDataBatchingTab(),
            'Decoding Strategy': EvalDecodingTab(), 'Problem Definition': EvalProblemTab()
        }
        self.file_system_tabs_map = {
            "Update Settings": FileSystemUpdateTab(), "Delete Settings": FileSystemDeleteTab(),
            "Cryptography Settings": FileSystemCryptographyTab(),
        }
        self.other_tabs_map = {
            "Execute Script": RunScriptsTab(), "Program Test Suite": TestSuiteTab()
        }
        self.all_tabs = {
            'Train Model': self.train_tabs_map, 'Generate Data': self.gen_data_tabs_map,
            'Evaluate': self.eval_tabs_map, 'Test Simulator': self.test_sim_tabs_map,
            'File System Tools': self.file_system_tabs_map, 'Other Tools': self.other_tabs_map
        }
        # --- End Tab Initialization ---

        # Tabs container
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Preview
        preview_layout = QVBoxLayout()
        preview_str = "Generated Command (Read-Only):" if self.test_only else "Generated Command:" 
        preview_label = QLabel(preview_str)
        preview_label.setStyleSheet("font-weight: 600; padding-top: 5px;")
        preview_layout.addWidget(preview_label)
        
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(180)
        self.preview.setStyleSheet(f"""
            QTextEdit {{
                background-color: #FAFAFA; /* Slightly different white for code */
                color: #333333;
                border: 1px solid {BORDER_COLOR};
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                padding: 10px;
                border-radius: 5px;
            }}
        """)
        preview_layout.addWidget(self.preview)

        # Restore logic starts here
        # -----------------------------------------------------
        self.command_combo.blockSignals(True)
        self.command_combo.setCurrentText(initial_window)
        self.command_combo.blockSignals(False)
        self.setup_tabs(initial_window)
        if initial_tab_index is not None:
            self.tabs.setCurrentIndex(initial_tab_index)
        # -----------------------------------------------------

        self.update_preview()
        self.tabs.currentChanged.connect(self.update_preview)

        # Preview and controls
        lower_layout = QHBoxLayout()
        lower_layout.setSpacing(15)
        # Preview
        lower_layout.addLayout(preview_layout, 3)

        # Controls
        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)

        # Reopen Button (Critical Action)
        self.reopen_button = QPushButton("Close and Reopen GUI")
        self.reopen_button.clicked.connect(self.close_and_reopen)
        self.reopen_button.setStyleSheet(f"""
            QPushButton {{ background-color: {SECONDARY_ACCENT_COLOR}; }}
            QPushButton:hover {{ background-color: {SECONDARY_HOVER_COLOR}; }}
        """)
        control_layout.addWidget(self.reopen_button)

        # --- Secondary/Utility Button Style ---
        secondary_button_style = f"""
            QPushButton {{
                background-color: #ECF0F1;
                color: {TEXT_COLOR};
            }}
            QPushButton:hover {{
                background-color: {BORDER_COLOR};
            }}
        """
        
        # Refresh Button (Utility)
        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.clicked.connect(self.update_preview)
        self.refresh_button.setStyleSheet(secondary_button_style)
        control_layout.addWidget(self.refresh_button)

        # Copy Button (Utility)
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.copy_button.setStyleSheet(secondary_button_style)
        control_layout.addWidget(self.copy_button)

        # Run Button (Primary Action)
        self.run_button = QPushButton("Run Command (simulated)" if self.test_only else "Run Command")
        self.run_button.clicked.connect(self.run_command)
        self.run_button.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {PRIMARY_ACCENT_COLOR}; 
                padding: 12px; /* Bigger padding for primary action */
            }}
            QPushButton:hover {{ background-color: {PRIMARY_HOVER_COLOR}; }}
        """)
        control_layout.addWidget(self.run_button)
        control_layout.addStretch()

        # Notes Label
        suffix_notes = "\n• Run is simulated here." if self.test_only else ""
        notes_label = QLabel("Notes:\n• Leave fields empty to use defaults\n• Cost weights of 0 are ignored\n• Use Refresh to update preview" + suffix_notes)
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet(f"font-size: 11px; color: {MUTED_TEXT_COLOR}; padding-top: 5px;")
        control_layout.addWidget(notes_label)

        lower_layout.addLayout(control_layout, 1)

        main_layout.addLayout(lower_layout)

    # --- NO CHANGES REQUIRED FOR LOGIC METHODS BELOW THIS LINE ---
    # The styling changes are all self-contained in __init__
    # The logic for enabling/disabling the Run button is now handled
    # by the :disabled pseudo-state in the main stylesheet,
    # so the setStyleSheet calls in run_command are no longer needed.

    def close_and_reopen(self):
        """Hides the current window and triggers the external restart."""
        current_tab_index = self.tabs.currentIndex()

        self.hide() # Hide the window immediately
        if self.restart_callback:
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
        # The :disabled state in the main stylesheet now handles the style.
        self.run_button.setDisabled(True)
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
        
        # Re-enable the button. The style is automatically restored.
        self.run_button.setDisabled(False)
