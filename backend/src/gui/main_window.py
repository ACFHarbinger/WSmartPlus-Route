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
    RLCostsTab, RLDataTab, RLModelTab, FileSystemScriptsTab,
    GenDataGeneralTab, GenDataProblemTab, GenDataAdvancedTab,
    RLOptimizerTab, RLOutputTab, RLTrainingTab, TestSuiteTab,
    TestSimAdvancedTab, TestSimIOTab, FileSystemCryptographyTab,
    TestSimSettingsTab, TestSimPolicyParamsTab, FileSystemDeleteTab,
    EvalIOTab, EvalDataBatchingTab, EvalDecodingTab, EvalProblemTab,
    MetaRLTrainParserTab, HyperParamOptimParserTab, FileSystemUpdateTab,
)


# --- MODERN STYLING CONSTANTS ---
# Based on a dark, professional theme with a vibrant accent color.
PRIMARY_ACCENT_COLOR = "#00BFA5"  # Teal/Cyan for emphasis (e.g., Refresh, active tabs)
SECONDARY_ACCENT_COLOR = "#FF5252"  # Red/Pink for critical actions (e.g., Reopen/Reset)
BACKGROUND_COLOR = "#2C3E50"      # Dark Slate Blue/Wet Asphalt (Main Background)
FOREGROUND_COLOR = "#ECF0F1"      # Light Gray/Cloud (Text/Foreground Elements)
CONTAINER_BG_COLOR = "#34495E"    # Darker container background
TEXT_COLOR = FOREGROUND_COLOR

class MainWindow(QWidget):
    def __init__(self, test_only=False, initial_window='Train Model', restart_callback=None, initial_tab_index=0):
        super().__init__()
        self.test_only = test_only
        self.restart_callback = restart_callback
        self.setWindowTitle("Neural Combinatorial Optimization — Configuration GUI")
        self.resize(900, 700)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Apply Global Stylesheet for a Dark Theme
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-family: Arial, Helvetica, sans-serif;
                font-size: 12px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QComboBox, QTextEdit {{
                border: 1px solid {CONTAINER_BG_COLOR};
                padding: 5px;
                background-color: {CONTAINER_BG_COLOR};
                selection-background-color: {PRIMARY_ACCENT_COLOR};
                color: {TEXT_COLOR};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid; 
            }}
            QTabWidget::pane {{ /* The tab content area */
                border: 1px solid {CONTAINER_BG_COLOR};
                background-color: {CONTAINER_BG_COLOR};
            }}
            QTabBar::tab {{
                background: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                padding: 8px 15px;
                border: 1px solid {CONTAINER_BG_COLOR};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background: {CONTAINER_BG_COLOR};
                color: {PRIMARY_ACCENT_COLOR};
                font-weight: bold;
                border-top: 3px solid {PRIMARY_ACCENT_COLOR};
            }}
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10) # Add a little space between sections

        # Title
        title = QLabel("Neural Combinatorial Optimization")
        title.setStyleSheet(f"""
            font-size: 24px; 
            font-weight: bold; 
            padding: 10px 0;
            color: {PRIMARY_ACCENT_COLOR}; 
        """)
        main_layout.addWidget(title)

        # Command selection
        command_layout = QHBoxLayout()
        command_layout.addWidget(QLabel("Select Command:"))
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
            "Execute Script": FileSystemScriptsTab(), "Program Test Suite": TestSuiteTab()
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
        if self.test_only:
            preview_layout.addWidget(QLabel("Generated Command (Read-Only):"))
        else:
            preview_layout.addWidget(QLabel("Generated Command:"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(self.test_only)
        self.preview.setMaximumHeight(180) # Increased height slightly
        self.preview.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1E2B38; /* Slightly darker for a code look */
                color: #A9B7C6; /* Light blue-gray for code */
                border: 1px solid {PRIMARY_ACCENT_COLOR};
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11px;
                padding: 10px;
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
        # Preview
        lower_layout.addLayout(preview_layout, 3)

        # Controls
        control_layout = QVBoxLayout()

        # Helper function for common button styling
        def get_button_style(bg_color, hover_color, text_color="white"):
            return f"""
                QPushButton {{
                    background-color: {bg_color};
                    color: {text_color};
                    font-weight: bold;
                    border: none;
                    padding: 8px;
                    border-radius: 4px; /* Rounded corners */
                    text-transform: uppercase;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QPushButton:pressed {{
                    background-color: {bg_color};
                }}
                QPushButton:disabled {{
                    background-color: #5D6D7E; /* Muted gray for disabled state */
                    color: #AAB7B8;
                }}
            """

        # Reopen Button (Critical Action)
        self.reopen_button = QPushButton("Close and Reopen GUI")
        self.reopen_button.clicked.connect(self.close_and_reopen)
        self.reopen_button.setStyleSheet(get_button_style(SECONDARY_ACCENT_COLOR, "#D32F2F"))
        control_layout.addWidget(self.reopen_button)

        # Refresh Button (Utility/Primary Accent)
        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.clicked.connect(self.update_preview)
        self.refresh_button.setStyleSheet(get_button_style("#7D8D9D", "#62707E"))
        control_layout.addWidget(self.refresh_button)

        # Copy Button (Utility)
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.copy_button.setStyleSheet(get_button_style("#7D8D9D", "#62707E")) # Soft Gray/Blue
        control_layout.addWidget(self.copy_button)

        # Run Button (Primary Action)
        self.run_button = QPushButton("Run Command (simulated)" if self.test_only else "Run Command")
        self.run_button.clicked.connect(self.run_command)
        # Separate style for run button to easily manage disabled state
        self.run_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {PRIMARY_ACCENT_COLOR};
                color: white;
                font-weight: bold;
                border: none;
                padding: 10px; /* Bigger padding */
                border-radius: 4px;
                text-transform: uppercase;
            }}
            QPushButton:hover {{
                background-color: #00897B; /* Darker accent */
            }}
            QPushButton:disabled {{
                background-color: #5D6D7E; /* Muted gray for disabled state */
                color: #AAB7B8;
            }}
        """)
        control_layout.addWidget(self.run_button)
        control_layout.addStretch()

        # Notes Label
        suffix_notes = "\n• Run is simulated here." if self.test_only else ""
        notes_label = QLabel("Notes:\n• Leave fields empty to use defaults\n• Cost weights of 0 are ignored\n• Use Refresh to update preview" + suffix_notes)
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet("font-size: 10px; color: #95A5A6; padding-top: 5px;") # Lighter gray for notes
        control_layout.addWidget(notes_label)

        lower_layout.addLayout(control_layout, 1)

        main_layout.addLayout(lower_layout)

        self.setLayout(main_layout)

        # Final update call (redundant, but safe)
        self.update_preview()
        # Connect tab change signal (re-connect is not needed, but doesn't hurt)
        self.tabs.currentChanged.connect(self.update_preview)

    # All methods below (close_and_reopen, get_actual_command, setup_tabs, 
    # on_command_changed, update_preview, copy_to_clipboard, run_command)
    # remain the same as the original, save for potential stylistic updates 
    # to run_command's button re-enabling logic to match the new style.

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
        # Disable button with the disabled style
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

        # Re-enable button with the original style
        self.run_button.setDisabled(False)