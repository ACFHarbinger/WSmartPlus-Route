import re
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QComboBox, QTextEdit, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QApplication,
    QTabWidget, QPushButton, QWidget, QLabel, QMessageBox
)
from . import SimulationResultsWindow
from ..tabs import (
    InputAnalysisTab, OutputAnalysisTab,
    RLCostsTab, RLDataTab, RLModelTab, RunScriptsTab,
    GenDataGeneralTab, GenDataProblemTab, GenDataAdvancedTab,
    RLOptimizerTab, RLOutputTab, RLTrainingTab, TestSuiteTab,
    TestSimAdvancedTab, TestSimIOTab, FileSystemCryptographyTab,
    TestSimSettingsTab, TestSimPolicyParamsTab, FileSystemDeleteTab,
    EvalIOTab, EvalDataBatchingTab, EvalDecodingTab, EvalProblemTab,
    MetaRLTrainParserTab, HyperParamOptimParserTab, FileSystemUpdateTab,
)
from ..styles.globals import (
    BORDER_COLOR, MUTED_TEXT_COLOR,
    TEXT_COLOR, LIGHT_QSS, DARK_QSS, 
)


class MainWindow(QWidget):
    def __init__(self, test_only=False, initial_window='Train Model', restart_callback=None, initial_tab_index=0):
        super().__init__()
        self.process = None
        self.output_buffer = ""
        self.results_window = None # Stores the SimulationResultsWindow instance
        self.test_only = test_only
        self.restart_callback = restart_callback
        self.setWindowTitle("Machine Learning Models and Operations Research Solvers for Combinatorial Optimization Problems")
        self.setMinimumSize(1080, 900)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Theme tracking
        self.current_theme = 'light'

        # Apply Global Stylesheet for a Modern Light Theme
        self.setStyleSheet(LIGHT_QSS)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12) # Add space between main sections
        main_layout.setContentsMargins(15, 15, 15, 15) # Add padding to window

        # Title
        self.title_label = QLabel("Machine Learning and Operations Research for Combinatorial Optimization")
        self.title_label.setObjectName("mainTitleLabel")
        main_layout.addWidget(self.title_label)

        # Command selection
        command_layout = QHBoxLayout()

        self.command_label = QLabel("Select Command:")
        self.command_label.setObjectName("commandSelectLabel")
        command_layout.addWidget(self.command_label)
        
        self.command_combo = QComboBox()
        # --- CHANGED: Added 'Analysis' to the list ---
        self.command_combo.addItems(['Train Model', 'Generate Data', 'Evaluate', 'Test Simulator', 'Data Analysis', 'File System Tools', 'Other Tools'])
        self.command_combo.currentTextChanged.connect(self.on_command_changed)
        self.command_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        command_layout.addWidget(self.command_combo)
        
        command_layout.addStretch()

        # --- Theme Toggle Button ---
        self.theme_toggle_button = QPushButton("🎨")
        self.theme_toggle_button.setObjectName("themeToggleButton")
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        self.theme_toggle_button.setToolTip("Toggle Light/Dark Mode")
        self.theme_toggle_button.setFixedSize(25, 25)
        command_layout.addWidget(self.theme_toggle_button)
        
        main_layout.addLayout(command_layout)

        self.train_tabs_map = {
            "Data": RLDataTab(), "Model": RLModelTab(), "Training": RLTrainingTab(),
            "Optimizer": RLOptimizerTab(), "Cost Weights": RLCostsTab(), "Output": RLOutputTab(),
            "Hyper-Parameter Optimization": HyperParamOptimParserTab(), "Meta-Learning": MetaRLTrainParserTab()
        }
        self.gen_data_tabs_map = {
            "General Output": GenDataGeneralTab(), "Problem Definition": GenDataProblemTab(), "Advanced Settings": GenDataAdvancedTab()
        }
        
        settings_tab = TestSimSettingsTab()
        io_tab = TestSimIOTab(settings_tab=settings_tab) 
        self.test_sim_tabs_map = {
            "Simulator Settings": settings_tab, 
            "Policy Parameters": TestSimPolicyParamsTab(),
            "IO Settings": io_tab, 
            "Advanced Settings": TestSimAdvancedTab()
        }

        self.eval_tabs_map = {
            'IO Settings': EvalIOTab(), 'Data Configurations': EvalDataBatchingTab(),
            'Decoding Strategy': EvalDecodingTab(), 'Problem Definition': EvalProblemTab()
        }
        
        self.analysis_tabs_map = {
            "Input Analysis": InputAnalysisTab(),
            "Output Analysis": OutputAnalysisTab()
        }

        self.file_system_tabs_map = {
            "Update Settings": FileSystemUpdateTab(), "Delete Settings": FileSystemDeleteTab(),
            "Cryptography Settings": FileSystemCryptographyTab(),
        }

        self.other_tabs_map = {
            "Execute Script": RunScriptsTab(), "Program Test Suite": TestSuiteTab()
        }
        
        # --- CHANGED: Add 'Analysis' to all_tabs ---
        self.all_tabs = {
            'Train Model': self.train_tabs_map, 
            'Generate Data': self.gen_data_tabs_map,
            'Evaluate': self.eval_tabs_map, 
            'Test Simulator': self.test_sim_tabs_map,
            'Data Analysis': self.analysis_tabs_map,
            'File System Tools': self.file_system_tabs_map, 
            'Other Tools': self.other_tabs_map
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
        self.preview.setObjectName("previewTextEdit") 
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(180)
        preview_layout.addWidget(self.preview)

        # Restore logic starts here
        self.command_combo.blockSignals(True)
        self.command_combo.setCurrentText(initial_window)
        self.command_combo.blockSignals(False)
        self.setup_tabs(initial_window)
        if initial_tab_index is not None:
            self.tabs.setCurrentIndex(initial_tab_index)

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

        # Reopen Button 
        self.reopen_button = QPushButton("Close and Reopen GUI")
        self.reopen_button.setObjectName("reopenButton")
        self.reopen_button.clicked.connect(self.close_and_reopen)
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
        
        # Refresh Button 
        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.clicked.connect(self.update_preview)
        self.refresh_button.setStyleSheet(secondary_button_style)
        control_layout.addWidget(self.refresh_button)

        # Copy Button 
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.copy_button.setStyleSheet(secondary_button_style)
        control_layout.addWidget(self.copy_button)

        # Run Button 
        self.run_button = QPushButton("Run Command (simulated)" if self.test_only else "Run Command")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self.run_command)
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

    # --- THEME AND STYLING METHODS ---
    def toggle_theme(self):
        """Toggles the application stylesheet between light and dark mode."""
        if self.current_theme == 'light':
            self.current_theme = 'dark'
            self.setStyleSheet(DARK_QSS)
        else:
            self.current_theme = 'light'
            self.setStyleSheet(LIGHT_QSS)

    # --- LOGIC METHODS ---

    def close_and_reopen(self):
        """Hides the current window and triggers the external restart."""
        current_tab_index = self.tabs.currentIndex()

        self.hide() 
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
            'Analysis': 'analysis', # --- CHANGED: Added mapping ---
        }
        actual_command = command_mapping.get(main_command_display)
        
        # ... (Rest of logic remains same, Analysis generally doesn't generate a CLI command but we map it safely)
        
        if main_command_display == 'Train Model':
            current_tab_title = self.tabs.tabText(self.tabs.currentIndex())
            if current_tab_title == "Hyper-Parameter Optimization":
                actual_command = 'hp_optim'
            elif current_tab_title == "Meta-Learning":
                actual_command = 'mrl_train'
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
            script_tab_widget = self.other_tabs_map['Execute Script']
            if hasattr(script_tab_widget, 'get_params'):
                script_params = script_tab_widget.get_params()
                script_name = script_params.pop('script', None)
                if script_name:
                    if sys.platform.startswith('linux'):
                        actual_command = f'scripts/{script_name}'
                        if not actual_command.endswith('.sh'): actual_command += '.sh'
                    elif sys.platform.startswith('win'):
                        actual_command = f'scripts\\{script_name}'
                        if not actual_command.endswith('.bat'): actual_command += '.bat'

        return actual_command if actual_command else main_command_display

    def setup_tabs(self, command):
        """Dynamically loads the correct set of tabs based on the command."""
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)

        if command in self.all_tabs:
            tab_set = self.all_tabs[command]
            for title, tab_widget in tab_set.items():
                self.tabs.addTab(tab_widget, title)
        else:
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
        main_command_display = self.command_combo.currentText()
        
        # --- CHANGED: Analysis tabs are purely GUI tools, no command preview needed really ---
        if main_command_display == 'Analysis':
            self.preview.setPlainText("# Analysis tools run directly within the GUI.\n# No command line argument required.")
            return
            
        actual_command = self.get_actual_command(main_command_display)
        all_params = {}
        regex = r"(?<!-)-(?!-)"
        
        if main_command_display in ['File System Tools', 'Other Tools']:
            current_tab_widget = self.tabs.currentWidget()
            if hasattr(current_tab_widget, 'get_params'):
                all_params.update(current_tab_widget.get_params())
        elif main_command_display == 'Train Model':
            current_tab_title = self.tabs.tabText(self.tabs.currentIndex())
            for title, tab_widget in self.train_tabs_map.items():
                if hasattr(tab_widget, 'get_params'):
                    is_base_tab = title not in ["Hyper-Parameter Optimization", "Meta-Learning"]
                    is_active_special_tab = (
                        title == current_tab_title and
                        title in ["Hyper-Parameter Optimization", "Meta-Learning"]
                    )
                    if is_base_tab or is_active_special_tab:
                        all_params.update(tab_widget.get_params())
        else:
            for i in range(self.tabs.count()):
                tab_widget = self.tabs.widget(i)
                if hasattr(tab_widget, 'get_params'):
                    all_params.update(tab_widget.get_params())

        if actual_command.startswith('scripts/') and 'script' in all_params:
            all_params.pop('script')
            cmd_parts = [f"bash {actual_command}"] if sys.platform.startswith('linux') else [actual_command]
        else:
            cmd_parts = [f"python main.py {actual_command}"]

        for key, value in all_params.items():
            if value is None or value == "":
                continue

            if isinstance(value, bool):
                if key in ['mask_inner', 'mask_logits'] and value is False:
                    cmd_parts.append(f"--no_{re.sub(regex, '_', key)}")
                elif value is True:
                    cmd_parts.append(f"--{re.sub(regex, '_', key)}")
            elif isinstance(value, (int, float)):
                cmd_parts.append(f"--{re.sub(regex, '_', key)} {value}")
            elif isinstance(value, str):
                list_keys = [
                    'focus_graph', 'input_keys', 'graph_sizes', 'data_distributions',
                    'policies', 'pregular_level', 'plastminute_cf',
                    'lookahead_configs', 'gurobi_param', 'hexaly_param',
                ]
                if key in list_keys:
                    parts = value.split()
                    cmd_parts.append(f"--{key}")
                    cmd_parts.extend(parts)
                elif ' ' in value or '"' in value or "'" in value or key == 'update_operation':
                    cmd_parts.append(f"--{key} '{value}'")
                else:
                    cmd_parts.append(f"--{key} {value}")
        command_str = " \\\n  ".join(cmd_parts)
        self.preview.setPlainText(command_str)

    def copy_to_clipboard(self):
        """Copy command to clipboard"""
        self.update_preview()
        clipboard = QApplication.clipboard()
        clipboard.setText(self.preview.toPlainText())
        QMessageBox.information(self, "Copied:", self.preview.toPlainText())

    def run_command(self):
        """Starts the external command using QProcess and opens the results window."""
        
        # --- CHANGED: Prevent running shell commands for Analysis tabs ---
        if self.command_combo.currentText() == 'Analysis':
            QMessageBox.information(self, "Info", "Use the buttons inside the Analysis tabs to load files.")
            return
            
        self.run_button.setDisabled(True)
        self.update_preview()
        
        command_str = self.preview.toPlainText()
        shell_command = command_str.replace(" \\\n  ", " ")
        
        if self.test_only:
            QMessageBox.information(
                self, "Command Simulation",
                f"The following command would be executed:\n\n{command_str}\n\n(Execution is simulated in this environment)."
            )
            self.run_button.setDisabled(False)
            return

        main_command = self.command_combo.currentText()
        is_simulation = main_command == 'Test Simulator'
        
        # --- CLOSE EXISTING RESULTS WINDOW BEFORE STARTING NEW PROCESS ---
        if self.results_window and self.results_window.isVisible():
            self.results_window.close()
            self.results_window = None
        
        if is_simulation:
            test_sim_tab = self.test_sim_tabs_map['Simulator Settings'] 
            policy_names = ['Unknown Policy'] # Default fallback
            if hasattr(test_sim_tab, 'get_params'):
                policies_str = test_sim_tab.get_params().get('policies', '')
                policy_names = policies_str.split() if policies_str else ['Unknown Policy']
            
            self.results_window = SimulationResultsWindow(policy_names)
            self.results_window.show()
        else:
            self.results_window = None
            
        if self.process is not None:
            self.process.terminate()
            self.process.waitForFinished(100) 

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.on_command_finished)

        program = 'sh' if sys.platform.startswith('linux') or sys.platform.startswith('darwin') else 'cmd'
        
        if program == 'sh':
            arguments = ['-c', shell_command]
        elif program == 'cmd':
            arguments = ['/C', shell_command]
        else:
            parts = shell_command.split()
            program = parts[0]
            arguments = parts[1:]

        print(f"Starting process: {program} {' '.join(arguments)}")
        self.process.start(program, arguments)
        
        if not self.process.waitForStarted(200):
            error_msg = self.process.errorString()
            QMessageBox.critical(self, "Error", f"Failed to start external process: {error_msg}")
            self.on_command_finished(1, QProcess.ExitStatus.CrashExit)

    def read_output(self):
        """Reads output and feeds it to the results window for plotting."""
        output_bytes = self.process.readAllStandardOutput()
        output = output_bytes.data().decode()
        
        self.output_buffer += output

        if self.results_window:
            # Data parsing is fast and is done here
            self.output_buffer = self.results_window.parse_buffer(self.output_buffer)

        non_structural_output = [line for line in output.splitlines() if not line.startswith("GUI_")]
        if non_structural_output:
            print("\n".join(non_structural_output))
            sys.stdout.flush()

    def on_command_finished(self, exit_code, exit_status):
        """Called when the external command finishes."""
        
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            msg = "Command execution finished successfully."
            if self.results_window: 
                self.results_window.status_label.setText("Simulation Complete: Success")
        else:
            msg = f"Command failed with exit code: {exit_code}"
            if self.results_window: 
                self.results_window.status_label.setText(f"Simulation Failed (Code: {exit_code})")
            QMessageBox.critical(self, "Error", msg)

        self.process = None 
        self.run_button.setDisabled(False)

    def read_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        print(data, end='')

    def read_stderr(self):
        data = self.process.readAllStandardError().data().decode()
        print(data, end='')

    def closeEvent(self, event):
        """Ensures all active threads and external windows are closed before main app exit."""
        
        # 1. Close the SimulationResultsWindow if it's currently open (from a running process)
        if self.results_window and self.results_window.isVisible():
            self.results_window.close()
        
        # 2. Explicitly shut down Data Analysis and Output Analysis tabs (which manage their own workers/windows)
        for tab in self.analysis_tabs_map.values():
            if hasattr(tab, 'shutdown'):
                tab.shutdown()
        
        # 3. Terminate any running QProcess
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished(1000)

        super().closeEvent(event)