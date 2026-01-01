import sys
import re

from PySide6.QtCore import QObject, Signal


class UIMediator(QObject):
    """
    Mediator class to handle communication between the Main Window and various Tabs.
    It manages tab registration, listens for parameter changes, and generates the command preview.
    """
    command_updated = Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.tabs = {}
        self.current_command = None

    def register_tab(self, command_name, tab_name, tab_widget):
        """Registers a tab with the mediator."""
        if command_name not in self.tabs:
            self.tabs[command_name] = {}
        
        self.tabs[command_name][tab_name] = tab_widget
        
        # Connect to paramsChanged signal if it exists
        if hasattr(tab_widget, 'paramsChanged'):
            tab_widget.paramsChanged.connect(self.update_preview)

    def set_current_command(self, command):
        """Updates the current selected command."""
        self.current_command = command
        self.update_preview()

    def get_actual_command(self, main_command_display):
        """Maps the display name to the command-line argument dynamically."""
        # Logic moved from MainWindow
        command_mapping = {
            'Train Model': 'train',
            'Generate Data': 'gen_data',
            'Evaluate': 'eval',
            'Test Simulator': 'test_sim',
            'Analysis': 'analysis',
        }
        actual_command = command_mapping.get(main_command_display, main_command_display)
        
        # Dynamic handling based on active tab for specific commands
        if self.main_window and hasattr(self.main_window, 'tabs'):
             current_tab_title = self.main_window.tabs.tabText(self.main_window.tabs.currentIndex())
             
             if main_command_display == 'Train Model':
                 if current_tab_title == "Hyper-Parameter Optimization":
                     actual_command = 'hp_optim'
                 elif current_tab_title == "Meta-Learning":
                     actual_command = 'mrl_train'
             elif main_command_display == 'File System Tools':
                 actual_command = 'file_system '
                 if current_tab_title == "Cryptography Settings":
                     actual_command += 'cryptography'
                 elif current_tab_title == "Update Settings":
                     actual_command += 'update'
                 elif current_tab_title == "Delete Settings":
                     actual_command += 'delete'
             elif main_command_display == 'Other Tools':
                 if current_tab_title == 'Execute Script':
                     actual_command = 'scripts'
                 elif current_tab_title == 'Program Test Suite':
                     actual_command = 'test_suite'

        # Script handling
        if actual_command == 'scripts':
             # We need to access the script tab specifically if possible, 
             # or just let get_params() logic handle it, but here we need the name to form the command.
             # Accessing via self.tabs if registered correctly
             script_tab = self.tabs.get('Other Tools', {}).get('Execute Script')
             if script_tab and hasattr(script_tab, 'get_params'):
                 script_params = script_tab.get_params()
                 script_name = script_params.get('script', None)
                 if script_name:
                     if sys.platform.startswith('linux'):
                         actual_command = f'scripts/{script_name}'
                         if not actual_command.endswith('.sh'): actual_command += '.sh'
                     elif sys.platform.startswith('win'):
                         actual_command = f'scripts\\{script_name}'
                         if not actual_command.endswith('.bat'): actual_command += '.bat'
                         
        return actual_command

    def update_preview(self):
        """Generates the command string based on the current command and tab parameters."""
        if not self.current_command:
            return

        if self.current_command == 'Analysis':
            self.command_updated.emit("# Analysis tools run directly within the GUI.\n# No command line argument required.")
            return

        actual_command = self.get_actual_command(self.current_command)
        all_params = {}
        
        # Collect parameters based on the command type
        if self.current_command in ['File System Tools', 'Other Tools']:
            # Active tab only
             if self.main_window:
                current_tab = self.main_window.tabs.currentWidget()
                if hasattr(current_tab, 'get_params'):
                     all_params.update(current_tab.get_params())
        elif self.current_command == 'Train Model':
            # Base tabs + potentially active special tab
            if self.current_command in self.tabs:
                train_tabs = self.tabs[self.current_command]
                current_tab_title = self.main_window.tabs.tabText(self.main_window.tabs.currentIndex())
                
                for title, tab_widget in train_tabs.items():
                    if hasattr(tab_widget, 'get_params'):
                        is_base_tab = title not in ["Hyper-Parameter Optimization", "Meta-Learning"]
                        is_active_special_tab = (
                            title == current_tab_title and
                            title in ["Hyper-Parameter Optimization", "Meta-Learning"]
                        )
                        if is_base_tab or is_active_special_tab:
                            all_params.update(tab_widget.get_params())
        else:
            # Aggregate from all tabs for this command
            if self.current_command in self.tabs:
                for tab_widget in self.tabs[self.current_command].values():
                     if hasattr(tab_widget, 'get_params'):
                         all_params.update(tab_widget.get_params())

        # Construct Command String
        if actual_command.startswith('scripts/') and 'script' in all_params:
             all_params.pop('script') # Remove script name from params as it's in command
             cmd_parts = [f"bash {actual_command}"] if sys.platform.startswith('linux') else [actual_command]
        else:
             cmd_parts = [f"python main.py {actual_command}"]

        regex = r"(?<!-)-(?!-)"
        for key, value in all_params.items():
            if value is None or value == "":
                continue

            if isinstance(value, bool):
                clean_key = re.sub(regex, '_', key)
                if key in ['mask_inner', 'mask_logits'] and value is False:
                    cmd_parts.append(f"--no_{clean_key}")
                elif value is True:
                    cmd_parts.append(f"--{clean_key}")
            elif isinstance(value, (int, float)):
                 clean_key = re.sub(regex, '_', key)
                 cmd_parts.append(f"--{clean_key} {value}")
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
        self.command_updated.emit(command_str)
