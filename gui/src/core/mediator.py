"""Mediator class for handling communication between different GUI components."""

import re
import sys

from PySide6.QtCore import QObject, Signal


class UIMediator(QObject):
    """
    Mediator class to handle communication between the Main Window and various Tabs.
    It manages tab registration, listens for parameter changes, and generates the command preview.
    """

    command_updated = Signal(str)

    def __init__(self, main_window):
        """Initialize the Mediator.

        Args:
            main_window: The main application window.
        """
        # Pass main_window as parent if it's a QObject to ensure proper cleanup
        parent = main_window if isinstance(main_window, QObject) else None
        super().__init__(parent)
        self.main_window = main_window
        self.tabs = {}
        self.current_command = None

    def register_tab(self, command_name, tab_name, tab_widget):
        """Registers a tab with the mediator."""
        if command_name not in self.tabs:
            self.tabs[command_name] = {}

        self.tabs[command_name][tab_name] = tab_widget

        # Connect to paramsChanged signal if it exists
        if hasattr(tab_widget, "paramsChanged"):
            tab_widget.paramsChanged.connect(self.update_preview)

    def set_current_command(self, command):
        """Updates the current selected command."""
        self.current_command = command
        self.update_preview()

    def get_actual_command(self, main_command_display):
        """Maps the display name to the command-line argument dynamically."""
        actual_command = self._get_base_command_mapping(main_command_display)
        actual_command = self._handle_dynamic_tab_command(main_command_display, actual_command)

        if actual_command == "scripts":
            actual_command = self._format_script_command()

        return actual_command

    def _get_base_command_mapping(self, display_name):
        """Returns the base command string for a display name."""
        mapping = {
            "Train Model": "train",
            "Generate Data": "gen_data",
            "Evaluate": "eval",
            "Test Simulator": "test_sim",
            "Analysis": "analysis",
        }
        return mapping.get(display_name, display_name)

    def _handle_dynamic_tab_command(self, display_name, current_cmd):
        """Adjusts the command based on the active tab's context."""
        if not (self.main_window and hasattr(self.main_window, "tabs")):
            return current_cmd

        idx = self.main_window.tabs.currentIndex()
        tab_title = self.main_window.tabs.tabText(idx)

        if display_name == "Train Model":
            if tab_title == "Hyper-Parameter Optimization":
                return "hp_optim"
            if tab_title == "Meta-Learning":
                return "mrl_train"

        if display_name == "File System Tools":
            sub_cmds = {
                "Cryptography Settings": "cryptography",
                "Update Settings": "update",
                "Delete Settings": "delete",
            }
            sub = sub_cmds.get(tab_title)
            return f"file_system {sub}" if sub else "file_system "

        if display_name == "Other Tools":
            if tab_title == "Execute Script":
                return "scripts"
            if tab_title == "Program Test Suite":
                return "test_suite"

        return current_cmd

    def _format_script_command(self):
        """Formats the script execution command based on the OS."""
        script_tab = self.tabs.get("Other Tools", {}).get("Execute Script")
        if not (script_tab and hasattr(script_tab, "get_params")):
            return "scripts"

        script_params = script_tab.get_params()
        script_name = script_params.get("script")
        if not script_name:
            return "scripts"

        if sys.platform.startswith("linux"):
            ext = ".sh"
            path_sep = "/"
        else:
            ext = ".bat"
            path_sep = "\\"

        full_name = script_name if script_name.endswith(ext) else f"{script_name}{ext}"
        return f"scripts{path_sep}{full_name}"

    def update_preview(self):
        """Generates the command string based on the current command and tab parameters."""
        if not self.current_command:
            return

        if self.current_command == "Analysis":
            self.command_updated.emit(
                "# Analysis tools run directly within the GUI.\n# No command line argument required."
            )
            return

        actual_command = self.get_actual_command(self.current_command)
        all_params = self._collect_tab_parameters()
        command_str = self._construct_command_string(actual_command, all_params)
        self.command_updated.emit(command_str)

    def _collect_tab_parameters(self):
        """Collects combined parameters from all relevant tabs."""
        all_params = {}
        if self.current_command in ["File System Tools", "Other Tools"]:
            if self.main_window:
                current_tab = self.main_window.tabs.currentWidget()
                if hasattr(current_tab, "get_params"):
                    all_params.update(current_tab.get_params())

        elif self.current_command == "Train Model":
            if self.current_command in self.tabs:
                train_tabs = self.tabs[self.current_command]
                current_tab_title = self.main_window.tabs.tabText(self.main_window.tabs.currentIndex())

                for title, tab_widget in train_tabs.items():
                    if hasattr(tab_widget, "get_params"):
                        is_special = title in ["Hyper-Parameter Optimization", "Meta-Learning"]
                        if not is_special or title == current_tab_title:
                            all_params.update(tab_widget.get_params())

        elif self.current_command in self.tabs:
            for tab_widget in self.tabs[self.current_command].values():
                if hasattr(tab_widget, "get_params"):
                    all_params.update(tab_widget.get_params())

        return all_params

    def _construct_command_string(self, actual_command, all_params):
        """Formats the final command string from parameters."""
        # Split logic into parts
        if actual_command.startswith(("scripts/", "scripts\\")) and "script" in all_params:
            all_params.pop("script")
            is_linux = sys.platform.startswith("linux")
            cmd_parts = [f"bash {actual_command}"] if is_linux else [actual_command]
        else:
            cmd_parts = [f"python main.py {actual_command}"]

        regex = r"(?<!-)-(?!-)"
        for key, value in all_params.items():
            if value is None or value == "":
                continue

            if isinstance(value, bool):
                clean_key = re.sub(regex, "_", key)
                if key in ["mask_inner", "mask_logits"] and value is False:
                    cmd_parts.append(f"--no_{clean_key}")
                elif value is True:
                    cmd_parts.append(f"--{clean_key}")
            elif isinstance(value, (int, float)):
                clean_key = re.sub(regex, "_", key)
                cmd_parts.append(f"--{clean_key} {value}")
            elif isinstance(value, str):
                list_keys = [
                    "focus_graph",
                    "input_keys",
                    "graph_sizes",
                    "data_distributions",
                    "policies",
                    "pregular_level",
                    "plastminute_cf",
                    "lookahead_configs",
                    "gurobi_param",
                    "hexaly_param",
                ]
                if key in list_keys:
                    parts = value.split()
                    cmd_parts.append(f"--{key}")
                    cmd_parts.extend(parts)
                elif " " in value or '"' in value or "'" in value or key == "update_operation":
                    cmd_parts.append(f"--{key} '{value}'")
                else:
                    cmd_parts.append(f"--{key} {value}")

        return " \\\n  ".join(cmd_parts)
