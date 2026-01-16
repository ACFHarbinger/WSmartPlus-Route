"""
Terminal User Interface (TUI) Module.

This module implements a rich, interactive command-line interface for the WSmart-Route
system using `rich` and `prompt_toolkit`. It provides a menu-driven experience for
configuring simulations, training models, and managing data.
"""
import os
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich import print as rprint
from rich.text import Text

from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import RadioList, Label, Frame

from logic.src.cli.registry import get_parser
from logic.src.cli.train_parser import validate_train_args
from logic.src.cli.sim_parser import validate_test_sim_args, validate_eval_args
from logic.src.cli.data_parser import validate_gen_data_args
from logic.src.cli.fs_parser import validate_file_system_args
from logic.src.cli.gui_parser import validate_gui_args
from logic.src.cli.test_suite_parser import validate_test_suite_args

# Custom style for prompt-toolkit
tui_style = Style.from_dict({
    'dialog': 'bg:#2b2b2b #ffffff',
    'dialog.body': 'bg:#2b2b2b #ffffff',
    'dialog.border': '#00ff00',
    'radio-area': 'bg:#333333',
    'radio-selected': '#00ff00 bold',
    'button': 'bg:#444444 #ffffff',
    'button.focused': 'bg:#00ff00 #000000',
})

class TerminalUI:
    """
    Terminal User Interface for WSmart+ Route CLI.
    """
    def __init__(self):
        """Initialize the TUI with console, parser, and command definitions."""
        self.console = Console()
        self.parser = get_parser()
        self.command_map = {
            "train": "Training for neural model",
            "mrl_train": "Meta-RL training",
            "hp_optim": "Hyperparameter optimization",
            "gen_data": "Generate training data",
            "eval": "Evaluate a model",
            "test_sim": "Run simulation test",
            "file_system": "File system operations",
            "gui": "Launch the GUI",
            "test_suite": "Run the test suite"
        }

    def display_header(self):
        """Displays a premium header."""
        header_text = Text("WSmart+ Route TUI", style="bold green")
        self.console.print(Panel(header_text, subtitle="Interactive Command Center", border_style="bright_blue"))

    def _quick_select(self, title: str, text: str, values: List[Tuple[str, str]]) -> Optional[str]:
        """A reusable one-click selection menu using prompt-toolkit."""
        radio_list = RadioList(values)
        kb = KeyBindings()

        @kb.add('enter')
        def _(event):
            # Ensure the highlighted one is selected as current
            radio_list.current_value = radio_list.values[radio_list._selected_index][0]
            event.app.exit(result=radio_list.current_value)

        @kb.add('c-c')
        @kb.add('escape')
        def _(event):
            event.app.exit(result=None)

        # Custom handlers for arrows to sync selection (asterisk) with highlight
        @radio_list.control.key_bindings.add('up')
        def _(event):
            radio_list._selected_index = (radio_list._selected_index - 1) % len(radio_list.values)
            radio_list.current_value = radio_list.values[radio_list._selected_index][0]

        @radio_list.control.key_bindings.add('down')
        def _(event):
            radio_list._selected_index = (radio_list._selected_index + 1) % len(radio_list.values)
            radio_list.current_value = radio_list.values[radio_list._selected_index][0]

        # Override RadioList internal enter behavior to submit
        radio_list.control.key_bindings.add('enter')(lambda event: event.app.exit(result=radio_list.values[radio_list._selected_index][0]))

        container = Frame(
            HSplit([
                Label(f"{text} (Enter to select, Esc to cancel)\n"),
                radio_list,
            ]),
            title=title,
            style='class:dialog'
        )

        app = Application(
            layout=Layout(container, focused_element=radio_list),
            key_bindings=kb,
            full_screen=False,
            mouse_support=True,
            style=tui_style
        )
        return app.run()

    def _prompt(self, message: str, default: str = "", is_int: bool = False, is_bool: bool = False, choices: List[str] = None) -> Any:
        """A prompt-toolkit based prompt that supports Esc to cancel and Rich colors."""
        import re
        
        # Convert Rich markup to ANSI for prompt_toolkit
        # Use a temporary console to capture output without printing to stdout
        with self.console.capture() as capture:
            self.console.print(message, end="")
        ansi_message = capture.get()
        
        suffix = ""
        if choices:
            suffix += f" [{'|'.join(choices)}]"
        
        suffix += f" [{default}]: " if str(default) else ": "
        
        prompt_message = ANSI(ansi_message + suffix)

        kb = KeyBindings()
        @kb.add('escape')
        @kb.add('c-c')
        def _(event):
            event.app.exit(result="__CANCEL__")

        try:
            if is_bool:
                # For boolean, simple y/n prompt
                defaults_text = " [y/n]"
                # We can reuse the ansi message but append y/n
                bool_prompt = ANSI(ansi_message + defaults_text + suffix)
                
                result = prompt(bool_prompt, default="y" if default is True else "n" if default is False else "", key_bindings=kb)
                if result == "__CANCEL__" or result is None: return None
                return result.lower().startswith('y')
            
            # Create a completer if choices are provided
            completer = None
            if choices:
                from prompt_toolkit.completion import WordCompleter
                completer = WordCompleter(choices)

            result = prompt(prompt_message, default=str(default), key_bindings=kb, completer=completer)
            if result == "__CANCEL__" or result is None:
                return None
                
            if is_int:
                return int(result)
            return result
        except (ValueError, KeyboardInterrupt):
            return None

    def select_subcommand(self) -> Optional[str]:
        """Shows the main menu to select a subcommand."""
        values = [(k, f"{k}: {v}") for k, v in self.command_map.items()]
        return self._quick_select("Main Menu", "Select an operation to perform:", values)

    def configure_command(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Guides the user through configuring arguments for a specific command.
        """
        self.console.clear()
        self.display_header()
        
        panel = Panel(
            f"Configuring [bold blue]{command}[/bold blue]\n[dim]{self.command_map.get(command)}[/dim]",
            border_style="cyan",
            expand=False
        )
        self.console.print(panel)
        self.console.print()

        if command == "test_sim":
            res = self._form_test_sim()
        elif command == "train":
            res = self._form_train()
        elif command == "mrl_train":
            res = self._form_mrl_train()
        elif command == "hp_optim":
            res = self._form_hp_optim()
        elif command == "eval":
            res = self._form_eval()
        elif command == "gen_data":
            res = self._form_gen_data()
        elif command == "file_system":
            res = self._form_file_system()
        else:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] Form for '{command}' is not fully interactive yet.")
            if self._prompt("Proceed with default values?", default=False, is_bool=True):
                return {"command": command}
            return None
        
        return res

    def _form_test_sim(self) -> Optional[Dict[str, Any]]:
        """Form for test_sim command."""
        opts = {}
        
        val = self._prompt("[bold green]Policies[/bold green] (space separated)", default="regular alns")
        if val is None: return None
        opts["policies"] = val.split()
        
        opts["days"] = self._prompt("[bold green]Simulation days[/bold green]", default=31, is_int=True)
        if opts["days"] is None: return None
        
        opts["size"] = self._prompt("[bold green]Graph size[/bold green]", default=50, is_int=True)
        if opts["size"] is None: return None
        
        opts["area"] = self._prompt("[bold green]Area[/bold green]", default="riomaior")
        if opts["area"] is None: return None
        
        opts["waste_type"] = self._prompt("[bold green]Waste type[/bold green]", default="plastic")
        if opts["waste_type"] is None: return None
        
        opts["n_samples"] = self._prompt("[bold green]Number of samples[/bold green]", default=1, is_int=True)
        if opts["n_samples"] is None: return None
        
        opts["real_time_log"] = self._prompt("[bold green]Enable real-time log window?[/bold green]", default=False, is_bool=True)
        if opts["real_time_log"] is None: return None
        
        # Hidden defaults
        opts["checkpoint_dir"] = "temp"
        opts["output_dir"] = "output"
        opts["seed"] = 42
        return opts

    def _form_train(self) -> Optional[Dict[str, Any]]:
        """Form for train command."""
        opts = {}
        opts["model"] = self._prompt("[bold green]Model type[/bold green]", default="am", choices=["am", "tam", "ddam"])
        if opts["model"] is None: return None
        
        opts["problem"] = self._prompt("[bold green]Problem[/bold green]", default="vrpp", choices=["wcvrp", "vrpp", "cwcvrp"])
        if opts["problem"] is None: return None
        
        opts["graph_size"] = self._prompt("[bold green]Graph size[/bold green]", default=20, is_int=True)
        if opts["graph_size"] is None: return None
        
        opts["batch_size"] = self._prompt("[bold green]Batch size[/bold green]", default=256, is_int=True)
        if opts["batch_size"] is None: return None
        
        opts["n_epochs"] = self._prompt("[bold green]Number of epochs[/bold green]", default=25, is_int=True)
        if opts["n_epochs"] is None: return None
        
        opts["train_time"] = self._prompt("[bold green]Train over multiple days?[/bold green]", default=False, is_bool=True)
        if opts["train_time"] is None: return None
        
        return opts

    def _form_mrl_train(self) -> Optional[Dict[str, Any]]:
        """Form for mrl_train command."""
        opts = {}
        opts["model"] = self._prompt("[bold green]Model type[/bold green]", default="am", choices=["am", "tam", "ddam"])
        if opts["model"] is None: return None
        
        opts["problem"] = self._prompt("[bold green]Problem[/bold green]", default="vrpp", choices=["wcvrp", "vrpp", "cwcvrp"])
        if opts["problem"] is None: return None
        
        opts["n_iterations"] = self._prompt("[bold green]Number of iterations[/bold green]", default=100, is_int=True)
        if opts["n_iterations"] is None: return None
        
        opts["meta_batch_size"] = self._prompt("[bold green]Meta batch size[/bold green]", default=4, is_int=True)
        if opts["meta_batch_size"] is None: return None
        
        return opts

    def _form_hp_optim(self) -> Optional[Dict[str, Any]]:
        """Form for hp_optim command."""
        opts = {}
        opts["model"] = self._prompt("[bold green]Model type[/bold green]", default="am", choices=["am", "tam", "ddam"])
        if opts["model"] is None: return None
        
        opts["problem"] = self._prompt("[bold green]Problem[/bold green]", default="vrpp", choices=["wcvrp", "vrpp", "cwcvrp"])
        if opts["problem"] is None: return None
        
        opts["n_trials"] = self._prompt("[bold green]Number of trials[/bold green]", default=50, is_int=True)
        if opts["n_trials"] is None: return None
        
        return opts

    def _form_eval(self) -> Optional[Dict[str, Any]]:
        """Form for eval command."""
        opts = {}
        opts["datasets"] = self._prompt("[bold green]Datasets[/bold green] (space separated paths)", default="datasets/vrpp_20.pkl")
        if opts["datasets"] is None: return None
        opts["datasets"] = opts["datasets"].split()
        
        opts["decode_type"] = self._prompt("[bold green]Decode type[/bold green]", default="greedy", choices=["greedy", "sampling", "bs"])
        if opts["decode_type"] is None: return None
        
        opts["eval_batch_size"] = self._prompt("[bold green]Eval batch size[/bold green]", default=256, is_int=True)
        if opts["eval_batch_size"] is None: return None
        
        return opts

    def _form_gen_data(self) -> Optional[Dict[str, Any]]:
        """Form for gen_data command."""
        opts = {}
        opts["problem"] = self._prompt("[bold green]Problem type[/bold green]", default="vrpp", choices=["wcvrp", "vrpp", "cwcvrp"])
        if opts["problem"] is None: return None
        
        gs = self._prompt("[bold green]Graph size[/bold green]", default=50, is_int=True)
        if gs is None: return None
        opts["graph_sizes"] = [gs]
        
        opts["dataset_size"] = self._prompt("[bold green]Dataset size[/bold green]", default=10000, is_int=True)
        if opts["dataset_size"] is None: return None
        
        opts["data_type"] = self._prompt("[bold green]Data type[/bold green]", default="virtual", choices=["virtual", "real"])
        if opts["data_type"] is None: return None
        
        return opts

    def _form_file_system(self) -> Optional[Dict[str, Any]]:
        """Form for file_system command."""
        sub_comm = self._quick_select(
            "File System", 
            "Select subcommand:",
            [("update", "Update entries"), ("delete", "Delete entries"), ("cryptography", "Crypto ops")]
        )
        
        if not sub_comm: return None
        
        opts = {"fs_command": sub_comm}
        if sub_comm == "delete":
            opts["log"] = self._prompt("Delete logs?", default=True, is_bool=True)
            if opts["log"] is None: return None
            
            opts["data"] = self._prompt("Delete datasets?", default=False, is_bool=True)
            if opts["data"] is None: return None
            
            opts["test_checkpoint"] = self._prompt("Delete checkpoints?", default=True, is_bool=True)
            if opts["test_checkpoint"] is None: return None
            
            opts["delete_preview"] = True
        return opts

    def run(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Main TUI loop. Returns the selected command and options."""
        while True:
            self.console.clear()
            self.display_header()
            
            command = self.select_subcommand()
            if not command:
                return None
                
            opts = self.configure_command(command)
            if not opts:
                continue

            # Confirm and show summary
            self.console.print("\n[bold green]Configuration Summary:[/bold green]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Parameter")
            table.add_column("Value")
            for k, v in opts.items():
                table.add_row(k, str(v))
            self.console.print(table)

            if self._prompt("\n[bold cyan]Run this command?[/bold cyan]", default=True, is_bool=True):
                return command, opts
            else:
                res = self._prompt("Adjust parameters or select new command? (No to exit)", default=True, is_bool=True)
                if res is None or res is False:
                    return None

def launch_tui() -> Optional[Tuple[str, Dict[str, Any]]]:
    """Entry point for the TUI."""
    tui = TerminalUI()
    return tui.run()

if __name__ == "__main__":
    launch_tui()
