import argparse
import signal
import sys
import threading
import traceback

from logic.src.cli import ConfigsParser, add_gui_args, validate_gui_args
from logic.src.utils.definitions import CTRL_C_TIMEOUT, ICON_FILE
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from gui.src.windows import MainWindow, SimulationResultsWindow


def launch_results_window(policy_names, log_path):
    """Initializes QApplication and launches the SimulationResultsWindow."""
    app = QApplication(sys.argv)

    # Create and show the window
    results_window = SimulationResultsWindow(policy_names, log_path)
    results_window.show()

    sys.exit(app.exec())


def run_app_gui(opts):
    app = QApplication(sys.argv)
    try:
        app_icon = QIcon(ICON_FILE)
        app.setWindowIcon(app_icon)
    except Exception:
        pass

    if "app_style" in opts and opts["app_style"] is not None:
        app.setStyle(opts["app_style"])  # Set application style

    current_window = None  # Non-local variable to hold the reference to the main window

    # Create a custom signal handler that works with Qt
    def handle_interrupt(signum, frame):
        """Handle Ctrl+C signal by gracefully closing the application"""
        print("\nCtrl+C received - closing application...")
        if current_window is not None:
            current_window.close()
        quit()
        # Force exit if app doesn't quit quickly
        threading.Timer(CTRL_C_TIMEOUT, lambda: sys.exit(1)).start()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, handle_interrupt)

    def launch_gui(test_only, tab_index=0):
        """Creates and shows a new instance of the MainWindow."""
        nonlocal current_window

        # 1. Check if an old instance exists
        if current_window is not None:
            current_window.close()

        # 2. Create the new window instance, passing the saved tab_index
        current_window = MainWindow(
            test_only=test_only,
            restart_callback=launch_gui,
            initial_tab_index=tab_index,
            initial_window=(current_window.command_combo.currentText() if current_window else "Train Model"),
        )
        current_window.show()

    launch_gui(opts["test_only"], tab_index=0)

    # Install a custom event filter to catch the interrupt
    old_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        return app.exec()
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, old_handler)


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="WSR Simulator Test Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_gui_args(parser)
    try:
        parsed_args = parser.parse_process_args(sys.argv[1:], "gui")
        args = validate_gui_args(parsed_args)
        exit_code = run_app_gui(args)
    except (argparse.ArgumentError, AssertionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        parser.print_help()
        exit_code = 1
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print("\n" + e)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
