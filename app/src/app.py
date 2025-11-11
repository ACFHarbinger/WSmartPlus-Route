import sys
import signal
import threading

from .utils.definitions import CTRL_C_TIMEOUT
from .utils.arg_parser import parse_params
from .gui.main_window import MainWindow
from PySide6.QtWidgets import QApplication


def run_app_gui(opts):
    app = QApplication(sys.argv)
    if 'app_style' in opts and opts['app_style'] is not None:
        app.setStyle(opts['app_style']) # Set application style
    
    current_window = None # Non-local variable to hold the reference to the main window
    
    # Create a custom signal handler that works with Qt
    def handle_interrupt(signum, frame):
        """Handle Ctrl+C signal by gracefully closing the application"""
        print("\nCtrl+C received - closing application...")
        if current_window is not None:
            current_window.close()
        app.quit()
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
            initial_window=current_window.command_combo.currentText() if current_window else 'Train Model'
        )
        current_window.show()
    
    launch_gui(opts['test_only'], tab_index=0)
    
    # Install a custom event filter to catch the interrupt
    old_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        return app.exec()
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, old_handler)
    


if __name__ =="__main__":
    try:
        args = parse_params()
        exit_code = run_app_gui(args)
    finally:
        sys.stdout.flush()
        sys.exit(exit_code)