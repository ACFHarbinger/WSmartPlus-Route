import os
import sys

from backend.src import run_app_gui

# Ensure that your root directory is on the path if needed
sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    args = {'test_only': False}
    run_app_gui(args)