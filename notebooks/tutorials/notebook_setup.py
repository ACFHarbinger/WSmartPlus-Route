import os
import sys

IN_COLAB = {}
HOME_DIRECTORY = {}
SETUP_EXECUTED = {}


def setup_home_directory(notebook_name):
    setup_executed = False if notebook_name not in SETUP_EXECUTED else SETUP_EXECUTED[notebook_name]
    if not setup_executed:
        try:
            home_dir = os.path.dirname(os.getcwd())
            sys.path.insert(0, home_dir)
            print(f"Setup completed - added home_dir to system path: {home_dir}")

            # Update the global variables
            globals()["SETUP_EXECUTED"][notebook_name] = True
            globals()["HOME_DIRECTORY"][notebook_name] = home_dir
            return home_dir
        except Exception as e:
            globals()["SETUP_EXECUTED"][notebook_name] = False
            globals()["HOME_DIRECTORY"][notebook_name] = None
            print("Failed to setup home directory:", e.__traceback__)
            return None
    else:
        home_dir = HOME_DIRECTORY[notebook_name]
        print("Already added home_dir to system path:", home_dir)
        return home_dir


def import_google_colab_libs():
    from google.colab import drive, files

    return drive, files


def setup_google_colab(notebook_name):
    in_colab = False if notebook_name not in IN_COLAB else IN_COLAB[notebook_name]
    if not in_colab:
        try:
            drive, files = import_google_colab_libs()

            # Update the global variable
            globals()["IN_COLAB"][notebook_name] = True
            print("Completed setup of Google Colab libraries!")
            return True, drive, files
        except Exception:
            globals()["IN_COLAB"][notebook_name] = False
            return False, None, None
    else:
        return True, *import_google_colab_libs()
