"""
Command builder logic for RunScriptsTab.
"""

import sys


def get_command(params):
    """
    Generate the command string to run the selected script.

    Args:
        params (dict): Dictionary of parameters from ScriptParametersWidget.

    Returns:
        str: The formatted shell command.
    """
    if not params:
        return ""

    script_name = params["script"]
    command_parts = []

    if script_name in ["test_sim", "train", "hyperparam_optim", "gen_data"]:
        command_parts.append(f"python main.py {script_name.replace('_', ' ')}")
    elif script_name == "setup_env":
        if sys.platform.startswith("linux"):
            command_parts.append(f"scripts/setup_env.sh {params.get('manager', 'uv')}")
        else:
            command_parts.append(f"scripts\\setup_env.bat {params.get('manager', 'uv')}")
    elif script_name in ["slim_slurm", "slurm"]:
        command_parts.append(f"bash {script_name}.sh")

    if params.get("verbose"):
        command_parts.append("--verbose")

    if params.get("cores") and script_name in ["test_sim", "slurm"]:
        command_parts.append(f"-nc {params['cores']}")

    return " ".join(command_parts)
