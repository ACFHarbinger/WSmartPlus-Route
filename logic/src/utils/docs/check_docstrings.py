"""
Docstring Compliance Checker.

This script traverses Python files in specified directories or file paths,
parses them using the `ast` module, and checks for the presence of docstrings
in modules, classes, and functions.

It identifies missing docstrings to help maintain documentation standards
across the codebase and displays them in a formatted table using `rich`.

Usage:
    python check_docstrings.py <path1> [path2 ...]
"""

import ast
import os
import sys

# Try importing rich for beautiful output; provide a helpful error if missing.
try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Error: The 'rich' library is required for this script.")
    print("Please install it using: pip install rich")
    sys.exit(1)

# Initialize the console for printing
console = Console()


def check_path(path):
    """
    Check a single file for missing docstrings using AST parsing.

    Args:
        path (str): The file path to check.

    Returns:
        list: A list of dictionaries containing details about missing docstrings.
              Keys: 'path', 'type', 'name'.
    """
    missing_items = []
    if os.path.isfile(path):
        if not path.endswith(".py"):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            console.print(f"[bold red]Syntax error in {path}[/bold red]")
            return []

        # Check Module Docstring
        if not ast.get_docstring(tree):
            missing_items.append({"path": path, "type": "Module", "name": "<module>"})

        # Check Class and Function Docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip private members (starting with _ but not __)
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue

                if not ast.get_docstring(node):
                    kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    missing_items.append({"path": path, "type": kind, "name": node.name})
    return missing_items


def check_docstrings_recursive(directory):
    """
    Recursively check a directory for Python files with missing docstrings.

    Args:
        directory (str): The root directory to search.

    Returns:
        list: A list of dictionaries describing missing docstrings.
    """
    missing_items = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            missing_items.extend(check_path(os.path.join(root, file)))
    return missing_items


def display_results(errors):
    """
    Render the list of errors in a beautiful table using Rich.

    Args:
        errors (list): A list of dictionaries with error details.
    """
    if not errors:
        console.print(":check_mark_button: [bold green]Success! No missing docstrings found.[/bold green]")
        return

    # Create a table
    table = Table(
        title=f"Docstring Compliance Report ({len(errors)} Issues Found)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
    )

    # Add columns
    table.add_column("File Path", style="dim")
    table.add_column("Type", justify="center")
    table.add_column("Name", style="bold yellow")

    # Add rows
    for error in errors:
        # Color code the 'Type' column for better readability
        type_style = "blue" if error["type"] == "Class" else "green" if error["type"] == "Function" else "white"

        table.add_row(error["path"], f"[{type_style}]{error['type']}[/{type_style}]", error["name"])

    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python check_docstrings.py <path1> [path2 ...]")
        sys.exit(1)

    all_missing = []

    with console.status("[bold green]Scanning files...[/bold green]"):
        for arg in sys.argv[1:]:
            if os.path.isfile(arg):
                all_missing.extend(check_path(arg))
            elif os.path.isdir(arg):
                all_missing.extend(check_docstrings_recursive(arg))
            else:
                console.print(f"[bold yellow]Skipping invalid path: {arg}[/bold yellow]")

    display_results(all_missing)
