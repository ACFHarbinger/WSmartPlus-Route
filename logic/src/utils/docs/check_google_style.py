"""check_google_style.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import check_google_style
"""

import argparse
import ast
import os
import re
import sys
from typing import Dict, List

# Try importing rich for beautiful output
try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Error: The 'rich' library is required for this script.")
    print("Please install it using: pip install rich")
    sys.exit(1)

# Initialize Console
console = Console()

# --- Configuration ---
SECTION_ALIASES = {
    "ARGS": "Args",
    "ARGUMENTS": "Args",
    "PARAMETERS": "Args",
    "PARAMS": "Args",
    "ATTRIBUTES": "Attributes",
    "ATTRS": "Attributes",
    "EXAMPLE": "Example",
    "EXAMPLES": "Example",
    "RETURN": "Returns",
    "RETURNS": "Returns",
    "YIELD": "Yields",
    "YIELDS": "Yields",
    "RAISES": "Raises",
    "WARNS": "Warns",
    "NOTE": "Note",
    "NOTES": "Note",
    "TODO": "Todo",
}


class GoogleStyleValidator(ast.NodeVisitor):
    """GoogleStyleValidator class.

    Attributes:
        attr (Type): Description of attribute.
    """

    def __init__(self, filepath):
        """Initialize Class.

        Args:
            filepath (Any): Description of filepath.
        """
        self.filepath = filepath
        self.violations = []
        self.current_class = None

    def add_violation(self, node, message):
        """Add violation.

        Args:
            node (Any): Description of node.
            message (Any): Description of message.
        """
        lineno = getattr(node, "lineno", 1)
        name = getattr(node, "name", "module")
        # Store raw data; formatting happens later
        self.violations.append({"line": lineno, "context": type(node).__name__, "name": name, "message": message})

    def _parse_docstring_sections(self, docstring: str) -> Dict[str, str]:
        """parse docstring sections.

        Args:
            docstring (str): Description of docstring.

        Returns:
            Any: Description of return value.
        """
        if not docstring:
            return {}

        sections = {}
        header_pattern = re.compile(
            r"^\s*(" + "|".join(set(SECTION_ALIASES.keys())) + r"):\s*$", re.MULTILINE | re.IGNORECASE
        )

        matches = list(header_pattern.finditer(docstring))

        for i, match in enumerate(matches):
            raw_header = match.group(1).upper()
            normalized_header = SECTION_ALIASES.get(raw_header, raw_header.title())
            start_index = match.end()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(docstring)
            sections[normalized_header] = docstring[start_index:end_index]

        return sections

    def _check_missing_args(self, node, sections):
        """check missing args.

        Args:
            node (Any): Description of node.
            sections (Any): Description of sections.
        """
        args_to_check = []
        for arg in node.args.args:
            if arg.arg not in ("self", "cls"):
                args_to_check.append(arg.arg)
        for arg in node.args.kwonlyargs:
            args_to_check.append(arg.arg)
        if node.args.vararg:
            args_to_check.append(node.args.vararg.arg)
        if node.args.kwarg:
            args_to_check.append(node.args.kwarg.arg)

        if not args_to_check:
            return

        if "Args" not in sections:
            self.add_violation(node, f"Missing 'Args' section. Expected: {', '.join(args_to_check)}")
            return

        args_content = sections["Args"]
        missing = []
        for arg in args_to_check:
            pattern = re.compile(rf"^\s*{re.escape(arg)}\s*(\(|:)", re.MULTILINE)
            if not pattern.search(args_content):
                missing.append(arg)

        if missing:
            self.add_violation(node, f"Missing description for arguments: {', '.join(missing)}")

    def _check_returns_yields(self, node, sections):
        """check returns yields.

        Args:
            node (Any): Description of node.
            sections (Any): Description of sections.
        """
        has_yield = False
        has_return_value = False

        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child is not node:
                    continue
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                has_yield = True
            elif isinstance(child, ast.Return) and child.value is not None:
                has_return_value = True

        if has_yield and "Yields" not in sections:
            self.add_violation(node, "Function yields data but missing 'Yields' section")

        if has_return_value and node.name != "__init__" and "Returns" not in sections:
            self.add_violation(node, "Function returns data but missing 'Returns' section")

    def visit_Module(self, node):
        """Visit module.

        Args:
            node (Any): Description of node.
        """
        docstring = ast.get_docstring(node)
        if not docstring:
            self.add_violation(node, "Missing module-level docstring")
        else:
            sections = self._parse_docstring_sections(docstring)
            if "Attributes" not in sections:
                self.add_violation(node, "Module docstring missing 'Attributes' section")
            if "Example" not in sections:
                self.add_violation(node, "Module docstring missing 'Example' section")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit classdef.

        Args:
            node (Any): Description of node.
        """
        prev_class = self.current_class
        self.current_class = node.name
        docstring = ast.get_docstring(node)
        if not docstring:
            self.add_violation(node, "Missing class docstring")
        else:
            sections = self._parse_docstring_sections(docstring)
            if "Attributes" not in sections:
                self.add_violation(node, "Class docstring missing 'Attributes' section")
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        """Visit functiondef.

        Args:
            node (Any): Description of node.
        """
        self._validate_function(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit asyncfunctiondef.

        Args:
            node (Any): Description of node.
        """
        self._validate_function(node)

    def _validate_function(self, node):
        """validate function.

        Args:
            node (Any): Description of node.
        """
        docstring = ast.get_docstring(node)
        if not docstring:
            self.add_violation(node, "Missing function docstring")
            return
        sections = self._parse_docstring_sections(docstring)
        self._check_missing_args(node, sections)
        self._check_returns_yields(node, sections)


def analyze_file(filepath):
    """Analyze file.

    Args:
    filepath (Any): Description of filepath.

    Returns:
        Any: Description of return value.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        validator = GoogleStyleValidator(filepath)
        validator.visit(tree)
        return validator.violations
    except SyntaxError:
        return [{"line": 0, "context": "Error", "name": "File", "message": "Syntax Error - Could not parse"}]
    except Exception as e:
        return [{"line": 0, "context": "Error", "name": "File", "message": str(e)}]


def display_report(all_violations: List[dict]):
    """Renders the final report using Rich Tables."""
    if not all_violations:
        console.print(
            Panel(
                "[bold green]SUCCESS[/bold green]: All scanned files comply with Google Docstring standards!",
                title="Validation Complete",
                expand=False,
            )
        )
        sys.exit(0)

    # create a table
    table = Table(
        title=f"Google Style Violations ({len(all_violations)} Found)",
        box=box.ROUNDED,
        header_style="bold white",
    )

    table.add_column("Location", style="dim cyan", no_wrap=True)
    table.add_column("Context", style="bold")
    table.add_column("Issue", style="red")

    # Sort by filepath then line number for readability
    sorted_violations = sorted(all_violations, key=lambda x: (x["filepath"], x["line"]))

    last_file = None
    for v in sorted_violations:
        # Group visually by adding a section break if the file changes
        if last_file and v["filepath"] != last_file:
            table.add_section()

        last_file = v["filepath"]

        # Color code the context
        ctx = v["context"]
        if ctx == "Module":
            ctx_style = "[blue]Module[/blue]"
        elif ctx == "ClassDef":
            ctx_style = "[yellow]Class[/yellow]"
        elif ctx == "Error":
            ctx_style = "[bold red]ERROR[/bold red]"
        else:
            ctx_style = f"[green]{ctx}[/green]"

        # Format location (File:Line)
        location = f"{v['filepath']}:{v['line']}"

        table.add_row(location, ctx_style, v["message"])

    console.print(table)
    console.print(f"\n[bold red]FAILURE:[/bold red] Found {len(all_violations)} issues.")
    sys.exit(1)


def main():
    """Main."""
    parser = argparse.ArgumentParser(description="Strict Google Style Docstring Validator.")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to scan")
    args = parser.parse_args()

    skip_dirs = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "dist", "build"}
    all_violations = []

    # Intro
    console.rule("[bold cyan]Napoleon Google Style Validator[/bold cyan]")
    console.print(
        "Checking for: [yellow]Attributes[/yellow] (Modules/Classes), [yellow]Example[/yellow] (Modules), [yellow]Args/Returns[/yellow] (Functions)",
        justify="center",
        style="dim",
    )
    print()

    # Collect targets
    targets = []
    if os.path.isfile(args.path):
        targets.append(("", "", [os.path.basename(args.path)]))
    else:
        for root, dirs, files in os.walk(args.path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            targets.append((root, dirs, files))

    # Run Analysis with Spinner
    with console.status("[bold green]Scanning codebase...[/bold green]", spinner="dots"):
        for root, _, files in targets:
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)

                    # Calculate clean relative path for display
                    if os.path.isfile(args.path):
                        display_path = os.path.basename(full_path)
                    else:
                        try:
                            display_path = os.path.relpath(full_path, os.getcwd())
                        except ValueError:
                            display_path = full_path

                    violations = analyze_file(full_path)

                    # Inject filepath into violation dicts for the reporter
                    for v in violations:
                        v["filepath"] = display_path
                        all_violations.append(v)

    # Show results
    display_report(all_violations)


if __name__ == "__main__":
    main()
