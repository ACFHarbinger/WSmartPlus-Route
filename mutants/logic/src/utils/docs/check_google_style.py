import argparse
import ast
import os
import re
import sys
from typing import Dict

# --- Configuration ---
# Valid headers based on Sphinx Napoleon Google Style
# Mapping alias -> Standard Name
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

# Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


class GoogleStyleValidator(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.violations = []
        self.current_class = None

    def add_violation(self, node, message):
        lineno = getattr(node, "lineno", 1)
        name = getattr(node, "name", "module")
        self.violations.append({"line": lineno, "context": type(node).__name__, "name": name, "message": message})

    def _parse_docstring_sections(self, docstring: str) -> Dict[str, str]:
        """
        Extracts sections (Args, Returns, etc.) from a docstring.
        Ignores indentation but respects the 'Header:' syntax.
        """
        if not docstring:
            return {}

        sections = {}
        # Regex matches "Header:" at the start of a line (ignoring whitespace)
        # It captures the header name.
        header_pattern = re.compile(
            r"^\s*(" + "|".join(set(SECTION_ALIASES.keys())) + r"):\s*$", re.MULTILINE | re.IGNORECASE
        )

        matches = list(header_pattern.finditer(docstring))

        for i, match in enumerate(matches):
            raw_header = match.group(1).upper()
            normalized_header = SECTION_ALIASES.get(raw_header, raw_header.title())

            start_index = match.end()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(docstring)

            content = docstring[start_index:end_index]
            sections[normalized_header] = content

        return sections

    def _check_missing_args(self, node, sections):
        """Checks if all function arguments are documented in 'Args'."""
        # 1. Collect arguments from the AST
        # Filter out 'self' and 'cls'
        args_to_check = []

        # Standard args
        for arg in node.args.args:
            if arg.arg not in ("self", "cls"):
                args_to_check.append(arg.arg)

        # Keyword-only args
        for arg in node.args.kwonlyargs:
            args_to_check.append(arg.arg)

        # *args and **kwargs
        if node.args.vararg:
            args_to_check.append(node.args.vararg.arg)
        if node.args.kwarg:
            args_to_check.append(node.args.kwarg.arg)

        if not args_to_check:
            return

        # 2. Check if 'Args' section exists
        if "Args" not in sections:
            # We allow 'Parameters' via the alias mapping
            self.add_violation(node, f"Missing 'Args' section. Expected: {', '.join(args_to_check)}")
            return

        # 3. Check if specific args are mentioned
        args_content = sections["Args"]
        missing = []
        for arg in args_to_check:
            # Regex: Look for the arg name at the start of a line, or followed by ( or :
            # Matches: "  arg_name (int):" or "  arg_name:"
            pattern = re.compile(rf"^\s*{re.escape(arg)}\s*(\(|:)", re.MULTILINE)
            if not pattern.search(args_content):
                missing.append(arg)

        if missing:
            self.add_violation(node, f"Missing description for arguments: {', '.join(missing)}")

    def _check_returns_yields(self, node, sections):
        """Analyzes function body to see if Returns/Yields is required."""
        has_yield = False
        has_return_value = False

        # Walk the function body nodes to find Return/Yield
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child is not node:
                    continue  # Don't look inside nested functions

            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                has_yield = True
            elif isinstance(child, ast.Return) and child.value is not None:
                has_return_value = True

        # Check Yields
        if has_yield and "Yields" not in sections:
            self.add_violation(node, "Function yields data but missing 'Yields' section")

        # Check Returns (Skip for __init__)
        if has_return_value and node.name != "__init__" and "Returns" not in sections:
            self.add_violation(node, "Function returns data but missing 'Returns' section")

    def visit_Module(self, node):
        docstring = ast.get_docstring(node)

        if not docstring:
            self.add_violation(node, "Missing module-level docstring")
        else:
            sections = self._parse_docstring_sections(docstring)

            # Constraint: Module must have 'Attributes'
            if "Attributes" not in sections:
                self.add_violation(node, "Module docstring missing 'Attributes' section")

            # Constraint: Module must have 'Example' (or Examples)
            if "Example" not in sections:
                self.add_violation(node, "Module docstring missing 'Example' section")

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name

        docstring = ast.get_docstring(node)

        if not docstring:
            self.add_violation(node, "Missing class docstring")
        else:
            sections = self._parse_docstring_sections(docstring)

            # Constraint: Class must have 'Attributes'
            if "Attributes" not in sections:
                self.add_violation(node, "Class docstring missing 'Attributes' section")

        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        self._validate_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._validate_function(node)

    def _validate_function(self, node):
        # Optional: Skip private functions if desired
        # if node.name.startswith('_') and node.name != '__init__': return

        docstring = ast.get_docstring(node)
        if not docstring:
            self.add_violation(node, "Missing function docstring")
            return

        sections = self._parse_docstring_sections(docstring)

        # 1. Validate Arguments
        self._check_missing_args(node, sections)

        # 2. Validate Returns / Yields
        self._check_returns_yields(node, sections)


def analyze_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        validator = GoogleStyleValidator(filepath)
        validator.visit(tree)
        return validator.violations
    except SyntaxError:
        return [
            {"line": 0, "context": "Error", "name": "File", "message": "Syntax Error - Could not parse Python file"}
        ]
    except Exception as e:
        return [{"line": 0, "context": "Error", "name": "File", "message": str(e)}]


def main():
    parser = argparse.ArgumentParser(description="Strict Google Style Docstring Validator (Napoleon Standards).")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to scan")
    args = parser.parse_args()

    total_issues = 0
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "dist", "build"}

    print(f"{CYAN}Initializing Napoleon Google Style Check...{RESET}")
    print(
        f"Checking for: {YELLOW}Attributes{RESET} (Modules/Classes), {YELLOW}Example{RESET} (Modules), {YELLOW}Args/Returns{RESET} (Functions)"
    )
    print("=" * 100)
    print(f"{'File':<50} | {'Line':<5} | {'Context':<10} | {'Issue'}")
    print("-" * 100)

    # Determine if path is file or dir
    targets = []
    if os.path.isfile(args.path):
        targets.append(("", "", [os.path.basename(args.path)]))
    else:
        for root, dirs, files in os.walk(args.path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            targets.append((root, dirs, files))

    found_issues = False

    for root, dirs, files in targets:
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                if os.path.isfile(args.path):
                    full_path = args.path  # Handle single file case

                violations = analyze_file(full_path)

                if violations:
                    found_issues = True
                    # Calculate relative path for cleaner output
                    try:
                        rel_path = os.path.relpath(full_path, start=os.getcwd())
                    except ValueError:
                        rel_path = full_path

                    for v in violations:
                        total_issues += 1

                        # Color logic
                        ctx_color = (
                            GREEN if v["context"] == "Module" else (YELLOW if v["context"] == "ClassDef" else RESET)
                        )

                        print(
                            f"{rel_path:<50} | {v['line']:<5} | {ctx_color}{v['context']:<10}{RESET} | {v['message']}"
                        )

    print("-" * 100)
    if found_issues:
        print(f"{RED}FAILED: Found {total_issues} docstring issues.{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}SUCCESS: Codebase complies with project documentation standards.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
