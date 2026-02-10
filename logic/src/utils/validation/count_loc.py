"""count_loc.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import count_loc
"""

import argparse
import ast
import os
from typing import Dict, Set

from rich.console import Console
from rich.table import Table


def get_docstring_lines(source: str) -> Set[int]:
    """Get docstring lines.

    Args:
    source (str): Description of source.

    Returns:
        Any: Description of return value.
    """
    doc_lines: Set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return doc_lines
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            const_node = node.body[0].value
            if isinstance(const_node.value, str):
                start = node.body[0].lineno
                end = getattr(node.body[0], "end_lineno", start)
                for i in range(start, end + 1):
                    doc_lines.add(i)
    return doc_lines


def analyze_file(filepath: str) -> Dict[str, int]:
    """Analyze file.

    Args:
    filepath (str): Description of filepath.

    Returns:
        Any: Description of return value.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines()
    except (UnicodeDecodeError, PermissionError):
        return {"code": 0, "comment": 0, "docstring": 0, "total": 0}

    docstring_lines_set = get_docstring_lines(source)
    n_code, n_comments, n_docs = 0, 0, 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if i in docstring_lines_set:
            n_docs += 1
        elif stripped.startswith("#"):
            n_comments += 1
        else:
            n_code += 1

    return {"code": n_code, "comment": n_comments, "docstring": n_docs, "total": n_code + n_comments + n_docs}


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--sort", choices=["code", "comment", "docstring", "total"], default="total")
    parser.add_argument("--limit", type=int, default=30, help="Number of files to show")
    args = parser.parse_args()

    console = Console()
    file_data = []
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules"}

    with console.status("[bold green]Analyzing codebase...", spinner="dots"):
        for root, dirs, files in os.walk(args.path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, args.path)
                    metrics = analyze_file(full_path)
                    if metrics["total"] > 0:
                        file_data.append({"path": rel_path, **metrics})

    file_data.sort(key=lambda x: x[args.sort], reverse=True)

    # Create the Rich Table
    table = Table(
        title=f"Codebase Analysis (Sorted by {args.sort.upper()})", title_style="bold magenta", show_footer=True
    )

    table.add_column("File Path", style="cyan", no_wrap=False, footer="TOTALS")
    table.add_column("Code", justify="right", style="green", footer=str(sum(d["code"] for d in file_data)))
    table.add_column("Comments", justify="right", style="yellow", footer=str(sum(d["comment"] for d in file_data)))
    table.add_column("Docstrings", justify="right", style="blue", footer=str(sum(d["docstring"] for d in file_data)))
    table.add_column("Total", justify="right", style="bold white", footer=str(sum(d["total"] for d in file_data)))

    for d in file_data[: args.limit]:
        table.add_row(d["path"], str(d["code"]), str(d["comment"]), str(d["docstring"]), str(d["total"]))

    console.print(table)
    if len(file_data) > args.limit:
        console.print(f"[dim]... and {len(file_data) - args.limit} more files.[/dim]")


if __name__ == "__main__":
    main()
