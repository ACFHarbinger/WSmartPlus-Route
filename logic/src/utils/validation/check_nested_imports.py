import argparse
import ast
import os
from pathlib import Path
from typing import List, Set, Tuple


def is_type_checking_block(node: ast.stmt) -> bool:
    """Check if a node is an `if TYPE_CHECKING:` block."""
    if not isinstance(node, ast.If):
        return False

    if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
        return True
    return bool(isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")


def is_import_error_try_block(node: ast.stmt) -> bool:
    """Check if a node is a `try...except ImportError:` block."""
    if not isinstance(node, ast.Try):
        return False

    for handler in node.handlers:
        if handler.type is None:
            continue
        if isinstance(handler.type, ast.Name) and handler.type.id in ("ImportError", "ModuleNotFoundError"):
            return True
        if isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name) and elt.id in ("ImportError", "ModuleNotFoundError"):
                    return True
    return False


def is_suppress_import_error_block(node: ast.stmt) -> bool:
    """Check if a node is a `with contextlib.suppress(ImportError):` block."""
    if not isinstance(node, ast.With):
        return False

    for item in node.items:
        if isinstance(item.context_expr, ast.Call):
            func = item.context_expr.func

            # Check if the function being called is 'suppress' or 'contextlib.suppress'
            is_suppress = False
            if (
                isinstance(func, ast.Name)
                and func.id == "suppress"
                or isinstance(func, ast.Attribute)
                and func.attr == "suppress"
            ):
                is_suppress = True

            if is_suppress:
                # Check if ImportError or ModuleNotFoundError is passed as an argument
                for arg in item.context_expr.args:
                    if isinstance(arg, ast.Name) and arg.id in ("ImportError", "ModuleNotFoundError"):
                        return True
    return False


def extract_all_imports(node: ast.AST) -> Set[ast.AST]:
    """Recursively extract all import nodes from within a given AST node."""
    imports = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            imports.add(child)
    return imports  # type: ignore[return-value]


def analyze_file(filepath: Path) -> List[Tuple[int, str]]:
    """Parses a Python file and returns a list of nested imports."""
    nested_imports = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    valid_top_level_imports = set()
    found_non_import = False

    # 1. Identify valid top-level imports
    for i, node in enumerate(tree.body):
        # Check for module docstring
        is_docstring = False
        if (
            i == 0
            and isinstance(node, ast.Expr)
            and (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                or hasattr(ast, "Str")
                and isinstance(node.value, ast.Str)
            )
        ):
            is_docstring = True

        if is_docstring:
            continue

        if not found_non_import:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                valid_top_level_imports.add(node)
            elif (
                is_type_checking_block(node) or is_import_error_try_block(node) or is_suppress_import_error_block(node)
            ):
                valid_top_level_imports.update(extract_all_imports(node))  # type: ignore[arg-type]
            else:
                # Any other statement closes the top-level window
                found_non_import = True

    # 2. Walk the AST to find nested imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in valid_top_level_imports:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:  # ast.ImportFrom
                module = node.module or ""
                level = node.level or 0
                prefix = ("." * level) + module
                names = [f"{prefix}.{alias.name}" if prefix else alias.name for alias in node.names]

            for name in names:
                nested_imports.append((node.lineno, name))

    return sorted(nested_imports, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Find nested/delayed imports in Python files.")
    parser.add_argument("directory", type=str, help="The target directory to scan")
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        default=[],
        help="Directory names to exclude from the search (e.g. .venv node_modules tests)",
    )
    args = parser.parse_args()

    target_root = Path(args.directory)
    if not target_root.is_dir():
        print(f"Error: Directory '{target_root}' does not exist.")
        return

    exclude_set = set(args.exclude)

    print(f"Scanning '{target_root}'...")
    if exclude_set:
        print(f"Excluding directories: {', '.join(exclude_set)}")
    print("=" * 60)

    files_found = 0
    total_imports = 0

    for root, dirs, files in os.walk(target_root):
        dirs[:] = [d for d in dirs if d not in exclude_set]

        for filename in files:
            if filename.endswith(".py"):
                filepath = Path(root) / filename
                results = analyze_file(filepath)

                if results:
                    files_found += 1
                    print(f"\n📄 {filepath}")
                    for line_no, import_name in results:
                        print(f"   Line {line_no:<4} | 📦 {import_name}")
                        total_imports += 1

    print("\n" + "=" * 60)
    print(f"Done! Found {total_imports} nested imports across {files_found} files.")


if __name__ == "__main__":
    main()
