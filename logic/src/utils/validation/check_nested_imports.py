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
            is_suppress = False
            if (
                isinstance(func, ast.Name)
                and func.id == "suppress"
                or isinstance(func, ast.Attribute)
                and func.attr == "suppress"
            ):
                is_suppress = True

            if is_suppress:
                for arg in item.context_expr.args:
                    if isinstance(arg, ast.Name) and arg.id in ("ImportError", "ModuleNotFoundError"):
                        return True
    return False


def is_header_assignment(node: ast.stmt) -> bool:
    """
    Check if a node is a header-safe assignment.
    Allowed: Constants, os.environ updates, and Logger initialization.
    """
    if not isinstance(node, ast.Assign):
        return False

    # 1. Check for os.environ[...] = ...
    for target in node.targets:
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Attribute) and target.value.attr == "environ":
                return True
            if isinstance(target.value, ast.Name) and target.value.id == "environ":
                return True

    # 2. Check for Logger initialization (e.g., logger = get_pylogger(__name__))
    if isinstance(node.value, ast.Call):
        func = node.value.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            func_name = f"{func.value.id}.{func.attr}"

        if func_name in ("get_pylogger", "getLogger", "logging.getLogger"):
            return True

    # 3. Allow simple constant values (True, False, None, strings, numbers)
    return isinstance(node.value, (ast.Constant, ast.NameConstant, ast.Num, ast.Str))


def is_header_setup_call(node: ast.stmt) -> bool:
    """Check for common setup calls like matplotlib.use() or warnings.filterwarnings()."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False

    call = node.value
    func_name = ""

    # Handle matplotlib.use or warnings.filterwarnings
    if isinstance(call.func, ast.Attribute):
        if isinstance(call.func.value, ast.Name):
            func_name = f"{call.func.value.id}.{call.func.attr}"
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id

    safe_calls = {
        "matplotlib.use",
        "warnings.filterwarnings",
        "warnings.simplefilter",
        "os.environ.setdefault",
        "sys.path.append",
        "torch.set_default_dtype",
    }

    return func_name in safe_calls


def get_factory_line_ranges(tree: ast.AST) -> List[Tuple[int, int]]:
    """Identifies the start and end lines of all classes containing 'Factory'."""
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "Factory" in node.name:
            start = node.lineno
            # end_lineno is Python 3.8+. Fallback to walking children if not available.
            if hasattr(node, "end_lineno"):
                end = node.end_lineno
            else:
                end = max((n.lineno for n in ast.walk(node) if hasattr(n, "lineno")), default=start)
            ranges.append((start, end))
    return ranges  # type: ignore[return-value]


def extract_all_imports(node: ast.AST) -> Set[ast.AST]:
    """Recursively extract all import nodes from within a given AST node."""
    imports = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            imports.add(child)
    return imports  # type: ignore[return-value]


def analyze_file(filepath: Path, ignore_factories: bool = False) -> List[Tuple[int, str]]:
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

    factory_ranges = get_factory_line_ranges(tree) if ignore_factories else []

    # 1. Identify valid top-level imports
    for i, node in enumerate(tree.body):
        is_docstring = (
            i == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, (ast.Constant, getattr(ast, "Str", type(None))))
        )
        if is_docstring:
            continue

        if not found_non_import:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                valid_top_level_imports.add(node)
            elif (
                is_type_checking_block(node) or is_import_error_try_block(node) or is_suppress_import_error_block(node)
            ):
                valid_top_level_imports.update(extract_all_imports(node))  # type: ignore[arg-type]
            elif is_header_assignment(node) or is_header_setup_call(node):
                continue
            else:
                found_non_import = True

    # 2. Walk the AST to find nested imports
    for node in ast.walk(tree):  # type: ignore[assignment]
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in valid_top_level_imports:
            # Check if this node is inside an ignored factory class
            if any(start <= node.lineno <= end for start, end in factory_ranges):
                continue

            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
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
    parser.add_argument("-e", "--exclude", nargs="+", default=[], help="Directories to exclude")
    parser.add_argument(
        "-i", "--ignore_factories", action="store_true", help="Ignore nested imports inside Factory classes"
    )
    args = parser.parse_args()

    target_root = Path(args.directory)
    if not target_root.is_dir():
        print(f"Error: Directory '{target_root}' does not exist.")
        return

    print(f"Scanning '{target_root}'...")
    if args.ignore_factories:
        print("Ignoring imports inside Factory classes.")
    print("=" * 60)

    files_found = 0
    total_imports = 0
    for root, dirs, files in os.walk(target_root):
        dirs[:] = [d for d in dirs if d not in args.exclude]
        for filename in files:
            if filename.endswith(".py"):
                filepath = Path(root) / filename
                results = analyze_file(filepath, ignore_factories=args.ignore_factories)
                if results:
                    files_found += 1
                    print(f"\n📄 {filepath}")
                    for line_no, name in results:
                        print(f"   Line {line_no:<4} | 📦 {name}")
                        total_imports += 1

    print("\n" + "=" * 60)
    print(f"Done! Found {total_imports} nested imports across {files_found} files.")


if __name__ == "__main__":
    main()
