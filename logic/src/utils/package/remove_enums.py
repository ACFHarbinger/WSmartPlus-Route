"""Script to clean up and remove the enums module and GlobalRegistry from WSmart-Route."""

import re
import shutil
from pathlib import Path


def get_project_root() -> Path:
    """Find WSmart-Route root directory."""
    return Path(__file__).resolve().parents[4]


def remove_path(path: Path):
    """Delete a file or directory safely."""
    if not path.exists():
        return
    root = get_project_root()
    rel_path = path.relative_to(root) if path.is_absolute() else path
    if path.is_dir():
        print(f"Removing directory: {rel_path}")
        shutil.rmtree(path)
    else:
        print(f"Removing file: {rel_path}")
        path.unlink()


def process_python_file(file_path: Path):
    """Remove GlobalRegistry imports and decorators from a python file."""
    try:
        content = file_path.read_text(errors="ignore")

        original_content = content

        # Remove imports from logic.src.enums
        content = re.sub(
            r"(?m)^\s*(?:from|import)\s+logic\.src\.enums\b.*$\n?",
            "",
            content,
        )

        # Remove @GlobalRegistry.register(...) decorators (handles multi-line as well)
        content = re.sub(
            r"(?s)@GlobalRegistry\.register\s*\([^)]*?\)\s*\n?",
            "",
            content,
        )

        if content != original_content:
            print(f"Cleaned enums from: {file_path.relative_to(get_project_root())}")
            file_path.write_text(content)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    root = get_project_root()
    print(f"Project root is: {root}")

    # 1. Delete the enums directory
    enums_dir = root / "logic/src/enums"
    remove_path(enums_dir)

    # 2. Process all Python files in logic/src and logic/test
    for p in root.glob("logic/**/*.py"):
        if "remove_enums.py" in p.name:
            continue
        process_python_file(p)

    print("\n--- Enums Cleanup Complete! ---")


if __name__ == "__main__":
    main()
