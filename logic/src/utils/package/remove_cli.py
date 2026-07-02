"""Script to remove the CLI module and related entry-point files from WSmart-Route."""

import re
import shutil
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    root = get_project_root()
    rel = path.relative_to(root)
    if path.is_dir():
        print(f"Removing directory: {rel}")
        shutil.rmtree(path)
    else:
        print(f"Removing file: {rel}")
        path.unlink()


def remove_constants_init_import(root: Path, module_name: str) -> None:
    """Comment out a wildcard import from logic/src/constants/__init__.py."""
    init_path = root / "logic/src/constants/__init__.py"
    if not init_path.exists():
        return
    content = init_path.read_text(errors="ignore")
    pattern = rf"(?m)^(from logic\.src\.constants\.{re.escape(module_name)} import \*.*)$"
    new_content = re.sub(pattern, r"# \1  # AUTO-REMOVED", content)
    if new_content != content:
        print(f"Updated constants/__init__.py: commented out {module_name} import")
        init_path.write_text(new_content)


def patch_logic_init(init_path: Path) -> None:
    """Comment out parser_dispatch import and remove parser_entry_point from __all__."""
    if not init_path.exists():
        return
    content = init_path.read_text(errors="ignore")
    content = re.sub(
        r"(?m)^(from logic\.controllers\.parser_dispatch import parser_entry_point.*)$",
        r"# \1  # AUTO-REMOVED",
        content,
    )
    content = re.sub(r'"parser_entry_point",?\s*', "", content)
    content = re.sub(r"'parser_entry_point',?\s*", "", content)
    print(f"Patched: {init_path.relative_to(get_project_root())}")
    init_path.write_text(content)


def main() -> None:
    root = get_project_root()
    print(f"Project root: {root}")

    to_delete = [
        root / "logic/src/cli",
        root / "logic/src/file_system.py",
        root / "logic/controllers/parser_dispatch.py",
    ]
    for path in to_delete:
        remove_path(path)

    patch_logic_init(root / "logic/__init__.py")

    # stats.py constants are only used in the CLI — remove them with it.
    remove_path(root / "logic/src/constants/stats.py")
    remove_constants_init_import(root, "stats")

    print("\n--- CLI Cleanup Complete! ---")


if __name__ == "__main__":
    main()
