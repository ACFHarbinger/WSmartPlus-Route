"""Script to clean up and remove the UI module from WSmart-Route."""

import re
import shutil
from pathlib import Path


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


def comment_justfile(justfile_path: Path):
    """Comment out Streamlit dashboard command in justfile."""
    if not justfile_path.exists():
        return
    print("Commenting out Streamlit dashboard target in justfile...")
    content = justfile_path.read_text(errors="ignore")

    content = re.sub(
        r"(?m)^(dashboard:\s*\n\s+.*)$",
        r"# \1 # AUTO-COMMENTED",
        content,
    )
    justfile_path.write_text(content)


def update_dockerfile(dockerfile_path: Path):
    """Update Dockerfile ENTRYPOINT to call main.py instead of Streamlit."""
    if not dockerfile_path.exists():
        return
    print("Updating Dockerfile entrypoint...")
    content = dockerfile_path.read_text(errors="ignore")
    content = re.sub(
        r'ENTRYPOINT\s+\["uv",\s*"run",\s*"streamlit",\s*"run",\s*"logic/dashboard_entry.py",.*\]',
        'ENTRYPOINT ["uv", "run", "python", "main.py"]',
        content,
    )
    dockerfile_path.write_text(content)


def main():
    root = get_project_root()
    print(f"Project root is: {root}")

    # 1. Delete UI folders and files
    to_delete = [
        root / "logic/src/ui",
        root / "logic/src/utils/ui",
        root / "logic/dashboard_entry.py",
        root / "logic/test/unit/ui",
        root / "logic/configs/ui",
        root / "logic/src/constants/dashboard.py",
    ]

    for path in to_delete:
        remove_path(path)

    # Update constants/__init__.py to remove dashboard import
    remove_constants_init_import(root, "dashboard")

    # 2. Modify justfile
    comment_justfile(root / "justfile")

    # 3. Update Dockerfile
    update_dockerfile(root / "docker/Dockerfile")

    print("\n--- UI Cleanup Complete! ---")


if __name__ == "__main__":
    main()
