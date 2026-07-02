"""Script to clean up and remove the security module and file system commands from WSmart-Route."""

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


def patch_setup_env(file_path: Path):
    """Patch setup_env.py to remove security imports and decrypt usages."""
    if not file_path.exists():
        return
    print(f"Patching setup_env.py: {file_path}")
    content = file_path.read_text(errors="ignore")

    # Remove security imports
    content = re.sub(
        r"from logic\.src\.utils\.security import decrypt_file_data, load_key",
        "# from logic.src.utils.security import decrypt_file_data, load_key  # AUTO-REMOVED",
        content,
    )

    # Replace load_key and decrypt_file_data usages with ValueError
    content = re.sub(
        r"key\s*=\s*load_key\(symkey_name=symkey_name, env_filename=env_filename or \"\.env\"\)\s*\n\s*data\s*=\s*decrypt_file_data\(key, gplic_path\)",
        'raise ValueError("Symmetric cryptography is disabled because security utilities are removed.")',
        content,
    )

    file_path.write_text(content)


def patch_google_maps(file_path: Path):
    """Patch google.py to remove security imports and decrypt usages."""
    if not file_path.exists():
        return
    print(f"Patching google.py: {file_path}")
    content = file_path.read_text(errors="ignore")

    # Remove security imports
    content = re.sub(
        r"from logic\.src\.utils\.security import decrypt_file_data, load_key",
        "# from logic.src.utils.security import decrypt_file_data, load_key  # AUTO-REMOVED",
        content,
    )

    # Replace load_key and decrypt_file_data usages with ValueError
    content = re.sub(
        r"sym_key\s*=\s*load_key\(kwargs\[\"symkey_name\"\],\s*kwargs\[\"env_filename\"\]\)\s*\n\s*api_key\s*=\s*decrypt_file_data\(sym_key,\s*kwargs\[\"gapik_file\"\]\)",
        'raise ValueError("Symmetric cryptography is disabled because security utilities are removed.")',
        content,
    )

    file_path.write_text(content)


def patch_cli_registry(file_path: Path):
    """Patch registry.py to remove files_parser definition."""
    if not file_path.exists():
        return
    print(f"Patching CLI registry: {file_path}")
    content = file_path.read_text(errors="ignore")

    # Remove imports
    content = re.sub(
        r"from logic\.src\.cli\.fs_parser import add_files_args",
        "# from logic.src.cli.fs_parser import add_files_args  # AUTO-REMOVED",
        content,
    )

    # Comment out files subcommand registration
    pattern = r"(?s)(# Files\s*\n\s*files_parser\s*=\s*subparsers\.add_parser\(\"file_system\".*?\n\s*add_files_args\(files_parser\))"
    content = re.sub(
        pattern,
        r"# \1  # AUTO-REMOVED",
        content,
    )

    file_path.write_text(content)


def patch_cli_init(file_path: Path):
    """Patch cli/__init__.py to remove fs_parser validations."""
    if not file_path.exists():
        return
    print(f"Patching CLI init: {file_path}")
    content = file_path.read_text(errors="ignore")

    # Remove imports
    content = re.sub(
        r"from logic\.src\.cli\.fs_parser import add_files_args, validate_file_system_args",
        "# from logic.src.cli.fs_parser import add_files_args, validate_file_system_args  # AUTO-REMOVED",
        content,
    )

    # Comment out validation block
    pattern = r"(?s)(if command == \"file_system\":\s*\n\s*#.*?\n\s*return \(\"file_system\", inner_comm\), opts)"
    content = re.sub(
        pattern,
        r"# \1  # AUTO-REMOVED",
        content,
    )

    # Remove add_files_args from __all__
    content = re.sub(r'"add_files_args",?\s*', "", content)

    file_path.write_text(content)


def patch_parser_dispatch(file_path: Path):
    """Patch parser_dispatch.py to remove file_system imports and handlers."""
    if not file_path.exists():
        return
    print(f"Patching parser_dispatch.py: {file_path}")
    content = file_path.read_text(errors="ignore")

    # Remove imports
    pattern_import = r"(?s)from logic\.src\.file_system import\s*\(\s*delete_file_system_entries,\s*perform_cryptographic_operations,\s*update_file_system_entries,\s*\)"
    content = re.sub(
        pattern_import,
        "# from logic.src.file_system import ... # AUTO-REMOVED",
        content,
    )

    # Replace file_system dispatch logic with raise
    pattern_dispatch = r"(?s)(assert comm == \"file_system\".*?perform_cryptographic_operations\(opts\))"
    replacement = 'raise ValueError("File system operations are disabled because security and file system utilities are removed.")'
    content = re.sub(
        pattern_dispatch,
        replacement,
        content,
    )

    file_path.write_text(content)


def main():
    root = get_project_root()
    print(f"Project root is: {root}")

    # 1. Delete security directory and file_system.py, fs_parser.py
    remove_path(root / "logic/src/utils/security")
    remove_path(root / "logic/src/file_system.py")
    remove_path(root / "logic/src/cli/fs_parser.py")

    # 2. Delete test files
    remove_path(root / "logic/test/unit/test_file_system.py")
    remove_path(root / "logic/test/fixtures/security_fixtures.py")
    remove_path(root / "logic/test/unit/utils/security/test_crypto_utils.py")

    # 3. Patch dependent files
    patch_setup_env(root / "logic/src/utils/configs/setup_env.py")
    patch_google_maps(root / "logic/src/data/network/google.py")
    patch_cli_registry(root / "logic/src/cli/registry.py")
    patch_cli_init(root / "logic/src/cli/__init__.py")
    patch_parser_dispatch(root / "logic/controllers/parser_dispatch.py")

    print("\n--- Security Cleanup Complete! ---")


if __name__ == "__main__":
    main()
