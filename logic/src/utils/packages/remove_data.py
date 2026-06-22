"""Script to filter and remove unwanted datasets, distributions, and network strategies in WSmart-Route."""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Set


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


def parse_keep_list(input_str: str) -> List[str]:
    """Parse comma-separated list of names to keep."""
    if not input_str.strip():
        return []
    return [name.strip().lower() for name in input_str.split(",") if name.strip()]


def clean_init_file(init_path: Path, deleted_module_names: Set[str]):
    """Clean up imports, __all__, and registry dicts in __init__.py for deleted modules."""
    if not init_path.exists() or not deleted_module_names:
        return

    content = init_path.read_text(errors="ignore")
    lines = content.splitlines()
    new_lines = []

    deleted_classes = set()

    for line in lines:
        # Check if line imports from a deleted module
        # Format: from .xxx import yyy or from .sub.xxx import yyy
        match = re.search(r"from\s+\.([a-zA-Z0-9_.]+)\s+import\s+(.+)", line)
        is_deleted = False
        if match:
            imported_from = match.group(1)
            # Get the leaf module name
            module_name = imported_from.split(".")[-1]
            if module_name in deleted_module_names:
                is_deleted = True
                # Extract class/function name(s)
                import_clause = match.group(2)
                # Handle 'Class as Class' or 'Class as Alias'
                for part in import_clause.split(","):
                    part = part.strip()
                    if " as " in part:
                        alias = part.split(" as ")[-1].strip()
                        deleted_classes.add(alias)
                    else:
                        deleted_classes.add(part)

        if is_deleted:
            # Comment out or omit import line
            new_lines.append(f"# {line}  # AUTO-REMOVED")
        else:
            new_lines.append(line)

    content = "\n".join(new_lines)

    # Now remove deleted classes from __all__ list and registries
    for cls in deleted_classes:
        # Remove from __all__ (handles double or single quotes)
        content = re.sub(rf'"{re.escape(cls)}",?\s*', "", content)
        content = re.sub(rf"'{re.escape(cls)}',?\s*", "", content)

        # Remove from DISTRIBUTION_REGISTRY or STRATEGIES dictionaries
        # Format: "key": Class, or 'key': Class,
        content = re.sub(rf'"[a-zA-Z0-9_]+"\s*:\s*{re.escape(cls)},?\s*', "", content)
        content = re.sub(rf"'[a-zA-Z0-9_]+'\s*:\s*{re.escape(cls)},?\s*", "", content)

    # Clean up trailing/consecutive commas in lists/dicts that might have been left
    content = re.sub(r",\s*([\]}])", r"\1", content)

    init_path.write_text(content)


def filter_datasets(root: Path, keep_names: List[str]):
    """Filter files in logic/src/data/datasets/pytorch and logic/src/data/datasets/simulation."""
    datasets_dir = root / "logic/src/data/datasets"
    if not datasets_dir.exists():
        return

    print(f"\nFiltering datasets (keeping: {keep_names})...")
    deleted_modules = set()

    for sub in ["pytorch", "simulation"]:
        sub_dir = datasets_dir / sub
        if not sub_dir.exists():
            continue
        for p in sub_dir.glob("*.py"):
            if p.name == "__init__.py":
                continue
            stem = p.stem.lower()
            # Check if this dataset should be kept
            keep = False
            for name in keep_names:
                if name in stem:
                    keep = True
                    break
            if not keep:
                deleted_modules.add(stem)
                remove_path(p)

    clean_init_file(datasets_dir / "__init__.py", deleted_modules)


def filter_distributions(root: Path, keep_names: List[str]):
    """Filter files in logic/src/data/distributions."""
    dist_dir = root / "logic/src/data/distributions"
    if not dist_dir.exists():
        return

    print(f"\nFiltering distributions (keeping: {keep_names})...")
    deleted_modules = set()

    for p in dist_dir.glob("*.py"):
        if p.name in ["__init__.py", "base.py"]:
            continue
        stem = p.stem.lower()
        keep = False
        for name in keep_names:
            if name in stem:
                keep = True
                break
        if not keep:
            deleted_modules.add(stem)
            remove_path(p)

    clean_init_file(dist_dir / "__init__.py", deleted_modules)


def filter_network(root: Path, keep_names: List[str]):
    """Filter files in logic/src/data/network."""
    network_dir = root / "logic/src/data/network"
    if not network_dir.exists():
        return

    print(f"\nFiltering network strategies (keeping: {keep_names})...")
    deleted_modules = set()

    for p in network_dir.glob("*.py"):
        if p.name == "__init__.py":
            continue
        stem = p.stem.lower()
        keep = False
        for name in keep_names:
            if name in stem:
                keep = True
                break
        if not keep:
            deleted_modules.add(stem)
            remove_path(p)

    clean_init_file(network_dir / "__init__.py", deleted_modules)


def main():
    parser = argparse.ArgumentParser(description="Filter dataset, distribution, and network modules.")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated datasets to keep")
    parser.add_argument("--distributions", type=str, default="", help="Comma-separated distributions to keep")
    parser.add_argument("--network", type=str, default="", help="Comma-separated network strategies to keep")

    args = parser.parse_args()
    root = get_project_root()

    keep_datasets = parse_keep_list(args.datasets)
    keep_distributions = parse_keep_list(args.distributions)
    keep_network = parse_keep_list(args.network)

    if keep_datasets:
        filter_datasets(root, keep_datasets)
    else:
        print("No datasets specified to keep. Keeping all datasets.")

    if keep_distributions:
        filter_distributions(root, keep_distributions)
    else:
        print("No distributions specified to keep. Keeping all distributions.")

    if keep_network:
        filter_network(root, keep_network)
    else:
        print("No network strategies specified to keep. Keeping all network strategies.")

    print("\n--- Data Filtering Complete! ---")


if __name__ == "__main__":
    main()
