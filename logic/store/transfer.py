"""Shared tree-transfer helpers for the upload/download CLI entry points."""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath

from logic.store.providers import CloudProvider


def collect_files(base: Path, pattern: str = "*") -> list[Path]:
    """Recursively collect files under *base* whose names match *pattern*."""
    if base.is_file():
        return [base]
    return sorted(p for p in base.rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, pattern))


def upload_tree(provider: CloudProvider, files: list[Path], base: Path, dest: str, dry_run: bool = False) -> int:
    """Upload *files* to *dest*, preserving their paths relative to *base*."""
    count = 0
    for f in files:
        rel = f.relative_to(base) if base.is_dir() else Path(f.name)
        remote = str(PurePosixPath(dest.strip("/")) / PurePosixPath(*rel.parts))
        if dry_run:
            print(f"  [dry-run] {f} → {provider.name}:{remote}")
        else:
            print(f"  Uploading {f} → {provider.name}:{remote}")
            provider.upload_file(f, remote)
        count += 1
    return count


def download_tree(
    provider: CloudProvider, remote_dir: str, target: Path, pattern: str = "*", dry_run: bool = False
) -> int:
    """Recursively download *remote_dir* into *target*, filtering file names by *pattern*."""
    count = 0
    for entry in provider.list_dir(remote_dir):
        remote = f"{remote_dir.strip('/')}/{entry['name']}" if remote_dir.strip("/") else entry["name"]
        if entry["is_dir"]:
            count += download_tree(provider, remote, target / entry["name"], pattern, dry_run)
        elif fnmatch.fnmatch(entry["name"], pattern):
            local = target / entry["name"]
            if dry_run:
                print(f"  [dry-run] {provider.name}:{remote} → {local}")
            else:
                print(f"  Downloading {provider.name}:{remote} → {local}")
                provider.download_file(remote, local)
            count += 1
    return count
