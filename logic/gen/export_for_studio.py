"""Package simulation run artefacts into a Studio-compatible ``.wsroute`` bundle.

Creates a zip archive with a ``manifest.json`` index and all recognised data
files from a run output directory (CSV, JSON/JSONL, YAML, NPZ, PKL, PT, Parquet).

Usage
-----
    uv run python logic/gen/export_for_studio.py assets/output/my_run
    uv run python logic/gen/export_for_studio.py assets/output/my_run -o exports/run.wsroute
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BUNDLE_VERSION = "1"
INCLUDE_EXTENSIONS = frozenset(
    {".csv", ".json", ".jsonl", ".yaml", ".yml", ".npz", ".pkl", ".pt", ".parquet", ".td"}
)


def collect_files(source: Path) -> list[Path]:
    """Return all bundle-eligible files under *source*, sorted by relative path."""
    return sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in INCLUDE_EXTENSIONS
    )


def export_bundle(source: Path, output: Path) -> int:
    """Write a ``.wsroute`` zip bundle. Returns the number of packaged files."""
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Source must be a directory: {source}")

    files = collect_files(source)
    manifest = {
        "version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "file_count": len(files),
        "files": [str(f.relative_to(source)) for f in files],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for file_path in files:
            zf.write(file_path, str(file_path.relative_to(source)))

    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a WSmart-Route Studio .wsroute bundle")
    parser.add_argument("source", type=Path, help="Run output directory to package")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .wsroute path (default: <source>.wsroute next to source dir)",
    )
    args = parser.parse_args()

    source = args.source
    output = args.output or (source.parent / f"{source.name}.wsroute")
    count = export_bundle(source, output)
    print(f"Wrote {output} ({count} data files + manifest)")


if __name__ == "__main__":
    main()
