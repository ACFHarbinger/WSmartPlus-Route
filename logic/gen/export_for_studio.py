"""Package simulation run artefacts into a Studio-compatible ``.wsroute`` bundle.

Creates a zip archive with a ``manifest.json`` index and all recognised data
files from a run output directory (CSV, JSON/JSONL, YAML, NPZ, PKL, PT, Parquet).
Optional ``--arrow`` emits Arrow IPC sidecars (``.arrow``) for each CSV so the
Studio DuckDB-Wasm pipeline can ingest without re-parsing.

Usage
-----
    uv run python logic/gen/export_for_studio.py assets/output/my_run
    uv run python logic/gen/export_for_studio.py assets/output/my_run -o exports/run.wsroute
    uv run python logic/gen/export_for_studio.py assets/output/my_run --arrow
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BUNDLE_VERSION = "1"
INCLUDE_EXTENSIONS = frozenset(
    {".csv", ".json", ".jsonl", ".yaml", ".yml", ".npz", ".pkl", ".pt", ".parquet", ".td", ".arrow"}
)


def collect_files(source: Path) -> list[Path]:
    """Return all bundle-eligible files under *source*, sorted by relative path."""
    return sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in INCLUDE_EXTENSIONS
    )


def csv_to_arrow_ipc(csv_path: Path, arrow_path: Path) -> None:
    """Write a CSV table as an Arrow IPC file (§G.8 Arrow sidecar)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    table = pacsv.read_csv(csv_path)
    arrow_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def build_arrow_sidecars(source: Path, csv_files: list[Path]) -> list[Path]:
    """Emit ``.arrow`` IPC sidecars next to each CSV under a temp staging tree."""
    sidecars: list[Path] = []
    for csv_path in csv_files:
        rel = csv_path.relative_to(source)
        arrow_rel = rel.with_suffix(".arrow")
        arrow_path = source / arrow_rel
        if arrow_path.exists():
            sidecars.append(arrow_path)
            continue
        csv_to_arrow_ipc(csv_path, arrow_path)
        sidecars.append(arrow_path)
    return sidecars


def export_bundle(source: Path, output: Path, *, include_arrow: bool = False) -> int:
    """Write a ``.wsroute`` zip bundle. Returns the number of packaged files."""
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Source must be a directory: {source}")

    files = collect_files(source)
    arrow_sidecars: list[Path] = []
    if include_arrow:
        csv_files = [f for f in files if f.suffix.lower() == ".csv"]
        arrow_sidecars = build_arrow_sidecars(source, csv_files)
        files = sorted(set(files) | set(arrow_sidecars))

    manifest = {
        "version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "file_count": len(files),
        "arrow_sidecars": len(arrow_sidecars),
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
    parser.add_argument(
        "--arrow",
        action="store_true",
        help="Emit Arrow IPC (.arrow) sidecars for each CSV before bundling",
    )
    args = parser.parse_args()

    source = args.source
    output = args.output or (source.parent / f"{source.name}.wsroute")
    count = export_bundle(source, output, include_arrow=args.arrow)
    arrow_note = " (with Arrow IPC sidecars)" if args.arrow else ""
    print(f"Wrote {output} ({count} data files + manifest){arrow_note}")


if __name__ == "__main__":
    main()
