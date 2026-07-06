"""
Upload a selected subset of the simulation results in assets/output/ to a
cloud drive (OneDrive/SharePoint, Google Drive or Dropbox).

The subset is selected by horizon directory (e.g. 30days, 90days), area
directory (e.g. riomaior100_plastic), distribution (emp, gamma3), run name
(e.g. sl_ftsp) and a filename glob — any filter left unset matches everything.

Credentials per provider are documented in logic/store/config.py.

Usage
-----
    # Everything from the 90-day runs, to Dropbox
    uv run python -m logic.store.upload.results --provider dropbox --horizons 90days

    # Only Rio Maior N=100 empirical logs from the 30-day runs, to OneDrive
    uv run python -m logic.store.upload.results --provider onedrive \\
        --horizons 30days --areas riomaior100_plastic --dists emp \\
        --pattern "log_*.json" --dest WSmart-Route/results
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path, PurePosixPath

from logic.store.providers import get_provider

OUTPUT_DIR = Path("assets/output")


def _match(name: str, allowed: list[str] | None) -> bool:
    return not allowed or any(fnmatch.fnmatch(name, a) for a in allowed)


def select_results(
    output_dir: Path,
    horizons: list[str] | None,
    areas: list[str] | None,
    dists: list[str] | None,
    runs: list[str] | None,
    pattern: str,
) -> list[Path]:
    """Collect result files matching the directory-level and filename filters."""
    files: list[Path] = []
    for horizon_dir in sorted(d for d in output_dir.iterdir() if d.is_dir()):
        if not _match(horizon_dir.name, horizons):
            continue
        for area_dir in sorted(d for d in horizon_dir.iterdir() if d.is_dir()):
            if not _match(area_dir.name, areas):
                continue
            for dist_dir in sorted(d for d in area_dir.iterdir() if d.is_dir()):
                if not _match(dist_dir.name, dists):
                    continue
                for run_dir in sorted(d for d in dist_dir.iterdir() if d.is_dir()):
                    if not _match(run_dir.name, runs):
                        continue
                    files += sorted(
                        f for f in run_dir.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, pattern)
                    )
    return files


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",")] if value else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", required=True, choices=["onedrive", "sharepoint", "gdrive", "dropbox"])
    p.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Local results root")
    p.add_argument("--horizons", default=None, help="Comma-separated horizon dirs (e.g. 30days,90days)")
    p.add_argument("--areas", default=None, help="Comma-separated area dir globs (e.g. riomaior*)")
    p.add_argument("--dists", default=None, help="Comma-separated distribution dirs (e.g. emp,gamma3)")
    p.add_argument("--runs", default=None, help="Comma-separated run dir globs (e.g. sl_*,la_ftsp)")
    p.add_argument("--pattern", default="*", help="Filename glob (e.g. 'log_*.json')")
    p.add_argument("--dest", default="WSmart-Route/results", help="Destination folder on the drive")
    p.add_argument("--dry-run", action="store_true", help="List the selection without uploading")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"Results dir not found: {output_dir}")

    files = select_results(
        output_dir, _csv(args.horizons), _csv(args.areas), _csv(args.dists), _csv(args.runs), args.pattern
    )
    if not files:
        raise SystemExit("No result files match the given filters.")
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"Selected {len(files)} files ({total_mb:.1f} MB) from {output_dir}")

    provider = None if args.dry_run else get_provider(args.provider)
    for f in files:
        remote = str(PurePosixPath(args.dest.strip("/")) / PurePosixPath(*f.relative_to(output_dir).parts))
        if provider is None:
            print(f"  [dry-run] {f} → {args.provider}:{remote}")
        else:
            print(f"  Uploading {f} → {provider.name}:{remote}")
            provider.upload_file(f, remote)
    print("Done.")


if __name__ == "__main__":
    main()
