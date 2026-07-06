"""
Upload generated PowerPoint presentations to OneDrive / SharePoint.

Uses the Microsoft Graph API (device-code sign-in on first use; the refresh
token is cached under ~/.wsmart_route/). By default the newest .pptx in
assets/windows/ is uploaded to the signed-in user's OneDrive; pass --site-id
(or export MSGRAPH_SITE_ID) to target a SharePoint site's default document
library instead.

Required environment: MSGRAPH_CLIENT_ID (and optionally MSGRAPH_TENANT_ID,
MSGRAPH_SITE_ID) — see logic/store/config.py.

Usage
-----
    uv run python -m logic.store.upload.presentation
    uv run python -m logic.store.upload.presentation \\
        --file assets/windows/wsmart_route_results.pptx \\
        --dest "Presentations/WSmart-Route" \\
        --site-id "<sharepoint-site-id>"
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

from logic.store.providers import OneDriveProvider

PRESENTATIONS_DIR = Path("assets/windows")


def newest_presentation() -> Path:
    decks = sorted(PRESENTATIONS_DIR.glob("*.pptx"), key=lambda p: p.stat().st_mtime)
    if not decks:
        raise SystemExit(f"No .pptx files found under {PRESENTATIONS_DIR}")
    return decks[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=None,
                   help="Presentation to upload (default: newest .pptx in assets/windows/)")
    p.add_argument("--all", action="store_true", help="Upload every .pptx in assets/windows/")
    p.add_argument("--dest", default="WSmart-Route/presentations",
                   help="Destination folder on the drive")
    p.add_argument("--site-id", default=None,
                   help="SharePoint site id (default: personal OneDrive, or MSGRAPH_SITE_ID)")
    p.add_argument("--dry-run", action="store_true", help="List what would be uploaded and exit")
    args = p.parse_args()

    if args.all:
        files = sorted(PRESENTATIONS_DIR.glob("*.pptx"))
        if not files:
            raise SystemExit(f"No .pptx files found under {PRESENTATIONS_DIR}")
    else:
        files = [Path(args.file) if args.file else newest_presentation()]
    for f in files:
        if not f.exists():
            raise SystemExit(f"File not found: {f}")

    if args.dry_run:
        for f in files:
            print(f"  [dry-run] {f} → {PurePosixPath(args.dest.strip('/')) / f.name}")
        return

    provider = OneDriveProvider(site_id=args.site_id)
    target = "SharePoint site" if provider.site_id else "OneDrive"
    print(f"Uploading {len(files)} presentation(s) to {target} folder '{args.dest}'")
    for f in files:
        remote = str(PurePosixPath(args.dest.strip("/")) / f.name)
        print(f"  Uploading {f} → {remote}")
        provider.upload_file(f, remote)
    print("Done.")


if __name__ == "__main__":
    main()
