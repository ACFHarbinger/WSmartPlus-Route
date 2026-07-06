"""
Credential and cache configuration for the cloud storage providers.

All secrets come from environment variables so nothing sensitive is committed:

Microsoft Graph (OneDrive / SharePoint)
    MSGRAPH_CLIENT_ID     Azure AD app registration (public client) — required
    MSGRAPH_TENANT_ID     Directory (tenant) id, default "common"
    MSGRAPH_SITE_ID       Optional SharePoint site id; when set, uploads target
                          the site's default document library instead of the
                          signed-in user's OneDrive

Google Drive
    GDRIVE_ACCESS_TOKEN   Short-lived OAuth2 access token, or
    GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN
                          for automatic access-token refresh

Dropbox
    DROPBOX_ACCESS_TOKEN  Access token, or
    DROPBOX_APP_KEY / DROPBOX_APP_SECRET / DROPBOX_REFRESH_TOKEN
                          for automatic access-token refresh

Device-flow tokens for Microsoft Graph are cached in ~/.wsmart_route/ so the
interactive sign-in only happens once per machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("WSR_STORE_CACHE", str(Path.home() / ".wsmart_route")))


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value else default


def require_env(name: str, hint: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing environment variable {name} — {hint}")
    return value


def load_token_cache(name: str) -> dict:
    path = CACHE_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_token_cache(name: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)
