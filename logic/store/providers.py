"""
Cloud drive providers used by logic/store/upload and logic/store/download.

Three providers share one small interface (upload_file / download_file /
list_dir), all implemented with plain ``requests`` against the vendors' REST
APIs — no heavyweight SDK dependencies:

  - OneDriveProvider   Microsoft Graph API (personal OneDrive or a SharePoint
                       site document library); interactive device-code sign-in
                       cached under ~/.wsmart_route/
  - GoogleDriveProvider  Google Drive v3 API (access token or refresh token)
  - DropboxProvider    Dropbox HTTP API v2 (access token or refresh token)

Use ``get_provider("onedrive" | "gdrive" | "dropbox")`` to obtain an instance.
Remote paths are always POSIX-style ("folder/subfolder/file.ext") relative to
the drive root; intermediate folders are created on demand.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from logic.store.config import env, load_token_cache, require_env, save_token_cache

CHUNK = 8 * 1024 * 1024  # 8 MiB upload chunks


class CloudProvider(ABC):
    """Minimal file-transfer interface shared by every cloud drive."""

    name: str = "abstract"

    @abstractmethod
    def upload_file(self, local: Path, remote_path: str) -> None:
        """Upload *local* to *remote_path* (POSIX path relative to drive root)."""

    @abstractmethod
    def download_file(self, remote_path: str, local: Path) -> None:
        """Download *remote_path* into *local* (parent dirs created)."""

    @abstractmethod
    def list_dir(self, remote_path: str) -> list[dict]:
        """List a remote folder → [{"name": str, "is_dir": bool}]."""


# ── Microsoft Graph (OneDrive / SharePoint) ────────────────────────────────────


class OneDriveProvider(CloudProvider):
    """OneDrive / SharePoint document library via the Microsoft Graph API."""

    name = "onedrive"
    GRAPH = "https://graph.microsoft.com/v1.0"
    SCOPE = "Files.ReadWrite.All Sites.ReadWrite.All offline_access"
    CACHE_FILE = "msgraph_token.json"

    def __init__(self, site_id: str | None = None):
        self.client_id = require_env(
            "MSGRAPH_CLIENT_ID",
            "register a public-client app in Azure AD and export its application id",
        )
        self.tenant = env("MSGRAPH_TENANT_ID", "common")
        self.site_id = site_id or env("MSGRAPH_SITE_ID")
        self._token: str | None = None

    # -- auth -------------------------------------------------------------------

    def _login_base(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant}/oauth2/v2.0"

    def _device_code_flow(self) -> dict:
        r = requests.post(
            f"{self._login_base()}/devicecode",
            data={"client_id": self.client_id, "scope": self.SCOPE},
            timeout=30,
        )
        r.raise_for_status()
        flow = r.json()
        print(f"\n{flow['message']}\n")  # "visit https://microsoft.com/devicelogin and enter CODE"
        interval = int(flow.get("interval", 5))
        while True:
            time.sleep(interval)
            tok = requests.post(
                f"{self._login_base()}/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": self.client_id,
                    "device_code": flow["device_code"],
                },
                timeout=30,
            )
            body = tok.json()
            if tok.status_code == 200:
                return body
            if body.get("error") in ("authorization_pending", "slow_down"):
                continue
            raise SystemExit(f"Device-code sign-in failed: {body.get('error_description', body)}")

    def _refresh(self, refresh_token: str) -> dict | None:
        r = requests.post(
            f"{self._login_base()}/token",
            data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": refresh_token,
                "scope": self.SCOPE,
            },
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        cache = load_token_cache(self.CACHE_FILE)
        body = None
        if cache.get("refresh_token"):
            body = self._refresh(cache["refresh_token"])
        if body is None:
            body = self._device_code_flow()
        save_token_cache(
            self.CACHE_FILE,
            {"refresh_token": body.get("refresh_token", cache.get("refresh_token"))},
        )
        self._token = body["access_token"]
        return self._token

    def _headers(self, **extra: str) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}", **extra}

    def _drive(self) -> str:
        return f"{self.GRAPH}/sites/{self.site_id}/drive" if self.site_id else f"{self.GRAPH}/me/drive"

    # -- transfer ----------------------------------------------------------------

    def upload_file(self, local: Path, remote_path: str) -> None:
        size = local.stat().st_size
        item = f"{self._drive()}/root:/{remote_path}"
        if size <= 4 * 1024 * 1024:
            r = requests.put(
                f"{item}:/content",
                headers=self._headers(**{"Content-Type": "application/octet-stream"}),
                data=local.read_bytes(),
                timeout=120,
            )
            r.raise_for_status()
            return
        r = requests.post(
            f"{item}:/createUploadSession",
            headers=self._headers(**{"Content-Type": "application/json"}),
            json={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
            timeout=30,
        )
        r.raise_for_status()
        session_url = r.json()["uploadUrl"]
        with local.open("rb") as fh:
            offset = 0
            while offset < size:
                chunk = fh.read(CHUNK)
                end = offset + len(chunk) - 1
                cr = requests.put(
                    session_url,
                    headers={
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end}/{size}",
                    },
                    data=chunk,
                    timeout=300,
                )
                cr.raise_for_status()
                offset += len(chunk)

    def download_file(self, remote_path: str, local: Path) -> None:
        r = requests.get(
            f"{self._drive()}/root:/{remote_path}:/content",
            headers=self._headers(),
            timeout=300,
            stream=True,
        )
        r.raise_for_status()
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    def list_dir(self, remote_path: str) -> list[dict]:
        base = self._drive()
        url = f"{base}/root:/{remote_path}:/children" if remote_path else f"{base}/root/children"
        entries: list[dict] = []
        while url:
            r = requests.get(url, headers=self._headers(), timeout=60)
            r.raise_for_status()
            body = r.json()
            entries += [{"name": it["name"], "is_dir": "folder" in it} for it in body.get("value", [])]
            url = body.get("@odata.nextLink")
        return entries


# ── Google Drive ───────────────────────────────────────────────────────────────


class GoogleDriveProvider(CloudProvider):
    """Google Drive v3 API with path-based folder resolution."""

    name = "gdrive"
    API = "https://www.googleapis.com/drive/v3"
    UPLOAD = "https://www.googleapis.com/upload/drive/v3"

    def __init__(self):
        self._token: str | None = None
        self._folder_ids: dict[str, str] = {"": "root"}

    def _get_token(self) -> str:
        if self._token:
            return self._token
        token = env("GDRIVE_ACCESS_TOKEN")
        if not token:
            client_id = require_env(
                "GDRIVE_CLIENT_ID",
                "set GDRIVE_ACCESS_TOKEN, or GDRIVE_CLIENT_ID/GDRIVE_CLIENT_SECRET/GDRIVE_REFRESH_TOKEN",
            )
            r = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": require_env("GDRIVE_CLIENT_SECRET", "OAuth client secret"),
                    "refresh_token": require_env("GDRIVE_REFRESH_TOKEN", "OAuth refresh token"),
                    "grant_type": "refresh_token",
                },
                timeout=30,
            )
            r.raise_for_status()
            token = r.json()["access_token"]
        self._token = token
        return token

    def _headers(self, **extra: str) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}", **extra}

    def _find_child(self, parent_id: str, name: str, folders_only: bool = False) -> dict | None:
        q = f"name = '{name}' and '{parent_id}' in parents and trashed = false"
        if folders_only:
            q += " and mimeType = 'application/vnd.google-apps.folder'"
        r = requests.get(
            f"{self.API}/files",
            headers=self._headers(),
            params={"q": q, "fields": "files(id, name, mimeType)"},
            timeout=60,
        )
        r.raise_for_status()
        files = r.json().get("files", [])
        return files[0] if files else None

    def _folder_id(self, remote_dir: str, create: bool = True) -> str:
        remote_dir = remote_dir.strip("/")
        if remote_dir in self._folder_ids:
            return self._folder_ids[remote_dir]
        parent = "root"
        built = []
        for part in [p for p in remote_dir.split("/") if p]:
            built.append(part)
            key = "/".join(built)
            if key in self._folder_ids:
                parent = self._folder_ids[key]
                continue
            found = self._find_child(parent, part, folders_only=True)
            if found:
                parent = found["id"]
            elif create:
                r = requests.post(
                    f"{self.API}/files",
                    headers=self._headers(**{"Content-Type": "application/json"}),
                    json={"name": part, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
                    timeout=60,
                )
                r.raise_for_status()
                parent = r.json()["id"]
            else:
                raise FileNotFoundError(f"Google Drive folder not found: {key}")
            self._folder_ids[key] = parent
        return parent

    def upload_file(self, local: Path, remote_path: str) -> None:
        remote = remote_path.strip("/")
        folder, _, fname = remote.rpartition("/")
        parent = self._folder_id(folder)
        existing = self._find_child(parent, fname)
        meta = {"name": fname} if existing else {"name": fname, "parents": [parent]}
        # Resumable upload handles any size and lets us PUT the bytes in one stream.
        if existing:
            init_url = f"{self.UPLOAD}/files/{existing['id']}?uploadType=resumable"
            init = requests.patch(init_url, headers=self._headers(**{"Content-Type": "application/json"}),
                                  json=meta, timeout=60)
        else:
            init_url = f"{self.UPLOAD}/files?uploadType=resumable"
            init = requests.post(init_url, headers=self._headers(**{"Content-Type": "application/json"}),
                                 json=meta, timeout=60)
        init.raise_for_status()
        session = init.headers["Location"]
        with local.open("rb") as fh:
            r = requests.put(session, data=fh, timeout=600)
        r.raise_for_status()

    def download_file(self, remote_path: str, local: Path) -> None:
        remote = remote_path.strip("/")
        folder, _, fname = remote.rpartition("/")
        parent = self._folder_id(folder, create=False)
        found = self._find_child(parent, fname)
        if not found:
            raise FileNotFoundError(f"Google Drive file not found: {remote}")
        r = requests.get(
            f"{self.API}/files/{found['id']}",
            headers=self._headers(),
            params={"alt": "media"},
            timeout=300,
            stream=True,
        )
        r.raise_for_status()
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    def list_dir(self, remote_path: str) -> list[dict]:
        parent = self._folder_id(remote_path, create=False)
        entries: list[dict] = []
        token = None
        while True:
            params = {
                "q": f"'{parent}' in parents and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType)",
            }
            if token:
                params["pageToken"] = token
            r = requests.get(f"{self.API}/files", headers=self._headers(), params=params, timeout=60)
            r.raise_for_status()
            body = r.json()
            entries += [
                {"name": f["name"], "is_dir": f["mimeType"] == "application/vnd.google-apps.folder"}
                for f in body.get("files", [])
            ]
            token = body.get("nextPageToken")
            if not token:
                return entries


# ── Dropbox ────────────────────────────────────────────────────────────────────


class DropboxProvider(CloudProvider):
    """Dropbox HTTP API v2."""

    name = "dropbox"
    API = "https://api.dropboxapi.com/2"
    CONTENT = "https://content.dropboxapi.com/2"
    SESSION_THRESHOLD = 128 * 1024 * 1024

    def __init__(self):
        self._token: str | None = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        token = env("DROPBOX_ACCESS_TOKEN")
        if not token:
            key = require_env(
                "DROPBOX_APP_KEY",
                "set DROPBOX_ACCESS_TOKEN, or DROPBOX_APP_KEY/DROPBOX_APP_SECRET/DROPBOX_REFRESH_TOKEN",
            )
            r = requests.post(
                "https://api.dropboxapi.com/oauth2/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": require_env("DROPBOX_REFRESH_TOKEN", "OAuth refresh token"),
                },
                auth=(key, require_env("DROPBOX_APP_SECRET", "app secret")),
                timeout=30,
            )
            r.raise_for_status()
            token = r.json()["access_token"]
        self._token = token
        return token

    def _headers(self, api_arg: dict | None = None, content: bool = False) -> dict:
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        if api_arg is not None:
            headers["Dropbox-API-Arg"] = json.dumps(api_arg)
        if content:
            headers["Content-Type"] = "application/octet-stream"
        return headers

    @staticmethod
    def _abs(remote_path: str) -> str:
        return "/" + remote_path.strip("/")

    def upload_file(self, local: Path, remote_path: str) -> None:
        size = local.stat().st_size
        dest = self._abs(remote_path)
        if size <= self.SESSION_THRESHOLD:
            r = requests.post(
                f"{self.CONTENT}/files/upload",
                headers=self._headers({"path": dest, "mode": "overwrite", "mute": True}, content=True),
                data=local.read_bytes(),
                timeout=600,
            )
            r.raise_for_status()
            return
        with local.open("rb") as fh:
            start = requests.post(
                f"{self.CONTENT}/files/upload_session/start",
                headers=self._headers({"close": False}, content=True),
                data=fh.read(CHUNK),
                timeout=300,
            )
            start.raise_for_status()
            session_id = start.json()["session_id"]
            offset = CHUNK
            while True:
                chunk = fh.read(CHUNK)
                if not chunk:
                    break
                if offset + len(chunk) < size:
                    r = requests.post(
                        f"{self.CONTENT}/files/upload_session/append_v2",
                        headers=self._headers(
                            {"cursor": {"session_id": session_id, "offset": offset}, "close": False},
                            content=True,
                        ),
                        data=chunk,
                        timeout=300,
                    )
                else:
                    r = requests.post(
                        f"{self.CONTENT}/files/upload_session/finish",
                        headers=self._headers(
                            {
                                "cursor": {"session_id": session_id, "offset": offset},
                                "commit": {"path": dest, "mode": "overwrite", "mute": True},
                            },
                            content=True,
                        ),
                        data=chunk,
                        timeout=300,
                    )
                r.raise_for_status()
                offset += len(chunk)

    def download_file(self, remote_path: str, local: Path) -> None:
        r = requests.post(
            f"{self.CONTENT}/files/download",
            headers=self._headers({"path": self._abs(remote_path)}),
            timeout=600,
            stream=True,
        )
        r.raise_for_status()
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    def list_dir(self, remote_path: str) -> list[dict]:
        entries: list[dict] = []
        body = {"path": self._abs(remote_path) if remote_path.strip("/") else ""}
        url = f"{self.API}/files/list_folder"
        while True:
            r = requests.post(url, headers=self._headers(**{}), json=body, timeout=60)
            r.raise_for_status()
            data = r.json()
            entries += [
                {"name": e["name"], "is_dir": e[".tag"] == "folder"} for e in data.get("entries", [])
            ]
            if not data.get("has_more"):
                return entries
            url = f"{self.API}/files/list_folder/continue"
            body = {"cursor": data["cursor"]}


PROVIDERS = {
    "onedrive": OneDriveProvider,
    "sharepoint": OneDriveProvider,  # alias — set MSGRAPH_SITE_ID or pass site_id
    "gdrive": GoogleDriveProvider,
    "dropbox": DropboxProvider,
}


def get_provider(name: str, **kwargs) -> CloudProvider:
    """Instantiate a provider by name ("onedrive"/"sharepoint", "gdrive", "dropbox")."""
    try:
        cls = PROVIDERS[name.lower()]
    except KeyError:
        raise SystemExit(f"Unknown provider '{name}' (choose from: {', '.join(PROVIDERS)})") from None
    return cls(**kwargs) if cls is OneDriveProvider else cls()
