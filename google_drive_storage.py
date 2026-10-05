import io
import base64
import json
import os
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _load_credential_info():
    # Prefer the explicitly refreshed Base64 credential over any legacy/raw value
    # that may still exist on Render.
    encoded = os.getenv("GDRIVE_JSON_B64", "").strip()
    if encoded:
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            info = json.loads(decoded)
        except Exception as exc:
            raise RuntimeError("GDRIVE_JSON_B64 is not valid base64 service-account JSON") from exc
        return info, "GDRIVE_JSON_B64"

    raw = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
        return info, "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"

    secret_path = os.getenv(
        "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE",
        "/etc/secrets/google-drive-service-account.json",
    )
    if os.path.isfile(secret_path):
        try:
            with open(secret_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except Exception as exc:
            raise RuntimeError("Google Drive secret file is not valid JSON") from exc
        return info, secret_path

    raise RuntimeError("Google Drive credentials are not configured")


def credential_diagnostics():
    info, source = _load_credential_info()
    private_key = info.get("private_key", "") or ""
    return {
        "source": source,
        "type": info.get("type"),
        "project_id": info.get("project_id"),
        "client_email": info.get("client_email"),
        "private_key_id": info.get("private_key_id"),
        "token_uri": info.get("token_uri"),
        "private_key_has_pem_header": private_key.startswith("-----BEGIN PRIVATE KEY-----"),
        "private_key_has_pem_footer": private_key.rstrip().endswith("-----END PRIVATE KEY-----"),
        "private_key_length": len(private_key),
    }


def _credentials():
    info, _ = _load_credential_info()
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

def _service():
    return build("drive", "v3", credentials=_credentials(), cache_discovery=False)


def extract_folder_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("Google Drive folder URL or ID is required")
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", value)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise ValueError("Could not read Google Drive folder ID")


def list_images(folder_id: str):
    service = _service()
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id,name,mimeType,size,modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            orderBy="name",
        ).execute()
        for item in response.get("files", []):
            if item.get("mimeType", "").startswith("image/"):
                files.append(item)
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(file_id: str) -> bytes:
    service = _service()
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request, chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def get_file_metadata(file_id: str):
    service = _service()
    return service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,size,modifiedTime",
    ).execute()
