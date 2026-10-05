import io
import json
import os
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _credentials():
    raw = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    secret_path = os.getenv(
        "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE",
        "/etc/secrets/google-drive-service-account.json",
    )
    if os.path.isfile(secret_path):
        return service_account.Credentials.from_service_account_file(
            secret_path,
            scopes=SCOPES,
        )

    raise RuntimeError("Google Drive credentials are not configured")

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
