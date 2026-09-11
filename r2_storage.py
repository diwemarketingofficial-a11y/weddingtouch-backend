import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")

if not all([R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET]):
    raise RuntimeError("Missing one or more R2 environment variables")

r2 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

def upload_file(file_bytes: bytes, key: str, content_type: str) -> str:
    r2.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return key

def delete_file(key: str):
    r2.delete_object(
        Bucket=R2_BUCKET,
        Key=key,
    )

def create_download_url(key: str, expires_in: int = 3600) -> str:
    return r2.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": R2_BUCKET,
            "Key": key,
        },
        ExpiresIn=expires_in,
    )

def create_upload_url(
    key: str,
    content_type: str,
    expires_in: int = 3600,
) -> str:
    return r2.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": R2_BUCKET,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )

def file_exists(key: str) -> bool:
    try:
        r2.head_object(
            Bucket=R2_BUCKET,
            Key=key,
        )
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
