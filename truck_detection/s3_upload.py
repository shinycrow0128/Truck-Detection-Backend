import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def _get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION_NAME"),
        endpoint_url=os.getenv("ENDPOINT_URL"),
        config=Config(connect_timeout=20, read_timeout=20),
    )


def guess_content_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".mov"):
        return "video/quicktime"
    if lower.endswith(".avi"):
        return "video/x-msvideo"
    if lower.endswith(".mkv"):
        return "video/x-matroska"
    return "application/octet-stream"


def upload_video_to_s3(file_path: str) -> str | None:
    """
    Uploads a received video to S3 and returns a public URL (or None on failure).
    Requires env: BUCKET_NAME (+ AWS credentials via env/instance role).
    """
    filename = os.path.basename(file_path)
    object_key = f"videos/{filename}"
    bucket_name = os.getenv("BUCKET_NAME")

    s3 = _get_s3_client()
    if s3 is None:
        print("[S3] Not configured (missing AWS_S3_BUCKET) → skipping S3 upload")
        return None

    try:
        s3.upload_file(
            Filename=file_path,
            Bucket=bucket_name,
            Key=object_key,
            ExtraArgs={
                "ContentType": guess_content_type(file_path),
                "ContentDisposition": "inline",
            },
        )
        url = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{object_key}"

    except (BotoCoreError, ClientError) as e:
        print(f"[S3] Upload failed: {e}")
        return None

    return url


def upload_image_to_s3(file_path: str) -> str | None:
    """
    Uploads a detected image to S3 and returns a public URL (or None on failure).
    """
    object_key = f"images/{file_path}"
    bucket_name = os.getenv("BUCKET_NAME")

    s3 = _get_s3_client()
    if s3 is None:
        print("[S3] Not configured (missing AWS_S3_BUCKET) → skipping S3 upload")
        return None

    try:
        s3.upload_file(
            Filename=file_path,
            Bucket=bucket_name,
            Key=object_key,
            ExtraArgs={
                "ContentType": "image/jpg",
                "ContentDisposition": "inline",
            },
        )
        url = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{object_key}"

    except (BotoCoreError, ClientError) as e:
        print(f"[S3] Upload failed: {e}")
        return None

    return url
