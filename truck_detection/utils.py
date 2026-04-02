import os
import re
from datetime import datetime


def parse_timestamp_from_filename(filename: str) -> datetime | None:
    """Extracts timestamp like 20260219085003 from Brunswick_00_20260219085003.mp4"""
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(\d{14})", base)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def safe_delete(path: str | None) -> None:
    """Delete file if it exists — no crash on missing file."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
            print(f"  Deleted local file: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Could not delete {path}: {e}")
