from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os
import re
import time
import threading
from datetime import datetime
from collections import defaultdict
from botocore.config import Config
from pathlib import Path


# ─── AWS S3 ────────────────────────────────────────────────────────
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ─── YOLO + OpenCV ────────────────────────────────────────────────
from ultralytics import YOLO
import cv2
import numpy as np

# ─── Supabase ─────────────────────────────────────────────────────
from supabase import create_client, Client
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────
FIRST_MODEL_PATH = "first_step_model.pt"
SECOND_MODEL_PATH = "second_step_model.pt"
TRUCK_CLASS_ID = [0,1,2,3,4]
MIN_CONF = 0.7
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Supabase setup
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")   # ← use service_role key!
BUCKET_NAME = "truck_detections"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("!!! WARNING: Supabase URL or KEY not found in .env !!!")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[Supabase] Client initialized")

def _get_s3_client():
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION_NAME"),
        endpoint_url=os.getenv("ENDPOINT_URL"),
        config=Config(connect_timeout=20, read_timeout=20)
    )
    return s3_client

def _guess_content_type(path: str) -> str:
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
    Requires env: AWS_S3_BUCKET (+ AWS credentials via env/instance role).
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
                "ContentType": _guess_content_type(file_path),
                'ContentDisposition': 'inline'
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
    Requires env: AWS_S3_BUCKET (+ AWS credentials via env/instance role).
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
                "ContentType": 'image/jpg',
                'ContentDisposition': 'inline'
            },
        )
        url = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{object_key}"

    except (BotoCoreError, ClientError) as e:
        print(f"[S3] Upload failed: {e}")
        return None

    return url

def insert_video_row(supabase: Client, dt: datetime, video_url: str):
    try:
        response = (
            supabase.table("video")
            .insert({
                "date": dt.isoformat(),
                "video_url": video_url
            })
            .execute()
        )

        if response.data and len(response.data) > 0:
            inserted_row = response.data[0]
            video_id = inserted_row["id"]          # ← this is your auto-generated ID
            return video_id                         # or return the whole dict if you want more
        else:
            print("[Supabase] Insert appeared to succeed but no row returned")
            return None
    except Exception as e:
        print(f"[Supabase] Failed: {e}")
        return None

def get_camera_id(supabase: Client, camera_name: str) -> str | None:
    """
    Get camera id by camera_name.
    Returns the id (usually uuid string) or None if not found.
    """
    try:
        response = (
            supabase.table("camera")
            .select("id")
            .eq("camera_name", camera_name)
            .maybe_single()
            .execute()
        )
        
        if response.data:
            return response.data["id"]
        return None
        
    except Exception as e:
        print(f"Error while fetching camera '{camera_name}': {e}")
        return None

def get_truck_id(supabase: Client, truck_name: int | str) -> str | None:
    """
    Get truck id by truck_name.
    Returns the id (usually uuid string) or None if not found.
    """
    try:
        search_name = f"Truck-{int(truck_name) + 1}"
        response = (
            supabase.table("truck")
            .select("id")
            .eq("truck_name", search_name)
            .maybe_single()
            .execute()
        )
        
        if response.data:
            return response.data["id"]
        return None
        
    except Exception as e:
        print(f"Error while fetching truck '{truck_name}': {e}")
        return None

# Global models (lazy load)
_MODEL_DET  = None   # detection + tracking
_MODEL_CLS  = None   # classification (empty/full)

def get_first_model():
    global _MODEL_DET
    if _MODEL_DET is None:
        if not os.path.exists(FIRST_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {FIRST_MODEL_PATH}")
        _MODEL_DET = YOLO(FIRST_MODEL_PATH)
        print(f"[YOLO] Loaded model: {FIRST_MODEL_PATH}")
    return _MODEL_DET

def get_second_model():
    global _MODEL_CLS
    if _MODEL_CLS is None:
        if not os.path.exists(SECOND_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {SECOND_MODEL_PATH}")
        _MODEL_CLS = YOLO(SECOND_MODEL_PATH)
        print(f"[YOLO] Loaded model: {SECOND_MODEL_PATH}")
    return _MODEL_CLS

def parse_timestamp_from_filename(filename: str) -> datetime | None:
    """Extracts timestamp like 20260219085003 from Brunswick_00_20260219085003.mp4"""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Looking for 14-digit pattern: YYYYMMDDHHMMSS
    match = re.search(r'(\d{14})', base)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None

def get_direction(x):
    if x == "empty":
        return "OUTGOING"
    if x == "full":
        return "INCOMING"
    return None  # or raise an error, or return "unknown", etc.

def save_truck_detection(
    camera_id: str,
    truck_id: str | None,
    bin_status: str,
    truck_status: str,
    detection_time: datetime,
    image_path: str | None = None,
    video_path: str | None = None
):
    image_url = upload_image_to_s3(image_path)
    
    # Prepare record
    data = {
        "camera_id": get_camera_id(supabase, camera_id),
        "truck_id": get_truck_id(supabase, truck_id),
        "bin_status": bin_status.lower(),
        "truck_status": truck_status.lower(),
        "detected_at": detection_time.isoformat(),
        "image_url": image_url,
        "video_id": video_path,
    }

    try:
        response = supabase.table("truck_detections").insert(data).execute()
        if response.data:
            _safe_delete(image_path)
            return response.data[0]
        else:
            print("[Supabase] Insert returned no data")
    except Exception as e:
        print(f"[Supabase] Database insert failed: {e}")
        print("Data was:", data)

    return None

def _safe_delete(path: str | None):
    """Delete file if it exists — no crash on missing file"""
    if path and os.path.exists(path):
        try:
            os.remove(path)
            print(f"  Deleted local file: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Could not delete {path}: {e}")

def analyze_video_for_truck(video_path: str):
    """
    Returns (result: int, message: str, output_image_path: str or None, direction: str or None)
    direction will be "left", "right" or "unknown"
    """
    try:
        model1 = get_first_model()
        model2 = get_second_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "unknown", f"Cannot open video: {video_path}", None, None

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_image_path = f"{video_name}_truck.jpg"

        # ─── Tracking history ───────────────────────────────────────
        track_history = defaultdict(list)           # track_id → list of center_x
        track_confs   = defaultdict(list)           # track_id → list of confidences
        frame_count   = 0
        truck_frame_count_total = 0

        best_conf = 0.0
        best_frame = None
        best_box = None
        best_truck_id = None
        # best_track_id_final = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Run detection + tracking (persist tracks between frames)
            results = model1.track(
                frame,
                persist=True,
                conf=MIN_CONF,
                classes=[TRUCK_CLASS_ID],
                verbose=False
            )[0]

            if results.boxes.id is not None:   # tracking is active
                boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
                confs   = results.boxes.conf.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                ids     = results.boxes.id.cpu().numpy().astype(int)

                for box, conf, tid, cls_id in zip(boxes, confs, ids, cls_ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2

                    track_history[tid].append(center_x)
                    track_confs[tid].append(conf)

                    truck_frame_count_total += 1

                    # Keep best single frame for visualization
                    if conf > best_conf:
                        best_conf = conf
                        best_frame = frame.copy()
                        best_box = box
                        best_truck_id = cls_id
                        # best_track_id_final = tid
        cap.release()
        print(f" Total truck detections: {truck_frame_count_total} over {frame_count} frames")

        if not track_history:
            msg = "TRUCK NOT PRESENT (0) — no tracked objects"
            print(msg)                                      # no extra space
            return None, "unknown", msg, None, None

        elif truck_frame_count_total < 10:
            msg = f"TRUCK NOT CONFIRMED — only {truck_frame_count_total} detected frames"
            print(msg)
            return None, "unknown", msg, None, None

        # ─── Classify bin status (empty/full) ────────────────────────
        bin_status = "unknown"

        if best_frame is None or best_box is None:
            print("Cannot classify bin: no best frame or bounding box available")
        else:
            results_bin = model2.predict(
                source=video_path,
                conf=0.7,
                classes=[0, 1],
                stream=True,
                verbose=False
            )

            full_count = 0
            empty_count = 0

            for result in results_bin:
                if len(result.boxes) != 0:
                    best_idx = result.boxes.conf.argmax()
                    cls_id   = int(result.boxes.cls[best_idx])
                    if cls_id == 0:
                        empty_count += 1
                    else:
                        full_count += 1
            
            if (empty_count > full_count):
                bin_status = "empty"
            else:
                bin_status = "full"

        # ─── Draw best frame (optional improvement) ──────────────────
        if best_frame is not None and best_box is not None:

            x1, y1, x2, y2 = best_box
            label = (f"Truck {best_truck_id+1} : {get_direction(bin_status)}")
            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(best_frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(output_image_path, best_frame)
        msg = (f"Truck : {best_truck_id+1}")
        print(best_truck_id, bin_status, msg, output_image_path, get_direction(bin_status))
        return best_truck_id, bin_status, msg, output_image_path, get_direction(bin_status)

    except Exception as e:
        msg = f"Analysis failed: {str(e)}"
        print(f" ERROR: {msg}")
        return None, "unknown", msg, None, None


class ReolinkFTPHandler(FTPHandler):
    def on_file_received(self, file):
        print(f"\n[NEW FILE] {file}")
        try:
            if not os.path.exists(file):
                print(" File disappeared right after upload")
                return

            stat = os.stat(file)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            print(f" └─ Size: {size_mb:.2f} MB  ({stat.st_size:,} bytes)")
            print(f" └─ Modified: {mod_time}")

            lower_file = file.lower()
            if lower_file.endswith(VIDEO_EXTENSIONS):
                print(" → Video → starting truck detection...")

                def background_task():
                    dt = parse_timestamp_from_filename(file)
                    if dt is None:
                        dt = datetime.utcnow()
                        print(" Warning: could not parse timestamp from filename → using now()")

                    camera_id = os.path.basename(file).split('_')[0]

                    # 1) Upload original received file to S3
                    s3_url = upload_video_to_s3(file)

                    # 2) Save date + S3 url in Supabase table `videos`
                    if supabase is not None and s3_url:
                        video_id = insert_video_row(supabase, dt=dt, video_url=s3_url)

                    # # 3) Run detection pipeline (existing behavior)
                    truck_id, bin_status, message, img_path, direction = analyze_video_for_truck(file)

                    # if truck_id != None and img_path and supabase is not None:
                    #     print(" → Uploading to Supabase...")
                    #     save_truck_detection(
                    #         camera_id=camera_id,
                    #         truck_id=truck_id,          # ← change later if needed
                    #         bin_status=bin_status,
                    #         truck_status=direction,
                    #         detection_time=dt,
                    #         image_path=img_path,
                    #         video_path=video_id
                    #     )

                    print(" " + "─" * 70)

                threading.Thread(target=background_task, daemon=True).start()

            elif lower_file.endswith(('.jpg', '.jpeg', '.png')):
                print(" → Snapshot / image (no analysis)")
            else:
                print(" → Other file type")

        except Exception as e:
            print(f" Error processing file {file}: {e}")


def main():
    authorizer = DummyAuthorizer()
    authorizer.add_user("reolink", "H@rryP0tter", r"C:\reolink", perm="elradfmw")

    handler = ReolinkFTPHandler
    handler.authorizer = authorizer
    handler.banner = "Reolink FTP server ready."

    address = ('0.0.0.0', 21)
    server = FTPServer(address, handler)
    server.max_cons = 50
    server.max_cons_per_ip = 5

    print("FTP server started on port 21...")
    print("Waiting for Reolink uploads → truck analysis → Supabase upload (when truck confirmed)")
    server.serve_forever()


if __name__ == '__main__':
    main()