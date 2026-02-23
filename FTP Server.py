from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os
import re
import time
import threading
from datetime import datetime
from collections import defaultdict

# ─── YOLO + OpenCV ────────────────────────────────────────────────
from ultralytics import YOLO
import cv2
import numpy as np

# ─── Supabase ─────────────────────────────────────────────────────
from supabase import create_client, Client
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────
MODEL_PATH = "best.pt"
TRUCK_CLASS_ID = 1
MIN_CONF = 0.50
MIN_FRAMES_REQUIRED = 5
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

# Global YOLO model (lazy load)
_MODEL = None

def get_yolo_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        _MODEL = YOLO(MODEL_PATH)
        print(f"[YOLO] Loaded model: {MODEL_PATH}")
    return _MODEL

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
    if x == "left":
        return "outgoing"
    if x == "right":
        return "incoming"
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
    """Minimal version used inside FTP handler"""
    if supabase is None:
        print("Supabase not configured → skipping database save")
        return None

    image_url = None
    video_url = None

    try:
        ts_str = detection_time.strftime("%Y%m%d_%H%M%S")
        if image_path and os.path.exists(image_path):
            filename = f"{camera_id}/{ts_str}_truck.jpg"
            with open(image_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=filename,
                    file=f,
                    file_options={"content-type": "image/jpeg"}
                )
            image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)

        if video_path and os.path.exists(video_path):
            filename = f"{camera_id}/{ts_str}_truck.mp4"
            with open(video_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=filename,
                    file=f,
                    file_options={"content-type": "video/mp4"}
                )
            video_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)

    except Exception as e:
        print(f"Supabase Storage upload failed: {e}")
    
    # Prepare record
    data = {
        "camera_id": get_camera_id(supabase, camera_id),
        "truck_id": truck_id,
        "bin_status": bin_status.lower(),
        "truck_status": get_direction(truck_status).lower(),
        "detected_at": detection_time.isoformat(),
        "image_url": image_url,
        "video_url": video_url,
    }

    try:
        response = supabase.table("truck_detections").insert(data).execute()
        if response.data:
            return response.data[0]
        else:
            print("[Supabase] Insert returned no data")
    except Exception as e:
        print(f"[Supabase] Database insert failed: {e}")
        print("Data was:", data)

    return None

def analyze_video_for_truck(video_path: str):
    """
    Returns (result: int 0|1, message: str, output_image_path: str or None, direction: str or None)
    direction will be "left", "right" or "unknown"
    """
    try:
        model = get_yolo_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, f"Cannot open video: {video_path}", None, None

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
        best_track_id = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Run detection + tracking (persist tracks between frames)
            results = model.track(
                frame,
                persist=True,
                conf=MIN_CONF,
                classes=[TRUCK_CLASS_ID],
                verbose=False
            )[0]

            if results.boxes.id is not None:   # tracking is active
                boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
                confs   = results.boxes.conf.cpu().numpy()
                ids     = results.boxes.id.cpu().numpy().astype(int)

                for box, conf, tid in zip(boxes, confs, ids):
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
                        best_track_id = tid

        cap.release()
        print(f" Total truck detections: {truck_frame_count_total} over {frame_count} frames")

        if not track_history:
            msg = "TRUCK NOT PRESENT (0) — no tracked objects"
            print(f" {msg}")
            return 0, msg, None, None

        # ─── Find best track (most reliable + longest) ───────────────
        best_track_id_final = None
        best_avg_conf = 0.0
        best_length = 0

        for tid, conf_list in track_confs.items():
            if len(conf_list) < MIN_FRAMES_REQUIRED:
                continue
            avg_conf = np.mean(conf_list)
            if avg_conf > best_avg_conf or (avg_conf == best_avg_conf and len(conf_list) > best_length):
                best_avg_conf = avg_conf
                best_length = len(conf_list)
                best_track_id_final = tid

        if best_track_id_final is None:
            msg = f"TRUCK NOT PRESENT (0) — no track ≥ {MIN_FRAMES_REQUIRED} frames"
            print(f" {msg}")
            return 0, msg, None, None

        # ─── Determine direction from best track ─────────────────────
        xs = track_history[best_track_id_final]
        if len(xs) < 2:
            direction = "unknown"
        else:
            first_x = xs[0]
            last_x  = xs[-1]
            delta_x = last_x - first_x

            if abs(delta_x) < 15:   # ← tune this threshold (pixels)
                direction = "unknown / stationary"
            elif delta_x > 0:
                direction = "right"     # x increases → left-to-right
            else:
                direction = "left"

        # ─── Draw best frame (optional improvement) ──────────────────
        if best_frame is not None and best_box is not None:
            x1, y1, x2, y2 = best_box
            label = (f"ID {best_track_id_final}  {best_avg_conf:.2f}  "
                     f"({len(xs)} frames)  {direction}")
            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(best_frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            cv2.imwrite(output_image_path, best_frame)

        msg = (f"TRUCK DETECTED (1) — track ID {best_track_id_final} — "
               f"{len(xs)} frames ≥ {MIN_FRAMES_REQUIRED} — "
               f"direction: {direction} — saved: {output_image_path}")
        print(f" {msg}")

        return 1, msg, output_image_path, direction

    except Exception as e:
        msg = f"Analysis failed: {str(e)}"
        print(f" ERROR: {msg}")
        return 0, msg, None, None


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
                    result, message, img_path, direction = analyze_video_for_truck(file)

                    status = "POSITIVE (1)" if result == 1 else "NEGATIVE (0)"
                    print(f" [ANALYSIS] {status}")
                    print(f" → {message}")

                    if result == 1 and img_path and supabase is not None:
                        dt = parse_timestamp_from_filename(file)
                        if dt is None:
                            dt = datetime.utcnow()
                            print(" Warning: could not parse timestamp from filename → using now()")
                        print(" → Uploading to Supabase...")
                        save_truck_detection(
                            camera_id=os.path.basename(file).split('_')[0],
                            truck_id="e613cecc-8ac8-48d5-ad7f-74e025fb6a42",          # ← change later if needed
                            bin_status="full",
                            truck_status=direction,
                            detection_time=dt,
                            image_path=img_path,
                            video_path=file
                        )

                    if img_path and os.path.exists(img_path):
                        print(f" → Best frame: {img_path}")
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