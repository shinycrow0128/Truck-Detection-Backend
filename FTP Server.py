from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os
import time
import threading

# ─── YOLO + OpenCV imports ────────────────────────────────────────
from ultralytics import YOLO
import cv2
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────
MODEL_PATH = "best.pt"              # change if needed
TRUCK_CLASS_ID = 1                  # ← most important: your truck class ID
MIN_CONF = 0.50                     # minimum confidence to count as detection
MIN_FRAMES_REQUIRED = 20            # ← new: how many frames needed to consider "truck present"

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Global model (lazy load)
_MODEL = None

def get_yolo_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        _MODEL = YOLO(MODEL_PATH)
        print(f"[YOLO] Loaded model: {MODEL_PATH}")
    return _MODEL


def analyze_video_for_truck(video_path: str):
    """
    Returns (result: int 0|1, message: str, output_path: str or None)
    
    result = 1  → truck appeared in ≥ 20 frames → best frame is saved
    result = 0  → truck in < 20 frames → no image saved
    """
    try:
        model = get_yolo_model()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return 0, f"Cannot open video: {video_path}", None

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_image_path = f"{video_name}_truck.jpg"

        best_conf = 0.0
        best_frame = None
        best_box = None
        truck_frame_count = 0

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 20 == 0:
                print(f"  Processing frame {frame_count}...", end="\r")

            # Only look for trucks
            results = model.track(
                frame,
                persist=True,
                conf=MIN_CONF,
                classes=[TRUCK_CLASS_ID],
                verbose=False
            )[0]

            if len(results.boxes) > 0:
                truck_frame_count += 1

                # Keep track of the best-confidence detection
                boxes = results.boxes
                confs = boxes.conf.cpu().numpy()
                max_idx = np.argmax(confs)
                this_conf = confs[max_idx]

                if this_conf > best_conf:
                    best_conf = this_conf
                    best_frame = frame.copy()
                    best_box = boxes.xyxy[max_idx].cpu().numpy().astype(int)

        cap.release()

        print(f"  Total frames with truck: {truck_frame_count}/{frame_count}")

        if truck_frame_count >= MIN_FRAMES_REQUIRED and best_frame is not None:
            x1, y1, x2, y2 = best_box

            label = f"truck {best_conf:.2f}  ({truck_frame_count} frames)"

            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(best_frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imwrite(output_image_path, best_frame)

            msg = (f"TRUCK DETECTED (result=1) — {truck_frame_count} frames ≥ {MIN_FRAMES_REQUIRED} "
                   f"— saved: {output_image_path} (best conf={best_conf:.3f})")
            print(f"  {msg}")
            return 1, msg, output_image_path

        else:
            msg = f"TRUCK NOT CONSIDERED PRESENT (result=0) — only {truck_frame_count} frames < {MIN_FRAMES_REQUIRED}"
            print(f"  {msg}")
            return 0, msg, None

    except Exception as e:
        msg = f"Video analysis failed: {str(e)}"
        print(f"  ERROR: {msg}")
        return 0, msg, None


class ReolinkFTPHandler(FTPHandler):
    def on_file_received(self, file):
        print(f"\n[NEW FILE UPLOADED] {file}")

        try:
            if not os.path.exists(file):
                print("  Warning: File disappeared immediately after upload")
                return

            stat = os.stat(file)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

            print(f"  └─ Size:     {stat.st_size:,} bytes ({size_mb:.2f} MB)")
            print(f"  └─ Modified: {mod_time}")

            lower_file = file.lower()

            if lower_file.endswith(VIDEO_EXTENSIONS):
                print("  → Video file → starting truck analysis...")

                def background_analysis():
                    result, message, out_path = analyze_video_for_truck(file)
                    status = "POSITIVE (1)" if result == 1 else "NEGATIVE (0)"
                    print(f"  [ANALYSIS COMPLETE] {status}")
                    print(f"  → {message}")
                    if out_path:
                        print(f"  → Output image: {out_path}")
                    print("  " + "─" * 70)

                threading.Thread(target=background_analysis, daemon=True).start()

            elif lower_file.endswith(('.jpg', '.jpeg', '.png')):
                print("  → Image / snapshot")

            else:
                print("  → Other file type")

        except Exception as e:
            print(f"  Error processing {file}: {e}")


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
    server.serve_forever()


if __name__ == '__main__':
    main()