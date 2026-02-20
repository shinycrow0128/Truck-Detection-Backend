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
TRUCK_CLASS_ID = 1                  # ← most important: your truck class ID (COCO=2, custom model → check your data.yaml)
MIN_CONF = 0.30                     # minimum confidence to consider
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
    Process video → find frame with best truck detection → save annotated best frame
    Returns (success: bool, message: str, output_path: str or None)
    """
    try:
        model = get_yolo_model()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return False, f"Cannot open video: {video_path}", None

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_image_path = f"{video_name}_best_truck.jpg"

        best_conf = 0.0
        best_frame = None
        best_box = None
        best_track_id = None

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"  Processing frame {frame_count}...", end="\r")

            # Track → filter only truck class
            results = model.track(
                frame,
                persist=True,
                conf=MIN_CONF,
                classes=[TRUCK_CLASS_ID],      # ← crucial filter
                verbose=False
            )[0]

            if len(results.boxes) == 0:
                continue

            boxes = results.boxes
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None

            # Best in this frame
            max_idx = np.argmax(confs)
            this_conf = confs[max_idx]

            if this_conf > best_conf:
                best_conf = this_conf
                best_frame = frame.copy()
                best_box = boxes.xyxy[max_idx].cpu().numpy().astype(int)
                # best_class_id = int(clss[max_idx])   # we already know it's truck
                best_track_id = int(ids[max_idx]) if ids is not None else None

        cap.release()

        if best_frame is not None and best_conf >= MIN_CONF:
            x1, y1, x2, y2 = best_box

            label = f"truck {best_conf:.2f}"

            # Draw box + label
            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(best_frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imwrite(output_image_path, best_frame)

            msg = f"Best truck frame saved → {output_image_path} (conf={best_conf:.3f})"
            print(f"  {msg}")
            return True, msg, output_image_path

        else:
            msg = "No truck found with conf ≥ 0.3"
            print(f"  {msg}")
            return False, msg, None

    except Exception as e:
        msg = f"Video analysis failed: {str(e)}"
        print(f"  ERROR: {msg}")
        return False, msg, None


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

            # ─── VIDEO HANDLING ────────────────────────────────────────
            if lower_file.endswith(VIDEO_EXTENSIONS):
                print("  → Video file detected → starting truck analysis...")
                
                # Run in background thread so FTP doesn't block completely
                def background_analysis():
                    success, message, out_path = analyze_video_for_truck(file)
                    status = "SUCCESS" if success else "NO TRUCK / FAILED"
                    print(f"  [VIDEO ANALYSIS COMPLETE] {status}")
                    print(f"  → {message}")
                    if out_path:
                        print(f"  → Output image: {out_path}")

                threading.Thread(target=background_analysis, daemon=True).start()

            # ─── IMAGE HANDLING (optional) ─────────────────────────────
            elif lower_file.endswith(('.jpg', '.jpeg', '.png')):
                print("  → Image / snapshot")

            else:
                print("  → Other file type")

            print("  " + "─" * 60)

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