from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os
import time
from pathlib import Path

CONF = 0.8
CLASS_NAMES = {
    0: "1_empty",
    1: "1_full",
    2: "2_empty",
    3: "2_full",
    4: "3_empty",
    5: "3_full",
    6: "4_empty",
    7: "4_full",
    8: "5_empty",
    9: "5_full",
}

_MODEL = None


def _resolve_model_path() -> Path:
    here = Path(__file__).resolve().parent
    model_path = here / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Put best.pt in the same folder as FTP Server.py"
        )
    return model_path


def _get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        model_path = _resolve_model_path()
        _MODEL = YOLO(str(model_path))
    return _MODEL


def detect_truck(image_path: str) -> dict:
    model = _get_model()

    results = model.predict(
        source=image_path,
        conf=CONF,
        imgsz=640,
        verbose=True,
    )[0]

    if len(results.boxes) == 0:
        return {
            "detected": False,
            "message": "No truck detected (confidence below threshold)",
            "confidence_threshold": CONF,
        }

    best_idx = results.boxes.conf.argmax()
    cls_id = int(results.boxes.cls[best_idx])
    conf_val = float(results.boxes.conf[best_idx])

    label = CLASS_NAMES.get(cls_id, f"?? class {cls_id}")
    return {
        "detected": True,
        "label": label.upper(),
        "confidence": conf_val,
        "box_xyxy": results.boxes.xyxy[best_idx].tolist(),
        "num_detections": int(len(results.boxes)),
    }

class ReolinkFTPHandler(FTPHandler):
    def on_file_received(self, file):
        
        print(f"\n[NEW FILE UPLOADED] {file}")

        try:
            # Get file information immediately (no delay needed)
            if os.path.exists(file):
                stat = os.stat(file)

                size_mb = stat.st_size / (1024 * 1024)          # size in MB for readability
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))

                print(f"  └─ Size:     {stat.st_size:,} bytes ({size_mb:.2f} MB)")
                print(f"  └─ Modified: {mod_time}")
                print(f"  └─ Created:  {create_time} (Windows creation time)")

                # Optional extras you can add:
                # - File extension check
                if file.lower().endswith(('.jpg', '.jpeg')):
                    print("  → This is a snapshot!")

                    try:
                        print("\n" + "═" * 60)
                        pred = detect_truck(file)
                        if not pred.get("detected"):
                            print(f"→ {pred.get('message', 'No truck detected')}")
                        else:
                            print(f"Prediction : {pred['label']}")
                            print(f"Confidence : {pred['confidence']:.3f}")
                            print(f"Box coords : {pred['box_xyxy']}")
                            print(f"Number of detections: {pred['num_detections']}")
                        print("═" * 60)

                        out_path = Path(file).with_suffix(Path(file).suffix + ".truck.txt")
                        out_path.write_text(str(pred), encoding="utf-8")
                        print(f"[SAVED] {out_path}")
                    except Exception as e:
                        print(f"[DETECTION ERROR] {e}")
                elif file.lower().endswith('.mp4'):
                    print("  → This is a video clip!")
            else:
                print("  Warning: File disappeared right after upload?!")

        except Exception as e:
            print(f"  Error getting info for {file}: {e}")


def main():
    # Create a dummy authorizer for "virtual" users
    authorizer = DummyAuthorizer()
    authorizer.add_user("reolink", "H@rryP0tter", r"C:\reolink", perm="elradfmw")

    # Instantiate FTP handler
    handler = ReolinkFTPHandler
    handler.authorizer = authorizer

    # Optional: customize banner or other settings
    handler.banner = "Reolink FTP server ready."

    # Listen on all interfaces, port 21
    address = ('0.0.0.0', 21)

    server = FTPServer(address, handler)
    server.max_cons = 50
    server.max_cons_per_ip = 5

    print("FTP server started on port 21...")
    server.serve_forever()

if __name__ == '__main__':
    main()