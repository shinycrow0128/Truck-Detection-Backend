import os
import threading
import time
from datetime import datetime

from pyftpdlib.handlers import FTPHandler

from truck_detection import config
from truck_detection.supabase_ops import insert_video_row
from truck_detection.utils import parse_timestamp_from_filename


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
            if lower_file.endswith(config.VIDEO_EXTENSIONS):
                print(" → Video → starting truck detection...")

                def background_task():
                    from truck_detection.s3_upload import upload_video_to_s3
                    from truck_detection.video_analysis import analyze_video_for_truck

                    dt = parse_timestamp_from_filename(file)
                    if dt is None:
                        dt = datetime.utcnow()
                        print(" Warning: could not parse timestamp from filename → using now()")

                    camera_id = os.path.basename(file).split("_")[0]

                    s3_url = upload_video_to_s3(file)

                    video_id = None
                    if config.supabase is not None and s3_url:
                        video_id = insert_video_row(config.supabase, dt=dt, video_url=s3_url)

                    truck_id, bin_status, message, img_path, direction = analyze_video_for_truck(file)

                    # if truck_id != None and img_path and config.supabase is not None:
                    #     print(" → Uploading to Supabase...")
                    #     save_truck_detection(
                    #         camera_id=camera_id,
                    #         truck_id=truck_id,
                    #         bin_status=bin_status,
                    #         truck_status=direction,
                    #         detection_time=dt,
                    #         image_path=img_path,
                    #         video_path=video_id,
                    #     )

                    print(" " + "─" * 70)

                threading.Thread(target=background_task, daemon=True).start()

            elif lower_file.endswith((".jpg", ".jpeg", ".png")):
                print(" → Snapshot / image (no analysis)")
            else:
                print(" → Other file type")

        except Exception as e:
            print(f" Error processing file {file}: {e}")
