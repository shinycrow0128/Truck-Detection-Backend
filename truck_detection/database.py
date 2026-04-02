from datetime import datetime

from truck_detection import config
from truck_detection.s3_upload import upload_image_to_s3
from truck_detection.supabase_ops import get_camera_id, get_truck_id
from truck_detection.utils import safe_delete


def save_truck_detection(
    camera_id: str,
    truck_id: str | None,
    bin_status: str,
    truck_status: str,
    detection_time: datetime,
    image_path: str | None = None,
    video_path: str | None = None,
):
    sb = config.supabase
    if sb is None:
        print("[Supabase] Not configured — skip save_truck_detection")
        return None

    image_url = upload_image_to_s3(image_path)

    data = {
        "camera_id": get_camera_id(sb, camera_id),
        "truck_id": get_truck_id(sb, truck_id),
        "bin_status": bin_status.lower(),
        "truck_status": truck_status.lower(),
        "detected_at": detection_time.isoformat(),
        "image_url": image_url,
        "video_id": video_path,
    }

    try:
        response = sb.table("truck_detections").insert(data).execute()
        if response.data:
            safe_delete(image_path)
            return response.data[0]
        print("[Supabase] Insert returned no data")
    except Exception as e:
        print(f"[Supabase] Database insert failed: {e}")
        print("Data was:", data)

    return None
