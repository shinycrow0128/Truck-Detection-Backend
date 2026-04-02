import os
from collections import defaultdict

import cv2

from truck_detection.config import MIN_CONF, TRUCK_CLASS_ID
from truck_detection.models import get_first_model, get_second_model
from truck_detection.utils import parse_timestamp_from_filename

__all__ = ["analyze_video_for_truck", "get_direction", "parse_timestamp_from_filename"]


def get_direction(x: str) -> str | None:
    if x == "empty":
        return "OUTGOING"
    if x == "full":
        return "INCOMING"
    return None


def analyze_video_for_truck(video_path: str):
    """
    Returns a list of tuples:
      (truck_id, bin_status, msg, best_frame_image_path, direction)

    - `truck_id` is the 0-based truck class id coming from the tracker/model.
    - Only truck types with at least `MIN_TRUCK_FRAMES` detected frames are returned.
    - If no truck reaches the threshold, returns an empty list.
    """
    try:
        model1 = get_first_model()
        model2 = get_second_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        MIN_TRUCK_FRAMES = 20

        track_history = defaultdict(list)
        frame_count = 0
        truck_frame_count_total = 0
        truck_type_frame_counts = defaultdict(int)  # key: class id, value: frames where this class appears
        truck_track_frame_counts = defaultdict(int)  # key: tracker id, value: frames where this track id appears

        best_conf = 0.0
        best_frame = None
        best_box = None
        best_truck_id = None
        # Per truck type, keep the highest-confidence frame+box so we can draw the correct label later.
        best_per_type = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = model1.track(
                frame,
                persist=True,
                conf=MIN_CONF,
                classes=[TRUCK_CLASS_ID],
                verbose=False,
            )[0]

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)

                for box, conf, tid, cls_id in zip(boxes, confs, ids, cls_ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2
                    cls_key = int(cls_id)

                    track_history[tid].append(center_x)

                    truck_frame_count_total += 1
                    truck_track_frame_counts[int(tid)] += 1
                    truck_type_frame_counts[cls_key] += 1

                    prev_type_best = best_per_type.get(cls_key)
                    if prev_type_best is None or conf > prev_type_best[0]:
                        # Store a copy so later selections don't mutate the reference frame.
                        best_per_type[cls_key] = (float(conf), frame.copy(), box)

                    if conf > best_conf:
                        best_conf = conf
                        best_frame = frame.copy()
                        best_box = box
                        best_truck_id = cls_key
        cap.release()
        per_type = dict(
            sorted(truck_type_frame_counts.items(), key=lambda kv: kv[1], reverse=True)
        )
        per_track = dict(
            sorted(truck_track_frame_counts.items(), key=lambda kv: kv[1], reverse=True)
        )
        print(
            f" Total truck detections: {truck_frame_count_total} over {frame_count} frames"
            f" | per truck type: {per_type}"
            f" | per track id: {per_track}"
        )

        if not track_history:
            msg = "TRUCK NOT PRESENT (0) — no tracked objects"
            print(msg)
            return []

        # Filter trucks: if a truck appears in the video less than 20 frames, ignore it.
        counts_sorted_all = sorted(
            truck_type_frame_counts.items(), key=lambda kv: kv[1], reverse=True
        )
        qualified = [(t, c) for (t, c) in counts_sorted_all if c >= MIN_TRUCK_FRAMES]
        if not qualified:
            counts_str_all = ", ".join([f"Truck {t + 1}: {c} frames" for t, c in counts_sorted_all])
            msg = (
                f"TRUCK NOT CONFIRMED — {counts_str_all} "
                f"(min {MIN_TRUCK_FRAMES} frames each)"
            )
            print(msg)
            return []

        bin_status = "unknown"
        results_bin = model2.predict(
            source=video_path,
            conf=0.7,
            classes=[0, 1],
            stream=True,
            verbose=False,
        )

        full_count = 0
        empty_count = 0
        for result in results_bin:
            if len(result.boxes) != 0:
                best_idx = result.boxes.conf.argmax()
                cls_id = int(result.boxes.cls[best_idx])
                if cls_id == 0:
                    empty_count += 1
                else:
                    full_count += 1

        bin_status = "empty" if empty_count > full_count else "full"
        direction = get_direction(bin_status)

        truck_infos: list[tuple[int, str, str, str, str | None]] = []
        for truck_type, count in qualified:
            best_type_entry = best_per_type.get(int(truck_type))
            if best_type_entry is None:
                continue

            _conf, type_frame, type_box = best_type_entry
            if type_frame is None or type_box is None:
                continue

            x1, y1, x2, y2 = type_box
            label = f"Truck {truck_type + 1} : {direction}"
            cv2.rectangle(type_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(
                type_frame,
                label,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Naming requested: `{video_name}_Truck-{n}.jpg` (example: Brunswick_...Truck-1.jpg)
            out_path = f"{video_name}Truck-{truck_type + 1}.jpg"
            cv2.imwrite(out_path, type_frame)

            msg = f"Truck {truck_type + 1} ({count} frames)"
            print(truck_type, bin_status, msg, out_path, direction)
            truck_infos.append((truck_type, bin_status, msg, out_path, direction))

        return truck_infos

    except Exception as e:
        msg = f"Analysis failed: {str(e)}"
        print(f" ERROR: {msg}")
        return []
