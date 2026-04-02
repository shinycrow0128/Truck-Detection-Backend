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
    Returns (result, message, output_image_path, direction).
    direction will be "OUTGOING", "INCOMING", or None (via get_direction).
    """
    try:
        model1 = get_first_model()
        model2 = get_second_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "unknown", f"Cannot open video: {video_path}", None, None

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_image_path = f"{video_name}_truck.jpg"

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
            return None, "unknown", msg, None, None

        if truck_frame_count_total < 10:
            msg = f"TRUCK NOT CONFIRMED — only {truck_frame_count_total} detected frames"
            print(msg)
            return None, "unknown", msg, None, None

        bin_status = "unknown"

        if best_frame is None or best_box is None:
            print("Cannot classify bin: no best frame or bounding box available")
        else:
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

            if empty_count > full_count:
                bin_status = "empty"
            else:
                bin_status = "full"

        # If multiple trucks are present in the video, pick the truck type with the most detected frames.
        # This matches the requested behavior like: Truck 1 = 65 frames, Truck 2 = 40 frames.
        counts_str = ""
        counts_sorted: list[tuple[int, int]] = []
        if truck_type_frame_counts:
            counts_sorted = sorted(
                truck_type_frame_counts.items(), key=lambda kv: kv[1], reverse=True
            )
            counts_str = ", ".join(
                [
                    f"Truck {truck_type + 1}: {count} frames"
                    for truck_type, count in counts_sorted
                ]
            )

        # Save one best frame per truck type that appears.
        # - Primary truck (highest frame count) keeps the old output name: `{video_name}_truck.jpg`
        # - Others get suffixed names: `{video_name}_truck_{n}.jpg`
        saved_paths: list[str] = []
        primary_truck_type: int | None = None
        if counts_sorted:
            primary_truck_type = counts_sorted[0][0]
            best_truck_id = primary_truck_type

        for idx, (truck_type, _count) in enumerate(counts_sorted):
            best_type_entry = best_per_type.get(int(truck_type))
            if best_type_entry is None:
                continue

            _conf, type_frame, type_box = best_type_entry
            if type_frame is None or type_box is None:
                continue

            x1, y1, x2, y2 = type_box
            label = f"Truck {truck_type + 1} : {get_direction(bin_status)}"
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

            if idx == 0:
                out_path = output_image_path
            else:
                out_path = f"{video_name}_truck_{truck_type + 1}.jpg"

            cv2.imwrite(out_path, type_frame)
            saved_paths.append(out_path)

        selected_image_path = output_image_path if saved_paths else None
        selected_label = (
            f"Truck {best_truck_id + 1}" if best_truck_id is not None else "Truck unknown"
        )
        msg = (
            f"{counts_str} | selected: {selected_label} | images: {', '.join(saved_paths)}"
            if counts_str and saved_paths
            else selected_label
        )
        print(best_truck_id, bin_status, msg, output_image_path, get_direction(bin_status))
        return best_truck_id, bin_status, msg, selected_image_path, get_direction(bin_status)

    except Exception as e:
        msg = f"Analysis failed: {str(e)}"
        print(f" ERROR: {msg}")
        return None, "unknown", msg, None, None
