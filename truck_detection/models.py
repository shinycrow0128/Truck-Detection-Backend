import os

from ultralytics import YOLO

from truck_detection.config import FIRST_MODEL_PATH, SECOND_MODEL_PATH

_MODEL_DET = None
_MODEL_CLS = None


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
