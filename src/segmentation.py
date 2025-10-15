# segmentation.py
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.append(os.path.abspath("src"))
from exceptions import InvalidInputImage, NoDetectionsError, TargetNotFoundError

MODEL = None
DEVICE = None

def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = YOLO("../models/yolo11n-seg.pt")
    return MODEL

def prepare_image_for_yolo(img: np.ndarray) -> np.ndarray:
    """Ensure 3-channel uint8 image for YOLO (drop alpha, expand gray, contiguous)."""
    if img is None:
        raise InvalidInputImage("predict_segment: received None image")

    # Handle dimensionality
    if img.ndim == 2:
        # Gray -> 3-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        c = img.shape[2]
        if c >= 3:
            # Drop alpha channel regardless of order (RGBA/BGRA)
            img = img[:, :, :3]
        elif c >= 3:
            img = img[:, :, :3]
        elif c == 1:
            # 1 channel but stored as 3D -> treat as gray
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise InvalidInputImage(f"Unexpected channel count: {c}")
    else:
        raise InvalidInputImage(f"Unexpected image shape: {img.shape}")

    # Ensure uint8 in 0..255
    if img.dtype != np.uint8:
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(img)

def predict_segment(rgb_image, search_item):
    model = get_model()
    device = get_device()
    rgb_image = prepare_image_for_yolo(rgb_image)

    results = model.predict(rgb_image, device=device, verbose=False)
    if not results:
        raise NoDetectionsError("No YOLO results returned.")
    seg_image = rgb_image.copy()
    pts = None
    # Draw contours only for search_item
    for result in results:
        if result is None or result.boxes is None or result.masks is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes.data):
            class_id = int(box[-1])  # Extract class ID
            label = model.names[class_id]  # Get class name

            if label == search_item:  # Only draw if the object is the given search_item
                pts = np.array(mask, dtype=np.int32)
                cv2.polylines(seg_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Green contour
    if pts is None:
        raise TargetNotFoundError(f"'{search_item}' not found.")

    return pts, seg_image
