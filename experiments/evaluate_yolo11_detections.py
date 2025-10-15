# evaluate_yolo11_detections.py
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.append(os.path.abspath("src"))
from paths import PathManager

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODELS = {
    "yolo11n-seg": "../models/yolo11n-seg.pt",
    "yolo11s-seg": "../models/yolo11s-seg.pt",
    "yolo11m-seg": "../models/yolo11m-seg.pt",
    "yolo11l-seg": "../models/yolo11l-seg.pt",
    "yolo11x-seg": "../models/yolo11x-seg.pt",
}

ITEMS = {
    "apple": 507,
    "banana": 501,
}

OUTDIR = "results/yolo11_detection"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def run_detection(model_name, model_path, item, n_samples):
    model = YOLO(model_path)
    pm = PathManager(item)

    detail_rows = []
    t_item_start = time.time()
    n_detected = 0

    for idx in range(n_samples):
        img = np.load(pm.get_rgb_image_path(idx))
        if img.shape[2] > 3:
            img = img[:, :, 0:3]

        results = model(img)
        detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
        contains_item = any(item in name.lower() for name in detected_classes)

        detail_rows.append({
            "model": model_name,
            "item": item,
            "dataset_id": idx,
            "detected": int(contains_item),
        })

        if contains_item:
            n_detected += 1

    duration = time.time() - t_item_start
    summary = {
        "model": model_name,
        "item": item,
        "detected": n_detected,
        "total": n_samples,
        "accuracy": n_detected / n_samples,
        "duration_sec": duration,
        "duration_per_item_sec": duration / n_samples,
    }
    return detail_rows, summary

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--outdir", default="./results/yolo11_detections", help="Output directory")
    args = ap.parse_args()

    all_details = []
    all_summaries = []

    for model_name, model_path in MODELS.items():
        print(f"[INFO] Running detection with {model_name}")
        for item, n_samples in ITEMS.items():
            detail_rows, summary = run_detection(model_name, model_path, item, n_samples)
            all_details.extend(detail_rows)
            all_summaries.append(summary)

    # Speichern
    df_details = pd.DataFrame(all_details)
    df_summ = pd.DataFrame(all_summaries)

    os.makedirs(args.outdir, exist_ok=True)
    df_details.to_csv(os.path.join(args.outdir, "detection_details.csv"), index=False)
    df_summ.to_csv(os.path.join(args.outdir, "detection_summary.csv"), index=False)

    print(f"[DONE] Wrote results to {OUTDIR}/")

if __name__ == "__main__":
    main()