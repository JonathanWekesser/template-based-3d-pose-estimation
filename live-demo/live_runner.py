"""
Live 3D-Pose-Schätzung mit Intel RealSense-Kamera (Snapshot-Modus).
Drücke 's', um ein einzelnes Frame aufzunehmen und die Pose zu schätzen.
Drücke 'q', um das Programm zu beenden.
"""

import time
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
import sys
import os

sys.path.append(os.path.abspath("src"))
from camera import Camera
from config import load_config_yaml
from mask import masked_image
from template import TemplateLoader
from paths import PathManager
from segmentation import predict_segment
from pose_initializer import PoseInitializer
from exceptions import TargetNotFoundError, SegmentationError


# -----------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------

def scene_from_depth(depth_img: np.ndarray, cam: Camera) -> o3d.geometry.PointCloud:
    """Erstellt eine Punktwolke aus einem Tiefenbild unter Verwendung der Kameraparameter."""
    fx, fy, cx, cy = cam.get_camera_intrinsics()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=cam.width, height=cam.height,
        fx=fx, fy=fy, cx=cx, cy=cy
    )
    # RealSense gibt Tiefen in Millimetern -> Umwandlung in Meter
    depth_m = depth_img.astype(np.float32) / 1000.0
    return o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_m),
        intrinsic=intrinsic,
        extrinsic=np.eye(4)
    )


def process_snapshot(rgb, depth, item_name, cam, templates, pose_init):
    """Segmentiert das RGB-Bild, erstellt die Punktwolke und führt die Pose-Schätzung durch."""
    print("📸 Snapshot aufgenommen – Verarbeitung läuft...")

    try:
        pts, masked_rgb = predict_segment(rgb, item_name)
    except (TargetNotFoundError, SegmentationError) as e:
        print(f"❌ Segmentierung fehlgeschlagen: {e}")
        return

    full_scene = scene_from_depth(depth, cam)
    masked_depth = masked_image(pts, depth)
    scene = scene_from_depth(masked_depth, cam)

    best_template, best_ev, best_transform = None, None, None
    for template in templates:
        t0 = time.time()
        transformation, ev = pose_init.process_one(scene, template.pcd)
        duration = time.time() - t0

        if best_ev is None or ev.fitness > best_ev.fitness:
            best_template = template
            best_ev = ev
            best_transform = transformation

    print(f"✅ Beste Übereinstimmung: {best_template.name}, Fitness={best_ev.fitness:.3f}")

    # Optional: Maskiertes RGB anzeigen
    cv2.imshow("RGB (masked)", masked_rgb)
    cv2.waitKey(1)

    # Visualisierung der Szene und des registrierten Templates
    vis = o3d.visualization.Visualizer()
    vis.create_window("Pose Snapshot", width=800, height=600)
    vis_full = o3d.geometry.PointCloud(full_scene)
    vis_scene = o3d.geometry.PointCloud(scene)
    vis_template = o3d.geometry.PointCloud(best_template.pcd).transform(best_transform)
    vis_full.paint_uniform_color([0.2, 0.2, 0.2]) # Volle Szene grau
    vis_scene.paint_uniform_color([0, 1, 0])   # Szene: grün
    vis_template.paint_uniform_color([0, 0, 1])  # Template: blau
    vis.add_geometry(vis_full)
    vis.add_geometry(vis_scene)
    vis.add_geometry(vis_template)
    vis.run()
    vis.destroy_window()




# -----------------------------------------------------
# Hauptprogramm
# -----------------------------------------------------

def main():
    # 1. Konfiguration und Setup
    item_name = "apple"  # Beispielobjekt
    cfg = load_config_yaml("configs/universal.yaml")
    cam = Camera.from_yaml("data/camera_intrinsics.yaml", "D435i")
    pm = PathManager(item_name)
    templates = TemplateLoader(pm).load_templates(limit=3)

    pose_init = PoseInitializer(cfg, item_name, cam)

    # 2. RealSense-Setup
    pipeline = rs.pipeline()
    rs_config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    device = rs_config.resolve(pipeline_wrapper).get_device()
    usb_type = device.get_info(rs.camera_info.usb_type_descriptor)

    if "3" in usb_type:
        print("✅ USB 3 erkannt - volle Auflösung")
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    else:
        print("⚠️  USB 2.0 erkannt – Reduziere Auflösung und Framerate")
        rs_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        rs_config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

    pipeline.start(rs_config)
    print("\n🎥 Live-Stream gestartet. Drücke 's' für Snapshot, 'q' zum Beenden.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data())
            rgb = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_TURBO)
            images = np.hstack((rgb, depth_colormap))

            cv2.imshow("Live RGB and Depth", images)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                process_snapshot(rgb, depth, item_name, cam, templates, pose_init)
            elif key == ord('q'):
                print("👋 Beende Live-Stream.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
