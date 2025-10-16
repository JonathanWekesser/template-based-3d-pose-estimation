# live-demo/demo.py
import time
import rclpy
from bridge.ros_camera_bridge import ROSCameraBridge
from camera import Camera
from config import load_config_yaml
from pose_initializer import PoseInitializer
from template import TemplateLoader
from paths import PathManager
from visualization.live_visualizer import LiveVisualizer
from mask import masked_image
from segmentation import predict_segment, TargetNotFoundError
from pcd_utils import depth_to_pcd

def main():
    rclpy.init()
    node = ROSCameraBridge()
    print("[INFO] ROS Camera Bridge gestartet.")

    # Setup deiner Pipeline
    pm = PathManager("apple")  # Beispielobjekt
    cfg = load_config_yaml("config.yaml")
    cam = Camera.from_yaml(pm.get_camera_path(), "realsense_d435")
    tl = TemplateLoader(pm)
    templates = tl.load_all()
    pose_init = PoseInitializer(cfg, "apple", cam)
    visualizer = LiveVisualizer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            rgb, depth = node.get_latest_frames()
            if rgb is None or depth is None:
                continue

            print("[INFO] Neues Frame empfangen, starte Segmentierung…")
            try:
                mask_pts, seg_img = predict_segment(rgb, "apple")
            except TargetNotFoundError:
                print("[WARN] Objekt nicht erkannt – warte auf nächste Frames…")
                continue

            # masked_depth = masked_image(mask_pts, depth)
            # scene = pose_init.prep.prepare(
            #     pose_init.cam_to_pointcloud(masked_depth)
            # )
            masked_depth = masked_image(mask_pts, depth)
            scene_raw = depth_to_pcd(masked_depth, cam)
            scene = pose_init.prep.prepare(scene_raw)

            print("[INFO] Registrierung läuft…")
            t_final, ev = pose_init.process_one(scene, templates[0].pcd)
            print(ev)

            # Visualisierung
            vis_scene = scene
            vis_template = templates[0].pcd.transform(t_final)
            visualizer.update(vis_scene, vis_template)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("[INFO] Demo beendet.")
    finally:
        visualizer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
