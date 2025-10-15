# runner.py
import copy
import logging
import os
import sys
import time

import numpy as np
import open3d as o3d
import pandas as pd
from open3d.camera import PinholeCameraIntrinsic as o3dPinholeCameraIntrinsic
from tqdm import tqdm

sys.path.append(os.path.abspath("src"))
from dataset_loader import DatasetLoader
from mask import masked_image
from pose_initializer import PoseInitializer
from segmentation import predict_segment
from exceptions import TargetNotFoundError, SegmentationError
from evaluation import get_better_eval

RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]
BLACK = [0, 0, 0]

class ExperimentRunner:
    def __init__(self, config, item_name, cam, templates, out_dir=None, logger=None):
        self.config = config
        self.item_name = item_name
        self.cam = cam
        self.templates = templates
        self.out_dir = out_dir
        self.logger = logger or logging.getLogger("ExperimentRunner")
        self.dl = DatasetLoader(self.item_name)

    def run_range(self, idmin: int, idmax: int, return_only_best=False):
        pose_init = PoseInitializer(self.config, self.item_name, self.cam)
        rows = []
        if return_only_best:
            best_rows = []

        with tqdm(total=(idmax - idmin + 1) * len(self.templates), desc="Experiments", leave=True) as pbar:
            for idx in range(idmin, idmax + 1):
                rgb, depth, t_gt = self.dl(idx)
                try:
                    pts, _ = predict_segment(rgb, self.item_name)
                except TargetNotFoundError as e:
                    self.logger.warning(f"[idx {idx}] {e}")
                    pbar.update(len(self.templates))
                    continue
                except SegmentationError as e:
                    self.logger.error(f"[idx {idx}] Segmentation failed: {e}")
                    pbar.update(len(self.templates))
                    continue

                masked_depth = masked_image(pts, depth)
                scene = self._scene_from_depth(masked_depth)

                if return_only_best:
                    best_template = None
                    best_ev = None
                    best_transformation = None
                    best_duration = None

                self.logger.debug(f"========== idx: {idx} ==========")
                for template in self.templates:
                    t0 = time.time()
                    transformation, ev = pose_init.process_one(scene, template.pcd, t_gt)
                    duration = time.time() - t0
                    self.logger.debug(f"----- template: {template.name} -----")
                    self.logger.debug(f"Registration time: {duration:.2f}s")
                    self.logger.debug(ev)
                    self.logger.debug(f"Transformation: \n{transformation}")

                    rows.append(self._create_csv_row(idx, template.name, transformation, ev, duration))

                    if self.config.visualize:
                        full_scene = self._scene_from_depth(depth)
                        vis_scene = copy.deepcopy(scene)
                        vis_template = copy.deepcopy(template.pcd).transform(transformation)

                        full_scene.paint_uniform_color(BLACK)
                        vis_scene.paint_uniform_color(GREEN)
                        vis_template.paint_uniform_color(BLUE)

                        if self.out_dir is not None:
                            self._render_and_save([full_scene, vis_scene, vis_template], idx, template.name)

                    if return_only_best:
                        chosen = get_better_eval(ev, best_ev)
                        if chosen is ev:
                            best_template = template
                            best_transformation = transformation
                            best_ev = ev
                            best_duration = duration

                    pbar.update(1)

                if return_only_best:
                    if best_template is None:
                        # --- No successful template for this scene ---
                        self.logger.warning(
                            f"[idx {idx}] No partial template produced a better evaluation; "
                            "writing failure row with NaNs."
                        )
                        # Identity transform as placeholder
                        T_identity = np.eye(4)
                        # Build a failure row with NaN metrics so downstream can count failures
                        fail_row = {
                            "dataset_id": idx,
                            "template": "__no_success__",
                            "fitness": np.nan,
                            "inlier_rmse": np.nan,
                            "translation_error": np.nan,
                            "rotation_error": np.nan,
                            "correspondence_set_size": 0,
                            "transformation": T_identity.tolist(),
                            "duration": np.nan,
                        }
                        # Ensure CSV for partial-best exists as well:
                        best_rows.append(fail_row)
                    else:
                        best_rows.append(
                            self._create_csv_row(
                                idx, best_template.name, best_transformation, best_ev, best_duration
                            )
                        )
        if self.out_dir is not None:
            self._save_as_csv(rows)

        if return_only_best:
            return best_rows
        else:
            return rows

    def _create_csv_row(self, idx, template_name, transformation, evaluation, duration):
        return {
            "dataset_id": idx,
            "template": template_name,
            "fitness": evaluation.fitness,
            "inlier_rmse": evaluation.inlier_rmse,
            "translation_error": evaluation.translation_error,
            "rotation_error": evaluation.rotation_error,
            "correspondence_set_size": np.asarray(evaluation.correspondence_set).shape[0],
            "transformation": transformation.tolist(),
            "duration": duration,
        }

    def _scene_from_depth(self, depth_img):
        fx, fy, cx, cy = self.cam.get_camera_intrinsics()
        intrinsic = o3dPinholeCameraIntrinsic(
            width=self.cam.width, height=self.cam.height,
            fx=fx, fy=fy, cx=cx, cy=cy
        )
        return o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_img),
            intrinsic=intrinsic, extrinsic=np.eye(4)
        )

    def _save_as_csv(self, rows):
        os.makedirs(self.out_dir, exist_ok=True)
        df = pd.DataFrame(rows)

        csv_path = os.path.join(self.out_dir, "results.csv")
        df.to_csv(csv_path, index=False)

        self.logger.info(f"Results saved in: {csv_path}")
        return csv_path

    def _render_and_save(self, geometries, idx, t_name):
        out_path = os.path.join(self.out_dir, f"visualization/{idx}_{t_name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Minimal offscreen rendering via Open3D O3DVisualizer or Filament backend
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        for g in geometries: vis.add_geometry(g)
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1]); ctr.set_up([0, -1, 0]); ctr.set_lookat([0, 0, 0]); ctr.set_zoom(0.7)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(out_path)
        vis.destroy_window()
