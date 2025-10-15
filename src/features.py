# features.py
import open3d as o3d

class FeatureExtractor:
    def __init__(self, config):
        self.cfg = config

    def compute_fpfh(self, pcd):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.fpfh_radius,
                max_nn=self.cfg.fpfh_max_nn
            )
        )
