# preprocessing.py
import open3d as o3d

class Preprocessor:
    def __init__(self, config):
        self.cfg = config

    def prepare(self, pcd: o3d.geometry.PointCloud):
        if self.cfg.stat_nb_neighbors > 0 and self.cfg.stat_std_ratio > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.cfg.stat_nb_neighbors,
                std_ratio=self.cfg.stat_std_ratio
            )
        pcd = pcd.voxel_down_sample(self.cfg.voxel_size)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.normal_radius,
                max_nn=self.cfg.normal_max_nn
            )
        )
        return pcd
