# registration.py
import copy

import open3d as o3d


class Registration:
    def __init__(self, config):
        self.cfg = config

    def ransac(self, src, tgt, src_fpfh, tgt_fpfh):
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            copy.deepcopy(src), copy.deepcopy(tgt),
            src_fpfh, tgt_fpfh,
            mutual_filter=False,
            max_correspondence_distance=self.cfg.ransac_corr_dist,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=self.cfg.ransac_n,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.cfg.ransac_edge_length),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.cfg.ransac_corr_dist),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.cfg.ransac_max_iter, self.cfg.ransac_confidence
            )
        )

    def icp(self, src, tgt, init_trans):
        method = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
            if self.cfg.icp_variant == "p2l"
            else o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        return o3d.pipelines.registration.registration_icp(
            copy.deepcopy(src), copy.deepcopy(tgt),
            max_correspondence_distance=self.cfg.icp_corr_dist,
            init=init_trans,
            estimation_method=method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.cfg.icp_max_iter
            ),
        )
