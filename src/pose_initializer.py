# pose_initializer.py
import numpy as np

from camera import Camera
from config import RegistrationConfig
from evaluation import Evaluation
from features import FeatureExtractor
from preprocessing import Preprocessor
from registration import Registration


def prealign(src, tgt):
    t = np.eye(4)
    t[:3, 3] = tgt.get_center() - src.get_center()
    return t

class PoseInitializer:
    def __init__(self,
                 config: RegistrationConfig,
                 item_name: str,
                 cam: Camera,
        ):
        self.cfg = config
        self.item_name = item_name
        self.cam = cam
        self.prep = Preprocessor(config)
        self.feat = FeatureExtractor(config)
        self.reg = Registration(config)

    def process_one(self, scene, model, t_gt: np.ndarray | None = None):
        # Scene
        scene = self.prep.prepare(scene)
        scene_fpfh = self.feat.compute_fpfh(scene)

        # Model
        model = self.prep.prepare(model)
        t_prealign = prealign(model, scene)
        model = model.transform(t_prealign)
        model_fpfh = self.feat.compute_fpfh(model)

        # Registration
        reg_ransac = self.reg.ransac(model, scene, model_fpfh, scene_fpfh)
        reg_icp = self.reg.icp(model, scene, reg_ransac.transformation)
        t_final = reg_icp.transformation @ t_prealign

        ev = Evaluation(model, scene, self.cfg.voxel_size, t_final, t_gt)

        return t_final, ev
