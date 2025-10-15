# dataset_loader.py
import numpy as np

from paths import PathManager


class DatasetLoader:
    """
    Helper class to load RGB, Depth and Ground Truth for a given dataset index.
    Wraps PathManager to provide a unified interface for experiments.
    """

    def __init__(self, item: str, path_manager: PathManager | None = None):
        self.item = item
        self.pm = path_manager or PathManager(item)

    def __call__(self, idx: int):
        """
        Loads one dataset entry.

        Parameters
        ----------
        idx : int
            Dataset index

        Returns
        -------
        rgb : np.ndarray
            RGB image
        depth : np.ndarray
            Depth image
        T_gt : np.ndarray
            Ground truth transformation matrix (4x4)
        """
        rgb = np.load(self.pm.get_rgb_image_path(idx))
        depth = np.load(self.pm.get_depth_image_path(idx))
        T_gt = np.genfromtxt(self.pm.get_orientation_path(idx))
        return rgb, depth, T_gt