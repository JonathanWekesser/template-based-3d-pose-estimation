# paths.py
import glob
import os

from dotenv import load_dotenv


class PathManager():

    def __init__(self, item):
        load_dotenv()
        self.item = item
        self.base_path = os.getenv("BASE_PATH")
        self.item_path = self.base_path + f"{item}/"
        self.model_path = self.item_path + f"{item}.stl"

        self.data_path = os.getenv("DATA_PATH")

    def get_base_path(self):
        return self.base_path
    
    def get_item_path(self):
        return self.item_path
    
    def get_model_path(self):
        return self.model_path

    def get_rgb_image_path(self, id):
        return self.item_path + f"rgbddata/{self.item}{id}_color.npy"

    def get_depth_image_path(self, id):
        return self.item_path + f"rgbddata/{self.item}{id}_depth.npy"

    def get_orientation_path(self, id):
        return self.item_path + f"rgbddata/{self.item}{id}.txt"

    def get_data_path(self):
        return self.data_path

    def get_camera_path(self):
        return self.data_path + "camera_intrinsics.yaml"

    def get_model_pcd_path(self):
        return self.data_path + f"{self.item}/{self.item}.ply"

    def get_template_paths(self):
        template_dir = self.data_path + f"{self.item}/"
        file_names = f"{self.item}_template_*.ply"
        return sorted(glob.glob(os.path.join(template_dir, file_names)))
