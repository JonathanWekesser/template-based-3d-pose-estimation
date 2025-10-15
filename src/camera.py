# camera.py
import yaml

class Camera:
    def __init__(self, fx, fy, cx, cy, width=640, height=480):
        """
        Initialize Camera object.

        Parameters
        ----------
        fx : float
            Focal length in x direction.
        fy : float
            Focal length in y direction.
        cx : float
            Principal point x coordinate.
        cy : float
            Principal point y coordinate.
        width : int, optional
            Image width. Defaults to 640.
        height : int, optional
            Image height. Defaults to 480.
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @classmethod
    def from_yaml(cls, path: str, camera_name: str):
        """
        Load camera parameters from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.
        camera_name : str
            Key of the camera in the YAML file.

        Returns
        -------
        Camera
            A Camera instance.
        """
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        cam_cfg = config[camera_name]
        intr = cam_cfg["intrinsics"]
        size = cam_cfg.get("image_size", {"width": 640, "height": 480})

        return cls(
            fx=intr["fx"],
            fy=intr["fy"],
            cx=intr["cx"],
            cy=intr["cy"],
            width=size["width"],
            height=size["height"]
        )

    def get_camera_intrinsics(self):
        """
        Returns the camera intrinsic parameters.
        
        Returns:
            tuple: A tuple containing the focal lengths (fx, fy) and the principal point (cx, cy).
        """
        return self.fx, self.fy, self.cx, self.cy

    def get_image_size(self):
        """
        Returns the width and height of the images taken by the camera.
        
        Returns:
            tuple: A tuple containing the width and height of the image.
        """
        return self.width, self.height
