import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic as o3dPinholeCameraIntrinsic

def scene_from_depth(cam, depth_img):
    fx, fy, cx, cy = cam.get_camera_intrinsics()
    intrinsic = o3dPinholeCameraIntrinsic(
        width=cam.width, height=cam.height,
        fx=fx, fy=fy, cx=cx, cy=cy
    )
    return o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_img),
        intrinsic=intrinsic, extrinsic=np.eye(4)
    )

def show_image(image, title, convert_to_bgr=False):
    if convert_to_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(image)
    plt.title(title)
    plt.show()

