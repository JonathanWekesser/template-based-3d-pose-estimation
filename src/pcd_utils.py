# src/pcd_utils.py
# Purpose: Pure geometry utilities (depth/rgbd -> point cloud) using camera intrinsics.

import numpy as np
import open3d as o3d

def make_intrinsic(cam) -> o3d.camera.PinholeCameraIntrinsic:
    """Build Open3D intrinsic from your Camera object."""
    fx, fy, cx, cy = cam.get_camera_intrinsics()
    return o3d.camera.PinholeCameraIntrinsic(
        width=cam.width, height=cam.height, fx=fx, fy=fy, cx=cx, cy=cy
    )

def depth_to_pcd(
    depth: np.ndarray,
    cam,
    *,
    depth_unit: str | None = None,
    depth_trunc: float = np.inf,
    stride: int = 1,
    valid_only: bool = True,
    extrinsic: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """
    Convert a depth image into a point cloud using camera intrinsics.

    Parameters
    ----------
    depth : np.ndarray
        Depth image. Can be float32 meters or uint16 millimeters.
    cam : Camera-like
        Must provide intrinsics (fx, fy, cx, cy) and width/height.
    depth_unit : {'m','mm',None}
        If None, unit is auto-inferred from dtype/value range.
    depth_trunc : float
        Truncate depths beyond this distance (in meters).
    stride : int
        Pixel sampling stride.
    valid_only : bool
        Project only valid depth.
    extrinsic : np.ndarray
        4x4 camera-to-world transform (defaults to identity).

    Returns
    -------
    o3d.geometry.PointCloud
    """
    if extrinsic is None:
        extrinsic = np.eye(4, dtype=np.float32)

    # --- Normalize to float32 meters ---
    d = depth
    if depth_unit is None:
        # Heuristic: uint16 -> likely millimeters; float -> likely meters
        if d.dtype == np.uint16:
            depth_unit = 'mm'
        else:
            depth_unit = 'm'
    if depth_unit == 'mm':
        d = d.astype(np.float32) / 1000.0
    else:
        d = d.astype(np.float32)

    # Replace NaNs/negatives with 0 (Open3D ignores zeros)
    d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)

    # --- Build Open3D images and intrinsics ---
    intrinsic = make_intrinsic(cam)
    depth_img = o3d.geometry.Image(d)

    # --- Create point cloud ---
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        depth_scale=1.0,          # already meters
        depth_trunc=float(depth_trunc) if np.isfinite(depth_trunc) else np.inf,
        stride=int(stride),
        project_valid_depth_only=bool(valid_only),
    )
    return pcd

def rgbd_to_pcd(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    cam,
    *,
    depth_unit: str | None = None,
    depth_trunc: float = np.inf,
    stride: int = 1,
    valid_only: bool = True,
    extrinsic: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Optional helper: build colored point cloud from aligned RGB + depth."""
    import cv2  # local import to keep dependency boundaries clear

    if extrinsic is None:
        extrinsic = np.eye(4, dtype=np.float32)

    # Prepare depth in meters
    if depth_unit is None and depth.dtype == np.uint16:
        depth_unit = 'mm'
    if depth_unit == 'mm':
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)
    depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0).astype(np.float32)

    # Open3D expects RGB order
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_m)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=1.0,
        depth_trunc=float(depth_trunc) if np.isfinite(depth_trunc) else np.inf,
        convert_rgb_to_intensity=False,
    )
    intrinsic = make_intrinsic(cam)
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic=intrinsic, extrinsic=extrinsic
    )
