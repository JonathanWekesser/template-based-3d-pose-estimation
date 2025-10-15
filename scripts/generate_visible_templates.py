#!/usr/bin/env python3
"""
generate_visible_templates.py

Create visibility-based point cloud templates from viewpoints around an STL mesh,
using Open3D's RaycastingScene (headless, no OpenGL context required).

Example:
python generate_visible_templates.py \
  --mesh data/banana/banana.stl \
  --outdir data/template_experiment/banana/lookat \
  --num-templates 12 \
  --num-points 10000 \
  --radius 0.5 \
  --elev-list -30,30 \
  --width 640 --height 480 \
  --fov 60 \
  --seed 42
"""

import argparse
import math
import os
import sys
import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic, PinholeCameraParameters
from pathlib import Path
from typing import Tuple, List
import time

# ---------- Math & camera helpers ----------

def fov_to_intrinsics(width: int, height: int, fov_deg: float) -> Tuple[float, float, float, float]:
    """Compute pinhole intrinsics from horizontal FOV in degrees.
    Assumes square pixels and principal point at image center."""
    fov_rad = math.radians(fov_deg)
    fx = (0.5 * width) / math.tan(0.5 * fov_rad)
    fy = fx  # square pixels
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy

def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Return camera-to-world transform (4x4) where +Z is forward, +X right, +Y up in camera frame."""
    forward = center - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)
    R = np.stack([right, true_up, forward], axis=1)  # columns are basis vectors
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T  # camera-to-world

def spherical_view(center: np.ndarray, radius: float, az_deg: float, el_deg: float) -> np.ndarray:
    """Place the camera on a sphere around 'center' using azimuth (deg) and elevation (deg)."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = center[0] + radius * math.cos(el) * math.cos(az)
    y = center[1] + radius * math.cos(el) * math.sin(az)
    z = center[2] + radius * math.sin(el)
    return np.array([x, y, z], dtype=np.float32)

def make_rays(width: int, height: int, fx: float, fy: float, cx: float, cy: float, T_cam_world: np.ndarray) -> o3d.core.Tensor:
    """Create a (W*H, 6) tensor of rays [ox, oy, oz, dx, dy, dz] in world coordinates.
    Uses +Z forward pinhole. Pixel y is inverted to make +Y camera-up."""
    # pixel grid
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # shape HxW

    # directions in camera coords (+Z forward)
    x_cam = (uu - cx) / fx
    y_cam = -(vv - cy) / fy  # invert so +Y is up
    z_cam = np.ones_like(x_cam, dtype=np.float32)

    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # HxWx3
    # normalize
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-12
    dirs_cam /= norms

    # transform to world
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    H, W = height, width
    dirs_world = dirs_cam.reshape(-1, 3) @ R.T  # (H*W,3)
    origins_world = np.repeat(t[None, :], H * W, axis=0)

    rays_np = np.concatenate([origins_world, dirs_world], axis=1).astype(np.float32)  # (N,6)
    return o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)

def cast_visible_points(scene: o3d.t.geometry.RaycastingScene, rays: o3d.core.Tensor) -> np.ndarray:
    """Cast rays and return Nx3 numpy array of hit points (first intersections only)."""
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()  # (N,)
    # Valid hits have finite t
    valid = np.isfinite(t_hit)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    rays_np = rays.numpy()
    o = rays_np[:, :3]
    d = rays_np[:, 3:]
    pts = o[valid] + t_hit[valid, None] * d[valid]
    return pts.astype(np.float32)

def downsample_random(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly choose up to k points from the array. If fewer exist, return all."""
    n = points.shape[0]
    if n <= k:
        return points
    idx = rng.choice(n, size=k, replace=False)
    return points[idx]


# ---------------- Visualization helper ----------------
def _visualize_view(
    mesh_legacy: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
    width: int, height: int,
    fx: float, fy: float, cx: float, cy: float,
    T_cw: np.ndarray,
    duration: float = 0.0,
):
    """Show mesh + point cloud from the same camera pose.
    - duration==0.0 -> block until the window is closed by user
    - duration>0.0  -> show for given seconds, then auto-close
    """
    # Build camera parameters (Open3D expects world->camera extrinsic)
    extrinsic = np.linalg.inv(T_cw)  # world-to-camera
    intr = PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    params = PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = extrinsic

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Template Preview", width=width, height=height, visible=True)
    mesh_draw = mesh_legacy.compute_vertex_normals()
    vis.add_geometry(mesh_draw)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)  # Open3D >= 0.15
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)  # fallback for older versions

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size = 2.0

    if duration <= 0.0:
        vis.run()  # blocking window
        vis.destroy_window()
        return

    t_end = time.time() + duration
    while time.time() < t_end:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)
    vis.destroy_window()


# ---------- Main generation ----------

def generate_templates(
    mesh_path: str,
    outdir: str,
    num_templates: int,
    num_points: int,
    radius: float,
    elev_list: List[float],
    width: int,
    height: int,
    fov_deg: float,
    seed: int = 0,
    prefix: str = None,
    visualize: bool = False,
    viz_sec: float = 0.0,
):
    """Generate 'num_templates' visibility-only point cloud templates around the mesh."""
    rng = np.random.default_rng(seed)

    # Load mesh (legacy) and compute center
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_legacy is None or len(mesh_legacy.triangles) == 0:
        raise ValueError(f"Failed to load mesh or empty: {mesh_path}")
    mesh_legacy.compute_vertex_normals()
    center = np.asarray(mesh_legacy.get_axis_aligned_bounding_box().get_center(), dtype=np.float32)

    # Setup raycasting scene (tensor version)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    _ = scene.add_triangles(mesh_tensor)  # geometry_id (unused here)

    # Camera intrinsics
    fx, fy, cx, cy = fov_to_intrinsics(width, height, fov_deg)

    # Distribute azimuths around 360°, cycling through elevation list
    Path(outdir).mkdir(parents=True, exist_ok=True)
    base = Path(mesh_path).stem if prefix is None else prefix

    # Precompute how many azimuths we need per elevation to reach num_templates
    per_el = math.ceil(num_templates / max(1, len(elev_list)))
    az_list = np.linspace(0.0, 360.0, per_el, endpoint=False)

    saved = 0
    for i, el in enumerate(elev_list):
        for az in az_list:
            if saved >= num_templates:
                break

            # Camera placement and orientation
            eye = spherical_view(center, radius, az, el)
            T_cw = look_at(eye, center)

            # Build rays and cast
            rays = make_rays(width, height, fx, fy, cx, cy, T_cw)
            pts = cast_visible_points(scene, rays)

            if pts.shape[0] == 0:
                print(f"[WARN] View az={az:.1f}°, el={el:.1f}° produced 0 visible points. Skipping.")
                continue

            # Randomly downsample to desired num_points
            pts_ds = downsample_random(pts, num_points, rng)

            # Save as PLY
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_ds.astype(np.float64)))
            # optional: orient normals outward via KDTree or skip to keep it lightweight
            out_name = f"{base}_template_{saved:03d}_az{int(round(az))}_el{int(round(el))}.ply"
            out_path = os.path.join(outdir, out_name)
            o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=False)
            print(f"[OK] Saved template {saved+1}/{num_templates}: {out_path} ({pts_ds.shape[0]} pts)")


            # --- Optional visualization from the same camera pose ---
            if visualize:
                try:
                    _visualize_view(mesh_legacy, pcd, width, height, fx, fy, cx, cy, T_cw, duration=viz_sec)
                except Exception as e:
                    print(f"[WARN] Visualization failed (continuing): {e}")
            saved += 1

        if saved >= num_templates:
            break

    if saved < num_templates:
        print(f"[INFO] Only {saved}/{num_templates} templates created. "
              f"Consider increasing image resolution or adjusting radius/FOV/elevations.")

def parse_elev_list(s: str) -> List[float]:
    """Parse a comma-separated list of elevations in degrees (e.g., '-30,0,30')."""
    s = s.strip()
    if not s:
        return [0.0]
    return [float(x) for x in s.split(",")]

def main():
    ap = argparse.ArgumentParser(description="Generate visibility-only point cloud templates from a mesh.")
    ap.add_argument("--mesh", required=True, help="Path to STL/OBJ/PLY mesh (e.g., data/banana/banana.stl)")
    ap.add_argument("--outdir", required=True, help="Output directory for templates")
    ap.add_argument("--num-templates", type=int, required=True, help="Number of templates to generate")
    ap.add_argument("--num-points", type=int, default=10000, help="Target number of points per template")
    ap.add_argument("--radius", type=float, default=0.5, help="Camera radius around mesh center (mesh units)")
    ap.add_argument("--elev-list", type=parse_elev_list, default="0", help="Comma-separated elevations in deg (e.g., '-30,30')")
    ap.add_argument("--width", type=int, default=640, help="Ray grid width (pixels)")
    ap.add_argument("--height", type=int, default=480, help="Ray grid height (pixels)")
    ap.add_argument("--fov", type=float, default=60.0, help="Horizontal field of view in degrees")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for downsampling")
    ap.add_argument("--prefix", type=str, default=None, help="Filename prefix (default: mesh stem)")
    ap.add_argument("--visualize", action="store_true", help="Show an Open3D window for each generated template")
    ap.add_argument("--viz_sec", type=float, default=0.0, help="Seconds to keep the window open (0 = wait for manual close)")
    args = ap.parse_args()

    generate_templates(
        mesh_path=args.mesh,
        outdir=args.outdir,
        num_templates=args.num_templates,
        num_points=args.num_points,
        radius=args.radius,
        elev_list=args.elev_list if isinstance(args.elev_list, list) else parse_elev_list(args.elev_list),
        width=args.width,
        height=args.height,
        fov_deg=args.fov,
        seed=args.seed,
        prefix=args.prefix,
        visualize=args.visualize,
        viz_sec=args.viz_sec,
    )

if __name__ == "__main__":
    main()
