# live-demo/visualization/live_visualizer.py
import open3d as o3d
import numpy as np

class LiveVisualizer:
    """Manages live 3D visualization of registration results."""

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Live Pose Estimation", width=1280, height=720)
        self.scene_geom = None
        self.template_geom = None
        self.full_geom = None

        self._init_view()

    def _init_view(self):
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.7)

    def update(self, scene, template, full_scene=None):
        # Clear old geometries
        self.vis.clear_geometries()
        # Paint colors
        if full_scene is not None:
            full_scene.paint_uniform_color([0, 0, 0])
            self.vis.add_geometry(full_scene)
        scene.paint_uniform_color([0, 1, 0])       # green
        template.paint_uniform_color([0, 0, 1])    # blue
        self.vis.add_geometry(scene)
        self.vis.add_geometry(template)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
