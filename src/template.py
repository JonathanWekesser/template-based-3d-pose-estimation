# template.py
import os
import open3d as o3d


class Template:
    def __init__(self, pcd, name):
        self.pcd = pcd
        self.name = name


class TemplateLoader:
    """
    Loads full model and template point clouds for a given item.
    """

    def __init__(self, path_manager):
        self.pm = path_manager

    def load_full_model(self):
        pcd = o3d.io.read_point_cloud(self.pm.get_model_pcd_path())
        name = "full_model"
        return Template(pcd, name)

    def load_templates(self, limit=None, step=None):
        paths = self.pm.get_template_paths()
        if step:
            paths = paths[::step]
        if limit:
            paths = paths[:limit]

        templates = []
        for path in paths:
            pcd = o3d.io.read_point_cloud(path)
            name = os.path.basename(path)
            templates.append(Template(pcd, name))
        return templates

    def load_all(self):
        templates = [self.load_full_model()]
        templates.extend(self.load_templates())
        return templates
