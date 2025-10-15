# registration_cli.py
import argparse
import os
import sys

import yaml

sys.path.append(os.path.abspath("src"))
from config import RegistrationConfig, load_config_yaml
from paths import PathManager
from camera import Camera
from dataset_loader import DatasetLoader
from template import TemplateLoader
from runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Run PoseInitializer with a given config and dataset range.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--item", required=True, help="Object name (e.g. apple, banana)")
    parser.add_argument("--cam", default="D435i", help="Camera profile name from YAML")
    parser.add_argument("--idmin", type=int, required=True, help="First dataset index (inclusive)")
    parser.add_argument("--idmax", type=int, required=True, help="Last dataset index (inclusive)")
    parser.add_argument("--outdir", default="./results/cli_run", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load config
    cfg = load_config_yaml(args.config)

    # Setup paths, camera, dataset
    pm = PathManager(args.item)
    cam = Camera.from_yaml(pm.get_camera_path(), args.cam)

    # Model + Templates
    loader = TemplateLoader(pm)
    templates = loader.load_all()

    # Run
    runner = ExperimentRunner(cfg, args.item, cam, templates, out_dir=args.outdir)
    runner.run_range(args.idmin, args.idmax)

if __name__ == "__main__":
    main()
