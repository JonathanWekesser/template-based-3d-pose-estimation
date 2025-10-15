#!/usr/bin/env python3
"""
final_compare_full_vs_partial_runner.py

Run your ExperimentRunner twice per item:
(1) FULL model (single template) and
(2) PARTIAL templates (ALL templates discovered via PathManager),
picking the best template per scene (return_only_best=True).

Results are written to separate output folders, and a combined CSV
is optionally created for convenience.

Example:
python final_compare_full_vs_partial_runner.py \
  --items apple banana orange knife bottle \
  --idmin 0 --idmax 199 \
  --config configs/universal.yaml \
  --outdir results/final_compare \
  --visualize 0
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import open3d as o3d
import pandas as pd

sys.path.append(os.path.abspath("src"))
from config import RegistrationConfig
from camera import Camera
from paths import PathManager
from template import Template, TemplateLoader
from runner import ExperimentRunner

# ------------------------------ CLI ------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare FULL model vs PARTIAL templates via ExperimentRunner.")
    p.add_argument("--items", nargs="+", required=True, help="Item names, e.g., apple banana orange knife bottle.")
    p.add_argument("--idmin", type=int, required=True, help="Minimum dataset id (inclusive).")
    p.add_argument("--idmax", type=int, required=True, help="Maximum dataset id (inclusive).")

    # Config loading
    p.add_argument("--config", required=True, help="Path to RegistrationConfig YAML (universelle Konfiguration).")
    p.add_argument("--visualize", type=int, default=None, help="Override config.visualize (0/1).")

    # Camera intrinsics (optional; if omitted, try a sensible default)
    p.add_argument("--width", type=int, default=640, help="Image width (px).")
    p.add_argument("--height", type=int, default=480, help="Image height (px).")
    p.add_argument("--fx", type=float, default=615.0, help="Camera fx (px).")
    p.add_argument("--fy", type=float, default=615.0, help="Camera fy (px).")
    p.add_argument("--cx", type=float, default=320.0, help="Camera cx (px).")
    p.add_argument("--cy", type=float, default=240.0, help="Camera cy (px).")

    # Output & runtime
    p.add_argument("--outdir", default="results/final_compare", help="Base output directory.")
    p.add_argument("--join-csv", type=int, default=1, help="Join both results into a combined CSV (0/1).")

    # Template thinning (optional fast debug)
    p.add_argument("--limit-templates", type=int, default=None, help="Use at most this many partial templates.")
    p.add_argument("--template-step", type=int, default=None, help="Use every k-th partial template.")

    return p.parse_args()


# ------------------------------ Utilities ------------------------------
def load_config_from_yaml(path: str, override_visualize: Optional[int]) -> RegistrationConfig:
    """Load RegistrationConfig from YAML; override visualize if requested."""
    cfg = RegistrationConfig.from_yaml(path) if hasattr(RegistrationConfig, "from_yaml") else None
    if cfg is None:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = RegistrationConfig(**data)  # type: ignore

    if override_visualize is not None:
        # Set visualize flag explicitly
        try:
            cfg.visualize = bool(int(override_visualize))
        except Exception:
            cfg.visualize = bool(override_visualize)

    return cfg


def build_camera(args: argparse.Namespace) -> Camera:
    """Construct a Camera object matching runner expectations."""
    # If your Camera has a factory (e.g., Camera.realsense_d435i()), use that instead.
    cam = Camera(
        width=args.width,
        height=args.height,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
    )
    return cam


def load_full_template(pm: PathManager) -> Template:
    """Load the full model point cloud as a single Template."""
    pcd_path = pm.get_model_pcd_path()
    pcd = o3d.io.read_point_cloud(pcd_path)
    return Template(name="full_model", pcd=pcd)


def load_partial_templates(pm: PathManager,
                           limit: Optional[int] = None,
                           step: Optional[int] = None) -> List[Template]:
    """Load all partial templates available via PathManager; optionally thin the set."""
    paths = pm.get_template_paths()
    if step:
        paths = paths[::step]
    if limit:
        paths = paths[:limit]

    templates: List[Template] = []
    for p in paths:
        pcd = o3d.io.read_point_cloud(p)
        templates.append(Template(name=os.path.basename(p), pcd=pcd))
    return templates


def save_manifest(base: Path,
                  item: str,
                  cfg: RegistrationConfig,
                  cam: Camera,
                  args: argparse.Namespace,
                  full_count: int,
                  partial_count: int) -> None:
    """Persist a small manifest with run parameters for reproducibility."""
    base.mkdir(parents=True, exist_ok=True)
    manifest = {
        "item": item,
        "idmin": args.idmin,
        "idmax": args.idmax,
        "config": args.config,
        "visualize": getattr(cfg, "visualize", None),
        "camera": {
            "width": cam.width, "height": cam.height,
            "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
        },
        "limit_templates": args.limit_templates,
        "template_step": args.template_step,
        "templates_full_count": full_count,
        "templates_partial_count": partial_count,
    }
    with open(base / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_combined_csv(out_full: Path, out_part: Path, out_combined: Path) -> None:
    """Join results.csv from both conditions into a single CSV with a 'condition' column."""
    f_full = out_full / "results.csv"
    f_part = out_part / "results.csv"
    if not f_full.exists() or not f_part.exists():
        return
    df_full = pd.read_csv(f_full)
    df_full["condition"] = "full"
    df_part = pd.read_csv(f_part)
    df_part["condition"] = "partial_best"
    df = pd.concat([df_full, df_part], ignore_index=True)
    df.to_csv(out_combined, index=False)


# ------------------------------ Main ------------------------------
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("final_compare")

    # Load common config & camera
    config = load_config_from_yaml(args.config, args.visualize)
    cam = build_camera(args)

    for item in args.items:
        logger.info(f"=== Item: {item} | ids [{args.idmin}, {args.idmax}] ===")

        pm = PathManager(item)
        loader = TemplateLoader(pm)
        full_template = loader.load_full_model()
        partial_templates = loader.load_templates()

        # Output structure
        base_out_item = Path(args.outdir) / item
        out_full = base_out_item / "full"
        out_part = base_out_item / "partial_all"
        out_full.mkdir(parents=True, exist_ok=True)
        out_part.mkdir(parents=True, exist_ok=True)

        # Save manifest
        save_manifest(base_out_item, item, config, cam, args, 1, len(partial_templates))

        # --- Run FULL (single template) ---
        logger.info(f"[FULL] templates: 1")
        runner_full = ExperimentRunner(
            config=config,
            item_name=item,
            cam=cam,
            templates=[full_template],
            out_dir=str(out_full),
            logger=logging.getLogger(f"runner:{item}:full")
        )
        rows_full = runner_full.run_range(args.idmin, args.idmax, return_only_best=False)
        # Persist (runner already saved CSV if out_dir is set)

        # --- Run PARTIAL (ALL templates, pick best per scene) ---
        logger.info(f"[PARTIAL] templates: {len(partial_templates)} (best-of per scene)")
        runner_part = ExperimentRunner(
            config=config,
            item_name=item,
            cam=cam,
            templates=partial_templates,
            out_dir=str(out_part),
            logger=logging.getLogger(f"runner:{item}:partial")
        )
        rows_part_best = runner_part.run_range(args.idmin, args.idmax, return_only_best=True)
        # Persist a best-only CSV for partials
        if rows_part_best:
            df_best = pd.DataFrame(rows_part_best)
            out_best_csv = out_part / "results.csv"
            df_best.to_csv(out_best_csv, index=False)
            logger.info(f"[PARTIAL] Best-only results saved in: {out_best_csv}")

        # --- Join CSVs (optional) ---
        if args.join_csv:
            out_combined = base_out_item / "results_combined.csv"
            write_combined_csv(out_full, out_part, out_combined)
            logger.info(f"[COMBINED] {out_combined}")

    logger.info("All items processed.")


if __name__ == "__main__":
    main()
