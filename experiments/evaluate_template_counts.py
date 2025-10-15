# evaluate_template_counts.py
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import open3d as o3d

sys.path.append(os.path.abspath("src"))

from runner import ExperimentRunner                # noqa
from paths import PathManager                      # noqa
from camera import Camera                          # noqa
from config import load_config_yaml                # noqa
from template import Template                      # noqa


def load_plys_from_dir(directory: str, pattern: str = "*.ply"):
    """Load all point clouds from a directory and return list of Template."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    templates = []
    for p in paths:
        pcd = o3d.io.read_point_cloud(p)
        name = os.path.basename(p)
        templates.append(Template(pcd, name))
    return templates


def best_per_scene(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the row with max 'fitness' per dataset_id."""
    if df.empty:
        return df
    idx = df.groupby("dataset_id")["fitness"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the impact of the number of templates (k) per directory")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--item", required=True, help="Object name (e.g. apple, banana)")
    parser.add_argument("--cam", default="D435i", help="Camera profile name from YAML")
    parser.add_argument("--idmin", type=int, required=True, help="First dataset index (inclusive)")
    parser.add_argument("--idmax", type=int, required=True, help="Last dataset index (inclusive)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--template_dirs", nargs="+", required=True, help="Directories containing template files")

    # summary/selection knobs
    parser.add_argument("--criterion", default="mean_best_fitness",
                        choices=["mean_best_fitness", "median_best_fitness", "mean_inlier_rmse", "runtime_s"],
                        help="Metric to select the 'best' folder")
    args = parser.parse_args()

    # Build shared resources once (config & camera objects)
    cfg = load_config_yaml(args.config)
    pm = PathManager(args.item)
    cam = Camera.from_yaml(pm.get_camera_path(), args.cam)

    # to collect per-directory summaries
    summary_rows = []

    for template_dir in args.template_dirs:
        dir_label = os.path.basename(os.path.normpath(template_dir))
        out_dir = os.path.join(args.outdir, dir_label)

        # Load templates from this directory
        templates = load_plys_from_dir(template_dir, "*.ply")
        if len(templates) == 0:
            print(f"[WARN] no templates found in: {template_dir}")
            continue

        # Run experiment for this directory (all templates)
        runner = ExperimentRunner(cfg, args.item, cam, templates, out_dir)
        rows = runner.run_range(args.idmin, args.idmax)
        df = pd.DataFrame(rows)

        # Aggregate a minimal but informative summary for this directory
        if df.empty:
            summary_rows.append({
                "dir": dir_label,
                "n_templates": len(templates),
                "n_rows": 0,
                "n_scenes": 0,
                "mean_best_fitness": np.nan,
                "median_best_fitness": np.nan,
                "mean_inlier_rmse": np.nan,
                "runtime_s": np.nan,
            })
            continue

        winners = best_per_scene(df)
        runtime_s = float(df["duration"].sum())

        summary_rows.append({
            "dir": dir_label,
            "n_templates": len(templates),
            "n_rows": int(df.shape[0]),
            "n_scenes": int(winners.shape[0]),
            "mean_best_fitness": float(winners["fitness"].mean()),
            "median_best_fitness": float(winners["fitness"].median()),
            "mean_inlier_rmse": float(winners["inlier_rmse"].mean()),
            "runtime_s": runtime_s,
        })

    # Write global summary and pick best folder
    if len(summary_rows) == 0:
        print("[WARN] no directories summarized.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)

    # selection rule: higher is better for fitness, lower is better for rmse/runtime
    ascending = True if args.criterion in ["mean_inlier_rmse", "runtime_s"] else False
    summary_df = summary_df.sort_values(by=args.criterion, ascending=ascending).reset_index(drop=True)

    summary_path = os.path.join(args.outdir, "summary_dirs.csv")
    summary_df.to_csv(summary_path, index=False)

    best_dir = summary_df.iloc[0]["dir"]
    best_val = summary_df.iloc[0][args.criterion]

    print("\n=== Summary over template directories ===")
    print(summary_df.to_string(index=False))
    print(f"\n=> Best folder by '{args.criterion}': {best_dir}  (value={best_val})")
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
