# evaluate_configs.py
"""
Evaluate (at least) two YAML configs on the same dataset range and compare their performance.
Tailored to the current ExperimentRunner signature:
    ExperimentRunner(config, item_name, cam, templates, out_dir=None, logger=None)

For each config:
  1) Run the pipeline on [idmin, idmax]
  2) Keep only the best template result per scene (max fitness, then min inlier_rmse)
  3) Save per-config CSVs:
       - results_raw.csv
       - results_best_per_scene.csv
After all runs:
  4) Aggregate mean/median metrics per config
  5) Compute relative deltas vs. best config per metric
  6) Write a CSV summary: summary_compare_configs.csv

Design goals:
  - Zero optional deps (CSV only).
  - Resilient column normalization & scene-key detection (supports your runner's 'dataset_id').
  - Robust runner construction for your current API.
  - Robust template loading:
      * Prefer your TemplateLoader if available.
      * Otherwise, build lightweight wrappers with .pcd and .name from PathManager paths.

Comments are in English; CLI/help text in German.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make local src module imports resolvable
sys.path.append(os.path.abspath("src"))
from paths import PathManager
from camera import Camera
from config import RegistrationConfig
from runner import ExperimentRunner

# --- Optional imports that may or may not exist in your repo ---
try:
    # Prefer your project-specific TemplateLoader if present
    from template_loader import TemplateLoader  # type: ignore
except Exception:
    TemplateLoader = None  # type: ignore

try:
    import open3d as o3d
except Exception:
    o3d = None  # type: ignore


# ----------------------------- Utilities ----------------------------- #
def ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def read_config(path: str) -> RegistrationConfig:
    """Load RegistrationConfig from YAML (supports either .from_yaml() or **yaml)."""
    import yaml
    if hasattr(RegistrationConfig, "from_yaml"):
        return RegistrationConfig.from_yaml(path)  # type: ignore
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return RegistrationConfig(**raw)


def build_camera_from_repo(pm: PathManager, cam_name: str) -> Camera:
    """Prefer the repo's Camera.from_yaml API if available."""
    if hasattr(Camera, "from_yaml"):
        return Camera.from_yaml(pm.get_camera_path(), cam_name)  # type: ignore
    # Fallback constructor
    return Camera(cam_name)  # type: ignore


class SimpleTemplate:
    """Minimal wrapper to satisfy runner expectations: attributes .pcd and .name."""
    def __init__(self, pcd, name: str):
        self.pcd = pcd
        self.name = name


def load_templates(pm: PathManager) -> List[SimpleTemplate]:
    """
    Load templates as objects with .pcd and .name.
    Strategy:
      1) If a TemplateLoader with .load_all() exists and already returns such objects, use it.
      2) Else, read model + template paths via PathManager and wrap into SimpleTemplate.
    """
    # 1) Try project-specific TemplateLoader
    if TemplateLoader is not None:
        try:
            loader = TemplateLoader(pm)
            # Many variants exist; try to be flexible:
            if hasattr(loader, "load_all"):
                t = loader.load_all()
                # If repo returns (templates, names) tuple, convert; else assume ready
                if isinstance(t, tuple) and len(t) == 2:
                    templates, names = t
                    if o3d is None:
                        raise RuntimeError("open3d is required for template wrapping.")
                    wrapped = []
                    # Case A: templates are file paths -> read
                    if templates and isinstance(templates[0], (str, Path)):
                        for path, name in zip(templates, names):
                            pcd = o3d.io.read_point_cloud(str(path))
                            wrapped.append(SimpleTemplate(pcd, name))
                        return wrapped
                    # Case B: templates are Open3D point clouds -> wrap
                    else:
                        for pcd, name in zip(templates, names):
                            wrapped.append(SimpleTemplate(pcd, name))
                        return wrapped
                else:
                    # Assume it's already a list of objects with .pcd and .name
                    return t  # type: ignore
        except Exception:
            # Fall back to manual loading if TemplateLoader path diverges
            pass

    # 2) Manual loading via PathManager
    if o3d is None:
        raise RuntimeError("open3d is required to read point clouds; please install it.")

    # Full model
    templates: List[SimpleTemplate] = []
    model_pcd_path = pm.get_model_pcd_path()
    if model_pcd_path and Path(model_pcd_path).exists():
        model_pcd = o3d.io.read_point_cloud(str(model_pcd_path))
        templates.append(SimpleTemplate(model_pcd, "full_model"))

    # Partial templates
    for p in pm.get_template_paths():
        name = Path(p).name
        pcd = o3d.io.read_point_cloud(str(p))
        templates.append(SimpleTemplate(pcd, name))

    if not templates:
        raise RuntimeError("No templates found. Check your PathManager configuration.")
    return templates


def detect_scene_key(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a column that uniquely identifies the scene/sample.
    Includes your runner's 'dataset_id'.
    """
    candidates = [
        "dataset_id", "scene_id", "idx", "id", "dataset_idx", "image_id", "frame_id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column aliases for robust downstream processing."""
    ren = {
        # time
        "duration": "runtime_s",
        "time_s": "runtime_s",
        # rmse
        "inlier_RMSE": "inlier_rmse",
        # rotation/translation
        "rot_error": "rotation_error",
        "r_error_deg": "rotation_error",
        "trans_error": "translation_error",
        "t_error_m": "translation_error",
        # fitness variants
        "best_fitness": "fitness",
        "mean_best_fitness": "fitness_mean",
        "median_best_fitness": "fitness_median",
        # correspondences
        "corr_size": "correspondence_set_size",
        "n_corr": "correspondence_set_size",
        "num_correspondences": "correspondence_set_size",
    }
    present = {k: v for k, v in ren.items() if k in df.columns}
    if present:
        df = df.rename(columns=present)
    return df


def select_best_per_scene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly one row per scene:
      - Prefer row with *maximum fitness* (higher is better)
      - If tie/absent, prefer row with *minimum inlier_rmse* (lower is better)
      - If neither exists, keep the first occurrence per scene
    """
    if df.empty:
        return df.copy()

    df = normalize_columns(df)
    key = detect_scene_key(df)
    if key is None:
        # No grouping possible; return as-is
        return df.copy()

    sort_cols: List[str] = []
    ascending: List[bool] = []

    if "fitness" in df.columns:
        sort_cols.append("fitness")
        ascending.append(False)  # higher is better
    if "inlier_rmse" in df.columns:
        sort_cols.append("inlier_rmse")
        ascending.append(True)   # lower is better

    if sort_cols:
        df_sorted = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
        return df_sorted.groupby(key, as_index=False).head(1).reset_index(drop=True)

    return df.groupby(key, as_index=False).head(1).reset_index(drop=True)


def aggregate_config(df_best: pd.DataFrame) -> pd.Series:
    """
    Aggregate metrics over scenes for one config.
    Returns a flat 1-D Series (mean & median for each metric present).
    """
    spec = {
        "fitness": ["mean", "median"],
        "inlier_rmse": ["mean", "median"],
        "translation_error": ["mean", "median"],
        "rotation_error": ["mean", "median"],
        "runtime_s": ["mean", "median"],
        "correspondence_set_size": ["mean", "median"],
    }

    out: Dict[str, float] = {}
    for col, aggs in spec.items():
        if col not in df_best.columns:
            continue
        s = df_best[col]
        for a in aggs:
            if a == "mean":
                out[f"{col}__mean"] = float(s.mean())
            elif a == "median":
                out[f"{col}__median"] = float(s.median())

    return pd.Series(out, dtype=float)


def add_relative_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage deltas vs. the best config per metric:
      - fitness*: higher is better
      - others:   lower is better
    """
    if summary.empty:
        return summary

    out = summary.copy()
    for col in out.columns:
        if col == "__config":
            continue
        if not np.issubdtype(out[col].dtype, np.number):
            continue

        higher_is_better = col.startswith("fitness__")
        if higher_is_better:
            best = out[col].max()
            out[col + "__Δ%"] = 100.0 * (out[col] - best) / (best if best else np.nan)
        else:
            best = out[col].min()
            out[col + "__Δ%"] = 100.0 * (out[col] - best) / (best if best else np.nan)
    return out


# ----------------------------- CLI & Main ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vergleicht optimierte YAML-Konfigurationen gegen eine Baseline auf demselben Datensatzbereich; pro Szene zählt nur das beste Template-Ergebnis."
    )
    # Dataset/core
    p.add_argument("--item", required=True, help="Objektname (z. B. apple, banana)")
    p.add_argument("--cam", default="D435i", help="Kameraprofilname (z. B. D435i)")
    p.add_argument("--idmin", type=int, required=True, help="Erster Datensatzindex (inklusive)")
    p.add_argument("--idmax", type=int, required=True, help="Letzter Datensatzindex (inklusive)")

    # Two-config convenience:
    p.add_argument("--baseline", type=str, help="Baseline YAML-Config")
    p.add_argument("--optimized", type=str, help="Optimierte YAML-Config")

    # Alternatively: arbitrary many configs
    p.add_argument("--configs", nargs="+", help="Liste von YAML-Config-Dateien")
    p.add_argument("--names", nargs="*", default=None, help="Optionale Anzeigenamen (gleiche Länge wie --configs)")

    # Output
    p.add_argument("--outdir", required=True, help="Zielordner für Ergebnisse & Zusammenfassung (CSV)")

    return p.parse_args()


def resolve_configs(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    """Resolve config paths and display names from CLI flags."""
    if args.configs:
        cfgs = args.configs
        if args.names:
            if len(args.names) != len(cfgs):
                raise SystemExit("Fehler: --names muss dieselbe Anzahl wie --configs haben.")
            names = args.names
        else:
            names = [Path(c).stem for c in cfgs]
        return cfgs, names

    # Fallback: two-config mode
    if not (args.baseline and args.optimized):
        raise SystemExit("Bitte entweder --configs ... ODER --baseline und --optimized angeben.")
    cfgs = [args.baseline, args.optimized]
    names = ["baseline", "optimized"]
    return cfgs, names


def main():
    args = parse_args()

    out_root = Path(args.outdir).resolve()
    ensure_dir(out_root)

    cfg_paths, cfg_names = resolve_configs(args)

    # Shared infra
    pm = PathManager(args.item)
    cam = build_camera_from_repo(pm, args.cam)
    templates = load_templates(pm)

    # Run each config
    summary_rows: List[pd.Series] = []

    for cfg_path, cfg_name in zip(cfg_paths, cfg_names):
        cfg = read_config(cfg_path)
        out_cfg = out_root / cfg_name
        ensure_dir(out_cfg)

        # Build runner with current signature
        runner = ExperimentRunner(config=cfg, item_name=args.item, cam=cam, templates=templates, out_dir=str(out_cfg))

        # Run identical dataset range
        rows = runner.run_range(args.idmin, args.idmax)
        df_raw = pd.DataFrame(rows)
        df_raw["__config"] = cfg_name

        # Persist raw results (CSV)
        df_raw.to_csv(out_cfg / "results_raw.csv", index=False)

        # Keep only the best template per scene
        df_best = select_best_per_scene(df_raw)
        df_best["__config"] = cfg_name
        df_best.to_csv(out_cfg / "results_best_per_scene.csv", index=False)

        # Aggregate (flat 1-D)
        s = aggregate_config(df_best)
        s["__config"] = cfg_name
        summary_rows.append(s)

    # Assemble summary table
    if not summary_rows:
        print("Keine Daten aggregiert – bitte Pfade/Configs prüfen.", file=sys.stderr)
        sys.exit(1)

    summary = pd.concat(summary_rows, axis=1).T.reset_index(drop=True)

    # Reorder with __config first
    if "__config" in summary.columns:
        cols = ["__config"] + [c for c in summary.columns if c != "__config"]
        summary = summary[cols].copy()

    # Relative deltas
    summary = add_relative_deltas(summary)

    # Save summary (CSV only)
    summary.to_csv(out_root / "summary_compare_configs.csv", index=False)

    # Console printout
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== Vergleich: Baseline vs. Optimiert (best-per-scene) ===")
        print(summary.fillna("–").to_string(index=False))


if __name__ == "__main__":
    main()
