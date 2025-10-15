# evaluate_sampling_strategies.py
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

sys.path.append(os.path.abspath("src"))
from camera import Camera
from config import RegistrationConfig, load_config_yaml
from dataset_loader import DatasetLoader
from paths import PathManager
from runner import ExperimentRunner
from template import Template, TemplateLoader

# ---------------------------
# Logger
# ---------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateSamplingStrategiesExperiment")


# ---------------------------
# Sampling & spacing
# ---------------------------
def nearest_neighbor_stats(points: np.ndarray, subsample: int = 5000, seed: int = 0):
    """Compute k=2 nearest-neighbor distance stats on a subsample."""
    n = points.shape[0]
    if n < 2:
        return dict(mean_nn=np.nan, std_nn=np.nan, min_nn=np.nan, max_nn=np.nan, n_eval=0)
    rs = np.random.RandomState(seed)
    idx = np.arange(n)
    if subsample and subsample < n:
        idx = rs.choice(idx, size=subsample, replace=False)

    pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdt = o3d.geometry.KDTreeFlann(pcd_all)
    dists = []
    for i in idx:
        _, _, d2 = kdt.search_knn_vector_3d(points[i], 2)  # self + nearest neighbor
        if len(d2) >= 2:
            dists.append(np.sqrt(d2[1]))
    if not dists:
        return dict(mean_nn=np.nan, std_nn=np.nan, min_nn=np.nan, max_nn=np.nan, n_eval=0)
    d = np.asarray(dists, float)
    return dict(
        mean_nn=float(d.mean()),
        std_nn=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        min_nn=float(d.min()),
        max_nn=float(d.max()),
        n_eval=int(len(d)),
    )

def theoretical_spacing(surface_area: float, n_points: int) -> float:
    """Heuristic spacing for near-uniform surface samples: ~ sqrt(area / N)."""
    return float(np.sqrt(surface_area / n_points)) if (surface_area > 0 and n_points > 0) else np.nan

def sample_model(mesh: o3d.geometry.TriangleMesh, n_points: int, method: str) -> o3d.geometry.PointCloud:
    """Sample mesh as point cloud via Poisson or Uniform; estimate normals if missing."""
    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=int(n_points), init_factor=5.0)
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    return pcd

def write_template_summary(
    rows,
    outdir,
    item,
    metrics=None,
    filename=None,
    rounding=4,
    template_col=None,
):
    """
    Writes summary statistics to a CSV file.
    """
    if not rows:
        logger.warning("No rows -> no summary written.")
        return None, pd.DataFrame()

    df = pd.DataFrame(rows)

    # Template-Spalte bestimmen
    if template_col is not None:
        if template_col not in df.columns:
            raise KeyError(f"'{template_col}' column is missing. Available cols: {list(df.columns)}")
    else:
        candidates = ["template_name", "template", "t_name", "name", "model_name"]
        template_col = next((c for c in candidates if c in df.columns), None)
        if template_col is None:
            raise KeyError(
                "No template column found. Expected one of "
                f"{candidates}, but got: {list(df.columns)}"
            )

    # Default-Metriken (nur die, die existieren)
    if metrics is None:
        candidate_metrics = [
            "fitness",
            "inlier_rmse",
            "translation_error",
            "rotation_error",
            "duration",
            "correspondence_set_size",
        ]
        metrics = [c for c in candidate_metrics if c in df.columns]

    # Numerik erzwingen/filtern
    numeric_metrics = []
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        numeric_metrics.append(c)
    if not numeric_metrics:
        # Falls gar keine Metrikspalten existieren, nur n pro Template ausgeben
        summary = df.groupby(template_col, dropna=False).size().rename("n").reset_index()
    else:
        # Mittelwerte je Template + Anzahl
        summary = (
            df.groupby(template_col, dropna=False)[numeric_metrics]
              .mean(numeric_only=True)
              .reset_index()
        )
        summary["n"] = (
            df.groupby(template_col, dropna=False).size().astype(int).values
        )

        # Runden nur numerische Spalten
        num_cols = summary.select_dtypes(include="number").columns
        summary[num_cols] = summary[num_cols].round(rounding)

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, filename or f"{item}_summary.csv")
    summary.to_csv(csv_path, index=False)
    return csv_path, summary


# ---------------------------
# Plots
# ---------------------------
def detect_template_col(df: pd.DataFrame) -> str:
    """Find the column that carries the template name."""
    candidates = ["template_name", "template", "t_name", "name", "model_name"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No template column found. Expected one of {candidates}, got: {list(df.columns)}")

def parse_method_and_points(name: str) -> tuple[str, int | None]:
    """
    Parse method and n_points from a template file name, e.g.:
    '<item>_full_model_poisson_10000.pcd' -> ('poisson', 10000)
    Falls das Muster nicht passt, wird ('unknown', None) zurückgegeben.
    """
    # Accept both with and without extension; be lenient with underscores
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]
    # Try to capture the last two underscore-separated tokens: <method>_<n>
    m = re.search(r"(poisson|uniform)[_\-](\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    # Fallback: look for *_poisson_* or *_uniform_* and any trailing int
    if "poisson" in base:
        nums = re.findall(r"(\d+)", base)
        return "poisson", int(nums[-1]) if nums else None
    if "uniform" in base:
        nums = re.findall(r"(\d+)", base)
        return "uniform", int(nums[-1]) if nums else None
    return "unknown", None

def ensure_method_points(df: pd.DataFrame, template_col: str) -> pd.DataFrame:
    """Add 'method' and 'n_points' columns parsed from the template column."""
    if "method" in df.columns and "n_points" in df.columns:
        return df
    methods, ns = [], []
    for v in df[template_col].astype(str).tolist():
        m, n = parse_method_and_points(v)
        methods.append(m)
        ns.append(n)
    out = df.copy()
    out["method"] = methods
    out["n_points"] = ns
    return out

def save_lineplot(df, x, y, hue, title, ylabel, outpath):
    """Generic line plot using matplotlib (no seaborn)."""
    plt.figure(figsize=(8, 5))
    # One line per hue category
    for key, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        plt.plot(sub[x], sub[y], marker="o", label=str(key))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def create_plots(outdir: str, item: str, summary_df: pd.DataFrame, rows_df: pd.DataFrame):
    """
    Create a set of plots:
      - metric vs n_points (one line per method) for mean metrics
      - success_rate vs n_points (computed from per-row results)
    Also writes a method/points-aggregated CSV for convenience.
    """
    os.makedirs(outdir, exist_ok=True)

    # Detect the template column in both frames (can be different shapes)
    tcol_sum = detect_template_col(summary_df)
    tcol_rows = detect_template_col(rows_df)

    # Add 'method' and 'n_points'
    sum_enriched = ensure_method_points(summary_df, tcol_sum)
    rows_enriched = ensure_method_points(rows_df, tcol_rows)

    # Build an aggregated table by (method, n_points)
    # For summary_df, its means are already per-template. We average across identical (method, n_points)
    # in case mehrere Templates pro (method, n_points) vorhanden sind (z.B. future variants).
    metrics = [c for c in ["fitness", "inlier_rmse", "translation_error", "rotation_error", "duration"]
               if c in sum_enriched.columns]

    by_mp = (
        sum_enriched
        .dropna(subset=["n_points"])
        .groupby(["method", "n_points"], dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )

    # Success rate: proportion of rows with correspondence_set_size > 0
    success_series = pd.to_numeric(rows_enriched.get("correspondence_set_size", pd.Series(dtype=float)), errors="coerce")
    rows_enriched = rows_enriched.assign(success=(success_series > 0).astype(float))
    sr_by_mp = (
        rows_enriched
        .dropna(subset=["n_points"])
        .groupby(["method", "n_points"], dropna=False)["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )

    # Merge success rate into the metric table
    full_aggr = by_mp.merge(sr_by_mp, on=["method", "n_points"], how="outer")
    # Persist the aggregation for inspection
    full_aggr_path = os.path.join(outdir, f"{item}_summary_by_method_points.csv")
    full_aggr.to_csv(full_aggr_path, index=False)
    logger.info(f"Wrote aggregated summary: {full_aggr_path}")

    # Produce line plots for each available metric
    for m in metrics:
        p = os.path.join(outdir, f"{item}_{m}_vs_points.png")
        save_lineplot(
            df=full_aggr.dropna(subset=["n_points", m]),
            x="n_points", y=m, hue="method",
            title=f"{item}: {m} vs #points",
            ylabel=m,
            outpath=p,
        )
        logger.info(f"Wrote plot: {p}")

    # Success rate plot
    if "success_rate" in full_aggr.columns:
        p = os.path.join(outdir, f"{item}_success_rate_vs_points.png")
        save_lineplot(
            df=full_aggr.dropna(subset=["n_points", "success_rate"]),
            x="n_points", y="success_rate", hue="method",
            title=f"{item}: success rate vs #points",
            ylabel="success_rate",
            outpath=p,
        )
        logger.info(f"Wrote plot: {p}")


# ---------------------------
# Main experiment
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Sweep model sizes & sampling; measure registration quality/runtime.")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--mesh", required=True, help="Path to input mesh (.stl/.ply/.obj)")
    ap.add_argument("--item", required=True, help="Object name for PoseInitializer")
    ap.add_argument("--cam", default="D435i", help="Camera profile")
    ap.add_argument("--idmin", type=int, required=True, help="First dataset index (inclusive)")
    ap.add_argument("--idmax", type=int, required=True, help="Last dataset index (inclusive)")
    ap.add_argument("--outdir", default="./results/model_sampling", help="Output directory")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 20000],
                    help="Target number of model points")
    ap.add_argument("--methods", choices=["poisson", "uniform", "both"], default="both",
                    help="Sampling method(s)")
    ap.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log level")
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logger.setLevel(args.log_level)

    # Load config
    cfg = load_config_yaml(args.config)

    # Setup paths, camera, dataset
    pm = PathManager(args.item)
    cam = Camera.from_yaml(pm.get_camera_path(), args.cam)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty() or not mesh.has_triangles():
        raise RuntimeError(f"Invalid mesh: {args.mesh}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    methods = ["poisson", "uniform"] if args.methods == "both" else [args.methods]

    templates = []

    for method in methods:
        for n in args.sizes:
            logger.info(f"Sampling {method} with {n} points")
            model_pcd = sample_model(mesh, n, method)
            name = f"{args.item}_full_model_{method}_{n}.pcd"
            templates.append(Template(model_pcd, name))

    runner = ExperimentRunner(cfg, args.item, cam, templates, args.outdir, logger=logger)
    rows = runner.run_range(args.idmin, args.idmax)

    csv_path, summary_df = write_template_summary(rows, args.outdir, args.item)
    logger.info(f"Wrote summary: {csv_path}")

    if args.plots:
        # Convert rows (list[dict]) to DataFrame for plotting
        rows_df = pd.DataFrame(rows)
        try:
            create_plots(args.outdir, args.item, summary_df, rows_df)
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
