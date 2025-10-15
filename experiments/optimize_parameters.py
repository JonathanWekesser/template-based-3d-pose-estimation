# optimize_parameters.py
"""
Hyperparameter optimization for PoseInitializer parameters using Optuna.
Results are stored under results/optimize_parameters_<item>/, with one subfolder per trial.
"""

import argparse
import logging
import math
import os
import random
import sys
import time

import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Make local src module imports resolvable
sys.path.append(os.path.abspath("src"))
from runner import ExperimentRunner
from camera import Camera
from config import RegistrationConfig
from paths import PathManager
from template import TemplateLoader
from dataset_loader import DatasetLoader

# ------------------------------ Logger ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateSamplingStrategiesExperiment")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Optuna optimization for PoseInitializer via ExperimentRunner.")
    p.add_argument("--item", default="apple")
    p.add_argument("--cam", default="D435i")
    p.add_argument("--idmin", type=int, default=0)
    p.add_argument("--idmax", type=int, default=49)
    p.add_argument("--outdir", default="./results/optimize_parameters")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--templates", type=str, default="both", choices=["full", "partial", "both"])
    return p.parse_args()

# ------------------------------
# Utilities
# ------------------------------
REQUIRED_COLS = [
    "fitness",
    "inlier_rmse",
    "translation_error",
    "rotation_error",
    "correspondence_set_size",  # align with runner.py output
]

def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")

def safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")

def composite_score(df: pd.DataFrame) -> float:
    """Combine fitness and error metrics into a single bounded score.
    Larger is better. Return a large negative value if metrics are invalid.
    """
    # Early exit if no data was produced
    if df is None or len(df) == 0:
        return -1e6

    med_fit   = safe_median(df["fitness"])
    mean_rmse = safe_mean(df["inlier_rmse"])
    mean_t    = safe_mean(df["translation_error"])
    mean_r    = safe_mean(df["rotation_error"])

    # Success rate = share of rows with at least one correspondence
    sr_series = pd.to_numeric(df["correspondence_set_size"], errors="coerce")
    success_rate = float((sr_series > 0).mean()) if len(sr_series) else float("nan")

    # If any component is NaN or success is 0, treat as failed trial
    if any(math.isnan(x) for x in [med_fit, mean_rmse, mean_t, mean_r, success_rate]) or success_rate == 0.0:
        return -1e6

    # Light penalty formulation to keep magnitude reasonable across units
    penalty = (mean_rmse) + (mean_t * 100.0 * 0.1) + (mean_r / 90.0 * 0.1)
    return float((med_fit * success_rate) / (1.0 + penalty))

# ------------------------------
# Optuna param suggestion
# ------------------------------
def suggest_params(trial: optuna.trial.Trial) -> RegistrationConfig:
    """Sample a RegistrationConfig from Optuna's search space."""
    params = {
        "voxel_size":        trial.suggest_float("voxel_size", 0.002, 0.02, log=True),
        "stat_nb_neighbors": trial.suggest_int("stat_nb_neighbors", 10, 100),
        "stat_std_ratio":    trial.suggest_float("stat_std_ratio", 0.1, 2.0, log=True),
        "normal_radius":     trial.suggest_float("normal_radius", 0.01, 0.06, log=True),
        "normal_max_nn":     trial.suggest_int("normal_max_nn", 20, 100),
        "fpfh_radius":       trial.suggest_float("fpfh_radius", 0.01, 0.1, log=True),
        "fpfh_max_nn":       trial.suggest_int("fpfh_max_nn", 30, 200),
        "ransac_corr_dist":  trial.suggest_float("ransac_corr_dist", 0.003, 0.05, log=True),
        "ransac_n":          trial.suggest_int("ransac_n", 3, 6),
        "ransac_edge_length": trial.suggest_float("ransac_edge_length", 0.7, 0.99),
        "ransac_max_iter":    trial.suggest_int("ransac_max_iter", 4000, 200000, log=True),
        "ransac_confidence":  trial.suggest_float("ransac_confidence", 0.90, 0.999),
        "icp_variant":        trial.suggest_categorical("icp_variant", ["p2p", "p2l"]),
        "icp_corr_dist":      trial.suggest_float("icp_corr_dist", 0.001, 0.05, log=True),
        "icp_max_iter":       trial.suggest_int("icp_max_iter", 20, 2000, log=True),
        "visualize":          False,  # visualization slows down HPO
    }
    return RegistrationConfig(**params)

# ------------------------------
# Trial runner
# ------------------------------
def run_trial(item, cam_name, idmin, idmax, cfg: RegistrationConfig, outdir, trial_number, templates_to_use) -> pd.DataFrame:
    """Run one trial end-to-end and return a DataFrame with results.

    Notes:
    - Creates a per-trial directory to avoid file overwrites from the runner.
    - Converts the runner's list[dict] rows into a DataFrame.
    """
    # Create per-trial folder, e.g., .../trial_0000/
    trial_dir = os.path.join(outdir, f"trial_{trial_number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Camera + templates
    pm = PathManager(item)
    cam = Camera.from_yaml(pm.get_camera_path(), cam_name)

    loader = TemplateLoader(pm)
    if templates_to_use == "full":
        templates = loader.load_full_model()
    elif templates_to_use == "partial":
        templates = loader.load_templates()
    else:
        templates = loader.load_all()

    # Run experiment
    runner = ExperimentRunner(cfg, item, cam, templates, trial_dir)
    rows = runner.run_range(idmin, idmax, return_only_best=True)

    df = pd.DataFrame(rows)

    # Ensure required columns exist (robustness)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = 0.0

    # Save (runner also wrote its own results.csv, but we save our view explicitly)
    df.to_csv(os.path.join(trial_dir, "results.csv"), index=False)
    return df

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()

    # Seed handling
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    # Result directory includes item name
    out_dir = os.path.abspath(f"{args.outdir}_{args.item}")
    os.makedirs(out_dir, exist_ok=True)

    # SQLite storage to keep studies persistent and resumable
    storage = f"sqlite:///{os.path.join(out_dir, 'optuna.sqlite3')}"

    # Create or load study
    study = optuna.create_study(
        study_name=f"pose_init_opt_{args.item}",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=args.seed, multivariate=True, group=True),
        pruner=MedianPruner(n_warmup_steps=3),  # note: no trial.report() → pruning is effectively off
    )

    # Objective wrapper
    def objective(trial: optuna.trial.Trial) -> float:
        cfg = suggest_params(trial)
        t0 = time.time()
        try:
            df = run_trial(args.item, args.cam, args.idmin, args.idmax, cfg, out_dir, trial.number, args.templates)
        except Exception as e:
            # Record the exception for later inspection and mark as failed
            trial.set_user_attr("exception", str(e))
            return -1e6
        finally:
            trial.set_user_attr("runtime_s", time.time() - t0)

        # Compute score + attach useful aggregates
        score = composite_score(df)
        trial.set_user_attr("median_fitness", safe_median(df["fitness"]))
        trial.set_user_attr("mean_rmse",     safe_mean(df["inlier_rmse"]))
        trial.set_user_attr("mean_t_err",    safe_mean(df["translation_error"]))
        trial.set_user_attr("mean_r_err",    safe_mean(df["rotation_error"]))
        sr = float((pd.to_numeric(df["correspondence_set_size"], errors="coerce") > 0).mean())
        trial.set_user_attr("success_rate",  sr)

        return score

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=True)

    # Best results to stdout
    print("Best configuration:", study.best_params)
    print(f"Best score: {study.best_value:.6f}")

    # Save best params to YAML
    with open(os.path.join(out_dir, f"optuna_best_{args.item}.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "best_params": study.best_params,
                "best_value": float(study.best_value),
                "item": args.item,
                "id_range": [args.idmin, args.idmax],
                "templates_used": args.templates,
                "seed": args.seed,
            },
            f,
            sort_keys=False,
        )

    # Save a flat trial summary CSV for quick filtering/plotting
    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "state": str(t.state),
            "value": float(t.value) if t.value is not None else None,
            "runtime_s": t.user_attrs.get("runtime_s"),
            "exception": t.user_attrs.get("exception"),
            "median_fitness": t.user_attrs.get("median_fitness"),
            "mean_rmse": t.user_attrs.get("mean_rmse"),
            "mean_t_err": t.user_attrs.get("mean_t_err"),
            "mean_r_err": t.user_attrs.get("mean_r_err"),
            "success_rate": t.user_attrs.get("success_rate"),
            **{f"param_{k}": v for k, v in t.params.items()},
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"summary_trials_{args.item}.csv"), index=False)

    # Optional static plots (uses matplotlib backend of Optuna; safe-guarded)
    if args.plots:
        try:
            from optuna.visualization import matplotlib as ovm
            import matplotlib.pyplot as plt

            fig1 = ovm.plot_optimization_history(study)
            fig1.figure.savefig(os.path.join(out_dir, f"opt_history_{args.item}.png"), bbox_inches="tight")
            plt.close(fig1.figure)

            # Param importances may fail if study has pruned/failed trials only
            fig2 = ovm.plot_param_importances(study)
            fig2.figure.savefig(os.path.join(out_dir, f"param_importances_{args.item}.png"), bbox_inches="tight")
            plt.close(fig2.figure)
        except Exception as e:
            # Do not crash on plotting issues; users can ignore or inspect later
            print(f"[WARN] Failed to create plots: {e}")

if __name__ == "__main__":
    main()
