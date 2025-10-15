# evaluation.py
import numpy as np
import open3d as o3d


class Evaluation:
    def __init__(self, source, target, threshold, t_est=None, t_gt=None):
        """
        Constructor for Evaluation class.

        Parameters
        ----------
        source : open3d.geometry.PointCloud
            Source point cloud for registration
        target : open3d.geometry.PointCloud
            Target point cloud for registration
        threshold : float
            Distance threshold for ICP
        t_est : 4x4 numpy array
            Estimated transformation matrix
        t_gt : 4x4 numpy array
            Ground truth transformation matrix

        Attributes
        ----------
        ev : open3d.pipelines.registration.RegistrationResult
            Evaluation result
        fitness : float
            Fitness score of the registration
        inlier_rmse : float
            Root mean squared error of the inlier correspondences
        correspondence_set : open3d.utility.Vector2iVector
            Set of correspondences between source and target point clouds
        translation_error : float
            Euclidean distance between the estimated translation vector and ground truth translation vector
        rotation_error : float
            Angle between the estimated rotation matrix and ground truth rotation matrix in degrees
        """
        ev = o3d.pipelines.registration.evaluate_registration(source, target, threshold)
        self.eval = ev
        self.fitness = ev.fitness
        self.inlier_rmse = ev.inlier_rmse
        self.correspondence_set = ev.correspondence_set
        self.correspondence_set_size = len(self.correspondence_set)
        if t_est is not None and t_gt is not None:
            self.translation_error = translation_error(t_est, t_gt)
            self.rotation_error = rotation_error(t_est, t_gt)
        else:
            self.translation_error = None
            self.rotation_error = None

    def __str__(self):
        lines = [
            "Evaluation Results",
            f"  Fitness:                 {self.fitness:.3f}",
            f"  Inlier RMSE:             {self.inlier_rmse:.4f}",
            f"  Correspondence set size: {self.correspondence_set_size}",
        ]
        if self.translation_error is None or self.rotation_error is None:
            lines.append("  Translation error:       N/A (no ground truth)")
            lines.append("  Rotation error:          N/A (no ground truth)")
        else:
            lines.append(f"  Translation error:       {self.translation_error:.3f} m")
            lines.append(f"  Rotation error:          {self.rotation_error:.2f} °")

        return "\n".join(lines)


def translation_error(t_est, t_gt):
    """Euclidean distance between estimated and ground truth translation vectors."""
    t_est_vec = t_est[:3, 3]
    t_gt_vec = t_gt[:3, 3]
    return np.linalg.norm(t_est_vec - t_gt_vec)

def rotation_error(t_est, t_gt):
    """
    Computes the angular error between two rotation matrices (in degrees).
    """
    R_est = t_est[:3, :3]
    R_gt  = t_gt[:3, :3]

    # Compute the angle of rotation difference
    cos_theta = (np.trace(R_est.T @ R_gt) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def compare_evaluations(evals, sort_by="fitness", reverse=True, top_n=1, verbose=True):
    """
    Compares a list of Evaluation objects and prints out the top results.

    Parameters
    ----------
    evals : list of Evaluation
        List of Evaluation objects to compare
    sort_by : str, optional
        Attribute to sort the evaluations by. Must be one of 'fitness', 'rotation_error', 'translation_error', 'inlier_rmse'
    reverse : bool, optional
        Sort in descending order if True, otherwise in ascending order
    top_n : int, optional
        Number of top results to print out
    verbose : bool, optional
        Whether to print out the results

    Returns
    -------
    sorted_evals : list of Evaluation
        Top results sorted by the specified attribute
    """

    # Sanity check
    if sort_by not in ['fitness', 'rotation_error', 'translation_error', 'inlier_rmse']:
        raise ValueError(f"sort_by must be one of: 'fitness', 'rotation_error', 'translation_error', 'inlier_rmse'")

    sorted_evals = sorted(evals, key=lambda e: getattr(e, sort_by), reverse=reverse)

    if verbose:
        print(f"\n--- Top {top_n} results (sorted by '{sort_by}') ---")
        for i, ev in enumerate(sorted_evals[:top_n]):
            print(f"\nResult {i+1}:")
            print(ev)

    return sorted_evals[:top_n]

def get_better_eval(a, b):
    if a is None: return b
    if b is None: return a
    
    if a.fitness > b.fitness and a.inlier_rmse < b.inlier_rmse:
        return a
    elif a.fitness < b.fitness and a.inlier_rmse > b.inlier_rmse:
        return b
    
    if a.inlier_rmse < b.inlier_rmse and a.correspondence_set_size > b.correspondence_set_size:
        return a
    elif a.inlier_rmse > b.inlier_rmse and a.correspondence_set_size < b.correspondence_set_size:
        return b

    if a.correspondence_set_size > b.correspondence_set_size:
        return a
    elif a.correspondence_set_size < b.correspondence_set_size:
        return b

    return a
