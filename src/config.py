# config.py
from dataclasses import dataclass

import yaml


@dataclass
class RegistrationConfig:
    voxel_size: float = 0.005
    stat_nb_neighbors: int = 50
    stat_std_ratio: float = 0.7
    normal_radius: float = 0.1
    normal_max_nn: int = 30
    fpfh_radius: float = 0.025
    fpfh_max_nn: int = 100
    ransac_corr_dist: float = 0.008
    ransac_n: int = 4
    ransac_edge_length: float = 0.9
    ransac_max_iter: int = 100000
    ransac_confidence: float = 0.99
    icp_variant: str = "p2p"
    icp_corr_dist: float = 0.005
    icp_max_iter: int = 100
    visualize: bool = False

def load_config_yaml(path: str) -> RegistrationConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return RegistrationConfig(**raw)
