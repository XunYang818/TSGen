"""Spatial distribution divergence (Spatial JSD)."""
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


def points_to_grid_distribution(
    points: List[Tuple[float, float]],
    grid_size: float = 3.0
) -> Dict[Tuple[int, int], float]:
    """
    Convert trajectory points to grid cell occupancy distribution.

    Args:
        points: List of trajectory points [(x, y), ...]
        grid_size: Grid cell size in decimeters, default 3.0 (30cm)

    Returns:
        Dictionary mapping grid cell -> normalized occupancy ratio
    """
    if not points:
        return {}

    cell_counts = defaultdict(int)
    for x, y in points:
        cell_x = int(x // grid_size)
        cell_y = int(y // grid_size)
        cell_counts[(cell_x, cell_y)] += 1

    total = sum(cell_counts.values())
    return {cell: count / total for cell, count in cell_counts.items()}


def kl_divergence(p: Dict, q: Dict, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: Distribution P
        q: Distribution Q
        epsilon: Smoothing term to avoid division by zero

    Returns:
        KL divergence value
    """
    all_keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for key in all_keys:
        p_val = p.get(key, epsilon)
        q_val = q.get(key, epsilon)
        if p_val > epsilon:
            kl += p_val * np.log(p_val / q_val)
    return kl


def js_divergence(p: Dict, q: Dict) -> float:
    """
    Compute Jensen-Shannon divergence.
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p: Distribution P
        q: Distribution Q

    Returns:
        JS divergence value, range [0, ln(2)] ~ [0, 0.693]
    """
    if not p and not q:
        return 0.0
    if not p or not q:
        return np.log(2)

    all_keys = set(p.keys()) | set(q.keys())
    m = {}
    for key in all_keys:
        m[key] = 0.5 * (p.get(key, 0) + q.get(key, 0))

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_spatial_jsd(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    grid_size: float = 3.0
) -> float:
    """
    Compute spatial distribution divergence (Spatial JSD) between two trajectories.

    Args:
        points1: Points list from trajectory 1
        points2: Points list from trajectory 2
        grid_size: Grid cell size in decimeters, default 3.0

    Returns:
        JSD value, higher means greater difference in spatial utilization patterns
    """
    dist1 = points_to_grid_distribution(points1, grid_size)
    dist2 = points_to_grid_distribution(points2, grid_size)
    return js_divergence(dist1, dist2)
