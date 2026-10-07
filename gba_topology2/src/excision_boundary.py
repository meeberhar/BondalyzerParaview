"""Excision Boundary Subroutine for gba_topology2.

Profiles the scalar field f(phi) along closed 1D excision boundary loops on a triangulated
sphere, extracts boundary ports (valley and ridge ports), and independently verifies
the topological index formula:
    chi = 1 - k
where k is the number of valley ports (and ridge ports).

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Interpolates field values along 1D boundary points using inverse distance weighting.
- Resolves:
    k = 2 -> 4 ports (2 valleys, 2 ridges), chi = -1 (simple saddle)
    k = 3 -> 6 ports (3 valleys, 3 ridges), chi = -2 (monkey saddle)
    k = 4 -> 8 ports (4 valleys, 4 ridges), chi = -3 (4-fold octupolar cross)
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.micro_cluster import BoundaryRingPoint

__all__ = [
    "BoundaryPort",
    "ExcisionBoundaryResult",
    "extract_boundary_ports",
    "interpolate_scalar_at_points",
    "profile_boundary_loop",
]


@dataclass(frozen=True)
class BoundaryPort:
    """Represents a discrete port along an excision boundary loop."""

    port_id: str
    port_type: str  # "valley" (local 1D min on loop) or "ridge" (local 1D max on loop)
    phi_rad: float
    phi_deg: float
    position: tuple[float, float, float]
    scalar_value: float


@dataclass(frozen=True)
class ExcisionBoundaryResult:
    """Container for 1D excision boundary profiling and port analysis."""

    boundary_points: list[BoundaryRingPoint]
    profile_values: list[float]
    valley_ports: list[BoundaryPort]
    ridge_ports: list[BoundaryPort]
    fold_order: int  # k = number of valley ports
    implied_euler_index: int  # chi = 1 - k
    is_topologically_balanced: bool  # len(valley_ports) == len(ridge_ports)


def interpolate_scalar_at_points(
    query_pts: npt.NDArray[np.floating[Any]],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    k_neighbors: int = 4,
    power: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Interpolate scalar field at query points on sphere using Inverse Distance Weighting (IDW).

    Args:
        query_pts: (M, 3) coordinates of sample points.
        mesh_pts: (N, 3) coordinates of mesh vertices.
        f_vals: (N,) scalar values at mesh vertices.
        k_neighbors: Number of nearest neighbors to average (default 4).
        power: IDW power parameter (default 2.0).

    Returns:
        (M,) interpolated scalar values.
    """
    q = np.asarray(query_pts, dtype=np.float64)
    m = np.asarray(mesh_pts, dtype=np.float64)
    f = np.asarray(f_vals, dtype=np.float64)

    # Normalize vectors to unit sphere for robust chord distance
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm = np.where(q_norm < 1e-12, 1.0, q_norm)
    q_unit = q / q_norm

    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    m_norm = np.where(m_norm < 1e-12, 1.0, m_norm)
    m_unit = m / m_norm

    interp_vals = np.zeros(len(q), dtype=np.float64)

    # Batch compute dot products: (M, N)
    dots = np.dot(q_unit, m_unit.T)
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance: theta = arccos(dot)
    ang_dists = np.arccos(dots)

    for i in range(len(q)):
        row_dists = ang_dists[i]
        # Check for exact or near-exact match
        min_idx = int(np.argmin(row_dists))
        if row_dists[min_idx] < 1e-7:
            interp_vals[i] = f[min_idx]
            continue

        # Find k nearest neighbors
        k_idx = np.argpartition(row_dists, k_neighbors)[:k_neighbors]
        d_k = row_dists[k_idx]
        weights = 1.0 / (d_k**power)
        sum_w = float(np.sum(weights))
        interp_vals[i] = float(np.sum(weights * f[k_idx]) / sum_w)

    return interp_vals


def extract_boundary_ports(
    profile_values: list[float],
    ring_points: list[BoundaryRingPoint],
    smoothing_window: int = 3,
    min_prominence_frac: float = 0.05,
) -> tuple[list[BoundaryPort], list[BoundaryPort]]:
    """Detect alternating local minima (valleys) and maxima (ridges) on a closed periodic 1D profile.

    Args:
        profile_values: Sampled scalar values around the 1D closed loop.
        ring_points: Corresponding BoundaryRingPoint objects.
        smoothing_window: Moving average window for noise reduction (odd int, default 3).
        min_prominence_frac: Minimum peak/valley prominence fraction of profile span (default 0.05).

    Returns:
        Tuple of (valley_ports, ridge_ports).
    """
    n = len(profile_values)
    if n < 4:
        return [], []

    vals = np.array(profile_values, dtype=np.float64)
    v_span = float(np.ptp(vals))
    min_prominence = max(v_span * min_prominence_frac, 1e-12)

    # Periodic moving-average smoothing if requested
    if 1 < smoothing_window < n:
        half_w = smoothing_window // 2
        padded = np.concatenate([vals[-half_w:], vals, vals[:half_w]])
        kernel = np.ones(smoothing_window) / float(smoothing_window)
        smooth_vals = np.convolve(padded, kernel, mode="valid")
    else:
        smooth_vals = vals

    # Step 1: Find raw discrete local extrema on the 1D ring
    raw_min_indices: list[int] = []
    raw_max_indices: list[int] = []

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        curr_v = float(smooth_vals[i])
        prev_v = float(smooth_vals[prev_idx])
        next_v = float(smooth_vals[next_idx])

        if curr_v < prev_v and curr_v <= next_v:
            raw_min_indices.append(i)
        elif curr_v > prev_v and curr_v >= next_v:
            raw_max_indices.append(i)

    # Step 2: Filter by true topological prominence (peak-to-valley amplitude)
    # For each raw minimum, compute height difference relative to adjacent raw maxima
    valleys: list[BoundaryPort] = []
    for idx in raw_min_indices:
        v_curr = float(smooth_vals[idx])
        if raw_max_indices:
            max_vals = [float(smooth_vals[m]) for m in raw_max_indices]
            peak_height = max(max_vals) - v_curr
            if peak_height >= min_prominence or v_span < 1e-12:
                rp = ring_points[idx]
                valleys.append(
                    BoundaryPort(
                        port_id=f"VPORT_{len(valleys) + 1}",
                        port_type="valley",
                        phi_rad=rp.phi_rad,
                        phi_deg=float(math.degrees(rp.phi_rad)),
                        position=rp.position,
                        scalar_value=float(vals[idx]),
                    )
                )
        else:
            rp = ring_points[idx]
            valleys.append(
                BoundaryPort(
                    port_id=f"VPORT_{len(valleys) + 1}",
                    port_type="valley",
                    phi_rad=rp.phi_rad,
                    phi_deg=float(math.degrees(rp.phi_rad)),
                    position=rp.position,
                    scalar_value=float(vals[idx]),
                )
            )

    # For each raw maximum, compute height above valleys
    ridges: list[BoundaryPort] = []
    for idx in raw_max_indices:
        v_curr = float(smooth_vals[idx])
        if raw_min_indices:
            min_vals = [float(smooth_vals[m]) for m in raw_min_indices]
            valley_depth = v_curr - min(min_vals)
            if valley_depth >= min_prominence or v_span < 1e-12:
                rp = ring_points[idx]
                ridges.append(
                    BoundaryPort(
                        port_id=f"RPORT_{len(ridges) + 1}",
                        port_type="ridge",
                        phi_rad=rp.phi_rad,
                        phi_deg=float(math.degrees(rp.phi_rad)),
                        position=rp.position,
                        scalar_value=float(vals[idx]),
                    )
                )
        else:
            rp = ring_points[idx]
            ridges.append(
                BoundaryPort(
                    port_id=f"RPORT_{len(ridges) + 1}",
                    port_type="ridge",
                    phi_rad=rp.phi_rad,
                    phi_deg=float(math.degrees(rp.phi_rad)),
                    position=rp.position,
                    scalar_value=float(vals[idx]),
                )
            )

    return valleys, ridges


def profile_boundary_loop(
    ring_points: list[BoundaryRingPoint],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    smoothing_window: int = 3,
    min_prominence_frac: float = 0.05,
) -> ExcisionBoundaryResult:
    """Profile a 1D excision boundary loop, extract ports, and compute implied Euler index.

    Args:
        ring_points: List of BoundaryRingPoint sample points forming the closed loop.
        mesh_pts: (N, 3) sphere mesh vertices.
        f_vals: (N,) scalar field values.
        smoothing_window: Moving average smoothing window (default 3).
        min_prominence_frac: Prominence threshold fraction (default 0.05).

    Returns:
        ExcisionBoundaryResult with profile values, valley ports, ridge ports,
        fold order k, and implied index chi = 1 - k.
    """
    if not ring_points:
        return ExcisionBoundaryResult(
            boundary_points=[],
            profile_values=[],
            valley_ports=[],
            ridge_ports=[],
            fold_order=0,
            implied_euler_index=1,
            is_topologically_balanced=True,
        )

    pts_arr = np.array([rp.position for rp in ring_points], dtype=np.float64)
    interp_vals = interpolate_scalar_at_points(
        query_pts=pts_arr,
        mesh_pts=mesh_pts,
        f_vals=f_vals,
    )
    profile_list = [float(v) for v in interp_vals]

    valleys, ridges = extract_boundary_ports(
        profile_values=profile_list,
        ring_points=ring_points,
        smoothing_window=smoothing_window,
        min_prominence_frac=min_prominence_frac,
    )

    k_valleys = len(valleys)
    k_ridges = len(ridges)
    is_balanced = (k_valleys == k_ridges) and (k_valleys > 0)

    # Fold order is number of valley ports (or ridge ports)
    fold_order = k_valleys if is_balanced else max(k_valleys, k_ridges)
    implied_chi = 1 - fold_order if fold_order > 0 else 1

    return ExcisionBoundaryResult(
        boundary_points=ring_points,
        profile_values=profile_list,
        valley_ports=valleys,
        ridge_ports=ridges,
        fold_order=fold_order,
        implied_euler_index=implied_chi,
        is_topologically_balanced=is_balanced,
    )
