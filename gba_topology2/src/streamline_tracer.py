"""Streamline and Separatrix Tracing Subroutine for gba_topology2.

Provides continuous, facet-based streamline integration along steepest ascent
(unstable manifolds / ridges) and steepest descent (stable manifolds / valleys)
on a spherical triangulated mesh.

Core Capabilities:
- Trace continuous streamline trajectories across triangle facets from arbitrary
  seed points (e.g., effective saddles, boundary ports).
- Trace 1D separatrix manifold branches connecting effective critical points.
- Smooth streamline paths on S^2 using spherical geodesic relaxation.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.facet_gradient import (
    FacetGradientField,
    find_containing_triangle,
    trace_facet_step,
)

__all__ = [
    "StreamlinePath",
    "smooth_spherical_streamline",
    "trace_streamline",
]


@dataclass(frozen=True)
class StreamlinePath:
    """A continuous trajectory curve on a spherical mesh."""

    # List of (3,) coordinates along the path
    points: list[tuple[float, float, float]]
    # Sequence of triangle indices traversed
    traversed_triangles: list[int]
    # Total path geodesic length in radians
    angular_length_rad: float
    # Total Euclidean arc length in Angstrom
    spatial_length: float
    # Termination status: "max_steps", "stationary", "boundary", "converged"
    status: str


def trace_streamline(
    seed_point: tuple[float, float, float] | npt.NDArray[np.float64],
    direction_sign: float,  # +1.0 for ascent (ridge), -1.0 for descent (valley)
    grad_field: FacetGradientField,
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    sphere_radius: float,
    max_steps: int = 200,
    stop_radius_rad: float = 0.01,
    target_points: list[tuple[float, float, float]] | None = None,
) -> StreamlinePath:
    """Trace a continuous streamline across triangle facets from seed_point.

    Args:
        seed_point: (3,) starting point on the sphere surface.
        direction_sign: +1.0 for steepest ascent, -1.0 for steepest descent.
        grad_field: Precomputed FacetGradientField.
        pts: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle indices.
        sphere_radius: Sphere radius R.
        max_steps: Maximum number of triangle facet traversals.
        stop_radius_rad: Angular radius around target points to terminate integration.
        target_points: Optional list of target destination coordinates (e.g. extrema).

    Returns:
        StreamlinePath containing trajectory points and traversal metrics.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    tris_arr = np.asarray(triangles, dtype=np.int32)
    p_curr = np.asarray(seed_point, dtype=np.float64)

    # Project to exact sphere radius
    norm_p = float(np.linalg.norm(p_curr))
    if norm_p > 1e-12:
        p_curr = (p_curr / norm_p) * sphere_radius

    t_curr = find_containing_triangle(p_curr, pts_arr, tris_arr)
    if t_curr < 0:
        return StreamlinePath(
            points=[(float(p_curr[0]), float(p_curr[1]), float(p_curr[2]))],
            traversed_triangles=[],
            angular_length_rad=0.0,
            spatial_length=0.0,
            status="seed_outside_mesh",
        )

    path_pts: list[tuple[float, float, float]] = [
        (float(p_curr[0]), float(p_curr[1]), float(p_curr[2]))
    ]
    traversed_triangles: list[int] = [t_curr]
    total_ang = 0.0
    total_dist = 0.0
    status = "max_steps"

    target_arr = np.array(target_points, dtype=np.float64) if target_points else None

    for _step in range(max_steps):
        # Check proximity to any target point
        if target_arr is not None and len(target_arr) > 0:
            dots = np.dot(target_arr / sphere_radius, p_curr / sphere_radius)
            dots = np.clip(dots, -1.0, 1.0)
            min_ang = float(np.min(np.arccos(dots)))
            if min_ang <= stop_radius_rad:
                status = "converged"
                break

        p_next, t_next, _edge_idx = trace_facet_step(
            p_start=p_curr,
            triangle_idx=t_curr,
            direction_sign=direction_sign,
            grad_field=grad_field,
            pts=pts_arr,
            triangles=tris_arr,
            sphere_radius=sphere_radius,
        )

        step_dist = float(np.linalg.norm(p_next - p_curr))
        if step_dist < 1e-8:
            status = "stationary"
            break

        # Check for loop / revisiting an edge point
        if any(np.linalg.norm(np.array(prev_p) - p_next) < 1e-7 for prev_p in path_pts):
            path_pts.append((float(p_next[0]), float(p_next[1]), float(p_next[2])))
            status = "loop_detected"
            break

        cos_step = min(
            max(float(np.dot(p_curr, p_next)) / (sphere_radius**2), -1.0), 1.0
        )
        total_ang += math.acos(cos_step)
        total_dist += step_dist

        path_pts.append((float(p_next[0]), float(p_next[1]), float(p_next[2])))
        p_curr = p_next

        if t_next < 0:
            status = "boundary"
            break

        t_curr = t_next
        traversed_triangles.append(t_curr)

    return StreamlinePath(
        points=path_pts,
        traversed_triangles=traversed_triangles,
        angular_length_rad=total_ang,
        spatial_length=total_dist,
        status=status,
    )


def smooth_spherical_streamline(
    path: StreamlinePath,
    sphere_radius: float,
    iterations: int = 3,
    alpha: float = 0.3,
) -> StreamlinePath:
    """Apply geodesic Laplacian smoothing to a streamline on S^2, tethering endpoints.

    Args:
        path: Input StreamlinePath.
        sphere_radius: Sphere radius R.
        iterations: Number of smoothing passes.
        alpha: Relaxation factor (0.0 to 0.5).

    Returns:
        Smoothed StreamlinePath projected strictly onto the sphere surface.
    """
    pts_list = list(path.points)
    n_pts = len(pts_list)
    if n_pts <= 2:
        return path

    arr = np.array(pts_list, dtype=np.float64)

    for _ in range(iterations):
        smoothed = arr.copy()
        for i in range(1, n_pts - 1):
            # Geodesic relaxation between predecessor and successor
            laplacian = 0.5 * (arr[i - 1] + arr[i + 1]) - arr[i]
            p_new = arr[i] + alpha * laplacian
            norm_p = float(np.linalg.norm(p_new))
            if norm_p > 1e-12:
                smoothed[i] = (p_new / norm_p) * sphere_radius
        arr = smoothed

    smoothed_pts: list[tuple[float, float, float]] = [
        (float(arr[k, 0]), float(arr[k, 1]), float(arr[k, 2])) for k in range(n_pts)
    ]

    # Recompute length metrics
    total_dist = 0.0
    total_ang = 0.0
    for k in range(n_pts - 1):
        p0, p1 = arr[k], arr[k + 1]
        total_dist += float(np.linalg.norm(p1 - p0))
        cos_a = min(max(float(np.dot(p0, p1)) / (sphere_radius**2), -1.0), 1.0)
        total_ang += math.acos(cos_a)

    return StreamlinePath(
        points=smoothed_pts,
        traversed_triangles=path.traversed_triangles,
        angular_length_rad=total_ang,
        spatial_length=total_dist,
        status=path.status,
    )
