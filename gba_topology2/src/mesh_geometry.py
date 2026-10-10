"""Mesh Geometry Subroutine.

Calculates geometric properties of a spherical triangle mesh:
- Sphere radius R (mean radial distance of vertices from the sphere centre)
- Average edge length l_bar_e across unique mesh edges
- Angular pitch delta_theta_mesh = 2 * arcsin(l_bar_e / (2 * R)) in radians (and degrees)

Coordinates fed to these routines are expected to be centre-relative: R is a
distance from the atom's nucleus, and the pitch formula is a chord relation on
a sphere of radius R. ``sphere_frame.compute_atom_frame`` recovers that centre
from an off-origin mesh.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "analyze_mesh_geometry",
    "compute_average_edge_length",
    "compute_mesh_pitch",
    "compute_sphere_radius",
    "extract_unique_edges",
]


def compute_sphere_radius(pts: npt.NDArray[np.floating[Any]]) -> float:
    """Compute the mean sphere radius from vertex coordinates.

    Args:
        pts: (N, 3) array of vertex coordinates, measured from the sphere
            centre. Passing world coordinates for an off-origin atom returns
            the mean distance from the world origin, which is inflated.

    Returns:
        Mean Euclidean distance of vertices from the sphere centre.

    Raises:
        ValueError: If pts is empty or has incorrect dimensions.
    """
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"pts must be a non-empty (N, 3) array, got shape {pts.shape}")
    norms = np.linalg.norm(pts, axis=1)
    return float(np.mean(norms))


def extract_unique_edges(
    triangles: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.int64]:
    """Extract sorted unique undirected edges from triangle connectivity.

    Args:
        triangles: (M, 3) array of triangle vertex indices.

    Returns:
        (E, 2) array of unique undirected edges, each with [min_idx, max_idx].

    Raises:
        ValueError: If triangles is empty or has incorrect shape.
    """
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(
            f"triangles must be a non-empty (M, 3) array, got shape {triangles.shape}"
        )

    edges_set: set[tuple[int, int]] = set()
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        e0 = (min(i0, i1), max(i0, i1))
        e1 = (min(i1, i2), max(i1, i2))
        e2 = (min(i0, i2), max(i0, i2))
        edges_set.add(e0)
        edges_set.add(e1)
        edges_set.add(e2)

    return np.array(sorted(edges_set), dtype=np.int64)


def compute_average_edge_length(
    pts: npt.NDArray[np.floating[Any]],
    edges: npt.NDArray[np.integer[Any]],
) -> float:
    """Compute average Euclidean edge length across unique mesh edges.

    Args:
        pts: (N, 3) array of vertex coordinates.
        edges: (E, 2) array of edge vertex indices.

    Returns:
        Average Euclidean edge length l_bar_e.

    Raises:
        ValueError: If edges is empty.
    """
    if edges.shape[0] == 0:
        raise ValueError("edges array must not be empty.")
    p0 = pts[edges[:, 0]]
    p1 = pts[edges[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    return float(np.mean(lengths))


def compute_mesh_pitch(
    avg_edge_length: float, sphere_radius: float
) -> tuple[float, float]:
    """Compute the angular mesh pitch from average edge length and sphere radius.

    delta_theta_mesh = 2 * arcsin(l_bar_e / (2 * R))

    Args:
        avg_edge_length: Average edge length l_bar_e.
        sphere_radius: Sphere radius R.

    Returns:
        Tuple of (delta_theta_rad, delta_theta_deg).

    Raises:
        ValueError: If radius <= 0 or edge length exceeds diameter.
    """
    if sphere_radius <= 0:
        raise ValueError(f"sphere_radius must be positive, got {sphere_radius}")
    ratio = avg_edge_length / (2.0 * sphere_radius)
    ratio_clipped = min(max(ratio, -1.0), 1.0)
    delta_theta_rad = 2.0 * float(np.arcsin(ratio_clipped))
    delta_theta_deg = float(np.degrees(delta_theta_rad))
    return delta_theta_rad, delta_theta_deg


def analyze_mesh_geometry(
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    sphere_radius: float | None = None,
) -> dict[str, Any]:
    """Perform full geometric analysis on a spherical triangle mesh.

    ``sphere_radius`` must be measured from the sphere's own centre, so ``pts``
    is expected to be centre-relative (see ``sphere_frame.compute_atom_frame``).
    For a mesh centred on the world origin the two coincide, which is why an
    off-origin atom silently produced an inflated radius and pitch here.

    Args:
        pts: (N, 3) vertex coordinates, centre-relative.
        triangles: (M, 3) triangle vertex indices.
        sphere_radius: Optional authoritative radius R. When given it replaces
            the mean-norm estimate and is the radius used for the mesh pitch,
            e.g. the value the zone's Aux SphereRadius metadata reports.

    Returns:
        Dictionary containing:
        - 'num_vertices': int
        - 'num_triangles': int
        - 'num_edges': int
        - 'sphere_radius': float (R used for the pitch)
        - 'avg_edge_length': float (mean l_e)
        - 'mesh_pitch_rad': float (delta_theta in radians)
        - 'mesh_pitch_deg': float (delta_theta in degrees)
        - 'unique_edges': (E, 2) ndarray of int64

    Raises:
        ValueError: If sphere_radius is given and is not positive.
    """
    if sphere_radius is not None:
        if sphere_radius <= 0:
            raise ValueError(
                f"sphere_radius override must be positive, got {sphere_radius}"
            )
        r = float(sphere_radius)
    else:
        r = compute_sphere_radius(pts)
    edges = extract_unique_edges(triangles)
    avg_l = compute_average_edge_length(pts, edges)
    pitch_rad, pitch_deg = compute_mesh_pitch(avg_l, r)

    return {
        "num_vertices": int(pts.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "num_edges": int(edges.shape[0]),
        "sphere_radius": r,
        "avg_edge_length": avg_l,
        "mesh_pitch_rad": pitch_rad,
        "mesh_pitch_deg": pitch_deg,
        "unique_edges": edges,
    }
