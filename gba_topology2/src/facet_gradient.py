"""Facet Surface Gradient and Triangle Adjacency Subroutine.

Provides exact piecewise-linear gradient computation on triangular facets of a
spherical mesh:
- Triangle unit normals and areas
- Triangle constant surface gradient: \\nabla_S f
- Dual mesh half-edge and edge-adjacency map for rapid face-to-face traversal
- Exact ray / segment crossing calculations across triangle boundary edges on S^2

Mathematical Formulation:
    On triangle T = (v_0, v_1, v_2) with vertex scalar values f_0, f_1, f_2:
    The 3D affine gradient in the plane of T is:
        \\nabla f = \\frac{1}{2 A_T} \\sum_{i=0}^2 f_i (\\hat{n}_T \\times \\vec{e}_i)
    where \\vec{e}_i is the directed edge opposite to vertex v_i, and \\hat{n}_T
    is the outward unit normal of triangle T.
    Since vertices reside on a sphere of radius R, \\nabla f is already tangent
    to the planar facet T. The sphere-tangent component at any point \\vec{p} on T
    is obtained by projecting out the radial normal \\hat{n}_p = \\vec{p} / \\|\\vec{p}\\|:
        \\nabla_S f(\\vec{p}) = \\nabla f - (\\nabla f \\cdot \\hat{n}_p) \\hat{n}_p
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "FacetGradientField",
    "MeshAdjacency",
    "build_mesh_adjacency",
    "compute_facet_gradients",
    "find_containing_triangle",
    "trace_facet_step",
]


@dataclass(frozen=True)
class MeshAdjacency:
    """Topology structure storing triangle-to-triangle neighbor connectivity."""

    # Map from directed edge (v_start, v_end) -> triangle index
    directed_edge_to_triangle: dict[tuple[int, int], int]
    # (M, 3) array: neighbors[t, i] is the triangle adjacent across edge opposite to vertex i (or -1)
    triangle_neighbors: npt.NDArray[np.int32]
    # (M, 3) array of triangle unit normals
    triangle_normals: npt.NDArray[np.float64]
    # (M,) array of triangle planar areas
    triangle_areas: npt.NDArray[np.float64]


@dataclass(frozen=True)
class FacetGradientField:
    """Computed piecewise-linear surface gradient field on a triangular mesh."""

    # (M, 3) array of constant planar gradient vectors \\nabla f for each triangle
    facet_gradients: npt.NDArray[np.float64]
    # (M,) array of gradient magnitudes \\|\\nabla f\\|
    gradient_magnitudes: npt.NDArray[np.float64]
    # Topology and adjacency map
    adjacency: MeshAdjacency


def build_mesh_adjacency(
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
) -> MeshAdjacency:
    """Build triangle neighbor adjacency and compute geometric normals and areas.

    Args:
        pts: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle vertex indices.

    Returns:
        MeshAdjacency dataclass with connectivity and face geometry.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    tris_arr = np.asarray(triangles, dtype=np.int32)
    n_triangles = tris_arr.shape[0]

    # Compute triangle normals and areas
    p0 = pts_arr[tris_arr[:, 0]]
    p1 = pts_arr[tris_arr[:, 1]]
    p2 = pts_arr[tris_arr[:, 2]]

    e01 = p1 - p0
    e02 = p2 - p0
    cross = np.cross(e01, e02)
    cross_norms = np.linalg.norm(cross, axis=1)

    # Guard against zero-area triangles
    valid = cross_norms > 1e-14
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / cross_norms[valid, np.newaxis]
    areas = 0.5 * cross_norms

    # Build directed edge map and neighbor table
    directed_edges: dict[tuple[int, int], int] = {}
    for t_idx in range(n_triangles):
        v0, v1, v2 = (
            int(tris_arr[t_idx, 0]),
            int(tris_arr[t_idx, 1]),
            int(tris_arr[t_idx, 2]),
        )
        directed_edges[(v0, v1)] = t_idx
        directed_edges[(v1, v2)] = t_idx
        directed_edges[(v2, v0)] = t_idx

    # neighbors[t, i] is neighbor across edge opposite to vertex i:
    # i=0 (opp v0) -> edge (v1, v2) (matching twin is (v2, v1))
    # i=1 (opp v1) -> edge (v2, v0) (matching twin is (v0, v2))
    # i=2 (opp v2) -> edge (v0, v1) (matching twin is (v1, v0))
    neighbors = np.full((n_triangles, 3), -1, dtype=np.int32)
    for t_idx in range(n_triangles):
        v0, v1, v2 = (
            int(tris_arr[t_idx, 0]),
            int(tris_arr[t_idx, 1]),
            int(tris_arr[t_idx, 2]),
        )
        neighbors[t_idx, 0] = directed_edges.get((v2, v1), -1)
        neighbors[t_idx, 1] = directed_edges.get((v0, v2), -1)
        neighbors[t_idx, 2] = directed_edges.get((v1, v0), -1)

    return MeshAdjacency(
        directed_edge_to_triangle=directed_edges,
        triangle_neighbors=neighbors,
        triangle_normals=normals,
        triangle_areas=areas,
    )


def compute_facet_gradients(
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    adjacency: MeshAdjacency | None = None,
) -> FacetGradientField:
    """Compute the constant surface gradient vector \\nabla f on each triangle facet.

    Formula on triangle T = (v_0, v_1, v_2):
        \\nabla f = \\frac{1}{2 A_T} \\left( f_0 (\\hat{n}_T \\times (v_2 - v_1)) +
                                            f_1 (\\hat{n}_T \\times (v_0 - v_2)) +
                                            f_2 (\\hat{n}_T \\times (v_1 - v_0)) \\right)

    Args:
        pts: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle vertex indices.
        f_vals: (N,) scalar field values at vertices.
        adjacency: Optional pre-computed MeshAdjacency structure.

    Returns:
        FacetGradientField containing per-triangle gradient vectors.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    tris_arr = np.asarray(triangles, dtype=np.int32)
    f_arr = np.asarray(f_vals, dtype=np.float64)

    if adjacency is None:
        adj = build_mesh_adjacency(pts_arr, tris_arr)
    else:
        adj = adjacency

    n_triangles = tris_arr.shape[0]
    p0 = pts_arr[tris_arr[:, 0]]
    p1 = pts_arr[tris_arr[:, 1]]
    p2 = pts_arr[tris_arr[:, 2]]

    f0 = f_arr[tris_arr[:, 0], np.newaxis]
    f1 = f_arr[tris_arr[:, 1], np.newaxis]
    f2 = f_arr[tris_arr[:, 2], np.newaxis]

    n_T = adj.triangle_normals
    areas = adj.triangle_areas

    # Directed opposite edges
    e0 = p2 - p1  # opposite v0
    e1 = p0 - p2  # opposite v1
    e2 = p1 - p0  # opposite v2

    cross0 = np.cross(n_T, e0)
    cross1 = np.cross(n_T, e1)
    cross2 = np.cross(n_T, e2)

    grad = f0 * cross0 + f1 * cross1 + f2 * cross2

    valid = areas > 1e-14
    inv_2a = np.zeros(n_triangles, dtype=np.float64)
    inv_2a[valid] = 1.0 / (2.0 * areas[valid])

    grad = grad * inv_2a[:, np.newaxis]
    mags = np.linalg.norm(grad, axis=1)

    return FacetGradientField(
        facet_gradients=grad,
        gradient_magnitudes=mags,
        adjacency=adj,
    )


def find_containing_triangle(
    point: tuple[float, float, float] | npt.NDArray[np.float64],
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    candidate_triangles: list[int] | None = None,
) -> int:
    """Find the triangle index whose planar projection contains the given point.

    Uses barycentric coordinates on the spherical triangle projection.

    Args:
        point: (3,) coordinates of target point on the sphere.
        pts: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle vertex indices.
        candidate_triangles: Optional restricted list of triangle indices to search.

    Returns:
        Index of the containing triangle, or -1 if not found.
    """
    p_arr = np.asarray(point, dtype=np.float64)
    pts_arr = np.asarray(pts, dtype=np.float64)
    tris_arr = np.asarray(triangles, dtype=np.int32)

    search_list = (
        candidate_triangles if candidate_triangles is not None else range(len(tris_arr))
    )

    best_t = -1
    best_min_bary = -float("inf")

    for t_idx in search_list:
        v0, v1, v2 = (
            pts_arr[tris_arr[t_idx, 0]],
            pts_arr[tris_arr[t_idx, 1]],
            pts_arr[tris_arr[t_idx, 2]],
        )
        # Compute barycentric coordinates of ray passing through p_arr onto triangle (v0, v1, v2)
        # Using scalar triple product (volume coordinates)
        v01 = np.cross(v0, v1)
        v12 = np.cross(v1, v2)
        v20 = np.cross(v2, v0)

        det = float(np.dot(v0, v12))
        if abs(det) < 1e-14:
            continue

        b0 = float(np.dot(p_arr, v12)) / det
        b1 = float(np.dot(p_arr, v20)) / det
        b2 = float(np.dot(p_arr, v01)) / det

        min_b = min(b0, b1, b2)
        if min_b >= -1e-6:
            return t_idx
        if min_b > best_min_bary:
            best_min_bary = min_b
            best_t = t_idx

    return best_t if best_min_bary >= -0.05 else -1


def trace_facet_step(
    p_start: npt.NDArray[np.float64],
    triangle_idx: int,
    direction_sign: float,  # +1.0 for ascent, -1.0 for descent
    grad_field: FacetGradientField,
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    sphere_radius: float,
) -> tuple[npt.NDArray[np.float64], int, int]:
    """Integrate a streamline step across triangle_idx from p_start to its boundary exit edge.

    Within the facet, the gradient vector \\nabla f is constant. The streamline follows:
        \\vec{v} = \\text{sign} \\cdot \\nabla_S f(p_{\\text{start}})
    and intersects one of the three boundary great-circle arcs of the triangle.

    Args:
        p_start: (3,) starting point on the sphere surface within triangle_idx.
        triangle_idx: Current triangle index.
        direction_sign: +1.0 for ascending (ridge), -1.0 for descending (valley).
        grad_field: Precomputed FacetGradientField.
        pts: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle indices.
        sphere_radius: Sphere radius R.

    Returns:
        Tuple of (p_exit, next_triangle_idx, exit_edge_idx) where:
        - p_exit is the exact (3,) exit point on the boundary edge
        - next_triangle_idx is the neighboring triangle index (-1 if boundary reached)
        - exit_edge_idx is local edge index (0, 1, or 2) opposite to vertex i
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    tris_arr = np.asarray(triangles, dtype=np.int32)
    p_curr = np.asarray(p_start, dtype=np.float64)

    grad_T = grad_field.facet_gradients[triangle_idx]
    norm_p = float(np.linalg.norm(p_curr))
    n_p = p_curr / norm_p if norm_p > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Project facet gradient to sphere tangent plane at p_curr
    v_tan = direction_sign * (grad_T - float(np.dot(grad_T, n_p)) * n_p)
    norm_v = float(np.linalg.norm(v_tan))

    if norm_v < 1e-14:
        # Stationary point / zero gradient facet
        return p_curr, -1, -1

    v_dir = v_tan / norm_v

    # Three edges of current triangle:
    # edge 0: (v1, v2) opposite v0
    # edge 1: (v2, v0) opposite v1
    # edge 2: (v0, v1) opposite v2
    v_indices = [int(tris_arr[triangle_idx, i]) for i in range(3)]
    edge_pairs = [
        (v_indices[1], v_indices[2]),
        (v_indices[2], v_indices[0]),
        (v_indices[0], v_indices[1]),
    ]

    best_exit_p: npt.NDArray[np.float64] | None = None
    best_dist = float("inf")
    best_edge_local = -1

    for e_local, (va, vb) in enumerate(edge_pairs):
        pa = pts_arr[va]
        pb = pts_arr[vb]
        # Great-circle normal of edge arc: n_edge = pa x pb
        n_edge = np.cross(pa, pb)
        norm_ne = float(np.linalg.norm(n_edge))
        if norm_ne < 1e-14:
            continue
        n_edge /= norm_ne

        # Intersection of streamline great-circle (n_stream = p_curr x v_dir) with edge great-circle n_edge
        # Line of intersection vector: d_int = n_stream x n_edge
        n_stream = np.cross(p_curr, v_dir)
        d_int = np.cross(n_stream, n_edge)
        norm_di = float(np.linalg.norm(d_int))
        if norm_di < 1e-14:
            continue
        d_int /= norm_di

        # Pick forward intersection (dot with v_dir > 0)
        if float(np.dot(d_int, v_dir)) < 0:
            d_int = -d_int

        p_cand = d_int * sphere_radius

        # Check if p_cand lies between pa and pb along edge (positive barycentric weights)
        # Using vector triple product projection along edge
        e_vec = pb - pa
        e_len2 = float(np.dot(e_vec, e_vec))
        if e_len2 > 1e-14:
            proj_t = float(np.dot(p_cand - pa, e_vec)) / e_len2
            # Allow slight epsilon margin on edge segment
            if -0.01 <= proj_t <= 1.01:
                # Angular distance from p_curr
                cos_ang = min(max(float(np.dot(p_curr / norm_p, d_int)), -1.0), 1.0)
                dist = math.acos(cos_ang)
                if 1e-6 < dist < best_dist:
                    best_dist = dist
                    best_exit_p = p_cand
                    best_edge_local = e_local

    if best_exit_p is None or best_edge_local == -1:
        # Step forward small fixed angular delta along v_dir as fallback
        dtheta = 0.01
        p_step = (math.cos(dtheta) * n_p + math.sin(dtheta) * v_dir) * sphere_radius
        next_t = find_containing_triangle(p_step, pts_arr, tris_arr)
        return p_step, next_t, -1

    next_triangle = int(
        grad_field.adjacency.triangle_neighbors[triangle_idx, best_edge_local]
    )
    return best_exit_p, next_triangle, best_edge_local
