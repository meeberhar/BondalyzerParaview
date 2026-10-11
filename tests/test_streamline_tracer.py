"""Unit tests for facet_gradient and streamline_tracer modules."""

import numpy as np
import pytest

from gba_topology2.src.facet_gradient import (
    build_mesh_adjacency,
    compute_facet_gradients,
    find_containing_triangle,
)
from gba_topology2.src.streamline_tracer import (
    smooth_spherical_streamline,
    trace_streamline,
)


def _regular_octahedron(radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Construct regular octahedron on a sphere of radius R."""
    pts = np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    return pts, triangles


def test_build_mesh_adjacency_octahedron() -> None:
    """Test triangle adjacency on a regular closed octahedron."""
    pts, triangles = _regular_octahedron(radius=1.0)
    adj = build_mesh_adjacency(pts, triangles)

    assert adj.triangle_neighbors.shape == (8, 3)
    # Every triangle in a closed manifold sphere has 3 valid neighbor triangles (no boundary edges -1)
    assert np.all(adj.triangle_neighbors >= 0)
    assert adj.triangle_normals.shape == (8, 3)
    # Planar area of each equilateral triangle with edge sqrt(2) is sqrt(3)/2 ~ 0.866025
    expected_area = np.sqrt(3.0) / 2.0
    assert np.allclose(adj.triangle_areas, expected_area, atol=1e-6)


def test_compute_facet_gradients_linear_field() -> None:
    """Test facet gradient computation on field f = z: \\nabla f should point in +z direction on all facets."""
    pts, triangles = _regular_octahedron(radius=1.0)
    # f = z
    f_vals = pts[:, 2].copy()

    gf = compute_facet_gradients(pts, triangles, f_vals)
    assert gf.facet_gradients.shape == (8, 3)

    # For top triangles (faces 0..3), z-component of gradient must be positive
    for t_idx in range(4):
        assert gf.facet_gradients[t_idx, 2] > 0.0


def test_find_containing_triangle() -> None:
    """Test point location on spherical triangles."""
    pts, triangles = _regular_octahedron(radius=1.0)
    # Point near (1, 1, 1)/sqrt(3) should be inside triangle 0 (vertices 0, 2, 4)
    p_test = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    t_idx = find_containing_triangle(p_test, pts, triangles)
    assert t_idx == 0


def test_trace_streamline_ascent_and_descent() -> None:
    """Test continuous streamline integration along field f = z."""
    pts, triangles = _regular_octahedron(radius=1.0)
    f_vals = pts[:, 2].copy()  # minimum at (0, 0, -1), maximum at (0, 0, 1)
    gf = compute_facet_gradients(pts, triangles, f_vals)

    # Start near equator in triangle 0, step upwards (+1.0 ascent)
    p_start = np.array([0.7, 0.7, 0.1], dtype=np.float64)
    p_start /= np.linalg.norm(p_start)

    path_up = trace_streamline(
        seed_point=p_start,
        direction_sign=+1.0,
        grad_field=gf,
        pts=pts,
        triangles=triangles,
        sphere_radius=1.0,
        max_steps=20,
    )

    assert len(path_up.points) >= 2
    # Ascent must reach the top apex (0, 0, 1)
    assert path_up.points[1][2] > path_up.points[0][2]
    assert np.isclose(path_up.points[1][2], 1.0, atol=1e-6)

    # Descending path (-1.0 descent) must decrease z coordinate towards equator/bottom
    path_down = trace_streamline(
        seed_point=p_start,
        direction_sign=-1.0,
        grad_field=gf,
        pts=pts,
        triangles=triangles,
        sphere_radius=1.0,
        max_steps=20,
    )
    assert len(path_down.points) >= 2
    assert path_down.points[1][2] < path_down.points[0][2]


def test_smooth_spherical_streamline() -> None:
    """Test spherical geodesic relaxation preserves endpoints and stays on S^2."""
    pts, triangles = _regular_octahedron(radius=1.0)
    f_vals = pts[:, 2].copy()
    gf = compute_facet_gradients(pts, triangles, f_vals)

    p_start = np.array([0.7, 0.7, 0.1], dtype=np.float64)
    p_start /= np.linalg.norm(p_start)

    raw_path = trace_streamline(p_start, +1.0, gf, pts, triangles, 1.0, max_steps=10)
    smoothed = smooth_spherical_streamline(
        raw_path, sphere_radius=1.0, iterations=5, alpha=0.3
    )

    assert len(smoothed.points) == len(raw_path.points)
    # Endpoints must remain tethered
    assert np.allclose(smoothed.points[0], raw_path.points[0], atol=1e-12)
    assert np.allclose(smoothed.points[-1], raw_path.points[-1], atol=1e-12)
    # Every point must lie exactly on the unit sphere
    for pt in smoothed.points:
        assert pytest.approx(np.linalg.norm(np.array(pt)), abs=1e-12) == 1.0
