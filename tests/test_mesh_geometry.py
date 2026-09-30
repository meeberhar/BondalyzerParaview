"""Unit tests for mesh_geometry module."""

import numpy as np
import pytest

from gba_topology2.src.mesh_geometry import (
    analyze_mesh_geometry,
    compute_average_edge_length,
    compute_mesh_pitch,
    compute_sphere_radius,
    extract_unique_edges,
)


def test_regular_octahedron_mesh_geometry() -> None:
    """Test geometry metrics on a regular octahedron inscribed in a unit sphere."""
    # 6 vertices on axes
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    # 8 faces
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

    res = analyze_mesh_geometry(pts, triangles)
    assert res["num_vertices"] == 6
    assert res["num_triangles"] == 8
    assert res["num_edges"] == 12
    assert pytest.approx(res["sphere_radius"], abs=1e-7) == 1.0

    # In regular octahedron on unit sphere, all edges have length sqrt(2)
    expected_edge_len = np.sqrt(2.0)
    assert pytest.approx(res["avg_edge_length"], abs=1e-7) == expected_edge_len

    # delta_theta = 2 * arcsin(sqrt(2) / 2) = 2 * arcsin(1 / sqrt(2)) = 2 * (pi/4) = pi/2 = 90 deg
    assert pytest.approx(res["mesh_pitch_rad"], abs=1e-7) == np.pi / 2.0
    assert pytest.approx(res["mesh_pitch_deg"], abs=1e-7) == 90.0


def test_edge_cases_and_exceptions() -> None:
    """Test validation errors for invalid inputs."""
    with pytest.raises(ValueError):
        compute_sphere_radius(np.empty((0, 3)))

    with pytest.raises(ValueError):
        extract_unique_edges(np.empty((0, 3), dtype=np.int32))

    with pytest.raises(ValueError):
        compute_average_edge_length(np.zeros((5, 3)), np.empty((0, 2), dtype=np.int64))

    with pytest.raises(ValueError):
        compute_mesh_pitch(1.0, 0.0)
