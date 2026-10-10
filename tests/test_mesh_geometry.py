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


def _octahedron(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular octahedron centred at an arbitrary point.

    Args:
        center: (3,) octahedron centre.
        radius: Circumscribed radius.

    Returns:
        Tuple of (pts, triangles).
    """
    base = (
        np.array(
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
        * radius
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
    return base + center, triangles


def test_geometry_is_translation_invariant_after_centring() -> None:
    """Centring an offset mesh reproduces the centred mesh's radius, edges and pitch."""
    center = np.array([2.0, -1.5, 0.8], dtype=np.float64)
    pts_local, triangles = _octahedron(np.zeros(3), 0.412733)
    pts_world, _ = _octahedron(center, 0.412733)

    local = analyze_mesh_geometry(pts_local, triangles)
    # The pre-fix behaviour: analysing world coordinates directly.
    world = analyze_mesh_geometry(pts_world, triangles)
    # The fix: translate to the atom frame first.
    centred = analyze_mesh_geometry(pts_world - center, triangles)

    assert pytest.approx(local["sphere_radius"], rel=1e-12) == 0.412733
    # World coordinates inflate R, so the pitch shrinks: the clustering scale
    # k * delta_theta is corrupted by exactly this factor.
    assert world["sphere_radius"] > local["sphere_radius"] * 1.5
    assert world["mesh_pitch_rad"] < local["mesh_pitch_rad"]

    assert pytest.approx(centred["sphere_radius"], rel=1e-12) == local["sphere_radius"]
    assert (
        pytest.approx(centred["avg_edge_length"], rel=1e-12) == local["avg_edge_length"]
    )
    assert (
        pytest.approx(centred["mesh_pitch_rad"], rel=1e-12) == local["mesh_pitch_rad"]
    )


def test_sphere_radius_override_drives_pitch() -> None:
    """An authoritative radius replaces the mean-norm estimate and sets the pitch."""
    pts, triangles = _octahedron(np.zeros(3), 1.0)
    default = analyze_mesh_geometry(pts, triangles)

    # Halving the radius doubles the chord ratio, so the pitch must grow.
    override = analyze_mesh_geometry(pts, triangles, sphere_radius=0.5)

    assert pytest.approx(override["sphere_radius"], rel=1e-15) == 0.5
    assert override["mesh_pitch_rad"] > default["mesh_pitch_rad"]
    expected_rad, expected_deg = compute_mesh_pitch(default["avg_edge_length"], 0.5)
    assert pytest.approx(override["mesh_pitch_rad"], rel=1e-12) == expected_rad
    assert pytest.approx(override["mesh_pitch_deg"], rel=1e-12) == expected_deg
    # Edge lengths are a mesh property, independent of the radius source.
    assert (
        pytest.approx(override["avg_edge_length"], rel=1e-15)
        == default["avg_edge_length"]
    )


def test_sphere_radius_override_validation() -> None:
    """A non-positive radius override is rejected."""
    pts, triangles = _octahedron(np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        analyze_mesh_geometry(pts, triangles, sphere_radius=0.0)
    with pytest.raises(ValueError):
        analyze_mesh_geometry(pts, triangles, sphere_radius=-1.0)
