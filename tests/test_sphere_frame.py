"""Unit tests for the sphere_frame module (local atom coordinate frame).

The regression these tests exist for: every angular formula in gba_topology2
measures directions as ``p / ||p||``, which is only the outward normal when the
sphere is centred on the world origin. Measured on ethene3.plt, atom C1 sits
1.2618 A off the origin with a true radius of 0.4127 A, and the naive
``mean(||x||)`` radius of 1.3067 A inflates R by 3.17x while shrinking the mesh
pitch by 0.32x. The tests below pin that failure mode and the fix.
"""

import math

import numpy as np
import pytest

from gba_topology2.src.sphere_frame import (
    compute_atom_frame,
    radial_offset,
    to_global,
    to_global_array,
    to_local,
    to_local_array,
)

OFFSET = np.array([2.0, -1.5, 0.8], dtype=np.float64)
RADIUS = 0.412733


def _fibonacci_sphere(
    num_points: int = 1200, radius: float = RADIUS, center: np.ndarray | None = None
) -> np.ndarray:
    """Generate a quasi-uniform point set on a sphere.

    Args:
        num_points: Number of vertices.
        radius: Sphere radius in Angstrom.
        center: Sphere centre; None selects the world origin.

    Returns:
        (num_points, 3) float64 array of vertex coordinates.
    """
    indices = np.arange(0, num_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    pts = np.column_stack([x, y, z])
    if center is not None:
        pts = pts + np.asarray(center, dtype=np.float64)
    return pts


def test_compute_atom_frame_origin_sphere() -> None:
    """A sphere at the origin recovers center ~ 0 and its exact radius."""
    pts = _fibonacci_sphere(center=None)
    center, radius = compute_atom_frame(pts)

    # A discrete shell samples the sphere quasi-uniformly, not exactly.
    assert np.linalg.norm(center) < 1e-3 * RADIUS
    assert pytest.approx(radius, abs=1e-9) == RADIUS


def test_compute_atom_frame_translated_sphere() -> None:
    """A sphere translated off the origin recovers its centre and radius exactly."""
    pts = _fibonacci_sphere(center=OFFSET)
    center, radius = compute_atom_frame(pts)

    assert np.allclose(center, OFFSET, atol=1e-12)
    assert pytest.approx(radius, abs=1e-9) == RADIUS


def test_naive_origin_radius_is_inflated_for_offset_atom() -> None:
    """A/B guard: mean(||x||) about the origin must NOT equal the true radius.

    Disabling the centring (i.e. using the world origin as the centre) makes
    this test fail, which is the whole point: it pins the bug the frame fixes.
    """
    centered = _fibonacci_sphere(center=None)
    offset = _fibonacci_sphere(center=OFFSET)

    _, radius_centered = compute_atom_frame(centered)
    _, radius_offset = compute_atom_frame(offset)

    # The true radius is translation invariant.
    assert pytest.approx(radius_offset, rel=1e-9) == radius_centered

    naive = float(np.mean(np.linalg.norm(offset, axis=1)))
    assert naive > radius_offset * 1.5
    # The pre-fix code used exactly this mean(||x||) as the sphere radius, so
    # every angle it derived was measured from the world origin, not the nucleus.
    assert naive == pytest.approx(
        float(np.mean(np.linalg.norm(offset - np.zeros(3), axis=1))), rel=1e-15
    )


def test_compute_atom_frame_matches_measured_ethene3_geometry() -> None:
    """Reproduce the ethene3 C1 numbers the inspector script reports."""
    center = np.array([1.261797, -0.000090, -0.000193], dtype=np.float64)
    pts = _fibonacci_sphere(num_points=4000, radius=0.412725, center=center)

    recovered, radius = compute_atom_frame(pts)

    assert np.linalg.norm(recovered) > 1.0  # nucleus is far off the origin
    assert pytest.approx(radius, abs=1e-6) == 0.412725
    naive = float(np.mean(np.linalg.norm(pts, axis=1)))
    assert naive / radius == pytest.approx(3.17, abs=0.1)


def test_compute_atom_frame_validation() -> None:
    """Malformed and degenerate inputs raise ValueError."""
    with pytest.raises(ValueError):
        compute_atom_frame(np.empty((0, 3)))
    with pytest.raises(ValueError):
        compute_atom_frame(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        compute_atom_frame(np.zeros(3))
    with pytest.raises(ValueError):
        compute_atom_frame(np.zeros((6, 3)))  # coincident vertices -> R == 0


def test_local_global_round_trip() -> None:
    """to_global(to_local(p)) returns the original world point."""
    pts = _fibonacci_sphere(num_points=200, center=OFFSET)
    for p in pts[:25]:
        local = to_local(p, OFFSET)
        assert np.allclose(local, p - OFFSET, atol=1e-15)
        back = to_global(local, OFFSET)
        assert np.allclose(back, p, atol=1e-15)


def test_array_helpers_match_scalar_helpers() -> None:
    """The (N, 3) helpers agree with the per-point helpers."""
    pts = _fibonacci_sphere(num_points=150, center=OFFSET)
    local = to_local_array(pts, OFFSET)
    global_back = to_global_array(local, OFFSET)

    assert local.shape == pts.shape
    assert np.allclose(local, np.array([to_local(p, OFFSET) for p in pts]), atol=1e-15)
    assert np.allclose(global_back, pts, atol=1e-15)
    # In the centred frame every vertex sits at the recovered radius.
    assert np.allclose(np.linalg.norm(local, axis=1), RADIUS, atol=1e-9)


def test_radial_offset_lifts_along_local_normal() -> None:
    """A radial lift grows the centre-relative distance and keeps the direction."""
    pts = _fibonacci_sphere(num_points=64, center=OFFSET)
    local = to_local_array(pts, OFFSET)

    for i in (0, 7, 33):
        p = local[i]
        lifted = np.array(radial_offset(p, 1.008), dtype=np.float64)
        assert np.linalg.norm(lifted) == pytest.approx(RADIUS * 1.008, rel=1e-12)
        assert np.allclose(lifted / np.linalg.norm(lifted), p / np.linalg.norm(p))


def test_radial_offset_about_world_origin_is_the_bug() -> None:
    """A/B guard: scaling a WORLD point radially about the origin is not a lift.

    The pre-fix viewer scaled ring points by 1.008 in world coordinates. For an
    off-origin atom that moves points along the origin direction, not along the
    atom's normal, so the ring no longer sits at a constant height above the
    sphere. Centring first removes the effect.
    """
    pts = _fibonacci_sphere(num_points=64, center=OFFSET)

    world_lifted = np.array([np.array(p) * 1.008 for p in pts], dtype=np.float64)
    world_heights = np.linalg.norm(world_lifted - OFFSET, axis=1)
    local_lifted = np.array(
        [
            to_global(radial_offset(p, 1.008), OFFSET)
            for p in to_local_array(pts, OFFSET)
        ]
    )
    local_heights = np.linalg.norm(local_lifted - OFFSET, axis=1)

    # Centred lift: every ring point sits at exactly 1.008 R above the nucleus.
    assert np.allclose(local_heights, RADIUS * 1.008, rtol=1e-12)
    # World-space lift: the height above the sphere varies across the ring.
    assert world_heights.max() - world_heights.min() > 1e-3 * RADIUS


def _octahedron(radius: float, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular octahedron mesh.

    Args:
        radius: Circumscribed radius.
        center: (3,) centre in the frame the vertices should be expressed in.

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


def test_pipeline_is_translation_invariant_after_centring() -> None:
    """Morse detection and clustering agree on the centred and offset meshes.

    Centring is a pure translation, so every topological result must be
    identical when each mesh is analysed in its own frame. Analysing the offset
    mesh in world coordinates is what produced the spurious and split clusters.
    """
    from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
    from gba_topology2.src.micro_cluster import compute_micro_clusters
    from gba_topology2.src.morse_detector import compute_discrete_morse_cps

    center = np.array([1.261797, -0.000090, -0.000193], dtype=np.float64)
    pts_local, triangles = _octahedron(RADIUS, np.zeros(3))
    pts_world, _ = _octahedron(RADIUS, center)

    # f = z measured from the nucleus; both meshes carry the same field.
    f_local = pts_local[:, 2].copy()
    f_world = (pts_world - center)[:, 2].copy()

    def _run(pts: np.ndarray, f_vals: np.ndarray) -> tuple[list[int], list[int]]:
        morse = compute_discrete_morse_cps(
            pts, triangles, f_vals, tie_break_epsilon=1e-15
        )
        geom = analyze_mesh_geometry(pts, triangles)
        cps = morse.minima + morse.maxima + morse.saddles
        clusters = compute_micro_clusters(
            cps=cps,
            delta_theta_mesh_rad=float(geom["mesh_pitch_rad"]),
            k_pitch=2.0,
            sphere_radius=float(geom["sphere_radius"]),
        )
        return (
            [len(morse.minima), len(morse.maxima), len(morse.saddles)],
            [cl.local_euler_index for cl in clusters],
        )

    counts_local, chi_local = _run(pts_local, f_local)
    counts_centred, chi_centred = _run(pts_world - center, f_world)
    counts_world, chi_world = _run(pts_world, f_world)

    # Centring reproduces the reference run exactly.
    assert counts_centred == counts_local
    assert chi_centred == chi_local
    # Analysing world coordinates changes the answer: the bug.
    assert counts_world != counts_local or chi_world != chi_local
