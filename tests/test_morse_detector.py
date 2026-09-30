"""Unit tests for morse_detector module."""

import numpy as np
import pytest

from gba_topology2.src.morse_detector import compute_discrete_morse_cps


def test_regular_octahedron_height_function() -> None:
    """Test Morse detection on regular octahedron with height function f = z."""
    pts = np.array(
        [
            [1.0, 0.0, 0.0],  # v0: z = 0
            [-1.0, 0.0, 0.0],  # v1: z = 0
            [0.0, 1.0, 0.0],  # v2: z = 0
            [0.0, -1.0, 0.0],  # v3: z = 0
            [0.0, 0.0, 1.0],  # v4: z = 1 (Maximum)
            [0.0, 0.0, -1.0],  # v5: z = -1 (Minimum)
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

    # Let f = z + slight epsilon to break exact ties at equator
    f_vals = pts[:, 2] + np.array([0.01, -0.01, 0.02, -0.02, 0.0, 0.0])

    res = compute_discrete_morse_cps(pts, triangles, f_vals)
    assert res.is_valid_euler
    assert res.euler_characteristic == 2
    assert len(res.maxima) == 1
    assert len(res.minima) == 1
    assert len(res.saddles) == 0
    assert res.maxima[0].vertex_id == 4
    assert res.minima[0].vertex_id == 5


def test_saddle_configuration() -> None:
    """Test a saddle on octahedron with quadrupole field f = x^2 - y^2."""
    pts = np.array(
        [
            [1.0, 0.0, 0.0],  # v0: x=1, y=0 -> f = 1 (Max)
            [-1.0, 0.0, 0.0],  # v1: x=-1, y=0 -> f = 1 (Max)
            [0.0, 1.0, 0.0],  # v2: x=0, y=1 -> f = -1 (Min)
            [0.0, -1.0, 0.0],  # v3: x=0, y=-1 -> f = -1 (Min)
            [0.0, 0.0, 1.0],  # v4: x=0, y=0 -> f = 0 (Saddle)
            [0.0, 0.0, -1.0],  # v5: x=0, y=0 -> f = 0 (Saddle)
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

    # Quadrupole function f(x, y, z) = x^2 - y^2 + small perturbation to avoid exact ties if needed
    # Link of v4 (north pole): cycle (0 - 2 - 1 - 3 - 0)
    # At v4: f = 0.
    # At v0 (f=1 > 0), v2 (f=-1 < 0), v1 (f=1 > 0), v3 (f=-1 < 0)
    # Lower link of v4: {v2, v3} disjoint! => 2 connected components => Saddle of multiplicity 1!
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2

    res = compute_discrete_morse_cps(pts, triangles, f_vals)
    assert res.is_valid_euler
    assert res.euler_characteristic == 2
    assert len(res.maxima) == 2
    assert len(res.minima) == 2
    assert len(res.saddles) == 2  # v4 and v5 are both saddles
    assert {s.vertex_id for s in res.saddles} == {4, 5}
    assert {m.vertex_id for m in res.maxima} == {0, 1}
    assert {m.vertex_id for m in res.minima} == {2, 3}


def test_input_validation() -> None:
    """Test error handling on dimension mismatch."""
    with pytest.raises(ValueError):
        compute_discrete_morse_cps(
            np.zeros((4, 2)), np.zeros((2, 3), dtype=np.int32), np.zeros(4)
        )
    with pytest.raises(ValueError):
        compute_discrete_morse_cps(
            np.zeros((4, 3)), np.zeros((2, 2), dtype=np.int32), np.zeros(4)
        )
    with pytest.raises(ValueError):
        compute_discrete_morse_cps(
            np.zeros((4, 3)), np.zeros((2, 3), dtype=np.int32), np.zeros(5)
        )
