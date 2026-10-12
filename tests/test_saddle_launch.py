"""Unit tests for the Saddle Launch Vector subroutine (Basin Locator Step 2).

The load-bearing case is the 3-fold monkey saddle: the Hessian is degenerate at
its centre, so the cyclic-link sector scan is the only method here that gets
its three ridge and three valley branches. The octahedron quadrupole pins the
simple k = 2 case against the Morse detector's own multiplicity.
"""

import math

import numpy as np
import pytest

from gba_topology2.src.morse_detector import CriticalPoint, compute_discrete_morse_cps
from gba_topology2.src.saddle_launch import (
    build_cyclic_link,
    compute_all_saddle_launch_seeds,
    extract_saddle_launch_seed,
)

OCTA_PTS = np.array(
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
OCTA_TRIS = np.array(
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


def _monkey_mesh(radius: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sphere with a north pole, 6 equatorial vertices and a south pole.

    The field f = sin(3*theta) sampled at theta = j*60 + 30 degrees alternates
    +1/-1 around the ring, so both poles see 3 upper and 3 lower link sectors:
    genuine 3-fold (monkey) saddles with multiplicity 2.
    """
    top = np.array([0.0, 0.0, radius])
    bot = np.array([0.0, 0.0, -radius])
    ring = []
    fvals = [0.0]
    for j in range(6):
        theta = math.radians(60.0 * j + 30.0)
        ring.append([radius * math.cos(theta), radius * math.sin(theta), 0.0])
        fvals.append(math.sin(3.0 * theta))
    fvals += [0.0]

    pts = np.array([top] + ring + [bot], dtype=np.float64)
    tris = []
    for j in range(6):
        nxt = 1 + ((j + 1) % 6)
        tris.append([0, 1 + j, nxt])
        tris.append([7, nxt, 1 + j])
    return pts, np.array(tris, dtype=np.int32), np.array(fvals, dtype=np.float64)


def test_build_cyclic_link_octahedron_pole() -> None:
    """The north pole of the octahedron has a 4-cycle link."""
    link = build_cyclic_link(4, OCTA_TRIS)
    assert sorted(link) == [0, 1, 2, 3]

    # Consecutive entries (cyclically) must be genuine link edges: the link of
    # vertex 4 is the cycle 0-2-1-3-0.
    allowed = {(0, 2), (2, 0), (2, 1), (1, 2), (1, 3), (3, 1), (3, 0), (0, 3)}
    for i in range(len(link)):
        pair = (link[i], link[(i + 1) % len(link)])
        assert pair in allowed, f"{pair} is not a link edge of vertex 4"


def test_build_cyclic_link_isolated_vertex_is_empty() -> None:
    """A vertex referenced by no triangle has an empty link."""
    assert build_cyclic_link(99, OCTA_TRIS) == []


def test_simple_saddle_gives_two_ascent_and_two_descent() -> None:
    """Quadrupole on the octahedron: k = 2, matching the detector's multiplicity."""
    f_vals = OCTA_PTS[:, 0] ** 2 - OCTA_PTS[:, 1] ** 2
    res = compute_discrete_morse_cps(OCTA_PTS, OCTA_TRIS, f_vals, tie_break_epsilon=0.0)

    sad = next(s for s in res.saddles if s.vertex_id == 4)
    seed = extract_saddle_launch_seed(
        saddle=sad,
        mesh_pts=OCTA_PTS,
        triangles=OCTA_TRIS,
        f_vals=f_vals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=math.radians(90.0),
    )

    assert seed.fold_order == 2
    assert len(seed.ascent_branches) == 2
    assert len(seed.descent_branches) == 2
    assert len(seed.separatrix_directions) == 4
    assert seed.index_agrees, "observed fold order must match mu + 1"

    # Ascent sectors sit above the saddle value, descent below.
    assert all(
        b.sector_centroid_value > seed.saddle_value for b in seed.ascent_branches
    )
    assert all(
        b.sector_centroid_value < seed.saddle_value for b in seed.descent_branches
    )


def test_monkey_saddle_gives_three_and_three() -> None:
    """k = 3 monkey saddle: 3 ridge + 3 valley branches, Hessian-blind case."""
    pts, tris, fvals = _monkey_mesh()
    sad = CriticalPoint(
        vertex_id=0,
        cp_type="saddle",
        value=0.0,
        position=(0.0, 0.0, 1.0),
        multiplicity=2,
    )

    seed = extract_saddle_launch_seed(
        saddle=sad,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=math.radians(30.0),
    )

    assert seed.fold_order == 3
    assert len(seed.ascent_branches) == 3
    assert len(seed.descent_branches) == 3
    assert len(seed.separatrix_directions) == 6

    # The three ridge branches are 120 degrees apart, and so are the valleys.
    for branches in (seed.ascent_branches, seed.descent_branches):
        dirs = [np.array(b.tangent_direction) for b in branches]
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                ang = math.degrees(
                    math.acos(float(np.clip(np.dot(dirs[i], dirs[j]), -1, 1)))
                )
                assert abs(ang - 120.0) < 1e-4, f"branch pair {i},{j} at {ang} deg"


def test_monkey_saddle_index_agrees_with_detector() -> None:
    """The detector's multiplicity 2 must match the observed fold order 3."""
    pts, tris, fvals = _monkey_mesh()
    res = compute_discrete_morse_cps(pts, tris, fvals, tie_break_epsilon=0.0)
    assert res.euler_characteristic == 2

    sad = next(s for s in res.saddles if s.vertex_id == 0)
    assert sad.multiplicity == 2

    seed = extract_saddle_launch_seed(
        saddle=sad,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=math.radians(30.0),
    )
    assert seed.fold_order == 3
    assert seed.index_agrees


def test_launch_positions_lie_on_the_sphere() -> None:
    """Geodesic stepping must keep every seed exactly at radius R."""
    pts, tris, fvals = _monkey_mesh(radius=0.412733)
    sad = CriticalPoint(
        vertex_id=0,
        cp_type="saddle",
        value=0.0,
        position=(0.0, 0.0, 0.412733),
        multiplicity=2,
    )
    seed = extract_saddle_launch_seed(
        saddle=sad,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=0.412733,
        delta_theta_mesh_rad=math.radians(30.0),
    )
    for b in seed.all_branches:
        r = float(np.linalg.norm(b.launch_position))
        assert abs(r - 0.412733) < 1e-12, f"seed at radius {r}"


def test_tangent_directions_are_perpendicular_to_the_normal() -> None:
    """Launch directions must be surface tangents, never radial."""
    pts, tris, fvals = _monkey_mesh()
    sad = CriticalPoint(
        vertex_id=0,
        cp_type="saddle",
        value=0.0,
        position=(0.0, 0.0, 1.0),
        multiplicity=2,
    )
    seed = extract_saddle_launch_seed(
        saddle=sad,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=math.radians(30.0),
    )
    normal = np.array([0.0, 0.0, 1.0])
    for b in seed.all_branches:
        assert abs(float(np.dot(b.tangent_direction, normal))) < 1e-12


def test_launch_step_pitch_scales_the_displacement() -> None:
    """The pitch multiplier must move the seed proportionally along the sphere."""
    pts, tris, fvals = _monkey_mesh()
    sad = CriticalPoint(
        vertex_id=0,
        cp_type="saddle",
        value=0.0,
        position=(0.0, 0.0, 1.0),
        multiplicity=2,
    )
    pitch = math.radians(30.0)

    seed = extract_saddle_launch_seed(
        saddle=sad,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=pitch,
        launch_step_pitch=0.35,
    )
    b = seed.ascent_branches[0]
    centre = np.array([0.0, 0.0, 1.0])
    ang = math.acos(float(np.clip(np.dot(centre, np.array(b.launch_position)), -1, 1)))
    assert abs(ang - 0.35 * pitch) < 1e-9


def test_extremum_yields_no_branches() -> None:
    """A maximum has no sign change around its link, so nothing is launched."""
    pts, tris, fvals = _monkey_mesh()
    # Raise the north pole well above everything: it becomes a maximum.
    fvals = fvals.copy()
    fvals[0] = 10.0
    res = compute_discrete_morse_cps(pts, tris, fvals, tie_break_epsilon=0.0)
    mx = next(m for m in res.maxima if m.vertex_id == 0)

    seed = extract_saddle_launch_seed(
        saddle=mx,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=math.radians(30.0),
    )
    assert seed.ascent_branches == []
    assert seed.descent_branches == []
    assert seed.fold_order == 0
    assert not seed.index_agrees


def test_invalid_launch_step_pitch_raises() -> None:
    """A non-positive launch offset would place the seed on the degenerate centre."""
    pts, tris, fvals = _monkey_mesh()
    sad = CriticalPoint(
        vertex_id=0,
        cp_type="saddle",
        value=0.0,
        position=(0.0, 0.0, 1.0),
        multiplicity=2,
    )
    with pytest.raises(ValueError, match="launch_step_pitch must be > 0"):
        extract_saddle_launch_seed(
            saddle=sad,
            mesh_pts=pts,
            triangles=tris,
            f_vals=fvals,
            sphere_radius=1.0,
            delta_theta_mesh_rad=math.radians(30.0),
            launch_step_pitch=0.0,
        )


def test_compute_all_seeds_preserves_order_and_count() -> None:
    """The batch entry point returns one seed per saddle, in input order."""
    pts, tris, fvals = _monkey_mesh()
    res = compute_discrete_morse_cps(pts, tris, fvals, tie_break_epsilon=0.0)
    seeds = compute_all_saddle_launch_seeds(
        saddles=res.saddles,
        mesh_pts=pts,
        triangles=tris,
        f_vals=fvals,
        sphere_radius=1.0,
        delta_theta_mesh_rad=math.radians(30.0),
    )
    assert len(seeds) == len(res.saddles)
    assert [s.vertex_id for s in seeds] == [s.vertex_id for s in res.saddles]
    # Both poles are 3-fold in this field.
    assert all(s.fold_order == 3 for s in seeds)
