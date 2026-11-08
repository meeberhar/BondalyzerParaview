"""Unit tests for :mod:`gba_topology2.src.ridge_refiner`.

The refiner pulls a gradient-traced connector onto the ridge crest while keeping
its real corners sharp. Every test is built on a synthetic field with a known
answer, and the set is organised around the two confusions the design exists to
resolve:

- **Sawtooth chatter is not a corner.** In flat ground the ascent zigzags across
  the crest with one-pitch teeth. The point-to-point turning angle is huge, and so
  is the windowed chord deflection -- what saves the detector is that the teeth
  give EVERY vertex the same deflection, so the baseline test cancels them.
- **Sustained curvature is not a corner either.** A ridge that curves smoothly
  deflects every vertex in the window by the same amount. The latitude circle is
  the clean example and the harshest one available: at colatitude 0.20 rad it
  turns 56.7 deg per window, above the 35 deg threshold, at EVERY vertex. An
  absolute threshold calls the whole arc corners. The baseline test reports none,
  which makes it the most important assertion in this file.

The frame contract is tested explicitly: the refiner works on centred unit
radials and exchanges world-frame polylines with its caller, so refining through
an off-origin centre must give the same geometry as refining in the centred frame.
Pd_20K sits at the origin and cannot see this class of bug; ethene3 at 1.2618 A
can.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.excision_boundary import interpolate_scalar_at_points
from gba_topology2.src.ridge_refiner import (
    CORNER_EXCESS_DEG,
    CORNER_WINDOW_PITCH,
    KNOT_MAX_PITCH,
    NUDGE_MAX_PITCH,
    SEGMENT_SAMPLE_DEG,
    RidgeRefinement,
    _scan_crests,
    chord_deflection_deg,
    detect_corners,
    probe_gradient_ridge,
    refine_gradient_ridge,
)
from gba_topology2.src.sphere_frame import (
    angular_distance,
    rotate_towards,
    to_global,
    to_local,
)

_N_MESH = 1200
_R = 1.0
_PITCH = math.radians(3.0)

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def _fibonacci_sphere(
    num_points: int = _N_MESH, radius: float = _R
) -> npt.NDArray[np.float64]:
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _unit(v: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _local(
    p: npt.ArrayLike, center: tuple[float, float, float] | None
) -> npt.NDArray[np.float64]:
    """World point -> centred unit radial. ``None`` means already centred."""
    arr = np.asarray(p, dtype=np.float64)
    return _unit(to_local(arr, center) if center else arr)


def _angle_deg(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    d = float(np.clip(float(np.dot(_unit(a), _unit(b))), -1.0, 1.0))
    return math.degrees(math.acos(d))


def _latitude_circle(
    theta: float, n: int, step_rad: float, latitude_pitch: float = 0.0
) -> list[npt.NDArray[np.float64]]:
    """Points on a small circle of colatitude ``theta``, ``step_rad`` apart.

    ``latitude_pitch`` lifts the circle off the great circle by that many mesh
    pitches, which is how a path running parallel to a crest, slightly below it,
    is built. ``theta = pi/2`` is the great circle itself.
    """
    off = latitude_pitch * _PITCH
    sin_t = math.sin(theta)
    if abs(sin_t) < 1e-12:
        raise ValueError("theta must not be a pole")
    dphi = step_rad / sin_t
    out: list[npt.NDArray[np.float64]] = []
    for i in range(n):
        phi = i * dphi
        in_plane = math.cos(phi) * _X + math.sin(phi) * _Y
        p = math.sin(theta) * in_plane + math.cos(theta) * _Z
        out.append(_unit(math.cos(off) * p + math.sin(off) * _Z))
    return out


def _great_circle_path(
    n: int, step_pitch: float = 1.0
) -> list[npt.NDArray[np.float64]]:
    """A straight path: an arc of the equator, ``step_pitch`` pitches per step."""
    return _latitude_circle(math.pi / 2.0, n, step_pitch * _PITCH)


def _kinked_path(
    n_leg: int, turn_deg: float, step_pitch: float = 1.0
) -> tuple[list[npt.NDArray[np.float64]], int]:
    """Two straight legs meeting at a sharp corner. Returns (path, corner index).

    The first leg runs along the equator; at the apex the path turns ``turn_deg``
    in the tangent plane and continues on a great circle. The turn is a genuine
    kink in the AVERAGE direction, which is what a corner has to be.

    ``turn_deg`` stays well under 90 on purpose: at exactly 90 the second leg runs
    onto the pole and the fixture degenerates, so a passing test would prove
    nothing about a generic elbow.
    """
    step = step_pitch * _PITCH
    leg1 = _latitude_circle(math.pi / 2.0, n_leg, step)
    apex = leg1[-1]
    tangent = _unit(apex - leg1[-2])
    outward = _unit(np.cross(tangent, apex))
    dir2 = _unit(
        math.cos(math.radians(turn_deg)) * tangent
        + math.sin(math.radians(turn_deg)) * outward
    )
    out = list(leg1)
    for i in range(1, n_leg):
        lam = i * step
        out.append(_unit(np.asarray(rotate_towards(apex, dir2, lam, 1.0))))
    return out, n_leg - 1


def _bowed_then_kinked(
    theta: float, n1: int, n2: int, turn_deg: float, step_pitch: float = 1.0
) -> tuple[list[npt.NDArray[np.float64]], int]:
    """A smoothly curving path with ONE sharp kink in the middle.

    The bow is a latitude circle, which carries a large sustained deflection at
    every vertex. This is the case an absolute threshold cannot handle: the corner
    must be found ON TOP of a baseline that is already near the threshold.
    """
    step = step_pitch * _PITCH
    leg1 = _latitude_circle(theta, n1, step)
    apex = leg1[-1]
    tangent = _unit(apex - leg1[-2])
    outward = _unit(np.cross(tangent, apex))
    dir2 = _unit(
        math.cos(math.radians(turn_deg)) * tangent
        + math.sin(math.radians(turn_deg)) * outward
    )
    out = list(leg1)
    for i in range(1, n2):
        lam = i * step
        out.append(_unit(np.asarray(rotate_towards(apex, dir2, lam, 1.0))))
    return out, n1 - 1


def _sawtooth_path(
    n: int, amplitude_pitch: float = 0.5, step_pitch: float = 1.0
) -> list[npt.NDArray[np.float64]]:
    """A straight path with alternating cross-track teeth: chatter, not corners.

    Each vertex sits ``amplitude`` pitches off the great circle, sign alternating
    every step, so the point-to-point turning angle is enormous while the average
    direction over several steps is exactly straight.
    """
    step = step_pitch * _PITCH
    amp = amplitude_pitch * _PITCH
    out: list[npt.NDArray[np.float64]] = []
    for i in range(n):
        lam = i * step
        off = amp if i % 2 == 0 else -amp
        base = math.cos(lam) * _X + math.sin(lam) * _Y
        out.append(_unit(math.cos(off) * base + math.sin(off) * _Z))
    return out


def _ridge_field(
    pts: npt.NDArray[np.float64],
    pole: npt.NDArray[np.float64],
    width_rad: float,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    """A ridge: the field peaks on the great circle whose pole is ``pole``.

    Depends only on the distance to that circle, so the crest is the whole circle
    and the gradient ALONG it vanishes. That isolation is deliberate: a transverse
    nudge is the only motion this field permits, so a test built on it cannot pass
    by sliding along the ridge.
    """
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    cross = np.arcsin(np.clip(u @ pole, -1.0, 1.0))
    return height * np.exp(-((cross / width_rad) ** 2))


def _corner_field(
    pts: npt.NDArray[np.float64],
    pole_a: npt.NDArray[np.float64],
    pole_b: npt.NDArray[np.float64],
    width_rad: float,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Two ridges crossing at a corner; the field is the max of the two crests."""
    return np.maximum(
        _ridge_field(pts, pole_a, width_rad, height),
        _ridge_field(pts, pole_b, width_rad, height),
    )


def _as_world(
    path: list[npt.NDArray[np.float64]], center: tuple[float, float, float] | None
) -> list[tuple[float, float, float]]:
    """Centred unit radials scaled to R and lifted into the world frame."""
    out: list[tuple[float, float, float]] = []
    for u in path:
        pos = (float(u[0] * _R), float(u[1] * _R), float(u[2] * _R))
        out.append(to_global(pos, center) if center else pos)
    return out


def _lat_deg(u: npt.ArrayLike) -> float:
    """Latitude in degrees: the cross-track distance from the equator."""
    return math.degrees(
        math.asin(float(np.clip(float(np.dot(_unit(u), _Z)), -1.0, 1.0)))
    )


def _refine(
    path: list[npt.NDArray[np.float64]],
    f: npt.NDArray[np.float64],
    pts: npt.NDArray[np.float64],
    center: tuple[float, float, float] | None = None,
    **kwargs: Any,
) -> RidgeRefinement:
    """Refine a centred fixture, handling the frame conversion for the caller.

    The mesh is translated by ``center`` alongside the path, so the world-frame
    call exercises the same geometry as the centred one -- which is what
    :func:`test_refinement_is_frame_independent` compares.
    """
    mesh = pts if center is None else pts + np.array(center)
    return refine_gradient_ridge(
        arc_points=_as_world(path, center),
        mesh_pts=mesh,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# corner detection
# ---------------------------------------------------------------------------


def test_sawtooth_chatter_is_not_a_corner() -> None:
    """Teeth cancel over the window, and their deflection is uniform: baseline wins."""
    u = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    naive = chord_deflection_deg(u, 1)
    assert max(naive[1:-1]) > 60.0, "fixture must produce huge point-to-point turns"

    corners, deflect, w = detect_corners(u, _PITCH)
    assert w >= 2, "the window must span several steps for the teeth to cancel"
    assert corners == ()
    mid = np.array(deflect[w:-w])
    assert float(mid.max() - np.median(mid)) < CORNER_EXCESS_DEG, (
        "chatter must not rise above its own baseline"
    )


def test_sharp_corner_is_detected() -> None:
    """A 70 deg kink in the average direction promotes exactly one vertex, at the apex."""
    path, apex = _kinked_path(21, turn_deg=70.0)
    u = np.array(path)
    corners, _, _ = detect_corners(u, _PITCH)
    assert len(corners) == 1
    assert abs(corners[0] - apex) <= 1, "the knot must sit at the apex of the bend"


def test_sustained_curvature_is_not_a_corner() -> None:
    """The latitude circle: 56.7 deg of deflection at EVERY vertex, zero corners.

    This is the probe's measurement, pinned as a regression. A small circle of
    colatitude 0.20 rad turns 56.7 deg per window at every interior vertex -- far
    above the 35 deg threshold -- so an absolute threshold calls the entire arc a
    sequence of corners. The baseline test reports none, because the deflection
    never rises above the path's own median.
    """
    for theta in (0.20, 0.30, 0.45):
        u = np.array(_latitude_circle(theta, 61, _PITCH))
        corners, deflect, w = detect_corners(u, _PITCH)
        mid = np.array(deflect[w:-w])
        assert corners == (), f"theta={theta}: sustained curvature read as corners"
        assert float(mid.max() - np.median(mid)) < 1.0, "deflection must be uniform"

    for theta in (0.20, 0.30):
        u = np.array(_latitude_circle(theta, 61, _PITCH))
        _, deflect, w = detect_corners(u, _PITCH)
        assert float(max(deflect[w:-w])) > CORNER_EXCESS_DEG, (
            f"theta={theta}: fixture must stay a hard case, i.e. deflect a lot"
        )


def test_a_corner_on_a_curved_ridge_is_still_found() -> None:
    """The baseline test must not cost sensitivity: a kink on a bowed path is found.

    The bow carries a large sustained deflection at every vertex, which is already
    above the absolute threshold. The kink must be the ONE vertex that rises above
    that baseline. The turn is 90 deg, not 70: measured, a 70 deg kink on a
    0.30-rad bow raises the deflection only 25 deg above the baseline, correctly
    below the 35 deg threshold, because the windowed chord smooths a bend that is
    shallower than the curvature it sits on.
    """
    path, apex = _bowed_then_kinked(0.30, 21, 21, turn_deg=90.0)
    u = np.array(path)
    _, deflect, w = detect_corners(u, _PITCH)
    assert float(np.median(deflect[w:-w])) > 20.0, "the bow must carry a real baseline"
    corners, _, _ = detect_corners(u, _PITCH)
    assert len(corners) == 1
    assert abs(corners[0] - apex) <= 2


def test_window_scales_with_the_path_step() -> None:
    """A path sampled finer than the mesh gets more vertices in the same window.

    The window is a PHYSICAL width (several mesh pitches), because that is the
    scale the sawtooth lives on. Halving the vertex spacing must double the vertex
    count of the window, and the corner must still be found.
    """
    coarse, apex_c = _kinked_path(21, 70.0, step_pitch=1.0)
    fine, apex_f = _kinked_path(41, 70.0, step_pitch=0.5)
    uc, uf = np.array(coarse), np.array(fine)
    _, _, w_c = detect_corners(uc, _PITCH)
    corners_f, _, w_f = detect_corners(uf, _PITCH)
    assert w_f == 2 * w_c
    assert len(corners_f) == 1
    assert abs(corners_f[0] - apex_f) <= 2
    assert abs(detect_corners(uc, _PITCH)[0][0] - apex_c) <= 1


def test_too_short_a_path_has_no_corners() -> None:
    """Fewer vertices than a window needs: no corners, and no crash."""
    u = np.array(_great_circle_path(5))
    corners, deflect, _ = detect_corners(u, _PITCH)
    assert corners == ()
    assert len(deflect) in (0, len(u))


# ---------------------------------------------------------------------------
# transverse relaxation
# ---------------------------------------------------------------------------


def test_an_off_crest_path_settles_onto_the_ridge() -> None:
    """A path running parallel to the crest, one pitch below it, is pulled up."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    before = np.array([abs(_lat_deg(u)) for u in path])
    assert float(before[6:-6].mean()) > 2.0, "fixture must start off the crest"

    res = _refine(path, f, pts)
    assert res.changed
    after = np.array([abs(_lat_deg(_local(p, None))) for p in res.path])
    mid = slice(6, -6)
    assert float(after[mid].mean()) < 0.35 * float(before[mid].mean()), (
        f"interior latitude {after[mid].mean():.2f} deg should collapse toward 0 "
        f"from {before[mid].mean():.2f}"
    )


def test_the_taper_peaks_in_the_middle_of_a_segment() -> None:
    """Displacement is largest at the segment centre and zero at the pinned ends.

    This is the physical observation as an assertion: the crest is best defined
    mid-segment, where the cross-ridge slope is steep. Near the saddle the field is
    flat and near the maximum the along-contour variation is slow, so the ends are
    the least trustworthy place to move.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    world = _as_world(path, None)
    res = _refine(path, f, pts)
    shifts = [
        _angle_deg(_local(p, None), _local(q, None)) for p, q in zip(res.path, world)
    ]
    assert shifts[0] == 0.0 and shifts[-1] == 0.0
    n = len(path)
    middle = shifts[n // 3 : 2 * n // 3]
    ends = shifts[1:4] + shifts[-4:-1]
    assert float(np.mean(middle)) > 2.0 * float(np.mean(ends)), (
        f"the envelope must put the displacement in the middle: "
        f"middle {np.mean(middle):.2f} vs ends {np.mean(ends):.2f}"
    )
    # The envelope, not a peak index: sin(pi t) is flat to first order at t = 0.5,
    # so the largest single displacement lands anywhere in the middle third, and a
    # vertex whose window resolves no crest stays put by design. What must hold is
    # that the middle moves far more than the ends, as a distribution.
    assert float(np.median(middle)) > float(np.max(ends)), (
        f"the median middle vertex must move further than the most mobile end: "
        f"{np.median(middle):.2f} vs {np.max(ends):.2f}"
    )


def test_endpoints_never_move() -> None:
    """Dirichlet anchors: the launch port and the maximum are fixed to 1e-12."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    world = _as_world(path, None)
    res = _refine(path, f, pts)
    assert res.anchor_indices[0] == 0
    assert res.anchor_indices[-1] == len(path) - 1
    for i in (0, len(path) - 1):
        assert math.dist(res.path[i], world[i]) < 1e-12, f"endpoint {i} moved"


def test_the_field_rises_along_the_refined_path() -> None:
    """Non-vacuity: the refinement moves the path UP the field, not sideways."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    res = _refine(path, f, pts)
    before = interpolate_scalar_at_points(np.array(path[2:-2]), pts, f)
    after = interpolate_scalar_at_points(
        np.array([_local(p, None) * _R for p in res.path[2:-2]]), pts, f
    )
    assert float(np.mean(after)) > float(np.mean(before)) + 1e-4


# ---------------------------------------------------------------------------
# knots
# ---------------------------------------------------------------------------


def test_a_knot_stays_sharp_and_inside_its_disc() -> None:
    """The corner is preserved and the knot's shift is bounded.

    The knot is a 2D node: it may move within ``KNOT_MAX_PITCH`` to find the crest.
    The disc is the constraint that keeps the corner sharp -- a knot free to travel
    would slide to wherever the field is highest and drag both segments with it,
    which is the blunting the piecewise design exists to prevent.
    """
    pts = _fibonacci_sphere()
    f = _corner_field(pts, _Z, _X, width_rad=4.0 * _PITCH)
    path, _ = _kinked_path(21, turn_deg=70.0)
    u = np.array(path)
    corners, _, _ = detect_corners(u, _PITCH)
    assert len(corners) == 1

    res = _refine(path, f, pts)
    assert res.corner_indices == corners
    after = np.array([_local(p, None) for p in res.path])
    k = corners[0]
    turn_after = chord_deflection_deg(after, 1)[k]
    assert turn_after > CORNER_EXCESS_DEG, (
        f"corner blunted to {turn_after:.1f} deg, below the corner threshold"
    )
    assert _angle_deg(u[k], after[k]) <= KNOT_MAX_PITCH * math.degrees(_PITCH) + 1e-9


def test_a_corner_is_not_blunted_by_the_neighbouring_relaxation() -> None:
    """Interior relaxation must not leak across a knot.

    The taper vanishes at the segment ends, so the vertices flanking a corner
    barely move and the elbow stays an elbow. A global smoothing pass would round
    it, which is the failure the piecewise design exists to prevent.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path, _ = _kinked_path(21, turn_deg=70.0)
    u = np.array(path)
    res = _refine(path, f, pts)
    assert len(res.corner_indices) == 1
    after = np.array([_local(p, None) for p in res.path])
    k = res.corner_indices[0]
    for i in (k - 1, k + 1):
        assert _angle_deg(u[i], after[i]) < 1.0, f"vertex {i} next to the knot moved"
    assert chord_deflection_deg(after, 1)[k] > 45.0


def test_a_transited_excision_centre_is_pinned() -> None:
    """The path is knotted to the centre of every region it was carried across.

    A transit is a straight line over ground the field cannot speak for, so the
    centre is a position the pipeline vouches for by construction. The refinement
    must not move it.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    centre = _as_world([path[15]], None)[0]
    res = _refine(path, f, pts, transited_centres=[centre])
    assert 15 in res.anchor_indices
    assert math.dist(res.path[15], centre) < 1e-12
    assert res.changed, "the rest of the path must still relax"


# ---------------------------------------------------------------------------
# frame contract and degenerate inputs
# ---------------------------------------------------------------------------


def test_refinement_is_frame_independent() -> None:
    """Refining through an off-origin centre equals refining in the centred frame.

    The refiner converts world positions to centred unit radials, works there, and
    converts back. If any step used the world origin as the sphere centre the two
    results would differ -- and Pd_20K, whose atom sits at the origin, would not
    show it.

    ``mesh_pts`` stays CENTRED in both calls, which is the documented contract: the
    IDW kernel normalises the mesh vertices themselves, so translating the mesh
    would change the interpolant and compare two different fields.
    """
    center = (1.2618, -0.0001, -0.0002)
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)

    centred = _refine(path, f, pts)
    world = refine_gradient_ridge(
        arc_points=_as_world(path, center),
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
    )
    assert centred.corner_indices == world.corner_indices
    a = np.array([_local(p, None) for p in centred.path])
    b = np.array([_local(p, center) for p in world.path])
    assert float(np.max(np.linalg.norm(a - b, axis=1))) < 1e-9


def test_a_path_with_no_interior_is_returned_unchanged() -> None:
    """Every vertex pinned: nothing to relax, and the caller is told so."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    world = _as_world(path, None)
    res = _refine(path, f, pts, transited_centres=world)
    assert not res.changed
    assert res.path == tuple(world)


def test_a_path_too_short_to_refine_is_returned_unchanged() -> None:
    """The caller may hand every connector, including a two-point geodesic."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    short = [(0.0, 0.0, _R), (_R, 0.0, 0.0)]
    res = refine_gradient_ridge(
        arc_points=short,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert res.path == tuple(short)
    assert not res.changed
    assert res.max_shift_deg == 0.0


def test_an_on_crest_path_barely_moves() -> None:
    """A path already on the ridge stays on it: the residual is the noise floor.

    Non-vacuity for the tests above, which rely on the offset being the reason
    anything moves. The assertion is a BOUND, not exactness: the field arrives
    through IDW, whose interpolant ripples at the mesh scale, so the crest estimate
    on a crest has a measured spread of ~0.1 pitch (median 0.06-0.09 pitch across
    ridge widths) against the 1.0 pitch recovered from an off-crest path. Demanding
    a shift of zero would demand an exact fit to a noisy interpolant, and the
    honest statement of what this estimator guarantees is the ratio: the residual
    must be small next to the signal it recovers.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _great_circle_path(31)
    res = _refine(path, f, pts)
    pitch_deg = math.degrees(_PITCH)
    assert res.max_shift_deg < 0.25 * pitch_deg, (
        f"on-crest drift {res.max_shift_deg:.2f} deg exceeds a quarter pitch"
    )
    assert res.mean_shift_deg < 0.07 * pitch_deg

    # The same estimator, given a real offset, recovers it: the bound above is a
    # noise floor, not a dead mechanism.
    off = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    res_off = _refine(off, f, pts)
    assert res_off.mean_shift_deg > 8.0 * res.mean_shift_deg


def test_constants_are_used_as_defaults() -> None:
    """The published constants are the defaults, so tuning them tunes the pipeline.

    Two halves: naming the constants explicitly must reproduce the default call
    exactly (they are the defaults, not a coincidence), and narrowing the search
    must reduce the displacement (the constants are wired to the mechanism, not
    inert). The second is stated as a strict inequality in the direction the
    geometry forces: a window half as wide cannot reach as far across the ridge.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    default = _refine(path, f, pts)
    explicit = _refine(
        path,
        f,
        pts,
        window_pitch=CORNER_WINDOW_PITCH,
        excess_deg=CORNER_EXCESS_DEG,
        nudge_max_pitch=NUDGE_MAX_PITCH,
        knot_max_pitch=KNOT_MAX_PITCH,
    )
    assert explicit.path == default.path
    assert explicit.corner_indices == default.corner_indices

    narrow = _refine(path, f, pts, nudge_max_pitch=0.5 * NUDGE_MAX_PITCH)
    assert narrow.max_shift_deg < default.max_shift_deg
    assert default.max_shift_deg > 0.0


# ---------------------------------------------------------------------------
# displacement smoothing
# ---------------------------------------------------------------------------


def _rippled_ridge_field(
    pts: npt.NDArray[np.float64],
    width_rad: float,
    amp: float,
    period_pitch: float,
) -> npt.NDArray[np.float64]:
    """A ridge carrying a mesh-scale ripple: what IDW interpolation actually is.

    The ripple is azimuthal, so it moves the apparent crest at every vertex by a
    different amount, which is the property that defeats a per-vertex estimate.
    ``amp`` is small in absolute terms (1% of a peak of height 1) and that is the
    point: the noise does not have to be large to dominate a correction of the
    same size.
    """
    base = _ridge_field(pts, _Z, width_rad)
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    az = np.arctan2(u @ _Y, u @ _X)
    return np.asarray(base + amp * np.sin(2.0 * math.pi * az / (period_pitch * _PITCH)))


def _turn_sum(u: npt.NDArray[np.float64]) -> float:
    """Total point-to-point turning of a polyline, degrees. The roughness measure."""
    total = 0.0
    for i in range(1, len(u) - 1):
        t = u[i]
        a = u[i] - u[i - 1]
        b = u[i + 1] - u[i]
        a = a - float(np.dot(a, t)) * t
        b = b - float(np.dot(b, t)) * t
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-15 or nb < 1e-15:
            continue
        total += math.degrees(
            math.acos(float(np.clip(float(np.dot(a / na, b / nb)), -1, 1)))
        )
    return total


def test_the_displacement_is_fitted_not_applied_vertex_by_vertex() -> None:
    """On a rippled field, applying per-vertex crest estimates makes it ROUGHER.

    This is the regression the real field taught us. On ethene3 curvedness the
    raw per-vertex version took the point-to-point turns above 35 deg, away from
    the corner, from 6 to 15 and the total turning from 628 to 1374 deg, for a
    field gain of 0.03% of span -- it bought noise. The crest offset of a ridge
    from a path is a smooth function of position along the path, so the estimates
    are fitted along the segment before any of them is applied, which averages the
    noise ALONG the path exactly as the parabola fit averages it ACROSS the ridge.

    Measured on this fixture: the raw fit multiplies the total turning by ~155,
    the smoothed one by ~5, and the smoothed path sits CLOSER to the crest.
    """
    pts = _fibonacci_sphere()
    f = _rippled_ridge_field(pts, 4.0 * _PITCH, amp=0.008, period_pitch=2.0)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    u0 = np.array(path)
    base_turn = _turn_sum(u0)
    assert base_turn < 30.0, (
        "the input path is straight; roughness must start near zero"
    )

    raw = _refine(path, f, pts, fit_degree=-1)
    smooth = _refine(path, f, pts)
    u_raw = np.array([_local(p, None) for p in raw.path])
    u_sm = np.array([_local(p, None) for p in smooth.path])

    assert _turn_sum(u_raw) > 20.0 * base_turn, (
        f"fixture must reproduce the failure: raw gave {_turn_sum(u_raw):.0f} "
        f"vs base {base_turn:.0f}"
    )
    assert _turn_sum(u_sm) < 0.1 * _turn_sum(u_raw), (
        f"the fit must remove the injected roughness: {_turn_sum(u_sm):.0f} vs "
        f"{_turn_sum(u_raw):.0f}"
    )
    # And it must not buy smoothness by doing nothing.
    assert smooth.mean_shift_deg > 1.0
    f_raw = float(_ridge_field(u_raw, _Z, 4.0 * _PITCH).mean())
    f_sm = float(_ridge_field(u_sm, _Z, 4.0 * _PITCH).mean())
    f_base = float(_ridge_field(u0, _Z, 4.0 * _PITCH).mean())
    assert f_sm > f_base, "the refinement must still climb toward the crest"
    assert f_sm >= f_raw - 1e-9, (
        f"smoothing must not cost accuracy: {f_sm:.4f} vs raw {f_raw:.4f}"
    )


def test_fit_degree_zero_shifts_the_segment_rigidly() -> None:
    """A constant fit moves the whole segment by the average offset.

    The degree is a real control, not decoration: degree 0 is the mean of the
    per-vertex estimates, so it must land between the extremes and be smoother
    still. This pins the wiring of the parameter.
    """
    pts = _fibonacci_sphere()
    f = _rippled_ridge_field(pts, 4.0 * _PITCH, amp=0.008, period_pitch=2.0)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    const = _refine(path, f, pts, fit_degree=0)
    quad = _refine(path, f, pts, fit_degree=2)
    u_c = np.array([_local(p, None) for p in const.path])
    u_q = np.array([_local(p, None) for p in quad.path])
    assert const.mean_shift_deg > 1.0
    # The ordering is the assertion. A constant offset under a sin taper is a
    # smooth bulge, and a bulge has turning, so the total is legitimately larger
    # than the straight input path's; what must hold is that fewer degrees of
    # freedom give a smoother result, and that both beat the raw estimates.
    assert _turn_sum(u_c) <= _turn_sum(u_q) + 1e-9, (
        "a constant fit cannot be rougher than a quadratic one"
    )
    assert _turn_sum(u_q) < 0.1 * _turn_sum(
        np.array([_local(p, None) for p in _refine(path, f, pts, fit_degree=-1).path])
    )


# ---------------------------------------------------------------------------
# ridge probe
# ---------------------------------------------------------------------------


def test_probe_steps_an_off_crest_path_onto_the_crest() -> None:
    """The probe finds the crest half a pitch away, from the field's curvature."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.5)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    accepted = [s for s in probe.samples if s.point.accepted]
    assert accepted, f"nothing accepted: {[s.point.reason for s in probe.samples]}"
    # The step is the full offset, recovered from the Hessian, not from the path.
    assert 0.3 < probe.median_offset_pitch < 0.7, (
        f"median offset {probe.median_offset_pitch:.3f} pitch, expected ~0.5"
    )
    # And it lands on the crest: latitude ~0, against the path's ~1.5 deg.
    lat = [abs(_lat_deg(s.point.position_unit)) for s in accepted]
    assert max(lat) < 0.5, f"ridge points off the crest: max {max(lat):.2f} deg"
    # The rise is real and published: over 0.5% of the field's span.
    assert min(s.point.gain_fraction for s in accepted) > 0.005


def test_probe_refuses_a_crest_beyond_the_measured_reach() -> None:
    """A crest further than RIDGE_REACH_PITCH is a DIFFERENT ridge: refused.

    The bound is measured, not chosen -- see RIDGE_REACH_PITCH. This test is what
    keeps it a bound: a path 1.5 pitches off the crest must be told "out of
    reach", not quietly stepped two thirds of the way there. The probe reports a
    correction, never a re-routing.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.5)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    assert probe.samples, "the path is long enough to sample"
    assert probe.n_accepted == 0, (
        f"a 1.5-pitch offset was stepped to: {probe.median_offset_pitch} pitch"
    )
    reasons = {s.point.reason for s in probe.samples}
    assert "out_of_reach" in reasons, f"refused for the wrong reason: {reasons}"


def test_probe_points_ignore_the_sawtooth() -> None:
    """THE test the refinement suite lacked: the ridge is the crest, not the teeth.

    A sawtooth path crosses the crest every step, so its vertices swing +/-1.5 deg
    in latitude. The probe's accepted points must sit on the crest, not follow the
    zigzag. This is the property the Entry-058 suite never asserted, which is why
    every test in it passed on a refinement that moved nothing.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _sawtooth_path(41, amplitude_pitch=0.5)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    accepted = [s for s in probe.samples if s.point.accepted]
    assert len(accepted) >= 3, "too few accepted to judge the crest"
    lat = [abs(_lat_deg(s.point.position_unit)) for s in accepted]
    assert max(lat) < 0.5, (
        f"probe points followed the teeth: max |lat| {max(lat):.2f} deg "
        f"(teeth swing ~1.5 deg)"
    )
    # The path's own vertices at the sample positions are off the crest; the probe
    # points are not. Non-vacuity: the test measures the correction, not a flat line.
    from_lat = [abs(_lat_deg(s.from_unit)) for s in accepted]
    assert max(from_lat) > 3.0 * max(lat)


def test_probe_marks_only_the_corner_not_the_endpoints() -> None:
    """Knots are the sharp continuous bends. The launch port and maximum are pinned
    for the segment structure but are not knots, and teeth are never knots."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path, apex = _kinked_path(21, turn_deg=70.0)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    assert len(probe.knot_indices) == 1
    assert abs(probe.knot_indices[0] - apex) <= 1
    assert 0 not in probe.knot_indices
    assert len(path) - 1 not in probe.knot_indices
    # The knot position is the world-frame copy of that vertex.
    assert len(probe.knot_positions_world) == 1
    assert (
        _angle_deg(
            _local(probe.knot_positions_world[0], None), path[probe.knot_indices[0]]
        )
        < 1e-9
    )

    saw = _sawtooth_path(41, amplitude_pitch=0.5)
    probe_saw = probe_gradient_ridge(
        _as_world(saw, None), pts, f, _R, _PITCH, center=None
    )
    assert probe_saw.knot_indices == (), "sawtooth teeth promoted to knots"


def test_probe_samples_are_evenly_spaced_along_the_path() -> None:
    """Samples sit SEGMENT_SAMPLE_DEG apart in arc length, endpoints excluded."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    assert len(probe.samples) >= 5
    alongs = [s.along_deg for s in probe.samples]
    assert alongs == sorted(alongs), "samples out of path order"
    gaps = [b - a for a, b in pairwise(alongs)]
    assert all(abs(g - SEGMENT_SAMPLE_DEG) < 1.0 for g in gaps), (
        f"uneven spacing: {[f'{g:.1f}' for g in gaps]}"
    )
    assert alongs[0] > 0.0 and alongs[-1] < alongs[0] + 90.0


def test_probe_rejects_a_valley() -> None:
    """A field that curves the wrong way has no ridge to step onto. Refused, with
    the reason recorded -- not silently returned as 'the path is already right'."""
    pts = _fibonacci_sphere()
    # A trough along the equator: f is MINIMISED at latitude 0, so the Hessian's
    # most-negative direction is not across a crest.
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    cross = np.arcsin(np.clip(u @ _Z, -1.0, 1.0))
    f = np.exp(-((cross / (4.0 * _PITCH)) ** 2))
    f = f.max() - f  # invert: now a valley
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=1.0)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    reasons = {s.point.reason for s in probe.samples}
    assert "no_ridge" in reasons, f"valley not refused as no_ridge: {reasons}"


def test_probe_refusals_are_recorded_not_dropped() -> None:
    """Every sample appears in the output, accepted or not, with its reason.

    A probe that silently omitted the hard cases would report a clean ridge on a
    field that has one only where it is easy. The refusals are the evidence about
    where the criterion does not apply.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _sawtooth_path(41, amplitude_pitch=0.5)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    assert probe.samples
    assert all(s.point.reason for s in probe.samples)
    assert probe.n_accepted == sum(1 for s in probe.samples if s.point.accepted)
    assert probe.n_accepted < len(probe.samples), (
        "fixture must produce refusals for this test to mean anything"
    )


def test_probe_never_moves_the_path() -> None:
    """The probe is a measurement. It has no path output, and the input is intact."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.5)
    world = _as_world(path, None)
    snapshot = [tuple(p) for p in world]
    probe_gradient_ridge(world, pts, f, _R, _PITCH, center=None)
    assert [tuple(p) for p in world] == snapshot, "the probe mutated its input"

    # The stronger property: each sample's FROM position lies on the traced path and
    # its ridge point does not. A probe that quietly reported the ridge as the path
    # would pass every other test in this file.
    probe = probe_gradient_ridge(world, pts, f, _R, _PITCH, center=None)
    accepted = [s for s in probe.samples if s.point.accepted]
    assert accepted
    u_path = np.array([_unit(p) for p in path])

    def dist_to_path(p: npt.NDArray[np.float64]) -> float:
        """Angular distance from ``p`` to the polyline, not to its vertex set.

        Samples are placed BETWEEN vertices, so a vertex-set distance is half a
        step by construction and measures the sampling, not the geometry.
        """
        best = math.inf
        for a, b in pairwise(u_path):
            n = _unit(np.cross(a, b))
            if n is None:
                best = min(best, angular_distance(p, a))
                continue
            # Inside the segment: distance to the great circle it lies on.
            s_ab = angular_distance(a, b)
            s_pa = angular_distance(p, a)
            s_pb = angular_distance(p, b)
            if s_pa + s_pb <= s_ab * 1.000001:
                best = min(
                    best, abs(math.asin(float(np.clip(float(np.dot(p, n)), -1.0, 1.0))))
                )
            else:
                best = min(best, s_pa, s_pb)
        return best

    on = [dist_to_path(s.from_unit) for s in accepted]
    off = [dist_to_path(s.point.position_unit) for s in accepted]
    assert max(on) < 0.2 * _PITCH, "a sample was not on the path it probes"
    assert min(off) > 0.5 * max(on), (
        f"ridge points sit on the path: the probe corrected nothing "
        f"(max on-path {max(on) / _PITCH:.3f} pitch, min off {min(off) / _PITCH:.3f})"
    )


def test_probe_is_frame_independent() -> None:
    """Probing through an off-origin centre equals probing in the centred frame."""
    center = (1.2618, -0.0001, -0.0002)
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.5)
    centred = probe_gradient_ridge(
        _as_world(path, None), pts, f, _R, _PITCH, center=None
    )
    assert centred.n_accepted > 0, "fixture must produce accepted samples"
    world = probe_gradient_ridge(
        _as_world(path, center), pts, f, _R, _PITCH, center=center
    )
    assert centred.knot_indices == world.knot_indices
    assert len(centred.samples) == len(world.samples)
    # Both probes report CENTRED unit radials, so they compare directly: the frame
    # conversion is the probe's internal business, and a step that used the world
    # origin as the sphere centre would show up here as a disagreement. The
    # tolerance is float round-off through the translation (1e-6 deg on a 1.2618 A
    # offset), not a property of the algorithm.
    for a, b in zip(centred.samples, world.samples):
        assert _angle_deg(a.point.position_unit, b.point.position_unit) < 1e-4
        assert _angle_deg(a.from_unit, b.from_unit) < 1e-4
        assert a.point.accepted == b.point.accepted


def test_probe_on_a_short_path_is_empty() -> None:
    """A caller may hand every connector, including a two-point geodesic."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    probe = probe_gradient_ridge(
        [(0.0, 0.0, _R), (_R, 0.0, 0.0)], pts, f, _R, _PITCH, center=None
    )
    assert probe.samples == ()
    assert probe.knot_indices == ()
    assert probe.n_accepted == 0


def test_scan_finds_crests_on_both_sides_of_the_sample() -> None:
    """The two-sided scan is the mechanism that answers "is the ridge the other way?".

    A field with two crests along one line, +1.5 pitch (height 0.90) and -1.6 pitch
    (height 1.00). The scan must return BOTH, with the correct signs, and report the
    taller one as the higher value -- which is what the probe publishes as
    ``other_side``, so the operator's question is answered per sample by the shipped
    tool.

    Tested on the scan directly, not through ``ridge_point_at``: the selection rule
    (Newton steps to the NEAR crest, the reach refuses the far one) is covered by
    :func:`test_probe_steps_an_off_crest_path_onto_the_crest` and
    :func:`test_probe_refuses_a_crest_beyond_the_measured_reach`. This covers the
    other half: that the far side is SEEN at all.

    The mesh is 8000 points, not the suite's 1200: at 1200 the vertex spacing is 2.8
    pitches, so a crest 1.2 pitches wide is not representable in the interpolant and
    the scan correctly sees a monotone slope. A fixture cannot resolve structure
    finer than the mesh that carries it.
    """
    pts = _fibonacci_sphere(8000)
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    lat = np.arcsin(np.clip(u @ _Z, -1.0, 1.0))
    width = 1.2 * _PITCH
    near = 0.90 * np.exp(-(((lat - 1.5 * _PITCH) / width) ** 2))
    far = 1.00 * np.exp(-(((lat + 1.6 * _PITCH) / width) ** 2))
    f = np.maximum(near, far)

    crests = _scan_crests(_X, _Z, 2.5 * _PITCH, 0.25 * _PITCH, _R, pts, f, None, 0.02)
    assert len(crests) == 2, f"expected two crests, got {crests}"
    neg = [c for c in crests if c[0] < 0.0]
    pos = [c for c in crests if c[0] > 0.0]
    assert len(neg) == 1 and len(pos) == 1, f"crests not on both sides: {crests}"
    assert 1.2 < pos[0][0] / _PITCH < 1.8, (
        f"near crest misplaced: {pos[0][0] / _PITCH:.2f}"
    )
    assert -2.0 < neg[0][0] / _PITCH < -1.2, (
        f"far crest misplaced: {neg[0][0] / _PITCH:.2f}"
    )
    assert neg[0][1] > pos[0][1], (
        f"the taller crest was not the higher value: {neg[0][1]:.3f} vs {pos[0][1]:.3f}"
    )
