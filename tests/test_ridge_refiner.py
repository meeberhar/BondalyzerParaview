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
import pytest

from gba_topology2.src.ridge_refiner import (
    CORNER_EXCESS_DEG,
    NODE_PITCH,
    SEGMENT_SAMPLE_DEG,
    RidgeRefinement,
    _max_deviation_deg,
    _scan_crests,
    chord_deflection_deg,
    coarse_structural_curve,
    detect_chatter_runs,
    detect_corners,
    point_turn_deg,
    probe_gradient_ridge,
    smooth_gradient_ridge,
    smooth_sawtooth_runs,
    structural_curve_ridge,
)
from gba_topology2.src.sphere_frame import (
    angular_distance,
    rotate_towards,
    slerp,
    sphere_exp,
    sphere_log,
    squad_control,
    squad_point,
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


def _sawtooth_with_corner(
    n_leg: int,
    turn_deg: float,
    amplitude_pitch: float = 0.5,
    step_pitch: float = 1.0,
) -> tuple[list[npt.NDArray[np.float64]], int]:
    """A sawtooth on both legs of a sharp corner. Returns (path, corner index).

    The harshest fixture for the smoother: chatter everywhere AND a real bend that
    must survive. The caller passes the corner index as a knot; the smoother must
    remove the teeth on each leg while keeping the elbow sharp and passing exactly
    through the apex.
    """
    step = step_pitch * _PITCH
    amp = amplitude_pitch * _PITCH
    leg1 = _sawtooth_path(n_leg, amplitude_pitch, step_pitch)
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
        off = amp if i % 2 == 0 else -amp
        base = np.asarray(rotate_towards(apex, dir2, lam, 1.0))
        bu = _unit(base)
        perp = _unit(np.cross(bu, dir2))
        out.append(_unit(bu + off * perp))
    return out, n_leg - 1


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


def _smooth_world(
    path: list[npt.NDArray[np.float64]],
    knots: tuple[int, ...] = (),
    center: tuple[float, float, float] | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], RidgeRefinement]:
    """Smooth a centred fixture through the world-frame wrapper, back to radials.

    The path is lifted to the world frame (translated by ``center``) and handed to
    :func:`smooth_gradient_ridge`, which converts back to centred radials internally.
    The result is converted to centred unit radials again so a caller can compare it
    against the centred-frame smoothing vertex for vertex -- the frame contract the
    whole module rests on, invisible on a centred atom and so tested explicitly.

    Returns:
        ``(path_unit, refinement)``: the smoothed curve as centred unit radials, and
        the raw result with its metrics.
    """
    res = smooth_gradient_ridge(
        arc_points=_as_world(path, center),
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
        **kwargs,
    )
    return np.array([_local(p, center) for p in res.path]), res


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


# ---------------------------------------------------------------------------
# geodesic interpolation primitives (sphere_frame)
# ---------------------------------------------------------------------------


def test_sphere_log_exp_round_trip() -> None:
    """exp(log(a->b)) lands on b, and the log's norm is the central angle."""
    a = _X
    b = _unit(_X + _Y)  # 45 deg away
    lg = sphere_log(a, b)
    assert lg is not None
    assert math.degrees(float(np.linalg.norm(lg))) == pytest.approx(45.0, abs=1e-9)
    back = sphere_exp(a, lg)
    assert back is not None
    assert _angle_deg(back, b) < 1e-9


def test_sphere_log_degenerate_cases() -> None:
    """Coincident and antipodal points have no unique log; both return None."""
    assert sphere_log(_X, _X) is None
    assert sphere_log(_X, -_X) is None
    assert slerp(_X, -_X, 0.5) is None


def test_squad_interpolates_endpoints_and_is_geodesic_exact() -> None:
    """SQUAD passes through its knots exactly and adds no bulge to a great circle."""
    n = 9
    pts = np.array(
        [_unit(math.cos(i * 0.2) * _X + math.sin(i * 0.2) * _Y) for i in range(n)]
    )
    for i in range(n - 1):
        qa = squad_control(pts[max(i - 2, 0)], pts[i], pts[i + 1])
        qb = squad_control(pts[i], pts[i + 1], pts[min(i + 2, n - 1)])
        assert qa is not None and qb is not None
        a = squad_point(pts[i], qa, qb, pts[i + 1], 0.0)
        b = squad_point(pts[i], qa, qb, pts[i + 1], 1.0)
        assert a is not None and b is not None
        assert _angle_deg(a, pts[i]) < 1e-9, "SQUAD must interpolate its start knot"
        assert _angle_deg(b, pts[i + 1]) < 1e-9, "SQUAD must interpolate its end knot"
        # A geodesic's control points lie on it: no point leaves the equator.
        for t in (0.25, 0.5, 0.75):
            p = squad_point(pts[i], qa, qb, pts[i + 1], t)
            assert p is not None
            assert abs(_lat_deg(p)) < 1e-9, "a straight line must not bulge"


# ---------------------------------------------------------------------------
# chatter detection
# ---------------------------------------------------------------------------


def test_chatter_runs_found_on_sawtooth_not_on_smooth_paths() -> None:
    """A sawtooth is one long run; a great circle, a latitude arc, a clean kink are none."""
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    runs = detect_chatter_runs(saw, (), _PITCH)
    assert runs, "a pure sawtooth must be detected as chatter"
    lo, hi = runs[0]
    assert lo <= 2 and hi >= len(saw) - 3, "the run should span the teeth"

    straight = np.array(_great_circle_path(41))
    assert detect_chatter_runs(straight, (), _PITCH) == ()

    bowed = np.array(_latitude_circle(0.20, 41, _PITCH))
    assert detect_chatter_runs(bowed, (), _PITCH) == (), (
        "sustained curvature is not chatter"
    )

    kink, apex = _kinked_path(21, turn_deg=70.0)
    assert detect_chatter_runs(np.array(kink), (apex,), _PITCH) == (), (
        "a real corner is not chatter"
    )


def test_a_knot_is_never_a_tooth() -> None:
    """Passing the corner index as a knot excludes it from the tooth set."""
    path, apex = _sawtooth_with_corner(25, turn_deg=70.0)
    u = np.array(path)
    runs = detect_chatter_runs(u, (apex,), _PITCH)
    for lo, hi in runs:
        assert not (lo <= apex <= hi), "a knot fell inside a chatter run"


# ---------------------------------------------------------------------------
# smooth_sawtooth_runs
# ---------------------------------------------------------------------------


def test_smoothing_removes_the_teeth() -> None:
    """THE property the Entry-058 suite lacked: the total turning collapses.

    A sawtooth pays a huge point-to-point turn at every tooth, so a path that
    travels straight across the crest accumulates thousands of degrees of turning
    the ridge does not have. Smoothing removes the teeth, and the total turning
    drops by an order of magnitude.
    """
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    res = smooth_sawtooth_runs(saw, (), _PITCH)
    assert res.changed
    assert res.total_turn_before_deg > 1000.0, "fixture must be chattering"
    assert res.total_turn_after_deg < 0.15 * res.total_turn_before_deg, (
        f"teeth survived: {res.total_turn_before_deg:.0f} -> "
        f"{res.total_turn_after_deg:.0f} deg"
    )


def test_smoothed_sawtooth_sits_on_the_midline() -> None:
    """The curve follows the midline the teeth oscillate about, not a tooth.

    This is the defect the pure-sawtooth fixture catches: pinning the curve to the
    run's entry/exit VERTEX lands it on a tooth (latitude ~1.5 deg), a whole tooth
    off the crest. Anchoring at the bounding-edge MIDPOINT puts it on the midline.
    """
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    res = smooth_sawtooth_runs(saw, (), _PITCH)
    # Interior of the curve, away from the two pinned endpoint teeth.
    interior = res.path_unit[3:-3]
    lat = [abs(_lat_deg(p)) for p in interior]
    assert max(lat) < 0.5, (
        f"curve a tooth off the midline: max |lat| {max(lat):.2f} deg "
        f"(teeth swing ~1.5 deg)"
    )


def test_smoothing_preserves_endpoints_and_anchors() -> None:
    """The launch port and the maximum are physical positions; the curve keeps them."""
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    res = smooth_sawtooth_runs(saw, (), _PITCH)
    assert _angle_deg(res.path_unit[0], saw[0]) < 1e-9
    assert _angle_deg(res.path_unit[-1], saw[-1]) < 1e-9
    # Every knot the caller pinned is an anchor the curve passes through.
    path, apex = _sawtooth_with_corner(25, turn_deg=70.0)
    u = np.array(path)
    res2 = smooth_sawtooth_runs(u, (apex,), _PITCH)
    assert float(apex) in res2.anchors, "the knot is not an anchor"
    dists = [_angle_deg(q, u[apex]) for q in res2.path_unit]
    assert min(dists) < 1e-6, "the curve does not pass through the knot"


def test_smoothing_keeps_a_real_corner_sharp() -> None:
    """A knot embedded in chatter: the teeth go, the elbow stays.

    The apex is pinned, so the curve is free to kink there (no tangent continuity
    across a knot). The output must carry a sharp turn at the knot -- the opposite
    of smoothing the corner away.
    """
    path, apex = _sawtooth_with_corner(25, turn_deg=70.0)
    u = np.array(path)
    res = smooth_sawtooth_runs(u, (apex,), _PITCH)
    assert res.changed
    # Find the output vertex at the knot and require a large turn there.
    dists = [_angle_deg(q, u[apex]) for q in res.path_unit]
    i = int(np.argmin(dists))
    turns = point_turn_deg(res.path_unit)
    assert turns[i] > 30.0, f"the corner was blunted: turn at knot {turns[i]:.1f} deg"


def test_smoothing_leaves_a_smooth_path_untouched() -> None:
    """No chatter -> the input is returned verbatim, changed False.

    A great circle, a latitude arc, and a clean kink carry no teeth, so the
    smoother must not resample, rescale, or nudge them. This is the non-vacuity
    guard: the previous tests all move geometry, this one proves the detector's
    negatives are respected.
    """
    for name, path in [
        ("straight", _great_circle_path(41)),
        ("bowed", _latitude_circle(0.20, 41, _PITCH)),
    ]:
        u = np.array(path)
        res = smooth_sawtooth_runs(u, (), _PITCH)
        assert not res.changed, f"{name} path was modified"
        assert res.path_unit.shape == u.shape
        assert np.allclose(res.path_unit, u)


def test_smoothed_output_is_finer_than_the_mesh() -> None:
    """Runs are emitted at SMOOTH_OUTPUT_PITCH spacing: more, closer vertices.

    The point of fixed arc spacing is that the smooth line is visibly finer than
    the traced one it replaces. A 41-vertex sawtooth (one pitch per step) becomes a
    curve with more vertices spaced 0.45 pitch apart.
    """
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    res = smooth_sawtooth_runs(saw, (), _PITCH)
    assert len(res.path_unit) > len(saw), (
        f"resampling did not refine: {len(saw)} -> {len(res.path_unit)}"
    )
    # The interior spacing is at most the requested pitch (plus the geodesic
    # discretisation of the arc), never the full mesh pitch of the teeth.
    steps = [
        math.degrees(angular_distance(res.path_unit[i], res.path_unit[i + 1]))
        for i in range(3, len(res.path_unit) - 4)
    ]
    assert max(steps) < math.degrees(_PITCH), (
        f"output coarser than the mesh: max step {max(steps):.2f} deg"
    )


def test_smoothing_is_deterministic() -> None:
    """The same path in, the same curve out. Pure geometry, no field, no RNG."""
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    a = smooth_sawtooth_runs(saw, (), _PITCH)
    b = smooth_sawtooth_runs(saw, (), _PITCH)
    assert a.path_unit.shape == b.path_unit.shape
    assert np.allclose(a.path_unit, b.path_unit)


def test_smoothing_reports_its_cost() -> None:
    """max_deviation_deg is the largest angular distance from the curve to the path.

    Published, not hidden -- the same discipline as ``completed_leg_deg``. The
    curve must stay within a tooth of the traced path (it is smoothing the teeth,
    not re-routing the leg), and the metric must be a real positive number.
    """
    saw = np.array(_sawtooth_path(41, amplitude_pitch=0.5))
    res = smooth_sawtooth_runs(saw, (), _PITCH)
    assert res.max_deviation_deg > 0.0
    # A tooth is ~1.5 deg of latitude; the curve stays within about a tooth of the
    # zigzag it replaces.
    assert res.max_deviation_deg < 3.0, (
        f"curve strayed far from the traced path: {res.max_deviation_deg:.2f} deg"
    )


def test_smoothing_a_short_path_is_a_no_op() -> None:
    """Below _MIN_PATH vertices there is nothing to smooth; the input is returned."""
    short = np.array(_great_circle_path(4))
    res = smooth_sawtooth_runs(short, (), _PITCH)
    assert not res.changed
    assert np.allclose(res.path_unit, short)


def test_smoothing_is_frame_independent() -> None:
    """Smoothing through an off-origin centre gives the centred frame's geometry.

    The wrapper converts world -> centred radials, smooths, and scales back. For an
    atom at the origin the two frames coincide, which is why every frame bug in this
    project has been invisible on Pd_20K; ethene3's centre is 1.2618 A off the origin
    and can see it. The mesh is irrelevant here -- the smoother samples no field --
    so only the path and the centre are moved together.
    """
    center = (1.2618, -0.4407, 0.3311)
    saw = _sawtooth_path(41, amplitude_pitch=0.5)
    centred = smooth_sawtooth_runs(np.array(saw), (), _PITCH)
    lifted, _ = _smooth_world(saw, center=center)
    assert centred.changed and len(lifted) == len(centred.path_unit), (
        "the two frames produced different amounts of work"
    )
    dev = max(_angle_deg(a, b) for a, b in zip(lifted, centred.path_unit, strict=True))
    assert dev < 1e-6, f"frame-dependent geometry: max deviation {dev:.2e} deg"


def test_smoothing_preserves_a_transited_centre() -> None:
    """A vertex pinned as an excision transit stays exactly where it was.

    The transit is a straight carry across ground the field cannot speak for, so the
    pipeline vouches for that position; the curve must pass through it, not smooth
    across it.
    """
    saw = _sawtooth_path(41, amplitude_pitch=0.5)
    u = np.array(saw)
    res = smooth_sawtooth_runs(u, (), _PITCH, anchors=[20])
    assert res.changed
    assert 20.0 in res.anchors, "the transited vertex is not an anchor"
    dists = [_angle_deg(q, u[20]) for q in res.path_unit]
    assert min(dists) < 1e-9, "the curve does not pass through the transited centre"


def _deviation_bruteforce_deg(
    point: npt.NDArray[np.float64],
    polyline: npt.NDArray[np.float64],
    sample_deg: float = 0.3,
) -> float:
    """Sampled point-to-polyline distance: the reference the vectorised metric is
    checked against. Slow and obviously correct, which is what a reference is for.
    The sample step is coarse enough to keep the test fast; the closed form is
    exact, so the agreement bound is the sampling error, not a geometry bug."""
    best = math.inf
    for a, b in pairwise(polyline):
        n = max(2, math.ceil(math.degrees(angular_distance(a, b)) / sample_deg))
        for k in range(n + 1):
            q = a + (b - a) * (k / n)
            # An antipodal segment's chord passes through the centre, where the
            # radial is undefined; the closed form treats that segment as endpoint
            # distance, so the reference skips the singular samples too.
            if float(np.linalg.norm(q)) < 1e-9:
                continue
            pu = _unit(q)
            if pu is not None:
                best = min(best, math.degrees(angular_distance(point, pu)))
    return 0.0 if math.isinf(best) else best


def test_max_deviation_matches_a_bruteforce_reference() -> None:
    """The vectorised closed form equals a sampled walk, on every hard case.

    The metric is the cost of the whole smoothing call -- the loop form cost 99.8%
    of it -- so it was rewritten in array form, and an array rewrite of a geometric
    predicate is exactly where a sign or a mask goes quietly wrong. The fixtures are
    chosen to exercise each branch: perpendicular foot inside the arc, foot outside
    (endpoint wins), a run of coincident vertices (zero-length segments), and an
    antipodal pair (no unique great circle).
    """
    rng = np.random.default_rng(7)
    polys = [
        np.array([_unit(v) for v in [_X, _X + _Y, _Y, _Y + _Z]]),
        np.array(
            [
                _unit(math.cos(t) * _X + math.sin(t) * _Y)
                for t in np.linspace(0, 1.2, 12)
            ]
        ),
        np.array([_unit(rng.normal(size=3)) for _ in range(12)]),
        np.array([_unit(_X)] * 5 + [_unit(_Y)] * 5),
    ]
    for poly in polys:
        pts = np.array([_unit(rng.normal(size=3)) for _ in range(24)])
        ref = max(_deviation_bruteforce_deg(p, poly) for p in pts)
        got = _max_deviation_deg(pts, poly)
        assert abs(got - ref) < 0.2, f"vectorised {got:.4f} vs brute {ref:.4f}"


def test_max_deviation_treats_an_antipodal_segment_as_its_endpoints() -> None:
    """No unique great circle through antipodal points, so the arc is the endpoints.

    A sampled reference cannot check this: the chord between antipodes passes
    through the centre, and normalising points near it sweeps the whole
    perpendicular plane, so the sample is measuring a different (undefined) object.
    The contract is that such a segment contributes its endpoint distance and is
    not allowed to report a spurious zero.
    """
    poly = np.array([_unit(_X), -_unit(_X), _unit(_Y)])
    # Z is 90 deg from all three vertices and on neither arc's interior.
    assert _max_deviation_deg(np.array([_unit(_Z)]), poly) == pytest.approx(
        90.0, abs=1e-6
    )
    # unit(-X+Y) lies ON the second arc (-X -> Y), so its distance is 0; the
    # antipodal first arc must not contribute a spurious smaller value.
    on_arc = _unit(-_X + _Y)
    assert _max_deviation_deg(np.array([on_arc]), poly) == pytest.approx(0.0, abs=1e-6)
    # unit(X+Y) is 45 from X and 45 from Y, on neither arc's interior.
    mid = _unit(_X + _Y)
    assert _max_deviation_deg(np.array([mid]), poly) == pytest.approx(45.0, abs=1e-6)


def test_max_deviation_is_zero_for_a_path_against_itself() -> None:
    """Every vertex of a polyline lies on it, so the deviation must vanish."""
    path = np.array(_great_circle_path(41))
    assert _max_deviation_deg(path, path) == pytest.approx(0.0, abs=1e-9)
    assert _max_deviation_deg(np.zeros((0, 3)), path) == 0.0


def _winding_then_kink() -> tuple[npt.NDArray[np.float64], int]:
    """A tight spiral, a long clean leg, then one real 45 deg kink.

    The shape of the ethene3 shape-index connector: a stretch that winds violently
    (sustained, uniform, huge windowed deflection) followed by ground where the path
    is quiet and contains the one bend that matters.

    Returns:
        ``(path, apex_index)``.
    """
    spiral: list[npt.NDArray[np.float64]] = []
    for i in range(200):
        lam = i * 0.30
        r = 0.10 + 0.0004 * i
        p = (
            math.cos(lam) * _X * r
            + math.sin(lam) * _Y * r
            + _Z * math.sqrt(max(1e-9, 1.0 - r * r))
        )
        spiral.append(_unit(p))
    leg = [
        _unit(math.cos(i * _PITCH) * _X + math.sin(i * _PITCH) * _Y + 0.1 * _Z)
        for i in range(1, 41)
    ]
    apex = leg[-1]
    tangent = _unit(apex - leg[-2])
    outward = _unit(np.cross(tangent, apex))
    dir2 = _unit(
        math.cos(math.radians(45.0)) * tangent + math.sin(math.radians(45.0)) * outward
    )
    tail = [
        _unit(apex * math.cos(i * _PITCH) + dir2 * math.sin(i * _PITCH))
        for i in range(1, 21)
    ]
    return np.array(spiral + leg + tail), len(spiral) + len(leg) - 1


def test_a_corner_survives_when_the_rest_of_the_path_is_winding() -> None:
    """The baseline is LOCAL, so a churning stretch cannot hide a real bend.

    Measured on the ethene3 shape-index connector: it spirals half a turn inside an
    excision region, and a GLOBAL median baseline promoted 31 knots inside that
    region while the real 43 deg bend at the launch port was judged against the
    spiral and missed. One statistic, failing in both directions at once.

    The fixture reproduces both halves. The winding sets a global median of 68 deg,
    above the kink's 45, so a global baseline loses it; a local baseline reads the
    quiet ground the kink sits in and promotes it.
    """
    path, apex = _winding_then_kink()
    # Non-vacuity: with a baseline window as wide as the path, the winding wins and
    # the corner is lost -- this is the bug, asserted so the fix is measured.
    global_, _, _ = detect_corners(path, _PITCH, baseline_mult=50.0)
    assert not any(abs(c - apex) <= 3 for c in global_), (
        f"fixture no longer reproduces the failure: {global_}"
    )
    corners, _, _ = detect_corners(path, _PITCH)
    assert any(abs(c - apex) <= 3 for c in corners), (
        f"the real bend was lost to the winding's baseline: {corners}"
    )


def test_a_local_baseline_does_not_promote_sustained_curvature() -> None:
    """The other half: localising the baseline must not turn a bow into corners.

    A latitude circle deflects by the same amount at every vertex, so its LOCAL
    median equals its GLOBAL median and the excess test still reports nothing. If
    this test fails, the change has traded a missed corner for a knot storm.
    """
    for theta in (0.20, 0.35, 0.50):
        u = np.array(_latitude_circle(theta, 61, _PITCH))
        corners, _, _ = detect_corners(u, _PITCH)
        assert corners == (), f"theta={theta}: sustained curvature promoted {corners}"


def test_local_baseline_matches_the_global_one_on_a_homogeneous_path() -> None:
    """On a path that is one thing, local and global agree -- the safety property.

    The reason the change is safe on the fields that already worked: a homogeneous
    path has a local median equal to its global median, so every verdict is
    unchanged. Verified on the fixtures the corner tests are built from.
    """
    for name, path in [
        ("sawtooth", _sawtooth_path(41, amplitude_pitch=0.5)),
        ("great circle", _great_circle_path(41)),
        ("bowed", _latitude_circle(0.20, 41, _PITCH)),
    ]:
        u = np.array(path)
        for mult in (1.0, 6.0, 50.0):
            got, _, _ = detect_corners(u, _PITCH, baseline_mult=mult)
            assert got == (), f"{name}: baseline_mult={mult} promoted {got}"


# ---------------------------------------------------------------------------
# coarse structural curve
# ---------------------------------------------------------------------------


def _curve(
    path: list[npt.NDArray[np.float64]],
    center: tuple[float, float, float] | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], Any]:
    """Curve a centred fixture through the world-frame wrapper, back to radials.

    The wrapper detects its own corners, so a caller passes no knot set -- which is
    what makes the frame comparison test the detection as well as the geometry.
    """
    res = structural_curve_ridge(
        arc_points=_as_world(path, center),
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
        **kwargs,
    )
    return np.array([_local(p, center) for p in res.path]), res


def test_coarse_curve_is_built_from_few_nodes() -> None:
    """Nodes are spaced NODE_PITCH pitches: a handful, not one per mesh step.

    The description length of the branch is its arc divided by the node spacing.
    A 41-vertex path stepping one pitch covers 40 pitches, so at 8-pitch spacing it
    carries ~5 interior nodes plus the endpoints -- not 41.
    """
    path = _great_circle_path(41)
    u = np.array(path)
    res = coarse_structural_curve(u, (), _PITCH)
    span = 40 * math.degrees(_PITCH)
    expected = span / (NODE_PITCH * math.degrees(_PITCH))
    assert abs(res.n_nodes - (expected + 1)) <= 2, (
        f"{res.n_nodes} nodes for a {span:.0f} deg arc at "
        f"{NODE_PITCH} pitch (expected ~{expected + 1:.0f})"
    )


def test_coarse_curve_removes_the_winding() -> None:
    """THE property the coarse spacing exists for: the spacing is a filter cutoff.

    Measured on the ethene3 shape-index connector, which spirals half a turn inside
    an excision region: a curve at 1-pitch node spacing reproduces the winding almost
    exactly (65935 of 66010 deg of total turning) -- a faithful copy of a path that
    should not be followed -- while at 8 pitches the turning drops to 651 deg. The
    spacing decides which features survive, so it is a design parameter, not a
    density knob.

    The fixture is a deep sawtooth, which carries the same property at mesh scale:
    1-pitch nodes land on the teeth and reproduce them, 8-pitch nodes straddle them
    and average them out.
    """
    u = np.array(_sawtooth_path(121, amplitude_pitch=2.0))
    fine = coarse_structural_curve(u, (), _PITCH, node_pitch=1.0)
    coarse = coarse_structural_curve(u, (), _PITCH)
    before = fine.total_turn_before_deg
    assert before > 10000.0, "fixture must be winding hard"
    assert fine.total_turn_after_deg > 0.9 * before, (
        f"1-pitch spacing should reproduce the winding, got "
        f"{fine.total_turn_after_deg:.0f} of {before:.0f}"
    )
    assert coarse.total_turn_after_deg < 0.1 * fine.total_turn_after_deg, (
        f"coarse spacing did not filter the winding: "
        f"{fine.total_turn_after_deg:.0f} -> {coarse.total_turn_after_deg:.0f}"
    )


def test_coarse_curve_passes_through_endpoints_and_knots() -> None:
    """The endpoints and every knot are nodes: the curve interpolates them exactly."""
    path, apex = _sawtooth_with_corner(25, turn_deg=70.0)
    u = np.array(path)
    res = coarse_structural_curve(u, (apex,), _PITCH)
    assert apex in res.node_indices, "the knot is not a node"
    for idx in (0, apex, len(u) - 1):
        dists = [_angle_deg(q, u[idx]) for q in res.path_unit]
        assert min(dists) < 1e-6, f"the curve misses node {idx}"


def test_coarse_curve_keeps_a_knot_sharp() -> None:
    """A knot is a node, so the curve is free to kink there -- it is not smoothed.

    The coarse spacing removes winding and chatter; it must not blunt a real bend.
    The turn at the knot's output vertex stays large.
    """
    path, apex = _sawtooth_with_corner(25, turn_deg=70.0)
    u = np.array(path)
    res = coarse_structural_curve(u, (apex,), _PITCH)
    dists = [_angle_deg(q, u[apex]) for q in res.path_unit]
    i = int(np.argmin(dists))
    turns = point_turn_deg(res.path_unit)
    assert turns[i] > 30.0, f"the corner was blunted: {turns[i]:.1f} deg at the knot"


def test_coarse_curve_reproduces_a_straight_path() -> None:
    """A great circle is described by a great circle: no invented curvature.

    The licence to depart from the ascent is for filtering winding, not for bending
    straight ground. A curve through nodes on a geodesic stays on it.
    """
    path = _great_circle_path(41)
    u = np.array(path)
    res = coarse_structural_curve(u, (), _PITCH)
    assert res.max_deviation_deg < 0.05, (
        f"curve left the geodesic: {res.max_deviation_deg:.3f} deg"
    )


def test_coarse_curve_is_frame_independent() -> None:
    """Curving through an off-origin centre gives the centred frame's geometry."""
    center = (1.2618, -0.4407, 0.3311)
    # A fixture whose corner IS detectable, so the knot set is non-empty and the
    # frame comparison covers the corner path too. (A corner embedded in a full
    # sawtooth is not detected -- the known limitation recorded in the journal --
    # so _sawtooth_with_corner would make this test vacuous on the knot half.)
    path, apex = _kinked_path(21, turn_deg=70.0)
    u = np.array(path)
    # The wrapper detects its own corners, so the centred call is given the same
    # knot set it would find -- which also pins down that detection is frame
    # independent, the precondition for the comparison to mean anything.
    corners = detect_corners(u, _PITCH)[0]
    assert abs(corners[0] - apex) <= 2, (
        f"fixture must put a detectable corner at the apex, got {corners}"
    )
    centred = coarse_structural_curve(u, corners, _PITCH)
    lifted, res = _curve(path, center=center)
    assert res.corner_indices == corners, "corner detection is frame-dependent"
    assert len(lifted) == len(centred.path_unit)
    dev = max(_angle_deg(a, b) for a, b in zip(lifted, centred.path_unit, strict=True))
    assert dev < 1e-6, f"frame-dependent geometry: {dev:.2e} deg"


def test_coarse_curve_is_deterministic() -> None:
    """Same path in, same curve out. Pure geometry, no field, no RNG."""
    path, apex = _sawtooth_with_corner(25, turn_deg=70.0)
    u = np.array(path)
    a = coarse_structural_curve(u, (apex,), _PITCH)
    b = coarse_structural_curve(u, (apex,), _PITCH)
    assert a.path_unit.shape == b.path_unit.shape
    assert np.allclose(a.path_unit, b.path_unit)


def test_coarse_curve_pins_a_transited_centre() -> None:
    """A vertex pinned as an excision transit is a node the curve passes through."""
    path = _sawtooth_path(41, amplitude_pitch=0.5)
    u = np.array(path)
    res = coarse_structural_curve(u, (), _PITCH, anchors=[20])
    assert 20 in res.node_indices, "the transited vertex is not a node"
    dists = [_angle_deg(q, u[20]) for q in res.path_unit]
    assert min(dists) < 1e-6, "the curve does not pass through the transited centre"
