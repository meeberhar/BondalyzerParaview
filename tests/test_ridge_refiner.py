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

The frame contract is tested explicitly: the refiner works on centered unit
radials and exchanges world-frame polylines with its caller, so refining through
an off-origin center must give the same geometry as refining in the centered frame.
Pd_20K sits at the origin and cannot see this class of bug; ethene3 at 1.2618 A
can.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from gba_topology2.src import ridge_refiner
from gba_topology2.src.ridge_refiner import (
    _CREST_ON_CREST_PITCH,
    ARC_GRADIFY_NODES,
    CORNER_EXCESS_DEG,
    GRADIFY_OUTPUT_PITCH,
    NODE_PITCH,
    QUAD_FIT_MIN_GAIN_FRACTION,
    QUAD_FIT_NM_MAX_ITERS,
    QUAD_FIT_SEED_ANGLES,
    QUAD_FIT_SEED_RADII,
    RIDGE_REACH_PITCH,
    SEGMENT_SAMPLE_DEG,
    ArcGradification,
    QuadBezierFit,
    RidgeRefinement,
    _max_deviation_deg,
    _quad_basis,
    _quad_control,
    _robust_field_span,
    _sample_field,
    _scan_crests,
    chord_deflection_deg,
    coarse_structural_curve,
    detect_chatter_runs,
    detect_corners,
    fit_quad_bezier,
    gradify_arc,
    gradify_connections,
    point_turn_deg,
    probe_gradient_ridge,
    quad_bezier_points,
    ridge_point_at,
    smooth_gradient_ridge,
    smooth_sawtooth_runs,
    span_field_integral,
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
    """World point -> centered unit radial. ``None`` means already centered."""
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
    """Centered unit radials scaled to R and lifted into the world frame."""
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
    """Smooth a centered fixture through the world-frame wrapper, back to radials.

    The path is lifted to the world frame (translated by ``center``) and handed to
    :func:`smooth_gradient_ridge`, which converts back to centered radials internally.
    The result is converted to centered unit radials again so a caller can compare it
    against the centered-frame smoothing vertex for vertex -- the frame contract the
    whole module rests on, invisible on a centered atom and so tested explicitly.

    Returns:
        ``(path_unit, refinement)``: the smoothed curve as centered unit radials, and
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


def test_probe_accepts_a_path_already_on_the_crest() -> None:
    """A path that IS the crest is a measurement, not a refusal.

    The regression this pins: the gain gate asks whether the step lands on ground
    HIGHER than the sample, and a correct zero-length step cannot answer yes -- it lands
    where it started. Before the ``on_crest`` acceptance every already-correct node was
    refused, carried unchanged, and the arc was reported un-gradifiable when it was in
    fact a gradient path.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.0)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    assert probe.samples, "the path is long enough to sample"
    assert probe.n_accepted == len(probe.samples), (
        f"an on-crest path was refused: {[s.point.reason for s in probe.samples]}"
    )
    assert all(s.point.reason == "on_crest" for s in probe.samples)
    # The step is below the on-crest tolerance, i.e. below the mesh's own resolution.
    assert max(abs(s.point.signed_offset_rad) for s in probe.samples) <= (
        _CREST_ON_CREST_PITCH * _PITCH
    )


def test_on_crest_does_not_wave_a_reachable_crest_through() -> None:
    """The tolerance is a quarter pitch, so a reachable crest is still stepped to.

    A path 0.5 pitch off the crest is outside ``_CREST_ON_CREST_PITCH``, so the
    on-crest shortcut must not fire: the probe has to find the crest and move.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.5)
    probe = probe_gradient_ridge(_as_world(path, None), pts, f, _R, _PITCH, center=None)
    accepted = [s for s in probe.samples if s.point.accepted]
    assert accepted, f"nothing accepted: {[s.point.reason for s in probe.samples]}"
    assert all(s.point.reason == "ok" for s in accepted)
    assert 0.3 < probe.median_offset_pitch < 0.7


def test_robust_field_span_ignores_a_spike() -> None:
    """Two spike vertices must not set the gain gate for the whole field.

    Measured on ethene3 atom 1 curvedness: 2 vertices of 10081 hold a global span of
    0.4802 while the p5-p95 band is 0.003531. A gate normalised by the global span then
    demands a rise two orders of magnitude above the field's own structure, so every
    real step on the crest is refused -- which is exactly what made 74% of the nodes on
    Pd_20K come back carried instead of measured.
    """
    pts = _fibonacci_sphere()
    base = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    spiked = base.copy()
    spiked[0] += 1000.0
    spiked[1] += 1000.0

    global_span = float(np.max(spiked) - np.min(spiked))
    robust = _robust_field_span(spiked)
    assert global_span > 900.0, "the fixture did not build a spike"
    assert robust < 2.0, f"robust span {robust:.3f} still carries the spike"
    assert robust > 0.5, f"robust span {robust:.3f} trimmed the ridge itself"

    # The property, end to end: a spike in the field must not change the answer.
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.5)
    world = _as_world(path, None)
    clean = probe_gradient_ridge(world, pts, base, _R, _PITCH, center=None)
    noisy = probe_gradient_ridge(world, pts, spiked, _R, _PITCH, center=None)
    assert clean.n_accepted > 0, "the fixture has no crest"
    assert noisy.n_accepted == clean.n_accepted, (
        f"a spike changed the answer: {[s.point.reason for s in noisy.samples]}"
    )
    assert noisy.median_offset_pitch == pytest.approx(
        clean.median_offset_pitch, rel=1e-9
    )

    # And the defect, one step, both ways: the SAME sample, field and tangent, gated by
    # the span a spike would have produced and by the robust span. Only the gate
    # differs, and the answer flips -- that is the whole fix in one comparison.
    accepted = next(s for s in noisy.samples if s.point.accepted)
    tangent = np.cross(np.asarray(_Z, dtype=np.float64), accepted.from_unit)
    tangent = tangent / float(np.linalg.norm(tangent))
    robust_step = ridge_point_at(
        accepted.from_unit,
        tangent,
        pts,
        spiked,
        _R,
        _PITCH,
        span_override=_robust_field_span(spiked),
    )
    assert robust_step.accepted, f"the robust span was refused: {robust_step.reason}"
    gated = ridge_point_at(
        accepted.from_unit,
        tangent,
        pts,
        spiked,
        _R,
        _PITCH,
        span_override=global_span,
    )
    assert not gated.accepted, "the global span was expected to refuse this step"
    assert gated.reason == "no_crest"


def test_robust_field_span_degenerate_inputs() -> None:
    """A short or collapsed field degrades to the plain span, never to zero."""
    assert _robust_field_span(np.asarray([], dtype=np.float64)) == 0.0
    short = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert _robust_field_span(short) == 3.0  # under 20 values: no trim
    flat = np.full(64, 2.5, dtype=np.float64)
    assert _robust_field_span(flat) == 0.0, "a constant field has no span"


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
    """Probing through an off-origin center equals probing in the centered frame."""
    center = (1.2618, -0.0001, -0.0002)
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 31, _PITCH, latitude_pitch=0.5)
    centered = probe_gradient_ridge(
        _as_world(path, None), pts, f, _R, _PITCH, center=None
    )
    assert centered.n_accepted > 0, "fixture must produce accepted samples"
    world = probe_gradient_ridge(
        _as_world(path, center), pts, f, _R, _PITCH, center=center
    )
    assert centered.knot_indices == world.knot_indices
    assert len(centered.samples) == len(world.samples)
    # Both probes report CENTRED unit radials, so they compare directly: the frame
    # conversion is the probe's internal business, and a step that used the world
    # origin as the sphere center would show up here as a disagreement. The
    # tolerance is float round-off through the translation (1e-6 deg on a 1.2618 A
    # offset), not a property of the algorithm.
    for a, b in zip(centered.samples, world.samples):
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
    """Smoothing through an off-origin center gives the centered frame's geometry.

    The wrapper converts world -> centered radials, smooths, and scales back. For an
    atom at the origin the two frames coincide, which is why every frame bug in this
    project has been invisible on Pd_20K; ethene3's center is 1.2618 A off the origin
    and can see it. The mesh is irrelevant here -- the smoother samples no field --
    so only the path and the center are moved together.
    """
    center = (1.2618, -0.4407, 0.3311)
    saw = _sawtooth_path(41, amplitude_pitch=0.5)
    centered = smooth_sawtooth_runs(np.array(saw), (), _PITCH)
    lifted, _ = _smooth_world(saw, center=center)
    assert centered.changed and len(lifted) == len(centered.path_unit), (
        "the two frames produced different amounts of work"
    )
    dev = max(_angle_deg(a, b) for a, b in zip(lifted, centered.path_unit, strict=True))
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
    assert min(dists) < 1e-9, "the curve does not pass through the transited center"


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
            # An antipodal segment's chord passes through the center, where the
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
    through the center, and normalising points near it sweeps the whole
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
    """Curve a centered fixture through the world-frame wrapper, back to radials.

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
    """Curving through an off-origin center gives the centered frame's geometry."""
    center = (1.2618, -0.4407, 0.3311)
    # A fixture whose corner IS detectable, so the knot set is non-empty and the
    # frame comparison covers the corner path too. (A corner embedded in a full
    # sawtooth is not detected -- the known limitation recorded in the journal --
    # so _sawtooth_with_corner would make this test vacuous on the knot half.)
    path, apex = _kinked_path(21, turn_deg=70.0)
    u = np.array(path)
    # The wrapper detects its own corners, so the centered call is given the same
    # knot set it would find -- which also pins down that detection is frame
    # independent, the precondition for the comparison to mean anything.
    corners = detect_corners(u, _PITCH)[0]
    assert abs(corners[0] - apex) <= 2, (
        f"fixture must put a detectable corner at the apex, got {corners}"
    )
    centered = coarse_structural_curve(u, corners, _PITCH)
    lifted, res = _curve(path, center=center)
    assert res.corner_indices == corners, "corner detection is frame-dependent"
    assert len(lifted) == len(centered.path_unit)
    dev = max(_angle_deg(a, b) for a, b in zip(lifted, centered.path_unit, strict=True))
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
    assert min(dists) < 1e-6, "the curve does not pass through the transited center"


# ---------------------------------------------------------------------------
# gradification: bending a geodesic connector onto the crest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeConnection:
    """The minimal record shape :func:`gradify_connections` reads and replaces.

    ``MaxConnection`` and ``MinConnection`` differ in their endpoint field names;
    the gradifier must not care, and a fixture with NEITHER name proves it reads
    only ``method``/``arc_points`` and writes only the gradify fields. The gradify
    fields carry the same names and defaults as the real records', because
    ``dataclasses.replace`` can only write fields that exist.
    """

    saddle_id: int
    method: str
    arc_points: tuple[tuple[float, float, float], ...]
    mutual: bool = True
    gradified: bool = False
    gradify_nodes: int = 0
    gradify_moved: int = 0
    gradify_carried: int = 0
    gradify_pinned: int = 0
    gradify_median_offset_pitch: float = 0.0
    gradify_max_offset_pitch: float = 0.0
    gradify_residual_pitch: float = 0.0
    gradify_status: str = ""


class _Region:
    """An :class:`GradifyRegion` by duck type: a center radial and an angular radius."""

    def __init__(self, centre_unit: npt.NDArray[np.float64], radius_rad: float) -> None:
        self._centre = np.asarray(centre_unit, dtype=np.float64)
        self.radius_rad = float(radius_rad)

    @property
    def centre_unit(self) -> npt.NDArray[np.float64]:
        return self._centre


def _geodesic_arc(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], n: int = 25
) -> list[npt.NDArray[np.float64]]:
    """The great-circle chord from ``a`` to ``b`` as ``n`` centered unit radials.

    Built with :func:`slerp`, so it is a true geodesic: the bulge away from a
    parallel small circle is the whole point of the tests that use it.
    """
    out: list[npt.NDArray[np.float64]] = []
    for i in range(n):
        p = slerp(a, b, i / (n - 1))
        assert p is not None
        out.append(_unit(p))
    return out


def _offset_endpoints(
    span_deg: float, offset_pitch: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two endpoints at latitude ``offset_pitch`` pitches, ``span_deg`` apart.

    The crest of :func:`_ridge_field` with ``pole=_Z`` is the equator, so the
    geodesic between two points at equal latitude bows AWAY from the crest by
    ``separation/2 * tan(latitude)`` -- 11 degrees for a 24 degree span at half a
    pitch. Fixtures therefore live in the few-degree regime, where a geodesic is a
    slightly bowed chord exactly as a real connector is, and their offsets stay
    inside RIDGE_REACH_PITCH so the field can speak for every node.
    """
    off = offset_pitch * _PITCH
    half = math.radians(span_deg) / 2.0
    a = _unit(
        math.cos(off) * (math.cos(half) * _X + math.sin(half) * _Y) + math.sin(off) * _Z
    )
    b = _unit(
        math.cos(off) * (math.cos(-half) * _X + math.sin(-half) * _Y)
        + math.sin(off) * _Z
    )
    return a, b


def _lat_endpoints(
    span_deg: float,
    lat_a_pitch: float,
    lat_b_pitch: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two endpoints at latitudes ``lat_a_pitch``/``lat_b_pitch``, ``span_deg`` apart.

    The general case of :func:`_offset_endpoints`: a geodesic between UNEQUAL
    latitudes runs monotonically across the interval between them, which is the only
    way to build an arc whose nodes straddle RIDGE_REACH_PITCH -- a parallel-chord
    arc keeps a nearly constant offset (24 deg at 0.85 pitch bulges by 0.02), so on
    it the reach gate accepts ALL the nodes or NONE of them.
    """
    half = math.radians(span_deg) / 2.0

    def end(lat_pitch: float, sign: float) -> npt.NDArray[np.float64]:
        off = lat_pitch * _PITCH
        return _unit(
            math.cos(off) * (math.cos(half) * _X + sign * math.sin(half) * _Y)
            + math.sin(off) * _Z
        )

    return end(lat_a_pitch, 1.0), end(lat_b_pitch, -1.0)


def _gradify_world(
    path: list[npt.NDArray[np.float64]],
    pts: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    center: tuple[float, float, float] | None = None,
    **kwargs: Any,
) -> ArcGradification:
    """Gradify a centered fixture through the world-frame wrapper."""
    return gradify_arc(
        arc_points=_as_world(path, center),
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
        **kwargs,
    )


def _lat_pitch(u: npt.ArrayLike) -> float:
    """Signed latitude of a radial, in MESH PITCHES (crest = equator = 0).

    :func:`_lat_deg` returns degrees; dividing that by a radian pitch invents a
    factor of 57.3, which is exactly the kind of unit slip these fixtures exist to
    keep out.
    """
    v = _unit(u)
    return math.asin(float(np.clip(float(np.dot(v, _Z)), -1.0, 1.0))) / _PITCH


def _plane_pitch(arc: Sequence[Any]) -> npt.NDArray[np.float64]:
    """Unit normal of the great circle through ``arc``, for off-plane measurement.

    Distance to the geodesic's VERTEX set is a tessellation measure, not a geometry
    one: the emitted line is resampled at ~1 mesh pitch, so with a coarse fixture a
    vertex that never moved can sit half a segment from the nearest reference vertex.
    Distance to the great circle's PLANE is exact, and it is the plane a pin promises
    to hold.
    """
    a = _unit(np.asarray(arc[0], dtype=np.float64))
    b = _unit(np.asarray(arc[-1], dtype=np.float64))
    normal = np.cross(a, b)
    n = float(np.linalg.norm(normal))
    assert n > 1e-9, "fixture: an antipodal arc has no unique great circle"
    return normal / n


def _path_units_of(res: ArcGradification) -> list[npt.NDArray[np.float64]]:
    """The gradified path as centered unit radials (the fixture is at the origin)."""
    return [_unit(np.asarray(p)) for p in res.path]


def _node_lats(
    res: ArcGradification, center: tuple[float, float, float] | None = None
) -> list[float]:
    """Signed latitude (mesh pitches) of every emitted vertex, world -> radials."""
    return [_lat_pitch(_local(np.asarray(q), center)) for q in res.path]


def test_gradify_pulls_a_geodesic_onto_the_crest() -> None:
    """The core promise: a straight arc that missed the ridge is bent onto it.

    The input is a great circle between two points 0.5 pitch off the crest, so its
    MIDPOINT is further off than its ends (a geodesic bulges away from a parallel
    small circle) -- the exact failure the operator reported. Every node is inside
    RIDGE_REACH_PITCH, so the field can speak for all of them.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=4.0, offset_pitch=0.5)
    arc = _geodesic_arc(a, b)
    res = _gradify_world(arc, pts, f)

    assert res.status == "ok", res.status
    assert res.changed
    assert res.n_nodes == ARC_GRADIFY_NODES, "the node budget was not spent"
    assert res.n_moved == res.n_nodes, "measurable ground was left unstepped"
    assert res.n_carried == 0 and res.n_pinned == 2, (res.n_carried, res.n_pinned)

    before = [abs(_lat_pitch(u)) for u in arc]
    after = [abs(v) for v in _node_lats(res)]
    assert min(before) > 0.3, f"the fixture is not off-crest: {min(before):.3f}"
    assert max(after[1:-1]) < 0.3 * min(before), (
        f"gradification left {max(after[1:-1]):.3f} pitch of offset "
        f"(the input never comes closer than {min(before):.3f})"
    )
    # The step the node took is what the field asked for: ~0.3-0.5 pitch.
    assert 0.2 < res.median_offset_pitch < 0.6, res.median_offset_pitch
    assert res.max_offset_pitch <= RIDGE_REACH_PITCH + 1e-9, res.max_offset_pitch


def test_gradify_endpoints_are_bit_stable() -> None:
    """Endpoints do not move, not one ulp: basin-boundary joints depend on it.

    ``basin_loop`` welds arcs into a cycle by comparing joint positions, and the
    arc's identity (which saddle, which CP) is its endpoint. A gradified arc whose
    endpoint moved by a nanoradian would silently re-key the basin boundary.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=10.0, offset_pitch=0.5)
    world = _as_world(_geodesic_arc(a, b), None)
    res = gradify_arc(world, pts, f, _R, _PITCH)
    assert res.changed
    assert res.path[0] == tuple(world[0]), "the start endpoint moved"
    assert res.path[-1] == tuple(world[-1]), "the end endpoint moved"


def test_gradify_off_origin_centre_matches_the_centred_frame() -> None:
    """Gradification is frame independent -- Pd_20K cannot see this class of bug.

    The center is where the project's frame bugs have always hidden: an atom at the
    origin is its own frame, so every fixture that only tests ``center=None`` passes
    for the wrong reason.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=10.0, offset_pitch=0.5)
    arc = _geodesic_arc(a, b)
    center = (1.2618, -0.44, 0.31)
    in_centred = _gradify_world(arc, pts, f)
    in_world = _gradify_world(arc, pts, f, center=center)
    assert in_centred.changed and in_world.changed
    for p, q in zip(in_centred.path, in_world.path):
        u = _unit(np.asarray(p))
        v = _unit(_local(np.asarray(q), center))
        assert angular_distance(u, v) < 1e-9, (p, q)


def test_gradify_short_path_is_a_no_op() -> None:
    """Fewer than _MIN_PATH vertices: nothing to place nodes on, path unchanged."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=10.0, offset_pitch=0.5)
    short = _geodesic_arc(a, b, n=4)
    world = _as_world(short, None)
    res = gradify_arc(world, pts, f, _R, _PITCH)
    assert res.status == "degenerate"
    assert not res.changed
    assert res.path == tuple(tuple(p) for p in world)


def test_gradify_without_a_field_is_a_no_op() -> None:
    """An empty mesh is reported, not swallowed: the arc stays drawn and straight."""
    a, b = _offset_endpoints(span_deg=10.0, offset_pitch=0.5)
    world = _as_world(_geodesic_arc(a, b), None)
    res = gradify_arc(
        world,
        np.zeros((0, 3)),
        np.zeros(0),
        _R,
        _PITCH,
    )
    assert res.status == "no_field"
    assert res.path == tuple(tuple(p) for p in world)


def test_gradify_field_silence_refuses_everything_and_changes_nothing() -> None:
    """Where the crest is beyond the measured reach, the arc stays straight.

    The offset is 1.5 mesh pitches at the ends and more at the middle, against
    RIDGE_REACH_PITCH = 0.72: EVERY node is refused ``out_of_reach``, so there is no
    acceptance to carry a step from, and the honest answer is the geodesic the ray
    produced -- reported as ``all_refused``, counted, not silently blanked. This is
    the same measured bound the probe test asserts; gradification inherits it rather
    than inventing its own.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=4.0, offset_pitch=1.5)
    world = _as_world(_geodesic_arc(a, b), None)
    res = gradify_arc(world, pts, f, _R, _PITCH)
    assert res.status == "all_refused", res.status
    assert not res.changed
    assert res.n_moved == 0
    assert res.n_nodes > 0, "the nodes were placed and graded"
    assert res.median_offset_pitch == 0.0 and res.max_offset_pitch == 0.0
    assert res.path == tuple(tuple(p) for p in world), "an untouched arc changed"


def test_gradify_describes_the_bend_at_no_coarser_than_pitch() -> None:
    """The emitted polyline is described finely enough for the Step 6 smoother.

    The smoother's windows are pitch-scaled, so what matters is that NO STEP is
    coarser than the field's pitch (a coarse chain smooths a line the bend was never
    in). It is NOT that the spacing equals the traced connectors' 1.0 pitch: the node
    budget is a COUNT (the plan's ``n_nodes``, default 20), so on a 24 deg span the
    steps land near 0.4 pitch by construction. Denser is safe, sparser is not, and
    the path is not over-described either (a gradified arc is a polyline, not a
    point cloud).
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=24.0, offset_pitch=0.5)
    arc = _geodesic_arc(a, b)
    res = _gradify_world(arc, pts, f)
    u = [_unit(np.asarray(p)) for p in res.path]
    steps = np.array([angular_distance(x, y) / _PITCH for x, y in pairwise(u)])
    assert steps.max() <= 1.05 * GRADIFY_OUTPUT_PITCH, steps.max()
    assert float(np.median(steps)) <= 1.0 * GRADIFY_OUTPUT_PITCH
    assert steps.min() > 0.0
    geodesic_len = sum(angular_distance(x, y) / _PITCH for x, y in pairwise(arc))
    assert sum(steps) <= geodesic_len * 1.15, (sum(steps), geodesic_len)
    assert len(u) > 5, "the smoother needs a polyline, not a segment"
    assert len(u) <= 400, len(u)
    # The bend is real and it is sideways: interior vertices leave the great circle,
    # the endpoints do not.
    dev = [min(angular_distance(v, w) / _PITCH for w in np.array(arc)) for v in u]
    assert max(dev[1:-1]) > 0.3, dev
    assert dev[0] == 0.0 and dev[-1] == 0.0, "an endpoint moved"


def test_gradify_pins_an_excision_region_and_uses_its_own_crossings_as_ports() -> None:
    """Region ground is unmeasurable, so the arc keeps the great circle there.

    Inside the disc the mesh was condensed to the boundary ring: a Hessian fitted on
    region ground reports the ring's arithmetic, not a crest. The rule is the traced
    connector's, in geometric form -- the arc's own entry/exit crossings are the
    ports and the transit between them is a straight carry.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=8.0 * _PITCH)
    # 40 deg span at the project's own 25-vertex description length: 1.65 pitch per
    # segment, so a 3 pitch disc (5.5 deg -- a real excision radius) covers the middle
    # vertices and leaves 9-segment free spans on both sides for the ports to bound.
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    arc = _geodesic_arc(a, b)
    mid = arc[len(arc) // 2]
    region = _Region(mid, 3.0 * _PITCH)
    res = _gradify_world(arc, pts, f, regions=(region,))
    assert res.status == "ok", res.status
    # n_pinned counts the LAYOUT pins: the two CP centres plus the transit's entry port.
    assert res.n_pinned >= 3, "the region crossing was not used as a port"

    normal = _plane_pitch(arc)
    u_out = [_unit(np.asarray(p)) for p in res.path]
    # The geodesic's own vertices inside the disc (the fixture's disc spans 11 of the
    # 25 vertices; computed, not typed, so the tessellation is the fixture's business).
    inside = [
        i for i in range(len(arc)) if angular_distance(arc[i], mid) <= 3.0 * _PITCH
    ]
    assert len(inside) >= 5, "fixture: the disc covers no ground"
    assert inside[0] > 1 and inside[-1] < len(arc) - 2, "fixture: the disc eats an end"
    for i in inside:
        assert any(angular_distance(v, arc[i]) < 1e-12 for v in u_out), (
            f"the geodesic's disc vertex {i} is not in the gradified path"
        )
    # Away from the two BRIDGE segments (last stepped node to near port, far port to
    # first stepped node), every vertex inside the disc is on the great circle. The
    # bridge is the single slerp that hands the line from measured ground to the
    # transit, and its sub-vertices are on-plane to within one step of the port.
    for i, v in enumerate(u_out):
        d = angular_distance(v, mid)
        if d < 3.0 * _PITCH - 1.2 * _PITCH:
            assert abs(float(np.dot(v, normal))) < 1e-9, (
                f"vertex {i} at {d / _PITCH:.3f} pitch was bent inside the region"
            )
    # And OUTSIDE the disc the same arc did get bent (selected by distance from the
    # disc, since u_out is the emitted path, not the 25-vertex geodesic).
    bent = [
        i
        for i, v in enumerate(u_out[1:-1], start=1)
        if angular_distance(v, mid) > 3.5 * _PITCH
    ]
    assert bent, "fixture: nothing lies outside the disc"
    # Thresholds in this file are MESH PITCHES (a 0.5 pitch bend is the whole point of
    # the fixture), so compare in pitch, not in the raw radians np.dot returns.
    assert any(abs(float(np.dot(u_out[i], normal))) / _PITCH > 0.1 for i in bent), [
        round(abs(float(np.dot(u_out[i], normal))) / _PITCH, 3) for i in bent
    ]


def test_gradify_pins_every_vertex_inside_a_region() -> None:
    """A region's transit travels, and the rim beside it is not stepped.

    Two things the pin rule owes the drawing: the geodesic's OWN vertices inside the
    disc must reach the emitted path (a chain that carried only the two ports would
    connect them with a chord and cut the corner off the great circle), and the first
    STEPPED node must sit a full segment outside the rim -- stepping the vertex
    immediately outside a pin leaves a kink the smoother would then have to defend a
    corner the field never claimed.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=8.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    arc = _geodesic_arc(a, b)
    mid = arc[len(arc) // 2]
    bare = _gradify_world(arc, pts, f)
    pinned = _gradify_world(arc, pts, f, regions=(_Region(mid, 3.0 * _PITCH),))
    assert bare.n_pinned == 2, "an arc without regions is pinned at its centres only"
    # The transit costs a port: the layout gains a pin, and the straight stretch across
    # the disc is the geodesic's own transit.
    assert pinned.n_pinned > bare.n_pinned, (bare.n_pinned, pinned.n_pinned)
    # And the ground beside the disc is stepped NO closer than one segment: stepping the
    # rim vertex itself is the kink the rule exists to prevent.
    assert pinned.n_nodes < bare.n_nodes, "the disc took no ground from the nodes"
    normal = _plane_pitch(arc)
    u_pinned = _path_units_of(pinned)
    stepped = [
        i for i, v in enumerate(u_pinned) if abs(float(np.dot(v, normal))) > 1e-9
    ]
    assert stepped, "nothing was stepped at all: fixture is broken"
    near_rim = [
        i
        for i in stepped
        if angular_distance(u_pinned[i], mid) < 3.0 * _PITCH + 0.5 * _PITCH
    ]
    assert not near_rim, [
        round(angular_distance(u_pinned[i], mid) / _PITCH, 3) for i in near_rim
    ]


def test_gradify_carries_a_refusal_between_two_accepted_neighbours() -> None:
    """A node the field refused inherits its neighbours' step, in between them.

    The connector has to be ONE line, so silence in the middle is filled from the
    ground on either side -- and it is reported (``n_carried``), never hidden. The
    magnitudes here are the probe's own, so the interpolation is checked by
    sandwich: a carried node moves further than the near neighbour, less than the
    far one, on the same side of the crest.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=8.0 * _PITCH)
    # Endpoints 0.55 and 0.95 pitch off the crest over a 20 deg span: the geodesic
    # runs from +0.55 to +0.95, so its nodes cross the 0.72 reach -- the near end is
    # accepted, the far end refused. A parallel-chord arc cannot do this (its offset
    # is constant to within 0.02 pitch), which is why the fixture uses unequal
    # latitudes; see :func:`_lat_endpoints`.
    a, b = _lat_endpoints(span_deg=20.0, lat_a_pitch=0.55, lat_b_pitch=0.95)
    arc = _geodesic_arc(a, b)
    res = _gradify_world(arc, pts, f, n_nodes=12)
    assert res.n_moved > 0, "no acceptance to carry from"
    assert res.n_moved < res.n_nodes, "the far nodes were not refused"
    assert res.n_carried > 0, "nothing was refused between the acceptances"
    assert res.n_carried == res.n_nodes - res.n_moved, (res.n_carried, res.n_nodes)
    geodesic_lats = [_lat_pitch(u) for u in arc]
    out_lats = _node_lats(res)
    # The bend happened: the drawn line comes strictly closer to the crest than the
    # great circle did. (Per-vertex monotonicity is NOT promised -- a Newton step can
    # land a hair past the crest, and a path that overshoots by 0.01 pitch is on it.)
    assert min(out_lats) < min(geodesic_lats) - 0.2, (
        min(out_lats),
        min(geodesic_lats),
    )


def test_gradify_nodes_land_only_on_spans_with_room() -> None:
    """Largest-remainder allocation: nodes go on measurable spans, not on pins.

    A span of one segment has no room for a node, and a region's interior gets none
    at all: the node budget is spent on the ground between the pins, in proportion
    to that ground's arc length.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=8.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    arc = _geodesic_arc(a, b)
    regions = (
        _Region(arc[9], 2.0 * _PITCH),
        _Region(arc[16], 2.0 * _PITCH),
    )
    res = _gradify_world(arc, pts, f, regions=regions, n_nodes=12)
    assert res.status == "ok", res.status
    assert res.n_nodes > 0, "the free ground got no nodes"
    # Layout pins: the two centres plus the ports the regions contribute.
    assert res.n_pinned >= 3, res.n_pinned
    normal = _plane_pitch(arc)
    u_out = [_unit(np.asarray(p)) for p in res.path]
    for region in regions:
        inside = [
            v
            for v in u_out
            if angular_distance(v, region.centre_unit) <= region.radius_rad
        ]
        assert inside, "the arc skipped a region: fixture is broken"
        for v in inside:
            assert abs(float(np.dot(v, normal))) < 1e-9, "a region's transit was bent"


def test_gradify_all_pinned_arc_is_reported_as_pinned() -> None:
    """An arc that lies entirely on region/center ground is returned unchanged."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=6.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=6.0, offset_pitch=0.2)
    arc = _geodesic_arc(a, b)
    big = _Region(arc[len(arc) // 2], 4.0 * _PITCH)
    world = _as_world(arc, None)
    res = gradify_arc(world, pts, f, _R, _PITCH, regions=(big,))
    assert res.status == "pinned", res.status
    assert not res.changed
    assert res.n_nodes == 0
    assert res.path == tuple(tuple(p) for p in world)


def test_gradify_connections_gradifies_geodesics_and_leaves_traces_alone() -> None:
    """One rule, both polarities: the record's identity is untouched.

    ``method`` records how the ENDPOINT was found (the ray), and the viewer's color
    split plus the smoother's geodesic skip read it. So the gradified record keeps
    ``method="geodesic"`` and gains ``gradified=True``; every other field --
    saddle_id, branch, endpoint, status, mutual -- is preserved, which is what keeps
    the basin boundary's arc set and the degenerate-arc audit invariant.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    world = tuple(_as_world(_geodesic_arc(a, b), None))
    geodesic = _FakeConnection(1, "geodesic", world)
    traced = _FakeConnection(2, "gradient", world)
    out, stats = gradify_connections(
        [geodesic, traced],
        pts,
        f,
        _R,
        _PITCH,
        n_nodes=12,
    )
    assert [c.saddle_id for c in out] == [1, 2], "input order is preserved"
    g = out[0]
    assert g is not geodesic
    assert g.gradified is True
    assert g.method == "geodesic", "the method tag names the endpoint search"
    assert g.saddle_id == 1 and g.mutual is True, "the record's identity changed"
    assert g.arc_points[0] == world[0] and g.arc_points[-1] == world[-1]
    assert g.arc_points != world, "the geometry was not gradified"
    # Placed nodes: the request is a BUDGET, and pinned ground is not offered to
    # the field, so the count is what was placed -- and every placed node is
    # accounted for as measured (moved) or inherited (carried).
    assert 0 < g.gradify_nodes <= 25, g.gradify_nodes
    assert g.gradify_nodes == g.gradify_moved + g.gradify_carried, (
        g.gradify_nodes,
        g.gradify_moved,
        g.gradify_carried,
    )
    assert g.gradify_moved > 0
    assert g.gradify_status == "ok"
    # The traced record is the SAME object, not a copy: that is how a caller tells
    # an untouched record from one the field refused.
    assert out[1] is traced
    assert out[1].gradified is False

    assert stats["n_arcs"] == 1, "gradient records are not arcs"
    assert stats["n_gradified"] == 1
    # The node count is the requested BUDGET; the moved/carry split is the field's
    # answer, measured per node downstream, so only the accounting identity is pinned.
    assert stats["n_nodes"] == 12
    assert stats["n_moved"] > 0
    assert stats["n_moved"] + stats["n_carried"] == stats["n_nodes"]
    assert stats["median_offset_pitch"] > 0.0, "the accepted steps are published"
    # The residual is the SAME criterion asked a second time at the bent positions:
    # one Newton step lands near the crest where the quadratic model is good, so it
    # must be much smaller than the step it replaced -- and it must not be exactly
    # zero, which would mean the measurement was never taken.
    assert 0.0 < stats["residual_median_pitch"] < stats["median_offset_pitch"]
    assert "scope" in stats


def test_gradify_connections_counts_refusals_without_hiding_them() -> None:
    """Every arc is accounted for under exactly one status."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=4.0, offset_pitch=1.5)
    world = tuple(_as_world(_geodesic_arc(a, b), None))
    out, stats = gradify_connections(
        [_FakeConnection(1, "geodesic", world)], pts, f, _R, _PITCH
    )
    assert out[0] is out[0] and not out[0].gradified
    assert out[0].arc_points == world, "a refused arc was rewritten"
    assert stats["n_arcs"] == 1
    assert stats["n_gradified"] == 0
    assert stats["n_all_refused"] == 1


def test_a_descent_arc_follows_a_valley_floor_not_the_ridge_above_it() -> None:
    """The regression behind a descending boundary that rendered as a great circle.

    A descent arc runs saddle -> MINIMUM, so it follows a valley FLOOR. The ridge
    rule reads a node's most-NEGATIVE eigenvector as its across-crest axis; on a
    valley floor that eigenvector is the ALONG direction, so the along-path gate
    refuses every free node and the arc is returned bit-identical to the geodesic
    it started as. ``polarity="minimum"`` asks the valley's own question instead.
    """
    pts = _fibonacci_sphere()
    valley = -_ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    world = _as_world(_geodesic_arc(a, b), None)

    # What the code did before `polarity` existed: the ascent rule, asked of a valley.
    default = gradify_arc(world, pts, valley, _R, _PITCH, n_nodes=12)
    assert not default.changed, "the ascent rule must not move a valley arc"
    assert default.status == "all_refused", default.status
    assert default.n_moved == 0 and default.n_carried == 0
    assert default.path == tuple(tuple(p) for p in world), "a refused arc was rewritten"

    # The same nodes, asked the valley's own question.
    descent = gradify_arc(
        world, pts, valley, _R, _PITCH, n_nodes=12, polarity="minimum"
    )
    assert descent.changed, "a descending arc must be gradified"
    assert descent.n_moved > 0
    assert descent.status == "ok"
    # The floor is the equator (latitude 0 pitch); the arc started 0.5 pitch off it
    # and its ends are pinned there, so the middle must reach the floor.
    assert min(_node_lats(descent)) < 0.25, _node_lats(descent)


def test_the_ascent_rule_refuses_every_node_of_a_valley_arc() -> None:
    """The mechanism, at the level the operator can read: the refusal reason.

    ``along_path`` is what the ascent rule answers with when its across-crest axis
    turns out to point down the path -- which is exactly what a valley floor does to
    the most-negative eigenvector. It is a re-routing refusal, not a "no ridge here"
    one, so a table full of them is a sign error, not a field that has nothing to say.
    """
    pts = _fibonacci_sphere()
    valley = -_ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    world = tuple(_as_world(_geodesic_arc(a, b), None))

    as_ridge = probe_gradient_ridge(world, pts, valley, _R, _PITCH)
    assert as_ridge.n_accepted == 0
    refused = [s.point.reason for s in as_ridge.samples if not s.point.accepted]
    assert refused, "the probe must report the samples it refused"
    assert "along_path" in refused, (
        "the ascent rule must answer a valley floor with its re-routing refusal: "
        f"{sorted(set(refused))}"
    )

    as_valley = probe_gradient_ridge(world, pts, valley, _R, _PITCH, polarity="minimum")
    assert as_valley.n_accepted > 0
    valley_reasons = [s.point.reason for s in as_valley.samples if not s.point.accepted]
    # ``no_crest`` at the arc's very ends is honest: the field is flat on a CP and
    # genuinely does not say where the floor is there. ``along_path`` must be gone.
    assert set(valley_reasons) <= {"no_crest"}, valley_reasons
    assert not any(r == "along_path" for r in valley_reasons)


def test_a_descent_arc_is_the_ascent_rule_on_the_negated_field() -> None:
    """One implementation, not two: ``polarity`` is a single negation.

    The descent path on ``-f`` must equal the ascent path on ``+f`` vertex for vertex.
    That is the strongest statement available that the two polarities cannot drift
    apart -- any future branch added to one side without its mirror fails here.
    """
    pts = _fibonacci_sphere()
    ridge = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    world = _as_world(_geodesic_arc(a, b), None)

    up = gradify_arc(world, pts, ridge, _R, _PITCH, n_nodes=10, polarity="maximum")
    down = gradify_arc(world, pts, -ridge, _R, _PITCH, n_nodes=10, polarity="minimum")
    assert up.changed and down.changed
    assert up.status == down.status == "ok"
    np.testing.assert_allclose(np.asarray(down.path), np.asarray(up.path), atol=1e-12)
    assert down.median_offset_pitch == pytest.approx(up.median_offset_pitch, abs=1e-12)


def test_gradify_connections_bends_a_valley_arc_only_with_the_descent_polarity() -> (
    None
):
    """The call-site contract: the descent table must pass ``polarity="minimum"``.

    Both production callers (``diagnose_params`` and the viewer) used to hand the
    descent table the ascent's rule, which is the whole of the reported symptom.
    """
    pts = _fibonacci_sphere()
    valley = -_ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=40.0, offset_pitch=0.5)
    world = tuple(_as_world(_geodesic_arc(a, b), None))

    out_default, stats_default = gradify_connections(
        [_FakeConnection(1, "geodesic", world)], pts, valley, _R, _PITCH, n_nodes=12
    )
    assert not out_default[0].gradified, "the ascent rule must leave the valley alone"
    assert stats_default["n_gradified"] == 0
    assert stats_default["n_all_refused"] == 1

    out, stats = gradify_connections(
        [_FakeConnection(1, "geodesic", world)],
        pts,
        valley,
        _R,
        _PITCH,
        n_nodes=12,
        polarity="minimum",
    )
    assert out[0].gradified is True
    assert out[0].gradify_moved > 0
    assert stats["n_gradified"] == 1
    assert stats["n_all_refused"] == 0
    assert stats["median_offset_pitch"] > 0.0


# ---------------------------------------------------------------------------
# quadratic Bézier fit: bending a geodesic onto the best integral, at once
# ---------------------------------------------------------------------------


def _fit_world(
    path: list[npt.NDArray[np.float64]],
    pts: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    center: tuple[float, float, float] | None = None,
    **kwargs: Any,
) -> QuadBezierFit:
    """Fit a centered fixture through the world-frame wrapper, at the fixture's R."""
    return fit_quad_bezier(
        arc_points=_as_world(path, center),
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
        **kwargs,
    )


def test_the_path_integral_of_a_constant_is_that_constant() -> None:
    """The objective's floor: a mean over a path cannot depend on the path."""
    pts = [np.array([math.cos(i * 0.3), math.sin(i * 0.3), 0.1 * i]) for i in range(8)]
    pts = [_unit(p) for p in pts]
    assert span_field_integral(pts, [2.5] * 8) == pytest.approx(2.5, abs=1e-12)
    assert span_field_integral([], []) == 0.0
    assert span_field_integral([pts[0]], [7.0]) == pytest.approx(7.0)


def test_the_path_integral_is_arc_length_weighted_not_node_weighted() -> None:
    """A node owns HALF of each segment touching it, so spacing matters.

    Sampling one side of a step function densely and the other side sparsely is the
    case a plain mean gets wrong: the mean of the samples says half-and-half, the
    integral along the path says mostly one side, and only the latter is a property
    of the CURVE. This is also why the objective cannot be gamed by adding nodes.
    """
    pts = [_unit(np.array([math.cos(t), math.sin(t), 0.0])) for t in (0.0, 0.5, 1.0)]
    # 1.0 on the first half of the arc (rows 0-1), 0.0 on the second.
    value = span_field_integral(pts, [1.0, 1.0, 0.0])
    assert value == pytest.approx(0.75, rel=1e-6), value


def test_the_quadratic_welds_both_endpoints_exactly() -> None:
    """The ladder's licence to exist: a fitted span starts and ends where it did.

    The 2t(1-t) blend vanishes at both ends, so the endpoints are the geodesic's own
    pins to machine precision at EVERY control. This is what lets a fitted arc keep
    the basin boundary's identity: the ring is keyed by (saddle, target) and welded
    at these points.
    """
    a, b = _offset_endpoints(span_deg=20.0, offset_pitch=0.5)
    basis = _quad_basis(a, b)
    assert basis is not None
    for du, dv in ((0.0, 0.0), (3.0 * _PITCH, 0.0), (0.0, 2.5 * _PITCH), (1.0, -2.0)):
        control = _quad_control(*basis, du * _PITCH, dv * _PITCH)
        curve = quad_bezier_points(a, b, control, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        assert np.allclose(curve[0], a, atol=1e-14), (du, dv)
        assert np.allclose(curve[-1], b, atol=1e-14), (du, dv)
        assert np.all(np.linalg.norm(curve, axis=1) == pytest.approx(1.0, abs=1e-12))


def test_the_straight_control_reproduces_the_great_circle() -> None:
    """The search's origin must be the arc we already have, exactly.

    Every accepted gain is measured against this control, so if C0 were anything
    other than the straight chord the gate would be comparing the bend to a fiction.
    """
    a, b = _offset_endpoints(span_deg=30.0, offset_pitch=0.5)
    basis = _quad_basis(a, b)
    assert basis is not None
    c0, _e1, _e2 = basis
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve = quad_bezier_points(a, b, c0, ts)
    normal = _plane_pitch([a, b])
    # The path, not the parameterisation: the 2t(1-t) blend pulls the SAMPLES toward the
    # span's middle (measured: 0.049 rad of slide at t=0.25), so the rows are not the
    # geodesic's own points at those parameters. What C0 guarantees -- and what the
    # baseline measurement needs -- is that the curve IS the great circle.
    assert np.allclose(curve[0], a, atol=1e-14)
    assert np.allclose(curve[-1], b, atol=1e-14)
    for point in curve:
        assert abs(float(np.dot(point, normal))) < 1e-12, (
            "C0 bent the curve off the plane"
        )


def test_a_broader_crest_pays_a_path_that_runs_along_it() -> None:
    """A monotone objective, checked directly: a broader crest pays a path that
    runs along it more, because the crest's WIDTH is the scale that sets how far
    off it a path can sit.

    The gate is a threshold on this number, so the number has to track the physics.
    The path is the latitude circle at 4 pitches, which runs parallel to the crest
    (the equator) at a constant distance: the narrower crest puts it well outside the
    falloff, the broader one puts it inside, and the integral orders them that way.
    """
    pts = _fibonacci_sphere()
    narrow = _ridge_field(pts, _Z, width_rad=2.0 * _PITCH)
    broad = _ridge_field(pts, _Z, width_rad=8.0 * _PITCH)
    path = _latitude_circle(math.pi / 2.0, 40, 2.0 * _PITCH, latitude_pitch=4.0)

    def mean(f: npt.NDArray[np.float64]) -> float:
        values = _sample_field(path, _R, pts, f, None)
        return span_field_integral(path, list(values))

    assert mean(broad) > mean(narrow) + 1e-3, (mean(broad), mean(narrow))
    assert 0.0 < mean(narrow) < mean(broad) <= 1.0, (mean(narrow), mean(broad))


def test_the_fit_bends_a_far_arc_the_local_step_cannot_reach() -> None:
    """Method 3's reason to exist, measured against method 2's published reach.

    The arc runs 3 mesh pitches from the crest: gradify's Newton step is capped at
    RIDGE_REACH_PITCH (0.72) and refuses every node, so the arc is returned
    bit-identical (the operator's complaint, in fixture form). The quadratic optimises
    the WHOLE path at once and its search is capped by its own knob, so it takes the
    bend, and the path's midpoint lands ON the crest.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=30.0, offset_pitch=3.0)
    arc = _geodesic_arc(a, b)
    world = _as_world(arc, None)

    local = gradify_arc(world, pts, f, _R, _PITCH, n_nodes=20)
    assert not local.changed, "the fixture stopped being out of the step's reach"
    assert local.status == "all_refused", local.status

    res = _fit_world(arc, pts, f, n_nodes=20, search_pitch=8.0, max_control_pitch=8.0)
    assert res.status == "ok", res.status
    assert res.changed
    assert res.n_spans == 1 and res.n_fitted == 1, (res.n_spans, res.n_fitted)
    assert res.max_offset_pitch > 1.0, "the bend is not a whole-path move"
    lats = [_lat_pitch(_local(np.asarray(q), None)) for q in res.path]
    assert min(abs(v) for v in lats) < 0.5, lats

    # The lever, measured: the search bound is the CONTROL's, and the span's middle
    # travels HALF as far (the blend peaks at 0.5). A 4-pitch search therefore cannot
    # cross the 3 pitches to this crest, and says so instead of pretending.
    short = _fit_world(arc, pts, f, n_nodes=20, search_pitch=4.0, max_control_pitch=4.0)
    assert short.max_offset_pitch <= 4.0 + 1e-12, short.max_offset_pitch
    short_lats = [_lat_pitch(_local(np.asarray(q), None)) for q in short.path]
    assert min(abs(v) for v in short_lats) > 1.0, short_lats
    assert res.mean_f_after > res.mean_f_before
    assert res.min_gain_fraction >= QUAD_FIT_MIN_GAIN_FRACTION


def test_the_fit_moves_out_of_the_great_circle_and_gradify_cannot() -> None:
    """The degree of freedom the ladder exists to buy, on a crest that needs it.

    A crest running 30 deg out of the arc's own plane touches the great circle at ONE
    point, so every other node of a straight arc is off it by a distance that grows
    linearly -- and no set of in-plane moves can fix it. The fit's `dv` (offset along
    the plane's normal) is the axis that answers, and the test reads it back out of the
    published control offsets.
    """
    pts = _fibonacci_sphere()
    arc0 = _great_circle_path(41, step_pitch=1.0)
    # A crest tilted 30 deg out of the arc's own plane, about the arc's end: it crosses
    # the equator at the arc's two ENDS, so every free node is off it, and the distance
    # is a latitude, which no move inside the equator's plane can reduce.
    pole = _unit(math.cos(math.radians(30.0)) * _Z + math.sin(math.radians(30.0)) * _Y)
    f = _ridge_field(pts, pole, width_rad=12.0 * _PITCH)
    _ = pole

    # Distance to the CREST, not latitude: the arc is the equator, which the tilted
    # crest crosses at its ends, and _lat_pitch measures the wrong great circle here.
    def crest_pitch(u: npt.NDArray[np.float64]) -> float:
        return (
            abs(math.asin(float(np.clip(float(np.dot(u, pole)), -1.0, 1.0)))) / _PITCH
        )

    straight_off = [crest_pitch(u) for u in arc0[1:-1]]
    # The crest crosses the arc's plane AT the arc's start, so the transverse distance
    # runs from ~0 to its maximum at the far end: the maximum is what the bend must
    # chase, and the near end is why a 0.72-pitch reach cannot chase it.
    assert max(straight_off) > 2.0, (
        f"fixture: the arc must run out of the crest, max {max(straight_off):.2f}"
    )
    assert min(straight_off) < 1.0, (
        f"fixture: the arc must not start on the crest, {min(straight_off):.2f}"
    )

    res = _fit_world(arc0, pts, f, n_nodes=20, search_pitch=8.0, max_control_pitch=8.0)
    assert res.changed, res.status
    assert res.n_fitted >= 1
    _du, dv = res.control_offsets[0]
    assert abs(dv) > 2.0, (
        f"the bend did not use the out-of-plane axis: {res.control_offsets}"
    )
    normal = _plane_pitch(arc0)
    off_plane = [
        abs(float(np.dot(_local(np.asarray(q), None), normal))) / _PITCH
        for q in res.path[1:-1]
    ]
    assert max(off_plane) > 1.0, "the fitted arc stayed in the arc's own plane"
    # The bend is a whole-path move, and the two ends are welded: a great circle reaching
    # this crest from the arc's mid-plane would have to leave the endpoints, which are
    # pinned. So the improvement is the ladder's alone, and gradify's reach (0.72 pitch)
    # cannot make it.
    local = gradify_arc(_as_world(arc0, None), pts, f, _R, _PITCH, n_nodes=20)
    assert not local.changed, "the fixture fell inside the Newton step's reach"


def test_the_fit_refuses_a_field_with_nothing_to_gain() -> None:
    """Zero gain is zero improvement: flat ground is left exactly straight.

    This is the property that keeps the ethene3 arcs (published: 0 of 6 bent, every
    node refused) unbent by the new mode, and it must be a measured refusal with the
    geometry returned verbatim, not an accident of the search.
    """
    pts = _fibonacci_sphere()
    f = np.ones_like(pts[:, 0])
    arc = _geodesic_arc(*_offset_endpoints(span_deg=30.0, offset_pitch=1.0))
    world = _as_world(arc, None)
    res = _fit_world(arc, pts, f, n_nodes=20)
    assert res.status == "no_field", res.status
    assert not res.changed
    assert res.path == tuple(tuple(p) for p in world), "a refused arc was rewritten"
    assert res.n_spans == 1 and res.n_rejected == 1, (res.n_spans, res.n_rejected)


def test_a_ripple_smaller_than_the_gate_does_not_move_the_arc() -> None:
    """Gate 3, at the constant's own scale: IDW ripple is not a crest.

    The threshold is an ABSOLUTE fraction of the field's robust span, measured on
    ethene3, Pd_20K and CO_plain precisely because a relative window makes a 1e-21
    ripple look like a 20% rise. A field with a 1e-4 ripple on a 1.0 plateau has a
    gain of ~1e-4/span, which is at the floor; the amplitude here is set below it.
    """
    pts = _fibonacci_sphere()
    rng = np.random.default_rng(1234)
    # The span must come from the FIELD, not from the ripple. On a plateau whose only
    # variation is 1e-6 of noise, the robust span IS the noise, and a fraction-of-span
    # gate compares a ripple against a ripple -- which is how a relative window lets
    # IDW dust through. So: a real crest of span ~1.0, the arc laid ON it (no bend can
    # gain), and the ripple on top.
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH) + 1e-6 * rng.standard_normal(
        len(pts)
    )
    arc = _geodesic_arc(*_offset_endpoints(span_deg=30.0, offset_pitch=0.0))
    world = _as_world(arc, None)
    res = _fit_world(arc, pts, f, n_nodes=20)
    assert res.field_span > 100.0 * 1e-6, res.field_span
    assert not res.changed, res.span_statuses
    assert res.status in ("rejected_gain", "mixed"), res.status
    assert res.path == tuple(tuple(p) for p in world)


def test_the_bend_is_capped_and_the_cap_is_the_policy_of_the_answer() -> None:
    """Gate 1: an unbounded integral maximiser runs to the nearest peak.

    The crest here is 10 pitches off the arc's midpoint, so the best bend inside the
    disc is the disc's own edge: the accepted offset saturates
    max_control_pitch and never exceeds it, and a control asked to travel further is
    refused outright. The cap is what keeps a shape correction a shape correction.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=30.0, offset_pitch=10.0))

    # A 1-pitch search does NOT stay home: a Gaussian crest of 12-pitch width gains
    # measurably from a 1-pitch bend (f: 0.50 -> 0.52), and the gate is 5e-4 of span, so
    # the bend is honest and accepted. What the cap guarantees is that it stops AT the
    # cap, and that a span 10 pitches from its crest is still 10 - 0.5 pitches away.
    tight = _fit_world(arc, pts, f, n_nodes=20, search_pitch=1.0, max_control_pitch=1.0)
    assert tight.n_fitted == 1, tight.span_statuses
    du, dv = tight.control_offsets[0]
    assert math.hypot(du, dv) == pytest.approx(1.0, abs=1e-9), (du, dv)
    lats = [abs(_lat_pitch(_local(np.asarray(q), None))) for q in tight.path[1:-1]]
    assert min(lats) > 9.0, f"the search overshot its own radius: {min(lats):.2f}"

    # The cap bites from ABOVE too, and it bites by CONSTRAINING the search rather than
    # by refusing the answer: `max_control_pitch < search_pitch` is the ordinary case, not
    # a corner, because the card's slider runs to 8.0 pitches against a policy cap of 4.0.
    # The first version of the fit bounded the search by the SLIDER, so every slider value
    # above the cap walked the control to a radius the acceptance rule then rejected, and
    # the method published n_quad_fitted = 0 with no hint that the knob had turned itself
    # off. Measured on the fixture above (search 3.0, cap 1.5): the old code refused the
    # span outright; now the offset stops at 1.494 pitches, the bend is accepted, and the
    # arc is 8.5 pitches from its crest -- still a polish, never a reroute.
    capped = _fit_world(
        arc, pts, f, n_nodes=20, search_pitch=3.0, max_control_pitch=1.5
    )
    assert capped.n_fitted == 1, capped.span_statuses
    du, dv = capped.control_offsets[0]
    assert math.hypot(du, dv) == pytest.approx(1.5, abs=1e-3), (du, dv)
    assert capped.max_offset_pitch <= 1.5 + 1e-9, "an over-cap offset was published"
    lats = [abs(_lat_pitch(_local(np.asarray(q), None))) for q in capped.path[1:-1]]
    assert min(lats) > 8.0, f"a wide search laundered a reroute: {min(lats):.2f}"
    # And a search NARROWER than the cap is bounded by the search, not the cap: the knob
    # still means something below the policy line.
    narrow = _fit_world(
        arc, pts, f, n_nodes=20, search_pitch=0.5, max_control_pitch=4.0
    )
    assert narrow.max_offset_pitch <= 0.5 + 1e-9, narrow.control_offsets


def test_the_fit_stays_out_of_a_cluster_it_never_entered() -> None:
    """Gate 1's named version: a CP the arc does not connect to is off-limits.

    A cluster's condensed ground is the highest on the sphere, and the integral is
    maximised by the peak of anything it can touch. Pinning (gate 2) protects the
    INSIDE of a disc the arc transits; this gate protects the discs it never entered,
    which is the case pinning cannot see. The fixture is the equator's own arc --
    whose great circle IS the crest, so the baseline is already optimal and no
    transverse move can gain -- lifted toward a spike that sits 12 pitches off the
    plane, inside a disc the straight arc misses by 7 pitches.
    """
    pts = _fibonacci_sphere()
    arc = _great_circle_path(40, step_pitch=1.0)
    # A cluster 4 pitches out of the arc's plane, with a disc of radius 3: the straight
    # arc misses it by a pitch, and the only way to drink its 40-unit peak is to bend
    # into it -- which is the move the gate exists to refuse.
    spike = _unit(
        math.radians(12.0) * _Z + np.asarray(arc[len(arc) // 2], dtype=np.float64)
    )
    f = _ridge_field(pts, _Z, width_rad=8.0 * _PITCH) + 40.0 * np.exp(
        -((np.arccos(np.clip(pts @ spike, -1.0, 1.0))) ** 2)
        / (2.0 * (4.0 * _PITCH) ** 2)
    )
    regions = [_Region(spike, 3.0 * _PITCH)]

    with_gate = _fit_world(arc, pts, f, regions=regions, search_pitch=8.0)
    assert with_gate.n_fitted == 0, with_gate.span_statuses
    assert with_gate.span_statuses == ("rejected_offroute",), with_gate.span_statuses
    assert not with_gate.changed

    # The same arc, with the disc NOT declared: the spike is fair game, so the
    # refusal above was the gate and not the field.
    without = _fit_world(arc, pts, f)
    assert without.n_fitted == 1, without.span_statuses
    assert without.changed


def test_the_fit_pins_cluster_ground_and_keeps_the_transit_straight() -> None:
    """Gate 2, and the CP-cluster promise of the plan, in one arc.

    A region sits on the arc's middle: its two crossings and every vertex inside it
    are pinned, so the transit through the excised ground stays on the great circle
    (the traced connector's bridge rule, in geometric form), and the fit optimises the
    two free spans on either side. A vertex put INSIDE the disc by interpolation would
    read the condensation's artifact as the ground's shape -- the reason the rule
    exists at all.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=40.0, offset_pitch=3.0))
    center = _unit(np.asarray(arc[len(arc) // 2], dtype=np.float64))
    radius = 5.0 * _PITCH
    res = _fit_world(
        arc,
        pts,
        f,
        n_nodes=20,
        regions=[_Region(center, radius)],
        search_pitch=4.0,
        max_control_pitch=8.0,
    )
    assert res.n_spans == 2, res.n_spans
    assert res.n_pinned >= 3, res.n_pinned
    assert res.status in ("ok", "mixed"), res.status
    lats = [_lat_pitch(_local(np.asarray(q), None)) for q in res.path]
    inside = [
        i
        for i, v in enumerate(res.path)
        if angular_distance(_local(np.asarray(v), None), center) <= radius
    ]
    assert inside, "the transit vanished"
    straight = [_lat_pitch(u) for u in arc]
    for i in inside:
        assert min(abs(lats[i] - s) for s in straight) < 0.02, (
            f"vertex {i} inside the disc left the great circle"
        )


def test_an_arc_that_is_all_pinned_ground_is_reported_as_pinned() -> None:
    """A region covering the whole arc leaves no control to move, and says so."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=10.0, offset_pitch=1.0))
    center = _unit(np.asarray(arc[len(arc) // 2], dtype=np.float64))
    res = _fit_world(arc, pts, f, regions=[_Region(center, 2.0)])
    assert res.status == "pinned", res.status
    assert res.n_spans == 0 and res.n_fitted == 0
    assert not res.changed
    assert res.path == tuple(tuple(p) for p in _as_world(arc, None))


def test_the_fit_descends_a_valley_with_the_descent_polarity() -> None:
    """The descent is the ascent's rule on the negated field, for the integral too.

    A saddle -> MINIMUM connector follows a valley FLOOR, so the objective is the
    MINIMUM of f along the path. Maximising f sends it up the flank: the mirror image
    of the polarity bug the basin code was built on. One negation at the top keeps a
    single implementation, so the two cannot drift.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    a, b = _offset_endpoints(span_deg=30.0, offset_pitch=3.0)
    arc = _geodesic_arc(a, b)
    ascent = _fit_world(
        arc, pts, f, n_nodes=20, search_pitch=6.0, max_control_pitch=6.0
    )
    descent = _fit_world(
        arc,
        pts,
        f,
        n_nodes=20,
        search_pitch=6.0,
        max_control_pitch=6.0,
        polarity="minimum",
    )
    assert ascent.changed and descent.changed, (ascent.status, descent.status)
    # High ground and low ground, on the same field: the crest is the equator here, so
    # the ascent pulls the span onto it and the descent pushes it down the flank.
    # The span's middle, not the path: both endpoints are PINNED at 3 pitches, so they
    # are in the emitted line whatever the fit decides, and min over the whole path is
    # their latitude.
    up = [abs(_lat_pitch(_local(np.asarray(q), None))) for q in ascent.path[3:-3]]
    down = [abs(_lat_pitch(_local(np.asarray(q), None))) for q in descent.path[3:-3]]
    assert min(up) < 2.0, f"the ascent rule did not climb to the crest: {min(up):.2f}"
    assert min(down) > 3.3, f"the descent rule did not leave the crest: {min(down):.2f}"
    assert ascent.path != descent.path, "the polarity changed nothing"
    # Both rise in their OWN units: the descent's published mean is the negated field,
    # so a higher number is lower ground. Reading these as f itself inverts the result.
    assert ascent.mean_f_after > ascent.mean_f_before
    assert descent.mean_f_after > descent.mean_f_before
    assert descent.mean_f_before < 0.0 < ascent.mean_f_before


def test_a_span_with_no_node_is_skipped_and_counted() -> None:
    """The node budget is a real limit, and a span that gets none says so.

    Nodes are allocated by arc length, so a budget smaller than the number of free
    spans leaves a span with nothing to measure: its only samples would be its two
    pins, whose field values do not depend on the control, so every candidate scores
    identically and the gain is exactly zero. It is skipped and counted, not silently
    dropped from the totals.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=60.0, offset_pitch=3.0))
    center = _unit(np.asarray(arc[len(arc) // 2], dtype=np.float64))
    res = _fit_world(
        arc,
        pts,
        f,
        n_nodes=1,
        regions=[_Region(center, 3.0 * _PITCH)],
        search_pitch=4.0,
        max_control_pitch=8.0,
    )
    assert res.n_spans == 2, res.n_spans
    assert "degenerate" in res.span_statuses, res.span_statuses
    assert res.n_fitted + res.n_rejected == res.n_spans
    assert len(res.control_offsets) == res.n_spans
    assert res.n_nodes >= 1


def test_degenerate_and_empty_inputs_are_refused_by_name() -> None:
    """A caller hands this every arc it has, without testing first."""
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=4.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=20.0, offset_pitch=1.0))
    world = _as_world(arc, None)

    short = _fit_world(arc[:1], pts, f)
    assert short.status == "degenerate" and not short.changed
    assert short.path == tuple(tuple(p) for p in _as_world(arc[:1], None))

    expected = tuple(tuple(p) for p in world)
    for res, why in (
        (
            fit_quad_bezier(world, pts, f, 0.0, _PITCH),
            "a sphere of radius 0",
        ),
        (
            fit_quad_bezier(world, pts, f, _R, 0.0),
            "a mesh of pitch 0",
        ),
        (
            fit_quad_bezier(world, np.zeros((0, 3)), f, _R, _PITCH),
            "no mesh",
        ),
        (
            fit_quad_bezier(world, pts, np.zeros(0), _R, _PITCH),
            "no field",
        ),
    ):
        assert res.status == "no_field", (why, res.status)
        assert not res.changed and res.path == expected


def test_an_off_origin_centre_matches_the_centred_frame() -> None:
    """The frame contract: the fit works on radials and exchanges world points.

    Pd_20K sits at the origin and cannot see this class of bug; ethene3 at 1.2618 A
    can. A control frame built from world positions would rotate with the atom.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=30.0, offset_pitch=3.0))
    center = (1.2618, -0.4, 0.7)

    centered = _fit_world(arc, pts, f, n_nodes=20, search_pitch=4.0)
    offset = _fit_world(arc, pts, f, center=center, n_nodes=20, search_pitch=4.0)
    assert centered.status == offset.status == "ok"
    assert np.allclose(
        offset.control_offsets[0], centered.control_offsets[0], atol=1e-9
    ), (offset.control_offsets, centered.control_offsets)
    # The centered run has no center to subtract: with center=None its world frame IS the
    # radial frame, which is exactly the contract this test checks.
    a_radial = np.asarray(centered.path[0])
    b_radial = np.asarray(offset.path[0]) - np.asarray(center)
    assert np.allclose(a_radial, b_radial, atol=1e-9), (a_radial, b_radial)
    assert np.allclose(
        np.asarray(centered.path[-1]),
        np.asarray(offset.path[-1]) - np.asarray(center),
        atol=1e-9,
    ), "the far weld moved with the center"
    assert max(
        abs(np.linalg.norm(np.asarray(q) - center)) for q in offset.path
    ) == pytest.approx(_R, rel=1e-9)


def test_the_search_is_a_bounded_number_of_field_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost of the mode is fixed by constants, not by the field's shape.

    A shape search over a spike map can degenerate into a global optimiser if the
    budget is open-ended, and the card hangs on a slider drag across 52 arcs. The
    bound is seeds x scan nodes + polish evaluations x full nodes, in ONE batched
    query per seed group; this counts the samples actually asked for.
    """
    pts = _fibonacci_sphere()
    f = _ridge_field(pts, _Z, width_rad=12.0 * _PITCH)
    arc = _geodesic_arc(*_offset_endpoints(span_deg=30.0, offset_pitch=3.0))

    def run(field: Any) -> int:
        calls = {"n": 0}
        real = ridge_refiner._sample_field

        def counting(
            directions: Any,
            radius: float,
            mesh: Any,
            values: Any,
            finder: Any = None,
        ) -> Any:
            calls["n"] += len(directions)
            return real(directions, radius, mesh, values, finder)

        monkeypatch.setattr(ridge_refiner, "_sample_field", counting)
        _fit_world(arc, pts, field, n_nodes=20, search_pitch=4.0)
        monkeypatch.undo()
        return calls["n"]

    samples = run(f)
    seeds = 1 + len(QUAD_FIT_SEED_RADII) * QUAD_FIT_SEED_ANGLES
    rows = 22  # 20 nodes + the span's two pins, the full grid the polish measures on
    scan_rows = 12  # the endpoints plus every QUAD_FIT_SCAN_STRIDE-th node
    budget = seeds * scan_rows + (QUAD_FIT_NM_MAX_ITERS + 3) * rows + 2 * rows
    assert samples <= budget, (samples, budget)
    # The budget is the constants, not a target: a span reaches its crest in a handful of
    # reflections and the simplex's noise-floor test stops it, so a quarter of the
    # allowance is unspent. 4x headroom on the scan row count is the batching slack.
    assert samples > 0.5 * budget, (samples, budget)

    # And the mode is affordable on a whole table: 52 arcs at the budget above is the
    # cost of one gradify pass, which is the comparison the card's slider depends on.
    assert samples < 2000, samples
