"""Piecewise Ridge Refinement for gradient-traced connectors.

A cyan connector is a steepest-ascent trajectory: it integrates
:math:`\\dot x = \\nabla f` in steps of one mesh pitch from a saddle's launch port
to a maximum. That is CLOSE to the ridge line joining the two critical points,
but it is not the ridge, for two separate reasons.

1. **It drifts off the crest.** Gradient ascent follows the local slope, and on a
   ridge the slope along the crest is small while the slope across it is large. A
   trajectory that starts a fraction of a pitch off-centre has no mechanism to
   return: nothing in :math:`\\nabla f` points at the crest once the path is on
   the flank, because the flank is where the gradient points. The result is a path
   that runs parallel to the ridge, slightly below it.
2. **It chatters in flat ground.** Where the ridge is wide and the field is
   nearly flat, the gradient is dominated by discretization. The path zigzags
   across the crest with a sawtooth whose teeth are one pitch long. The AVERAGE
   direction over several teeth is straight and correct; the point-to-point
   direction is noise.

The obvious fix -- smooth the polyline -- is wrong, because the path also contains
**real** sharp turns. Where the ridge itself has a corner (the ascent from MC16 to
MC12 on ethene3 curvedness), the contour lines that mark the ridge kink, and the
trajectory kinks with them. Blunting that corner with a spline draws a line that
does not exist in the field. The same applies where the path is carried straight
across an excision region: the endpoints of that carry are determined by the
boundary geometry, not by smoothing.

So the refinement is PIECEWISE, over a set of pinned nodes:

- **Anchors** are immovable. The launch port, the terminal maximum, and the centre
  of every excision region the path transits. These are positions the pipeline
  determined by other means and vouches for; a refinement that moved them would
  contradict the topology.
- **Knots** are the detected corners. They are real, so they stay sharp -- the
  refinement imposes no tangent continuity across them -- but the exact vertex
  where the bend happens is found in ground where the field is nearly flat, so it
  is the least trustworthy part of the corner. A knot is therefore allowed to
  shift within a small disc around its detected position, and it is the one node
  type that moves as a 2D point rather than along a 1D cross-track.
- **Sub-segments** between consecutive pinned nodes relax transversally: the crest
  is estimated at each interior vertex along the cross-track, those estimates are
  FITTED as one smooth field along the segment, and the fit is applied scaled by
  :math:`w(t) = \\sin(\\pi t)` so the displacement vanishes at both ends. The taper
  is what keeps the anchors pinned and the knots sharp without any special-casing,
  and it encodes the physical
  observation that the crest is best defined in the middle of a segment: at the
  saddle end the field is flat, at the maximum end the variation along the contour
  is slowly varying, and in between the ridge is dramatic.

Corner detection is the delicate part, and it is where the sawtooth has to be
separated from the corner. Both produce large point-to-point turning angles. The
discriminator is the WINDOW: over four or five steps the teeth cancel -- the
incoming and outgoing chords of a sawtooth are parallel -- while a real bend
persists. So the deflection is measured between chords spanning ``W`` steps, not
between adjacent segments.

The windowed deflection has a second, subtler problem, which is what the
:func:`detect_corners` excess test is for. A ridge that curves smoothly (a
latitude circle is the clean example) deflects by the SAME amount at every point
in the window. That is sustained curvature, not a corner, and an absolute
threshold on windowed deflection flags the whole arc as corners. A real corner is
a LOCALISED excursion: the deflection spikes at one vertex and returns to the
surrounding baseline. So a vertex is a corner when its deflection exceeds the
MEDIAN deflection of the path by ``CORNER_EXCESS_DEG``. The median is the
baseline of that path's own curvature, which makes the test scale-free: it does
not care whether the ridge is gently bowing or violently wiggling, only whether
this vertex bends more sharply than the path around it.

The crest estimate is fitted along the path for the same reason it is fitted across
the ridge. The field arrives through IDW, whose interpolant ripples at the mesh
scale with an amplitude comparable to the field change over one pitch, so the crest
estimated at one vertex disagrees with its neighbour by ~0.2 pitch. Applying those
estimates vertex by vertex injects the ripple into the geometry: measured on ethene3
curvedness it took the point-to-point turns above 35 deg from 6 to 15 and the total
turning from 628 to 1374 deg, for a field gain of 0.03% of span -- the refinement
made the path ROUGHER. A ridge does not wiggle at the mesh scale, so the offset of a
ridge from a path is a smooth function of position along it, and
:func:`_smooth_offsets` fits the per-vertex estimates as one field per sub-segment
before applying any of them. Roughness returns to exactly baseline and the field
gain is equal or better, because the raw version was discarding accuracy as well as
smoothness.

Frame contract: the refinement works in the CENTRED atom frame on unit radials,
and the polyline it is handed and returns is in the WORLD frame, because that is
what ``MaxConnection.arc_points`` carries. ``center`` converts between them. For a
centred atom the two coincide, which is why every frame bug in this project has
been invisible on Pd_20K -- so the tests use an off-origin centre.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from gba_topology2.src.excision_boundary import (
    KNearestFinder,
    interpolate_scalar_at_points,
    interpolate_scalar_at_points_local,
)
from gba_topology2.src.sphere_frame import (
    angular_distance,
    rotate_towards,
    to_global,
    to_local,
    unit,
)

__all__ = [
    "CORNER_EXCESS_DEG",
    "CORNER_WINDOW_PITCH",
    "DISPLACEMENT_FIT_DEGREE",
    "KNOT_MAX_PITCH",
    "NUDGE_MAX_PITCH",
    "NUDGE_SAMPLES",
    "RidgeRefinement",
    "chord_deflection_deg",
    "detect_corners",
    "refine_gradient_ridge",
]

# Half-width of the corner-detection window, in mesh-pitch units. The window must
# span several gradient steps so the sawtooth teeth cancel: a tooth is one pitch
# long, so a chord of four pitches averages four teeth. Shorter than this and the
# detector sees chatter; much longer and it starts to smear the corner it found,
# because the two chords straddling a wide window are no longer local to the bend.
CORNER_WINDOW_PITCH = 4.0

# How much a vertex's windowed deflection must exceed the path's MEDIAN deflection
# to count as a corner, in degrees. This is the localisation test described in the
# module docstring: sustained curvature raises the whole median, a corner raises
# one vertex. Measured on a latitude circle (pure sustained curvature) the excess
# is ~0 deg at every theta, so this constant is the gap between "the ridge bends
# here" and "the ridge curves".
CORNER_EXCESS_DEG = 35.0

# Transverse search half-width, in mesh-pitch units. One pitch is the finest
# structure the mesh represents, so a crest found more than ~1.5 pitches off the
# traced path is a different ridge, not a correction to this one.
NUDGE_MAX_PITCH = 1.5

# Cross-track samples per vertex (the search is over +/- NUDGE_MAX_PITCH, so this
# many steps each side plus the vertex itself).
NUDGE_SAMPLES = 6

# Degree of the polynomial fitted to the per-vertex crest offsets along a
# sub-segment, before they are applied. The estimate at one vertex is a local fit
# to a noisy interpolant and adjacent vertices disagree by ~0.2 pitch on the real
# fields, so applying them vertex by vertex injects that disagreement into the
# geometry. See :func:`_smooth_offsets`.
DISPLACEMENT_FIT_DEGREE = 2

# Radius of the disc a knot may move within, in mesh-pitch units. Smaller than the
# nudge width on purpose: the knot's position is the least certain quantity, and a
# knot that wanders drags both segments attached to it.
KNOT_MAX_PITCH = 1.0

# Polar grid used for the knot search: rings x spokes, evaluated at the knot's
# neighbourhood. Deterministic, derivative-free, and small enough to be cheap.
_KNOT_RINGS = 3
_KNOT_SPOKES = 8

# Minimum vertices a path must have to be refined. Below this there is no interior
# to relax and no window to measure.
_MIN_PATH = 5


@dataclass(frozen=True)
class RidgeRefinement:
    """The outcome of refining one gradient-traced connector.

    Attributes:
        path: The refined polyline, WORLD frame, start (launch port) first. Equal
            to the input when nothing moved.
        changed: True when any vertex moved. A path with no interior freedom --
            two anchors and nothing between them -- reports False, which is how a
            caller distinguishes "refined" from "nothing to do".
        corner_indices: Path indices promoted to knots, ascending.
        anchor_indices: Path indices pinned by construction: the two endpoints and
            the vertex nearest each transited excision centre.
        max_shift_deg: Largest angular displacement of any vertex. This is the
            cost of the refinement, published rather than hidden -- the same
            discipline as ``completed_leg_deg``.
        mean_shift_deg: Mean angular displacement over the interior vertices.
        n_moved: Number of vertices that moved.
    """

    path: tuple[tuple[float, float, float], ...]
    changed: bool
    corner_indices: tuple[int, ...] = field(default_factory=tuple)
    anchor_indices: tuple[int, ...] = field(default_factory=tuple)
    max_shift_deg: float = 0.0
    mean_shift_deg: float = 0.0
    n_moved: int = 0


def _path_units(
    points: Sequence[Sequence[float]],
    center: tuple[float, float, float] | None,
) -> npt.NDArray[np.float64]:
    """Convert a world-frame polyline to centred unit radials.

    Args:
        points: (M, 3) positions, world frame (centred when ``center`` is None).
        center: World-frame nucleus.

    Returns:
        (M, 3) float64 unit radials. Degenerate (coincident-with-centre) points
        keep a zero row; callers test for it.
    """
    out = np.zeros((len(points), 3), dtype=np.float64)
    for i, p in enumerate(points):
        arr = np.asarray(p, dtype=np.float64)
        local = np.asarray(to_local(arr, center), dtype=np.float64) if center else arr
        u = unit(local)
        if u is not None:
            out[i] = u
    return out


def _mean_step_rad(path_unit: npt.NDArray[np.float64]) -> float:
    """Mean angular step between consecutive vertices, in radians.

    Returns 0.0 for a path with fewer than two usable vertices, which every
    caller treats as "nothing to do".
    """
    if len(path_unit) < 2:
        return 0.0
    steps = [
        angular_distance(path_unit[i], path_unit[i + 1])
        for i in range(len(path_unit) - 1)
    ]
    positive = [s for s in steps if s > 0.0]
    return float(np.mean(positive)) if positive else 0.0


def _window_steps(
    path_unit: npt.NDArray[np.float64],
    delta_theta_mesh_rad: float,
    window_pitch: float,
) -> int:
    """Corner-detection window, in VERTEX steps.

    ``window_pitch`` is expressed in mesh-pitch units because that is the scale
    the sawtooth lives on -- a tooth is one pitch long. The path's own step is
    measured from its vertices, so a path sampled finer than the mesh (which the
    resampled arcs are) gets proportionally more vertices in the window, and the
    physical width of the window is what stays fixed.

    Args:
        path_unit: (M, 3) unit radials.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        window_pitch: Desired window half-width in mesh-pitch units.

    Returns:
        The window in vertices, at least 1 and at most ``M // 2`` so the chords
        stay inside the path.
    """
    if delta_theta_mesh_rad <= 0.0:
        return 1
    step = _mean_step_rad(path_unit)
    if step <= 0.0:
        return 1
    w = round(window_pitch * delta_theta_mesh_rad / step)
    return max(1, min(w, max(1, len(path_unit) // 2)))


def chord_deflection_deg(
    path_unit: npt.NDArray[np.float64],
    window_steps: int,
) -> list[float]:
    """Windowed turning angle at every vertex of a spherical polyline.

    At vertex ``i`` the deflection is the angle between the INCOMING chord
    ``p_i - p_{i-W}`` and the OUTGOING chord ``p_{i+W} - p_i``, both projected to
    the tangent plane at ``p_i``. Chords, not adjacent segments: over ``W`` steps
    the alternating teeth of a sawtooth cancel, so a path that zigzags while
    travelling straight has deflection near zero, while a genuine bend keeps its
    angle. That is the whole reason for the window.

    Args:
        path_unit: (M, 3) unit radials.
        window_steps: Chord half-length ``W`` in vertices.

    Returns:
        List of length M, in degrees. The ``W`` vertices at each end have no
        complete chord pair and are reported as 0.0, so the caller must ignore
        the margins -- :func:`detect_corners` does.
    """
    m = len(path_unit)
    w = max(1, int(window_steps))
    out = [0.0] * m
    for i in range(w, m - w):
        centre = path_unit[i]
        vin = unit(tangent_project_local(path_unit[i] - path_unit[i - w], centre))
        vout = unit(tangent_project_local(path_unit[i + w] - path_unit[i], centre))
        if vin is None or vout is None:
            continue
        dot = float(np.clip(float(np.dot(vin, vout)), -1.0, 1.0))
        out[i] = math.degrees(math.acos(dot))
    return out


def tangent_project_local(
    vector: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Tangential component of ``vector`` at the point with radial ``normal``.

    Thin alias for :func:`sphere_frame.tangent_project`, kept as a module-local
    name so the chord math reads as one operation.
    """
    v = np.asarray(vector, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    return v - float(np.dot(v, n)) * n


def detect_corners(
    path_unit: npt.NDArray[np.float64],
    delta_theta_mesh_rad: float,
    window_pitch: float = CORNER_WINDOW_PITCH,
    excess_deg: float = CORNER_EXCESS_DEG,
) -> tuple[tuple[int, ...], list[float], int]:
    """Find the vertices where the path's AVERAGE direction changes sharply.

    A vertex is a corner when its windowed deflection exceeds the median
    deflection of the path's interior by ``excess_deg``. The median is the
    baseline: a ridge that curves smoothly deflects every vertex in the window by
    about the same amount, and that is curvature, not a corner. Subtracting the
    baseline makes the test read the same on a straight path, a bowed path and a
    violently wiggling one.

    Local maxima only: a vertex must also be the largest deflection in its own
    window, so a corner promotes ONE vertex rather than a run of them, and the
    knot it becomes sits at the apex of the bend.

    Args:
        path_unit: (M, 3) unit radials.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to size the window.
        window_pitch: Window half-width in mesh-pitch units.
        excess_deg: Required rise above the baseline, in degrees.

    Returns:
        Tuple of (corner indices ascending, deflection list, window in vertices).
        Empty indices when the path is too short to carry a window.
    """
    m = len(path_unit)
    w = _window_steps(path_unit, delta_theta_mesh_rad, window_pitch)
    if m < 2 * w + 1:
        return (), [], w
    deflect = chord_deflection_deg(path_unit, w)
    interior = deflect[w : m - w]
    if not interior:
        return (), [], w
    baseline = float(np.median(interior))
    threshold = baseline + excess_deg
    corners: list[int] = []
    for i in range(w, m - w):
        here = deflect[i]
        if here <= threshold:
            continue
        span = deflect[i - w : i + w + 1]
        if here < max(span):
            continue
        if corners and i - corners[-1] < w:
            # Keep the sharper of two corners closer than a window apart; they are
            # one bend, and two knots a window apart leave no room to relax.
            if here > deflect[corners[-1]]:
                corners[-1] = i
            continue
        corners.append(i)
    return tuple(corners), deflect, w


def _transverse_frame(
    path_unit: npt.NDArray[np.float64],
    i: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """(radial, cross-track unit) at vertex ``i``.

    The along-path tangent is the central difference over the neighbours, which is
    the AVERAGE direction the windowed test just approved of -- so the cross-track
    is perpendicular to the smoothed path, not to the noisy local segment. That is
    what stops the nudge from chasing a sawtooth tooth.
    """
    r = path_unit[i]
    lo = path_unit[i - 1] if i > 0 else path_unit[i]
    hi = path_unit[i + 1] if i < len(path_unit) - 1 else path_unit[i]
    tau = unit(tangent_project_local(hi - lo, r))
    if tau is None:
        return None
    n = unit(np.cross(r, tau))
    if n is None:  # pragma: no cover - tau is tangent, so r x tau is never zero
        return None
    return r, n


def _offset_unit(
    origin_unit: npt.NDArray[np.float64],
    direction_unit: npt.NDArray[np.float64],
    angle_rad: float,
) -> npt.NDArray[np.float64] | None:
    """Step ``angle_rad`` along the great circle through ``origin_unit``, SIGNED.

    :func:`sphere_frame.rotate_towards` clamps a non-positive angle to a no-op and
    returns the origin. That is right for a forward extrapolation and fatal for a
    cross-track search: every negative sample collapses onto the vertex itself, so
    the search sees only one side of the ridge, and a path sitting on the high side
    of the crest finds no peak and never moves. The sign is therefore carried by the
    DIRECTION and the angle is passed as a magnitude.

    Args:
        origin_unit: Unit radial of the start point.
        direction_unit: Unit tangent giving the positive direction.
        angle_rad: Signed angular step in radians.

    Returns:
        The unit radial at ``|angle_rad|`` along ``sign(angle_rad) * direction``,
        or None when the rotation degenerates.
    """
    if angle_rad == 0.0:
        return origin_unit
    direction = direction_unit if angle_rad > 0.0 else -direction_unit
    moved = rotate_towards(origin_unit, direction, abs(angle_rad), 1.0)
    return unit(np.asarray(moved, dtype=np.float64))


def _sample_field(
    directions: Sequence[npt.NDArray[np.float64]],
    radius: float,
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    find_k_nearest: KNearestFinder | None,
) -> list[float]:
    """IDW the field at centred unit radials scaled to ``radius``."""
    if not directions:
        return []
    pts = np.array([np.asarray(d, dtype=np.float64) * radius for d in directions])
    if find_k_nearest is not None:
        vals = interpolate_scalar_at_points_local(pts, mesh_pts, f_vals, find_k_nearest)
    else:
        vals = interpolate_scalar_at_points(pts, mesh_pts, f_vals)
    return [float(v) for v in np.asarray(vals, dtype=np.float64)]


def _crest_offset(
    offsets: Sequence[float],
    values: Sequence[float],
) -> float | None:
    """Estimate the crest position of a cross-track profile, by least squares.

    A parabola is fitted to ``log f`` across the profile and its vertex returned.
    Three reasons this is the estimator and the peak sample is not:

    1. **Noise.** The field arrives through IDW, whose interpolant ripples at the
       mesh scale with an amplitude comparable to the field change over one pitch.
       Taking the largest sample selects the crest of that ripple, so the refinement
       chases interpolation noise -- the sawtooth, rebuilt inside the fix. A
       least-squares fit over the whole window averages it out.
    2. **Resolution.** The peak sample can only land on a sample, so the crest is
       quantised to the sample spacing. The parabola vertex is continuous.
    3. **Exactness.** A Gaussian ridge is a parabola in log space, so for the
       profile shape these fields actually have, the fit is not an approximation.

    Rejection is structural, not tuned: the fit must be CONCAVE (``a < 0``, so a
    crest exists in the window at all) and its vertex must lie INSIDE the sampled
    range (a crest beyond the window is not resolved by it, and clamping it to the
    edge would move the vertex by the full nudge width on the strength of no
    evidence). A flat window fails the first test; a window on one flank of a
    distant ridge fails the second.

    Args:
        offsets: Sample positions in radians, symmetric about 0.
        values: Field values at those positions.

    Returns:
        The signed crest offset in radians, or None when the window does not
        resolve a crest.
    """
    x = np.asarray(offsets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return None
    if float(np.min(y)) <= 0.0:
        # log is undefined; fall back to the raw values. A field that changes sign
        # across a ridge is not a ridge, so the fit degrades to "no crest".
        y = y - float(np.min(y)) + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ly = np.log(np.clip(y, 1e-300, None))
    if not np.all(np.isfinite(ly)):
        return None
    a, b, _ = np.polyfit(x, ly, 2)
    if not np.isfinite(a) or not np.isfinite(b) or a >= 0.0:
        return None
    crest = -b / (2.0 * a)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(crest) or crest < lo or crest > hi:
        return None
    return float(crest)


def _stationary_point(
    du: Sequence[float],
    dv: Sequence[float],
    values: Sequence[float],
    limit: float,
) -> tuple[float, float] | None:
    """Estimate a 2D crest position from a polar sample grid, by least squares.

    The two-dimensional counterpart of :func:`_crest_offset`: a full quadratic is
    fitted to ``log f`` over the neighbourhood and its stationary point returned.
    The same structural rejections apply -- the Hessian must be negative definite
    (the point is a local maximum, not a saddle or a minimum) and the point must
    lie within ``limit`` of the centre (a crest outside the search disc is not
    resolved by it).

    Args:
        du: Sample offsets along the path tangent, radians.
        dv: Sample offsets along the cross-track, radians.
        values: Field values at those samples.
        limit: Radius of the admissible disc, radians.

    Returns:
        ``(u, v)`` of the estimated crest, or None when the neighbourhood does not
        resolve one.
    """
    u = np.asarray(du, dtype=np.float64)
    v = np.asarray(dv, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(u) < 6 or not (len(u) == len(v) == len(y)):
        return None
    if float(np.min(y)) <= 0.0:
        y = y - float(np.min(y)) + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ly = np.log(np.clip(y, 1e-300, None))
    if not np.all(np.isfinite(ly)):
        return None
    basis = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    coef, *_ = np.linalg.lstsq(basis, ly, rcond=None)
    if not np.all(np.isfinite(coef)):
        return None
    _, c1, c2, c3, c4, c5 = (float(c) for c in coef)
    hess = np.array([[2.0 * c3, c4], [c4, 2.0 * c5]], dtype=np.float64)
    det = float(np.linalg.det(hess))
    if hess[0, 0] >= 0.0 or det <= 0.0:
        return None
    try:
        sol = np.linalg.solve(hess, -np.array([c1, c2], dtype=np.float64))
    except np.linalg.LinAlgError:  # pragma: no cover - det > 0 excludes this
        return None
    out = (float(sol[0]), float(sol[1]))
    if not all(math.isfinite(o) for o in out):
        return None
    if math.hypot(*out) > limit:
        return None
    return out


def _crest_offset_at(
    path_unit: npt.NDArray[np.float64],
    i: int,
    radius: float,
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    find_k_nearest: KNearestFinder | None,
    delta_theta_mesh_rad: float,
    max_pitch: float,
    samples: int,
) -> float | None:
    """Signed cross-track angle from vertex ``i`` to the estimated crest, radians.

    The search is 1D along the cross-track, over ``[-max_pitch, +max_pitch]`` mesh
    pitches, and the crest is the least-squares parabola vertex of the profile
    (see :func:`_crest_offset`), not its largest sample: the IDW interpolant
    ripples at the mesh scale, and picking the peak follows that ripple.

    This returns the OFFSET, not a position, so a caller can combine the estimates
    from a whole sub-segment before applying any of them -- which is what
    :func:`_smooth_offsets` is for.

    Returns:
        The signed offset in radians, or None when the vertex has no frame or its
        window resolves no crest inside it.
    """
    frame = _transverse_frame(path_unit, i)
    if frame is None:
        return None
    r, n = frame
    half = max_pitch * delta_theta_mesh_rad
    offsets = [-half + (2.0 * half) * k / samples for k in range(samples + 1)]
    dirs = [_offset_unit(r, n, o) for o in offsets]
    if any(d is None for d in dirs):  # pragma: no cover - unit of a rotated unit
        return None
    vals = _sample_field(
        [d for d in dirs if d is not None], radius, mesh_pts, f_vals, find_k_nearest
    )
    if len(vals) != len(dirs):
        return None
    return _crest_offset(offsets, vals)


def _smooth_offsets(
    fit_ts: Sequence[float],
    fit_values: Sequence[float],
    all_ts: Sequence[float],
    degree: int,
) -> list[float] | None:
    """Fit the per-vertex crest offsets as one smooth field along a sub-segment.

    Why this exists. The estimate at a single vertex is a local fit to a noisy
    interpolant, and on the real fields adjacent vertices disagree by ~0.2 pitch
    (median |d offset| = 0.18-0.22 on ethene3 curvedness, peaks to 1.67). Applying
    those estimates vertex by vertex injects that disagreement into the geometry:
    measured on the same field, the count of > 35 deg point-to-point turns AWAY
    from the corner went 6 -> 15 and the total turning 628 -> 1374 deg, for a field
    gain of 0.03% of span. The refinement made the path ROUGHER, which is the
    opposite of what was asked.

    The offset of a ridge from a path is a smooth function of position along the
    path -- a ridge does not wiggle at the mesh scale -- so the estimate is fitted,
    not applied. A least-squares polynomial averages the noise ALONG the path
    exactly as the parabola fit averages it ACROSS the ridge, and it fills the
    vertices whose own window resolved no crest, which is a second benefit: a
    missing local estimate is not evidence of zero displacement.

    The degree is capped by the number of usable samples so a short segment cannot
    be fitted exactly (a degree-n polynomial through n+1 points reproduces its
    noise and smooths nothing).

    Args:
        fit_ts: Normalised positions of the vertices with an estimate, in [0, 1].
        fit_values: Their crest offsets, radians.
        all_ts: Normalised positions to evaluate at, including vertices with no
            estimate of their own.
        degree: Preferred polynomial degree.

    Returns:
        The fitted offset at each entry of ``all_ts``, or None when there is
        nothing to fit.
    """
    ts = np.asarray(fit_ts, dtype=np.float64)
    vals = np.asarray(fit_values, dtype=np.float64)
    if len(ts) < 2 or len(ts) != len(vals):
        return None
    if int(degree) < 0:
        # No fit: apply each vertex's own estimate, and treat a vertex whose window
        # resolved no crest as zero displacement. This is the version that injected
        # mesh-scale noise, kept reachable so the smoothing can be A/B'd.
        by_t = {round(float(t), 9): float(v) for t, v in zip(ts, vals)}
        return [by_t.get(round(float(t), 9), 0.0) for t in all_ts]
    deg = max(min(int(degree), len(ts) - 2), 0)
    try:
        coef = np.polyfit(ts, vals, deg)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - deg >= 0
        return None
    fitted = np.polyval(coef, np.asarray(all_ts, dtype=np.float64))
    out = [float(v) for v in np.asarray(fitted, dtype=np.float64)]
    return out if all(math.isfinite(v) for v in out) else None


def _refine_knot(
    path_unit: npt.NDArray[np.float64],
    i: int,
    radius: float,
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    find_k_nearest: KNearestFinder | None,
    delta_theta_mesh_rad: float,
    max_pitch: float,
) -> npt.NDArray[np.float64] | None:
    """Seat a corner knot on the crest, within a small disc.

    A knot is a 2D node, not a 1D one: the corner's position is the least
    trustworthy quantity in the path, because the bend is found where the field
    along the ridge is nearly flat. So the search is over a polar grid on the
    tangent plane, and the objective is the field value -- the crest is the local
    maximum of ``f`` in every direction, including along the path.

    The disc is the constraint that keeps the corner sharp. A knot free to travel
    would slide to wherever the field is highest and drag both segments with it,
    which is exactly the blunting the piecewise design exists to prevent.

    Returns:
        The new unit radial, or None to leave the knot where it was detected.
    """
    r = path_unit[i]
    lo = path_unit[i - 1] if i > 0 else path_unit[i]
    hi = path_unit[i + 1] if i < len(path_unit) - 1 else path_unit[i]
    tau = unit(tangent_project_local(hi - lo, r))
    if tau is None:
        return None
    n = unit(np.cross(r, tau))
    if n is None:  # pragma: no cover
        return None
    half = max_pitch * delta_theta_mesh_rad
    du: list[float] = [0.0]
    dv: list[float] = [0.0]
    dirs: list[npt.NDArray[np.float64]] = [r]
    for ring in range(1, _KNOT_RINGS + 1):
        rad = half * ring / _KNOT_RINGS
        for spoke in range(_KNOT_SPOKES):
            ang = 2.0 * math.pi * spoke / _KNOT_SPOKES
            cu, cv = rad * math.cos(ang), rad * math.sin(ang)
            d = unit(cu * tau + cv * n)
            if d is None:  # pragma: no cover - tau, n are orthonormal
                continue
            moved = _offset_unit(r, d, rad)
            if moved is not None:
                du.append(cu)
                dv.append(cv)
                dirs.append(moved)
    clean = [d for d in dirs if d is not None]
    vals = _sample_field(clean, radius, mesh_pts, f_vals, find_k_nearest)
    if len(vals) != len(du):
        return None
    peak = _stationary_point(du, dv, vals, limit=half)
    if peak is None:
        return None
    u_off, v_off = peak
    dist = math.hypot(u_off, v_off)
    if dist <= 0.0:
        return None
    direction = unit(u_off * tau + v_off * n)
    if direction is None:  # pragma: no cover - dist > 0 keeps this off the axis
        return None
    return _offset_unit(r, direction, dist)


def _segment_bounds(
    m: int,
    pinned: Sequence[int],
) -> list[tuple[int, int]]:
    """Contiguous ``(lo, hi)`` index pairs of the sub-segments between pinned nodes.

    Args:
        m: Number of vertices.
        pinned: Ascending pinned indices, including 0 and m - 1.

    Returns:
        Pairs with at least one interior vertex, so a segment of two adjacent
        pinned nodes yields nothing to relax.
    """
    nodes = sorted({int(p) for p in pinned if 0 <= int(p) < m})
    if not nodes:
        nodes = [0, m - 1]
    if nodes[0] != 0:
        nodes.insert(0, 0)
    if nodes[-1] != m - 1:
        nodes.append(m - 1)
    return [(lo, hi) for lo, hi in pairwise(nodes) if hi - lo >= 2]


def refine_gradient_ridge(
    arc_points: Sequence[Sequence[float]],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    transited_centres: Sequence[Sequence[float]] = (),
    window_pitch: float = CORNER_WINDOW_PITCH,
    excess_deg: float = CORNER_EXCESS_DEG,
    nudge_max_pitch: float = NUDGE_MAX_PITCH,
    knot_max_pitch: float = KNOT_MAX_PITCH,
    fit_degree: int = DISPLACEMENT_FIT_DEGREE,
    passes: int = 2,
) -> RidgeRefinement:
    """Pull one gradient-traced connector onto the ridge, preserving its corners.

    The pipeline, per path:

    1. Detect corners by windowed chord deflection above the path's own baseline.
    2. Pin the endpoints, the corners, and the vertex nearest each transited
       excision centre.
    3. For each sub-segment between pinned nodes, estimate the crest offset at
       every interior vertex, FIT those offsets as one smooth field along the
       segment (see :func:`_smooth_offsets`), and apply the fit scaled by
       :math:`\\sin(\\pi t)` so the displacement vanishes at the nodes.
    4. Move each knot to the crest within a small disc, then relax the segments
       attached to it again, so the knot and its neighbours reach a joint position.

    Steps 3-4 iterate ``passes`` times: the knot wants to sit where the segments
    meet, and the segments want to meet at the knot.

    Args:
        arc_points: (M, 3) connector polyline, WORLD frame, launch port first.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on ``mesh_pts``.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus. None when the positions are already centred.
        find_k_nearest: Spatial index over ``mesh_pts``; strongly recommended.
        transited_centres: Centres of the excision regions the path crossed,
            WORLD frame (the region centres arrive centred; convert at the call
            site with ``to_global``). Each is pinned.
        window_pitch: Corner window half-width in mesh pitches.
        excess_deg: Corner threshold above the baseline, in degrees.
        nudge_max_pitch: Cross-track search half-width in mesh pitches.
        knot_max_pitch: Knot freedom radius in mesh pitches.
        fit_degree: Degree of the polynomial fitted to the crest offsets along a
            sub-segment before they are applied. 0 is a constant (the segment
            shifts rigidly across the ridge), -1 applies the raw per-vertex
            estimates, which is the version that injected mesh-scale noise.
        passes: Alternating knot / segment relaxation rounds.

    Returns:
        A :class:`RidgeRefinement`. The input is returned unchanged (``changed``
        False) for a path too short to refine, so a caller can hand it every
        connector without pre-filtering.
    """
    m = len(arc_points)
    if m < _MIN_PATH or sphere_radius <= 0.0 or delta_theta_mesh_rad <= 0.0:
        return RidgeRefinement(
            path=tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points),
            changed=False,
        )

    original = _path_units(arc_points, center)
    corners, _, _ = detect_corners(
        original, delta_theta_mesh_rad, window_pitch, excess_deg
    )

    pinned: set[int] = {0, m - 1}
    pinned.update(corners)
    for c in transited_centres:
        arr = np.asarray(c, dtype=np.float64)
        local = np.asarray(to_local(arr, center), dtype=np.float64) if center else arr
        u = unit(local)
        if u is None:
            continue
        d = [angular_distance(u, original[i]) for i in range(m)]
        pinned.add(int(np.argmin(d)))

    nodes = sorted(pinned)
    segments = _segment_bounds(m, nodes)
    if not segments:
        return RidgeRefinement(
            path=tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points),
            changed=False,
            corner_indices=corners,
            anchor_indices=tuple(nodes),
        )

    work = original.copy()
    knot_set = set(corners)
    for _ in range(max(1, passes)):
        # JACOBI, not Gauss-Seidel: every frame, chord and cross-track in this pass
        # is read from the snapshot, and every displacement is written to `work`.
        # Reading the live array makes each vertex's cross-track depend on where its
        # neighbour just moved, so a chain of small tilts compounds into a large
        # drift -- measured on a path already sitting on the crest, an in-place pass
        # walked it 4.3 deg off a ridge it had no reason to leave. A string method
        # relaxes against a fixed geometry; that is what keeps the relaxation a
        # correction rather than a random walk.
        base = work.copy()
        for lo, hi in segments:
            span = hi - lo
            free = [i for i in range(lo + 1, hi) if i not in knot_set]
            if not free:
                continue
            all_ts = [(i - lo) / span for i in free]
            est_ts: list[float] = []
            est_offs: list[float] = []
            for i, t in zip(free, all_ts):
                crest = _crest_offset_at(
                    base,
                    i,
                    sphere_radius,
                    mesh_pts,
                    f_vals,
                    find_k_nearest,
                    delta_theta_mesh_rad,
                    nudge_max_pitch,
                    NUDGE_SAMPLES,
                )
                if crest is None:
                    continue
                est_ts.append(t)
                est_offs.append(crest)
            fitted = _smooth_offsets(est_ts, est_offs, all_ts, fit_degree)
            if fitted is None:
                continue
            for i, t, off in zip(free, all_ts, fitted):
                frame = _transverse_frame(base, i)
                if frame is None:
                    continue
                r, n = frame
                taper = math.sin(math.pi * t)
                if taper <= 0.0:
                    continue
                # The fitted offset is a displacement in RADIANS along the
                # cross-track; the taper scales it so it vanishes at the nodes.
                # Clamped to the search width: a fit extrapolating past the
                # window at a segment end is not a measurement.
                limit = nudge_max_pitch * delta_theta_mesh_rad
                moved = _offset_unit(r, n, max(-limit, min(limit, taper * off)))
                if moved is not None:
                    work[i] = moved
        for i in corners:
            moved = _refine_knot(
                base,
                i,
                sphere_radius,
                mesh_pts,
                f_vals,
                find_k_nearest,
                delta_theta_mesh_rad,
                knot_max_pitch,
            )
            if moved is not None:
                work[i] = moved

    shifts = [math.degrees(angular_distance(original[i], work[i])) for i in range(m)]
    interior = [s for i, s in enumerate(shifts) if i not in (0, m - 1)]
    out: list[tuple[float, float, float]] = []
    for u in work:
        pos = (
            float(u[0] * sphere_radius),
            float(u[1] * sphere_radius),
            float(u[2] * sphere_radius),
        )
        out.append(to_global(pos, center) if center else pos)
    return RidgeRefinement(
        path=tuple(out),
        changed=any(s > 1e-9 for s in shifts),
        corner_indices=corners,
        anchor_indices=tuple(nodes),
        max_shift_deg=float(max(shifts, default=0.0)),
        mean_shift_deg=float(np.mean(interior)) if interior else 0.0,
        n_moved=sum(1 for s in shifts if s > 1e-9),
    )
