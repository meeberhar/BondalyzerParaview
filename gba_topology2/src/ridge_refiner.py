"""Sawtooth Smoothing for gradient-traced connectors.

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

This module fixes (2), the chatter, and is honest that it does not fix (1). The
first refinement built here (Entry 058) attacked (1) by nudging every vertex
CROSS-TRACK onto a crest estimate; it failed inspection because a cross-track
shift cannot remove a tooth -- the tooth IS the cross-track motion -- and the
crest estimate on a noisy IDW interpolant moved the path by a fraction of a pitch
that was invisible on screen. The probe (:func:`probe_gradient_ridge`) remains as
the diagnostic that measures (1): it draws the ridge the FIELD defines beside the
unchanged path, so the operator can see how far off-crest the ascent runs. The
smoother here (:func:`smooth_sawtooth_runs`) removes (2) as pure geometry, with no
field at all.

The obvious fix for chatter -- smooth the whole polyline -- is wrong, because the
path also contains **real** sharp turns. Where the ridge itself has a corner (the
ascent from MC16 to MC12 on ethene3 curvedness), the contour lines that mark the
ridge kink, and the trajectory kinks with them. Blunting that corner with a spline
draws a line that does not exist in the field. The same applies where the path is
carried straight across an excision region: the endpoints of that carry are
determined by the boundary geometry, not by smoothing.

So the smoothing is PIECEWISE, over a set of pinned nodes:

- **Anchors** are immovable. The launch port, the terminal maximum, the centre of
  every excision region the path transits, and the detected corners. These are
  positions the pipeline determined by other means and vouches for; a smoothing
  that moved them would contradict the topology.
- **Knots** are the detected corners. They are real, so they stay sharp -- the
  curve is built to pass through a knot and is free to kink there, imposing no
  tangent continuity across it. That is the property that makes a bend survive.
- **Chatter runs** are maximal stretches of consecutive teeth between two anchors.
  Each is deleted and replaced by ONE spherical Catmull-Rom (SQUAD) curve through
  its two bounding anchors. The curve's control point at an anchor is built from
  the anchor's neighbours IN THE ANCHOR POLYGON, not the raw path -- a Catmull-Rom
  through the teeth reproduces the teeth, while one through the nodes the teeth
  hang between draws the midline they oscillate about. The anchor at each end of a
  run is the MIDPOINT of the bounding edge, not the vertex: consecutive teeth sit
  at +amp and -amp, so the midline is crossed at the edge midpoint, and pinning to
  the vertex would leave the whole curve a tooth off the crest.

Only the runs are resampled, at a fixed arc spacing finer than the mesh
(:data:`SMOOTH_OUTPUT_PITCH`), so the smoothed line is visibly finer than the
traced one it replaces while every stretch the field was never in question keeps
its original vertices.

Chatter detection is the delicate part, and it is where the sawtooth has to be
separated from the corner. Both produce large point-to-point turning angles. The
discriminator is the WINDOW: over four or five steps the teeth cancel -- the
incoming and outgoing chords of a sawtooth are parallel -- while a real bend
persists. So a vertex is a TOOTH when its point-to-point turn is large (it kinks
here) AND its windowed chord deflection is not (the average direction does not
kink). A corner is the opposite: large in both. :func:`detect_corners` finds the
corners by the excess-over-median test below; :func:`detect_chatter_runs` finds
the teeth, and a knot is never a tooth.

The windowed deflection has a second, subtler problem, which is what the
:func:`detect_corners` excess test is for. A ridge that curves smoothly (a
latitude circle is the clean example) deflects by the SAME amount at every point
in the window. That is sustained curvature, not a corner, and an absolute
threshold on windowed deflection flags the whole arc as corners. A real corner is
a LOCALISED excursion: the deflection spikes at one vertex and returns to the
surrounding baseline. So a vertex is a corner when its deflection exceeds the
MEDIAN deflection of its NEIGHBOURHOOD by ``CORNER_EXCESS_DEG``. The median is
the baseline of that path's own curvature, which makes the test scale-free: it
does not care whether the ridge is gently bowing or violently wiggling, only
whether this vertex bends more sharply than the path around it.

The baseline is local -- a median over ``CORNER_BASELINE_MULT`` windows, not over
the whole path -- because a global median is one statistic for a path that may not
be one thing. The ethene3 shape-index connector that spirals half a turn inside
MC2 has 162 of 441 vertices deflecting above 150 deg; a global baseline promoted
31 knots inside a region whose field is declared untrustworthy by construction,
and the real 43 deg bend at the launch port was judged against the spiral and
missed. Both failures at once, from one statistic.

Frame contract: the smoothing works in the CENTRED atom frame on unit radials,
and the polyline it is handed and returns is in the WORLD frame, because that is
what ``MaxConnection.arc_points`` carries. ``center`` converts between them. For a
centred atom the two coincide, which is why every frame bug in this project has
been invisible on Pd_20K -- so the tests use an off-origin centre.

The probe (:func:`probe_gradient_ridge`) shares the corner detection, the pinned
segment structure and the frame contract, and measures the crest (1) that the
smoother leaves alone. It asks the FIELD for its ridge: fit log f over a wide
disc, take the eigenvector of the most-negative Hessian eigenvalue -- the
across-crest direction, a property of the field, not of the path -- and Newton-step
the sample to the point where the gradient has no component along the ridge. That
is the contour curvature extremum the operator identified by eye, and it is
localised even where the field is flat-topped, because a zero crossing of the
tangential derivative is sharp where an argmax of f is not. The probe draws; it
does not move.
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
    squad_control,
    squad_point,
    to_global,
    to_local,
    unit,
)

__all__ = [
    "CORNER_BASELINE_MULT",
    "CORNER_EXCESS_DEG",
    "CORNER_WINDOW_PITCH",
    "CURVE_OUTPUT_PITCH",
    "NODE_PITCH",
    "RIDGE_REACH_PITCH",
    "RIDGE_STENCIL_PITCH",
    "SEGMENT_SAMPLE_DEG",
    "SMOOTH_MIN_TEETH",
    "SMOOTH_OUTPUT_PITCH",
    "TOOTH_TURN_DEG",
    "RidgeProbe",
    "RidgeRefinement",
    "RidgeSample",
    "SawtoothSmoothing",
    "StructuralCurve",
    "chord_deflection_deg",
    "coarse_structural_curve",
    "detect_chatter_runs",
    "detect_corners",
    "point_turn_deg",
    "probe_gradient_ridge",
    "ridge_point_at",
    "smooth_gradient_ridge",
    "smooth_sawtooth_runs",
    "structural_curve_ridge",
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

# Half-width of the baseline neighbourhood, in corner windows. The baseline a
# corner must exceed is the median deflection of the path NEAR the vertex, not of
# the whole path: see :func:`detect_corners`. Sized in windows so it scales with
# the same logic the window does. Wide enough that a genuine sustained curve
# (a latitude circle, where every vertex deflects alike) still raises its own
# baseline and yields no corners -- measured identical to the global median on
# that fixture -- and narrow enough that a spiral confined to one part of the
# path cannot set the bar for the rest of it.
CORNER_BASELINE_MULT = 6.0

# --- ridge probe -----------------------------------------------------------
#
# The probe MEASURES where the ridge is; it never moves the path. It exists
# because the Entry-058 refinement failed inspection on the real field: the
# cross-track parabola estimator is a weak criterion (a gradient path is already
# at a local max of the noisy interpolant in the cross-track sense, so the
# estimate returns roughly where the path is -- measured median movement 0.06-0.40
# pitch), and its 0.5-pitch sampling sits at the IDW ripple's own scale. The
# probe replaces the criterion with the field's own curvature geometry (see
# :func:`ridge_point_at`) so the operator can SEE the ridge the algorithm found,
# as points, before any line is moved to it.

# Arc spacing of the probe samples along a sub-segment, in degrees. Five to six
# degrees is two to three mesh pitches on the real atoms -- fine enough to trace
# the crest, coarse enough that a 30-60 deg connector carries a readable handful
# of points.
SEGMENT_SAMPLE_DEG = 5.5

# Radius of the disc over which the field is fitted to get the gradient and
# Hessian, in mesh-pitch units. WIDE on purpose: the Hessian is a second-derivative
# object, and a stencil of ~2.5 pitches averages the mesh-scale IDW ripple out of
# it, where the 0.5-pitch cross-track samples of the old estimator were dominated
# by its phase.
RIDGE_STENCIL_PITCH = 2.5

# The Newton step to the crest is accepted up to this many pitches from the
# sample. A crest further away is a different ridge: the probe reports a
# correction, not a re-routing.
#
# MEASURED, then multiplied by 1.5 at the operator's instruction. Every sample of
# every gradient connector on ethene3 curvedness was fitted and stepped (110
# samples), and the step size separated cleanly by whether the crest it lands on is
# real -- gain over the sample, as a fraction of the field's span:
#
#   real crest (gain > 0.05% span, n=36): |step| median 0.88 deg, p90 0.92,
#                                         MAX 1.07 deg = 0.48 pitch
#   ripple     (gain < 0.05%,     n=74): |step| median 1.19 deg, MAX 2.39 deg,
#                                         gain median +0.000% -- no ridge at all
#
# The farthest a real ridge ever lies from the path is 1.07 deg, so the bound is
# 1.5 x 1.07 = 1.61 deg, which on this atom's 2.2278 deg pitch is 0.72 pitch. The
# constant is in PITCHES so the bound travels with the mesh: it is a statement about
# resolution, not about degrees.
#
# "1.5 x" is read as the search RADIUS from the path, which is what keeps the
# 1.07 deg sample inside with margin. Read as a full DIAMETER (reach 0.36 pitch)
# the bound would reject the very sample that defines it.
#
# Note the ripple population steps FARTHER than the real one (2.39 vs 1.07 deg).
# That is the signature of the flat region: with no crest nearby, the quadratic fit
# extrapolates, and a long step there is noise, not a distant ridge. The reach
# rejects it, and the gain gate rejects what survives.
RIDGE_REACH_PITCH = 0.72

# The step direction must be at least this cross-track: the eigenvector of the
# most-negative Hessian eigenvalue is the ACROSS-crest direction when the sample
# sits on a flank, and when it points mostly ALONG the path the local geometry is
# not a ridge the sample can step onto. sin(45 deg).
_RIDGE_MAX_ALONG_COS = math.sin(math.radians(45.0))

# Rings and spokes of the stencil disc (plus the centre): 25 field samples per
# probe point, one least-squares quadratic over them.
_STENCIL_RINGS = 3
_STENCIL_SPOKES = 8

# A crest on the cross-track scan must stand this fraction of the window's own
# value range above its neighbouring troughs to count. This is the ripple gate,
# and it is RELATIVE for the same reason the corner test is: a ridge on a curved
# path, a ridge on a flat one, and a ridge on a violently wiggling one all present
# the same prominence as a fraction of their local window, so one number serves
# every field. Measured on ethene3 curvedness: in the developed region (f > 25% of
# span) the cross-track profile has EXACTLY ONE crest in +/-6 deg at every sample,
# so the gate never fires there; in the flat region near the saddle (f < 5% of
# span) it finds 6-9 "crests" whose values agree to four decimals -- the IDW
# ripple -- and the gate is what stops the probe from stepping onto them.
_CREST_PROMINENCE_FRACTION = 0.02

# Scan resolution along the cross-track line, in radians. A quarter pitch
# resolves a crest whose width is a pitch, which is the finest structure the mesh
# carries; the parabola refinement below then places it continuously.
_CREST_SCAN_STEP_PITCH = 0.25

# A crest is only worth stepping to if it stands this fraction of the FIELD'S SPAN
# above the sample. The scan's prominence gate is relative to its own window, and
# in the flat ground near a saddle the window's entire range IS the IDW ripple, so
# a relative gate waves the ripple through. Measured on ethene3 curvedness, the
# gain of the best crest within reach over the sample's own value, as a fraction of
# span: f < 5% of span (n=82) median +0.0008, max +0.156 -- noise; f in 5-25%
# (n=22) min +0.167, median +10.1 -- a ridge; f > 25% (n=15) min +7.28 -- a ridge.
# The two populations do not overlap, so 0.05% rejects 81 of the 82 noise
# candidates and keeps every real one. This is an ABSOLUTE, physical gate: "the
# ridge must be a measurable rise in this field", not "a bump in this window".
_CREST_MIN_GAIN_FRACTION = 0.0005

# Polar grid used for the knot search: rings x spokes, evaluated at the knot's
# neighbourhood. Deterministic, derivative-free, and small enough to be cheap.
_KNOT_RINGS = 3
_KNOT_SPOKES = 8

# Minimum vertices a path must have to be refined. Below this there is no interior
# to relax and no window to measure.
_MIN_PATH = 5

# --- sawtooth smoothing ----------------------------------------------------
#
# The refinement that replaces the transverse nudge. Entry 058 moved the path
# CROSS-TRACK onto a crest estimate and the teeth survived, because a cross-track
# correction cannot remove a tooth: the tooth IS the cross-track motion. The teeth
# are removed by replacing the zigzag between two pinned nodes with ONE smooth
# curve through the nodes, which is a change to the path's shape along its length,
# not a shift across it.

# A vertex counts as a TOOTH when the point-to-point turn there exceeds this many
# degrees. The scale is the mesh: on ethene3 curvedness the traced path steps one
# pitch (2.23 deg) per vertex and the chatter swings +/-1.5 deg across the crest,
# which is a point-to-point turn of 60-90 deg. A real ridge turn is spread over
# many vertices and gives a small per-vertex turn, so this threshold separates the
# two WITHOUT a window -- the window is what the corner test uses, and it is
# deliberately NOT used here.
TOOTH_TURN_DEG = 30.0

# How far above the path's own baseline deflection a vertex may sit and still be
# chatter. A tooth has a huge point-to-point turn and a windowed deflection AT the
# baseline (the chords cancel); a corner has a deflection ABOVE it. A vertex that is
# both a sharp turn and a corner is a corner -- corners win. This is the margin that
# keeps a real bend from being smoothed away.
_CHATTER_MAX_EXCESS_DEG = CORNER_EXCESS_DEG * 0.5

# A run of fewer than this many teeth is left alone. One zigzag between two nodes is
# a single vertex's noise, not a sawtooth; replacing it with a curve is a coin flip
# about the field, and the module's rule is that a correction needs a population.
SMOOTH_MIN_TEETH = 3

# Arc spacing of the output curve, in mesh-pitch units. 0.45 pitch is 1.0 deg on
# ethene3's 2.23 deg pitch: finer than the mesh, so the smooth line is visibly
# finer than the traced one it replaces, and coarse enough that a 30 deg leg carries
# ~30 points rather than a thousand. In PITCHES so the density travels with the mesh
# resolution, the same reason RIDGE_REACH_PITCH is in pitches.
SMOOTH_OUTPUT_PITCH = 0.45

# --- coarse structural curve ------------------------------------------------
#
# The refinement the operator asked for after inspecting the shape-index field:
# "cant we just pick a number of points along these boundaries and put a smooth
# curve through them... a smooth boundary is essential to the ultimate visual 3D
# representation we will ultimately generate so its exact path is not what we
# ultimatly want."
#
# THE SPACING IS NOT A DENSITY KNOB, IT IS A LOW-PASS FILTER'S CUTOFF. Measured on
# the ethene3 shape-index connector that spirals half a turn inside MC2 (441
# vertices, total turning 66010 deg), one SQUAD through nodes spaced k pitches:
#
#   k =  1.0 pitch -> 3513 pts,   65935 deg turning, 0.34 deg from traced
#   k =  4.0 pitch ->  193 pts,    1806 deg turning, 1.60 deg from traced
#   k =  8.0 pitch ->   89 pts,     651 deg turning, 2.66 deg from traced
#   k = 12.0 pitch ->   65 pts,     532 deg turning, 2.77 deg from traced
#
# At 1 pitch the curve reproduces the spiral almost exactly (65935 of 66010 deg): a
# faithful copy of a path that should not be followed. At 8 the winding is averaged
# away, 100x less turning. SMOOTH_OUTPUT_PITCH = 0.45 sits BELOW the spiral's own
# scale, which is exactly why the Entry-061 smoother "helped to eliminate the saw
# tooth pattern but was applied over a too small a region".

# Spacing of the structural nodes, in mesh pitches. The shape of the curve is
# carried by the nodes; everything between them is interpolation. 8 pitches is the
# operator's choice, and the measurement above is what it buys.
NODE_PITCH = 8.0

# Arc spacing of the emitted polyline, in mesh pitches. Coarser than the traced
# path's own one-pitch steps: the record is being replaced by a rendering, and a
# curve whose control points are 8 pitches apart needs tessellation at a fraction of
# that and nothing more. 1 pitch gives 8 segments per node span, which is visually
# smooth and 4x coarser than the Entry-061 output.
CURVE_OUTPUT_PITCH = 1.0


@dataclass(frozen=True)
class RidgeRefinement:
    """The outcome of smoothing one gradient-traced connector.

    The world-frame result of :func:`smooth_gradient_ridge`. The vertex-count of
    ``path`` differs from the input's (runs are resampled at a finer arc spacing),
    so the cost is reported as a deviation from the traced polyline, not a
    per-vertex shift.

    Attributes:
        path: The smoothed polyline, WORLD frame, start (launch port) first. Equal
            to the input when nothing was smoothed.
        changed: True when any run was replaced. A path with no chatter reports
            False, which is how a caller distinguishes "smoothed" from "nothing to
            do".
        corner_indices: Path indices promoted to knots, ascending. Pinned; the
            curve passes through each and is free to kink there.
        anchor_indices: Input indices of every node the curve is pinned to: the two
            endpoints, every knot, the vertex nearest each transited excision
            centre, and the entry/exit vertex of each run.
        n_moved: Input vertices the curve replaced (the teeth).
        n_runs: Chatter runs the fidelity smoother replaced. Zero for the coarse
            curve, which detects nothing and replaces the whole path.
        n_nodes: Structural nodes the coarse curve passes through. Zero for the
            fidelity smoother, which has no node set.
        total_turn_before_deg: Sum of point-to-point turns over the input path.
        total_turn_after_deg: The same over the output. The gap is the chatter the
            smoother removed -- the number the operator is judging.
        max_deviation_deg: Largest angular distance from any output vertex to the
            INPUT polyline. The cost of the smoothing, published rather than hidden
            -- the same discipline as ``completed_leg_deg``.
    """

    path: tuple[tuple[float, float, float], ...]
    changed: bool
    corner_indices: tuple[int, ...] = field(default_factory=tuple)
    anchor_indices: tuple[int, ...] = field(default_factory=tuple)
    n_moved: int = 0
    n_runs: int = 0
    n_nodes: int = 0
    total_turn_before_deg: float = 0.0
    total_turn_after_deg: float = 0.0
    max_deviation_deg: float = 0.0


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
    baseline_mult: float = CORNER_BASELINE_MULT,
) -> tuple[tuple[int, ...], list[float], int]:
    """Find the vertices where the path's AVERAGE direction changes sharply.

    A vertex is a corner when its windowed deflection exceeds the median deflection
    of its NEIGHBOURHOOD by ``excess_deg``. The median is the baseline: a ridge that
    curves smoothly deflects every vertex in the window by about the same amount, and
    that is curvature, not a corner. Subtracting the baseline makes the test read the
    same on a straight path, a bowed path and a violently wiggling one.

    The baseline is LOCAL -- the median over ``baseline_mult`` windows on either
    side, not over the whole path. A global median is a statistic computed over the
    entire path, and a path that is not globally homogeneous defeats it in BOTH
    directions at once. Measured on the ethene3 shape-index connector that spirals
    0.5 turns inside MC2: the winding has a windowed deflection above 150 deg at 162
    of 441 vertices, so a global baseline promoted 31 knots inside a region whose
    field is declared untrustworthy by construction, while the real bend at the MS3
    launch port (43 deg, against a global threshold of 38.2) was judged against the
    spiral and missed. A local baseline gives that path 5 knots, all at the spiral's
    entry and exit, and leaves the curvedness connectors UNCHANGED (verified on all
    four: 1, 1, 3, 3 knots, identical indices) -- which is the property that makes
    the change safe: on a homogeneous path the local median IS the global one.

    Local maxima only: a vertex must also be the largest deflection in its own
    window, so a corner promotes ONE vertex rather than a run of them, and the
    knot it becomes sits at the apex of the bend.

    Args:
        path_unit: (M, 3) unit radials.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to size the window.
        window_pitch: Window half-width in mesh-pitch units.
        excess_deg: Required rise above the baseline, in degrees.
        baseline_mult: Half-width of the baseline neighbourhood, in corner windows.

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
    # The baseline neighbourhood, in vertices. The zero-deflection margins are
    # excluded from every median below, so the statistic is over measured vertices.
    half = max(w + 1, round(baseline_mult * w))
    corners: list[int] = []
    for i in range(w, m - w):
        here = deflect[i]
        lo = max(0, i - half)
        hi = min(m, i + half + 1)
        neighbourhood = [x for x in deflect[lo:hi] if x > 0.0]
        if not neighbourhood:
            continue
        if here <= float(np.median(neighbourhood)) + excess_deg:
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


def point_turn_deg(path_unit: npt.NDArray[np.float64]) -> list[float]:
    """Point-to-point turning angle at every vertex, in degrees.

    The angle between the incoming segment ``p_i - p_{i-1}`` and the outgoing
    ``p_{i+1} - p_i``, both projected to the tangent plane at ``p_i``. This is the
    NOISY sibling of :func:`chord_deflection_deg`: a sawtooth tooth has a huge value
    here and a baseline value there, while a real corner has a large value in BOTH.
    The two together are what separates chatter from a bend -- the windowed test
    alone cannot, because it is blind to the teeth it averages away.

    Args:
        path_unit: (M, 3) unit radials.

    Returns:
        List of length M, degrees. The two endpoints are 0.0 (no turn is defined
        with one neighbour).
    """
    m = len(path_unit)
    out = [0.0] * m
    for i in range(1, m - 1):
        centre = path_unit[i]
        vin = unit(tangent_project_local(centre - path_unit[i - 1], centre))
        vout = unit(tangent_project_local(path_unit[i + 1] - centre, centre))
        if vin is None or vout is None:
            continue
        dot = float(np.clip(float(np.dot(vin, vout)), -1.0, 1.0))
        out[i] = math.degrees(math.acos(dot))
    return out


def detect_chatter_runs(
    path_unit: npt.NDArray[np.float64],
    knots: Sequence[int],
    delta_theta_mesh_rad: float,
    window_pitch: float = CORNER_WINDOW_PITCH,
    tooth_turn_deg: float = TOOTH_TURN_DEG,
    min_teeth: int = SMOOTH_MIN_TEETH,
) -> tuple[tuple[int, int], ...]:
    """Maximal sawtooth runs, as inclusive ``(first_tooth, last_tooth)`` index pairs.

    A vertex is a TOOTH when its point-to-point turn is large AND its windowed
    chord deflection is not: the first says the path kinks here, the second says the
    AVERAGE direction does not, which is precisely a zigzag about a straight line. A
    vertex that is a sharp turn and a corner is a corner -- corners win, and a knot
    is never a tooth.

    Runs are maximal sets of consecutive teeth, kept when at least ``min_teeth``
    long. A single zigzag between two nodes is one vertex's noise, not a sawtooth,
    and replacing it is a coin flip about the field.

    Args:
        path_unit: (M, 3) unit radials.
        knots: Corner indices from :func:`detect_corners` (plus any other pinned
            node), which are excluded from the tooth set.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to size the window.
        window_pitch: Corner window half-width in mesh pitches.
        tooth_turn_deg: Minimum point-to-point turn that counts as a tooth.
        min_teeth: Shortest run that is smoothed.

    Returns:
        Ascending, non-overlapping ``(lo, hi)`` inclusive tooth-index pairs. Empty
        when the path carries no chatter.
    """
    m = len(path_unit)
    if m < _MIN_PATH:
        return ()
    turn = point_turn_deg(path_unit)
    w = _window_steps(path_unit, delta_theta_mesh_rad, window_pitch)
    deflect = chord_deflection_deg(path_unit, w)
    interior = deflect[w : m - w]
    baseline = float(np.median(interior)) if interior else 0.0
    knot_set = {int(k) for k in knots}

    tooth = [False] * m
    for i in range(1, m - 1):
        if i in knot_set:
            continue
        if turn[i] <= tooth_turn_deg:
            continue
        if deflect[i] - baseline > _CHATTER_MAX_EXCESS_DEG:
            continue
        tooth[i] = True

    runs: list[tuple[int, int]] = []
    i = 1
    while i < m - 1:
        if not tooth[i]:
            i += 1
            continue
        j = i
        while j + 1 < m - 1 and tooth[j + 1]:
            j += 1
        if j - i + 1 >= min_teeth:
            runs.append((i, j))
        i = j + 1
    return tuple(runs)


@dataclass(frozen=True)
class SawtoothSmoothing:
    """The outcome of smoothing the chatter out of one traced path.

    Attributes:
        path_unit: The smoothed polyline, CENTRED unit radials, start first. Equal
            to the input when nothing was smoothed.
        changed: True when any vertex was replaced.
        runs: The chatter runs that were smoothed, as ``(first_tooth, last_tooth)``
            inclusive indices into the INPUT path.
        anchors: Ascending fractional indices into the input path of every node the
            curve is pinned to. An integer is a real vertex (an endpoint, a knot, a
            caller-pinned node); a half-integer ``i + 0.5`` is the midpoint of the
            edge ``i -> i+1``, which is where a run is anchored so the curve sits on
            the midline the teeth oscillate about, not on a tooth.
        smoothed_indices: Input indices the curve REPLACED (the teeth). The output
            is not indexed like the input -- runs are emitted at a finer arc spacing
            -- so the correspondence is one-way, and the deviation metric is measured
            from the output back to the input polyline.
        total_turn_before_deg: Sum of point-to-point turns over the input path. This
            is the chatter's signature: a sawtooth pays a large turn at every tooth,
            so a path that travels straight across the crest accumulates turning
            that the ridge does not have.
        total_turn_after_deg: The same quantity on the output. Smoothing the teeth
            removes their turning, so this is the number the operator is judging.
        max_deviation_deg: Largest angular distance from any output vertex to the
            INPUT polyline. This replaces ``max_shift_deg``: once the output carries
            a different number of vertices there is no vertex-to-vertex shift to
            report, and a per-vertex metric would also hide the fact that a
            resampled curve passes between its samples.
    """

    path_unit: npt.NDArray[np.float64]
    changed: bool = False
    runs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    anchors: tuple[float, ...] = field(default_factory=tuple)
    smoothed_indices: tuple[int, ...] = field(default_factory=tuple)
    total_turn_before_deg: float = 0.0
    total_turn_after_deg: float = 0.0
    max_deviation_deg: float = 0.0


def _max_deviation_deg(
    points: npt.NDArray[np.float64],
    polyline: npt.NDArray[np.float64],
) -> float:
    """Largest angular distance from any of ``points`` to ``polyline``, in degrees.

    Closed form per segment, fully vectorised. The distance from ``p`` to the great
    circle through ``a`` and ``b`` is ``asin(|p . n|)`` where ``n`` is the circle's
    pole, and that perpendicular IS the distance to the arc exactly when the foot
    falls between the endpoints -- tested by the two cross-product signs below.
    Outside that, the nearer endpoint wins.

    The vectorisation matters: the per-vertex loop this replaces called
    ``np.cross``/``np.dot`` on 3-vectors inside a Python loop over points x
    segments, and on a 445-vertex traced path with a 293-vertex output that cost
    16.6 s -- 99.8% of the whole smoothing call, and 96% of the entire connect
    stage. The array form does the same arithmetic in one pass over (N, S) blocks.

    Args:
        points: (N, 3) unit radials.
        polyline: (M, 3) unit radials.

    Returns:
        The maximum, over ``points``, of the angular distance to the nearest point
        of ``polyline``. Zero when either is empty.
    """
    if len(points) == 0 or len(polyline) < 2:
        return 0.0
    p = np.asarray(points, dtype=np.float64)
    a = np.asarray(polyline[:-1], dtype=np.float64)
    b = np.asarray(polyline[1:], dtype=np.float64)

    pole = np.cross(a, b)
    pole_norm = np.linalg.norm(pole, axis=1, keepdims=True)
    # Degenerate (coincident) and antipodal segments have no unique great circle;
    # they are handled by the endpoint branch, so give them a zero pole, which makes
    # the perpendicular a constant 0 and the inside test fail.
    usable = pole_norm > 1e-15
    pole = np.divide(pole, pole_norm, out=np.zeros_like(pole), where=usable)

    perp = np.arcsin(np.clip(np.abs(p @ pole.T), 0.0, 1.0))  # (N, S)
    at_a = p @ np.cross(pole, a).T  # foot past a, along the arc
    at_b = p @ np.cross(b, pole).T  # foot before b, along the arc
    inside = (at_a >= 0.0) & (at_b >= 0.0) & usable.T

    d_a = np.arccos(np.clip(p @ a.T, -1.0, 1.0))
    d_b = np.arccos(np.clip(p @ b.T, -1.0, 1.0))
    dist = np.where(inside, perp, np.minimum(d_a, d_b))
    return float(math.degrees(dist.min(axis=1).max()))


def _squad_arc(
    a: npt.NDArray[np.float64],
    qa: npt.NDArray[np.float64],
    qb: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    step_rad: float,
) -> list[npt.NDArray[np.float64]]:
    """The SQUAD curve from ``a`` to ``b``, sampled at ``step_rad`` arc spacing.

    The number of samples is set by the GEODESIC distance between the anchors, so a
    long run is drawn finer in vertex count and the physical spacing is what stays
    fixed. The start is excluded (the caller already emitted the previous anchor) and
    the end is included, so consecutive arcs join without a duplicate vertex.

    Returns:
        List of unit radials, ``b`` last. Empty when the curve is degenerate.
    """
    span = angular_distance(a, b)
    if span <= 0.0 or step_rad <= 0.0:
        return []
    n = max(2, math.ceil(span / step_rad))
    out: list[npt.NDArray[np.float64]] = []
    for k in range(1, n + 1):
        p = squad_point(a, qa, qb, b, k / n)
        if p is None:
            continue
        u = unit(p)
        if u is not None:
            out.append(u)
    if not out:
        return []
    b_unit = unit(b)
    if b_unit is not None:
        out[-1] = b_unit
    return out


def smooth_sawtooth_runs(
    path_unit: npt.NDArray[np.float64],
    knots: Sequence[int],
    delta_theta_mesh_rad: float,
    anchors: Sequence[int] = (),
    output_pitch: float = SMOOTH_OUTPUT_PITCH,
    window_pitch: float = CORNER_WINDOW_PITCH,
    tooth_turn_deg: float = TOOTH_TURN_DEG,
    min_teeth: int = SMOOTH_MIN_TEETH,
) -> SawtoothSmoothing:
    """Replace each sawtooth run with one smooth spherical curve through its anchors.

    This is the refinement. The teeth are removed by changing the path's SHAPE along
    its length, not by shifting it across the ridge: Entry 058's transverse nudge
    moved every vertex a hair sideways and the zigzag survived, because the zigzag IS
    the sideways motion. Here the vertices between two anchors are deleted and a
    single curve is drawn between them.

    The curve is a spherical Catmull-Rom (SQUAD). Its control point at an anchor is
    :func:`sphere_frame.squad_control` applied to the anchor's neighbours IN THE
    ANCHOR POLYGON, not in the raw path -- which is the whole trick. A Catmull-Rom
    through the teeth reproduces the teeth; a Catmull-Rom through the nodes the teeth
    hang between draws the midline they oscillate about, because the midline is
    exactly what the anchor polygon is.

    The anchor at each end of a run is the MIDPOINT OF THE EDGE that bounds it, not
    the vertex. Consecutive teeth sit at +amp and -amp about the midline, so the
    midline is crossed at the midpoint of the edge between them; pinning the curve to
    the entry/exit VERTEX would pin it to a tooth and leave the whole curve a tooth's
    width off the crest. The two path endpoints, every knot, and every caller-pinned
    node are anchored exactly, because those are physical positions, not oscillations.

    Anchors are immovable. A knot keeps its angle because the curve is built to pass
    through it and is free to kink there -- no tangent continuity is imposed across a
    knot, which is the property that makes a real bend survive. Gaps that carry no
    chatter keep their original vertices untouched; only the runs are resampled, at
    ``output_pitch`` arc spacing, so the smoothed line is visibly finer than the mesh
    while the record's shape is preserved everywhere the field was never in question.

    Pure geometry: no field is sampled, so this runs on a path with no mesh beside
    it, and it is deterministic in its input.

    Args:
        path_unit: (M, 3) unit radials, CENTRED frame, launch port first.
        knots: Corner indices from :func:`detect_corners`. Sharp, real, pinned.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to size the window and
            the output spacing.
        anchors: Additional pinned indices (the transited excision centres, the
            vertex nearest each). Pinned like knots, and never a tooth.
        output_pitch: Arc spacing of the emitted curve, in mesh pitches.
        window_pitch: Corner window half-width in mesh pitches, for the tooth test.
        tooth_turn_deg: Minimum point-to-point turn that counts as a tooth.
        min_teeth: Shortest run that is smoothed.

    Returns:
        A :class:`SawtoothSmoothing`. When the path carries no chatter the input is
        returned with ``changed`` False, so a caller can hand it every connector.
    """
    m = len(path_unit)
    before = sum(point_turn_deg(path_unit)) if m >= 3 else 0.0
    if m < _MIN_PATH or delta_theta_mesh_rad <= 0.0:
        return SawtoothSmoothing(
            path_unit=np.asarray(path_unit, dtype=np.float64).copy(),
            total_turn_before_deg=before,
            total_turn_after_deg=before,
        )

    runs = detect_chatter_runs(
        path_unit,
        knots,
        delta_theta_mesh_rad,
        window_pitch,
        tooth_turn_deg,
        min_teeth,
    )
    if not runs:
        return SawtoothSmoothing(
            path_unit=np.asarray(path_unit, dtype=np.float64).copy(),
            total_turn_before_deg=before,
            total_turn_after_deg=before,
        )

    def _mid(i: int, j: int) -> npt.NDArray[np.float64]:
        a = path_unit[i]
        b = path_unit[j]
        m_unit = unit(a + b)
        return m_unit if m_unit is not None else a

    # The anchor polygon, as (fractional index, position) pairs. A run's ends sit at
    # the midpoints of the bounding edges (lo-0.5, hi+0.5); every pinned node sits on
    # its own integer index. The fractional keys order them along the path and mark
    # which gaps are smoothed (a run's entry->exit) versus kept verbatim.
    pinned: dict[float, npt.NDArray[np.float64]] = {
        0.0: path_unit[0],
        float(m - 1): path_unit[m - 1],
    }
    for k in knots:
        ki = int(k)
        if 0 < ki < m - 1:
            pinned[float(ki)] = path_unit[ki]
    for a in anchors:
        ai = int(a)
        if 0 < ai < m - 1:
            pinned[float(ai)] = path_unit[ai]
    smoothed: set[tuple[float, float]] = set()
    replaced: set[int] = set()
    for lo, hi in runs:
        pinned[lo - 0.5] = _mid(lo - 1, lo)
        pinned[hi + 0.5] = _mid(hi, hi + 1)
        smoothed.add((lo - 0.5, hi + 0.5))
        replaced.update(range(lo, hi + 1))

    keys = sorted(pinned)
    positions = [pinned[k] for k in keys]

    # Control point at each anchor, from its neighbours IN THE ANCHOR POLYGON.
    controls: list[npt.NDArray[np.float64]] = []
    for j in range(len(positions)):
        prev = positions[j - 1] if j > 0 else positions[j]
        nxt = positions[j + 1] if j < len(positions) - 1 else positions[j]
        q = squad_control(prev, positions[j], nxt)
        # A one-sided control (a path endpoint) falls back to the anchor itself,
        # which makes the arc through it a plain geodesic: no invented curvature at
        # a free end.
        controls.append(q if q is not None else positions[j])

    step_rad = max(1e-6, output_pitch * delta_theta_mesh_rad)
    out: list[npt.NDArray[np.float64]] = [positions[0]]
    for j in range(len(positions) - 1):
        lo_key, hi_key = keys[j], keys[j + 1]
        if (lo_key, hi_key) in smoothed:
            arc = _squad_arc(
                positions[j], controls[j], controls[j + 1], positions[j + 1], step_rad
            )
            if arc:
                out.extend(arc)
                continue
        # A verbatim gap: emit the original vertices strictly between the two anchor
        # keys, then the end anchor. Integer indices in (lo_key, hi_key).
        for i in range(math.ceil(lo_key), math.floor(hi_key) + 1):
            if lo_key < i < hi_key:
                out.append(path_unit[i])
        out.append(positions[j + 1])

    result = np.array(out, dtype=np.float64)
    after = sum(point_turn_deg(result)) if len(result) >= 3 else 0.0
    # Chunked so the (points x segments) working set stays bounded on a long path.
    max_dev = 0.0
    for s in range(0, len(result), 4096):
        max_dev = max(max_dev, _max_deviation_deg(result[s : s + 4096], path_unit))
    return SawtoothSmoothing(
        path_unit=result,
        changed=True,
        runs=runs,
        anchors=tuple(keys),
        smoothed_indices=tuple(sorted(replaced)),
        total_turn_before_deg=before,
        total_turn_after_deg=after,
        max_deviation_deg=max_dev,
    )


@dataclass(frozen=True)
class StructuralCurve:
    """One smooth curve through the structural nodes of a traced path.

    Attributes:
        path_unit: The curve, CENTRED unit radials, start first. Empty when the
            path was too short to carry a node pair.
        node_indices: Input vertex index of every node the curve passes through,
            ascending. The endpoints and the knots are exact; the rest are the
            vertices nearest the arc-length positions ``NODE_PITCH`` apart.
        n_nodes: Size of the node set -- the description length of the connection.
        total_turn_before_deg: Point-to-point turning of the traced path.
        total_turn_after_deg: The same over the curve. The gap is the winding and
            chatter the coarse spacing filtered out.
        max_deviation_deg: Largest angular distance from the curve to the traced
            polyline. INFORMATIONAL, not a cost: a coarse structural curve is
            asked to depart from the ascent by many pitches, and on the shape-index
            field it does (2.66 deg at 8 pitches). The Entry-061 smoother treated
            this as the quantity to minimise; that contract belongs to the
            fidelity curve, not to this one.
    """

    path_unit: npt.NDArray[np.float64]
    node_indices: tuple[int, ...] = field(default_factory=tuple)
    n_nodes: int = 0
    total_turn_before_deg: float = 0.0
    total_turn_after_deg: float = 0.0
    max_deviation_deg: float = 0.0


def _arc_length_nodes(
    path_unit: npt.NDArray[np.float64],
    step_rad: float,
) -> list[int]:
    """Vertex indices closest to arc-length positions ``step_rad`` apart.

    The nodes are chosen by ARC LENGTH along the traced path, not by vertex index,
    so the spacing is a physical distance and the same on a path sampled finely and
    on one sampled coarsely. The two endpoints are always included: they are the
    saddle port and the maximum, and the curve must start and finish there.

    Returns:
        Ascending, strictly increasing indices, first 0 and last M-1.
    """
    m = len(path_unit)
    if m < 2:
        return list(range(m))
    cum = [0.0]
    for i in range(m - 1):
        cum.append(cum[-1] + angular_distance(path_unit[i], path_unit[i + 1]))
    total = cum[-1]
    if total <= 0.0 or step_rad <= 0.0:
        return [0, m - 1]
    out: list[int] = [0]
    target = step_rad
    i = 0
    while target < total - 1e-12:
        while i < m - 2 and cum[i + 1] < target:
            i += 1
        # Nearest vertex to the target arc length, not the one before it.
        best = i + 1 if abs(cum[i + 1] - target) <= abs(cum[i] - target) else i
        if best > out[-1]:
            out.append(best)
        target += step_rad
    if out[-1] != m - 1:
        out.append(m - 1)
    return out


def coarse_structural_curve(
    path_unit: npt.NDArray[np.float64],
    knots: Sequence[int],
    delta_theta_mesh_rad: float,
    anchors: Sequence[int] = (),
    node_pitch: float = NODE_PITCH,
    output_pitch: float = CURVE_OUTPUT_PITCH,
) -> StructuralCurve:
    """One smooth curve through the structural nodes of a traced path.

    The traced ascent is replaced by a description: a handful of nodes at the
    places that carry meaning -- the launch port, the maximum, every detected
    corner, every caller-pinned centre, and points spaced ``node_pitch`` mesh
    pitches along the arc between them -- joined by a single spherical
    Catmull-Rom. Everything between the nodes is interpolation, and that is the
    point: the exact path of the ascent is a property of the integrator's step
    size, and the 3D representation wants the connection, not the integration.

    The spacing is a low-pass filter's cutoff, not a density knob. Measured on the
    ethene3 shape-index connector that spirals half a turn inside MC2 (441
    vertices, 66010 deg of total turning): at 1 pitch the curve reproduces the
    spiral almost exactly (65935 deg -- a faithful copy of a path that should not
    be followed), at 4 pitches it drops to 1806 deg, at 8 to 651 deg. The node
    spacing decides which features survive, and 8 pitches keeps the bends and
    discards the winding.

    Knots and caller-pinned anchors are inserted into the node set at their exact
    index, so a real corner is never interpolated across: the curve passes through
    it and is free to kink there. The endpoints are always nodes.

    Pure geometry: no field is sampled, and the output is deterministic in the
    input.

    Args:
        path_unit: (M, 3) unit radials, CENTRED frame, launch port first.
        knots: Corner indices from :func:`detect_corners`. Pinned, sharp.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to space the nodes.
        anchors: Additional pinned indices (the vertex nearest each transited
            excision centre).
        node_pitch: Node spacing in mesh pitches -- the filter cutoff.
        output_pitch: Tessellation spacing of the emitted polyline, in pitches.

    Returns:
        A :class:`StructuralCurve`. Empty ``path_unit`` when the path is too short
        to carry a node pair, so a caller can hand it every connector.
    """
    m = len(path_unit)
    before = sum(point_turn_deg(path_unit)) if m >= 3 else 0.0
    if m < 3 or delta_theta_mesh_rad <= 0.0:
        return StructuralCurve(
            path_unit=np.asarray(path_unit, dtype=np.float64).copy(),
            total_turn_before_deg=before,
            total_turn_after_deg=before,
        )

    nodes = set(_arc_length_nodes(path_unit, node_pitch * delta_theta_mesh_rad))
    nodes.add(0)
    nodes.add(m - 1)
    for k in knots:
        ki = int(k)
        if 0 <= ki < m:
            nodes.add(ki)
    for a in anchors:
        ai = int(a)
        if 0 <= ai < m:
            nodes.add(ai)
    keys = sorted(nodes)
    positions = [path_unit[i] for i in keys]
    if len(positions) < 2:
        return StructuralCurve(
            path_unit=np.asarray(path_unit, dtype=np.float64).copy(),
            node_indices=tuple(keys),
            n_nodes=len(keys),
            total_turn_before_deg=before,
            total_turn_after_deg=before,
        )

    controls: list[npt.NDArray[np.float64]] = []
    for j in range(len(positions)):
        prev = positions[j - 1] if j > 0 else positions[j]
        nxt = positions[j + 1] if j < len(positions) - 1 else positions[j]
        q = squad_control(prev, positions[j], nxt)
        controls.append(q if q is not None else positions[j])

    step_rad = max(1e-6, output_pitch * delta_theta_mesh_rad)
    out: list[npt.NDArray[np.float64]] = [positions[0]]
    for j in range(len(positions) - 1):
        arc = _squad_arc(
            positions[j], controls[j], controls[j + 1], positions[j + 1], step_rad
        )
        if arc:
            out.extend(arc)
        else:
            out.append(positions[j + 1])

    result = np.array(out, dtype=np.float64)
    after = sum(point_turn_deg(result)) if len(result) >= 3 else 0.0
    max_dev = 0.0
    for s in range(0, len(result), 4096):
        max_dev = max(max_dev, _max_deviation_deg(result[s : s + 4096], path_unit))
    return StructuralCurve(
        path_unit=result,
        node_indices=tuple(keys),
        n_nodes=len(keys),
        total_turn_before_deg=before,
        total_turn_after_deg=after,
        max_deviation_deg=max_dev,
    )


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


def smooth_gradient_ridge(
    arc_points: Sequence[Sequence[float]],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    transited_centres: Sequence[Sequence[float]] = (),
    window_pitch: float = CORNER_WINDOW_PITCH,
    excess_deg: float = CORNER_EXCESS_DEG,
    output_pitch: float = SMOOTH_OUTPUT_PITCH,
) -> RidgeRefinement:
    """Smooth the sawtooth chatter out of one traced connector. The refinement.

    The world-frame wrapper around :func:`smooth_sawtooth_runs`: it takes the
    polyline ``MaxConnection.arc_points`` carries, converts to centred unit
    radials, detects the corners (which are the knots the smoother pins), smooths
    the runs, and returns a world-frame polyline. No field is sampled -- the
    smoother is pure geometry -- so this needs neither the mesh nor the values,
    only the pitch (to size the window and the output spacing) and the radius (to
    scale the radials).

    The corners are detected with the SAME :func:`detect_corners` the probe uses, so
    the two agree on where the path genuinely bends; a knot is passed to the smoother
    as an anchor it may not move and may not smooth through.

    Args:
        arc_points: (M, 3) connector polyline, WORLD frame, launch port first.
        sphere_radius: Atom sphere radius R, to scale the output radials.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus. None when the positions are already centred.
        transited_centres: Centres of the excision regions the path crossed, WORLD
            frame. Each is pinned to the nearest vertex, exactly as the nudge did.
        window_pitch: Corner window half-width in mesh pitches.
        excess_deg: Corner threshold above the baseline, in degrees.
        output_pitch: Arc spacing of the emitted curve, in mesh pitches.

    Returns:
        A :class:`RidgeRefinement`. The input is returned unchanged (``changed``
        False) for a path too short to carry a run, so a caller can hand it every
        connector.
    """
    m = len(arc_points)
    passthrough = tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points)
    if m < _MIN_PATH or sphere_radius <= 0.0 or delta_theta_mesh_rad <= 0.0:
        return RidgeRefinement(path=passthrough, changed=False)

    original = _path_units(arc_points, center)
    corners, _, _ = detect_corners(
        original, delta_theta_mesh_rad, window_pitch, excess_deg
    )

    pinned: list[int] = []
    for c in transited_centres:
        arr = np.asarray(c, dtype=np.float64)
        local = np.asarray(to_local(arr, center), dtype=np.float64) if center else arr
        u = unit(local)
        if u is None:
            continue
        d = [angular_distance(u, original[i]) for i in range(m)]
        pinned.append(int(np.argmin(d)))

    res = smooth_sawtooth_runs(
        original,
        corners,
        delta_theta_mesh_rad,
        anchors=pinned,
        output_pitch=output_pitch,
        window_pitch=window_pitch,
    )
    out: list[tuple[float, float, float]] = []
    for u in res.path_unit:
        pos = (
            float(u[0] * sphere_radius),
            float(u[1] * sphere_radius),
            float(u[2] * sphere_radius),
        )
        out.append(to_global(pos, center) if center else pos)
    return RidgeRefinement(
        path=tuple(out),
        changed=res.changed,
        corner_indices=corners,
        anchor_indices=tuple(round(a) for a in res.anchors),
        max_deviation_deg=res.max_deviation_deg,
        total_turn_before_deg=res.total_turn_before_deg,
        total_turn_after_deg=res.total_turn_after_deg,
        n_runs=len(res.runs),
        n_moved=len(res.smoothed_indices),
    )


def structural_curve_ridge(
    arc_points: Sequence[Sequence[float]],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    transited_centres: Sequence[Sequence[float]] = (),
    window_pitch: float = CORNER_WINDOW_PITCH,
    excess_deg: float = CORNER_EXCESS_DEG,
    node_pitch: float = NODE_PITCH,
    output_pitch: float = CURVE_OUTPUT_PITCH,
) -> RidgeRefinement:
    """Replace one traced connector with a coarse smooth curve through its nodes.

    The world-frame wrapper around :func:`coarse_structural_curve`, with the same
    contract as :func:`smooth_gradient_ridge`: takes the polyline
    ``MaxConnection.arc_points`` carries, converts to centred unit radials, detects
    the corners, builds the curve, and returns a world-frame polyline. No field is
    sampled.

    The difference from the fidelity smoother is the target. ``smooth_gradient_ridge``
    preserves the ascent and removes only mesh-scale chatter, so its deviation from
    the traced path is the cost. This one is asked to describe the CONNECTION, so the
    deviation is expected and the useful numbers are the node count (the description
    length of the branch) and the turning removed.

    Args:
        arc_points: (M, 3) connector polyline, WORLD frame, launch port first.
        sphere_radius: Atom sphere radius R, to scale the output radials.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus. None when the positions are already centred.
        transited_centres: Centres of the excision regions the path crossed, WORLD
            frame. Each is pinned to the nearest vertex and becomes a node.
        window_pitch: Corner window half-width in mesh pitches.
        excess_deg: Corner threshold above the baseline, in degrees.
        node_pitch: Structural node spacing in mesh pitches -- the filter cutoff.
        output_pitch: Tessellation spacing of the emitted curve, in pitches.

    Returns:
        A :class:`RidgeRefinement`. ``changed`` is True whenever a curve was built,
        which for a path of ``_MIN_PATH`` vertices or more is always: unlike the
        fidelity smoother, this has nothing to detect first.
    """
    m = len(arc_points)
    passthrough = tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points)
    if m < _MIN_PATH or sphere_radius <= 0.0 or delta_theta_mesh_rad <= 0.0:
        return RidgeRefinement(path=passthrough, changed=False)

    original = _path_units(arc_points, center)
    corners, _, _ = detect_corners(
        original, delta_theta_mesh_rad, window_pitch, excess_deg
    )

    pinned: list[int] = []
    for c in transited_centres:
        arr = np.asarray(c, dtype=np.float64)
        local = np.asarray(to_local(arr, center), dtype=np.float64) if center else arr
        u = unit(local)
        if u is None:
            continue
        d = [angular_distance(u, original[i]) for i in range(m)]
        pinned.append(int(np.argmin(d)))

    res = coarse_structural_curve(
        original,
        corners,
        delta_theta_mesh_rad,
        anchors=pinned,
        node_pitch=node_pitch,
        output_pitch=output_pitch,
    )
    if len(res.path_unit) == 0:
        return RidgeRefinement(path=passthrough, changed=False)
    out: list[tuple[float, float, float]] = []
    for u in res.path_unit:
        pos = (
            float(u[0] * sphere_radius),
            float(u[1] * sphere_radius),
            float(u[2] * sphere_radius),
        )
        out.append(to_global(pos, center) if center else pos)
    return RidgeRefinement(
        path=tuple(out),
        changed=True,
        corner_indices=corners,
        anchor_indices=res.node_indices,
        max_deviation_deg=res.max_deviation_deg,
        total_turn_before_deg=res.total_turn_before_deg,
        total_turn_after_deg=res.total_turn_after_deg,
        n_runs=0,
        n_nodes=res.n_nodes,
        n_moved=m,
    )


# ---------------------------------------------------------------------------
# ridge probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgePoint:
    """One sample's step onto the ridge.

    Attributes:
        position_unit: The ridge point, CENTRED unit radial. Equals the sample
            when ``accepted`` is False.
        signed_offset_rad: The cross-track component of the step, signed along the
            sample's cross-track. This is the quantity the operator reads: how far,
            and which side, the ridge lies from the traced path.
        delta_rad: The full step length along the scan axis, signed.
        accepted: False when the local field geometry is not a ridge the sample can
            step onto; ``reason`` says which.
        reason: ``"ok"``, ``"no_frame"``, ``"fit_failed"``, ``"no_ridge"`` (the
            Hessian has no negative eigenvalue -- a valley, a flat basin, or noise),
            ``"along_path"`` (the across-crest direction is mostly along the path,
            so this is a re-routing), ``"out_of_reach"`` (the crest lies beyond
            :data:`RIDGE_REACH_PITCH`), or ``"no_crest"`` (the step lands on ground
            no higher than the sample: in the flat country near a saddle the field
            genuinely does not say where the ridge is, and saying so is the honest
            answer).
        gain_fraction: The field rise of the ridge point over the sample, as a
            fraction of the field's span. Zero on a refusal. This is what separates
            a real crest from a ripple peak, published per sample so the reader sees
            the difference rather than takes it on faith: measured on ethene3, real
            crests gain 0.17-26% of span, ripple peaks 0.0008%.
        other_side: ``(offset_rad, gain_fraction)`` of the most developed crest on
            the OPPOSITE side of the path from the one chosen, or None when the scan
            found none. The operator's question -- "is the ridge the other way?" --
            is answered by this field, per sample, in the shipped tool.
    """

    position_unit: npt.NDArray[np.float64]
    signed_offset_rad: float
    delta_rad: float
    accepted: bool
    reason: str
    gain_fraction: float = 0.0
    other_side: tuple[float, float] | None = None


@dataclass(frozen=True)
class RidgeSample:
    """One probe sample: a point on the traced path and the ridge beside it.

    Attributes:
        along_deg: Arc length from the sub-segment's start, degrees.
        from_unit: The sample's position ON THE PATH, centred unit radial.
        point: The ridge step from that sample.
    """

    along_deg: float
    from_unit: npt.NDArray[np.float64]
    point: RidgePoint


@dataclass(frozen=True)
class RidgeProbe:
    """The ridge geometry of one gradient-traced connector. Nothing is moved.

    Attributes:
        knot_indices: Path indices promoted to knots -- the sharp, CONTINUOUS bends.
            The two endpoints are pinned for the segment structure but are NOT
            knots, and sawtooth teeth never appear here: this is the operator's
            "sharp and continuous change in direction as compared to saw teeth".
        knot_positions_world: The knots in the WORLD frame, for drawing.
        samples: Every probe sample, in path order, accepted and refused alike.
        n_accepted: Samples that found a ridge within reach.
        median_offset_pitch: Median |signed offset| of the accepted samples, in mesh
            pitches -- the honest size of the correction. Entry 058 reported a max
            and the max was the knot; the median is what the operator sees.
        max_offset_pitch: The largest accepted offset, in pitches.
    """

    knot_indices: tuple[int, ...] = field(default_factory=tuple)
    knot_positions_world: tuple[tuple[float, float, float], ...] = field(
        default_factory=tuple
    )
    samples: tuple[RidgeSample, ...] = field(default_factory=tuple)
    n_accepted: int = 0
    median_offset_pitch: float = 0.0
    max_offset_pitch: float = 0.0


def _plane_basis(
    point_unit: npt.NDArray[np.float64],
    tangent_unit: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Orthonormal tangent-plane basis ``(along, across)`` at a point.

    ``along`` is the path's own direction, projected to the tangent plane;
    ``across`` is ``r x along``. The probe's step is measured in this basis, so
    ``e[0]`` is the along-path share of the ridge direction and ``e[1]`` the
    cross-track share.
    """
    along = unit(tangent_project_local(tangent_unit, point_unit))
    if along is None:
        return None
    across = unit(np.cross(point_unit, along))
    if across is None:  # pragma: no cover - along is tangent, so never parallel to r
        return None
    return along, across


def _ridge_frame(
    point_unit: npt.NDArray[np.float64],
    tangent_unit: npt.NDArray[np.float64],
    radius: float,
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    find_k_nearest: KNearestFinder | None,
    stencil_rad: float,
) -> (
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
    | None
):
    """Gradient and Hessian of ``log f`` at a point, from one wide disc fit.

    A quadratic is fitted to ``log f`` over a ring/spoke disc in the tangent plane
    and its gradient and Hessian read off the coefficients. Three reasons this is
    the criterion and a peak search is not:

    1. **The direction comes from the field.** The eigenvector of the most-negative
       Hessian eigenvalue is the across-crest direction wherever the sample sits on
       a flank. It does not depend on the path's own vertices, which is what breaks
       the circularity of a path-derived cross-track -- reading the path's
       ``i +/- 1`` neighbours makes the cross-track wiggle with the teeth it was
       supposed to be immune to.
    2. **A zero crossing, not an argmax.** The crest is where the gradient has no
       component along the ridge. On a broad flat-topped ridge the value of ``f``
       barely varies, so an argmax is set-valued and noise decides it; the
       tangential derivative is zero at the crest and grows away from it, so its
       zero is localised exactly where the field is flattest.
    3. **A wide stencil.** The Hessian is a second-derivative object, and a disc of
       ~2.5 pitches averages the mesh-scale IDW ripple out of it. The old estimator
       sampled every 0.5 pitch -- twice per ripple period -- and the fit was
       dominated by the ripple's phase.

    ``log f`` because a Gaussian ridge is exactly a paraboloid in log, so the fit is
    not an approximation for the profiles these fields have, and the gradient's zero
    set is identical to ``f``'s.

    Args:
        point_unit: Centred unit radial of the sample.
        tangent_unit: The path's direction there (any radial; it is projected).
        radius: Sphere radius, for the field samples.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values.
        find_k_nearest: Spatial index over ``mesh_pts``.
        stencil_rad: Disc radius in radians.

    Returns:
        ``(g, H, along, across)`` with ``g`` the 2-gradient, ``H`` the 2-Hessian in
        the ``(along, across)`` basis, and the basis vectors themselves; None when
        the basis or the fit degenerates.
    """
    basis = _plane_basis(point_unit, tangent_unit)
    if basis is None:
        return None
    along, across = basis
    du: list[float] = [0.0]
    dv: list[float] = [0.0]
    dirs: list[npt.NDArray[np.float64]] = [point_unit]
    for ring in range(1, _STENCIL_RINGS + 1):
        rad = stencil_rad * ring / _STENCIL_RINGS
        for spoke in range(_STENCIL_SPOKES):
            ang = 2.0 * math.pi * spoke / _STENCIL_SPOKES
            cu, cv = rad * math.cos(ang), rad * math.sin(ang)
            d = unit(cu * along + cv * across)
            if d is None:  # pragma: no cover - along, across are orthonormal
                continue
            moved = _offset_unit(point_unit, d, rad)
            if moved is None:  # pragma: no cover
                continue
            du.append(cu)
            dv.append(cv)
            dirs.append(moved)
    vals = _sample_field(dirs, radius, mesh_pts, f_vals, find_k_nearest)
    u = np.asarray(du, dtype=np.float64)
    v = np.asarray(dv, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    if len(u) < 6 or not (len(u) == len(v) == len(y)):
        return None
    if float(np.min(y)) <= 0.0:
        y = y - float(np.min(y)) + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ly = np.log(np.clip(y, 1e-300, None))
    if not np.all(np.isfinite(ly)):
        return None
    basis_mat = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    coef, *_ = np.linalg.lstsq(basis_mat, ly, rcond=None)
    if not np.all(np.isfinite(coef)):
        return None
    _, c1, c2, c3, c4, c5 = (float(c) for c in coef)
    grad = np.array([c1, c2], dtype=np.float64)
    hess = np.array([[2.0 * c3, c4], [c4, 2.0 * c5]], dtype=np.float64)
    return grad, hess, along, across


def _scan_crests(
    point_unit: npt.NDArray[np.float64],
    axis_unit: npt.NDArray[np.float64],
    half_rad: float,
    step_rad: float,
    radius: float,
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    find_k_nearest: KNearestFinder | None,
    prominence_fraction: float,
) -> list[tuple[float, float]]:
    """Crests along the cross-track line through a sample, both directions.

    The line is scanned at ``step_rad`` resolution over ``[-half_rad, +half_rad]``
    and every local maximum whose PROMINENCE clears the gate is returned, refined
    to sub-sample position by a parabola through its three samples.

    Prominence, not raw local-max: the IDW interpolant has a local maximum every
    pitch in the flat region, all at the same value to four decimals. A prominence
    gate measures how much a peak stands above the ground between it and the rest
    of the window, which is exactly the difference between "a crest" and "a bump in
    the noise", and it is scale-free: the same fraction serves a sharp ridge and a
    shallow one.

    Args:
        point_unit: The sample, centred unit radial.
        axis_unit: Unit tangent giving the positive scan direction.
        half_rad: Scan half-length, radians.
        step_rad: Scan resolution, radians.
        radius: Sphere radius for the field samples.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values.
        find_k_nearest: Spatial index over ``mesh_pts``.
        prominence_fraction: Minimum rise above the window's troughs, as a fraction
            of the window's own value range.

    Returns:
        ``(offset_rad, value)`` for each crest, offset signed along ``axis_unit``,
        sorted by offset. Empty when the profile has no prominent peak.
    """
    if half_rad <= 0.0 or step_rad <= 0.0:
        return []
    n = max(4, round(2.0 * half_rad / step_rad))
    offs = [-half_rad + 2.0 * half_rad * k / n for k in range(n + 1)]
    dirs = [_offset_unit(point_unit, axis_unit, o) for o in offs]
    if any(d is None for d in dirs):  # pragma: no cover - unit of a rotated unit
        return []
    vals = _sample_field(
        [d for d in dirs if d is not None], radius, mesh_pts, f_vals, find_k_nearest
    )
    if len(vals) != len(offs) or len(vals) < 3:
        return []
    arr = np.asarray(vals, dtype=np.float64)
    rng = float(arr.max() - arr.min())
    if not math.isfinite(rng) or rng <= 0.0:
        return []
    gate = prominence_fraction * rng
    out: list[tuple[float, float]] = []
    for i in range(1, len(arr) - 1):
        if not (arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]):
            continue
        if arr[i] == arr[i - 1] and arr[i] == arr[i + 1]:
            continue  # a flat run is not a peak
        left = float(arr[: i + 1].min())
        right = float(arr[i:].min())
        if float(arr[i]) - max(left, right) < gate:
            continue
        # Parabola refinement: the crest of the quadratic through the three
        # samples, so the position is continuous rather than quantised to the scan.
        y0, y1, y2 = float(arr[i - 1]), float(arr[i]), float(arr[i + 1])
        denom = y0 - 2.0 * y1 + y2
        shift = 0.0 if denom >= -1e-18 else 0.5 * (y0 - y2) / denom
        shift = max(-1.0, min(1.0, shift))
        out.append((offs[i] + shift * step_rad, y1 + 0.25 * (y0 - y2) * shift))
    return out


def ridge_point_at(
    point_unit: npt.NDArray[np.float64],
    tangent_unit: npt.NDArray[np.float64],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    find_k_nearest: KNearestFinder | None = None,
    stencil_pitch: float = RIDGE_STENCIL_PITCH,
    reach_pitch: float = RIDGE_REACH_PITCH,
    prominence_fraction: float = _CREST_PROMINENCE_FRACTION,
) -> RidgePoint:
    """Step one sample onto the most developed ridge within reach, either side.

    Two mechanisms, each doing the half it is good at:

    - **The Hessian gives the AXIS.** The eigenvector of the most-negative
      eigenvalue is the across-crest direction, a property of the field and not of
      the path. This is what breaks the circularity of a path-derived cross-track,
      which reads the path's own ``i +/- 1`` vertices and so wiggles with the
      sawtooth it was meant to be immune to.
    - **A scan along that axis gives the LOCATION.** The profile is sampled both
      ways out to ``reach_pitch``, crests are found by prominence, and the MOST
      DEVELOPED one wins -- highest field value, ties broken toward the sample.

    The Newton step alone would have supplied both, and its second half is the
    problem: ``-(g·e)/lambda_min`` lands on the NEAREST zero-crossing of the
    tangential gradient. Where a path runs along a shoulder, the nearest crest is
    the weak one and a more developed ridge on the other side is invisible to it.
    The operator's criterion -- "move around its contour to find the ridge" -- is a
    search along the line, and that is what this is; the Hessian narrows it to one
    line so the search is 1D and its reach is bounded.

    Refusals, all structural:

    - ``lambda_min >= 0``: the local geometry is a valley, a basin, or noise. There
      is no across-crest direction to scan.
    - ``|e · along| > sin 45deg``: the direction points mostly ALONG the path, so
      the sample is not on the flank of a ridge the path follows.
    - ``no_crest``: the scan found no prominent peak within reach. In the flat
      region near a saddle this is the honest answer -- the field does not say
      where the ridge is there, and a probe that invented one would be drawing
      interpolation noise and calling it a crest.

    Args:
        point_unit: The sample, CENTRED unit radial.
        tangent_unit: The path's direction at the sample.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        find_k_nearest: Spatial index over ``mesh_pts``.
        stencil_pitch: Hessian disc radius in mesh pitches.
        reach_pitch: Scan half-length, in mesh pitches. See :data:`RIDGE_REACH_PITCH`
            for how this bound was measured.
        prominence_fraction: Crest gate, as a fraction of the scan window's range.

    Returns:
        A :class:`RidgePoint`. ``accepted`` False with a ``reason`` when the local
        field does not present a reachable ridge; the sample position is returned
        unchanged in that case so a caller can still draw it.
    """
    p = unit(point_unit)
    if p is None:
        return RidgePoint(
            np.asarray(point_unit, dtype=np.float64), 0.0, 0.0, False, "no_frame"
        )
    frame = _ridge_frame(
        p,
        np.asarray(tangent_unit, dtype=np.float64),
        sphere_radius,
        mesh_pts,
        f_vals,
        find_k_nearest,
        stencil_pitch * delta_theta_mesh_rad,
    )
    if frame is None:
        return RidgePoint(p, 0.0, 0.0, False, "fit_failed")
    grad, hess, along, across = frame
    evals, evecs = np.linalg.eigh(hess)
    lam = float(evals[0])
    if not math.isfinite(lam) or lam >= 0.0:
        return RidgePoint(p, 0.0, 0.0, False, "no_ridge")
    e = np.asarray(evecs[:, 0], dtype=np.float64)
    if abs(float(e[0])) > _RIDGE_MAX_ALONG_COS:
        return RidgePoint(p, 0.0, 0.0, False, "along_path")
    axis = unit(float(e[0]) * along + float(e[1]) * across)
    if axis is None:  # pragma: no cover - e is unit, along/across orthonormal
        return RidgePoint(p, 0.0, 0.0, False, "fit_failed")
    reach = reach_pitch * delta_theta_mesh_rad
    # THE STEP IS NEWTON'S, NOT THE HIGHEST CREST IN THE WINDOW. The reason is the
    # flat top. On a broad crest the smooth field varies across a pitch by LESS than
    # the interpolant ripples, so a ripple peak sitting on the flank is higher in
    # VALUE than the true crest: measured on a synthetic flat-topped ridge, a
    # highest-value rule stepped a sample from latitude +0.718 deg to +1.249 deg --
    # away from a crest lying at 0. A zero crossing of the tangential derivative is
    # localised on a flat top; an argmax is not. So the crest is located by the fit,
    # gated by the field rise below, and the cross-track scan is used ONLY to report
    # what the other side offers.
    #
    # REACH FIRST, then the scan. The Newton step is one arithmetic operation on the
    # fit already in hand; the scan is 25 field evaluations. Checking reach first
    # refuses an unreachable crest cheaply, and reports the more informative reason:
    # "the crest is a different ridge" is a fact about the field, "the scan found no
    # peak" is a fact about the search window.
    delta = -float(np.dot(grad, e)) / lam
    if not math.isfinite(delta):
        return RidgePoint(p, 0.0, 0.0, False, "fit_failed")
    if abs(delta) > reach:
        return RidgePoint(p, 0.0, delta, False, "out_of_reach")
    crests = _scan_crests(
        p,
        axis,
        reach,
        _CREST_SCAN_STEP_PITCH * delta_theta_mesh_rad,
        sphere_radius,
        mesh_pts,
        f_vals,
        find_k_nearest,
        prominence_fraction,
    )
    moved = _offset_unit(p, axis, delta)
    if moved is None:  # pragma: no cover
        return RidgePoint(p, 0.0, delta, False, "fit_failed")
    # The crest must be a real rise over where the sample stands, measured against
    # the FIELD's span. The prominence gate inside the scan is relative to the scan
    # window, and in the flat ground near a saddle the window's entire range IS the
    # IDW ripple, so a relative gate passes ripple peaks there. Measured on ethene3
    # curvedness, the gain of the best crest within reach, as a fraction of span:
    # f < 5% of span (n=82): median +0.0008%, max +0.156% -- noise.
    # f in 5-25%   (n=22): min +0.167%, median +10.1% -- a ridge.
    # f > 25%      (n=15): min +7.28%,  median +9.3%  -- a ridge.
    # The two populations do not overlap, so a gate at 0.05% of span rejects 81 of
    # the 82 noise candidates and keeps every real one.
    f_here = _sample_field([p], sphere_radius, mesh_pts, f_vals, find_k_nearest)
    if not f_here:
        return RidgePoint(p, 0.0, 0.0, False, "fit_failed")
    span = float(np.max(f_vals) - np.min(f_vals)) if len(f_vals) else 0.0
    f_there = _sample_field([moved], sphere_radius, mesh_pts, f_vals, find_k_nearest)
    gain = (f_there[0] - f_here[0]) / span if (f_there and span > 0.0) else 0.0
    if span > 0.0 and gain < _CREST_MIN_GAIN_FRACTION:
        # The step lands on ground no higher than the sample: the field does not
        # present a ridge here. Near a saddle that is the truth, not a failure to
        # find an answer, and it is reported as a refusal with a reason.
        return RidgePoint(p, 0.0, delta, False, "no_crest")
    # What the OTHER side offers, for the report. The operator asked whether a more
    # developed ridge lies the other way; this is that number, per sample, in the
    # shipped tool rather than in a one-off script.
    opposite = [c for c in crests if c[0] * delta < 0.0]
    other: tuple[float, float] | None = None
    if opposite:
        best_opp = max(opposite, key=lambda c: c[1])
        other = (best_opp[0], (best_opp[1] - f_here[0]) / span)
    return RidgePoint(moved, delta, delta, True, "ok", gain, other)


def _resample_segment(
    path_unit: npt.NDArray[np.float64],
    lo: int,
    hi: int,
    step_rad: float,
) -> list[tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Points spaced ``step_rad`` apart along the polyline from ``lo`` to ``hi``.

    The samples lie ON THE EXISTING PATH, which is the initial guess; the ridge
    direction comes from the field, not from them. The two end nodes are excluded
    because they are pinned. The tangent at a sample is the direction of the
    polyline segment it falls in, projected to the tangent plane.

    Returns:
        List of ``(fraction_along, point_unit, tangent_unit)``.
    """
    pts = path_unit[lo : hi + 1]
    if len(pts) < 2:
        return []
    cum = [0.0]
    for k in range(len(pts) - 1):
        cum.append(cum[-1] + angular_distance(pts[k], pts[k + 1]))
    total = cum[-1]
    if total <= 0.0 or step_rad <= 0.0:
        return []
    out: list[tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    n = math.floor(total / step_rad)
    k = 0
    for j in range(1, n + 1):
        target = j * step_rad
        if target >= total - 1e-12:
            break
        while k < len(cum) - 2 and cum[k + 1] <= target:
            k += 1
        span = cum[k + 1] - cum[k]
        frac = 0.0 if span <= 0.0 else (target - cum[k]) / span
        a = pts[k]
        b = pts[k + 1]
        p = unit(a + frac * (b - a))
        if p is None:
            continue
        out.append((target / total, p, b - a))
    return out


def probe_gradient_ridge(
    arc_points: Sequence[Sequence[float]],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    window_pitch: float = CORNER_WINDOW_PITCH,
    excess_deg: float = CORNER_EXCESS_DEG,
    sample_deg: float = SEGMENT_SAMPLE_DEG,
    stencil_pitch: float = RIDGE_STENCIL_PITCH,
    reach_pitch: float = RIDGE_REACH_PITCH,
) -> RidgeProbe:
    """Measure the ridge beside one gradient-traced connector. Moves nothing.

    The pipeline:

    1. :func:`detect_corners` -- the windowed chord deflection above the path's own
       median baseline. This is reused UNCHANGED from the refinement, because it is
       exactly the operator's distinction: a sharp CONTINUOUS change in the average
       direction is a corner, a sawtooth is not.
    2. Pin the endpoints and the corners, and split the path into sub-segments.
    3. Space samples ``sample_deg`` apart along each sub-segment, ON THE PATH.
    4. Step each sample onto the most developed ridge within reach, scanning the
       across-crest direction the Hessian defines with :func:`ridge_point_at`, and
       record the refusals with their reasons.

    Args:
        arc_points: (M, 3) connector polyline, WORLD frame, launch port first.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on ``mesh_pts``.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus. None when the positions are already centred.
        find_k_nearest: Spatial index over ``mesh_pts``; strongly recommended.
        window_pitch: Corner window half-width in mesh pitches.
        excess_deg: Corner threshold above the baseline, in degrees.
        sample_deg: Probe sample spacing along the path, in degrees.
        stencil_pitch: Hessian disc radius in mesh pitches.
        reach_pitch: Largest accepted ridge step, in mesh pitches.

    Returns:
        A :class:`RidgeProbe`. A path too short to carry a segment yields an empty
        probe, so a caller can hand it every connector.
    """
    m = len(arc_points)
    if m < _MIN_PATH or sphere_radius <= 0.0 or delta_theta_mesh_rad <= 0.0:
        return RidgeProbe()

    original = _path_units(arc_points, center)
    corners, _, _ = detect_corners(
        original, delta_theta_mesh_rad, window_pitch, excess_deg
    )
    nodes = sorted({0, m - 1, *corners})
    step_rad = math.radians(sample_deg)

    samples: list[RidgeSample] = []
    # Arc length is reported CUMULATIVE along the whole path. A segment-local
    # reading restarts at zero after every knot, which puts two rows labelled
    # "5.5" in one table and makes the operator reconcile them against the bend.
    cum = [0.0]
    for i in range(m - 1):
        cum.append(cum[-1] + angular_distance(original[i], original[i + 1]))
    for lo, hi in _segment_bounds(m, nodes):
        for frac, p, tangent in _resample_segment(original, lo, hi, step_rad):
            rp = ridge_point_at(
                p,
                tangent,
                mesh_pts,
                f_vals,
                sphere_radius,
                delta_theta_mesh_rad,
                find_k_nearest=find_k_nearest,
                stencil_pitch=stencil_pitch,
                reach_pitch=reach_pitch,
            )
            arc = cum[lo] + frac * (cum[hi] - cum[lo])
            samples.append(RidgeSample(math.degrees(arc), p, rp))

    accepted = [s.point.signed_offset_rad for s in samples if s.point.accepted]
    pitch = delta_theta_mesh_rad
    return RidgeProbe(
        knot_indices=corners,
        knot_positions_world=tuple(
            to_global(
                (
                    float(original[i][0] * sphere_radius),
                    float(original[i][1] * sphere_radius),
                    float(original[i][2] * sphere_radius),
                ),
                center,
            )
            if center
            else (
                float(original[i][0] * sphere_radius),
                float(original[i][1] * sphere_radius),
                float(original[i][2] * sphere_radius),
            )
            for i in corners
        ),
        samples=tuple(samples),
        n_accepted=len(accepted),
        median_offset_pitch=float(np.median([abs(a) for a in accepted]) / pitch)
        if accepted
        else 0.0,
        max_offset_pitch=float(max((abs(a) for a in accepted), default=0.0) / pitch),
    )
