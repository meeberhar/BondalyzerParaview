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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any, Literal, Protocol, TypeVar, cast, runtime_checkable

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
    "ARC_GRADIFY_NODES",
    "CORNER_BASELINE_MULT",
    "CORNER_EXCESS_DEG",
    "CORNER_WINDOW_PITCH",
    "CURVE_OUTPUT_PITCH",
    "GRADIFY_MAX_NODES",
    "GRADIFY_OUTPUT_PITCH",
    "NODE_PITCH",
    "QUAD_FIT_MAX_CONTROL_PITCH",
    "QUAD_FIT_MIN_GAIN_FRACTION",
    "QUAD_FIT_NM_MAX_ITERS",
    "QUAD_FIT_NM_STEP_PITCH",
    "QUAD_FIT_OFFROUTE_MARGIN",
    "QUAD_FIT_SCAN_STRIDE",
    "QUAD_FIT_SEARCH_PITCH",
    "QUAD_FIT_SEED_ANGLES",
    "QUAD_FIT_SEED_RADII",
    "RIDGE_REACH_PITCH",
    "RIDGE_STENCIL_PITCH",
    "SEGMENT_SAMPLE_DEG",
    "SMOOTH_MIN_TEETH",
    "SMOOTH_OUTPUT_PITCH",
    "TOOTH_TURN_DEG",
    "ArcGradification",
    "GradifyPolarity",
    "QuadBezierFit",
    "RidgeProbe",
    "RidgeRefinement",
    "RidgeSample",
    "SawtoothSmoothing",
    "StructuralCurve",
    "chord_deflection_deg",
    "coarse_structural_curve",
    "detect_chatter_runs",
    "detect_corners",
    "fit_quad_bezier",
    "gradify_arc",
    "gradify_connections",
    "point_turn_deg",
    "polarity_field",
    "probe_gradient_ridge",
    "quad_bezier_points",
    "ridge_point_at",
    "smooth_gradient_ridge",
    "smooth_sawtooth_runs",
    "span_field_integral",
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

# A Newton step shorter than this many pitches is the quadratic model saying the
# crest is HERE: the sample already satisfies the local gradient condition, and the
# move needed is below the mesh's own resolution. Such a step is ACCEPTED as a
# measurement (``reason="on_crest"``) rather than refused as ``no_crest``.
#
# Why this is not just cosmetics. The gain gate above asks whether the step lands on
# GROUND HIGHER THAN THE SAMPLE, and a correct zero-length step cannot answer yes: it
# lands where it started. Refusing it makes every already-correct node look like a
# failure, so the arc is reported as un-gradifiable when it is in fact a gradient path
# -- measured on ethene3 atom 1, 71% of curvedness nodes and 95% of Pd20k atom 1's
# are within HALF a pitch of the crest, and the shipped rule carried all of them.
#
# The tolerance is a QUARTER pitch because that is what separates the two populations
# the field actually presents. The ripple population measured for the gain gate steps
# 0.54-1.07 pitch (median 1.19 deg, max 2.39 deg on a 2.2278 deg pitch), so a quarter
# pitch never waves a ripple peak through; the broad flat crest of a condensed field
# sits well inside it.
_CREST_ON_CREST_PITCH = 0.25

# Polar grid used for the knot search: rings x spokes, evaluated at the knot's
# neighbourhood. Deterministic, derivative-free, and small enough to be cheap.
_KNOT_RINGS = 3
_KNOT_SPOKES = 8

# Minimum vertices a path must have to be refined. Below this there is no interior
# to relax and no window to measure.
_MIN_PATH = 5

# --- arc gradification -----------------------------------------------------
#
# A geodesic connector is a great circle between two CP centres. It is straight by
# construction and the crest it connects is not, so the line the operator is shown
# as "the connection between this saddle and this maximum" leaves the ridge by up
# to half the crest's excursion. This section bends it onto the ridge WITHOUT
# changing its endpoints: the arc keeps its identity (same saddle, same CP, same
# topology) and only its geometry moves. It is method 2 of the operator's three --
# take the existing arc, put N nodes on it, and move each one SIDEWAYS until the
# local crest criterion is satisfied -- and it deliberately reuses the probe's own
# criterion (:func:`ridge_point_at`), the measurement the operator has been
# inspecting with "Probe Ridge Points (Diagnostic)", so the rule that MOVES the
# line is the same rule that VERIFIES it.
#
# Default number of free nodes placed on a geodesic for gradification. The operator
# asked for "a number of points (say, N = 20) along the arc", and 20 is what the
# diagnostic tool's node count settled on (inspect_max_connections' --arc-probe-nodes
# default). Nodes are interpolated on the GEODESIC, so the count is the resolution
# of the bend: the crest's excursion is a smooth function of arc length, so 20 nodes
# over a 30-60 deg connector samples it several times per degree of deflection.
ARC_GRADIFY_NODES = 20.0

# Ceiling of the node-count control. The geodesic itself carries ARC_SAMPLES + 1 =
# 25 vertices, and nodes are placed by arc length (the original vertex set is not
# the node set), so a count above 25 is legal -- it just buys sub-vertex resolution.
# Beyond this the cost is 25 stencil samples + 25 scan samples per node per arc,
# against a crest that a pitch-spaced output line cannot render any finer.
GRADIFY_MAX_NODES = 40.0

# Which critical point the connector is climbing to, or descending from. The two are
# MIRROR criteria on the same field, not two criteria: a ridge is a place where the
# field falls away on both sides, a valley a place where it rises on both sides, so
# the descent is the ascent run on the NEGATED field. That is exactly what
# :func:`polarity_field` does, which is why the whole local rule below stays single
# and sign-free -- flipping the sign once at the entry point turns every internal
# comparison (the Hessian's eigenvalue sign, the Newton step, the crest scan's local
# maxima, the gain gate's RISE) into its correct mirror without a second code path.
#
# The names follow the rest of the project (polarity_detector's "maximum"/"minimum"),
# and "maximum" is the default because the ascent table is the one the card's own
# numbers describe.
GradifyPolarity = Literal["maximum", "minimum"]


def polarity_field(
    f_vals: npt.NDArray[np.float64], polarity: str
) -> npt.NDArray[np.float64]:
    """The field as the local criterion must see it for this polarity.

    ``"maximum"`` hands the field back unchanged: a connector climbing to a maximum
    sits on a RIDGE, where the transverse curvature is negative and the crest is the
    highest ground across the path.

    ``"minimum"`` returns the negation: a connector descending to a minimum sits on a
    VALLEY floor, where the transverse curvature is positive and the floor is the
    LOWEST ground across the path. Negating turns that floor into a crest, so every
    downstream test -- the ``lam < 0`` gate that rejects a pathless direction, the
    Newton step, ``_scan_crests``' local maxima, and the gain gate's RISE -- is the
    exact mirror rather than a parallel implementation that can drift out of step
    with the ascent.

    Why this is needed at all, measured: on ethene3 atom 1 the descent arc
    ``SAD_1594 -> E-MIN_MS5`` has ``|e- . along| = 0.98-1.00`` at every free node,
    i.e. the most-NEGATIVE eigenvector points ALONG the path, and
    ``|e+ . along| = 0.07-0.64`` -- the transverse axis of a valley floor is the
    most-POSITIVE one. The ascent rule therefore took the along-path vector as its
    across-crest axis, found it had no transverse component, and refused every node
    ``along_path``; the arc came back bit-identical to the great circle it started
    as. That is the whole reason a descending connector still renders straight.

    The returned array is a fresh negation for ``"minimum"`` and the input itself for
    ``"maximum"``; a caller must not mutate it. The span the gain gate normalises by
    is INVARIANT under this sign flip (percentiles swap and the difference is the
    same), so a caller that measured a robust span before the flip keeps using it.

    Args:
        f_vals: (N,) field values on the mesh.
        polarity: ``"maximum"`` or ``"minimum"``. An unrecognised value is treated as
            ``"maximum"`` rather than raising, matching the tables' own default.

    Returns:
        ``f_vals`` itself for ``"maximum"``, a new negated array for ``"minimum"``.
    """
    if polarity == "minimum":
        return -f_vals
    return f_vals


# Output vertex spacing of a gradified arc, in mesh pitches. The nodes are ~1.5 deg
# apart on a 30-60 deg connector, which is coarse ground for a line the viewer draws
# as a polydata strip; the result is resampled to the same 1-pitch spacing the
# smoother's output uses (SMOOTH_OUTPUT_PITCH) so a gradified geodesic and a
# smoothed traced path are tessellated alike. This changes the DESCRIPTION LENGTH
# only, never the geometry: the polyline passes through the nodes either way.
GRADIFY_OUTPUT_PITCH = 1.0

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
        reason: ``"ok"``, ``"on_crest"`` (the Newton step is below
            :data:`_CREST_ON_CREST_PITCH`: the sample already satisfies the local
            gradient condition, so the ridge is here rather than elsewhere),
            ``"no_frame"``, ``"fit_failed"``, ``"no_ridge"`` (the
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
        axis_unit: The unit scan axis the step was measured along (the Hessian's
            most-negative eigenvector in the tangent plane), or None when the fit
            never got that far. Published on REFUSALS as well as acceptances: it is
            a property of the local field, not of the step's acceptance, and
            :func:`gradify_arc` needs an axis for a node it carries, whose step is
            inherited from its neighbours while its direction is its own.
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
    axis_unit: npt.NDArray[np.float64] | None = None
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


def _robust_field_span(f_vals: npt.NDArray[np.float64], trim: float = 0.005) -> float:
    """Field span with the extreme ``trim`` fraction discarded at each end.

    The gain gate is a fraction of the field's span, so a handful of spike vertices
    set the bar for the whole field. On ethene3 atom 1 curvedness, 2 vertices of 10081
    carry a curvature spike and hold a global span of 0.4802, while the p5-p95 band is
    0.003531 -- 0.74% of it. A gate normalised by the global span therefore demands a
    rise two orders of magnitude above what the field's own structure offers, and no
    legitimate sub-pitch correction can ever pass.

    The trim is 0.5% a side: wide enough to drop the spike population, narrow enough to
    leave a genuine maximum (which occupies one vertex in 10000) untouched. A field too
    short to trim degrades to the plain max - min, so a degenerate input cannot make
    the span zero or negative.

    Args:
        f_vals: (N,) field values.
        trim: Fraction discarded at each end.

    Returns:
        The trimmed span, or 0.0 for an empty array.
    """
    n = len(f_vals)
    if n == 0:
        return 0.0
    if n < 20:  # too short to trim meaningfully; max - min is the whole population
        return float(np.max(f_vals) - np.min(f_vals))
    lo = float(np.percentile(f_vals, 100.0 * trim))
    hi = float(np.percentile(f_vals, 100.0 * (1.0 - trim)))
    span = hi - lo
    # A field whose trimmed band collapsed (a constant, or a spike on nearly every
    # vertex) falls back to the plain span so the gate never becomes vacuous.
    return span if span > 0.0 else float(np.max(f_vals) - np.min(f_vals))


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
    on_crest_pitch: float = _CREST_ON_CREST_PITCH,
    span_override: float | None = None,
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
        on_crest_pitch: Steps shorter than this many pitches are accepted as
            ``on_crest``; see :data:`_CREST_ON_CREST_PITCH`.
        span_override: Field span used to normalise the gain gate, for a caller that
            has already measured it once. None recomputes ``max - min`` per call,
            which is what a single direct call wants; the geometry pipeline passes a
            robust span so one handful of spike vertices cannot set the gate for the
            whole field.

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
        return RidgePoint(p, 0.0, delta, False, "out_of_reach", axis_unit=axis)
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
        return RidgePoint(p, 0.0, delta, False, "fit_failed", axis_unit=axis)
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
    if span_override is not None:
        span = float(span_override)
    else:
        span = float(np.max(f_vals) - np.min(f_vals)) if len(f_vals) else 0.0
    f_there = _sample_field([moved], sphere_radius, mesh_pts, f_vals, find_k_nearest)
    gain = (f_there[0] - f_here[0]) / span if (f_there and span > 0.0) else 0.0
    # A step below the on-crest tolerance is the model saying the crest is HERE, and it
    # is accepted as a measurement rather than refused: a correct zero-length step lands
    # where it started, so the gain gate can never see a rise for it. See
    # _CREST_ON_CREST_PITCH for the population measurements that set the tolerance.
    already = abs(delta) <= on_crest_pitch * delta_theta_mesh_rad
    if span > 0.0 and gain < _CREST_MIN_GAIN_FRACTION and not already:
        # The step lands on ground no higher than the sample: the field does not
        # present a ridge here. Near a saddle that is the truth, not a failure to
        # find an answer, and it is reported as a refusal with a reason.
        return RidgePoint(p, 0.0, delta, False, "no_crest", axis_unit=axis)
    # What the OTHER side offers, for the report. The operator asked whether a more
    # developed ridge lies the other way; this is that number, per sample, in the
    # shipped tool rather than in a one-off script.
    other: tuple[float, float] | None = None
    opposite = [c for c in crests if c[0] * delta < 0.0]
    if opposite and span > 0.0:
        best_opp = max(opposite, key=lambda c: c[1])
        other = (best_opp[0], (best_opp[1] - f_here[0]) / span)
    return RidgePoint(
        moved,
        delta,
        delta,
        True,
        "on_crest" if already else "ok",
        gain,
        axis,
        other,
    )


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
    span_override: float | None = None,
    polarity: GradifyPolarity = "maximum",
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
        span_override: Field span for the gain gate. None measures the robust span
            once for the whole path; see :func:`_robust_field_span`.
        polarity: ``"maximum"`` for a connector climbing to a maximum (a RIDGE),
            ``"minimum"`` for one descending to a minimum (a VALLEY FLOOR). The
            descent is the ascent's own rule on the negated field, applied once here
            by :func:`polarity_field`, so the two polarities cannot drift apart.
            A probe that ignores this reports the ascent's ridge for a descent arc,
            which is a different question than the operator asked.

    Returns:
        A :class:`RidgeProbe`. A path too short to carry a segment yields an empty
        probe, so a caller can hand it every connector.
    """
    m = len(arc_points)
    if m < _MIN_PATH or sphere_radius <= 0.0 or delta_theta_mesh_rad <= 0.0:
        return RidgeProbe()

    # The descent is the ascent run on the negated field; see polarity_field for the
    # eigenvalue measurement that forced this. Done ONCE, before the span, so every
    # sample below is graded by the same criterion gradify_arc would apply.
    field = polarity_field(np.asarray(f_vals, dtype=np.float64), polarity)

    original = _path_units(arc_points, center)
    corners, _, _ = detect_corners(
        original, delta_theta_mesh_rad, window_pitch, excess_deg
    )
    nodes = sorted({0, m - 1, *corners})
    step_rad = math.radians(sample_deg)
    # Measured once for the whole path, not per sample: the gain gate is a fraction of
    # the field's span, and the robust span is what keeps a handful of spike vertices
    # from setting the bar for every sample on the crest. The probe and gradify_arc
    # must agree here, or the diagnostic reports a different ridge than the operator
    # would get.
    span = _robust_field_span(field) if span_override is None else float(span_override)

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
                field,
                sphere_radius,
                delta_theta_mesh_rad,
                find_k_nearest=find_k_nearest,
                stencil_pitch=stencil_pitch,
                reach_pitch=reach_pitch,
                span_override=span,
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


# --- arc gradification -----------------------------------------------------


@runtime_checkable
class GradifyRegion(Protocol):
    """The disc shape :func:`gradify_arc` needs to know what it cannot measure.

    ``separatrix_tracer.ExcisionRegion`` satisfies it. Taking the protocol instead
    of the class keeps this module free of the connection data model, the same
    decoupling as ``separatrix_tracer.TraceTarget`` and ``ExcisionSource`` -- and
    here it is also load-bearing against a cycle: ``max_connection`` imports this
    module, so this module must not import ``max_connection``, which is what pulls
    in ``separatrix_tracer``.

    Attributes:
        centre_unit: Region centre, CENTRED unit radial.
        radius_rad: Angular radius of the disc the ascent may cross but not end in.
    """

    @property
    def centre_unit(self) -> npt.NDArray[np.float64]: ...

    @property
    def radius_rad(self) -> float: ...


@runtime_checkable
class GradifiableConnection(Protocol):
    """The record shape :func:`gradify_connections` needs.

    Both ``MaxConnection`` and ``MinConnection`` satisfy it. The function reads
    ``method`` and ``arc_points`` and returns a replacement of the SAME type, so
    the ascent table and the descent table -- whose records carry different field
    names for the endpoint (``max_id`` / ``min_id``) -- are gradified by one rule.

    Attributes:
        method: ``"geodesic"`` or ``"gradient"``; the geometry the record carries.
        arc_points: World-frame connector polyline, saddle centre first.
    """

    @property
    def method(self) -> str: ...

    @property
    def arc_points(self) -> tuple[tuple[float, float, float], ...]: ...


_ConnectionT = TypeVar("_ConnectionT", bound=GradifiableConnection)


@dataclass(frozen=True)
class ArcGradification:
    """The result of bending one geodesic connector onto the field's crests.

    The world-frame output of :func:`gradify_arc`. Like :class:`RidgeRefinement`,
    the vertex count of ``path`` differs from the input's (the nodes are placed by
    arc length and the result is resampled), so nothing is reported per vertex: the
    counts are over NODES and the offsets are the measurement, not a shift.

    Attributes:
        path: The gradified polyline, WORLD frame, saddle centre first. Equal to
            the input when nothing moved.
        changed: True when at least one node stepped off the great circle. A path
            whose every node was refused or pinned reports False, which is how a
            caller tells "gradified" from "the field said nothing here".
        n_nodes: Free nodes placed on the geodesic and graded (pinned nodes are not
            counted -- they were never offered to the field).
        n_moved: Free nodes the field ACCEPTED, i.e. that stepped onto a crest.
        n_carried: Free nodes refused by the field that inherited an interpolated
            step from their accepted neighbours. Reported, never hidden: these are
            the positions the algorithm invented, not the ones it measured.
        n_pinned: Nodes held on the great circle by rule -- the two CP centres, every
            vertex inside an excision region, and each region's entry/exit crossing
            of the arc (the straight transit's own vertices, plus the outside vertex
            of each crossing segment).
        median_offset_pitch: Median |step| of the ACCEPTED nodes, in mesh pitches --
            the honest size of the correction, the same statistic the probe reports.
        max_offset_pitch: The largest accepted step, in pitches.
        residual_median_pitch: Median |step| the SAME criterion asks for a SECOND
            time, measured at each moved node's NEW position with the BENT line's own
            tangent, in pitches. This is the arc's error as a gradient path: a step
            near zero says the node is on the crest, and the drop from
            :attr:`median_offset_pitch` is the proof that the bend worked. It is not
            expected to be exactly zero -- a single Newton step converges to the crest
            only where the quadratic model is good, and the carry interpolates.
        residual_max_pitch: The largest such residual step.
        status: ``"ok"`` (nodes moved), ``"all_refused"`` (every free node was
            refused), ``"pinned"`` (no free node: the whole arc is centre, region,
            or crossing ground), ``"degenerate"`` (the path is too short to carry a
            node), ``"no_field"`` (the field inputs were unusable).
    """

    path: tuple[tuple[float, float, float], ...]
    changed: bool = False
    n_nodes: int = 0
    n_moved: int = 0
    n_carried: int = 0
    n_pinned: int = 0
    median_offset_pitch: float = 0.0
    max_offset_pitch: float = 0.0
    residual_median_pitch: float = 0.0
    residual_max_pitch: float = 0.0
    status: str = "ok"


def _polyline_length(path_unit: npt.NDArray[np.float64]) -> list[float]:
    """Cumulative arc length (radians) at every vertex of a unit-radial polyline."""
    cum = [0.0]
    for i in range(len(path_unit) - 1):
        cum.append(cum[-1] + angular_distance(path_unit[i], path_unit[i + 1]))
    return cum


def _slerp(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], t: float
) -> npt.NDArray[np.float64]:
    """Point at fraction ``t`` along the GREAT CIRCLE from unit radial ``a`` to ``b``.

    The segment walk of :func:`gradify_arc` must subdivide on the sphere, not in the
    chord: a linear interpolation's radial projection curves off the great circle
    through its endpoints by ~length²/8, which is 0.08 pitch on a 2.5 pitch segment --
    small, systematic, and exactly the deviation a PIN promises is zero. Slerp keeps a
    sub-vertex of a coplanar pair coplanar, to machine precision.

    Args:
        a: Start unit radial.
        b: End unit radial.
        t: Fraction of the arc, clamped to [0, 1].

    Returns:
        The interpolated unit radial (``a`` when the pair is degenerate).
    """
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-12:
        return a
    s = math.sin(omega)
    f = min(max(t, 0.0), 1.0)
    return (math.sin((1.0 - f) * omega) * a + math.sin(f * omega) * b) / s


def _interp_node(
    path_unit: npt.NDArray[np.float64],
    cum: Sequence[float],
    lo: int,
    hi: int,
    frac: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Point at ``frac`` of the arc length between vertices ``lo`` and ``hi``.

    The companion of :func:`_resample_segment` for a NODE COUNT rather than a
    spacing: it linearly interpolates along the polyline's chords and returns the
    unit radial with the chord direction that contains it (the tangent the crest
    criterion is measured against). ``frac`` 0 and 1 return the end vertices.

    Args:
        path_unit: (M, 3) centred unit radials of the path.
        cum: Cumulative arc length per vertex, as by :func:`_polyline_length`.
        lo: Start vertex index.
        hi: End vertex index, strictly greater than ``lo``.
        frac: Fraction of the span's arc length, clamped to [0, 1].

    Returns:
        ``(point_unit, tangent_unit)``, or None when the span is degenerate.
    """
    if hi <= lo:
        return None
    total = cum[hi] - cum[lo]
    a = path_unit[lo]
    b = path_unit[hi]
    if total <= 0.0:
        tangent = b - a
        return a, tangent
    target = cum[lo] + min(max(frac, 0.0), 1.0) * total
    k = lo
    while k < hi - 1 and cum[k + 1] <= target:
        k += 1
    span = cum[k + 1] - cum[k]
    f = 0.0 if span <= 0.0 else (target - cum[k]) / span
    p = unit(path_unit[k] + f * (path_unit[k + 1] - path_unit[k]))
    if p is None:  # pragma: no cover - chord of two distinct radials
        return None
    return p, path_unit[k + 1] - path_unit[k]


def _world(
    unit_radial: npt.NDArray[np.float64],
    sphere_radius: float,
    center: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """Scale a centred unit radial to the atom's radius and lift it to world frame."""
    pos = (
        float(unit_radial[0]) * sphere_radius,
        float(unit_radial[1]) * sphere_radius,
        float(unit_radial[2]) * sphere_radius,
    )
    return to_global(pos, center) if center is not None else pos


def _pinned_spans(
    geodesic: npt.NDArray[np.float64],
    regions: Sequence[GradifyRegion],
) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    """Vertices a gradified arc must hold on the great circle, and the spans between.

    Pinned, in order of the reason that pins it:

    - the two endpoints. An arc runs CP-centre to CP-centre, so both centres are
      always region ground: the connection's identity is welded to them and the
      basin boundary's joints depend on them being bit-stable.
    - every vertex inside a region. A region's interior is where the mesh was
      condensed to its fitted boundary, so a Hessian fitted there is a fit over the
      boundary ring -- the crest it reports is an artifact of the condensation.
    - each region's entry/exit CROSSING of this arc: the last vertex outside on the
      near side, the first outside on the far side, and the two disc vertices that
      bracket them. These are the arc's own ports -- the traced connector's rule in
      geometric form, bridge centre to boundary and carry boundary to boundary
      straight -- so the geodesic's transit segment survives untouched. The outside
      port is pinned so the first STEPPED node sits a full node-spacing from the
      straight stretch instead of kinking at the rim; the rule is deliberately
      limited to the crossings, because expanding every inside vertex's neighbours
      would eat the whole of a short arc's measurable ground and switch the
      connection off, which is the opposite of the request.

    Args:
        geodesic: (M, 3) centred unit radials of the great circle.
        regions: The CP excision regions.

    Returns:
        ``(sorted_pinned, spans, vertex_pinned)``. ``spans`` are the ``(lo, hi)`` pairs
        between consecutive pins that carry free ground; spans of one segment have no
        room for a node and are dropped. Empty ``spans`` means the whole arc is
        centre, region, or crossing ground. ``vertex_pinned[i]`` is True when vertex
        ``i`` stands on a pin -- a CP centre, a region's condensed ground, or a
        crossing port -- the mask :func:`_node_layout` uses to keep its nodes off that
        ground.
    """
    m = len(geodesic)
    pinned: set[int] = {0, m - 1}

    def mask_of(region: GradifyRegion) -> list[bool]:
        centre = np.asarray(region.centre_unit, dtype=np.float64)
        radius = float(region.radius_rad)
        return [angular_distance(geodesic[i], centre) <= radius for i in range(m)]

    masks = [mask_of(r) for r in regions]
    v_pinned = [False] * m
    v_pinned[0] = v_pinned[m - 1] = True
    for mask in masks:
        for i, inside in enumerate(mask):
            if inside:
                pinned.add(i)
                v_pinned[i] = True
        for i in range(1, m):
            if mask[i] != mask[i - 1]:
                # A crossing: pin BOTH vertices of the segment that leaves (or enters)
                # the disc. Pinning only the outside vertex leaves the transit starting
                # one segment INSIDE the disc, and the chord from it to the far side
                # then cuts across the region's interior.
                pinned.update({i - 1, i})
                v_pinned[i - 1] = True
                v_pinned[i] = True

    nodes = sorted(pinned)
    spans = [(lo, hi) for lo, hi in pairwise(nodes) if hi - lo >= 2]
    return nodes, spans, v_pinned


def _owned_vertex(cum: Sequence[float], length: float, lo: int, hi: int) -> int:
    """Index of the geodesic vertex whose segment starts at arc length ``length``.

    A free node is placed by ARC LENGTH inside a span, so the ground it stands on is
    the segment that contains it, and that segment's START vertex is the one whose pin
    state decides the node's fate. Deciding by a vertex index computed once at span
    entry (the shape of the first version of this rule) is a silent corruption: the
    index stops tracking the position when nodes are dropped for pinned ground, so a
    far-side pin gets stamped onto near-side nodes, they get dropped, and the chain is
    left with a straight chord across the whole region.

    Args:
        cum: Cumulative arc length per vertex, from :func:`_polyline_length`.
        length: The node's arc length along the geodesic.
        lo: First candidate vertex index (the span's start).
        hi: Last candidate vertex index (the span's end).

    Returns:
        The vertex index ``k`` in [lo, hi - 1] with ``cum[k] <= length < cum[k + 1]``.
    """
    k = lo
    while k < hi - 1 and cum[k + 1] <= length:
        k += 1
    return k


def _node_layout(
    geodesic: npt.NDArray[np.float64],
    cum: Sequence[float],
    spans: Sequence[tuple[int, int]],
    pins: Sequence[int],
    v_pinned: Sequence[bool],
    n_nodes: int,
) -> tuple[
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[bool],
]:
    """Lay ``n_nodes`` free nodes on the measurable ground between the pins.

    Nodes are placed by arc length, NOT on the geodesic's own vertices: the vertex
    set is a tessellation choice (25 points for any arc, long or short) and the node
    count is the resolution of the bend, so a 60 deg connector and a 12 deg one get
    the same number of measurements where they have measurable ground. The allocation
    is largest-remainder over the spans' arc lengths, so the counts sum to ``n_nodes``
    exactly and a span too short to earn a node gets none -- its ground is left to the
    pins around it. A node's tangent is the local geodesic direction (the chord across
    the segment it stands on) so a bent arc's nodes follow the arc, not the end-to-end
    line.

    A node that lands on pinned ground is DROPPED. Spans are delimited by pins, and a
    pin sits at each region's crossing ports, so the interior of a transit lies inside
    a span; placing a node there and pinning it afterwards only moves it by the
    resampling spacing, not by a measured amount. This keeps the promise that nothing
    standing on condensed ground is ever stepped, and it keeps the rim clear: a node
    whose owning segment BEGINS at a pinned vertex is dropped by the same test, because
    the ground under it is the pin's neighbourhood (see :func:`_owned_vertex`).

    The emitted lists are in PATH order, built by merging the PINS and the placed nodes
    by arc length (ties: the pin first). Every pin is emitted, including the whole run
    of a region's interior vertices: a chain that carried only a span's two end pins
    would connect them with one chord, and a chord across a disc the size of a real
    excision region cuts the corner off the great circle by ~length²/8 -- 0.25 pitch on
    a 16 deg span, which is precisely the bending the pin rule forbids. The geodesic's
    own vertices ARE the transit, so they travel.

    Args:
        geodesic: (M, 3) centred unit radials of the great circle.
        cum: Cumulative arc length per vertex, from :func:`_polyline_length`.
        spans: ``(lo, hi)`` pin pairs with room for nodes, from :func:`_pinned_spans`.
        pins: The pinned vertex indices, ascending, from :func:`_pinned_spans`. EVERY
            one of them is emitted: the pins are the chain's skeleton and the nodes are
            threaded between them.
        v_pinned: Per-vertex pin mask, from :func:`_pinned_spans`. A node standing on
            a segment that BEGINS at a pinned vertex is dropped: that ground is the
            pin's neighbourhood, and the pin's own position is what the transit needs.
        n_nodes: Free nodes to place in total.

    Returns:
        ``(positions, tangents, pinned)`` in path order: every pin and every placed
        free node, merged by arc length. ``pinned[i]`` is True when ``positions[i]`` is
        a pin, i.e. a vertex that was never offered to the field.
    """
    m = len(geodesic)
    v_indices = list(pins)
    free_length = sum(cum[hi] - cum[lo] for lo, hi in spans)
    quota = (
        [n_nodes * (cum[hi] - cum[lo]) / free_length for lo, hi in spans]
        if free_length > 0.0
        else [0.0] * len(spans)
    )
    counts = [int(q) for q in quota]
    short = n_nodes - sum(counts)
    order = sorted(
        range(len(spans)),
        key=lambda s: (-(quota[s] - counts[s]), -spans[s][1], spans[s][0]),
    )
    for s in order[: max(0, short)]:
        counts[s] += 1

    positions: list[npt.NDArray[np.float64]] = []
    tangents: list[npt.NDArray[np.float64]] = []
    pinned: list[bool] = []

    # (arc length, tie-break, position, tangent, is_pin): pins sort first at a shared
    # length, because a node sitting exactly on a port stands on the port's ground and
    # has already been dropped by the owner test.
    chain: list[
        tuple[float, int, npt.NDArray[np.float64], npt.NDArray[np.float64], bool]
    ]
    chain = []
    for index in v_indices:
        nxt = geodesic[index + 1] if index + 1 < m else geodesic[index]
        chain.append(
            (float(cum[index]), 0, geodesic[index], nxt - geodesic[index], True)
        )

    for span_index, (lo, hi) in enumerate(spans):
        for j in range(counts[span_index]):
            frac = (j + 1) / (counts[span_index] + 1)
            node = _interp_node(geodesic, cum, lo, hi, frac)
            if node is None:  # pragma: no cover - span has positive length
                continue
            length = cum[lo] + frac * (cum[hi] - cum[lo])
            owner = _owned_vertex(cum, length, lo, hi)
            if v_pinned[owner]:
                continue
            chain.append(
                (
                    float(length),
                    1,
                    node[0],
                    geodesic[owner + 1] - geodesic[owner],
                    False,
                )
            )

    chain.sort(key=lambda e: (e[0], e[1]))
    for _length, _tie, position, tangent, is_pin in chain:
        positions.append(position)
        tangents.append(tangent)
        pinned.append(is_pin)
    return positions, tangents, pinned


def gradify_arc(
    arc_points: Sequence[Sequence[float]],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    regions: Sequence[GradifyRegion] = (),
    n_nodes: float = ARC_GRADIFY_NODES,
    stencil_pitch: float = RIDGE_STENCIL_PITCH,
    reach_pitch: float = RIDGE_REACH_PITCH,
    output_pitch: float = GRADIFY_OUTPUT_PITCH,
    span_override: float | None = None,
    polarity: GradifyPolarity = "maximum",
) -> ArcGradification:
    """Bend a geodesic connector onto the crest, sideways, node by node.

    Method 2, in the operator's words: "take the existing arc, choose a number of
    points along the arc, and move them sideways so that each point satisfies a
    local gradient condition, essentially transforming each arc into a legit
    gradient path". Concretely:

    1. Resample the great circle to ``n_nodes`` FREE nodes, placed by arc length
       between the PINNED positions (:func:`_pinned_spans`), so the nodes live on
       measurable ground and not inside a disc where the field is condensed.
    2. Grade every free node with :func:`ridge_point_at`, the probe's own criterion:
       the Hessian's most-negative eigenvector gives the across-crest AXIS, a Newton
       step along it gives the crest's LOCATION, and the rise and reach gates refuse
       the ground where the field genuinely does not say where the ridge is.
    3. PIN what the field cannot speak for: the two CP centres (an arc runs centre to
       centre, so both ends are always pinned), every vertex inside an excision
       region, and each region's entry/exit CROSSING of this arc -- the outside vertex
       of the crossing segment is the port the straight transit runs to, exactly as a
       gradient connector's boundary crossing is its port, and pinning it keeps the
       first stepped node a full node-spacing from the straight stretch.
    4. CARRY the refusals: a refused, unpinned node takes a linear interpolation of
       its accepted neighbours' steps and moves along its OWN field-derived axis. The
       connector is a single line, so a node the field is silent about is filled in
       from the nodes it answered at, and the count is published (``n_carried``).
    5. Describe the result at ``output_pitch`` -- walking each segment of the stepped
       chain, so every pinned vertex survives as an emitted vertex and a region's
       transit stays the geodesic's own chord -- to match the description length of
       the traced paths the smoother eats.

    The endpoints never move, so the arc's identity -- which saddle, which CP, and
    therefore which basin boundary it bounds -- is untouched: this changes geometry
    only. Nothing here re-decides an endpoint.

    Args:
        arc_points: (M, 3) connector polyline, WORLD frame, saddle centre first.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on ``mesh_pts``.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus. None when the positions are already centred.
        find_k_nearest: Spatial index over ``mesh_pts``; strongly recommended.
        regions: The CP excision regions, whose ground is pinned (see step 3).
        n_nodes: Free nodes to place on the measurable part of the arc.
        stencil_pitch: Hessian disc radius in mesh pitches.
        reach_pitch: Largest accepted step, in mesh pitches.
        output_pitch: Vertex spacing of the emitted polyline, in mesh pitches.
        span_override: Field span for the gain gate. None measures the robust span
            once for the whole field (:func:`_robust_field_span`), which is what the
            geometry pipeline wants; pass an explicit value only to reproduce a
            caller's own normalisation.
        polarity: ``"maximum"`` for a connector climbing to a maximum (a RIDGE),
            ``"minimum"`` for one descending to a minimum (a VALLEY FLOOR). One
            negation at the top of this function turns every internal comparison
            into its mirror; see :func:`polarity_field` for the eigenvalue
            measurement that forced it and for what the ascent-only rule did to
            descending connectors.

    Returns:
        An :class:`ArcGradification`. Degenerate input (fewer than :data:`_MIN_PATH`
        vertices, a non-positive radius/pitch, an unusable field) returns the path
        unchanged with a status naming why, so a caller can hand this every arc it
        has without testing first.
    """
    m = len(arc_points)
    if (
        m < _MIN_PATH
        or sphere_radius <= 0.0
        or delta_theta_mesh_rad <= 0.0
        or len(mesh_pts) == 0
        or len(f_vals) == 0
    ):
        status = "degenerate" if m < _MIN_PATH else "no_field"
        return ArcGradification(
            path=tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points),
            status=status,
        )

    geodesic = _path_units(arc_points, center)
    cum = _polyline_length(geodesic)

    # --- pins: the ground the field cannot speak for -------------------------
    #
    # See :func:`_pinned_spans` for what is pinned and why -- condensed region
    # interiors, and each region's crossing ports with the vertex on each side. The
    # third return is the per-VERTEX mask :func:`_node_layout` grades its nodes on.
    pins, spans, v_pinned = _pinned_spans(geodesic, regions)
    if not spans:
        return ArcGradification(
            path=tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points),
            n_pinned=len(pins),
            status="pinned",
        )

    # --- place the free nodes, proportional to each span's arc length ---------
    layout_pos, layout_tan, layout_pinned = _node_layout(
        geodesic,
        cum,
        spans,
        pins,
        v_pinned,
        max(1, round(float(n_nodes))),
    )

    # --- grade, carry, move --------------------------------------------------
    pitch = delta_theta_mesh_rad
    # The descent is the ascent's own rule on the negated field: a valley floor's
    # transverse axis is its most-POSITIVE eigenvector, which the negated field turns
    # into the most-negative one every gate below expects. Negating ONCE here (rather
    # than branching at each comparison) keeps a single implementation, so the two
    # polarities cannot drift apart.
    field = polarity_field(f_vals, polarity)
    # Measured once for the whole arc, not per node: the gain gate is a fraction of the
    # field's span and the robust span is what keeps a couple of spike vertices from
    # setting the bar (see _robust_field_span). The span is invariant under the
    # negation, so this is the same number either way.
    span = _robust_field_span(field) if span_override is None else float(span_override)
    deltas: list[float] = []
    axes: list[npt.NDArray[np.float64] | None] = []
    accepted: list[bool] = []
    for i, pos in enumerate(layout_pos):
        if layout_pinned[i]:
            deltas.append(0.0)
            axes.append(None)
            accepted.append(False)
            continue
        rp = ridge_point_at(
            pos,
            layout_tan[i],
            mesh_pts,
            field,
            sphere_radius,
            delta_theta_mesh_rad,
            find_k_nearest=find_k_nearest,
            stencil_pitch=stencil_pitch,
            reach_pitch=reach_pitch,
            span_override=span,
        )
        deltas.append(float(rp.delta_rad) if rp.accepted else 0.0)
        # A REFUSED node keeps its axis: the direction is the field's, only the
        # magnitude is inherited. Without this a carried node has nothing to move
        # along and the connector leaves a straight gap wherever the reach failed.
        axis = (
            rp.axis_unit if rp.axis_unit is not None else unit(rp.position_unit - pos)
        )
        axes.append(unit(axis) if axis is not None else None)
        accepted.append(bool(rp.accepted))

    # The carried step: interpolate the accepted magnitudes in node-index space.
    # A node with no accepted neighbour on either side has nothing to inherit, and
    # is left on the great circle (it cannot be moved along an axis the field never
    # gave it, and a straight carry across the gap is what the pins do for the
    # regions).
    carried_flags = [False] * len(layout_pos)
    stamp = [i for i, a in enumerate(accepted) if a]
    if not stamp:
        return ArcGradification(
            path=tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points),
            n_nodes=sum(1 for p in layout_pinned if not p),
            n_pinned=sum(1 for p in layout_pinned if p),
            status="all_refused",
        )
    for i in range(len(layout_pos)):
        if layout_pinned[i] or accepted[i] or axes[i] is None:
            continue  # pragma: no cover - the axis is published on every fit
        left = [s for s in stamp if s < i]
        right = [s for s in stamp if s > i]
        if left and right:
            lo_s, hi_s = left[-1], right[0]
            w = (i - lo_s) / (hi_s - lo_s)
            deltas[i] = (1.0 - w) * deltas[lo_s] + w * deltas[hi_s]
        elif left:
            deltas[i] = deltas[left[-1]]
        else:
            deltas[i] = deltas[right[0]]
        # Every node the field refused between acceptances is CARRIED, including one
        # that inherits a zero step: the accounting identity n_nodes == n_moved +
        # n_carried must hold, because a caller reads the two counts as "measured"
        # versus "inherited" and there is no third kind of free node.
        carried_flags[i] = True

    out_units: list[npt.NDArray[np.float64]] = []
    for i, pos in enumerate(layout_pos):
        step = deltas[i]
        axis = axes[i]
        if layout_pinned[i] or axis is None or step == 0.0:
            out_units.append(pos)
            continue
        stepped = _offset_unit(pos, axis, step)
        out_units.append(stepped if stepped is not None else pos)

    # --- describe the bent line at the traced paths' spacing ------------------
    #
    # Per SEGMENT of the stepped chain, not by a uniform arc-length walk over the
    # whole line: every chain vertex must appear in the emitted path, because the
    # pins among them are the straight transit's definition. A uniform walk lands
    # between the vertices, and interpolating across a port puts a vertex inside the
    # disc a hair off the great circle -- invisible in the drawing, fatal to the
    # promise that condensed ground is never bent. Each segment is walked in
    # round(length / step) sub-steps (at least one) ON THE GREAT CIRCLE between its
    # endpoints (:func:`_slerp`), so the description length matches the traced paths
    # the Step 6 smoother eats and a sub-vertex of a pinned pair stays on the plane.
    gradified = np.asarray(out_units, dtype=np.float64)
    step_rad = max(output_pitch, 1e-9) * pitch
    emitted: list[npt.NDArray[np.float64]] = [gradified[0]]
    for i in range(len(gradified) - 1):
        lo, hi = gradified[i], gradified[i + 1]
        n_sub = max(1, round(angular_distance(lo, hi) / step_rad))
        for j in range(1, n_sub + 1):
            emitted.append(_slerp(lo, hi, j / n_sub))
    emitted[-1] = gradified[-1]  # the CP centre, bit-exact
    if len(emitted) < _MIN_PATH:
        # A very short arc cannot carry one-pitch vertices: hand back the stepped
        # nodes themselves rather than a 3-point sketch of them.
        emitted = list(gradified)

    path = tuple(_world(u, sphere_radius, center) for u in emitted)
    offsets = [abs(deltas[i]) for i, a in enumerate(accepted) if a]

    # --- the arc's error, measured twice ------------------------------------
    #
    # The same criterion, asked a second time at each moved node's NEW position, with
    # the BENT line's own tangent. This is the residual the operator asked method 3 to
    # minimise, and it is the proof that the bend worked: a Newton step lands on the
    # crest only where the quadratic model is good, so the second step is normally an
    # order of magnitude smaller than the first, and a residual that is NOT smaller
    # says the ground was too curved (or the step was carried) for one pass.
    residuals: list[float] = []
    for i, is_accepted in enumerate(accepted):
        if not is_accepted or layout_pinned[i]:
            continue
        lo = out_units[max(0, i - 1)]
        hi = out_units[min(len(out_units) - 1, i + 1)]
        rp2 = ridge_point_at(
            out_units[i],
            hi - lo,
            mesh_pts,
            field,
            sphere_radius,
            delta_theta_mesh_rad,
            find_k_nearest=find_k_nearest,
            stencil_pitch=stencil_pitch,
            reach_pitch=reach_pitch,
            span_override=span,
        )
        if rp2.accepted:
            residuals.append(abs(rp2.delta_rad))
    residual_med = float(np.median(residuals) / pitch) if residuals else 0.0
    residual_max = float(max(residuals, default=0.0) / pitch)

    return ArcGradification(
        path=path,
        changed=True,
        n_nodes=sum(1 for p in layout_pinned if not p),
        n_moved=sum(1 for a in accepted if a),
        n_carried=sum(1 for c in carried_flags if c),
        # The layout's own pins: the arc's two centres plus every span start, i.e. a
        # region's crossing port when regions exist. Pinned vertices DROPPED from the
        # node layout are not counted -- the node was never there to pin.
        n_pinned=sum(1 for p in layout_pinned if p),
        median_offset_pitch=float(np.median(offsets) / pitch) if offsets else 0.0,
        max_offset_pitch=float(max(offsets, default=0.0) / pitch),
        residual_median_pitch=residual_med,
        residual_max_pitch=residual_max,
        status="ok",
    )


def gradify_connections(
    connections: Sequence[_ConnectionT],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    regions: Sequence[GradifyRegion] = (),
    n_nodes: float = ARC_GRADIFY_NODES,
    polarity: GradifyPolarity = "maximum",
) -> tuple[list[_ConnectionT], dict[str, Any]]:
    """Gradify every straight connector in a table; report the table's numbers.

    The table-level wrapper of :func:`gradify_arc`, the twin of
    ``max_connection.refine_connections``. It is agnostic to WHICH TABLE it is given:
    the ascent table (``MaxConnection``) and the descent table (``MinConnection``)
    carry the connector in the same ``arc_points`` field and the geometry in the same
    frame, so ONE rule fixes both -- but the two tables need OPPOSITE SIGNS of the
    same rule, which is what ``polarity`` selects. A basin boundary built from
    descent arcs therefore inherits the improvement only if the caller passes
    ``"minimum"``; see :func:`polarity_field` for the measured consequence of not
    doing so. Each record it moves is replaced with the gradified polyline plus
    ``gradified=True`` and the node counts; records that refused, that were all-pinned,
    or that are too short are returned IDENTICAL (the same object, not a copy), which
    is how a caller tells the two apart.

    The method tag is NOT changed. ``method`` records how the ENDPOINT was found --
    the ray, for a geodesic -- and the viewer's colour split and the smoother's
    geodesic skip both read it; ``gradified`` is the geometry flag.

    Args:
        connections: The connection table, in any order.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on ``mesh_pts``.
        sphere_radius: Atom sphere radius R.
        center: World-frame nucleus. None when the positions are already centred.
        find_k_nearest: Spatial index over ``mesh_pts``; strongly recommended.
        regions: The CP excision regions whose ground is pinned.
        n_nodes: Free nodes per arc; see :data:`ARC_GRADIFY_NODES`.
        polarity: ``"maximum"`` for an ascent table climbing to maxima,
            ``"minimum"`` for a descent table descending to minima. One negation in
            :func:`gradify_arc` mirrors the whole local rule; see
            :func:`polarity_field`.

    Returns:
        ``(records, stats)``: the new list in input order, and the aggregate the
        viewer publishes as ``connection_stats["arc_gradify_stats"]`` -- counts by
        status, the node/moved/carried/pinned totals, and the median and max
        accepted step in pitches, with ``scope`` naming the set that was measured
        (the same discipline as ``params_schema.make_ridge_probe_stats``).
    """
    out: list[_ConnectionT] = []
    counts = {
        "n_arcs": 0,
        "n_gradified": 0,
        "n_all_refused": 0,
        "n_pinned": 0,
        "n_degenerate": 0,
        "n_nodes": 0,
        "n_moved": 0,
        "n_carried": 0,
        "n_pinned_nodes": 0,
    }
    medians: list[float] = []
    maxima: list[float] = []
    residuals: list[float] = []
    for conn in connections:
        if conn.method != "geodesic":
            out.append(conn)
            continue
        counts["n_arcs"] += 1
        res = gradify_arc(
            arc_points=conn.arc_points,
            mesh_pts=mesh_pts,
            f_vals=f_vals,
            sphere_radius=sphere_radius,
            delta_theta_mesh_rad=delta_theta_mesh_rad,
            center=center,
            find_k_nearest=find_k_nearest,
            regions=regions,
            n_nodes=n_nodes,
            polarity=polarity,
        )
        counts[f"n_{res.status}"] = counts.get(f"n_{res.status}", 0) + 1
        if not res.changed:
            out.append(conn)
            continue
        counts["n_gradified"] += 1
        counts["n_nodes"] += res.n_nodes
        counts["n_moved"] += res.n_moved
        counts["n_carried"] += res.n_carried
        counts["n_pinned_nodes"] += res.n_pinned
        medians.append(res.median_offset_pitch)
        maxima.append(res.max_offset_pitch)
        residuals.append(res.residual_median_pitch)
        out.append(
            cast(
                "_ConnectionT",
                replace(
                    # Every connection record in the project is a frozen dataclass and
                    # `replace` returns the type it is given; the Protocol deliberately
                    # does not claim `__dataclass_fields__`, so the typing is asserted
                    # at this one seam rather than widened onto the records.
                    cast(Any, conn),
                    arc_points=res.path,
                    gradified=True,
                    gradify_nodes=res.n_nodes,
                    gradify_moved=res.n_moved,
                    gradify_carried=res.n_carried,
                    gradify_pinned=res.n_pinned,
                    gradify_median_offset_pitch=res.median_offset_pitch,
                    gradify_max_offset_pitch=res.max_offset_pitch,
                    gradify_residual_pitch=res.residual_median_pitch,
                    gradify_status=res.status,
                ),
            )
        )
    stats: dict[str, Any] = {
        "on": True,
        "n_arcs": counts["n_arcs"],
        "n_gradified": counts["n_gradified"],
        "n_all_refused": counts["n_all_refused"],
        "n_pinned": counts["n_pinned"],
        "n_degenerate": counts["n_degenerate"],
        "n_nodes": counts["n_nodes"],
        "n_moved": counts["n_moved"],
        "n_carried": counts["n_carried"],
        "n_pinned_nodes": counts["n_pinned_nodes"],
        "median_offset_pitch": float(np.median(medians)) if medians else 0.0,
        "max_offset_pitch": float(max(maxima, default=0.0)),
        # The arc's error as a gradient path, measured by the SAME criterion a second
        # time at each moved node's new position (see :func:`gradify_arc`). The drop
        # from ``median_offset_pitch`` is the proof the bend worked; a residual that
        # is not smaller says the ground was too curved for one Newton pass.
        "residual_median_pitch": float(np.median(residuals)) if residuals else 0.0,
        "residual_max_pitch": float(max(residuals, default=0.0)),
        "n_nodes_target": round(float(n_nodes)),
        "scope": "every geodesic connector in the table, full field",
    }
    return out, stats


# --- quadratic Bézier arc fit (method 3, first rung) -----------------------
#
# The operator's third method: "start with the existing arc, and then change it to a
# sphere-surface-mapped polynomial. First as a quadratic which has a single control
# point that can be adjusted". Where :func:`gradify_arc` walks N nodes to the crest one
# local Newton step at a time (method 2), this section replaces the SHAPE: the span
# becomes a spherical quadratic Bézier -- :func:`sphere_frame.squad_point` at
# ``qa = qb = C``, whose ``2t(1-t)`` blend vanishes at both ends, so both welds are
# exact by construction -- and the ONE control point is placed to optimise the whole
# path at once.
#
# THE CRITERION IS THE FUNCTION'S OWN PATH INTEGRAL, not a gradient-alignment error.
# The plan asked for ``1 - |tangent . grad|`` summed along the path. Measured before
# building it, that objective is the weak signal of a much larger one: the condensed
# CURVEDNESS of a bonded system is a spike map whose crest lines are hundreds of times
# deeper than the arc-to-crest excursions this bend is sized for (Pd_20K atom 1, 8888
# vertices: p99.5 - median = 6.8e-2 against a 36 deg connector's 5.9e-5 excursion
# scale), and its gradient direction flips across the ridge at the mesh-pitch scale the
# samples land on. Summed over 20 nodes it therefore measures mostly the noise it stands
# in. The integral of ``f`` itself says the same physical thing -- a ridge is where ``f``
# is highest ACROSS the path -- with a 1/length that cancels the spike's depth, no
# derivative, no stencil, and one sample per node instead of the stencil's 25, and it is
# polarity-signed for free: maximise it saddle->max, negate once and maximise it
# saddle->min (see :func:`polarity_field`). The decisive cross-check -- does a curve
# fitted to the integral also satisfy the gradient criterion? -- is measured with the
# probe in journal.md, against a rule this fit never optimises.
#
# WHAT THE INTEGRAL CAN BUY THAT WE DO NOT WANT, and the four gates that stop it:
#
# 1. Running to a neighbouring CP's peak. The path is a quadratic through two FIXED
#    endpoints and ``|delta|`` is capped at ``search_pitch`` while searching and
#    :data:`QUAD_FIT_MAX_CONTROL_PITCH` when accepting, so it cannot reroute across the
#    sphere; and the off-route gate (:data:`QUAD_FIT_OFFROUTE_MARGIN`) refuses a bend
#    that walks the path into a CP disc the straight arc never entered.
# 2. Diving into a cluster to drink its condensed peak. A cluster's ground is PINNED by
#    :func:`_pinned_spans` -- both centres, every vertex inside a disc, and both
#    vertices of each crossing segment -- exactly as gradify pins it, so the fit only
#    ever moves free spans and every pin is emitted as a verbatim vertex.
# 3. Flat ground and IDW ripple. The gain gate is an ABSOLUTE fraction of the field's
#    ROBUST span (:func:`_robust_field_span`), the same constant and the same
#    normalisation gradify settled on after measuring that a relative window lets a
#    1e-21 ripple look like a 20% rise. A span of zero (a constant field) refuses the
#    whole arc -- deliberately NOT the ``if span > 0.0 and gain < ...`` shape, whose
#    ``span == 0`` branch accepts at zero gain.
# 4. A broad flat-topped ridge, where the integral is stationary over a wide band and
#    the argmax is a SET and not a point (the argument :func:`_ridge_frame` makes against
#    argmax, aimed at a criterion that carries no curvature information at all). Two
#    mitigations: the search STARTS at the straight chord and Nelder-Mead stops in the
#    basin it is dropped into, so the answer is the MINIMAL bend that captures the gain
#    rather than a global maximiser; and the offset cap is a cap, not a hint. Documented
#    limit: on a genuinely flat plateau the chosen bend is arbitrary within the plateau.
#    It is not worse, and the probe readout reports it honestly.
#
# The quadratic is the ladder's first rung only. :attr:`QuadBezierFit.control_offsets` is
# a per-span ``(du, dv)`` pair and a cubic is a per-span ``(du1, dv1, du2, dv2)``, so the
# search below is written to grow into a `pick_order` rung (the lowest order whose gain
# clears the gate) without restructuring -- see journal.md's next-unit list.

# Radius of the control-point search, in mesh pitches, measured from the chord's own
# midpoint. THE CONTROL IS A 2:1 LEVER, measured: a control moved 2 pitches along the
# bending axis pulls the span's middle 1.07 pitches off the great circle, because the
# squad's 2t(1-t) blend peaks at 0.5. So this number is TWICE the bend it buys, and the
# default of 2.0 pitches reaches a crest about one pitch off the arc's middle.
# Deliberately NOT :data:`RIDGE_REACH_PITCH`: a Newton step must be small because it is a
# LOCAL quadratic model, and this is a global search over a SHAPE. The population that
# motivated method 3 is the arc that runs two pitches from a crest gradify measured and
# carried -- Pd_20K's published numbers are 1291 carried nodes against 542 measured, a
# median offset of 28% of the pitch, and a residual bend that is local. -- and a 0.72-pitch reach cannot express that bend at all. Two pitches is a
# half-cell of search in every direction for 17 field evaluations per span.
QUAD_FIT_SEARCH_PITCH: float = 2.0

# Largest control offset ACCEPTED, in mesh pitches. A larger bend is a reroute and not a
# correction, and this is the second line of defence against the peak of a CP the arc was
# never connected to. The accepted offset is never smaller than the search radius (the
# search is clamped to its own bound), so the two constants are independent knobs: the
# first is the budget of the search, the second is the policy of the answer.
QUAD_FIT_MAX_CONTROL_PITCH: float = 4.0

# Smallest accepted improvement, as a fraction of the field's robust span. Copied from
# :data:`_CREST_MIN_GAIN_FRACTION`, whose value was MEASURED on ethene3, Pd_20K and
# CO_plain against the IDW ripple population (a rise of 20% of a 1e-21 local window is
# not an improvement; 0.05% of the field's real span is), and re-measured here on the
# same datasets -- see journal.md. Zero gain is zero improvement, so a flat span is left
# exactly straight, which is the property that keeps the ethene3 arcs (published: 0 of 6
# bent, every node refused) unbent here too.
QUAD_FIT_MIN_GAIN_FRACTION: float = 0.0005

# Off-route gate: refuse an accepted bend that carries a path node into a CP disc the
# STRAIGHT arc never entered, by more than this fraction of the disc's radius. The
# integral is maximised by the peak of anything it can touch, and a cluster's condensed
# ground is the highest ground on the sphere; pinning (gate 2) protects the ground INSIDE
# a disc the arc already transits, this gate protects the discs it does not. The margin is
# not zero because a straight arc's last vertex before a crossing sits a hair outside the
# rim, and a bend of a fraction of a pitch must not be refused for touching the rim of a
# disc the arc was already heading for.
QUAD_FIT_OFFROUTE_MARGIN: float = 0.02

# Seed radii of the grid scan, as multiples of the search radius. Four rings starting at
# half of it: the coarse scan exists only to choose a BASIN for Nelder-Mead, which walks
# the rest of the way, so a ring spacing finer than the simplex step
# (:data:`QUAD_FIT_NM_STEP_PITCH`) buys nothing and costs one field sample per node.
QUAD_FIT_SEED_RADII: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

# Seed angles of the grid scan. The objective is smooth in the control's polar angle --
# the basin's width is set by the crest's curvature, which resolves at the pitch scale and
# not at 30 deg -- and 12 rays bracket every basin while keeping the search isotropic: no
# privileged direction, which matters most for the nearly-straight arcs whose crest is a
# hair off the arc's own plane.
QUAD_FIT_SEED_ANGLES: int = 12

# Node-spacing of the seed scan, in nodes of the full grid. The scan ranks basins, it does
# not grade the answer, so it may subsample; the Nelder-Mead polish and the accepted
# measurement both use the FULL node grid, so the published gain is the whole-path
# integral at the resolution the card advertises.
QUAD_FIT_SCAN_STRIDE: int = 2

# Function evaluations of the Nelder-Mead polish (the initial triangle is free of this
# budget). 40 is roughly a dozen reflection/contraction rounds of a 2-simplex: enough to
# travel a full search radius from a seed at the simplex's step, short enough that a
# slider drag over 52 arcs stays interactive.
QUAD_FIT_NM_MAX_ITERS: int = 40

# Initial simplex edge of the polish, in mesh pitches. A quarter of the search radius:
# wide enough that the simplex sees the crest's curvature (which resolves at the pitch
# scale, not below it), narrow enough that it cannot jump out of the basin the scan chose.
QUAD_FIT_NM_STEP_PITCH: float = 0.25

# Nelder-Mead coefficients, the textbook 2D triple (mirror, expand, shrink), named so a
# future cubic rung tunes one table instead of three literals.
_QUAD_NM_MIRROR: float = 1.0
_QUAD_NM_EXPAND: float = 2.0
_QUAD_NM_SHRINK: float = 0.5

# Slerp degeneracy floor. Below this sine of the interior angle the interpolation is
# replaced by its limit -- the geodesic point itself -- which is what
# :func:`sphere_frame.slerp` does for a coincident pair and what the batched form must do
# to avoid a 0/0 for a control that sits exactly ON a node. Such a control prescribes no
# bend, which is the honest answer, and the gain gate then refuses it.
_QUAD_DEGENERATE_SIN: float = 1e-12

# Basis floor for a span's transverse frame: a pair of radials closer than this to
# parallel is a zero-length span, and closer than this to antipodal has no unique great
# circle. Both get the same treatment -- the span is skipped as degenerate.
_QUAD_DEGENERATE_BASIS: float = 1e-12

# Span statuses, and the arc statuses derived from them. A caller reads these to tell
# "the field offered no gain" from "the bend left the route" from "there was no free
# ground to bend", which is why refusals are named and not merely counted.
_QUAD_SPAN_FITTED = "fitted"
_QUAD_SPAN_GAIN = "rejected_gain"
_QUAD_SPAN_OFFROUTE = "rejected_offroute"
_QUAD_SPAN_DEGENERATE = "degenerate"

_QUAD_ARC_OK = "ok"
_QUAD_ARC_MIXED = "mixed"
_QUAD_ARC_GAIN = "rejected_gain"
_QUAD_ARC_PINNED = "pinned"
_QUAD_ARC_DEGENERATE = "degenerate"
_QUAD_ARC_NO_FIELD = "no_field"


@dataclass(frozen=True)
class QuadBezierFit:
    """The result of fitting one geodesic connector to the spherical quadratic.

    The world-frame output of :func:`fit_quad_bezier`. Like :class:`ArcGradification`,
    the vertex count of ``path`` differs from the input's (the nodes are placed by arc
    length and the result is resampled), so nothing is reported per vertex: the counts
    are over SPANS and NODES and the offsets are the measurement, not a shift. Every
    field has a default, so an unchanged arc is described by its path and a status.

    Attributes:
        path: The fitted polyline, WORLD frame, saddle centre first. Equal to the input
            when nothing moved.
        changed: True when at least one span accepted a bend. An arc whose every span
            refused is returned with its geometry untouched, which is how a caller tells
            "fitted" from "the field said nothing here".
        n_spans: Free spans the arc offered (:func:`_pinned_spans`), i.e. the number of
            control points this quadratic rung was allowed to move.
        n_fitted: Spans that accepted a bend.
        n_rejected: Spans that did not, for any reason; :attr:`span_statuses` is the
            split, and ``n_fitted + n_rejected == n_spans`` always.
        n_nodes: Free nodes placed on the geodesic and sampled (pinned nodes are not
            counted -- they were never offered to the field).
        n_pinned: Chain vertices held on the great circle by rule -- the two CP centres,
            every vertex inside an excision region, and each region's entry/exit crossing
            of the arc.
        max_offset_pitch: Largest accepted control offset, in mesh pitches. The honest
            size of the bend, and the number to watch against
            :data:`QUAD_FIT_MAX_CONTROL_PITCH`: an arc that saturates the cap is a
            reroute the gates failed to stop.
        min_gain_fraction: The SMALLEST accepted gain, in units of the field's robust
            span. The binding gate: :data:`QUAD_FIT_MIN_GAIN_FRACTION` is the floor, so a
            value sitting on the floor means the bend is a coin flip against the ripple.
        mean_gain_fraction: Mean accepted gain, same units.
        mean_f_before: Arc-length-weighted mean field along the STRAIGHT spans that were
            fitted, in field units (polarity-signed). Read with :attr:`mean_f_after`: the
            pair is the whole point of the fit, in the units the field is drawn in.
        mean_f_after: The same quantity along the FITTED spans.
        field_span: The robust span the gains were normalised by, in field units.
            Published so a reader can turn a fraction back into an absolute rise and see
            it against the field's own dynamic range.
        control_offsets: Per span, in span order, the accepted control's offset from the
            chord's midpoint, as ``(du, dv) / pitch``: ``du`` along the path's own direction
            at the span's middle, ``dv`` OUT of the arc's great-circle plane. ``dv`` is the
            bend -- the control moves the span's middle HALF as far as it moves itself, and
            only along this axis -- while ``du`` is nearly flat in the objective and
            re-samples the same great circle; see :func:`_quad_basis`. A skipped span
            reports ``(0.0, 0.0)``. A cubic rung extends each pair to four numbers, which is
            why this is a tuple of tuples and not a flat list of lengths.
        span_statuses: Per span, in span order: ``"fitted"``, ``"rejected_gain"``,
            ``"rejected_offroute"`` or ``"degenerate"``.
        status: ``"ok"`` (every free span fitted), ``"mixed"`` (some fitted, some not),
            ``"rejected_gain"`` (none fitted and every refusal was a gain refusal),
            ``"pinned"`` (no free span: the whole arc is centre, region, or crossing
            ground), ``"degenerate"`` (no span had a usable transverse frame),
            ``"no_field"`` (unusable field inputs, including a field of zero span, where
            the integral ranks nothing).
    """

    path: tuple[tuple[float, float, float], ...]
    changed: bool = False
    n_spans: int = 0
    n_fitted: int = 0
    n_rejected: int = 0
    n_nodes: int = 0
    n_pinned: int = 0
    max_offset_pitch: float = 0.0
    min_gain_fraction: float = 0.0
    mean_gain_fraction: float = 0.0
    mean_f_before: float = 0.0
    mean_f_after: float = 0.0
    field_span: float = 0.0
    control_offsets: tuple[tuple[float, float], ...] = ()
    span_statuses: tuple[str, ...] = ()
    status: str = _QUAD_ARC_OK


@dataclass(frozen=True)
class _QuadSpanFit:
    """One span's fit: the answer, the gates that decided it, the nodes it owns.

    Private to this module -- the caller assembles spans into a chain and publishes a
    :class:`QuadBezierFit` -- but a dataclass rather than a tuple because it carries ten
    values and a mis-ordered one is silent.

    Attributes:
        index: The span's position in the arc's span list, i.e. its index in
            :attr:`QuadBezierFit.control_offsets`.
        accepted: True when the bend passed every gate.
        status: One of the ``_QUAD_SPAN_*`` values.
        du_pitch: Accepted control offset along the in-plane normal, in mesh pitches.
        dv_pitch: The same along the plane's normal, in pitches.
        gain_fraction: ``(J_after - J_before) / field_span`` at the chosen control.
        integral_before: The arc-length-weighted mean field of the straight span, in
            field units.
        integral_after: The same along the fitted curve; equal to :attr:`integral_before`
            when the span was not fitted.
        length: The span's arc length in radians, the weight it carries in the arc-level
            :attr:`QuadBezierFit.mean_f_before` average.
        node_positions: The fitted positions of this span's interior nodes, in path
            order. Empty when the span was not fitted.
    """

    index: int
    accepted: bool = False
    status: str = _QUAD_SPAN_GAIN
    du_pitch: float = 0.0
    dv_pitch: float = 0.0
    gain_fraction: float = 0.0
    integral_before: float = 0.0
    integral_after: float = 0.0
    length: float = 0.0
    node_positions: tuple[npt.NDArray[np.float64], ...] = ()


@dataclass(frozen=True)
class _QuadSpanNodes:
    """The nodes one span owns, with the curve parameter each one stands on.

    Attributes:
        lo: Index of the span's start vertex in the geodesic (a pin).
        hi: Index of the span's end vertex in the geodesic (a pin).
        length: The span's arc length in radians.
        fracs: Each node's fraction of the span's arc length, ascending. For a great
            circle this IS the curve parameter, because slerp runs at constant angular
            speed, so the straight node at frac 0.4 lands on ``P(0.4)`` of the fitted
            curve: the fit deforms a span's interior without reparameterising it.
        slots: Each node's index in :attr:`_QuadLayout.chain`, so a fitted position can
            be written back into the chain it came from. The chain mixes pins and nodes in
            path order, and a span must address ITS nodes and no one else's.
    """

    lo: int
    hi: int
    length: float
    fracs: tuple[float, ...] = ()
    slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class _QuadLayout:
    """A geodesic's chain as the quadratic fit needs it: pins, and nodes by span.

    The twin of what :func:`_node_layout` returns, regrouped. The flat (positions,
    tangents, pinned) triple is right for gradify, which moves each node independently,
    and unusable for a fit, which optimises ONE control per span and so must know which
    nodes a span owns and where each sits in that span's parameter range. The allocation,
    the pin rule and the drop rule are :func:`_node_layout`'s own -- the same
    largest-remainder quota, the same ``_owned_vertex``/``v_pinned`` test -- so both
    polish modes measure and move the identical set of positions on the identical arc,
    which is what makes the card's two summaries comparable;
    ``test_quad_nodes_are_the_gradify_nodes`` is the guard against the two drifting.

    Attributes:
        chain: ``(position_unit, is_pin)`` for every pin and every placed node, in path
            order: the chain fitted nodes are substituted into and the emission walks.
        spans: One :class:`_QuadSpanNodes` per span, in span order.
        n_nodes: Placed free nodes, ``sum(len(s.fracs) for s in spans)``.
    """

    chain: tuple[tuple[npt.NDArray[np.float64], bool], ...] = ()
    spans: tuple[_QuadSpanNodes, ...] = ()
    n_nodes: int = 0


def _trapezoid_weights(segments: Sequence[float]) -> list[float]:
    """Arc-length weights of a polyline's vertices, for a trapezoid line integral.

    ``w[0] = s[0]/2``, ``w[k] = (s[k-1] + s[k])/2``, ``w[-1] = s[-1]/2``: the length of
    path each vertex is responsible for, so ``sum(w)`` is the total length -- the
    denominator that makes :func:`span_field_integral` a MEAN. That is load-bearing and
    not cosmetics: bending lengthens the path, so an unnormalised integral rewards a
    longer curve and not a ridge.

    Args:
        segments: The polyline's segment lengths (radians on the unit sphere).

    Returns:
        One weight per vertex (``len(segments) + 1``); empty for an empty polyline.
    """
    n = len(segments)
    if n == 0:
        return []
    if n == 1:
        return [segments[0] / 2.0, segments[0] / 2.0]
    weights = [segments[0] / 2.0]
    for k in range(1, n):
        weights.append((segments[k - 1] + segments[k]) / 2.0)
    weights.append(segments[n - 1] / 2.0)
    return weights


def span_field_integral(
    points_unit: Sequence[npt.NDArray[np.float64]],
    f_at_points: Sequence[float],
) -> float:
    """Arc-length-weighted mean of a field along a polyline (a trapezoid line integral).

    ``J = sum(f_k w_k) / sum(w_k)`` over the vertices, with the weights from
    :func:`_trapezoid_weights`. This is the objective :func:`fit_quad_bezier` maximises
    and the whole of the quadratic rung's criterion; the section header records why the
    integral of the function, and not a gradient-alignment sum, is the right thing to
    optimise on this field.

    The trapezoid rule is the honest quadrature for a polyline -- the field is known at
    the vertices, the path between them is a geodesic -- and it is exact for a linear
    integrand, so each endpoint carries half a segment.

    Args:
        points_unit: (K, 3) centred unit radials, in path order.
        f_at_points: (K,) field values at those positions, polarity-signed.

    Returns:
        The mean field along the path, in field units. 0.0 for fewer than two vertices,
        and the plain mean for a zero-length path: coincident vertices rank nothing, and
        a NaN here would poison an arc's whole set of statistics.
    """
    k = len(points_unit)
    if k == 0:
        return 0.0
    values = np.asarray(f_at_points, dtype=np.float64)
    if k == 1:
        return float(values[0])
    segments = [
        angular_distance(points_unit[i], points_unit[i + 1]) for i in range(k - 1)
    ]
    weights = np.asarray(_trapezoid_weights(segments), dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.dot(values, weights) / total)


def _quad_curve_batch(
    geodesic_pts: npt.NDArray[np.float64],
    control_units: npt.NDArray[np.float64],
    ts: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Evaluate a spherical quadratic for MANY controls at once.

    The batched form of :func:`quad_bezier_points`, and the reason the search is
    affordable: the seed scan is 49 controls x ~12 nodes, which is ONE field query of
    ~588 points rather than 588 queries of one point.

    ``squad(a, C, C, b, t)`` collapses, because the control geodesic ``slerp(C, C, t)``
    is ``C`` at every ``t``: the curve is the great circle from ``a`` to ``b`` blended
    toward ``C`` by the quadratic ``2t(1-t)``, which vanishes at both ends. The geodesic
    samples are therefore shared by every candidate and computed once, and each candidate
    costs one slerp toward its own control.

    Args:
        geodesic_pts: (K, 3) unit radials of the span's geodesic at ``ts``; row 0 is the
            span's start pin and row K-1 its end pin.
        control_units: (S, 3) candidate control radials, or one (3,) radial.
        ts: (K,) curve parameters, ascending, with ``ts[0] == 0`` and ``ts[-1] == 1``.

    Returns:
        (S, K, 3) unit radials. A candidate whose control coincides with, or is antipodal
        to, a geodesic sample returns that sample at the node.
    """
    # A single control may arrive as a (3,) radial; the batch arithmetic below needs
    # the (S, 3) form, and normalising here means no caller has to remember it.
    control_units = np.atleast_2d(np.asarray(control_units, dtype=np.float64))
    blend = np.asarray([2.0 * t * (1.0 - t) for t in ts], dtype=np.float64)[:, None]
    cos_om = np.clip(geodesic_pts @ control_units.T, -1.0, 1.0)  # (K, S)
    omega = np.arccos(cos_om)
    sin_om = np.sin(omega)
    flat = sin_om < _QUAD_DEGENERATE_SIN
    safe = np.where(flat, 1.0, sin_om)
    f_c = np.where(flat, 0.0, np.sin(blend * omega) / safe)
    f_g = np.where(flat, 1.0, np.sin((1.0 - blend) * omega) / safe)
    pts = (
        f_g.T[:, :, None] * geodesic_pts[None, :, :]
        + f_c.T[:, :, None] * control_units[:, None, :]
    )
    norms = np.linalg.norm(pts, axis=2, keepdims=True)
    good = (norms[:, :, 0] > _QUAD_DEGENERATE_SIN)[:, :, None]
    scaled = pts / np.where(good, norms, 1.0)
    return np.where(good, scaled, geodesic_pts[None, :, :])


def quad_bezier_points(
    a_unit: npt.NDArray[np.float64],
    b_unit: npt.NDArray[np.float64],
    control_unit: npt.NDArray[np.float64],
    ts: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Points of the spherical quadratic from ``a`` to ``b`` with control ``C``.

    The single-control SQUAD -- :func:`sphere_frame.squad_point` at ``qa = qb = C`` --
    i.e. the sphere-surface-mapped quadratic Bézier the operator's method 3 asks for at
    its lowest order. Both endpoints are exact BY CONSTRUCTION, because the ``2t(1-t)``
    blend vanishes at t=0 and t=1: no post-hoc clamping, no 2-ULP weld to argue with, and
    a basin ring built on these paths closes on the same CP centres as a ring built on
    the straight arcs (which is what gradify's cycle test checks, and what made that
    endpoint promise a test and not an assumption).

    Args:
        a_unit: Start unit radial (the span's saddle-side pin).
        b_unit: End unit radial (the span's CP-side pin).
        control_unit: The control point's unit radial.
        ts: Parameters to evaluate, ascending, in [0, 1].

    Returns:
        (len(ts), 3) unit radials. Rows where the interpolation degenerates fall back to
        the geodesic (or, when the geodesic itself is undefined, to the nearer endpoint),
        so a caller always gets a usable polyline.
    """
    geodesic = np.zeros((len(ts), 3), dtype=np.float64)
    for i, t in enumerate(ts):
        geodesic[i] = _slerp(a_unit, b_unit, float(t))
    return np.asarray(
        _quad_curve_batch(geodesic, np.atleast_2d(control_unit), ts)[0],
        dtype=np.float64,
    )


def _quad_basis(
    a_unit: npt.NDArray[np.float64],
    b_unit: npt.NDArray[np.float64],
) -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | None
):
    """A span's control frame: ``(C0, e_along, e_across)``, ON-TANGENT at ``C0``.

    ``C0`` is the midpoint of the geodesic -- the chord's own centre, and the control
    whose curve is the great circle itself, which is what makes the straight span the
    search's origin. The gate measures an improvement over the arc we already have, so the
    search starts on it and every accepted bend is a measured gain over it.

    The basis is built AT ``C0``, orthonormal in its tangent plane, and the two axes are
    NOT interchangeable. Measured on a 9.9-pitch arc at 3 pitches of latitude:

    - ``e_across = unit(a x b)``, the plane's normal, is THE BENDING AXIS. A control moved
      2 pitches along it pulls the span's middle 1.07 pitches off the great circle, toward
      a crest the plane does not contain -- the move a great circle can never make, which
      is the reason the ladder exists.
    - ``e_along``, the path's own direction at ``C0``, is the WEAK axis. A control displaced
      along it keeps the curve ON the great circle and only redistributes the samples along
      it (the blend pulls them toward the span's middle), so it changes the sampled
      integral by the field's ALONG-path variation and not by any bend. The objective is
      nearly flat along it, which the isotropic seed grid exposes honestly: the polish stops
      where the flatness starts, and an accepted offset is mostly ``e_across``.

    Building the frame at ``a`` instead -- the obvious choice, and what this function first
    did -- skews the chart: ``unit(b - (a.b)a)`` is tangent at ``a``, and its component along
    ``C0`` is sin(theta/2), 0.26 for a 10-pitch span and 0.71 for a 40-pitch one, so a
    requested ``(du, dv)`` is not the offset the control receives and the published
    ``control_offsets`` stop being a measurement. A basis vector must be orthogonal to the
    point it translates, which is why the frame is constructed at the point it moves.

    Args:
        a_unit: Start unit radial.
        b_unit: End unit radial.

    Returns:
        ``(C0, e_along, e_across)``, or None when the pair is (near) parallel or antipodal
        and the span has no transverse frame to search in.
    """
    dot_ab = float(np.dot(a_unit, b_unit))
    if dot_ab > 1.0 - _QUAD_DEGENERATE_BASIS:
        return None  # coincident: the span has no length to bend
    if dot_ab < -1.0 + _QUAD_DEGENERATE_BASIS:
        return None  # antipodal: no unique great circle, so no plane and no bend axis
    normal = np.cross(a_unit, b_unit)
    norm = float(np.linalg.norm(normal))
    if norm < _QUAD_DEGENERATE_BASIS:
        return None
    c0 = unit(a_unit + b_unit)
    if c0 is None:  # pragma: no cover - the antipodal pair returned above
        return None
    e_across = normal / norm
    e_along = unit(np.cross(c0, e_across))
    if (
        e_along is None
    ):  # pragma: no cover - c0 lies in the plane, so the cross is nonzero
        return None
    return c0, e_along, e_across


def _quad_control(
    c0: npt.NDArray[np.float64],
    e_along: npt.NDArray[np.float64],
    e_across: npt.NDArray[np.float64],
    du: float,
    dv: float,
) -> npt.NDArray[np.float64]:
    """The control point at offset ``(du, dv)`` radians from a frame's origin.

    A great-circle step of ``hypot(du, dv)`` along ``(du*e_along + dv*e_across)/|..|``.
    Because :func:`_quad_basis` returns an ON-TANGENT basis at ``C0``, that direction is a
    tangent vector at ``C0`` and the step is exact: the control's angular distance from
    ``C0`` IS ``hypot(du, dv)``, so the pitch-unit caps of :data:`QUAD_FIT_SEARCH_PITCH`
    mean the same thing in every direction, the search is isotropic by construction, and
    the published offset is the offset the control received.
    """
    distance = math.hypot(du, dv)
    if distance <= 0.0:
        return c0
    direction = (du * e_along + dv * e_across) / distance
    moved = _offset_unit(c0, direction, distance)
    return c0 if moved is None else moved


def _quad_clamp(
    du: float,
    dv: float,
    bound: float,
) -> tuple[float, float]:
    """Project an offset onto the admissible disc of radius ``bound`` radians."""
    distance = math.hypot(du, dv)
    if bound <= 0.0 or distance <= bound:
        return du, dv
    scale = bound / distance
    return du * scale, dv * scale


def _quad_nelder_mead(
    cost: Callable[[float, float], float],
    u0: float,
    v0: float,
    step: float,
    max_evals: int,
    bound: float,
) -> tuple[float, float, float]:
    """Minimise ``cost`` over a disc with a 2D Nelder-Mead simplex.

    Derivative-free, dependency-free, and the right optimiser here for three reasons: the
    objective needs no gradient (a gradient of this field is exactly what the stencil
    exists to avoid), a 2-simplex cannot jump a ridge the seed scan rejected, and it
    degrades gracefully -- on a flat plateau it stops where it was dropped, which is
    gate 4's whole mitigation.

    Args:
        cost: Objective, minimised, called with ``(du, dv)`` in radians.
        u0: Start offset along ``e1``, radians.
        v0: Start offset along ``e2``, radians.
        step: Initial simplex edge, radians.
        max_evals: Budget of ``cost`` evaluations (the initial triangle is free).
        bound: Radius of the admissible disc, radians. Every candidate is projected onto
            it before evaluation, so the search cannot spend evaluations outside it.

    Returns:
        ``(du, dv, cost)`` of the best vertex found.
    """
    vertices = [
        _quad_clamp(u0, v0, bound),
        _quad_clamp(u0 + step, v0, bound),
        _quad_clamp(u0, v0 + step, bound),
    ]
    values = [cost(*v) for v in vertices]
    evals = 0
    while evals < max_evals:
        order = sorted(range(3), key=lambda i: values[i])
        vertices = [vertices[i] for i in order]
        values = [values[i] for i in order]
        best, mid, worst = vertices[0], vertices[1], vertices[2]
        # Convergence: a simplex whose spread is at the float noise floor of its own value
        # cannot resolve another step, and the remaining iterations are arithmetic on
        # noise.
        if abs(values[2] - values[0]) <= 1e-14 * max(1.0, abs(values[0])):
            break
        centroid = ((best[0] + mid[0]) / 2.0, (best[1] + mid[1]) / 2.0)
        reflected = _quad_clamp(
            centroid[0] + _QUAD_NM_MIRROR * (centroid[0] - worst[0]),
            centroid[1] + _QUAD_NM_MIRROR * (centroid[1] - worst[1]),
            bound,
        )
        f_reflected = cost(*reflected)
        evals += 1
        if f_reflected < values[0]:
            expanded = _quad_clamp(
                centroid[0] + _QUAD_NM_EXPAND * (centroid[0] - worst[0]),
                centroid[1] + _QUAD_NM_EXPAND * (centroid[1] - worst[1]),
                bound,
            )
            f_expanded = cost(*expanded)
            evals += 1
            vertices[2], values[2] = (
                (expanded, f_expanded)
                if f_expanded < f_reflected
                else (reflected, f_reflected)
            )
        elif f_reflected < values[2]:
            # Accept the reflection, whether it landed in the top pair (the ordinary
            # case) or only beat the worst: both replace the worst vertex.
            vertices[2], values[2] = reflected, f_reflected
        else:
            contracted = _quad_clamp(
                centroid[0] + _QUAD_NM_SHRINK * (worst[0] - centroid[0]),
                centroid[1] + _QUAD_NM_SHRINK * (worst[1] - centroid[1]),
                bound,
            )
            f_contracted = cost(*contracted)
            evals += 1
            if f_contracted < values[2]:
                vertices[2], values[2] = contracted, f_contracted
                continue
            # Shrink: pull every vertex halfway to the best. The only branch that costs
            # more than one evaluation, and the one that makes the simplex converge.
            for i in (1, 2):
                point = _quad_clamp(
                    best[0] + _QUAD_NM_SHRINK * (vertices[i][0] - best[0]),
                    best[1] + _QUAD_NM_SHRINK * (vertices[i][1] - best[1]),
                    bound,
                )
                vertices[i] = point
                values[i] = cost(*point)
                evals += 1
    best_index = int(np.argmin(values))
    return vertices[best_index][0], vertices[best_index][1], float(values[best_index])


def _quad_scan_indices(n_nodes: int) -> list[int]:
    """The sample rows the seed scan uses: the endpoints, and every ``stride``-th node."""
    total = n_nodes + 2  # + the span's two pins
    stride = max(1, int(QUAD_FIT_SCAN_STRIDE))
    chosen = {0, total - 1}
    chosen.update(range(1, total - 1, stride))
    return sorted(chosen)


def _quad_node_layout(
    geodesic: npt.NDArray[np.float64],
    cum: Sequence[float],
    spans: Sequence[tuple[int, int]],
    pins: Sequence[int],
    v_pinned: Sequence[bool],
    n_nodes: int,
) -> _QuadLayout:
    """Lay ``n_nodes`` free nodes on the measurable ground, grouped by span.

    The allocation, the pin rule and the drop rule are :func:`_node_layout`'s, verbatim:
    largest-remainder over the spans' arc lengths, nodes placed by ARC LENGTH (not on the
    geodesic's own vertices, which are a tessellation choice), and a node whose owning
    segment begins at a pinned vertex DROPPED. What it adds is the per-span grouping, the
    ``frac`` each node stands on, and the chain slot each node occupies. It is written
    beside its twin rather than derived from it, because the grouping changes the return
    shape -- and the guard test compares the two node SETS position by position, so the
    pair cannot drift apart silently.

    Args:
        geodesic: (M, 3) centred unit radials of the great circle.
        cum: Cumulative arc length per vertex, from :func:`_polyline_length`.
        spans: ``(lo, hi)`` pin pairs with room for nodes, from :func:`_pinned_spans`.
        pins: The pinned vertex indices, ascending.
        v_pinned: Per-vertex pin mask, from :func:`_pinned_spans`.
        n_nodes: Free nodes to place in total.

    Returns:
        A :class:`_QuadLayout`.
    """
    length_of = [float(cum[hi] - cum[lo]) for lo, hi in spans]
    free_length = math.fsum(length_of)
    quota = (
        [n_nodes * length / free_length for length in length_of]
        if free_length > 0.0
        else [0.0] * len(spans)
    )
    counts = [int(q) for q in quota]
    short = n_nodes - sum(counts)
    order = sorted(
        range(len(spans)),
        key=lambda s: (-round(quota[s] - counts[s], 12), -spans[s][1], spans[s][0]),
    )
    for s in order[: max(0, short)]:
        counts[s] += 1

    # (arc length, tie, position, is_pin, span, frac): pins sort first at a shared
    # length, exactly as :func:`_node_layout` orders them, so the two chains are the same
    # sequence. The span and frac travel with the node so the grouping below reads the
    # SORTED chain and needs no identity key: slot = the entry's index in the chain.
    entries: list[tuple[float, int, npt.NDArray[np.float64], bool, int, float]] = [
        (float(cum[index]), 0, geodesic[index], True, -1, 0.0) for index in pins
    ]
    for si, (lo, hi) in enumerate(spans):
        for j in range(counts[si]):
            frac = (j + 1) / (counts[si] + 1)
            node = _interp_node(geodesic, cum, lo, hi, frac)
            if node is None:  # pragma: no cover - a span has positive length
                continue
            length = float(cum[lo] + frac * (cum[hi] - cum[lo]))
            if v_pinned[_owned_vertex(cum, length, lo, hi)]:
                continue
            entries.append((length, 1, node[0], False, si, frac))

    entries.sort(key=lambda e: (e[0], e[1]))
    grouped: list[list[tuple[float, int]]] = [[] for _ in spans]
    for slot, entry in enumerate(entries):
        if not entry[3]:
            grouped[entry[4]].append((entry[5], slot))

    span_nodes = [
        _QuadSpanNodes(
            lo=spans[si][0],
            hi=spans[si][1],
            length=length_of[si],
            fracs=tuple(f for f, _s in grouped[si]),
            slots=tuple(s for _f, s in grouped[si]),
        )
        for si in range(len(spans))
    ]
    return _QuadLayout(
        chain=tuple((e[2], e[3]) for e in entries),
        spans=tuple(span_nodes),
        n_nodes=sum(len(s.fracs) for s in span_nodes),
    )


def _fit_quad_span(
    index: int,
    span: _QuadSpanNodes,
    geodesic: npt.NDArray[np.float64],
    mesh_pts: npt.NDArray[np.float64],
    field: npt.NDArray[np.float64],
    sphere_radius: float,
    pitch: float,
    find_k_nearest: KNearestFinder | None,
    offroute: Sequence[tuple[npt.NDArray[np.float64], float]],
    search_pitch: float,
    max_control_pitch: float,
    min_gain_fraction: float,
    field_span: float,
) -> _QuadSpanFit:
    """Place one span's control point, and decide whether the bend is worth it.

    The whole per-span pipeline: frame, straight baseline, seeded scan, Nelder-Mead
    polish, gates. Split out of :func:`fit_quad_bezier` so a span is testable in
    isolation, which is the module's rule that every geometric step is a single-purpose
    routine.

    Args:
        index: The span's index in the arc.
        span: Its nodes, from :func:`_quad_node_layout`.
        geodesic: (M, 3) unit radials of the arc's great circle.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        field: (N,) the polarity-signed field.
        sphere_radius: Atom sphere radius R.
        pitch: The mesh pitch, radians.
        find_k_nearest: Spatial index over ``mesh_pts``.
        offroute: ``(centre, radius)`` of every CP disc the straight arc never entered.
        search_pitch: Control search radius in pitches -- how far the polish LOOKS.
        max_control_pitch: Largest accepted offset, in pitches -- how far it is ALLOWED
            to move the arc. The two are independent on purpose: a search wider than the
            cap finds its best at the disc's edge and is then REFUSED by the cap, which
            is the gate reporting "there is a crest out there, further than this arc may
            bend for it" instead of silently spending the wider search.
        min_gain_fraction: Smallest accepted gain, in units of ``field_span``.
        field_span: The robust span the gain is normalised by.

    Returns:
        A :class:`_QuadSpanFit`.
    """
    a_unit = geodesic[span.lo]
    b_unit = geodesic[span.hi]
    basis = _quad_basis(a_unit, b_unit)
    if basis is None:
        return _QuadSpanFit(
            index=index, status=_QUAD_SPAN_DEGENERATE, length=span.length
        )
    c0, e_along, e_across = basis
    if not span.fracs:
        # No node, nothing to measure: the only samples are the two pins, and a pin's
        # field value does not depend on the control, so every candidate has the same
        # integral and the gain is exactly zero. Skipped, not searched.
        return _QuadSpanFit(
            index=index, status=_QUAD_SPAN_DEGENERATE, length=span.length
        )

    ts = (0.0,) + tuple(span.fracs) + (1.0,)
    # The straight samples are the geodesic at the node fracs -- i.e. exactly where
    # :func:`_quad_node_layout` placed the nodes, rebuilt here from the same slerp so the
    # fit's parameter grid and the layout's arc lengths agree by construction.
    geodesic_pts = np.zeros((len(ts), 3), dtype=np.float64)
    geodesic_pts[0] = a_unit
    geodesic_pts[-1] = b_unit
    for i, frac in enumerate(span.fracs):
        geodesic_pts[i + 1] = _slerp(a_unit, b_unit, float(frac))
    scan_rows = _quad_scan_indices(len(span.fracs))
    scan_pts = geodesic_pts[scan_rows]
    scan_ts = [ts[i] for i in scan_rows]

    def integrals(
        controls: npt.NDArray[np.float64],
        samples: npt.NDArray[np.float64],
        params: Sequence[float],
    ) -> list[float]:
        """Mean field along one curve per control, at ``params``. ONE field query."""
        pts = _quad_curve_batch(samples, controls, params)
        k = len(params)
        values = _sample_field(
            list(pts.reshape(-1, 3)), sphere_radius, mesh_pts, field, find_k_nearest
        )
        return [
            span_field_integral(list(pts[s]), values[s * k : (s + 1) * k])
            for s in range(pts.shape[0])
        ]

    # --- the straight baseline: the arc we already have, at the node grid ---------
    #
    # Measured on the STRAIGHT samples and not on the (0, 0) candidate, because the
    # quadratic's parameterisation is compressed toward the middle even when the control
    # sits on the geodesic (the curve is the same set, traversed monotonically at
    # 2 - 6t(1-t) of the geodesic's angular speed), and the gate's job is to compare a
    # bend against the connector the record carries today.
    baseline_values = _sample_field(
        list(geodesic_pts), sphere_radius, mesh_pts, field, find_k_nearest
    )
    baseline = span_field_integral(list(geodesic_pts), baseline_values)

    # --- the seeds: one batch, then argmax ---------------------------------------
    seeds: list[tuple[float, float]] = [(0.0, 0.0)]
    total_offset = pitch * max(search_pitch, 1e-9)
    for radius in QUAD_FIT_SEED_RADII:
        r = total_offset * radius
        for spoke in range(QUAD_FIT_SEED_ANGLES):
            angle = 2.0 * math.pi * spoke / QUAD_FIT_SEED_ANGLES
            seeds.append((r * math.cos(angle), r * math.sin(angle)))
    seed_controls = np.asarray(
        [_quad_control(c0, e_along, e_across, du, dv) for du, dv in seeds],
        dtype=np.float64,
    )
    seed_values = integrals(seed_controls, scan_pts, scan_ts)
    du, dv = seeds[int(np.argmax(seed_values))]

    # --- the polish: the FULL node grid, so the measured gain is the published one
    def cost(du_: float, dv_: float) -> float:
        control = np.atleast_2d(_quad_control(c0, e_along, e_across, du_, dv_))
        return -integrals(control, geodesic_pts, ts)[0]

    du, dv, _ = _quad_nelder_mead(
        cost,
        du,
        dv,
        pitch * QUAD_FIT_NM_STEP_PITCH,
        QUAD_FIT_NM_MAX_ITERS,
        total_offset,
    )
    offset = math.hypot(du, dv)
    control = _quad_control(c0, e_along, e_across, du, dv)
    curve = _quad_curve_batch(geodesic_pts, np.atleast_2d(control), ts)[0]
    values = _sample_field(list(curve), sphere_radius, mesh_pts, field, find_k_nearest)
    fitted = span_field_integral(list(curve), values)
    gain = (fitted - baseline) / field_span
    # The CAP, not the search: `max(...)` here would let a wide search launder an
    # offset the policy forbids, which is the failure this line exists to catch.
    accept_offset = pitch * max(max_control_pitch, 0.0)

    def refusal(status: str) -> _QuadSpanFit:
        return _QuadSpanFit(
            index=index,
            status=status,
            gain_fraction=gain,
            integral_before=baseline,
            integral_after=fitted,
            length=span.length,
        )

    if offset > accept_offset + 1e-15:
        # Gate 1's cap. Reported as a gain refusal is wrong -- the bend was bigger than
        # the policy allows, which is a reroute -- so it gets its own name only when the
        # gain itself cleared the bar.
        return refusal(
            _QUAD_SPAN_OFFROUTE if gain >= min_gain_fraction else _QUAD_SPAN_GAIN
        )
    if gain < min_gain_fraction:
        return refusal(_QUAD_SPAN_GAIN)
    # Gate 1, on the ANSWER and not on every seed (which would make the search's cost
    # proportional to the number of discs): a bend that carries a node into a disc the
    # straight arc never entered is a reroute to a CP this arc does not connect to.
    for centre, radius in offroute:
        limit = radius * (1.0 + QUAD_FIT_OFFROUTE_MARGIN)
        if any(
            angular_distance(curve[i], centre) <= limit for i in range(1, len(ts) - 1)
        ):
            return refusal(_QUAD_SPAN_OFFROUTE)
    return _QuadSpanFit(
        index=index,
        accepted=True,
        status=_QUAD_SPAN_FITTED,
        du_pitch=du / pitch,
        dv_pitch=dv / pitch,
        gain_fraction=gain,
        integral_before=baseline,
        integral_after=fitted,
        length=span.length,
        node_positions=tuple(curve[i] for i in range(1, len(ts) - 1)),
    )


def fit_quad_bezier(
    arc_points: Sequence[Sequence[float]],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    regions: Sequence[GradifyRegion] = (),
    n_nodes: float = ARC_GRADIFY_NODES,
    search_pitch: float = QUAD_FIT_SEARCH_PITCH,
    max_control_pitch: float = QUAD_FIT_MAX_CONTROL_PITCH,
    min_gain_fraction: float = QUAD_FIT_MIN_GAIN_FRACTION,
    output_pitch: float = GRADIFY_OUTPUT_PITCH,
    span_override: float | None = None,
    polarity: GradifyPolarity = "maximum",
) -> QuadBezierFit:
    """Fit a geodesic connector to the spherical quadratic the field integrates best.

    Method 3's first rung, in the operator's words: "the existing arc ... a sphere surface
    mapped polynomial. First as a quadratic which has a single control point that can be
    adjusted". One control point per FREE SPAN (the arc's ground between two pins), placed
    by a seeded grid scan and a Nelder-Mead polish to maximise :func:`span_field_integral`
    -- the arc-length-weighted mean of the polarity-signed field along the span -- and
    accepted per span only when the improvement clears :data:`QUAD_FIT_MIN_GAIN_FRACTION`
    of the field's robust span, stays inside :data:`QUAD_FIT_MAX_CONTROL_PITCH`, and keeps
    the path out of every CP disc the straight arc never entered.

    What is pinned is :func:`gradify_arc`'s own list, by the same helper, for the same
    reasons: the two CP centres (an arc runs centre to centre, so both ends are always
    region ground, and the basin boundary's joints depend on them being bit-stable), every
    vertex inside an excision region (whose field is the condensation's artifact and not
    the ground's), and each region's entry/exit crossing with the vertex on each side --
    the traced connector's bridge rule, centre to boundary and boundary to boundary
    straight, in geometric form. CP-cluster handling is therefore not a special case added
    here: it is the rule the traced connectors and gradify already obey.

    Args:
        arc_points: (M, 3) connector polyline, WORLD frame, saddle centre first.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on ``mesh_pts``.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus. None when the positions are already centred.
        find_k_nearest: Spatial index over ``mesh_pts``; strongly recommended.
        regions: The CP excision regions. Their discs pin ground (gate 2) and define the
            off-route gate (gate 1); an empty sequence disables both, which is what a
            caller with no excision data should pass.
        n_nodes: Free nodes per arc; see :data:`ARC_GRADIFY_NODES`. Shared with
            :func:`gradify_arc` on purpose, so the card's two polish modes describe the
            same sampling of the same arc.
        search_pitch: Control search radius in mesh pitches; see
            :data:`QUAD_FIT_SEARCH_PITCH`.
        max_control_pitch: Largest accepted offset, in pitches.
        min_gain_fraction: Smallest accepted improvement, as a fraction of the field's
            robust span.
        output_pitch: Vertex spacing of the emitted polyline, in mesh pitches.
        span_override: Field span for the gain gate. None measures the robust span once
            for the whole field (:func:`_robust_field_span`), which is what the geometry
            pipeline wants; pass an explicit value only to reproduce a caller's own
            normalisation.
        polarity: ``"maximum"`` for a connector climbing to a maximum (maximise the
            integral: the ridge), ``"minimum"`` for one descending to a minimum (negate
            once, maximise: the valley floor). See :func:`polarity_field`.

    Returns:
        A :class:`QuadBezierFit`. Degenerate input (fewer than :data:`_MIN_PATH`
        vertices, a non-positive radius/pitch, an unusable field, a field of zero span)
        returns the path unchanged with a status naming why, so a caller can hand this
        every arc it has without testing first.
    """
    m = len(arc_points)
    straight = tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc_points)
    if (
        m < _MIN_PATH
        or sphere_radius <= 0.0
        or delta_theta_mesh_rad <= 0.0
        or len(mesh_pts) == 0
        or len(f_vals) == 0
    ):
        return QuadBezierFit(
            path=straight,
            status=_QUAD_ARC_DEGENERATE if m < _MIN_PATH else _QUAD_ARC_NO_FIELD,
        )

    geodesic = _path_units(arc_points, center)
    cum = _polyline_length(geodesic)
    pins, spans, v_pinned = _pinned_spans(geodesic, regions)
    if not spans:
        return QuadBezierFit(path=straight, n_pinned=len(pins), status=_QUAD_ARC_PINNED)

    pitch = delta_theta_mesh_rad
    # The descent is the ascent's own rule on the negated field, and the integral is
    # polarity-signed for free: maximising the negated field downhill IS minimising the
    # field itself. Negating ONCE here keeps a single implementation, so the ascent and
    # the descent cannot drift apart.
    field = polarity_field(f_vals, polarity)
    span = _robust_field_span(field) if span_override is None else float(span_override)
    if span <= 0.0:
        # A constant field: the integral ranks nothing and the gate's denominator is 0.
        # Refuse the arc rather than divide by it -- see gate 3.
        return QuadBezierFit(
            path=straight,
            n_spans=len(spans),
            n_rejected=len(spans),
            n_pinned=len(pins),
            status=_QUAD_ARC_NO_FIELD,
        )

    layout = _quad_node_layout(
        geodesic, cum, spans, pins, v_pinned, max(1, round(float(n_nodes)))
    )

    # Every CP disc the straight arc never entered. A region it DOES touch is fair
    # ground: its crossings are pinned and the transit between them stays straight.
    offroute: list[tuple[npt.NDArray[np.float64], float]] = []
    for region in regions:
        centre = np.asarray(region.centre_unit, dtype=np.float64)
        radius = float(region.radius_rad)
        touched = any(angular_distance(geodesic[i], centre) <= radius for i in range(m))
        if not touched:
            offroute.append((centre, radius))

    fits = [
        _fit_quad_span(
            index=si,
            span=node_span,
            geodesic=geodesic,
            mesh_pts=mesh_pts,
            field=field,
            sphere_radius=sphere_radius,
            pitch=pitch,
            find_k_nearest=find_k_nearest,
            offroute=offroute,
            search_pitch=search_pitch,
            max_control_pitch=max_control_pitch,
            min_gain_fraction=min_gain_fraction,
            field_span=span,
        )
        for si, node_span in enumerate(layout.spans)
    ]

    n_fitted = sum(1 for fit in fits if fit.accepted)
    ordered = sorted(fits, key=lambda f: f.index)
    common: dict[str, Any] = {
        "n_spans": len(spans),
        "n_fitted": n_fitted,
        "n_rejected": len(spans) - n_fitted,
        "n_nodes": layout.n_nodes,
        "n_pinned": sum(1 for _pos, is_pin in layout.chain if is_pin),
        "field_span": span,
        "control_offsets": tuple((f.du_pitch, f.dv_pitch) for f in ordered),
        "span_statuses": tuple(f.status for f in ordered),
    }
    if n_fitted == 0:
        if all(f.status == _QUAD_SPAN_DEGENERATE for f in fits):
            status = _QUAD_ARC_DEGENERATE
        elif all(f.status == _QUAD_SPAN_GAIN for f in fits):
            status = _QUAD_ARC_GAIN
        else:
            status = _QUAD_ARC_MIXED
        return QuadBezierFit(path=straight, status=status, **common)

    # --- assemble the chain ----------------------------------------------------
    #
    # Pins verbatim, fitted nodes at their fitted positions, refused nodes on the great
    # circle. The chain is then described at ``output_pitch`` exactly as gradify's is, for
    # the same reason: walking each SEGMENT, never a uniform arc-length walk over the whole
    # line, keeps every pin an emitted vertex, so a region's transit stays the geodesic's
    # own chord and no interpolation puts a vertex inside a disc.
    out_units = [pos for pos, _is_pin in layout.chain]
    for fit in fits:
        if not fit.accepted:
            continue
        node_span = layout.spans[fit.index]
        for slot, position in zip(node_span.slots, fit.node_positions):
            out_units[slot] = position

    fitted_chain = np.asarray(out_units, dtype=np.float64)
    step_rad = max(output_pitch, 1e-9) * pitch
    emitted: list[npt.NDArray[np.float64]] = [fitted_chain[0]]
    for i in range(len(fitted_chain) - 1):
        lo_p, hi_p = fitted_chain[i], fitted_chain[i + 1]
        n_sub = max(1, round(angular_distance(lo_p, hi_p) / step_rad))
        for j in range(1, n_sub + 1):
            emitted.append(_slerp(lo_p, hi_p, j / n_sub))
    emitted[-1] = fitted_chain[-1]  # the CP centre, bit-exact
    if len(emitted) < _MIN_PATH:  # pragma: no cover - a fit has >= _MIN_PATH slots
        emitted = list(fitted_chain)

    accepted_fits = [fit for fit in fits if fit.accepted]
    weights = [max(fit.length, 0.0) for fit in accepted_fits]
    total_weight = math.fsum(weights)
    gains = [fit.gain_fraction for fit in accepted_fits]
    offsets = [math.hypot(fit.du_pitch, fit.dv_pitch) for fit in accepted_fits]

    def weighted(values: list[float]) -> float:
        """Arc-length-weighted mean over the fitted spans (pins are not the fit's)."""
        if not values:
            return 0.0
        if total_weight <= 0.0:
            return float(np.mean(values))
        return float(np.dot(np.asarray(values), np.asarray(weights)) / total_weight)

    return QuadBezierFit(
        path=tuple(_world(u, sphere_radius, center) for u in emitted),
        changed=True,
        max_offset_pitch=float(max(offsets, default=0.0)),
        min_gain_fraction=float(min(gains, default=0.0)),
        mean_gain_fraction=float(np.mean(gains)) if gains else 0.0,
        mean_f_before=weighted([fit.integral_before for fit in accepted_fits]),
        mean_f_after=weighted([fit.integral_after for fit in accepted_fits]),
        status=_QUAD_ARC_OK if n_fitted == len(spans) else _QUAD_ARC_MIXED,
        **common,
    )
