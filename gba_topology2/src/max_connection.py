"""Maximum Connection Subroutine for gba_topology2 (Basin Locator Step 3).

A saddle's ascent (ridge) launch branch is the direction the separatrix leaves
the saddle on its way UP toward a maximum. This subroutine answers the question
that direction poses: *which* maximum is this branch aiming at?

The method: WALK THE RAY, CONE FOR TARGETS, DISC FOR OBSTACLES
--------------------------------------------------------------
The ray is a SOLVER, not the connector. Its job is to find the two endpoints of
a connection -- the saddle cluster centre and the maximum cluster centre -- and
the connector drawn is the single great circle between them. The port, the launch
bearing and the boundary crossing are all internal to the search.

Each branch supplies a ray on the sphere, launched from its PORT on the saddle's
excision boundary along the outward geodesic. Every VISIBLE CRITICAL POINT --
maxima, minima and saddles alike -- is scored by its position relative to that
ray, decomposed in the orthonormal frame ``(o, d, n)`` where ``o`` is the unit
origin, ``d`` the unit direction and ``n = o x d`` the pole of the ray's great
circle:

    along = atan2(t.d, t.o)      progress along the ray to the CP's CENTRE
    cross = asin(t.n)            perpendicular deviation from the ray

The candidates inside the forward sector (``0 < along <= max_arc_pitch *
delta_theta`` and ``|cross| <= cone_pitch * delta_theta``) are walked in
increasing ``along``, and the walk stops at the first CP that is decisive:

- a **maximum** ends the walk and the branch CONNECTS to it. A target is
  captured by the CONE. Where the ray crosses the maximum's excision boundary,
  ``entry = along - acos(cos rho / cos cross)``, is recorded as the ranking key
  (``along_track_deg``) and as how far the ray travels before it reaches the
  basin; it is not a vertex of the drawn connector.
- a **minimum or saddle** ends the walk as an OCCLUSION only if the great circle
  actually enters its excision disc, i.e. ``|cross| <= rho``. Otherwise the ray
  sails past it and the walk continues.

So the disc is the criterion for an OBSTACLE and the cone is the criterion for a
TARGET. Ties break on ``|cross|`` then on id, so the result is a pure function of
the input set and not of its order.

Why the cone captures a maximum, not the disc (measured, user-reported)
----------------------------------------------------------------------
Requiring the ray to enter the MAXIMUM's own boundary is tuned backwards. Every
raw maximum's excision radius is one mesh pitch -- 1.61 deg on Pd_20K -- while
the legitimate rays into a maximum ringed by three saddles arrive at cross-tracks
0.0, 1.35, 1.42 on MAX_7 and 0.0, 2.00, 1.98 on its symmetric twin MAX_4. A gate
at 1.61 sits inside that band, so it accepted all three of MAX_7's connections
and rejected two of MAX_4's: identical geometry, different answers, decided by
noise. The connector counts came out 3, 2, 2, 2, 2, 1, 1, 3 across the eight
symmetric maxima. Capturing by the cone, which is the directional criterion the
sector already encodes, gives all eight maxima exactly 3.

Why the disc blocks an obstacle, not the cone (measured, user-reported)
----------------------------------------------------------------------
Scoring an obstacle's CENTRE occludes on NEAR MISSES, because the cone is
necessarily wider than any single disc. On Pd_20K curvedness, saddle MC18's
branch 0 was blocked from its maximum MC38 by a minimum MIN_19165 at
``along 7.5, cross 7.4`` -- a 1.6 deg object the ray passes 7.4 deg clear of. It
is not an obstacle; the ray sails past it. It was a candidate only because the
9.6 deg cone is six times wider than its disc, and it won because its centre is
nearer than MC38's. Requiring the ray to enter the obstacle's disc takes the
occlusion count on that snapshot from 20 to 0 and MC18 connects to MC38.

Why the disc is an ELLIPSE evaluated toward the ray
---------------------------------------------------
An excision boundary around an elongated set is a fitted ellipse, not a circle.
Using its semi-major as a disc inflates the transverse reach by the aspect ratio:
on Pd_20K shape index the bridged corridor MS13 (a = 11.35, b = 2.43, aspect
10.6) occluded a ray passing 3.58 deg off its axis, because 3.58 < 11.35 while
3.58 > 2.78. :func:`directional_support_radius` evaluates the boundary in the
direction of the ray. This is the same failure the bridging gap test already
fixed for the same reason (``string_aggregator.cluster_directional_support_radius``),
whose docstring names it: an isotropic radius "turns a long thin corridor into a
disk of radius a and lets it swallow compact fragments sitting far off to the
side".

Why the previous rule was wrong (measured, user-reported)
--------------------------------------------------------
The first version took the smallest ``|cross|`` with ``along`` as a tie-break.
That scores AIM and never checks REACH, and it fails in three ways:

1. **Collinear maxima are the degenerate case, structurally.** A ray that
   passes through the maximum MC109 and on to MC143 has ``cross(MC109) = 0``
   and ``cross(MC143) = 0`` -- both lie on the great circle by construction, so
   cross-track is provably unable to separate them. Only ``along`` can, and it
   was the secondary key, consulted only on a bit-exact tie. On a dense field
   that is not rare: across the four saved Pd_20K snapshots the mean cross-track
   of a claim was 0.19-4.4 deg, i.e. many maxima sit essentially ON each other's
   rays, and float noise at 1e-11 rad was deciding a physical question.
2. **The pool was maxima-only**, so the rule could not express "passes through a
   critical point" at all -- it never looked at the critical points that were
   not maxima.
3. **The drawn arc is the saddle->winner geodesic**, so the renderer faithfully
   drew the occlusion the matcher had ignored. That is what made it visible.

Why the sector is a constant, not the branch's own sector width
---------------------------------------------------------------
The obvious per-branch, parameter-free cone is ``LaunchBranch.sector_width_rad``,
the angular gap between the two separatrix crossings bounding the branch's
ascent sector. Measured on the four snapshots it is unusable: on Curvedness 40
of 72 branches report width **0.0 deg** and several report **355-357 deg**; on V
four report 352.6 deg. It is the span between bounding crossings, which
degenerates whenever a sector contains all of them. So the half-angle is the
pitch-relative ``CONE_PITCH`` below, and the number of tunables stays at one.

This is deliberately the SIMPLEST possible connector: a great circle, no field
integration, no barrier test. Step 4 replaces it with the traced separatrix, and
the value of having it early is that the traced path has something to be
compared against -- a geodesic that lands on a different maximum than the
integrator does is a measurement, not a surprise.

The ray origin: the port, not the saddle centre
----------------------------------------------
The excision path (``excision_launch.py``) places ``launch_position`` on the
ring and ``tangent_direction`` at that port, pointing outward from the ring
centre. The saddle centre is the ring centre. Two rays are defensible:

- ``"port_tangent"`` (the DEFAULT): origin = the port, direction =
  ``tangent_direction``, the outward continuation measured AT the ring.
- ``"chord"``: origin = saddle centre, direction = the centre->port bearing.

They are the same great circle -- ``_outward_tangent`` builds the port direction
as ``radial * port - centre``, the parallel transport of the centre->port
geodesic, so both share the pole ``centre x port``. They differ ONLY in where
``along = 0`` sits, which shifts by the ring radius. That shift is the whole
reason the port is the origin: **the disc inside the excision boundary is the
degenerate core whose field is meaningless -- that is the entire reason the
boundary exists.** Launching from the centre means the connector's first ~ring
radius traverses that core, and any critical point the ring swallowed (a leaked
neighbour, per the RING > CENSUS finding) sits on the ray and occludes it. From
the port, every CP inside the boundary has ``along < 0`` and is automatically
excluded. Measured on Pd_20K curvedness, MC18's ring is 8.3 deg and the
occluding minima sit at along 7.5 -- inside the core, so the port origin removes
them by construction, and the disc test removes the off-ray ones.

The mode stays an API parameter (not a viewer control) because it cannot change
the winner on a clean field, and Step 4's integrator starts at the port.

Reciprocity
-----------
Claims are one-way: a branch claims the first critical point it runs into. A
connection is flagged ``mutual`` when it is the tightest claim on that maximum,
i.e. the maximum also points back at this branch. The flag is reported, never
used to gate, because a non-mutual claim is exactly the signal a crossing
separatrix produces -- hiding it would hide the phenomenon.

Occlusion is NOT reported as a mere absence. A branch blocked by a minimum or a
saddle carries ``status == "occluded"`` and the blocker's id, because "this ray
flies over a minimum on the way to that maximum" is a topological finding about
the field, not a failure to find a target.

Frame contract: seeds carry CENTRED positions (``SaddleLaunchSeed``), maxima
carry WORLD positions (``BasinCenter``). Pass ``center`` (the world-frame
nucleus) to map the latter into the centred frame; the whole computation runs
centred, where ``p / ||p||`` is the outward normal, and every emitted coordinate
is mapped back out with ``to_global`` so the viewer can draw it. Omit ``center``
only for origin-centred synthetic meshes, where the two frames coincide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.sphere_frame import (
    angular_distance,
    radial_offset,
    rotate_towards,
    tangent_project,
    to_global,
    to_local,
    unit,
)

__all__ = [
    "ARC_SAMPLES",
    "CONE_PITCH",
    "CONNECTION_STATUSES",
    "DIRECTION_MODES",
    "MAX_ARC_PITCH",
    "ConnectionRay",
    "ConnectionTarget",
    "DirectionMode",
    "DiscGeometry",
    "MaxConnection",
    "basin_centers_to_targets",
    "build_connection_pool",
    "connect_seed_to_maxima",
    "connection_ray",
    "directional_support_radius",
    "disc_entry_angle",
    "disc_geometry_from_cluster",
    "mark_mutual",
    "ray_frame",
    "seed_ray_direction",
    "summarize_connections",
    "target_frame_angles",
]

# The ray's initial bearing is measured from the saddle centre toward the port.
# "port_tangent" starts at the port and follows the outward direction measured
# there. Both lie on the same great circle (see the module docstring); the mode
# is an API parameter for the integrator, not a viewer control.
DIRECTION_MODES: tuple[str, ...] = ("chord", "port_tangent")

DirectionMode = Literal["chord", "port_tangent"]

_STATUS_CONNECTED = "connected"
_STATUS_OCCLUDED = "occluded"
_STATUS_NO_TARGET = "no_target"

_KIND_MAXIMUM = "maximum"

# The three verdicts a branch can receive, in decreasing order of success.
CONNECTION_STATUSES: tuple[str, ...] = (
    _STATUS_CONNECTED,
    _STATUS_OCCLUDED,
    _STATUS_NO_TARGET,
)

# Longest arc a branch may claim, in mesh-pitch units. 40 pitches is ~64 deg on
# Pd_20K (pitch 1.6058 deg). Measured on the saved Pd_20K snapshots, the largest
# saddle-to-first-hit separation is 55.2 deg (V) and 45.0 deg (curvedness), so the
# cap admits every observed connection while excluding the far side of the sphere.
# Under the first-hit rule this is a safety net, not the discriminator: the nearer
# hit wins on along-track long before the cap binds. It is what stops a branch in
# an empty sector from claiming a maximum on the far side of the atom.
MAX_ARC_PITCH = 40.0

# Half-angle of the forward sector a branch may claim within, in mesh-pitch
# units. 6.0 pitches is 9.6 deg on Pd_20K (pitch 1.6058) and 13.4 deg on
# ethene3 (2.2278). Chosen as the smallest pitch multiple that admits every
# cross-track observed on the four saved Pd_20K snapshots: the largest was 7.82
# deg (V, 6 maxima), i.e. 4.9 pitches, so 6.0 leaves headroom without opening
# the sector wide enough to admit a neighbour off the branch's own gateway. In
# pitch units for the same reason as every other threshold here: the relevant
# scale is the intrinsic grid, and it keeps the constant transferable between
# the 1.6 deg and 2.2 deg meshes.
CONE_PITCH = 6.0

# Segments per rendered arc. The arc is a pure display artefact (the match itself
# is closed-form), so this only controls how smooth the tube looks.
ARC_SAMPLES = 24


@dataclass(frozen=True)
class ConnectionTarget:
    """One visible critical point a branch can run into.

    The occlusion test needs every critical point, not just maxima: a ray that
    flies over a minimum on its way to a distant maximum is not a connection.
    This is the uniform shape the pool is built from, so a raw CP, an effective
    CP and a :class:`BasinCenter` all enter the same search.

    A critical point is a DISC, not a point. Its excision boundary is the circle
    of ``angular_radius_rad`` about its centre, and that disc is the region where
    its own field is degenerate -- the reason the boundary exists. So a ray
    "hits" a critical point when it enters the disc, which is what makes a
    maximum capturable and a minimum sitting off to the side a non-obstacle.

    Attributes:
        cp_id: Identifier of the critical point.
        polarity: ``"maximum"``, ``"minimum"`` or ``"saddle"``.
        position: Position in the frame the caller's positions are in (world
            when built from the viewer's UI dicts). Not required to be unit
            length -- :func:`connect_seed_to_maxima` normalises it after mapping
            into the centred frame.
        value: Representative scalar field value.
        angular_radius_rad: Angular radius of this CP's excision disc, radians.
            0.0 means a point target (no disc), the conservative choice when the
            radius is unknown.
        semi_major_rad: Semi-major angular radius of the fitted boundary ellipse.
            Defaults to ``angular_radius_rad``; equal to the semi-minor for a
            compact set, in which case the disc is exact.
        semi_minor_rad: Semi-minor angular radius, radians. Defaults to
            ``angular_radius_rad``.
        principal_axis_unit: Major-axis direction of the ellipse, centred frame.
            Zero vector means no axis, and the radius is taken isotropically.
    """

    cp_id: str
    polarity: str
    position: npt.NDArray[np.float64]
    value: float
    angular_radius_rad: float = 0.0
    semi_major_rad: float = -1.0
    semi_minor_rad: float = -1.0
    principal_axis_unit: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        # A target built with only the isotropic radius is a circle: both
        # semi-axes take it, so the directional radius returns it unchanged.
        if self.semi_major_rad < 0.0:
            object.__setattr__(self, "semi_major_rad", self.angular_radius_rad)
        if self.semi_minor_rad < 0.0:
            object.__setattr__(self, "semi_minor_rad", self.angular_radius_rad)

    def radius_toward(self, target_unit: npt.NDArray[np.float64]) -> float:
        """Excision-boundary radius in the direction of ``target_unit``.

        Returns the isotropic ``angular_radius_rad`` when the target has no
        fitted axis, so a compact set and a set of unknown shape behave exactly
        as they did before the ellipse was carried.
        """
        axis = np.asarray(self.principal_axis_unit, dtype=np.float64)
        if float(np.linalg.norm(axis)) <= 1e-12:
            return float(self.angular_radius_rad)
        return directional_support_radius(
            self.position, self.semi_major_rad, self.semi_minor_rad, axis, target_unit
        )


@dataclass(frozen=True)
class DiscGeometry:
    """The fitted excision boundary of one critical point, as an ellipse.

    A compact set has ``semi_major_rad == semi_minor_rad`` and the ellipse is a
    circle; a bridged corridor is elongated and the transverse radius is far
    smaller than the longitudinal one. The occlusion test needs the transverse
    value, so it must carry the shape, not just a scalar.

    Attributes:
        angular_radius_rad: Isotropic radius, the fallback when no axis is set.
        semi_major_rad: Semi-major angular radius, radians.
        semi_minor_rad: Semi-minor angular radius, radians.
        principal_axis_unit: Major-axis direction, centred frame. A zero vector
            means "no axis", and the radius is taken isotropically.
    """

    angular_radius_rad: float
    semi_major_rad: float
    semi_minor_rad: float
    principal_axis_unit: tuple[float, float, float] = (0.0, 0.0, 0.0)


def disc_geometry_from_cluster(cluster: MicroCluster) -> DiscGeometry:
    """Read a cluster's fitted boundary as a :class:`DiscGeometry`.

    A cluster with no fitted ellipse (morphology "compact") has its semi-axes set
    to the isotropic radius, so the geometry round-trips exactly.

    Args:
        cluster: The micro-cluster (or a corridor adapted to one).

    Returns:
        The disc geometry for its excision boundary.
    """
    axis = cluster.principal_axis_unit
    return DiscGeometry(
        angular_radius_rad=float(cluster.angular_radius_rad),
        semi_major_rad=float(cluster.semi_major_rad or cluster.angular_radius_rad),
        semi_minor_rad=float(cluster.semi_minor_rad or cluster.angular_radius_rad),
        principal_axis_unit=(float(axis[0]), float(axis[1]), float(axis[2])),
    )


def basin_centers_to_targets(
    centers: list[BasinCenter],
    discs: dict[str, DiscGeometry] | None = None,
) -> list[ConnectionTarget]:
    """Convert visible extrema into connection targets.

    Args:
        centers: Basin centers (maxima and minima) in the world frame, as
            produced by ``extract_basin_centers``.
        discs: Fitted excision boundary per entry, keyed by ``center_id``.
            Entries with no geometry get a zero radius and behave as points.

    Returns:
        One :class:`ConnectionTarget` per entry, positions left in the input
        frame -- :func:`connect_seed_to_maxima` maps them to the centred frame
        when ``center`` is given.
    """
    table = discs or {}
    out: list[ConnectionTarget] = []
    for c in centers:
        g = table.get(str(c.center_id))
        out.append(
            ConnectionTarget(
                cp_id=str(c.center_id),
                polarity=c.polarity,
                position=np.asarray(c.position, dtype=np.float64),
                value=float(c.value),
                angular_radius_rad=g.angular_radius_rad if g else 0.0,
                semi_major_rad=g.semi_major_rad if g else 0.0,
                semi_minor_rad=g.semi_minor_rad if g else 0.0,
                principal_axis_unit=(
                    np.asarray(g.principal_axis_unit, dtype=np.float64)
                    if g
                    else np.zeros(3, dtype=np.float64)
                ),
            )
        )
    return out


def build_connection_pool(
    basin_centers: list[BasinCenter],
    raw_saddles: list[CriticalPoint],
    effective_cps: list[dict[str, Any]],
    hidden_vertex_ids: set[int],
    center: tuple[float, float, float] | None = None,
    disc_by_vertex: dict[int, DiscGeometry] | None = None,
    disc_by_id: dict[str, DiscGeometry] | None = None,
) -> list[ConnectionTarget]:
    """Assemble every VISIBLE critical point into one occlusion pool.

    The first-hit rule can only express "this ray flies over a minimum/saddle on
    the way to that maximum" if the minimum/saddle is in the pool. Maxima and
    minima come from ``extract_basin_centers``; saddles do NOT -- that function
    emits extrema only. So the visible saddles are added here from two sources,
    which is the recurring "effective CPs have no mesh vertex" trap:

    - raw saddles that survived reduction (``vertex_id not in hidden``), whose
      ``CriticalPoint.position`` is in the CENTRED frame;
    - ``E-SAD`` entries of the viewer's effective-CP dicts, whose ``position`` is
      already in the WORLD frame (an effective CP has no vertex to scan).

    The two frames are reconciled here with ``to_global`` on the raw saddles, so
    every emitted position shares the world frame the basin centers use. Omit
    ``center`` only for origin-centred synthetic meshes, where the frames
    coincide -- on a real molecule they do not, and mixing them puts the raw
    saddles up to 4 R off the sphere.

    Args:
        basin_centers: Visible maxima and minima, world frame.
        raw_saddles: All raw saddles from the Morse detector, centred frame.
        effective_cps: The viewer's effective-CP dicts (world-frame positions).
        hidden_vertex_ids: Raw vertices collapsed into an effective entity; their
            saddles are not visible and must not occlude.
        center: World-frame nucleus, used to map the centred raw saddles out.
        disc_by_vertex: Fitted excision boundary per raw CP, keyed by mesh vertex
            id (``disc_geometry_from_cluster`` on its micro-cluster).
        disc_by_id: Fitted excision boundary per effective CP, keyed by entity id
            (the reduction unit's adapted cluster).

    Returns:
        A de-duplicated pool in the world frame (centred when ``center`` is
        None), ordered by id.
    """
    by_vertex = disc_by_vertex or {}
    by_id = disc_by_id or {}
    point = DiscGeometry(0.0, 0.0, 0.0)
    geo: dict[str, DiscGeometry] = dict(by_id)
    for c in basin_centers:
        if c.center_id in geo:
            continue
        if c.vertex_id >= 0 and c.vertex_id in by_vertex:
            geo[c.center_id] = by_vertex[c.vertex_id]

    pool: list[ConnectionTarget] = basin_centers_to_targets(basin_centers, geo)

    for sd in raw_saddles:
        if sd.vertex_id in hidden_vertex_ids:
            continue
        pos = (
            np.asarray(to_global(sd.position, center), dtype=np.float64)
            if center is not None
            else np.asarray(sd.position, dtype=np.float64)
        )
        g = by_vertex.get(sd.vertex_id, point)
        pool.append(
            ConnectionTarget(
                cp_id=f"SAD_{sd.vertex_id}",
                polarity="saddle",
                position=pos,
                value=float(sd.value),
                angular_radius_rad=g.angular_radius_rad,
                semi_major_rad=g.semi_major_rad,
                semi_minor_rad=g.semi_minor_rad,
                principal_axis_unit=np.asarray(g.principal_axis_unit, dtype=np.float64),
            )
        )

    for eff in effective_cps:
        if eff.get("type") != "E-SAD":
            continue
        pos = eff.get("position", (0.0, 0.0, 0.0))
        eid = str(eff.get("id_label", "E-SAD"))
        g = by_id.get(eid, point)
        pool.append(
            ConnectionTarget(
                cp_id=eid,
                polarity="saddle",
                position=np.asarray(pos, dtype=np.float64),
                value=float(eff.get("value", 0.0)),
                angular_radius_rad=g.angular_radius_rad,
                semi_major_rad=g.semi_major_rad,
                semi_minor_rad=g.semi_minor_rad,
                principal_axis_unit=np.asarray(g.principal_axis_unit, dtype=np.float64),
            )
        )

    # De-duplicate by id: a saddle can appear in more than one source, and a
    # duplicate would be scored twice with identical geometry.
    seen: set[str] = set()
    unique: list[ConnectionTarget] = []
    for t in pool:
        if t.cp_id in seen:
            continue
        seen.add(t.cp_id)
        unique.append(t)
    unique.sort(key=lambda t: t.cp_id)
    return unique


@dataclass(frozen=True)
class ConnectionRay:
    """A great-circle ray on the atom sphere, in the centred frame.

    Attributes:
        origin_unit: Unit radial vector of the ray's start.
        direction_unit: Unit tangent at the start giving the initial bearing.
        pole_unit: Unit normal of the ray's great circle, ``origin x direction``.
    """

    origin_unit: npt.NDArray[np.float64]
    direction_unit: npt.NDArray[np.float64]
    pole_unit: npt.NDArray[np.float64]


@dataclass(frozen=True)
class MaxConnection:
    """One ascent branch and the critical point it runs into first.

    Attributes:
        saddle_id: Source saddle's id (``"SAD_142"`` / ``"E-SAD_MC7"``).
        branch_index: Index of the branch within the seed's ascent list.
        max_id: Claimed maximum's ``center_id``. Set only when ``status`` is
            ``"connected"``.
        status: ``"connected"`` (the first hit was a maximum), ``"occluded"``
            (the first hit was a minimum or a saddle, so the ray flies over a
            critical point on the way), or ``"no_target"`` (nothing in the
            forward sector).
        direction_mode: Which ray definition produced this verdict.
        cross_track_deg: Perpendicular deviation of the hit from the ray, signed
            toward the ray's left (the pole). Negative for a hit on the other
            side. None when nothing was hit.
        along_track_deg: Arc-length along the ray at which the hit's excision
            disc is first entered. This is the quantity the rule minimises, and
            for a maximum it is where the connector MEETS the basin, not where
            its centre sits.
        separation_deg: Great-circle angle saddle-to-hit.
        mutual: True when this is the tightest claim on that maximum.
        arc_points: World-frame polyline of the FINAL CONNECTOR: the single
            great circle from the source saddle's centre to the hit's centre,
            sampled on the sphere. Empty when nothing was hit. The port and the
            boundary crossing are search internals and are not vertices here.
        saddle_value: Field value at the saddle.
        max_value: Field value at the claimed maximum.
        verified: True when the source seed's ring converged on the fold order
            the reduction reports (``index_agrees``).
        blocker_id: Id of the critical point that occluded this branch, and its
            polarity. Recorded so "why is there no arc here?" is answerable from
            the table without re-running the geometry.
    """

    saddle_id: str
    branch_index: int
    max_id: str | None
    status: str
    direction_mode: str
    cross_track_deg: float | None
    along_track_deg: float | None
    separation_deg: float | None
    mutual: bool
    arc_points: tuple[tuple[float, float, float], ...]
    saddle_value: float
    max_value: float | None
    verified: bool
    blocker_id: str | None = None
    blocker_polarity: str | None = None


def ray_frame(
    origin_unit: npt.NDArray[np.float64],
    direction_unit: npt.NDArray[np.float64],
) -> ConnectionRay:
    """Build the orthonormal ``(o, d, n)`` frame of a great-circle ray.

    Args:
        origin_unit: Unit radial vector of the ray's start.
        direction_unit: Unit tangent at the start.

    Returns:
        A :class:`ConnectionRay` whose pole is ``origin x direction``.
    """
    pole = np.cross(origin_unit, direction_unit)
    pole_unit = unit(pole)
    if pole_unit is None:
        # Degenerate: direction parallel to the radial. Fall back to any pole
        # perpendicular to the origin so the frame stays orthonormal.
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis, origin_unit))) > 0.9:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        pole_unit = unit(tangent_project(axis, origin_unit))
        if pole_unit is None:  # pragma: no cover - origin_unit is unit, so never
            pole_unit = axis
    return ConnectionRay(
        origin_unit=origin_unit, direction_unit=direction_unit, pole_unit=pole_unit
    )


def target_frame_angles(
    target_unit: npt.NDArray[np.float64],
    ray: ConnectionRay,
) -> tuple[float, float]:
    """Decompose a target direction into (along, cross) angles about a ray.

    ``along`` is the signed progress from the ray origin toward the target,
    positive ahead of the ray; ``cross`` is the latitude out of the ray's great
    circle, signed toward the pole. Both in radians.

    Args:
        target_unit: Unit radial vector of the candidate.
        ray: The ray frame.

    Returns:
        Tuple of (along_rad, cross_rad).
    """
    along = math.atan2(
        float(np.dot(target_unit, ray.direction_unit)),
        float(np.dot(target_unit, ray.origin_unit)),
    )
    cross = math.asin(
        float(np.clip(float(np.dot(target_unit, ray.pole_unit)), -1.0, 1.0))
    )
    return along, cross


def directional_support_radius(
    centroid_unit: npt.NDArray[np.float64],
    semi_major_rad: float,
    semi_minor_rad: float,
    principal_axis_unit: npt.NDArray[np.float64],
    target_unit: npt.NDArray[np.float64],
) -> float:
    """Angular radius of a fitted ellipse toward a target direction.

    Evaluates the polar form of the boundary ellipse,

        r(phi) = sqrt((a cos phi)^2 + (b sin phi)^2),

    at the bearing of ``target_unit`` on the tangent plane at ``centroid_unit``,
    with ``phi`` folded to [0, 90] degrees because the principal axis is an
    undirected line.

    Why the disc must be directional here. An excision boundary around an
    elongated set is an ELLIPSE, not a circle: a bridged corridor has a
    semi-major along the chain and a semi-minor across it. Treating it as a disc
    of radius ``a`` inflates the transverse extent by the aspect ratio, and the
    occlusion test then blocks rays that pass well clear of the set. Measured on
    Pd_20K shape index: corridor MS13 has a = 11.35, b = 2.43 (aspect 10.6). A
    ray to MAX_4 passed it at cross-track 3.58 deg -- outside the 2.78 deg
    transverse radius, inside the 11.35 deg isotropic one -- so the corridor
    occluded a maximum it never touched, and the eight raw maxima around the
    pattern lost one connection each.

    This is the same failure the bridging gap test already fixed for the same
    reason (``string_aggregator.cluster_directional_support_radius``), whose
    docstring names it: an isotropic radius "turns a long thin corridor into a
    disk of radius a and lets it swallow compact fragments sitting far off to
    the side". That function delegates here so the ellipse is evaluated in
    exactly one place.

    Args:
        centroid_unit: Unit radial of the ellipse centre, centred frame.
        semi_major_rad: Semi-major angular radius, radians.
        semi_minor_rad: Semi-minor angular radius, radians.
        principal_axis_unit: Major-axis direction (need not be normalised or
            tangent-projected; it is made tangent here).
        target_unit: Direction to evaluate toward, centred frame.

    Returns:
        Angular radius in radians toward the target. Falls back to the
        semi-major whenever the bearing or axis is undefined, which is the
        conservative choice: it never shrinks the reach on ambiguous geometry.
    """
    a = float(semi_major_rad)
    b = float(semi_minor_rad)
    if a <= 0.0:
        return 0.0
    if b <= 0.0:
        return a
    if abs(a - b) <= 1e-12:
        return a

    c = unit(centroid_unit)
    if c is None:
        return a
    t = np.asarray(target_unit, dtype=np.float64)
    bearing = unit(tangent_project(t, c))
    if bearing is None:
        return a
    axis = unit(tangent_project(np.asarray(principal_axis_unit, dtype=np.float64), c))
    if axis is None:
        return a

    cos_phi = abs(float(np.dot(bearing, axis)))
    cos2 = min(max(cos_phi * cos_phi, 0.0), 1.0)
    sin2 = max(0.0, 1.0 - cos2)
    return math.sqrt((a * a) * cos2 + (b * b) * sin2)


def disc_entry_angle(
    along_center_rad: float,
    cross_rad: float,
    disc_radius_rad: float,
) -> float | None:
    """Where a ray first touches a critical point's excision disc.

    A CP is a disc of angular radius ``rho`` about its centre, not a point. On
    the ray's great circle, a point at arc-length ``s`` from the origin sits at
    angular distance ``dist(s)`` from the centre with

        cos dist(s) = cos(c) * cos(s - a)

    where ``a`` is the centre's along-track and ``c`` its cross-track. Setting
    ``dist = rho`` gives the entry arc-length below. The solution exists exactly
    when ``|c| <= rho``: the great circle passes through the disc.

    Args:
        along_center_rad: Along-track angle of the disc centre.
        cross_rad: Cross-track angle of the disc centre (signed).
        disc_radius_rad: Angular radius of the disc. 0.0 degenerates to the
            centre point, whose entry is the centre itself.

    Returns:
        The arc-length at which the ray enters the disc, or None when the great
        circle misses the disc entirely.
    """
    if disc_radius_rad <= 0.0:
        return along_center_rad
    cos_c = math.cos(cross_rad)
    if cos_c <= 1e-15:
        return None
    ratio = math.cos(disc_radius_rad) / cos_c
    if ratio > 1.0:
        return None  # the great circle passes outside the disc
    return along_center_rad - math.acos(max(ratio, -1.0))


def connection_ray(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    direction_mode: DirectionMode,
    sphere_radius: float,
) -> ConnectionRay | None:
    """Build the great-circle ray of one launch branch.

    Args:
        seed: The source saddle.
        branch: One of the seed's ascent branches.
        direction_mode: ``"chord"`` starts at the saddle centre and aims at the
            port; ``"port_tangent"`` starts at the port along the port's own
            outward tangent.
        sphere_radius: Sphere radius, used only to normalise positions.

    Returns:
        The ray frame, or None when the geometry is degenerate (a centre at the
        nucleus, or a zero-length direction with no usable fallback).
    """
    centre = np.asarray(seed.saddle_position, dtype=np.float64)
    centre_unit = unit(centre)
    if centre_unit is None:
        return None
    port = np.asarray(branch.launch_position, dtype=np.float64)
    port_unit = unit(port)

    if direction_mode == "port_tangent":
        origin = port_unit if port_unit is not None else centre_unit
        raw = np.asarray(branch.tangent_direction, dtype=np.float64)
        direction = unit(tangent_project(raw, origin))
        if direction is None:
            return None
        return ray_frame(origin, direction)

    # "chord": the bearing from the saddle centre to the port. A port that
    # coincides with the centre carries no bearing, so fall back to the
    # branch's own tangent transported to the centre.
    if port_unit is not None:
        direction = unit(tangent_project(port_unit, centre_unit))
        if direction is not None:
            return ray_frame(centre_unit, direction)
    raw = np.asarray(branch.tangent_direction, dtype=np.float64)
    direction = unit(tangent_project(raw, centre_unit))
    if direction is None:
        return None
    return ray_frame(centre_unit, direction)


def seed_ray_direction(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    direction_mode: DirectionMode = "port_tangent",
    sphere_radius: float = 1.0,
) -> tuple[float, float, float] | None:
    """Unit initial bearing of one ascent branch, in the centred frame.

    A thin accessor over :func:`connection_ray` for callers (and tests) that
    want only the direction.

    Args:
        seed: The source saddle.
        branch: One of the seed's ascent branches.
        direction_mode: Which ray definition to use.
        sphere_radius: Sphere radius.

    Returns:
        The unit direction as a float tuple, or None when degenerate.
    """
    ray = connection_ray(seed, branch, direction_mode, sphere_radius)
    if ray is None:
        return None
    d = ray.direction_unit
    return (float(d[0]), float(d[1]), float(d[2]))


def _arc_between(
    origin_unit: npt.NDArray[np.float64],
    target_unit: npt.NDArray[np.float64],
    sphere_radius: float,
    center: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], ...]:
    """Sample the great-circle arc from a saddle to a maximum, in world frame.

    The connector is the shortest geodesic between the two endpoints, which is
    what "straight line on the sphere" means. It is built with the exponential
    map so every sample lies exactly on the sphere.

    Args:
        origin_unit: Unit radial of the saddle.
        target_unit: Unit radial of the maximum.
        sphere_radius: Sphere radius.
        center: World-frame nucleus, or None for an origin-centred mesh.

    Returns:
        ``ARC_SAMPLES + 1`` world-frame points, endpoints included.
    """
    span = angular_distance(origin_unit, target_unit)
    direction = unit(tangent_project(target_unit, origin_unit))

    def _emit(local: tuple[float, float, float]) -> tuple[float, float, float]:
        return to_global(local, center) if center is not None else local

    if direction is None or span <= 1e-12:
        # Coincident endpoints (or a target at the antipode of the projection):
        # there is no unique geodesic, so draw the straight chord between them.
        start = _emit(radial_offset(origin_unit, sphere_radius))
        end = _emit(radial_offset(target_unit, sphere_radius))
        return (start, end)

    out: list[tuple[float, float, float]] = []
    for i in range(ARC_SAMPLES + 1):
        frac = i / ARC_SAMPLES
        local = rotate_towards(origin_unit, direction, span * frac, sphere_radius)
        out.append(_emit(local))
    return tuple(out)


def connect_seed_to_maxima(
    seeds: list[SaddleLaunchSeed],
    targets: list[ConnectionTarget],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    max_arc_pitch: float = MAX_ARC_PITCH,
    cone_pitch: float = CONE_PITCH,
    direction_mode: DirectionMode = "port_tangent",
) -> list[MaxConnection]:
    """Find the first critical point each ascent branch runs into.

    Args:
        seeds: Saddle launch seeds, in the centred atom frame. Both the verified
            and the provisional (``k_mismatch``) sets are usually passed, so the
            connection table matches what the scene draws; each record carries
            ``verified`` so the two stay distinguishable.
        targets: EVERY visible critical point -- maxima, minima and saddles --
            each carrying its excision-disc radius. A branch that flies over a
            minimum toward a distant maximum is occluded, which the rule can only
            see if the minimum is in the pool. Build with
            :func:`build_connection_pool`.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to read the cap and
            the sector half-angle.
        center: World-frame nucleus. Required when ``targets`` carry world
            positions (they do, coming from ``extract_basin_centers``); the
            output arcs are then emitted in the world frame too.
        max_arc_pitch: Arc-length cap in mesh-pitch units.
        cone_pitch: Forward-sector half-angle in mesh-pitch units.
        direction_mode: ``"port_tangent"`` (default) launches from the port on
            the excision boundary; ``"chord"`` launches from the saddle centre.
            See the module docstring for why the port is the origin.

    Returns:
        One :class:`MaxConnection` per (seed, ascent branch), in input order.
        Deterministic: the winner depends only on the candidate SET, never on
        its order.
    """
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(
            f"direction_mode {direction_mode!r} must be one of {DIRECTION_MODES}"
        )
    max_arc_rad = max_arc_pitch * delta_theta_mesh_rad
    cone_rad = cone_pitch * delta_theta_mesh_rad

    prepared: list[ConnectionTarget] = []
    for t in targets:
        pos = np.asarray(t.position, dtype=np.float64)
        local = to_local(pos, center) if center is not None else pos
        u = unit(local)
        if u is None:
            continue
        # The principal axis is a DIRECTION, so it is translation invariant and
        # needs no to_local. It is normalised here because radius_toward only
        # tests it for zero-length and then projects it.
        axis = np.asarray(t.principal_axis_unit, dtype=np.float64)
        axis_unit = unit(axis)
        prepared.append(
            ConnectionTarget(
                cp_id=str(t.cp_id),
                polarity=t.polarity,
                position=u,
                value=t.value,
                angular_radius_rad=float(t.angular_radius_rad),
                semi_major_rad=float(t.semi_major_rad),
                semi_minor_rad=float(t.semi_minor_rad),
                principal_axis_unit=(
                    axis_unit
                    if axis_unit is not None
                    else np.zeros(3, dtype=np.float64)
                ),
            )
        )
    # Order the candidate list by id so a tie in the metrics resolves the same
    # way regardless of how the caller assembled the list.
    prepared.sort(key=lambda t: t.cp_id)

    out: list[MaxConnection] = []
    for seed in seeds:
        centre = np.asarray(seed.saddle_position, dtype=np.float64)
        centre_unit = unit(centre)
        for branch in seed.ascent_branches:
            ray = connection_ray(seed, branch, direction_mode, sphere_radius)
            if ray is None or centre_unit is None:
                # connection_ray returns None exactly when the centre is at the
                # nucleus, so the second test is the type narrowing for the
                # connector below; both are the same degenerate seed.
                out.append(_no_target(seed, branch, direction_mode))
                continue

            # WALK THE RAY OUTWARD, in increasing along-track, and stop at the
            # first critical point that is either a maximum (a target) or a disc
            # the ray actually enters (an obstacle).
            #
            # The disc test applies to OBSTACLES ONLY. A minimum or saddle blocks
            # the ray only if the great circle runs into its excision boundary:
            # measured on Pd_20K curvedness, MC18's branch 0 was blocked by
            # MIN_19165 at cross -7.41 with a 1.61 deg disc -- the ray sails past
            # it, so it is not an obstacle, and E-MAX_MC38 at cross -0.39 is the
            # true hit.
            #
            # A MAXIMUM is captured by the CONE, not by its disc. Requiring the
            # ray to hit the maximum's own boundary is tuned backwards: every raw
            # maximum's excision radius is one mesh pitch (1.61 deg on Pd_20K),
            # while the legitimate rays into a 3-saddle maximum arrive at
            # cross-tracks 0.0, 1.35, 1.42 (MAX_7) and 0.0, 2.00, 1.98 (MAX_4) --
            # the gate sits inside the measurement-noise band, so it accepted
            # MAX_7's three connections and rejected two of MAX_4's. The pattern
            # is symmetric; only the gate's luck is not. The cone is the
            # directional criterion, and the disc is where the arc STOPS.
            #
            # rho is the boundary radius TOWARD the ray, not the isotropic one.
            # An excision boundary around an elongated set is an ellipse, and
            # using its semi-major as a disc inflates the transverse reach by the
            # aspect ratio: on Pd_20K shape index the corridor MS13 (a=11.35,
            # b=2.43) occluded a ray passing 3.58 deg off its axis, because
            # 3.58 < 11.35 while 3.58 > 2.78. See directional_support_radius.
            #
            # The Entry 043 rule minimised |cross| with along as a tie-break,
            # which scored aim without checking reach: a ray through a near
            # maximum toward a far one has cross ~ 0 for BOTH, so cross-track
            # cannot separate them and the tie-break was float noise.
            candidates: list[tuple[float, float, int]] = []
            for idx, t in enumerate(prepared):
                # A branch's own saddle is not a candidate: the ray starts there.
                if t.cp_id == seed.saddle_id:
                    continue
                along, cross = target_frame_angles(t.position, ray)
                if along <= 0.0 or along > max_arc_rad:
                    continue
                if abs(cross) > cone_rad:
                    continue
                candidates.append((along, cross, idx))
            candidates.sort(key=lambda r: (r[0], abs(r[1]), r[2]))

            hit_idx = -1
            entry_rad = 0.0
            cross = 0.0
            occluder_idx = -1
            for along, cross_c, idx in candidates:
                t = prepared[idx]
                if t.polarity == _KIND_MAXIMUM:
                    # Captured by the CONE. The boundary entry is recorded as the
                    # ranking key (along_track_deg) and as where the ray meets the
                    # basin, but it is NOT a vertex of the drawn connector: the
                    # search's job is to find the two endpoints, and the connector
                    # is the single geodesic between them.
                    rho = t.radius_toward(ray.direction_unit)
                    boundary = disc_entry_angle(along, cross_c, rho)
                    hit_idx = idx
                    entry_rad = (
                        boundary if boundary is not None and boundary > 0.0 else along
                    )
                    cross = cross_c
                    break
                rho = t.radius_toward(ray.direction_unit)
                obstacle = disc_entry_angle(along, cross_c, rho)
                if obstacle is not None and obstacle > 0.0:
                    hit_idx, entry_rad, cross, occluder_idx = (
                        idx,
                        obstacle,
                        cross_c,
                        idx,
                    )
                    break

            if hit_idx < 0:
                out.append(_no_target(seed, branch, direction_mode))
                continue

            hit = prepared[hit_idx]
            connected = occluder_idx < 0
            out.append(
                MaxConnection(
                    saddle_id=seed.saddle_id,
                    branch_index=branch.sector_index,
                    max_id=hit.cp_id if connected else None,
                    status=_STATUS_CONNECTED if connected else _STATUS_OCCLUDED,
                    direction_mode=direction_mode,
                    cross_track_deg=math.degrees(cross),
                    along_track_deg=math.degrees(entry_rad),
                    separation_deg=math.degrees(
                        angular_distance(centre_unit, hit.position)
                    ),
                    mutual=False,
                    # THE FINAL CONNECTOR: one great circle from the saddle
                    # cluster centre to the maximum cluster centre. The ray, the
                    # port and the boundary crossing were the means of finding
                    # these two endpoints; the connector itself is the geodesic
                    # between them, so it has no kink where it meets the basin.
                    arc_points=_arc_between(
                        centre_unit, hit.position, sphere_radius, center
                    ),
                    saddle_value=float(seed.saddle_value),
                    max_value=float(hit.value) if connected else None,
                    verified=bool(seed.index_agrees),
                    blocker_id=None if connected else hit.cp_id,
                    blocker_polarity=None if connected else hit.polarity,
                )
            )

    return mark_mutual(out)


def _no_target(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    direction_mode: str,
) -> MaxConnection:
    """Build the record for a branch that claimed nothing.

    Args:
        seed: The source saddle.
        branch: The unclaimed ascent branch.
        direction_mode: The ray definition in force.

    Returns:
        A :class:`MaxConnection` with ``status == "no_target"`` and an empty arc.
    """
    return MaxConnection(
        saddle_id=seed.saddle_id,
        branch_index=branch.sector_index,
        max_id=None,
        status=_STATUS_NO_TARGET,
        direction_mode=direction_mode,
        cross_track_deg=None,
        along_track_deg=None,
        separation_deg=None,
        mutual=False,
        arc_points=(),
        saddle_value=float(seed.saddle_value),
        max_value=None,
        verified=bool(seed.index_agrees),
    )


def mark_mutual(connections: list[MaxConnection]) -> list[MaxConnection]:
    """Flag the tightest claim on each maximum as mutual.

    A maximum is "claimed by" every branch whose first hit is that maximum. The
    branch with the smallest cross-track is the one the maximum points back at,
    which is the reciprocal reading of the connection. Exactly one connection per
    maximum is flagged, and only connected ones are eligible (occluded and
    no-target records carry no ``max_id``).

    Args:
        connections: The connection list, in any order.

    Returns:
        A new list with ``mutual`` set; input order and all other fields are
        preserved.
    """
    winner: dict[str, tuple[float, float, int]] = {}
    for i, conn in enumerate(connections):
        if conn.max_id is None or conn.cross_track_deg is None:
            continue
        key = (abs(conn.cross_track_deg), float(conn.along_track_deg or 0.0), i)
        if conn.max_id not in winner or key < winner[conn.max_id]:
            winner[conn.max_id] = key

    out: list[MaxConnection] = []
    for i, conn in enumerate(connections):
        if conn.max_id is None:
            out.append(conn)
            continue
        is_mutual = winner.get(conn.max_id, (None, None, -1))[2] == i
        out.append(replace(conn, mutual=is_mutual))
    return out


def summarize_connections(connections: list[MaxConnection]) -> dict[str, Any]:
    """Reduce a connection list to the counters the viewer card shows.

    Args:
        connections: Output of :func:`connect_seed_to_maxima`.

    Returns:
        Dict of counts and angular statistics. Every float is finite, so the
        result is safe to write into a params snapshot and diff.
    """
    connected = [c for c in connections if c.status == _STATUS_CONNECTED]
    occluded = [c for c in connections if c.status == _STATUS_OCCLUDED]
    no_target = [c for c in connections if c.status == _STATUS_NO_TARGET]
    cross = [
        abs(float(c.cross_track_deg))
        for c in connected
        if c.cross_track_deg is not None
    ]
    sep = [float(c.separation_deg) for c in connected if c.separation_deg is not None]
    return {
        "direction_mode": connections[0].direction_mode if connections else "",
        "n_branches": len(connections),
        "n_connected": len(connected),
        "n_occluded": len(occluded),
        "n_no_target": len(no_target),
        "n_saddles_connected": len({c.saddle_id for c in connected}),
        "n_maxima_connected": len({c.max_id for c in connected}),
        "n_mutual": sum(1 for c in connections if c.mutual),
        "n_unverified": sum(1 for c in connected if not c.verified),
        "n_blocked_by_minimum": sum(
            1 for c in occluded if c.blocker_polarity == "minimum"
        ),
        "n_blocked_by_saddle": sum(
            1 for c in occluded if c.blocker_polarity == "saddle"
        ),
        "mean_cross_track_deg": sum(cross) / len(cross) if cross else 0.0,
        "max_cross_track_deg": max(cross) if cross else 0.0,
        "mean_separation_deg": sum(sep) / len(sep) if sep else 0.0,
        "max_separation_deg": max(sep) if sep else 0.0,
    }
