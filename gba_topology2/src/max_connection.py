"""Maximum Connection Subroutine for gba_topology2 (Basin Locator Step 3).

A saddle's ascent (ridge) launch branch is the direction the separatrix leaves
the saddle on its way UP toward a maximum. This subroutine answers the question
that direction poses: *which* maximum is this branch aiming at?

The method (straight-line / great-circle)
-----------------------------------------
Each branch supplies a ray on the sphere: an origin and an initial geodesic
direction. A candidate maximum is scored by its position relative to that ray,
decomposed in the orthonormal frame ``(o, d, n)`` where ``o`` is the unit origin,
``d`` the unit direction and ``n = o x d`` the pole of the ray's great circle:

    along = atan2(t.d, t.o)      progress along the ray, signed
    cross = asin(t.n)            perpendicular deviation from the ray

The winner is the maximum with the smallest ``|cross|`` among those in the
forward hemisphere (``along > 0``) and within the arc-length cap
(``along <= max_arc_pitch * delta_theta``). Ties break on ``along`` then on the
maximum id, so the result is a pure function of the input set and not of its
order.

This is deliberately the SIMPLEST possible connector: a great circle, no field
integration, no barrier test. Step 4 replaces it with the traced separatrix, and
the value of having it early is that the traced path has something to be
compared against -- a geodesic that lands on a different maximum than the
integrator does is a measurement, not a surprise.

Why cross-track and not plain proximity
---------------------------------------
"Nearest maximum to the launch point" ignores direction entirely: on a dense
field (Pd_20K V: 6 visible maxima, 1.6 deg mesh pitch) the nearest maximum is
almost always off to the side of where the branch points, and two opposite
branches of the same saddle claim the same extremum. Cross-track is directional
by construction, and reporting it alongside ``separation`` (the raw saddle-max
angle) keeps the two effects separable when a claim looks wrong.

The two ray definitions
-----------------------
The excision path (``excision_launch.py``) places ``launch_position`` on the
ring and ``tangent_direction`` at that port, pointing outward from the ring
centre. The saddle centre is the ring centre. So there are two defensible rays:

- ``"chord"``: origin = saddle centre, direction = the geodesic bearing from the
  centre to the port. This is the direction the separatrix had at the saddle
  itself, which is what "the launch seed minus the saddle position" means.
- ``"port_tangent"``: origin = the port, direction = ``tangent_direction``, the
  outward continuation measured AT the ring. This is where Step 4's integrator
  will start.

They are NOT the same arc, and they are the same great circle. ``_outward_tangent
`` builds the port direction as ``radial * port - centre``, which is the parallel
transport of the centre->port geodesic, so both rays lie on the plane spanned by
the centre and the port and share its pole ``centre x port``. They differ only in
where the arc begins, which shifts ``along_track`` by the ring radius and leaves
``cross_track`` -- the quantity the claim rule minimises -- untouched.

Measured on the three saved Pd_20K snapshots (V, curvedness, shape index; 24, 72
and 96 ascent branches): the two modes produce identical claim sets and identical
cross-track to 1e-11 deg. The mode is therefore kept as an API parameter, not a
viewer control -- it will matter to the integrator, which starts at the port, and
it is not worth an operator-facing knob that cannot change an answer. The default
is ``"chord"``, the literal reading of the request.

Reciprocity
-----------
Claims are one-way: a branch claims its best maximum. A connection is flagged
``mutual`` when it is the tightest claim on that maximum, i.e. the maximum also
points back at this branch. The flag is reported, never used to gate, because a
non-mutual claim is exactly the signal a crossing separatrix produces -- hiding
it would hide the phenomenon.

Frame contract: seeds carry CENTRED positions (``SaddleLaunchSeed``), maxima
carry WORLD positions (``BasinCenter``). Pass ``center`` (the world-frame
nucleus) to map the latter into the centred frame; the whole computation runs
centred, where ``p / ||p||`` is the outward normal, and every emitted coordinate
is mapped back out with ``to_global`` so the viewer can draw it. Omit ``center``
only for origin-centred synthetic meshes, where the two frames coincide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from gba_topology2.src.basin_center import BasinCenter
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
    "DIRECTION_MODES",
    "MAX_ARC_PITCH",
    "ConnectionRay",
    "DirectionMode",
    "MaxConnection",
    "connect_seed_to_maxima",
    "connection_ray",
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

# Longest arc a branch may claim, in mesh-pitch units. 40 pitches is ~64 deg on
# Pd_20K (pitch 1.6058 deg). Measured on the three saved Pd_20K snapshots, the
# largest saddle-to-claimed-maximum separation is 57.1 deg (shape index), 55.2
# (V) and 45.6 (curvedness), so the cap admits every observed connection while
# still excluding the far side of the sphere, where cross-track stops
# discriminating. It is a ceiling, not a tuned threshold: no claim is lost at
# this value on any field measured so far.
MAX_ARC_PITCH = 40.0

# Segments per rendered arc. The arc is a pure display artefact (the match itself
# is closed-form), so this only controls how smooth the tube looks.
ARC_SAMPLES = 24

_STATUS_CONNECTED = "connected"
_STATUS_NO_TARGET = "no_target"


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
    """One ascent branch and the maximum it claims.

    Attributes:
        saddle_id: Source saddle's id (``"SAD_142"`` / ``"E-SAD_MC7"``).
        branch_index: Index of the branch within the seed's ascent list.
        max_id: Claimed maximum's ``center_id``, or None when nothing qualified.
        status: ``"connected"`` or ``"no_target"``.
        direction_mode: Which ray definition produced this claim.
        cross_track_deg: Perpendicular deviation of the target from the ray,
            signed toward the ray's left (the pole). Negative for a target on
            the other side. None when there is no target.
        along_track_deg: Progress of the target along the ray from its origin.
        separation_deg: Great-circle angle saddle-to-maximum.
        mutual: True when this is the tightest claim on that maximum.
        arc_points: World-frame polyline saddle -> maximum, sampled on the
            great circle. Empty when there is no target.
        saddle_value: Field value at the saddle.
        max_value: Field value at the claimed maximum.
        verified: True when the source seed's ring converged on the fold order
            the reduction reports (``index_agrees``).
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
    direction_mode: DirectionMode = "chord",
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
    maxima: list[BasinCenter],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    max_arc_pitch: float = MAX_ARC_PITCH,
    direction_mode: DirectionMode = "chord",
) -> list[MaxConnection]:
    """Claim a maximum for every ascent branch of every seed.

    Args:
        seeds: Saddle launch seeds, in the centred atom frame. Both the verified
            and the provisional (``k_mismatch``) sets are usually passed, so the
            connection table matches what the scene draws; each record carries
            ``verified`` so the two stay distinguishable.
        maxima: Candidate targets. Entries whose polarity is not ``"maximum"``
            are ignored, so a full basin-center list may be passed.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to read the cap.
        center: World-frame nucleus. Required when ``maxima`` carry world
            positions (they do, coming from ``extract_basin_centers``); the
            output arcs are then emitted in the world frame too.
        max_arc_pitch: Arc-length cap in mesh-pitch units.
        direction_mode: ``"chord"`` or ``"port_tangent"``.

    Returns:
        One :class:`MaxConnection` per (seed, ascent branch), in input order.
        Deterministic: the winner depends only on the candidate SET, never on
        its order.
    """
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(
            f"direction_mode {direction_mode!r} must be one of {DIRECTION_MODES}"
        )
    targets = [m for m in maxima if m.polarity == "maximum"]
    max_arc_rad = max_arc_pitch * delta_theta_mesh_rad

    prepared: list[tuple[BasinCenter, npt.NDArray[np.float64]]] = []
    for m in targets:
        local = (
            np.asarray(to_local(m.position, center), dtype=np.float64)
            if center is not None
            else np.asarray(m.position, dtype=np.float64)
        )
        u = unit(local)
        if u is None:
            continue
        prepared.append((m, u))
    # Order the candidate list by id so a tie in the metrics resolves the same
    # way regardless of how the caller assembled the list.
    prepared.sort(key=lambda pair: str(pair[0].center_id))

    out: list[MaxConnection] = []
    for seed in seeds:
        centre = np.asarray(seed.saddle_position, dtype=np.float64)
        centre_unit = unit(centre)
        for branch in seed.ascent_branches:
            ray = connection_ray(seed, branch, direction_mode, sphere_radius)
            if ray is None:
                out.append(_no_target(seed, branch, direction_mode))
                continue

            best: tuple[float, float, int] | None = None
            best_idx = -1
            for idx, (_m, t_unit) in enumerate(prepared):
                along, cross = target_frame_angles(t_unit, ray)
                if along <= 0.0 or along > max_arc_rad:
                    continue
                key = (abs(cross), along, idx)
                if best is None or key < best:
                    best, best_idx = key, idx

            if best is None:
                out.append(_no_target(seed, branch, direction_mode))
                continue

            won, t_unit = prepared[best_idx]
            _cross, along = best[0], best[1]
            out.append(
                MaxConnection(
                    saddle_id=seed.saddle_id,
                    branch_index=branch.sector_index,
                    max_id=str(won.center_id),
                    status=_STATUS_CONNECTED,
                    direction_mode=direction_mode,
                    cross_track_deg=math.degrees(target_frame_angles(t_unit, ray)[1]),
                    along_track_deg=math.degrees(along),
                    separation_deg=math.degrees(
                        angular_distance(
                            centre_unit if centre_unit is not None else ray.origin_unit,
                            t_unit,
                        )
                    ),
                    mutual=False,
                    arc_points=_arc_between(
                        ray.origin_unit, t_unit, sphere_radius, center
                    ),
                    saddle_value=float(seed.saddle_value),
                    max_value=float(won.value),
                    verified=bool(seed.index_agrees),
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

    A maximum is "claimed by" every branch that selected it. The branch with the
    smallest cross-track is the one the maximum points back at, which is the
    reciprocal reading of the connection. Exactly one connection per maximum is
    flagged, and only connected ones are eligible.

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
        out.append(
            MaxConnection(
                saddle_id=conn.saddle_id,
                branch_index=conn.branch_index,
                max_id=conn.max_id,
                status=conn.status,
                direction_mode=conn.direction_mode,
                cross_track_deg=conn.cross_track_deg,
                along_track_deg=conn.along_track_deg,
                separation_deg=conn.separation_deg,
                mutual=is_mutual,
                arc_points=conn.arc_points,
                saddle_value=conn.saddle_value,
                max_value=conn.max_value,
                verified=conn.verified,
            )
        )
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
        "n_no_target": len(connections) - len(connected),
        "n_saddles_connected": len({c.saddle_id for c in connected}),
        "n_maxima_connected": len({c.max_id for c in connected}),
        "n_mutual": sum(1 for c in connections if c.mutual),
        "n_unverified": sum(1 for c in connected if not c.verified),
        "mean_cross_track_deg": sum(cross) / len(cross) if cross else 0.0,
        "max_cross_track_deg": max(cross) if cross else 0.0,
        "mean_separation_deg": sum(sep) / len(sep) if sep else 0.0,
        "max_separation_deg": max(sep) if sep else 0.0,
    }
