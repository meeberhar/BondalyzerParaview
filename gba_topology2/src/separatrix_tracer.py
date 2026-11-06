"""Separatrix Tracing Subroutine for gba_topology2 (Basin Locator Step 4).

Step 3 finds a saddle's terminal maximum by extrapolating a great circle from the
launch port and claiming the first maximum in its forward sector. That is exact
when the separatrix is roughly straight -- true on the dense metallic fields,
where every branch on Pd_20K connects -- and it fails when the ridge curves. On
ethene3 C1 (shape index) the single maximum sits 123.6 deg from the saddle, at
cross-track +/-44.9 deg and NEGATIVE along-track: it is behind the launch ray, so
no cone and no cap can reach it. The separatrix has to wrap around the sphere.

The method
----------
Follow the field. From the launch port, step along the surface gradient of the
condensed field -- the definition of a ridge line -- and stop when the path enters
a maximum's excision disc. The gradient is evaluated by central differences of the
same inverse-distance-weighted interpolant the excision rings are profiled with, so
the two stages measure the same field.

Why start at the port and not the saddle centre
-----------------------------------------------
At a saddle the gradient vanishes: that is what makes it a critical point. A
gradient-ascent step from the centre has no direction to take and the first step
would be pure noise. The port is on the excision boundary, outside the degenerate
core, where the gradient is well defined and already points uphill -- it is the
ridge gateway by construction. This is the same reason Step 3 launches from the
port (Entry 044): the disc inside the boundary is the region whose field is
meaningless.

Why the traced path is drawn, not replaced by a geodesic
--------------------------------------------------------
Step 3's connector is the single great circle between the two endpoints it found,
because its search was a straight line and the connector should be faithful to the
search. Here the search IS a curve, and the curve is the finding: the wrapping
around the sphere is the phenomenon. Replacing it with the geodesic between its
endpoints would draw a chord straight across the atom and erase exactly the
structure that made the geodesic rule fail.

Termination and guards
----------------------
- ARRIVE: the path enters a maximum's excision disc. The endpoint is that
  maximum's centre, matching the geodesic connector's endpoint convention.
- STALL: the field stops rising. Near a maximum the gradient vanishes, so a path
  that reaches a flat plateau without entering a disc is genuinely unresolved --
  a degenerate ridge, not a bug.
- BUDGET: a cap on cumulative arc length. A separatrix on a closed sphere can
  circle indefinitely; the cap bounds the work and reports the overrun.

Why the ascent TRANSITS an excision region
------------------------------------------
An excision boundary marks the region whose field is not trustworthy -- that is
the entire reason it exists, and the reason Step 3 launches from the boundary
rather than the centre (Entry 044). So an ascent must not be ALLOWED TO END
inside one: a termination there is a measurement taken in the noise, not a
finding about the field.

On ethene3 C1 (shape index) all four saddle ports do exactly that. The climb
leaves the port, runs out across the plateau, and terminates inside the excision
disc of MC1 or MC2 -- the two pole clusters, which are annihilated (net chi 0)
and so reduce to NOTHING: they are not in the critical-point pool, and their
interiors are the noisiest ground on the atom. Measured, the field along the
great circle from MC2's centre toward the spike rises monotonically
0.0059 -> 0.0063 -> 0.0073 -> 0.0094 -> 0.0141 -> 0.0372 -> 0.4001: the ridge is
there, and the ascent simply stopped in the noise before it could read it.

The rule: when the ascent terminates inside an excision region, continue in a
straight line THROUGH it -- along the great circle through the region's centre,
for twice the region's radius, which lands on the far rim -- and resume the
climb from there. The transit is deliberately not a gradient step: inside the
boundary the gradient IS the noise the boundary was drawn to exclude, so the only
defensible direction is the one the path already had, aimed at the nearest exit.
The field on the far side is trustworthy, and the ascent resumes there.

The region the port starts inside is excluded, and so is every region already
transited. The first is the saddle's own neighbourhood -- the launch port is ON
its excision boundary by construction, so without the exclusion the ascent would
transit at step 0 and never climb. The second stops it oscillating across the
same disc.

Measured on ethene3 shape index: the two branches that transit MC2 (E-SAD_MS3
br1 and E-SAD_MS4 br1 -- mirror partners, and they behave identically, which is
the free oracle) resume and ARRIVE at MAX_4864. The two that transit MC1 exit on
the far side of the spike's basin and correctly do not reach it: the field there
genuinely does not drain to MAX_4864. The rule resolves the cases the noise was
blocking and leaves the real ones alone.

Decoupling: this module does NOT import ``max_connection``. It consumes any object
satisfying :class:`TraceTarget`, which ``ConnectionTarget`` already does, and takes
the pool positions as a pre-mapped centred-frame dict. That keeps the dependency
one-way, so ``max_connection`` can import this module, and it keeps the frame
conversion in one place at the call site.

Frame contract: every position the tracer integrates is in the CENTRED atom frame,
where the nucleus is the origin and the outward normal of a surface point p is
p / ||p||. The pool arrives in the world frame (it is built from the viewer's UI
dicts), so the caller maps it through ``centred_positions`` first -- the same
``to_local`` Step 3 applies internally. Only the returned polyline is mapped back
out with ``to_global`` so the viewer can draw it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

if TYPE_CHECKING:
    # Type-only: max_connection imports this module, so a runtime import here
    # would be a cycle.
    from gba_topology2.src.max_connection import ConnectionTarget

__all__ = [
    "MAX_TRACE_ARC_PITCH",
    "STALL_PATIENCE",
    "TRACE_MAX_TRANSITS",
    "TRACE_STEP_PITCH",
    "ExcisionRegion",
    "ExcisionSource",
    "TraceOutcome",
    "TraceTarget",
    "centred_targets",
    "excision_regions",
    "trace_uphill",
]

# Ascent step, in mesh-pitch units. One pitch is the finest structure the mesh can
# represent, so a smaller step integrates detail the data does not contain and a
# larger one cuts corners on the ridge. 1.0 pitch is 1.6 deg on Pd_20K and 2.2 deg
# on ethene3.
TRACE_STEP_PITCH = 1.0

# Cumulative arc-length budget, in mesh-pitch units. 400 pitches is ~891 deg on
# ethene3, comfortably more than a great circle, so a path that exhausts it is
# spiralling and the answer is "unresolved", not "raise the budget".
MAX_TRACE_ARC_PITCH = 400.0

# Consecutive steps that must fail to raise the field before the ascent is called
# stalled. A single flat step is normal on a discretised ridge; a run of them means
# the path has found a plateau that is not a maximum.
STALL_PATIENCE = 6

# Progress, as a fraction of the field's global span, that counts as "rising".
_MIN_PROGRESS_FRACTION = 1e-9

# How many excision regions one ascent may cross. Four is generous: ethene3
# shape index needs exactly one per branch, and a path that has crossed four
# regions is wandering, not climbing. The bound exists so a degenerate field
# cannot make the tracer loop between two discs forever.
TRACE_MAX_TRANSITS = 4


@dataclass(frozen=True)
class ExcisionRegion:
    """A noisy disc the ascent may cross but must not end inside.

    Attributes:
        region_id: The owning cluster's id, for the record and for the
            already-transited set.
        centre_unit: Region centre, CENTRED unit radial -- the frame the ascent
            integrates in.
        radius_rad: Angular radius. The fitted boundary's SEMI-MAJOR, so the
            transit clears the whole region on any bearing, not just the bearings
            where the ellipse is narrow.
    """

    region_id: str
    centre_unit: npt.NDArray[np.float64]
    radius_rad: float


@runtime_checkable
class ExcisionSource(Protocol):
    """The cluster shape :func:`excision_regions` needs.

    ``MicroCluster`` satisfies it. Taking the protocol rather than the class keeps
    this module free of the data model, the same decoupling as :class:`TraceTarget`.
    """

    @property
    def cluster_id(self) -> str: ...

    @property
    def centroid(
        self,
    ) -> tuple[float, float, float] | npt.NDArray[np.float64]: ...

    @property
    def angular_radius_rad(self) -> float: ...


def excision_regions(
    sources: Sequence[ExcisionSource],
    center: tuple[float, float, float] | None = None,
) -> list[ExcisionRegion]:
    """Build the transit regions from the micro-cluster excision boundaries.

    Every cluster contributes a region, including the clusters the reduction
    annihilated. That is the point: MC1 and MC2 on ethene3 shape index are
    annihilated (net chi 0), so they are absent from the critical-point pool and
    invisible to the arrival test, yet their excision discs are exactly the noisy
    ground the ascent must not stop in. The pool and the regions are different
    questions -- "what can I connect to" and "where may I not end".

    Args:
        sources: Micro-clusters, CENTRED positions (the detector's frame).
        center: World-frame nucleus, for a source that carries world positions.
            None means the positions are already centred.

    Returns:
        Regions with unit-radial centres, degenerate clusters dropped.
    """
    out: list[ExcisionRegion] = []
    for cl in sources:
        pos = np.asarray(cl.centroid, dtype=np.float64)
        local = (
            pos
            if center is None
            else np.asarray(to_local(pos, center), dtype=np.float64)
        )
        u = unit(local)
        radius = float(cl.angular_radius_rad)
        if u is None or radius <= 0.0:
            continue
        out.append(
            ExcisionRegion(
                region_id=str(cl.cluster_id), centre_unit=u, radius_rad=radius
            )
        )
    return out


@runtime_checkable
class TraceTarget(Protocol):
    """The pool entry shape the tracer needs.

    ``ConnectionTarget`` satisfies this. The tracer knows nothing more about it, so
    the two modules have a one-way dependency.
    """

    @property
    def cp_id(self) -> str: ...

    @property
    def polarity(self) -> str: ...

    @property
    def position(self) -> npt.NDArray[np.float64]: ...

    def radius_toward(self, target_unit: npt.NDArray[np.float64]) -> float: ...


def centred_targets(
    targets: Sequence[ConnectionTarget],
    center: tuple[float, float, float] | None,
) -> list[ConnectionTarget]:
    """Map a pool into the centred frame, as unit radial positions.

    The pool is built in the world frame because that is what the viewer's UI dicts
    carry. The tracer integrates on the centred sphere, where ``p / ||p||`` is the
    outward normal, and it calls ``radius_toward``, which measures from the target's
    OWN position -- so the conversion must happen here, before the tracer sees the
    targets, or the ellipse is evaluated from a nucleus-offset centre and every
    arrival test is wrong. This is the recurring frame trap, which Pd_20K hides at
    the origin and ethene3 exposes at 1.2618 A.

    Args:
        targets: The pool, in the world frame.
        center: World-frame nucleus. None means the pool is already centred.

    Returns:
        New targets with unit centred positions, degenerate entries dropped.
    """
    out: list[ConnectionTarget] = []
    for t in targets:
        pos = np.asarray(t.position, dtype=np.float64)
        local = (
            pos
            if center is None
            else np.asarray(to_local(pos, center), dtype=np.float64)
        )
        u = unit(local)
        if u is None:
            continue
        out.append(replace(t, position=u))
    return out


@dataclass(frozen=True)
class TraceOutcome:
    """The result of one uphill gradient ascent.

    Attributes:
        status: ``"arrived"`` (entered a maximum's disc), ``"stalled"`` (the field
            stopped rising), ``"budget"`` (arc-length cap exhausted), ``"no_start"``
            (the port is degenerate), or ``"unreachable"`` (the maximum is not in
            the centred pool map).
        max_id: The reached maximum's id, set only when ``status`` is
            ``"arrived"``.
        path: World-frame polyline of the ascent, start first.
        arc_length_deg: Cumulative arc length travelled.
        n_steps: Number of integration steps taken.
        start_value: Field value at the launch port.
        end_value: Field value at the last sample.
        visited_ids: Ids of every non-maximum critical point whose disc the path
            entered, in first-visit order. A ridge line that runs through a
            minimum's basin on the way up is a finding worth seeing, not
            something to hide.
        transited_ids: Ids of the excision REGIONS the path crossed through, in
            crossing order. Set only when the caller supplies regions; a transit
            is a straight line drawn over ground the field cannot speak for, and
            the record says which ground that was.
    """

    status: str
    max_id: str | None
    path: tuple[tuple[float, float, float], ...]
    arc_length_deg: float
    n_steps: int
    start_value: float
    end_value: float
    visited_ids: tuple[str, ...] = field(default_factory=tuple)
    transited_ids: tuple[str, ...] = field(default_factory=tuple)


def _tangent_frame(
    p_unit: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """An orthonormal tangent basis at the unit radial ``p_unit``.

    The reference axis is picked by the smaller component so the cross product
    never degenerates, and the choice is a pure function of ``p_unit``, which keeps
    the integration deterministic.

    Args:
        p_unit: Unit radial vector.

    Returns:
        Tuple of (t1, t2), unit tangents with ``t2 = p x t1``.
    """
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(p_unit[2])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = unit(np.cross(ref, p_unit))
    if t1 is None:  # pragma: no cover - ref is never parallel to p_unit
        t1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t2 = unit(np.cross(p_unit, t1))
    if t2 is None:  # pragma: no cover - t1 is perpendicular to p_unit
        t2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return t1, t2


def _step(
    p_unit: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    angle_rad: float,
) -> npt.NDArray[np.float64]:
    """Advance ``angle_rad`` along the great circle from ``p_unit``.

    Re-normalised so accumulated float drift cannot move the path off the sphere.
    """
    moved = unit(
        np.asarray(rotate_towards(p_unit, direction, angle_rad, 1.0), dtype=np.float64)
    )
    return moved if moved is not None else p_unit


def trace_uphill(
    start_position: tuple[float, float, float] | npt.NDArray[np.float64],
    targets: Sequence[TraceTarget],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    step_pitch: float = TRACE_STEP_PITCH,
    max_arc_pitch: float = MAX_TRACE_ARC_PITCH,
    stall_patience: int = STALL_PATIENCE,
    excision_regions: Sequence[ExcisionRegion] = (),
) -> TraceOutcome:
    """Climb the field from a launch port until a maximum is reached.

    Args:
        start_position: The launch port, in the centred atom frame. Typically
            ``LaunchBranch.launch_position``.
        targets: The pool, in the CENTRED frame with unit-radial positions -- build
            with :func:`centred_targets`. Only entries whose polarity is
            ``"maximum"`` are arrival tests; the rest are recorded in
            ``visited_ids`` when the path crosses their disc.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on those vertices.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to read the step and
            the budget.
        center: World-frame nucleus, used to emit the path in world coordinates.
        find_k_nearest: Spatial index over ``mesh_pts``. Strongly recommended:
            without it every sample scans all N vertices.
        step_pitch: Ascent step in mesh-pitch units.
        max_arc_pitch: Cumulative arc-length budget in mesh-pitch units.
        stall_patience: Consecutive non-rising steps before the ascent stalls.
        excision_regions: Noisy discs the ascent may cross but must not END
            inside -- the micro-cluster excision boundaries. Omit (the default)
            to keep the pure ascent: a branch that terminates on flat ground
            stalls there, which is the Step 4 contract the tests pin. Supply
            them to get the transit rule described in the module docstring.

    Returns:
        A :class:`TraceOutcome`. ``status == "arrived"`` sets ``max_id`` to the
        maximum whose disc the path entered.
    """
    maxima = [t for t in targets if t.polarity == "maximum"]
    others = [t for t in targets if t.polarity != "maximum"]
    step_rad = step_pitch * delta_theta_mesh_rad
    budget_rad = max_arc_pitch * delta_theta_mesh_rad

    def _emit(p_local: npt.NDArray[np.float64]) -> tuple[float, float, float]:
        # center is None only for an origin-centred synthetic mesh, where the two
        # frames coincide and the position is emitted verbatim.
        if center is None:
            return (float(p_local[0]), float(p_local[1]), float(p_local[2]))
        return to_global(
            (float(p_local[0]), float(p_local[1]), float(p_local[2])), center
        )

    def _f(q_unit: npt.NDArray[np.float64]) -> float:
        q = np.array([q_unit * sphere_radius], dtype=np.float64)
        if find_k_nearest is not None:
            return float(
                interpolate_scalar_at_points_local(q, mesh_pts, f_vals, find_k_nearest)[
                    0
                ]
            )
        return float(interpolate_scalar_at_points(q, mesh_pts, f_vals)[0])

    def _outcome(
        status: str,
        max_id: str | None,
        path: list[tuple[float, float, float]],
        travelled: float,
        steps: int,
        f_start: float,
        f_here: float,
        visited: list[str],
        transited: list[str],
    ) -> TraceOutcome:
        return TraceOutcome(
            status=status,
            max_id=max_id,
            path=tuple(path),
            arc_length_deg=math.degrees(travelled),
            n_steps=steps,
            start_value=f_start,
            end_value=f_here,
            visited_ids=tuple(visited),
            transited_ids=tuple(transited),
        )

    def _region_at(
        q: npt.NDArray[np.float64],
        excluded: set[str],
    ) -> ExcisionRegion | None:
        """The first region the ascent may not end inside that contains ``q``.

        ``excluded`` carries the regions the port started inside and the ones
        already transited; both are off limits, the first because the launch port
        sits ON the saddle's own boundary and the second so the path cannot
        oscillate across the same disc.
        """
        for r in excision_regions:
            if r.region_id in excluded:
                continue
            if (
                r.radius_rad > 0.0
                and angular_distance(q, r.centre_unit) <= r.radius_rad
            ):
                return r
        return None

    p_unit = unit(np.asarray(start_position, dtype=np.float64))
    if p_unit is None:
        return _outcome(
            "no_start", None, [], 0.0, 0, float("nan"), float("nan"), [], []
        )

    path: list[tuple[float, float, float]] = [_emit(p_unit * sphere_radius)]
    # The centred unit radials parallel to `path`. The transit geometry needs
    # positions on the unit sphere, and `path` carries world-frame metres.
    units: list[npt.NDArray[np.float64]] = [p_unit.copy()]
    f_start = _f(p_unit)
    f_here = f_start
    span = float(np.ptp(f_vals))
    min_progress = _MIN_PROGRESS_FRACTION * (span if span > 0.0 else 1.0)

    visited: list[str] = []
    seen: set[str] = set()
    transited: list[str] = []
    travelled = 0.0
    steps = 0

    # The regions the port starts inside are never transited. The launch port is
    # ON the saddle's excision boundary by construction, and a saddle's boundary
    # overlaps the clusters it was reduced from, so without this the ascent would
    # transit at step 0 and never climb.
    excluded: set[str] = set()
    for r in excision_regions:
        if (
            r.radius_rad > 0.0
            and angular_distance(p_unit, r.centre_unit) <= r.radius_rad
        ):
            excluded.add(r.region_id)

    while True:
        flat_run = 0
        terminated = False
        status = "budget"
        max_id: str | None = None
        # The budget is PER SEGMENT, not per trace. A branch that spirals out its
        # allowance before reaching a region, transits, and then has nothing left
        # to climb with would report the transit as useless -- measured on ethene3
        # shape index, all four branches exhausted a shared budget this way and
        # arrived nowhere. The transit COUNT (TRACE_MAX_TRANSITS) bounds the total
        # work; the budget bounds each climb, which is what it means to be a cap
        # on how far an ascent may wander before the answer is "unresolved".
        segment_start = travelled

        while travelled - segment_start < budget_rad and not terminated:
            t1, t2 = _tangent_frame(p_unit)
            h = step_rad
            # Central differences on the surface. The common factor 1 / (2 R h)
            # is a positive scalar and only the direction is needed, so it is
            # dropped.
            g1 = _f(_step(p_unit, t1, h)) - _f(_step(p_unit, t1, -h))
            g2 = _f(_step(p_unit, t2, h)) - _f(_step(p_unit, t2, -h))
            direction = unit(g1 * t1 + g2 * t2)
            if direction is None:
                status = "stalled"
                break

            p_unit = _step(p_unit, direction, step_rad)
            path.append(_emit(p_unit * sphere_radius))
            units.append(p_unit.copy())
            travelled += step_rad
            steps += 1

            f_next = _f(p_unit)
            flat_run = flat_run + 1 if f_next - f_here < min_progress else 0
            f_here = f_next

            # Arrival is tested before the stall, because the gradient vanishes
            # at a maximum: reaching the top and going flat is success.
            for m in maxima:
                rho = m.radius_toward(p_unit)
                if rho > 0.0 and angular_distance(p_unit, m.position) <= rho:
                    status = "arrived"
                    max_id = str(m.cp_id)
                    terminated = True
                    break
            if terminated:
                break

            for t in others:
                if t.cp_id in seen:
                    continue
                rho = t.radius_toward(p_unit)
                if rho > 0.0 and angular_distance(p_unit, t.position) <= rho:
                    seen.add(t.cp_id)
                    visited.append(str(t.cp_id))

            if flat_run >= stall_patience:
                status = "stalled"
                break

        if status == "arrived" or not excision_regions:
            return _outcome(
                status,
                max_id,
                path,
                travelled,
                steps,
                f_start,
                f_here,
                visited,
                transited,
            )

        region = _region_at(p_unit, excluded)
        if region is None or len(transited) >= TRACE_MAX_TRANSITS:
            # The ascent ended on trustworthy ground: a real stall, a real
            # budget overrun, or the transit allowance spent. Report it as is.
            return _outcome(
                status,
                None,
                path,
                travelled,
                steps,
                f_start,
                f_here,
                visited,
                transited,
            )

        # TRANSIT. Walk the great circle the path was on, from the last point
        # outside the region, until the path is clear of every region it has not
        # already crossed. Deliberately not a gradient step: inside the boundary
        # the gradient is the noise the boundary exists to exclude.
        j = len(units) - 1
        while j > 0 and _region_at(units[j], excluded) is not None:
            j -= 1
        anchor = units[j]
        nxt = units[j + 1] if j + 1 < len(units) else p_unit
        heading = nxt - anchor
        heading = heading - float(np.dot(heading, anchor)) * anchor
        heading_unit = unit(heading)
        if heading_unit is None:  # pragma: no cover - consecutive samples differ
            return _outcome(
                status,
                None,
                path,
                travelled,
                steps,
                f_start,
                f_here,
                visited,
                transited,
            )

        q = anchor
        crossed = 0
        while crossed < 800:
            q = _step(q, heading_unit, step_rad)
            crossed += 1
            path.append(_emit(q * sphere_radius))
            units.append(q.copy())
            if _region_at(q, excluded) is None:
                break
        travelled += crossed * step_rad
        steps += crossed
        p_unit = q
        f_here = _f(p_unit)
        excluded.add(region.region_id)
        transited.append(region.region_id)
        # No budget check here: the budget is per climb segment (see above) and
        # the transit count bounds the whole trace, so the loop simply starts the
        # next climb from the far rim with a fresh allowance.
