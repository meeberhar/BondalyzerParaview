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
    "TRACE_STEP_PITCH",
    "TraceOutcome",
    "TraceTarget",
    "centred_targets",
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
    """

    status: str
    max_id: str | None
    path: tuple[tuple[float, float, float], ...]
    arc_length_deg: float
    n_steps: int
    start_value: float
    end_value: float
    visited_ids: tuple[str, ...] = field(default_factory=tuple)


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
        )

    p_unit = unit(np.asarray(start_position, dtype=np.float64))
    if p_unit is None:
        return _outcome("no_start", None, [], 0.0, 0, float("nan"), float("nan"), [])

    path: list[tuple[float, float, float]] = [_emit(p_unit * sphere_radius)]
    f_start = _f(p_unit)
    f_here = f_start
    span = float(np.ptp(f_vals))
    min_progress = _MIN_PROGRESS_FRACTION * (span if span > 0.0 else 1.0)

    visited: list[str] = []
    seen: set[str] = set()
    travelled = 0.0
    flat_run = 0
    steps = 0

    while travelled < budget_rad:
        t1, t2 = _tangent_frame(p_unit)
        h = step_rad
        # Central differences on the surface. The common factor 1 / (2 R h) is a
        # positive scalar and only the direction is needed, so it is dropped.
        g1 = _f(_step(p_unit, t1, h)) - _f(_step(p_unit, t1, -h))
        g2 = _f(_step(p_unit, t2, h)) - _f(_step(p_unit, t2, -h))
        direction = unit(g1 * t1 + g2 * t2)
        if direction is None:
            return _outcome(
                "stalled", None, path, travelled, steps, f_start, f_here, visited
            )

        p_unit = _step(p_unit, direction, step_rad)
        path.append(_emit(p_unit * sphere_radius))
        travelled += step_rad
        steps += 1

        f_next = _f(p_unit)
        flat_run = flat_run + 1 if f_next - f_here < min_progress else 0
        f_here = f_next

        # Arrival is tested before the stall, because the gradient vanishes at a
        # maximum: reaching the top and going flat is success, not failure.
        for m in maxima:
            rho = m.radius_toward(p_unit)
            if rho > 0.0 and angular_distance(p_unit, m.position) <= rho:
                return _outcome(
                    "arrived",
                    str(m.cp_id),
                    path,
                    travelled,
                    steps,
                    f_start,
                    f_here,
                    visited,
                )

        for t in others:
            if t.cp_id in seen:
                continue
            rho = t.radius_toward(p_unit)
            if rho > 0.0 and angular_distance(p_unit, t.position) <= rho:
                seen.add(t.cp_id)
                visited.append(str(t.cp_id))

        if flat_run >= stall_patience:
            return _outcome(
                "stalled", None, path, travelled, steps, f_start, f_here, visited
            )

    return _outcome("budget", None, path, travelled, steps, f_start, f_here, visited)
