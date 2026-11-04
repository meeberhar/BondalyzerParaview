"""Connector Refinement and Topology Audit (Basin Locator Step 5).

Step 3 draws a connector as a great circle and Step 4 finishes the branches the
straight ray cannot resolve by climbing the field. This module asks the question
both of them leave open: *is the connector the field agrees with?*

THE RULE
--------
Every connector is a claim -- "this saddle's ascent drains to this maximum". The
claim is tested by following the ridge: integrate the surface gradient uphill from
the branch's port and see which maximum's excision disc the path enters. Three
verdicts:

- ``confirmed`` -- the ridge enters the DISC of the claimed maximum. The great
  circle is a faithful summary of the separatrix, and drawing it is legitimate.
- ``rerouted`` -- the ridge enters a DIFFERENT pool maximum's disc (including a
  maximum the branch never claimed, e.g. a ray ``no_target`` the climb resolves).
  The ray's claim was wrong. The gradient is ground truth: the connection is
  corrected.
- ``dissolved`` -- the ridge stalls on non-maximum ground. There is no visible
  maximum at the end of this ascent, which is not a failure of the connector but
  a statement about the critical point set.

A branch whose connector is ALREADY a traced ridge (``method == "gradient"``,
Step 4) is confirmed by construction -- the ridge is the connector -- and an
occluded branch is a resolved verdict about the ray, not about the ascent.
Neither is re-tested.

THE AUDIT
---------
A dissolved branch is evidence about the reduction, and the evidence is a
position: the stall point. If a RAW maximum that the reduction hid sits within
one mesh pitch of it, the reduction over-merged -- the hidden maximum attracts
the ascent of the region around the saddle, which is the definition of a basin,
so it is a basin centre and must be visible. It is PROMOTED back into the pool
and the dissolved branches are re-resolved against the enlarged pool. If no
hidden maximum sits near the stall point, the ascent died on genuinely flat
ground: the basins touch at a measure-zero point. That is CORNER-SHARING, a real
adjacency, not a missing connection.

This is the resolution of the ethene3 shape-index case. The reduction bridges the
bump MC10 (f = 0.0058) into the saddle corridor MS1, leaving one visible maximum,
the spike MAX_4864 (f = 0.5799). All four saddle ports climb to the bump and stop:
the spike lies beyond a saddle, and gradient ascent is strictly uphill, so it is
unreachable. The topology said "pure maximum", the gradient says "max, saddle,
max", and the gradient is right -- the spike sits along a mirror plane and the
bumps are their own basins. Promoting MC10 makes the connection table consistent
with the basins.

WHY THIS IS CHEAP
-----------------
``trace_uphill`` integrates the FIELD; the pool enters only in the arrival test.
So refinement adds no new machinery and the audit needs no re-integration of
anything already traced: promotion enlarges the pool, and the enlarged pool is
what the (deterministic) ascent is tested against. The whole pass is a
re-resolution, which is why it is an option and not a rebuild of Step 3/4.

Frame contract: seeds carry CENTRED positions, the pool carries WORLD positions,
and the audit's stall points are kept as CENTRED UNIT radials -- the frame the
ascent integrates in, and the only frame in which an angular distance to a raw
maximum is meaningful. Every position handed to the viewer is mapped out with
``to_global``. Omit ``center`` only for origin-centred synthetic meshes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.excision_boundary import KNearestFinder
from gba_topology2.src.max_connection import ConnectionTarget, MaxConnection
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.separatrix_tracer import (
    TraceTarget,
    centred_targets,
    trace_uphill,
)
from gba_topology2.src.sphere_frame import angular_distance, to_global, to_local, unit

__all__ = [
    "CAUSE_CORNER",
    "CAUSE_NO_EVIDENCE",
    "CAUSE_PROMOTED",
    "REFINEMENT_VERDICTS",
    "REFINE_TOL_PITCH",
    "VERDICT_CONFIRMED",
    "VERDICT_DISSOLVED",
    "VERDICT_REROUTED",
    "VERDICT_RESOLVED",
    "VERDICT_RIDGE",
    "Promotion",
    "RefinementResult",
    "RefinementVerdict",
    "apply_refinement",
    "audit_dissolved",
    "promotion_targets",
    "refine_and_audit",
    "refine_connections",
    "summarize_refinement",
]

# Stall-to-hidden-maximum distance that counts as "the reduction over-merged", in
# mesh-pitch units. The bound is principled, not tuned: the stall is the LAST
# SAMPLE of an ascent whose step is one pitch, so it can sit up to a step short of
# the disc edge, and the disc a raw maximum carries is one pitch (Entry 046).
# Step + disc = 2 pitches. Measured on ethene3 shape index, the four stalls sit
# 0.6-3.4 deg from a hidden maximum (pitch 2.23 deg): at 1 pitch the mirror
# partners fall out of the tolerance and the promoted set is asymmetric; at 2 all
# four promote and the set {MAX_5, MAX_10052} is exactly mirror-symmetric.
REFINE_TOL_PITCH = 2.0

# The three verdicts, plus the two cases the pass declines to judge.
VERDICT_CONFIRMED = "confirmed"
VERDICT_REROUTED = "rerouted"
VERDICT_DISSOLVED = "dissolved"
VERDICT_RIDGE = "ridge"  # already a traced separatrix (Step 4): nothing to test
VERDICT_RESOLVED = "resolved"  # occluded: a verdict about the ray, not the ascent
REFINEMENT_VERDICTS: tuple[str, ...] = (
    VERDICT_CONFIRMED,
    VERDICT_REROUTED,
    VERDICT_DISSOLVED,
    VERDICT_RIDGE,
    VERDICT_RESOLVED,
)

# How the audit explains a stall.
CAUSE_PROMOTED = "promoted"  # a hidden maximum sits at the stall: over-merge
CAUSE_CORNER = "corner"  # flat ground: the basins touch at a point
CAUSE_NO_EVIDENCE = "no_evidence"  # no hidden maximum exists to consider


@dataclass(frozen=True)
class RefinementVerdict:
    """What the ridge says about one connector.

    Attributes:
        saddle_id: Source saddle's id.
        branch_index: Branch index within the seed's ascent list.
        verdict: One of :data:`REFINEMENT_VERDICTS`.
        claimed_max_id: The maximum the connector names, if any.
        ridge_max_id: The maximum the ridge actually enters, if any.
        stall_unit: Centred UNIT radial of the point where the ascent ended. The
            audit's evidence; None when the ascent arrived.
        ridge_arc_deg: Arc length the ascent travelled.
        endpoint_deviation_deg: Angle between the claimed maximum's centre and the
            ridge's terminal point. The size of the lie in the straight connector.
        ridge_path: World-frame polyline of the ascent, start first. The connector
        a REROUTED verdict installs; None for the branches that were not traced.
    """

    saddle_id: str
    branch_index: int
    verdict: str
    claimed_max_id: str | None = None
    ridge_max_id: str | None = None
    stall_unit: tuple[float, float, float] | None = None
    ridge_arc_deg: float | None = None
    endpoint_deviation_deg: float | None = None
    ridge_path: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class Promotion:
    """A raw maximum the reduction hid, restored by the audit.

    Attributes:
        vertex_id: Mesh vertex the raw maximum sits on.
        cp_id: Basin-center-style id (``"MAX_<vertex>"``), so a promoted entry
            joins the pool in the id space the viewer already uses.
        position: CENTRED position (the raw CriticalPoint's frame). Pool entries
            are mapped out with :func:`promotion_targets`.
        value: Field value at the maximum.
        distance_deg: Angle from the motivating stall point.
        source_saddle_id: The branch whose dissolution is the evidence.
        source_branch_index: Its branch index.
    """

    vertex_id: int
    cp_id: str
    position: tuple[float, float, float]
    value: float
    distance_deg: float
    source_saddle_id: str
    source_branch_index: int


@dataclass(frozen=True)
class RefinementResult:
    """The whole pass: verdicts, promotions, and the summary to record.

    Attributes:
        verdicts: One entry per connection, in the input order, AFTER
            re-resolution against the enlarged pool.
        promotions: Raw maxima restored by the audit, de-duplicated by vertex.
        causes: Per-branch audit outcome, keyed by ``(saddle_id, branch_index)``.
        stats: :func:`summarize_refinement` output.
    """

    verdicts: list[RefinementVerdict]
    promotions: list[Promotion]
    causes: dict[tuple[str, int], str]
    stats: dict[str, Any]


def _branch_ports(
    seeds: Sequence[SaddleLaunchSeed],
) -> dict[tuple[str, int], LaunchBranch]:
    """Index every ascent branch by ``(saddle_id, sector_index)``.

    Args:
        seeds: The launch seeds the connections were built from.

    Returns:
        Map from a connection's key to its branch, which carries the port the
        ascent starts from.
    """
    out: dict[tuple[str, int], LaunchBranch] = {}
    for seed in seeds:
        for branch in seed.ascent_branches:
            out[(seed.saddle_id, branch.sector_index)] = branch
    return out


def _centred_unit(
    position: tuple[float, float, float] | npt.NDArray[np.float64],
    center: tuple[float, float, float] | None,
) -> npt.NDArray[np.float64] | None:
    """Map a world-frame position to a centred unit radial.

    Args:
        position: World-frame point (or centred when ``center`` is None).
        center: World-frame nucleus.

    Returns:
        Unit vector, or None when the point sits on the nucleus.
    """
    pos = np.asarray(position, dtype=np.float64)
    local = (
        pos if center is None else np.asarray(to_local(pos, center), dtype=np.float64)
    )
    return unit(local)


def _as_tuple(v: npt.NDArray[np.float64] | None) -> tuple[float, float, float] | None:
    """Narrow an array to a plain tuple for the frozen record."""
    return None if v is None else (float(v[0]), float(v[1]), float(v[2]))


def _trace_verdict(
    saddle_id: str,
    branch_index: int,
    claimed_max_id: str | None,
    branch: LaunchBranch,
    centred_pool: Sequence[TraceTarget],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None,
    find_k_nearest: KNearestFinder | None,
) -> RefinementVerdict:
    """Climb from the port and classify one connector.

    Args:
        saddle_id: Source saddle's id.
        branch_index: Branch index.
        claimed_max_id: The connector's claim, if any.
        branch: The launch branch (the port the ascent starts from).
        centred_pool: The pool in the CENTRED unit-radial frame.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Mesh pitch.
        center: World-frame nucleus.
        find_k_nearest: Spatial index.

    Returns:
        confirmed / rerouted / dissolved, with the stall point when dissolved.
    """
    outcome = trace_uphill(
        start_position=branch.launch_position,
        targets=centred_pool,
        mesh_pts=mesh_pts,
        f_vals=f_vals,
        sphere_radius=sphere_radius,
        delta_theta_mesh_rad=delta_theta_mesh_rad,
        center=center,
        find_k_nearest=find_k_nearest,
    )
    claimed = next((t for t in centred_pool if t.cp_id == claimed_max_id), None)
    terminal = _centred_unit(outcome.path[-1], center) if outcome.path else None
    deviation = (
        math.degrees(angular_distance(claimed.position, terminal))
        if claimed is not None and terminal is not None
        else None
    )
    if outcome.status == "arrived" and outcome.max_id is not None:
        arrived = outcome.max_id == claimed_max_id
        return RefinementVerdict(
            saddle_id=saddle_id,
            branch_index=branch_index,
            verdict=VERDICT_CONFIRMED if arrived else VERDICT_REROUTED,
            claimed_max_id=claimed_max_id,
            ridge_max_id=outcome.max_id,
            ridge_arc_deg=outcome.arc_length_deg,
            endpoint_deviation_deg=deviation,
            ridge_path=outcome.path,
        )
    return RefinementVerdict(
        saddle_id=saddle_id,
        branch_index=branch_index,
        verdict=VERDICT_DISSOLVED,
        claimed_max_id=claimed_max_id,
        ridge_max_id=None,
        stall_unit=_as_tuple(terminal),
        ridge_arc_deg=outcome.arc_length_deg,
        endpoint_deviation_deg=deviation,
        ridge_path=outcome.path,
    )


def refine_connections(
    connections: Sequence[MaxConnection],
    seeds: Sequence[SaddleLaunchSeed],
    pool: Sequence[ConnectionTarget],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
) -> list[RefinementVerdict]:
    """Test every connector's claim by following the ridge.

    Args:
        connections: The Step 3/4 connection table.
        seeds: The launch seeds, for the ports the ascents start from.
        pool: The visible critical points, WORLD frame, with excision discs.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) condensed field values on ``mesh_pts``.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        center: World-frame nucleus.
        find_k_nearest: Spatial index over ``mesh_pts``.

    Returns:
        One :class:`RefinementVerdict` per connection, in the input order. A
        connection with no port (a seed the launch stage dropped) is reported
        ``resolved``: there is nothing to test.
    """
    ports = _branch_ports(seeds)
    centred_pool = centred_targets(pool, center)
    out: list[RefinementVerdict] = []
    for conn in connections:
        branch = ports.get((conn.saddle_id, conn.branch_index))
        if conn.method == "gradient":
            # The connector IS the ridge; claim and ascent are the same object.
            out.append(
                RefinementVerdict(
                    saddle_id=conn.saddle_id,
                    branch_index=conn.branch_index,
                    verdict=VERDICT_RIDGE,
                    claimed_max_id=conn.max_id,
                    ridge_max_id=conn.max_id,
                )
            )
            continue
        if conn.status == "occluded" or branch is None:
            out.append(
                RefinementVerdict(
                    saddle_id=conn.saddle_id,
                    branch_index=conn.branch_index,
                    verdict=VERDICT_RESOLVED,
                    claimed_max_id=conn.max_id,
                )
            )
            continue
        out.append(
            _trace_verdict(
                saddle_id=conn.saddle_id,
                branch_index=conn.branch_index,
                claimed_max_id=conn.max_id,
                branch=branch,
                centred_pool=centred_pool,
                mesh_pts=mesh_pts,
                f_vals=f_vals,
                sphere_radius=sphere_radius,
                delta_theta_mesh_rad=delta_theta_mesh_rad,
                center=center,
                find_k_nearest=find_k_nearest,
            )
        )
    return out


def audit_dissolved(
    stalls: Sequence[tuple[str, int, tuple[float, float, float]]],
    hidden_maxima: Sequence[CriticalPoint],
    delta_theta_mesh_rad: float,
    tol_pitch: float = REFINE_TOL_PITCH,
) -> tuple[list[Promotion], dict[tuple[str, int], str]]:
    """Attribute every stall to a hidden maximum, or call it corner-sharing.

    Args:
        stalls: ``(saddle_id, branch_index, stall_unit)`` triples, the stall in
            the CENTRED unit-radial frame -- the frame the ascent integrates in.
        hidden_maxima: Raw maxima the reduction collapsed, CENTRED positions
            (the Morse detector's frame, which is the stall's frame).
        delta_theta_mesh_rad: Mesh pitch, to read the tolerance.
        tol_pitch: Stall-to-maximum acceptance distance, in mesh pitches.

    Returns:
        ``(promotions, causes)``. Promotions are de-duplicated by vertex, keeping
        the nearest motivating stall. ``causes`` maps each audited stall to
        ``promoted`` (a hidden maximum was restored for it), ``corner`` (the
        stall is on flat ground, the basins touch at a point) or ``no_evidence``
        (no hidden maximum exists to consider).
    """
    tol_rad = tol_pitch * delta_theta_mesh_rad
    promotions: dict[int, Promotion] = {}
    causes: dict[tuple[str, int], str] = {}
    for saddle_id, branch_index, stall in stalls:
        key = (saddle_id, branch_index)
        best: tuple[float, CriticalPoint] | None = None
        for cp in hidden_maxima:
            pos = unit(np.asarray(cp.position, dtype=np.float64))
            if pos is None:
                continue
            d = float(angular_distance(np.asarray(stall, dtype=np.float64), pos))
            if best is None or d < best[0]:
                best = (d, cp)
        if best is None:
            causes[key] = CAUSE_NO_EVIDENCE
            continue
        d, cp = best
        if d > tol_rad:
            # The nearest hidden maximum is too far: the ascent died on flat
            # ground, not at a suppressed basin centre. Corner-sharing.
            causes[key] = CAUSE_CORNER
            continue
        promo = Promotion(
            vertex_id=int(cp.vertex_id),
            cp_id=f"MAX_{cp.vertex_id}",
            position=(
                float(cp.position[0]),
                float(cp.position[1]),
                float(cp.position[2]),
            ),
            value=float(cp.value),
            distance_deg=math.degrees(d),
            source_saddle_id=saddle_id,
            source_branch_index=branch_index,
        )
        prev = promotions.get(promo.vertex_id)
        if prev is None or promo.distance_deg < prev.distance_deg:
            promotions[promo.vertex_id] = promo
        causes[key] = CAUSE_PROMOTED
    return list(promotions.values()), causes


def promotion_targets(
    promotions: Sequence[Promotion],
    angular_radius_rad: float,
    center: tuple[float, float, float] | None = None,
) -> list[ConnectionTarget]:
    """Turn promotions into pool entries.

    The caller passes the disc radius the audit's tolerance implies (see
    :func:`refine_and_audit`): a promotion is justified by a stall within
    ``tol_pitch`` of the hidden maximum, so the basin boundary sits at the same
    radius and the arrival test can reach what the audit restored. Nothing
    downstream can tell a promoted maximum from one the reduction left visible,
    which is the point: the audit corrects the CP set, not the rule.

    Args:
        promotions: The audit's promotions, CENTRED positions.
        angular_radius_rad: Disc radius, in radians.
        center: World-frame nucleus, to map the positions out.

    Returns:
        Pool targets in the world frame, ordered by id.
    """
    out: list[ConnectionTarget] = []
    for p in promotions:
        pos = p.position if center is None else to_global(p.position, center)
        out.append(
            ConnectionTarget(
                cp_id=p.cp_id,
                polarity="maximum",
                position=np.asarray(pos, dtype=np.float64),
                value=p.value,
                angular_radius_rad=angular_radius_rad,
            )
        )
    out.sort(key=lambda t: t.cp_id)
    return out


def refine_and_audit(
    connections: Sequence[MaxConnection],
    seeds: Sequence[SaddleLaunchSeed],
    pool: Sequence[ConnectionTarget],
    hidden_maxima: Sequence[CriticalPoint],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    tol_pitch: float = REFINE_TOL_PITCH,
) -> RefinementResult:
    """Run the whole pass: test, audit, promote, re-resolve.

    The re-resolution climbs AGAIN for the dissolved branches, against the
    enlarged pool. The ascent path is identical -- the field did not change, only
    the arrival test's target set -- so this is a re-verdict, not a re-trace in
    any expensive sense, and it keeps the single source of truth for "where does
    this port drain" inside ``trace_uphill``.

    Args:
        connections: The Step 3/4 connection table.
        seeds: The launch seeds (the ports).
        pool: The visible critical points, world frame.
        hidden_maxima: Raw maxima the reduction collapsed, centred frame.
        mesh_pts: (N, 3) mesh vertices, centred frame.
        f_vals: (N,) field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Mesh pitch.
        center: World-frame nucleus.
        find_k_nearest: Spatial index.
        tol_pitch: The audit's acceptance distance, in pitches.

    Returns:
        The :class:`RefinementResult`. Verdicts are post-re-resolution.
    """
    verdicts = refine_connections(
        connections,
        seeds,
        pool,
        mesh_pts,
        f_vals,
        sphere_radius,
        delta_theta_mesh_rad,
        center,
        find_k_nearest,
    )
    stalls = [
        (v.saddle_id, v.branch_index, v.stall_unit)
        for v in verdicts
        if v.verdict == VERDICT_DISSOLVED and v.stall_unit is not None
    ]
    promotions, causes = audit_dissolved(
        stalls, hidden_maxima, delta_theta_mesh_rad, tol_pitch
    )
    if promotions:
        # The promoted disc is the AUDIT TOLERANCE, not one pitch. The promotion
        # was justified by a stall within tol_pitch of the maximum, so the basin
        # boundary is drawn at the same radius -- otherwise a stall 1.5 pitches
        # away promotes a maximum its own arrival test cannot reach, and the
        # branch dissolves against the evidence that just restored it. Measured on
        # ethene3 shape index: at 1 pitch, E-SAD_MS4 br1 (stall 3.4 deg from
        # MAX_5, tol 4.46 deg) promotes and then spirals the full 891 deg budget
        # without arriving.
        enlarged = centred_targets(
            [
                *pool,
                *promotion_targets(
                    promotions, tol_pitch * delta_theta_mesh_rad, center
                ),
            ],
            center,
        )
        ports = _branch_ports(seeds)
        promoted_ids = {p.cp_id for p in promotions}
        new_verdicts: list[RefinementVerdict] = []
        for v in verdicts:
            key = (v.saddle_id, v.branch_index)
            if (
                v.verdict == VERDICT_DISSOLVED
                and causes.get(key) == CAUSE_PROMOTED
                and key in ports
            ):
                rv = _trace_verdict(
                    saddle_id=v.saddle_id,
                    branch_index=v.branch_index,
                    claimed_max_id=None,
                    branch=ports[key],
                    centred_pool=enlarged,
                    mesh_pts=mesh_pts,
                    f_vals=f_vals,
                    sphere_radius=sphere_radius,
                    delta_theta_mesh_rad=delta_theta_mesh_rad,
                    center=center,
                    find_k_nearest=find_k_nearest,
                )
                # The re-resolution's arrival is a REROUTED verdict: the ridge
                # enters a maximum the connector did not claim (the promoted one,
                # or any other). The claim is preserved for the record.
                new_verdicts.append(
                    replace(
                        rv,
                        claimed_max_id=v.claimed_max_id,
                        verdict=(
                            VERDICT_REROUTED
                            if rv.ridge_max_id in promoted_ids
                            else rv.verdict
                        ),
                    )
                )
            else:
                new_verdicts.append(v)
        verdicts = new_verdicts
    stats = summarize_refinement(verdicts, promotions, causes)
    return RefinementResult(
        verdicts=verdicts, promotions=promotions, causes=causes, stats=stats
    )


def apply_refinement(
    connections: Sequence[MaxConnection],
    result: RefinementResult,
) -> list[MaxConnection]:
    """Rewrite the connection table per the ridge's verdicts.

    The gradient is ground truth: a REROUTED branch adopts the maximum the ridge
    enters, and its connector becomes the traced ridge line -- the same contract
    Step 4 established for the branches the ray never resolved. Confirmed and
    resolved branches keep their geodesic connectors (the ridge verified them);
    dissolved branches stay no_target, now with the audit's cause recorded in the
    stats. The input list is returned unchanged when no branch reroutes, which is
    the dense-field case: the pass is a no-op, byte-identical.

    Args:
        connections: The Step 3/4 connection table.
        result: The refinement pass, post-re-resolution.

    Returns:
        The corrected table, in the input order.
    """
    by_key = {(v.saddle_id, v.branch_index): v for v in result.verdicts}
    out: list[MaxConnection] = []
    for conn in connections:
        v = by_key.get((conn.saddle_id, conn.branch_index))
        if (
            v is None
            or v.verdict != VERDICT_REROUTED
            or v.ridge_max_id is None
            or len(v.ridge_path) < 2
        ):
            out.append(conn)
            continue
        hit = next((p for p in result.promotions if p.cp_id == v.ridge_max_id), None)
        out.append(
            replace(
                conn,
                max_id=v.ridge_max_id,
                status="connected",
                method="gradient",
                along_track_deg=v.ridge_arc_deg,
                # The old separation was measured to the claim the ridge
                # contradicted; the traced arc length is the honest extent.
                separation_deg=None,
                arc_points=v.ridge_path,
                max_value=hit.value if hit is not None else None,
                blocker_id=None,
                blocker_polarity=None,
            )
        )
    return out


def summarize_refinement(
    verdicts: Sequence[RefinementVerdict],
    promotions: Sequence[Promotion],
    causes: dict[tuple[str, int], str],
) -> dict[str, Any]:
    """Reduce the pass to the numbers that go in the snapshot.

    Args:
        verdicts: One per connection.
        promotions: The audit's promotions.
        causes: The per-branch audit outcomes.

    Returns:
        Counts by verdict and by cause, plus the mean and max endpoint deviation
        over the CONFIRMED branches -- the honest measure of how much the straight
        connector lies on this field, where the ridge it summarises is known.
    """

    def _count(name: str) -> int:
        return sum(1 for v in verdicts if v.verdict == name)

    dev = [
        float(v.endpoint_deviation_deg)
        for v in verdicts
        if v.verdict == VERDICT_CONFIRMED and v.endpoint_deviation_deg is not None
    ]
    return {
        "n_tested": len(verdicts),
        "n_confirmed": _count(VERDICT_CONFIRMED),
        "n_rerouted": _count(VERDICT_REROUTED),
        "n_dissolved": _count(VERDICT_DISSOLVED),
        "n_ridge": _count(VERDICT_RIDGE),
        "n_resolved": _count(VERDICT_RESOLVED),
        "n_promoted": len(promotions),
        "promoted_ids": sorted(p.cp_id for p in promotions),
        "n_corner": sum(1 for c in causes.values() if c == CAUSE_CORNER),
        "n_no_evidence": sum(1 for c in causes.values() if c == CAUSE_NO_EVIDENCE),
        "mean_endpoint_deviation_deg": sum(dev) / len(dev) if dev else 0.0,
        "max_endpoint_deviation_deg": max(dev) if dev else 0.0,
    }
