"""Unit tests for :mod:`gba_topology2.src.connector_refinement` (Step 5).

The rule under test: every connector is a claim, the ridge is the verdict.
confirmed = the ridge enters the claimed disc; rerouted = it enters a different
maximum's; dissolved = it stalls, and the stall point is then audited against the
raw maxima the reduction hid: near a hidden maximum -> over-merge, promote it;
on flat ground -> corner-sharing.

Fixtures are synthetic fields on a unit sphere with known ascent basins, the
same construction as the tracer tests. The ethene3 shape-index case is reproduced
in miniature: a small bump the ascent drains to, a tall spike it cannot reach
(separated by a saddle), the bump hidden by the reduction.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from gba_topology2.src.connector_refinement import (
    CAUSE_CORNER,
    CAUSE_NO_EVIDENCE,
    CAUSE_PROMOTED,
    VERDICT_CONFIRMED,
    VERDICT_DISSOLVED,
    VERDICT_REROUTED,
    VERDICT_RESOLVED,
    VERDICT_RIDGE,
    audit_dissolved,
    promotion_targets,
    refine_and_audit,
    refine_connections,
    summarize_refinement,
)
from gba_topology2.src.max_connection import ConnectionTarget, MaxConnection
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.sphere_frame import to_global

R = 1.0
PITCH = math.radians(3.0)
CENTER = (2.0, -1.5, 0.8)


def _pt(colat_deg: float, az_deg: float) -> tuple[float, float, float]:
    t = math.radians(colat_deg)
    p = math.radians(az_deg)
    return (
        math.sin(t) * math.cos(p),
        math.sin(t) * math.sin(p),
        math.cos(t),
    )


def _fib(n: int = 1200) -> npt.NDArray[np.float64]:
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    return np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


def _peak(
    pts: npt.NDArray[np.float64],
    center_unit: tuple[float, float, float],
    width_rad: float,
    height: float,
) -> npt.NDArray[np.float64]:
    c = np.array(center_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    d = np.arccos(np.clip(pts @ c, -1.0, 1.0))
    return height * np.exp(-((d / width_rad) ** 2))


def _branch(index: int, colat_deg: float, az_deg: float = 0.0) -> LaunchBranch:
    pos = np.array(_pt(colat_deg, az_deg), dtype=np.float64)
    origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent = pos - float(np.dot(pos, origin)) * origin
    tangent /= float(np.linalg.norm(tangent))
    return LaunchBranch(
        kind="ascent",
        sector_index=index,
        tangent_direction=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
        launch_position=(float(pos[0]), float(pos[1]), float(pos[2])),
        sector_width_rad=math.radians(60.0),
        sector_centroid_value=1.0,
    )


def _seed(saddle_id: str, branches: list[LaunchBranch]) -> SaddleLaunchSeed:
    return SaddleLaunchSeed(
        saddle_id=saddle_id,
        saddle_position=(0.0, 0.0, 1.0),
        saddle_value=0.5,
        vertex_id=-1,
        multiplicity=1,
        fold_order=2,
        index_agrees=True,
        ascent_branches=branches,
        descent_branches=[],
        ring_radius_rad=math.radians(3.0),
    )


def _target(
    cp_id: str,
    polarity: str,
    colat_deg: float,
    az_deg: float = 0.0,
    value: float = 1.0,
    disc_deg: float = 8.0,
    center: tuple[float, float, float] | None = None,
) -> ConnectionTarget:
    pos = _pt(colat_deg, az_deg)
    world = to_global(pos, center) if center is not None else pos
    return ConnectionTarget(
        cp_id=cp_id,
        polarity=polarity,
        position=np.array(world, dtype=np.float64),
        value=value,
        angular_radius_rad=math.radians(disc_deg),
    )


def _conn(
    saddle_id: str,
    branch_index: int,
    max_id: str | None,
    status: str = "connected",
    method: str = "geodesic",
) -> MaxConnection:
    return MaxConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        max_id=max_id,
        status=status,
        direction_mode="port_tangent",
        cross_track_deg=None,
        along_track_deg=None,
        separation_deg=None,
        mutual=False,
        arc_points=(),
        saddle_value=0.5,
        max_value=None,
        verified=True,
        method=method,
    )


def test_ridge_into_the_claimed_disc_is_confirmed() -> None:
    """A peak straight ahead: the ray's claim and the ridge agree."""
    pts = _fib()
    peak = (0.0, 0.0, 1.0)
    f = _peak(pts, peak, math.radians(35.0), 1.0)
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    conn = _conn("SAD_1", 0, "MAX_0")
    pool = [_target("MAX_0", "maximum", 0.0)]

    (v,) = refine_connections([conn], [seed], pool, pts, f, R, PITCH, center=None)
    assert v.verdict == VERDICT_CONFIRMED
    assert v.ridge_max_id == "MAX_0"
    assert v.stall_unit is None
    assert v.endpoint_deviation_deg is not None
    assert v.endpoint_deviation_deg < 20.0


def test_ridge_into_a_different_maximum_is_rerouted() -> None:
    """The ray claimed the far peak; the ridge drains to the near one."""
    pts = _fib()
    near = (0.0, 0.0, 1.0)
    far = (math.sin(2.4), 0.0, math.cos(2.4))
    f = _peak(pts, near, math.radians(30.0), 1.0) + _peak(
        pts, far, math.radians(30.0), 0.9
    )
    seed = _seed("SAD_1", [_branch(0, 15.0)])
    conn = _conn("SAD_1", 0, "MAX_FAR")
    pool = [_target("MAX_NEAR", "maximum", 0.0), _target("MAX_FAR", "maximum", 137.5)]

    (v,) = refine_connections([conn], [seed], pool, pts, f, R, PITCH, center=None)
    assert v.verdict == VERDICT_REROUTED
    assert v.claimed_max_id == "MAX_FAR"
    assert v.ridge_max_id == "MAX_NEAR"


def test_gradient_method_is_a_ridge_and_not_re_tested() -> None:
    """Step 4's traced connectors ARE ridges; the pass declines to re-judge them."""
    pts = _fib()
    f = _peak(pts, (0.0, 0.0, 1.0), math.radians(35.0), 1.0)
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    conn = _conn("SAD_1", 0, "MAX_0", method="gradient")

    (v,) = refine_connections(
        [conn], [seed], [_target("MAX_0", "maximum", 0.0)], pts, f, R, PITCH
    )
    assert v.verdict == VERDICT_RIDGE
    assert v.ridge_max_id == "MAX_0"


def test_occluded_is_resolved() -> None:
    """An occlusion is a verdict about the ray, not about the ascent."""
    pts = _fib()
    f = _peak(pts, (0.0, 0.0, 1.0), math.radians(35.0), 1.0)
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    conn = _conn("SAD_1", 0, None, status="occluded")

    (v,) = refine_connections(
        [conn], [seed], [_target("MAX_0", "maximum", 0.0)], pts, f, R, PITCH
    )
    assert v.verdict == VERDICT_RESOLVED


def test_missing_port_is_resolved() -> None:
    """A connection whose seed carries no branch cannot be tested; no crash."""
    pts = _fib()
    f = _peak(pts, (0.0, 0.0, 1.0), math.radians(35.0), 1.0)
    conn = _conn("SAD_9", 3, "MAX_0")

    (v,) = refine_connections(
        [conn], [], [_target("MAX_0", "maximum", 0.0)], pts, f, R, PITCH
    )
    assert v.verdict == VERDICT_RESOLVED


def test_ethene3_miniature_dissolves_then_promotes_then_reroutes() -> None:
    """The shape-index case in miniature: bump (hidden) + spike (visible).

    The port climbs the small bump and stalls: the spike is beyond a saddle and
    ascent is strictly uphill. The audit finds the bump in the hidden maxima
    within one pitch, promotes it, and the re-resolution arrives at it -- a
    REROUTED verdict naming the promoted maximum.
    """
    pts = _fib()
    bump = (math.sin(0.35), 0.0, math.cos(0.35))  # ~20 deg colatitude
    spike = (0.0, 0.0, 1.0)
    # Narrow peaks, 20 deg apart: the sum has a genuine saddle between them
    # (field ~0.21 at the colat-14 gap vs 0.30 at the bump), so the bump is its
    # own ascent basin and the spike is unreachable from the port.
    f = _peak(pts, bump, math.radians(8.0), 0.3) + _peak(
        pts, spike, math.radians(8.0), 1.0
    )
    # Port below the bump, climbing into it.
    seed = _seed("SAD_1", [_branch(0, 32.0)])
    conn = _conn("SAD_1", 0, None, status="no_target")
    pool = [_target("MAX_SPIKE", "maximum", 0.0, disc_deg=8.0)]
    hidden = [CriticalPoint(vertex_id=42, cp_type="maximum", value=0.3, position=bump)]

    result = refine_and_audit(
        connections=[conn],
        seeds=[seed],
        pool=pool,
        hidden_maxima=hidden,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=R,
        delta_theta_mesh_rad=PITCH,
        center=None,
    )
    (v,) = result.verdicts
    assert v.verdict == VERDICT_REROUTED
    assert v.ridge_max_id == "MAX_42"
    assert v.claimed_max_id is None
    assert len(result.promotions) == 1
    promo = result.promotions[0]
    assert promo.vertex_id == 42
    assert promo.cp_id == "MAX_42"
    assert promo.distance_deg < math.degrees(PITCH)
    assert result.causes[("SAD_1", 0)] == CAUSE_PROMOTED
    assert result.stats["n_promoted"] == 1
    assert result.stats["n_rerouted"] == 1


def test_stall_far_from_every_hidden_maximum_is_corner_sharing() -> None:
    """A stall on flat ground with the only hidden maximum far away: corner."""
    stalls = [("SAD_1", 0, (0.0, 0.0, 1.0))]
    hidden = [
        CriticalPoint(
            vertex_id=7, cp_type="maximum", value=0.1, position=(0.0, 1.0, 0.0)
        )
    ]

    promotions, causes = audit_dissolved(stalls, hidden, PITCH, tol_pitch=1.0)
    assert promotions == []
    assert causes[("SAD_1", 0)] == CAUSE_CORNER


def test_stall_with_no_hidden_maxima_is_no_evidence() -> None:
    """Nothing was hidden: the audit has no hypothesis to test."""
    stalls = [("SAD_1", 0, (0.0, 0.0, 1.0))]
    promotions, causes = audit_dissolved(stalls, [], PITCH)
    assert promotions == []
    assert causes[("SAD_1", 0)] == CAUSE_NO_EVIDENCE


def test_audit_dedupes_promotions_keeping_the_nearest_stall() -> None:
    """Two stalls draining to one hidden maximum: one promotion, nearest wins."""
    hidden = [
        CriticalPoint(
            vertex_id=9, cp_type="maximum", value=0.2, position=(0.0, 0.0, 1.0)
        )
    ]
    stalls = [
        ("SAD_1", 0, (math.sin(0.02), 0.0, math.cos(0.02))),  # ~1.1 deg
        ("SAD_2", 1, (0.0, 0.0, 1.0)),  # 0 deg
    ]

    promotions, causes = audit_dissolved(stalls, hidden, PITCH)
    assert len(promotions) == 1
    assert promotions[0].distance_deg == 0.0
    assert promotions[0].source_saddle_id == "SAD_2"
    assert causes[("SAD_1", 0)] == CAUSE_PROMOTED
    assert causes[("SAD_2", 1)] == CAUSE_PROMOTED


def test_promotion_targets_map_to_the_world_frame() -> None:
    """Pool entries carry world positions; the id space matches basin centers."""
    from gba_topology2.src.connector_refinement import Promotion

    promo = Promotion(
        vertex_id=42,
        cp_id="MAX_42",
        position=(0.0, 0.0, 1.0),
        value=0.3,
        distance_deg=0.5,
        source_saddle_id="SAD_1",
        source_branch_index=0,
    )
    (t,) = promotion_targets([promo], PITCH, CENTER)
    assert t.cp_id == "MAX_42"
    assert t.polarity == "maximum"
    assert np.allclose(t.position, np.array(to_global((0.0, 0.0, 1.0), CENTER)))
    assert math.degrees(t.angular_radius_rad) == pytest.approx(3.0)


def test_refinement_is_frame_consistent() -> None:
    """The same geometry offset from the origin gives the same verdict.

    The pool is built in the world frame and the seeds/field in the centred
    frame; a mixup is invisible at the origin and fatal at 1.26 A, so the
    confirmed case is run in both frames.
    """
    pts = _fib()
    peak = (0.0, 0.0, 1.0)
    f = _peak(pts, peak, math.radians(35.0), 1.0)
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    conn = _conn("SAD_1", 0, "MAX_0")

    (v0,) = refine_connections(
        [conn], [seed], [_target("MAX_0", "maximum", 0.0)], pts, f, R, PITCH
    )
    (vw,) = refine_connections(
        [conn],
        [seed],
        [_target("MAX_0", "maximum", 0.0, center=CENTER)],
        pts,
        f,
        R,
        PITCH,
        center=CENTER,
    )
    assert v0.verdict == vw.verdict == VERDICT_CONFIRMED
    assert vw.ridge_max_id == "MAX_0"


def test_summary_counts_verdicts_and_deviation() -> None:
    """The stats dict is the snapshot payload; check it is populated honestly."""
    from gba_topology2.src.connector_refinement import RefinementVerdict

    verdicts = [
        RefinementVerdict(
            saddle_id="S1",
            branch_index=0,
            verdict=VERDICT_CONFIRMED,
            endpoint_deviation_deg=4.0,
        ),
        RefinementVerdict(saddle_id="S1", branch_index=1, verdict=VERDICT_DISSOLVED),
        RefinementVerdict(saddle_id="S2", branch_index=0, verdict=VERDICT_RIDGE),
    ]
    stats = summarize_refinement(
        verdicts,
        promotions=[],
        causes={("S1", 1): CAUSE_CORNER},
    )
    assert stats["n_tested"] == 3
    assert stats["n_confirmed"] == 1
    assert stats["n_dissolved"] == 1
    assert stats["n_ridge"] == 1
    assert stats["n_corner"] == 1
    assert stats["mean_endpoint_deviation_deg"] == 4.0
    assert stats["max_endpoint_deviation_deg"] == 4.0


def test_apply_refinement_installs_the_ridge_and_leaves_confirmed_alone() -> None:
    """The corrected table: rerouted adopts the ridge's maximum and its polyline;
    confirmed keeps the geodesic; the no-branch case is byte-identical."""
    from dataclasses import replace

    from gba_topology2.src.connector_refinement import (
        RefinementResult,
        apply_refinement,
    )

    pts = _fib()
    f = _peak(pts, (0.0, 0.0, 1.0), math.radians(35.0), 1.0)
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    conn = _conn("SAD_1", 0, "MAX_0")
    pool = [_target("MAX_0", "maximum", 0.0)]
    result = refine_and_audit(
        connections=[conn],
        seeds=[seed],
        pool=pool,
        hidden_maxima=[],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=R,
        delta_theta_mesh_rad=PITCH,
        center=None,
    )
    # Confirmed: the table is untouched, object for object.
    applied = apply_refinement([conn], result)
    assert applied[0] is conn

    # A forced rerouted verdict installs the ridge as the connector.
    path = ((0.1, 0.0, 0.99), (0.05, 0.0, 0.999), (0.0, 0.0, 1.0))
    rerouted = replace(
        result.verdicts[0],
        verdict="rerouted",
        ridge_max_id="MAX_42",
        ridge_path=path,
        ridge_arc_deg=18.0,
    )
    forced = RefinementResult(
        verdicts=[rerouted],
        promotions=result.promotions,
        causes=result.causes,
        stats=result.stats,
    )
    (out,) = apply_refinement([conn], forced)
    assert out.max_id == "MAX_42"
    assert out.status == "connected"
    assert out.method == "gradient"
    assert out.arc_points == path
    assert out.along_track_deg == 18.0
    assert out.separation_deg is None
    # The input record is immutable: the original is intact.
    assert conn.max_id == "MAX_0"
