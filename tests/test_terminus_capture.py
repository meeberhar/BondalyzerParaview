"""Terminus capture: a traced connector lands on the CP its table names.

The defect these tests pin, reported by the operator and measured on
``ethene3_atom1_params.json`` (shape index, atom 1, stage "connect", mesh pitch 2.228 deg): a
gradient path declares arrival the moment a SAMPLE lands inside its target's disc, and the walk is
then free to keep going. When the target is an ISOLATED CP, about a mesh pitch from its centre, and
nobody notices. When the target is an EFFECTIVE CP -- one cluster standing for a family of reduced
CPs, on ground the reduction declared untrustworthy -- the disc is the CLUSTER's, up to 15.50 deg
wide here, and the walk runs all the way in to the centre EXACTLY, then spends its last step on a
spurious well inside the cluster. The emitted connector ends off the CP its own table names, at the
bottom of a dent the clustering already absorbed. Gap from the CP, measured over the 20 connector
rows, capture off -> on:

    SAD_1594   -> E-MIN_MS5     3.386 deg  (1.52 pitch)  -> 0.000
    SAD_1681   -> E-MIN_MS5     3.386 deg  (1.52 pitch)  -> 0.000
    SAD_8400   -> E-MIN_MS4     3.369 deg  (1.51 pitch)  -> 0.000
    SAD_8487   -> E-MIN_MS4     3.369 deg  (1.51 pitch)  -> 0.000
    E-SAD_MC1  -> E-MIN_MS4     3.369 deg  (1.51 pitch)  -> 0.000
    E-SAD_MC1  -> E-MIN_MS5     2.400 deg  (1.08 pitch)  -> 0.000   the operator's example
    ...                                                                     mean 1.110 -> 0.010

Six of the twenty rows end more than half a mesh pitch from the CP they name. With capture on, all
twenty end at 0.00-0.03 deg. Seventeen of them need NO drawn leg: the path had already touched the
centre, so truncating at the CLOSEST APPROACH is the whole fix and nothing is added. Three are
carried -- ``E-SAD_MC1 -> E-MIN_MS5`` at 9.73 deg (4.37 pitch) and the two descents onto
``E-MIN_MC4`` at 3.69 / 3.95 deg -- and every one of those is published as ``completed_leg_deg``,
because a carried leg is the part of a connector the field did not measure.

The fields here are analytic, with one deliberate touch: a spurious maximum added INSIDE the
target's disc, 12 deg off the true one, so a walk that is allowed to keep going after the disc
entry dies on it. That is the reported situation and nothing about it needs mocking.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from gba_topology2.src.max_connection import ConnectionTarget
from gba_topology2.src.params_schema import (
    PARAM_LEGACY_DEFAULT_OVERRIDES,
    PARAM_LEGACY_SCHEMA_VERSIONS,
    PARAMS_SCHEMA_VERSION,
    TRACE_CONNECTOR_STAT_KEYS,
    make_trace_connector_stats,
)
from gba_topology2.src.separatrix_tracer import (
    ExcisionRegion,
    TraceOutcome,
    trace_downhill,
    trace_uphill,
)

_R = 1.0
_PITCH = math.radians(3.0)  # ~3 deg, comparable to the real meshes
_N_MESH = 1200


class _Target:
    """Minimal pool entry: the tracer reads id, polarity, position, and radius_toward()."""

    def __init__(
        self,
        cp_id: str,
        polarity: str,
        position: npt.NDArray[np.float64],
        radius_rad: float,
    ) -> None:
        self.cp_id = cp_id
        self.polarity = polarity
        self.position = np.asarray(position, dtype=np.float64)
        self._radius_rad = radius_rad

    def radius_toward(self, target_unit: npt.NDArray[np.float64]) -> float:
        del target_unit  # isotropic: the stub has no ellipse
        return self._radius_rad


def _fibonacci_sphere(
    num_points: int = _N_MESH, radius: float = _R
) -> npt.NDArray[np.float64]:
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    return np.column_stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ]
    )


def _gaussian_peak(
    pts: npt.NDArray[np.float64],
    center_unit: npt.NDArray[np.float64],
    width_rad: float,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    d = np.arccos(
        np.clip(pts / np.linalg.norm(pts, axis=1, keepdims=True) @ center_unit, -1, 1)
    )
    return height * np.exp(-((d / width_rad) ** 2))


def _unit(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.asarray(v, dtype=np.float64)
    return np.asarray(out / np.linalg.norm(out), dtype=np.float64)


def _lat(deg: float) -> npt.NDArray[np.float64]:
    """A unit vector at ``deg`` from +z, in the x-z plane."""

    t = math.radians(deg)
    return np.array([math.sin(t), 0.0, math.cos(t)])


def _ang_deg(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(_unit(a) @ _unit(b))))))


class _Fixture:
    """One analytic field, its port, and the two maxima in it."""

    def __init__(self) -> None:
        self.pts = _fibonacci_sphere()
        self.true_cp = np.array([0.0, 0.0, 1.0])
        self.spurious = _lat(12.0)
        self.f = _gaussian_peak(
            self.pts, self.true_cp, width_rad=math.radians(45.0)
        ) + _gaussian_peak(
            self.pts, self.spurious, width_rad=math.radians(5.0), height=0.6
        )
        self.port = _lat(40.0)
        # 6 pitches = 18 deg, wider than the 12 deg offset of the spurious peak, so the
        # disc entry happens before the true CP and the walk is loose inside it.
        self.radius_deg = 18.0

    def targets(self) -> list[_Target]:
        return [
            _Target(
                "MAX_0",
                "maximum",
                self.true_cp,
                math.radians(self.radius_deg),
            )
        ]

    def ascend(
        self, *, capture: bool, regions: tuple[ExcisionRegion, ...] = ()
    ) -> TraceOutcome:
        return trace_uphill(
            start_position=self.port,
            targets=self.targets(),
            mesh_pts=self.pts,
            f_vals=self.f,
            sphere_radius=_R,
            delta_theta_mesh_rad=_PITCH,
            center=None,
            excision_regions=regions,
            terminus_capture=capture,
        )


# ---------------------------------------------------------------------------
# The reported case.
# ---------------------------------------------------------------------------


def test_the_fixture_reproduces_the_reported_defect() -> None:
    """Guard on the guard: without capture, the connector stops short of the CP it names."""

    fx = _Fixture()
    off = fx.ascend(capture=False)
    assert off.status == "arrived"
    assert off.max_id == "MAX_0"
    gap = _ang_deg(np.asarray(off.path[-1], float), fx.true_cp)
    assert gap > 2.0 * math.degrees(_PITCH), (
        f"fixture has no wide-disc arrival to fix ({gap=})"
    )
    assert off.completed_leg_deg is None, "capture off must draw nothing"


def test_capture_lands_the_connector_on_the_cp_centre() -> None:
    """The fix, stated as the operator stated it: hit the circle, finish at the centre."""

    fx = _Fixture()
    on = fx.ascend(capture=True)
    assert on.status == "arrived"
    assert on.max_id == "MAX_0"
    assert _ang_deg(np.asarray(on.path[-1], float), fx.true_cp) == pytest.approx(
        0.0, abs=1e-9
    )
    # The endpoint is exact and the cost is disclosed.
    assert on.completed_leg_deg is not None
    assert on.completed_leg_deg > 0.0


def test_capture_only_adds_the_straight_leg_and_keeps_the_measured_path() -> None:
    """What the capture writes is the leg, nothing else.

    The measured walk is kept byte-for-byte up to the truncation, and the run that follows is the
    geodesic to the centre, resampled at the step pitch. The disclosed leg is exactly the arc from
    the truncation point to the centre -- a measured distance, not a fudge factor.
    """

    fx = _Fixture()
    off = fx.ascend(capture=False)
    on = fx.ascend(capture=True)
    leg = float(on.completed_leg_deg or 0.0)
    n_leg = max(1, math.ceil(math.radians(leg) / _PITCH))
    head = len(on.path) - n_leg - 1  # index of the truncation point

    # The measured part is the uncaptured walk, unchanged, up to the truncation.
    np.testing.assert_allclose(
        np.asarray(on.path[: head + 1], float),
        np.asarray(off.path[: head + 1], float),
        atol=1e-12,
    )
    # The leg length IS the arc from the truncation point to the CP.
    assert leg == pytest.approx(
        _ang_deg(np.asarray(on.path[head], float), fx.true_cp), abs=1e-9
    )
    assert len(on.path) == head + 1 + n_leg
    # Every point stays on the sphere, and the path ends on the axis.
    radii = [float(np.linalg.norm(np.asarray(p, float))) for p in on.path]
    assert max(abs(r - _R) for r in radii) < 1e-9
    assert _ang_deg(np.asarray(on.path[-1], float), fx.true_cp) == pytest.approx(
        0.0, abs=1e-9
    )


def test_capture_is_off_by_default_for_every_untouched_caller() -> None:
    """Back-compat: the flag is additive, so a caller that never mentions it is unchanged."""

    fx = _Fixture()
    plain = trace_uphill(
        start_position=fx.port,
        targets=fx.targets(),
        mesh_pts=fx.pts,
        f_vals=fx.f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    explicit = fx.ascend(capture=False)
    np.testing.assert_allclose(
        np.asarray(plain.path, float), np.asarray(explicit.path, float), atol=0.0
    )
    assert plain.completed_leg_deg is None


def test_a_path_crossing_a_non_target_disc_is_not_cut_by_it() -> None:
    """The regression the operator's phrasing invites.

    "Terminate the gradient paths once they cross the circle" taken bare also cuts a path that
    clips some OTHER CP on the way to its own goal -- most visibly an ascent leaving a saddle that
    sits inside a minimum's disc, which is the separatrix the connector exists to show. The capture
    runs on the target the walk actually arrived on, so a crossing of a non-target stays a transit:
    the walk continues, records the CP it passed, and finishes on its own goal.
    """

    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(45.0))
    port = _lat(50.0)
    mid = _lat(25.0)  # on the way, with a disc (40 deg) wider than the peak's (6 deg)
    targets = [
        _Target("MAX_0", "maximum", peak, math.radians(6.0)),
        _Target("MIN_7", "minimum", mid, math.radians(40.0)),
    ]

    def run(capture: bool) -> TraceOutcome:
        return trace_uphill(
            start_position=port,
            targets=targets,
            mesh_pts=pts,
            f_vals=f,
            sphere_radius=_R,
            delta_theta_mesh_rad=_PITCH,
            center=None,
            terminus_capture=capture,
        )

    off, on = run(False), run(True)
    assert off.status == "arrived"
    assert "MIN_7" in off.visited_ids, "the fixture must exercise a crossing"
    # Capture changed nothing about the transit: same goal, CP recorded, walk not cut short.
    assert on.status == "arrived"
    assert on.max_id == off.max_id == "MAX_0"
    assert "MIN_7" in on.visited_ids
    # The measured walk survives intact; the only difference is the leg onto the peak's centre.
    np.testing.assert_allclose(
        np.asarray(on.path[: len(off.path)], float),
        np.asarray(off.path, float),
        atol=1e-12,
    )
    assert len(on.path) > len(off.path)
    assert _ang_deg(np.asarray(on.path[-1], float), peak) == pytest.approx(
        0.0, abs=1e-9
    )


def test_the_descent_mirror_lands_on_the_minimum() -> None:
    """``trace_downhill`` is the same walk on the negated field; the rule mirrors."""

    pts = _fibonacci_sphere()
    trough = np.array([0.0, 0.0, 1.0])
    spurious = _lat(12.0)
    f = -(
        _gaussian_peak(pts, trough, width_rad=math.radians(45.0))
        + _gaussian_peak(pts, spurious, width_rad=math.radians(5.0), height=0.6)
    )
    # The descent's real target type, not the ascent stub: ConnectionTarget is what
    # max_connection hands trace_downhill, and its __post_init__ turns the absent
    # ellipse into the isotropic radius -- the isolated-CP case.
    targets = [
        ConnectionTarget(
            cp_id="MIN_0",
            polarity="minimum",
            position=trough,
            value=float(f[0]),
            angular_radius_rad=math.radians(18.0),
        )
    ]

    def run(capture: bool) -> TraceOutcome:
        return trace_downhill(
            start_position=_lat(40.0),
            targets=targets,
            mesh_pts=pts,
            f_vals=f,
            sphere_radius=_R,
            delta_theta_mesh_rad=_PITCH,
            center=None,
            terminus_capture=capture,
        )

    off, on = run(False), run(True)
    assert off.status == "arrived"
    assert _ang_deg(np.asarray(off.path[-1], float), trough) > 2.0 * math.degrees(
        _PITCH
    )
    assert off.completed_leg_deg is None
    assert on.status == "arrived"
    assert on.max_id == "MIN_0"
    assert _ang_deg(np.asarray(on.path[-1], float), trough) == pytest.approx(
        0.0, abs=1e-9
    )
    assert on.completed_leg_deg is not None and on.completed_leg_deg > 0.0


def test_an_excised_terminus_is_still_landed_on_the_cp() -> None:
    """The two rules meet at the target's own disc and must not deadlock.

    Excision regions stop a walk that enters them, and a terminus CP's neighbourhood is exactly the
    kind of region that gets excised. The capture runs on the arrival the walk reported, so the
    connector cannot end in open space because a region was in the way.
    """

    fx = _Fixture()
    region = ExcisionRegion(
        region_id="Z_TRUE",
        centre_unit=fx.true_cp,
        radius_rad=math.radians(4.0),
    )
    out = fx.ascend(capture=True, regions=(region,))
    assert out.status == "arrived"
    assert _ang_deg(np.asarray(out.path[-1], float), fx.true_cp) == pytest.approx(
        0.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# Disclosure: the count and the longest drawn leg reach the card.
# ---------------------------------------------------------------------------


def test_published_stats_carry_the_capture_count_and_leg() -> None:
    """The producer names a completion ``n_completed``; the card must not lose it.

    The rename at the schema boundary is the only place the two vocabularies meet, so it is the
    place a silent 0 would be invisible in the UI and obvious here.
    """

    out = make_trace_connector_stats(
        {
            "n_branches": 10,
            "n_traced": 10,
            "n_trace_arrived": 10,
            "n_completed": 3,
            "max_completed_leg_deg": 9.73,
            "descent": {
                "n_completed": 3,
                "max_completed_leg_deg": 9.73,
                "n_traced": 10,
            },
        },
        on=True,
    )
    assert out["n_terminus_captured"] == 3
    assert out["max_terminus_leg_deg"] == pytest.approx(9.73)
    descent = out["descent"]
    assert descent is not None
    assert descent["n_terminus_captured"] == 3
    assert descent["max_terminus_leg_deg"] == pytest.approx(9.73)
    assert set(out) == set(TRACE_CONNECTOR_STAT_KEYS)


def test_published_stats_are_zero_when_tracing_is_off() -> None:
    out = make_trace_connector_stats(
        {"n_completed": 9, "max_completed_leg_deg": 40.0}, on=False
    )
    assert out["n_terminus_captured"] == 0
    assert out["max_terminus_leg_deg"] == 0.0
    assert out["descent"] is None


def test_capture_is_a_versioned_control_that_replays_old_snapshots_without_it() -> None:
    """A v13 file was recorded before the rule existed, so it must replay as it drew.

    The default is ON for a fresh session and OFF for every snapshot at or below 13. That is the
    only way both ``ethene3_atom1_params.json`` (v13, the file the operator reproduced with) and a
    new save show what was on screen when they were made.
    """

    assert PARAMS_SCHEMA_VERSION == 14
    per_version = PARAM_LEGACY_DEFAULT_OVERRIDES["terminus_capture"]
    assert set(per_version) == set(PARAM_LEGACY_SCHEMA_VERSIONS)
    assert all(v is False for v in per_version.values())


def test_capture_default_is_on_for_a_fresh_session() -> None:
    from gba_topology2.src.params_schema import DEFAULT_UI_STATE, PARAM_BOOL_KEYS

    assert DEFAULT_UI_STATE["terminus_capture"] is True
    assert "terminus_capture" in PARAM_BOOL_KEYS
