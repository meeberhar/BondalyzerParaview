"""Unit tests for :mod:`gba_topology2.src.separatrix_tracer`.

The tracer climbs the condensed field from a saddle's launch port to the maximum
its separatrix terminates on. These tests exercise it on synthetic fields with
known answers: a single smooth peak, a ridge that curves (so a straight ray would
miss), a flat plateau (no maximum to reach), and a field whose peak sits off the
great circle the port points along.

The frame contract is the point of most of these: the tracer integrates on the
CENTRED sphere, so a field built around a nucleus at the origin is the easy case,
and the tests deliberately include an off-origin atom to catch the recurring
world/centred mixup that Pd_20K hides and ethene3 exposes.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pytest

from gba_topology2.src.separatrix_tracer import (
    MAX_TRACE_ARC_PITCH,
    STALL_PATIENCE,
    TRACE_STEP_PITCH,
    TraceOutcome,
    centred_targets,
    trace_uphill,
)

# A mesh fine enough that the IDW interpolant resolves a peak of the angular width
# used below, and coarse enough that the brute-force kernel stays fast.
_N_MESH = 1200
_R = 1.0
_PITCH = math.radians(3.0)  # ~3 deg, comparable to the real meshes


def _fibonacci_sphere(
    num_points: int = _N_MESH, radius: float = _R
) -> npt.NDArray[np.float64]:
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _gaussian_peak(
    pts: npt.NDArray[np.float64],
    center_unit: npt.NDArray[np.float64],
    width_rad: float,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    """A single smooth maximum at ``center_unit``, decaying as cos of arc distance.

    ``exp(-(d/width)^2)`` on the sphere: monotone in the geodesic distance from the
    peak, so the surface gradient points straight at it from every direction and the
    ascent has a unique attractor.
    """
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    d = np.arccos(np.clip(u @ center_unit, -1.0, 1.0))
    return height * np.exp(-((d / width_rad) ** 2))


class _Target:
    """Minimal pool entry: the tracer only needs id, polarity, position, radius."""

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
        return self._radius_rad


def _max_target(
    center_unit: npt.NDArray[np.float64],
    radius_deg: float = 8.0,
    cp_id: str = "MAX_0",
) -> _Target:
    return _Target(cp_id, "maximum", center_unit, math.radians(radius_deg))


def test_climbs_to_a_single_peak() -> None:
    """A smooth peak is reached from a port on its far side; the path rises."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(35.0))
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(peak)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_0"
    assert out.end_value > out.start_value
    assert len(out.path) == out.n_steps + 1


def test_arrives_at_the_nearest_of_two_peaks() -> None:
    """With two maxima, the ascent terminates on the one its ridge drains to."""
    pts = _fibonacci_sphere()
    near = np.array([0.0, 0.0, 1.0])
    far = np.array([0.0, 1.0, 0.0])
    f = _gaussian_peak(pts, near, math.radians(30.0), 1.0) + _gaussian_peak(
        pts, far, math.radians(30.0), 0.9
    )
    port = np.array([0.2, 0.0, 0.98])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[
            _max_target(near, cp_id="MAX_NEAR"),
            _max_target(far, cp_id="MAX_FAR"),
        ],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_NEAR"


def test_curved_ridge_reached_where_a_straight_ray_would_miss() -> None:
    """The Step 4 case: the ridge to the maximum curves, so a straight ray aimed
    along the port's outward great circle does not follow it.

    Two peaks of unequal height create a saddle between them; a port placed off
    the symmetry plane climbs the curved ridge to the higher peak. The ascent
    arrives, and its path bows away from the straight great circle joining the
    port to the peak -- the curvature the geodesic connector cannot represent.
    """
    pts = _fibonacci_sphere()
    hi = np.array([0.0, 0.0, 1.0])  # the target maximum
    lo = np.array([math.sin(1.2), 0.0, math.cos(1.2)])  # a lower peak, ~69 deg on
    f = _gaussian_peak(pts, hi, math.radians(35.0), 1.0) + _gaussian_peak(
        pts, lo, math.radians(35.0), 0.7
    )
    # Off the y=0 symmetry plane, so the ascent ridge is not a great circle
    # through the two peaks.
    port = np.array([0.35, 0.62, 0.70])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(hi, radius_deg=10.0, cp_id="MAX_HI")],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_HI"

    # The path bows off the straight port->peak great circle: measure the largest
    # angular deviation of an interior point from that plane. A geodesic connector
    # has zero deviation; a curved ridge does not.
    n = np.cross(port, hi)
    n = n / np.linalg.norm(n)
    interior = np.array(out.path[1:-1]) / _R
    dev = np.abs(np.arccos(np.clip(interior @ n, -1.0, 1.0)) - math.pi / 2)
    assert math.degrees(float(dev.max())) > 5.0


def test_flat_plateau_stalls_without_a_maximum() -> None:
    """A constant field has no gradient: the ascent stalls, it does not crash."""
    pts = _fibonacci_sphere()
    f = np.full(len(pts), 0.5)
    port = np.array([0.0, 0.0, 1.0])

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(np.array([0.0, 1.0, 0.0]))],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "stalled"
    assert out.max_id is None
    assert out.n_steps <= STALL_PATIENCE + 2


def test_budget_exhausted_on_a_long_spiral() -> None:
    """A tiny budget cannot reach a distant peak: the answer is budget, not arrival."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(80.0))
    port = np.array([1.0, 0.0, 0.02])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(peak, radius_deg=4.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        max_arc_pitch=2.0,  # 2 pitches, far short of the ~86 deg to the peak
    )
    assert out.status == "budget"
    assert out.max_id is None


def test_degenerate_start_is_reported_not_raised() -> None:
    """A zero-length port has no direction: status no_start, empty path."""
    pts = _fibonacci_sphere()
    f = _gaussian_peak(pts, np.array([0.0, 0.0, 1.0]), math.radians(30.0))

    out = trace_uphill(
        start_position=(0.0, 0.0, 0.0),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]))],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "no_start"
    assert out.path == ()


def test_records_non_maximum_cps_the_path_crosses() -> None:
    """A saddle/min disc the ridge runs through is recorded, not hidden."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(45.0))
    port = np.array([0.5, 0.0, 0.87])
    port = port / np.linalg.norm(port)
    # A minimum sitting on the path between the port and the peak.
    mid = np.array([0.28, 0.0, 0.96])
    mid = mid / np.linalg.norm(mid)
    targets = [
        _max_target(peak, radius_deg=8.0),
        _Target("MIN_7", "minimum", mid, math.radians(6.0)),
    ]

    out = trace_uphill(
        start_position=port,
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert "MIN_7" in out.visited_ids


def test_path_is_emitted_in_the_world_frame() -> None:
    """With a nucleus offset, every path point sits on the sphere about it."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(40.0))
    port = np.array([0.4, 0.0, 0.92])
    port = port / np.linalg.norm(port)
    center = (1.2618, -0.0001, -0.0002)  # ethene3-like nucleus offset

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(peak)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
    )
    assert out.status == "arrived"
    c = np.array(center)
    radii = [math.dist(p, c) for p in out.path]
    assert all(abs(r - _R) < 1e-6 for r in radii)


def test_centred_targets_maps_world_positions_to_unit_radial() -> None:
    """centred_targets subtracts the nucleus and normalises, so radius_toward is
    measured from the correct centre."""
    from gba_topology2.src.max_connection import ConnectionTarget

    center = (2.0, 0.0, 0.0)
    world = np.array([2.0, 0.0, 3.0])  # centred: (0,0,3) -> unit (0,0,1)
    t = ConnectionTarget(
        cp_id="MAX_0",
        polarity="maximum",
        position=world,
        value=1.0,
        angular_radius_rad=math.radians(10.0),
    )

    (ct,) = centred_targets([t], center)
    assert np.allclose(ct.position, [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(ct.position), 1.0)


def test_centred_targets_drops_a_degenerate_entry() -> None:
    """A target sitting exactly on the nucleus has no direction; it is dropped."""
    from gba_topology2.src.max_connection import ConnectionTarget

    center = (1.0, 1.0, 1.0)
    t = ConnectionTarget(
        cp_id="MAX_0",
        polarity="maximum",
        position=np.array([1.0, 1.0, 1.0]),
        value=1.0,
        angular_radius_rad=math.radians(10.0),
    )
    assert centred_targets([t], center) == []


def test_centred_targets_none_center_is_identity_normalise() -> None:
    """center=None means the pool is already centred: only the normalise happens."""
    from gba_topology2.src.max_connection import ConnectionTarget

    t = ConnectionTarget(
        cp_id="MAX_0",
        polarity="maximum",
        position=np.array([0.0, 2.0, 0.0]),
        value=1.0,
        angular_radius_rad=math.radians(10.0),
    )
    (ct,) = centred_targets([t], None)
    assert np.allclose(ct.position, [0.0, 1.0, 0.0])


def test_constants_are_positive_and_ordered() -> None:
    """The step is finer than the budget, so a trace takes many steps."""
    assert TRACE_STEP_PITCH > 0.0
    assert MAX_TRACE_ARC_PITCH > TRACE_STEP_PITCH
    assert STALL_PATIENCE >= 1


def test_outcome_is_frozen_and_defaults_visited_empty() -> None:
    """TraceOutcome is an immutable value object; visited_ids defaults to empty."""
    out = TraceOutcome(
        status="arrived",
        max_id="MAX_0",
        path=(),
        arc_length_deg=1.0,
        n_steps=0,
        start_value=0.0,
        end_value=1.0,
    )
    assert out.visited_ids == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.status = "stalled"  # type: ignore[misc]
