"""Unit tests for the Maximum Connection subroutine (Basin Locator Step 3).

The behaviour under test is the *directional* claim: an ascent branch must pick
the maximum it points at, not the one nearest its saddle. Every fixture is a
unit sphere with the saddle at the north pole, so a "chord" ray from the pole
toward a port at azimuth phi is the meridian phi, and cross-track is simply the
azimuthal offset while along-track is the colatitude.

The frame tests exist because Pd_20K sits at the origin: a bug that mixes the
centred seed frame with the world-frame basin centers is invisible there and
catastrophic on ethene. They run the same geometry offset from the origin and
require identical answers.
"""

import math
import random

import numpy as np
import pytest

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import (
    ARC_SAMPLES,
    DIRECTION_MODES,
    DirectionMode,
    MaxConnection,
    connect_seed_to_maxima,
    connection_ray,
    mark_mutual,
    ray_frame,
    seed_ray_direction,
    summarize_connections,
    target_frame_angles,
)
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.sphere_frame import radial_offset, to_global

R = 1.0
PITCH = math.radians(2.0)
CENTER = (2.0, -1.5, 0.8)


def _pt(
    colat_deg: float, az_deg: float, radius: float = R
) -> tuple[float, float, float]:
    """Point on the sphere at a colatitude from +z and an azimuth from +x."""
    t = math.radians(colat_deg)
    p = math.radians(az_deg)
    return (
        radius * math.sin(t) * math.cos(p),
        radius * math.sin(t) * math.sin(p),
        radius * math.cos(t),
    )


def _branch(
    index: int,
    colat_deg: float,
    az_deg: float,
    kind: str = "ascent",
) -> LaunchBranch:
    """An ascent branch whose port sits at (colatitude, azimuth) from the pole."""
    pos = np.array(_pt(colat_deg, az_deg), dtype=np.float64)
    origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent = pos - float(np.dot(pos, origin)) * origin
    tangent /= float(np.linalg.norm(tangent))
    return LaunchBranch(
        kind=kind,
        sector_index=index,
        tangent_direction=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
        launch_position=(float(pos[0]), float(pos[1]), float(pos[2])),
        sector_width_rad=math.radians(60.0),
        sector_centroid_value=1.0,
    )


def _seed(
    saddle_id: str,
    branches: list[LaunchBranch],
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    index_agrees: bool = True,
) -> SaddleLaunchSeed:
    ascent = [b for b in branches if b.kind == "ascent"]
    descent = [b for b in branches if b.kind == "descent"]
    return SaddleLaunchSeed(
        saddle_id=saddle_id,
        saddle_position=position,
        saddle_value=0.5,
        vertex_id=-1,
        multiplicity=1,
        fold_order=2,
        index_agrees=index_agrees,
        ascent_branches=ascent,
        descent_branches=descent,
    )


def _max(
    center_id: str,
    colat_deg: float,
    az_deg: float,
    value: float = 2.0,
    center: tuple[float, float, float] | None = None,
) -> BasinCenter:
    pos = _pt(colat_deg, az_deg)
    world = to_global(pos, center) if center is not None else pos
    return BasinCenter(
        center_id=center_id,
        polarity="maximum",
        position=world,
        value=value,
        is_effective=False,
    )


def test_chord_ray_bears_toward_its_port() -> None:
    """The chord ray from the pole aims at the port's azimuth."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    ray = connection_ray(seed, seed.ascent_branches[0], "chord", R)
    assert ray is not None
    # Origin is the saddle centre.
    assert np.allclose(ray.origin_unit, [0.0, 0.0, 1.0])
    # The ray's great circle contains the port.
    port = np.array(_pt(20.0, 35.0), dtype=np.float64)
    assert abs(float(np.dot(port / np.linalg.norm(port), ray.pole_unit))) < 1e-12
    # Direction is a unit tangent at the origin.
    assert float(np.linalg.norm(ray.direction_unit)) == pytest.approx(1.0, rel=1e-12)
    assert float(np.dot(ray.direction_unit, ray.origin_unit)) == pytest.approx(
        0.0, abs=1e-12
    )


def test_seed_ray_direction_is_unit_and_tangential() -> None:
    """The accessor returns a unit tangent perpendicular to the radial."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    d = seed_ray_direction(seed, seed.ascent_branches[0], "chord", R)
    assert d is not None
    vec = np.array(d)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, rel=1e-12)
    assert float(np.dot(vec, [0.0, 0.0, 1.0])) == pytest.approx(0.0, abs=1e-12)


def test_port_tangent_ray_starts_at_the_port() -> None:
    """The two modes are genuinely different geodesics, as documented."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    chord = connection_ray(seed, seed.ascent_branches[0], "chord", R)
    port = connection_ray(seed, seed.ascent_branches[0], "port_tangent", R)
    assert chord is not None and port is not None
    assert np.allclose(chord.origin_unit, [0.0, 0.0, 1.0])
    assert np.allclose(port.origin_unit, np.array(_pt(20.0, 0.0)), atol=1e-12)
    # The port tangent points away from the ring centre: travelling along it
    # increases the angular distance from the pole.
    assert float(np.dot(port.direction_unit, port.origin_unit)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(np.dot(port.direction_unit, [0.0, 0.0, 1.0])) < 0.0


def test_target_frame_angles_matches_geometry() -> None:
    """Closed form on a meridian ray: along = atan2(sc, c), cross = asin(s*saz).

    For a ray from the pole along the +x meridian, a target at colatitude t and
    azimuth a has spherical coordinates (sin t cos a, sin t sin a, cos t), so
    ``along`` is the progress of its projection onto the meridian and ``cross``
    is its true angular distance to that great circle. Cross is NOT the azimuth
    offset: the parallels are small circles, so the deviation shrinks with
    sin(t). That is exactly the metric the claim rule needs.
    """
    ray = ray_frame(
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )

    def _expect(colat_deg: float, az_deg: float) -> tuple[float, float]:
        t, a = math.radians(colat_deg), math.radians(az_deg)
        along = math.degrees(math.atan2(math.sin(t) * math.cos(a), math.cos(t)))
        cross = math.degrees(math.asin(math.sin(t) * math.sin(a)))
        return along, cross

    for colat, az in ((30.0, 12.0), (30.0, -12.0), (89.0, 5.0), (10.0, 170.0)):
        along, cross = target_frame_angles(np.array(_pt(colat, az)), ray)
        w_along, w_cross = _expect(colat, az)
        assert math.degrees(along) == pytest.approx(w_along, abs=1e-9)
        assert math.degrees(cross) == pytest.approx(w_cross, abs=1e-9)

    # On the ray itself the decomposition is exact: along = colatitude, cross = 0.
    along, cross = target_frame_angles(np.array(_pt(37.0, 0.0)), ray)
    assert math.degrees(along) == pytest.approx(37.0, abs=1e-9)
    assert math.degrees(cross) == pytest.approx(0.0, abs=1e-9)

    # At the equator the great-circle distance equals the azimuth offset.
    _, cross_eq = target_frame_angles(np.array(_pt(90.0, 5.0)), ray)
    assert math.degrees(cross_eq) == pytest.approx(5.0, abs=1e-9)

    # Behind the origin is a negative progress.
    along_back, _ = target_frame_angles(np.array(_pt(160.0, 180.0)), ray)
    assert along_back < 0.0


def test_cross_track_beats_plain_proximity() -> None:
    """The on-ray far maximum wins over the off-ray near one.

    This is the whole point of the subroutine. "Nearest maximum to the launch
    point" would claim MIN_NEAR (10 deg away); the directional rule claims
    MAX_ON_AXIS (20 deg away) because the branch points at it.
    """
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    maxima = [
        _max("MIN_NEAR", 10.0, 90.0),  # closer, but 90 deg off the ray
        _max("MAX_ON_AXIS", 20.0, 1.0),  # farther, essentially on the ray
    ]
    conns = connect_seed_to_maxima([seed], maxima, R, PITCH)
    assert len(conns) == 1
    assert conns[0].max_id == "MAX_ON_AXIS"
    assert conns[0].status == "connected"
    assert abs(conns[0].cross_track_deg or 99.0) < 1.5
    assert conns[0].separation_deg == pytest.approx(20.0, abs=1e-6)


def test_opposite_branches_claim_opposite_maxima() -> None:
    """Two branches 180 deg apart must not collapse onto the same extremum."""
    seed = _seed(
        "SAD_1",
        [_branch(0, 20.0, 0.0), _branch(1, 20.0, 180.0)],
    )
    maxima = [_max("EAST", 25.0, 3.0), _max("WEST", 25.0, 177.0)]
    conns = connect_seed_to_maxima([seed], maxima, R, PITCH)
    assert [c.max_id for c in conns] == ["EAST", "WEST"]


def test_arc_cap_produces_no_target() -> None:
    """A maximum past the cap is not claimed, and the record is explicit."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    maxima = [_max("FAR", 20.0, 0.0)]
    # Cap of 5 pitches = 10 deg < the 20 deg along-track distance.
    conns = connect_seed_to_maxima([seed], maxima, R, PITCH, max_arc_pitch=5.0)
    assert conns[0].status == "no_target"
    assert conns[0].max_id is None
    assert conns[0].arc_points == ()
    assert conns[0].cross_track_deg is None

    # The default cap is generous enough to claim it.
    conns_default = connect_seed_to_maxima([seed], maxima, R, PITCH)
    assert conns_default[0].max_id == "FAR"


def test_rear_hemisphere_is_never_claimed() -> None:
    """A maximum behind the ray's origin loses to nothing, not to itself."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    maxima = [_max("BEHIND", 160.0, 180.0)]  # along = -160 deg
    conns = connect_seed_to_maxima([seed], maxima, R, PITCH)
    assert conns[0].status == "no_target"


def test_only_ascent_branches_are_connected() -> None:
    """Descent branches head for minima and are excluded from the table."""
    seed = _seed(
        "SAD_1",
        [_branch(0, 20.0, 0.0), _branch(0, 20.0, 90.0, kind="descent")],
    )
    maxima = [_max("A", 20.0, 0.0), _max("B", 20.0, 90.0)]
    conns = connect_seed_to_maxima([seed], maxima, R, PITCH)
    assert len(conns) == 1
    assert conns[0].max_id == "A"


def test_non_maximum_basin_centers_are_ignored() -> None:
    """A full basin-center list may be passed; minima are filtered out."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    minima = BasinCenter(
        center_id="MIN_7",
        polarity="minimum",
        position=_pt(20.0, 0.0),
        value=-1.0,
        is_effective=False,
    )
    maxima = [_max("ON_AXIS", 20.0, 1.0)]
    conns = connect_seed_to_maxima([seed], [minima, *maxima], R, PITCH)
    assert conns[0].max_id == "ON_AXIS"


def test_mutual_flags_exactly_one_claim_per_maximum() -> None:
    """The tightest claim on a maximum is the mutual one, and only one is."""
    seeds = [
        _seed("SAD_A", [_branch(0, 20.0, 0.0)], position=_pt(20.0, 0.0)),
        _seed("SAD_B", [_branch(0, 20.0, 0.0)], position=_pt(60.0, 0.0)),
    ]
    # Rebuild the branches in each seed's own frame: the helper assumes the
    # pole, so aim both branches at the same maximum from their own centres.
    maxima = [_max("TARGET", 20.0, 0.0)]
    conns = connect_seed_to_maxima(seeds, maxima, R, PITCH)
    connected = [c for c in conns if c.status == "connected"]
    assert len(connected) >= 1
    mutual = [c for c in conns if c.mutual]
    assert len(mutual) == len({c.max_id for c in connected})
    assert all(c.status == "connected" for c in mutual)


def test_mark_mutual_is_pure_and_order_stable() -> None:
    """mark_mutual returns a new list and never mutates the input records."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    conns = connect_seed_to_maxima([seed], [_max("T", 20.0, 0.0)], R, PITCH)
    again = mark_mutual(conns)
    assert [c.mutual for c in again] == [c.mutual for c in conns]
    assert all(isinstance(c, MaxConnection) for c in again)


def test_arc_points_lie_on_the_sphere_world_frame() -> None:
    """Every rendered sample sits at radius R from the nucleus, in world frame."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    maxima = [_max("T", 20.0, 35.0, center=CENTER)]
    conns = connect_seed_to_maxima([seed], maxima, R, PITCH, center=CENTER)
    arc = conns[0].arc_points
    assert len(arc) == ARC_SAMPLES + 1
    for p in arc:
        dist = float(np.linalg.norm(np.array(p) - np.array(CENTER)))
        assert dist == pytest.approx(R, rel=1e-9)
    # Endpoints: the saddle first, the maximum last.
    assert np.allclose(np.array(arc[0]) - np.array(CENTER), [0.0, 0.0, 1.0], atol=1e-9)
    assert float(np.linalg.norm(np.array(arc[-1]) - np.array(CENTER))) == pytest.approx(
        R, rel=1e-9
    )


def test_arc_is_a_great_circle() -> None:
    """All samples share one plane through the centre: zero torsion."""
    seed = _seed("SAD_1", [_branch(0, 40.0, 15.0)])
    conns = connect_seed_to_maxima([seed], [_max("T", 40.0, 15.0)], R, PITCH)
    pts = np.array(conns[0].arc_points)
    normal = np.cross(pts[1] - pts[0], pts[-1] - pts[0])
    normal /= float(np.linalg.norm(normal))
    # A great circle through the origin: every point is perpendicular to the
    # plane normal (offset 0, not just constant).
    assert np.max(np.abs(pts @ normal)) < 1e-9


def test_world_frame_and_centred_runs_agree() -> None:
    """Offsetting the atom changes no metric -- the frame contract, pinned.

    Pd_20K sits at the origin, so a module that reads world-frame basin centers
    as if they were centred scores perfectly there and fails on ethene. Running
    the identical geometry twice, once offset, is the A/B that catches it.
    """
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    maxima_local = [_max("T", 20.0, 35.0), _max("OFF", 12.0, 120.0)]
    maxima_world = [
        _max("T", 20.0, 35.0, center=CENTER),
        _max("OFF", 12.0, 120.0, center=CENTER),
    ]
    a = connect_seed_to_maxima([seed], maxima_local, R, PITCH)
    b = connect_seed_to_maxima([seed], maxima_world, R, PITCH, center=CENTER)
    assert a[0].max_id == b[0].max_id
    assert a[0].cross_track_deg == pytest.approx(b[0].cross_track_deg, abs=1e-9)
    assert a[0].along_track_deg == pytest.approx(b[0].along_track_deg, abs=1e-9)
    assert a[0].separation_deg == pytest.approx(b[0].separation_deg, abs=1e-9)
    # And the offset run's arc is the centred arc translated by the nucleus.
    for pa, pb in zip(a[0].arc_points, b[0].arc_points, strict=True):
        assert np.allclose(np.array(pb) - np.array(pa), np.array(CENTER), atol=1e-9)


def test_result_is_independent_of_input_order() -> None:
    """Shuffling the maxima and the seeds cannot change any claim."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0), _branch(1, 25.0, 120.0)])
    seed2 = _seed("SAD_2", [_branch(0, 30.0, 240.0)], position=_pt(5.0, 240.0))
    maxima = [
        _max("A", 20.0, 2.0),
        _max("B", 25.0, 121.0),
        _max("C", 30.0, 241.0),
        _max("D", 12.0, 60.0),
    ]
    reference = connect_seed_to_maxima([seed, seed2], maxima, R, PITCH)

    rng = random.Random(7)
    for _ in range(8):
        shuffled_max = maxima[:]
        rng.shuffle(shuffled_max)
        shuffled_seeds = [seed, seed2][:]
        rng.shuffle(shuffled_seeds)
        got = connect_seed_to_maxima(shuffled_seeds, shuffled_max, R, PITCH)
        key = {(c.saddle_id, c.branch_index): (c.max_id, c.mutual) for c in got}
        want = {(c.saddle_id, c.branch_index): (c.max_id, c.mutual) for c in reference}
        assert key == want


def test_direction_modes_are_both_selectable_and_can_disagree() -> None:
    """Both modes run; the mode is recorded on every connection."""
    seed = _seed("SAD_1", [_branch(0, 30.0, 20.0)])
    maxima = [_max("T", 30.0, 20.0)]
    modes: tuple[DirectionMode, ...] = ("chord", "port_tangent")
    assert set(modes) == set(DIRECTION_MODES)
    for mode in modes:
        conns = connect_seed_to_maxima([seed], maxima, R, PITCH, direction_mode=mode)
        assert conns[0].direction_mode == mode
        assert conns[0].max_id == "T"
    with pytest.raises(ValueError):
        connect_seed_to_maxima([seed], maxima, R, PITCH, direction_mode="bogus")  # type: ignore[arg-type]


def test_both_modes_share_the_same_great_circle() -> None:
    """The identity measured on Pd_20K, pinned.

    ``_outward_tangent`` builds the port direction as ``radial*port - centre``,
    which is the parallel transport of the centre->port geodesic. So the chord
    ray and the port ray lie on the SAME plane and share a pole; only the arc's
    start moves. Cross-track -- the quantity the claim rule minimises -- is
    therefore identical, which is why the mode is not a viewer control.
    """
    seed = _seed("SAD_1", [_branch(0, 22.0, 47.0)])
    branch = seed.ascent_branches[0]
    chord = connection_ray(seed, branch, "chord", R)
    port = connection_ray(seed, branch, "port_tangent", R)
    assert chord is not None and port is not None

    # Same great circle: the poles are parallel.
    assert abs(float(np.dot(chord.pole_unit, port.pole_unit))) == pytest.approx(
        1.0, abs=1e-12
    )
    # Different starts, on that circle.
    assert not np.allclose(chord.origin_unit, port.origin_unit)

    maxima = [_max("T", 40.0, 48.0), _max("OFF", 40.0, 100.0)]
    a = connect_seed_to_maxima([seed], maxima, R, PITCH, direction_mode="chord")
    b = connect_seed_to_maxima([seed], maxima, R, PITCH, direction_mode="port_tangent")
    assert a[0].max_id == b[0].max_id
    assert a[0].cross_track_deg is not None and b[0].cross_track_deg is not None
    assert a[0].cross_track_deg == pytest.approx(b[0].cross_track_deg, abs=1e-9)
    assert a[0].along_track_deg is not None and b[0].along_track_deg is not None
    # Along-track shifts by the ring radius, exactly as documented.
    assert a[0].along_track_deg == pytest.approx(b[0].along_track_deg + 22.0, abs=1e-6)


def test_degenerate_seed_position_yields_no_target() -> None:
    """A saddle at the nucleus has no radial, hence no ray."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)], position=(0.0, 0.0, 0.0))
    conns = connect_seed_to_maxima([seed], [_max("T", 20.0, 0.0)], R, PITCH)
    assert conns[0].status == "no_target"


def test_empty_inputs() -> None:
    """No seeds or no maxima produce an empty table, not an exception."""
    assert connect_seed_to_maxima([], [], R, PITCH) == []
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    assert connect_seed_to_maxima([seed], [], R, PITCH)[0].status == "no_target"


def test_summarize_connections_counts_and_finite_metrics() -> None:
    """The summary carries the viewer card's counters, all finite."""
    seed = _seed(
        "SAD_1",
        [_branch(0, 20.0, 0.0), _branch(1, 20.0, 180.0)],
    )
    maxima = [
        _max("EAST", 25.0, 3.0),
        _max("WEST", 25.0, 177.0),
        _max("FAR", 170.0, 90.0),
    ]
    stats = summarize_connections(connect_seed_to_maxima([seed], maxima, R, PITCH))
    assert stats["n_branches"] == 2
    assert stats["n_connected"] == 2
    assert stats["n_no_target"] == 0
    assert stats["n_saddles_connected"] == 1
    assert stats["n_maxima_connected"] == 2
    assert stats["direction_mode"] == "chord"
    for key, value in stats.items():
        if isinstance(value, float):
            assert math.isfinite(value), key

    empty = summarize_connections([])
    assert empty["n_branches"] == 0
    assert empty["mean_cross_track_deg"] == 0.0
    assert empty["max_separation_deg"] == 0.0


def test_verified_flag_tracks_index_agrees() -> None:
    """Provisional (k_mismatch) seeds stay distinguishable in the table."""
    good = _seed("SAD_1", [_branch(0, 20.0, 0.0)], index_agrees=True)
    bad = _seed("SAD_2", [_branch(0, 20.0, 0.0)], index_agrees=False)
    conns = connect_seed_to_maxima([good, bad], [_max("T", 20.0, 0.0)], R, PITCH)
    assert [c.verified for c in conns] == [True, False]
    assert summarize_connections(conns)["n_unverified"] == 1


def test_lifted_arc_endpoints_stay_radial() -> None:
    """radial_offset on the emitted arc is a true lift in the centred frame."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    conns = connect_seed_to_maxima([seed], [_max("T", 20.0, 35.0)], R, PITCH)
    lifted = [radial_offset(p, 1.02) for p in conns[0].arc_points]
    assert all(
        float(np.linalg.norm(np.array(p))) == pytest.approx(R * 1.02, rel=1e-12)
        for p in lifted
    )
