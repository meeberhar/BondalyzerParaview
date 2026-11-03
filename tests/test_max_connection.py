"""Unit tests for the Maximum Connection subroutine (Basin Locator Step 3).

The behaviour under test is the FIRST-HIT rule: an ascent branch claims the
nearest critical point it runs into along its great-circle ray, provided that
point is a maximum and lies inside the forward sector. The previous rule
minimised cross-track (aim) and never checked reach, so a ray passing through a
near maximum on its way to a far one claimed the far one -- both sit at
cross-track ~0, and the tie-break was float noise. These tests pin the fix.

Every fixture is a unit sphere with the saddle at the north pole, so a "chord"
ray from the pole toward a port at azimuth phi is the meridian phi: along-track
is the colatitude and cross-track is the great-circle distance to that meridian.

The frame tests exist because Pd_20K sits at the origin: a bug that mixes the
centred seed frame with the world-frame pool is invisible there and catastrophic
on ethene. They run the same geometry offset from the origin and require
identical answers.
"""

import math
import random

import numpy as np
import pytest

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import (
    ARC_SAMPLES,
    CONE_PITCH,
    CONNECTION_STATUSES,
    DIRECTION_MODES,
    ConnectionTarget,
    DirectionMode,
    basin_centers_to_targets,
    build_connection_pool,
    connect_seed_to_maxima,
    connection_ray,
    mark_mutual,
    ray_frame,
    seed_ray_direction,
    summarize_connections,
    target_frame_angles,
)
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.sphere_frame import radial_offset, to_global

R = 1.0
PITCH = math.radians(2.0)
CENTER = (2.0, -1.5, 0.8)
# Default forward-sector half-angle in degrees, at the module constant.
CONE_DEG = CONE_PITCH * math.degrees(PITCH)


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


def _target(
    cp_id: str,
    polarity: str,
    colat_deg: float,
    az_deg: float,
    value: float = 0.0,
) -> ConnectionTarget:
    """A raw pool entry (a saddle, typically) at a spherical position."""
    return ConnectionTarget(
        cp_id=cp_id,
        polarity=polarity,
        position=np.array(_pt(colat_deg, az_deg), dtype=np.float64),
        value=value,
    )


def _pool(
    *maxima: BasinCenter, extra: list[ConnectionTarget] | None = None
) -> list[ConnectionTarget]:
    """Basin centers -> targets, with optional extra (saddle) targets."""
    out = basin_centers_to_targets(list(maxima))
    if extra:
        out.extend(extra)
    return out


def test_chord_ray_bears_toward_its_port() -> None:
    """The chord ray from the pole aims at the port's azimuth."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    ray = connection_ray(seed, seed.ascent_branches[0], "chord", R)
    assert ray is not None
    assert np.allclose(ray.origin_unit, [0.0, 0.0, 1.0])
    port = np.array(_pt(20.0, 35.0), dtype=np.float64)
    assert abs(float(np.dot(port / np.linalg.norm(port), ray.pole_unit))) < 1e-12
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
    """The two modes start at different points, as documented."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    chord = connection_ray(seed, seed.ascent_branches[0], "chord", R)
    port = connection_ray(seed, seed.ascent_branches[0], "port_tangent", R)
    assert chord is not None and port is not None
    assert np.allclose(chord.origin_unit, [0.0, 0.0, 1.0])
    assert np.allclose(port.origin_unit, np.array(_pt(20.0, 0.0)), atol=1e-12)
    assert float(np.dot(port.direction_unit, port.origin_unit)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(np.dot(port.direction_unit, [0.0, 0.0, 1.0])) < 0.0


def test_target_frame_angles_matches_geometry() -> None:
    """Closed form on a meridian ray: along = atan2(sc, c), cross = asin(s*saz)."""
    ray = ray_frame(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))

    def _expect(colat_deg: float, az_deg: float) -> tuple[float, float]:
        t, a = math.radians(colat_deg), math.radians(az_deg)
        return (
            math.degrees(math.atan2(math.sin(t) * math.cos(a), math.cos(t))),
            math.degrees(math.asin(math.sin(t) * math.sin(a))),
        )

    for colat, az in ((30.0, 12.0), (30.0, -12.0), (89.0, 5.0), (10.0, 170.0)):
        along, cross = target_frame_angles(np.array(_pt(colat, az)), ray)
        w_along, w_cross = _expect(colat, az)
        assert math.degrees(along) == pytest.approx(w_along, abs=1e-9)
        assert math.degrees(cross) == pytest.approx(w_cross, abs=1e-9)

    along, cross = target_frame_angles(np.array(_pt(37.0, 0.0)), ray)
    assert math.degrees(along) == pytest.approx(37.0, abs=1e-9)
    assert math.degrees(cross) == pytest.approx(0.0, abs=1e-9)
    along_back, _ = target_frame_angles(np.array(_pt(160.0, 180.0)), ray)
    assert along_back < 0.0


# --- The first-hit rule: the regression the user reported ---------------------


def test_first_hit_beats_farther_on_axis() -> None:
    """A nearer maximum on the ray wins over a far one with cross-track ~0.

    This is the MC109 / MC8 case. The ray passes through MC109 (near) toward
    MC8 (far); both sit at cross-track ~0, so the old cross-track-minimising
    rule picked the far one on float noise. The first-hit rule picks the near
    one, and the maxima beyond MC109 are no longer reachable.
    """
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(
        _max("MAX_NEAR", 15.0, 0.001),  # along ~15, cross ~0.0003
        _max("MAX_FAR", 30.0, 0.0),  # along ~30, cross ~0
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].max_id == "MAX_NEAR"
    assert conns[0].status == "connected"
    assert conns[0].along_track_deg == pytest.approx(15.0, abs=0.05)


def test_minimum_on_the_ray_occludes_the_maximum_behind_it() -> None:
    """A minimum between the saddle and a maximum blocks the claim."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("MIN_BLOCK", "minimum", 15.0, 0.0)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "occluded"
    assert conns[0].max_id is None
    assert conns[0].blocker_id == "MIN_BLOCK"
    assert conns[0].blocker_polarity == "minimum"
    # The arc is drawn to the blocker, not past it.
    assert conns[0].along_track_deg == pytest.approx(15.0, abs=0.05)


def test_saddle_on_the_ray_occludes() -> None:
    """A saddle in the path is a separatrix crossing: the branch is occluded."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("SAD_BLOCK", "saddle", 12.0, 0.0)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "occluded"
    assert conns[0].blocker_id == "SAD_BLOCK"
    assert conns[0].blocker_polarity == "saddle"


def test_first_hit_is_a_maximum_so_it_connects() -> None:
    """When the nearest thing in the sector is a maximum, the branch connects."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(
        _max("MAX_NEAR", 15.0, 0.0),
        extra=[_target("SAD_FAR", "saddle", 25.0, 0.0)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_NEAR"
    assert conns[0].blocker_id is None


def test_source_saddle_is_not_its_own_target() -> None:
    """A branch never claims the saddle it launched from.

    The pool carries the source saddle (it is a visible critical point), so the
    search must skip it by id. Without the skip, a saddle target sitting along
    the ray would occlude the branch's own claim.
    """
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(
        _max("MAX_20", 20.0, 0.0),
        extra=[_target("SAD_1", "saddle", 10.0, 0.0)],  # the source, on the ray
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_20"


# --- The forward sector (cone) gate -------------------------------------------


def test_cone_gate_excludes_an_off_sector_near_target() -> None:
    """A nearer target outside the sector loses to a farther one inside it.

    The cone is what keeps the rule directional: without it, first-hit would
    claim the nearest critical point in a wide forward hemisphere, which is
    plain proximity again.
    """
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    # (25, 60): along ~13, cross ~21.5 -> outside a 12 deg cone.
    pool = _pool(
        _max("OFF_SECTOR", 25.0, 60.0),
        _max("ON_AXIS", 30.0, 0.0),
    )
    assert CONE_DEG < 21.5  # the fixture is only meaningful if the cone is tight
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].max_id == "ON_AXIS"


def test_cone_gate_with_nothing_inside_is_no_target() -> None:
    """A branch whose only forward critical point is off-sector finds nothing."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(_max("OFF_SECTOR", 25.0, 60.0))
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "no_target"
    assert conns[0].max_id is None
    assert conns[0].blocker_id is None


def test_arc_cap_produces_no_target() -> None:
    """A maximum past the arc cap is not claimed, and the record is explicit."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(_max("FAR", 20.0, 0.0))
    conns = connect_seed_to_maxima([seed], pool, R, PITCH, max_arc_pitch=5.0)
    assert conns[0].status == "no_target"
    assert conns[0].arc_points == ()
    # The default cap is generous enough to claim it.
    assert connect_seed_to_maxima([seed], pool, R, PITCH)[0].max_id == "FAR"


def test_rear_hemisphere_is_never_claimed() -> None:
    """A maximum behind the ray's origin is not a forward hit."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(_max("BEHIND", 160.0, 180.0))
    assert connect_seed_to_maxima([seed], pool, R, PITCH)[0].status == "no_target"


def test_opposite_branches_claim_opposite_maxima() -> None:
    """Two branches 180 deg apart must not collapse onto the same extremum."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0), _branch(1, 20.0, 180.0)])
    pool = _pool(_max("EAST", 25.0, 3.0), _max("WEST", 25.0, 177.0))
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert [c.max_id for c in conns] == ["EAST", "WEST"]


def test_only_ascent_branches_are_connected() -> None:
    """Descent branches head for minima and are excluded from the table."""
    seed = _seed(
        "SAD_1",
        [_branch(0, 20.0, 0.0), _branch(0, 20.0, 90.0, kind="descent")],
    )
    pool = _pool(_max("A", 20.0, 0.0), _max("B", 20.0, 90.0))
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert len(conns) == 1
    assert conns[0].max_id == "A"


# --- Pool construction (the frame trap) ---------------------------------------


def test_build_connection_pool_merges_extrema_and_saddles() -> None:
    """The pool carries maxima, minima AND saddles; saddles are not in basin
    centers, so build_connection_pool is the only thing that puts them in."""
    maxima = [_max("MAX_1", 20.0, 0.0)]
    minima = [
        BasinCenter(
            center_id="MIN_1",
            polarity="minimum",
            position=_pt(40.0, 0.0),
            value=-1.0,
            is_effective=False,
        )
    ]
    raw_sad = [
        CriticalPoint(vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0))
    ]
    pool = build_connection_pool([*maxima, *minima], raw_sad, [], set())
    by_id = {t.cp_id: t.polarity for t in pool}
    assert by_id == {"MAX_1": "maximum", "MIN_1": "minimum", "SAD_7": "saddle"}


def test_build_connection_pool_excludes_hidden_raw_saddles() -> None:
    """A raw saddle collapsed into an effective entity is not visible, so it
    must not occlude."""
    raw_sad = [
        CriticalPoint(
            vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0)
        ),
        CriticalPoint(
            vertex_id=8, cp_type="saddle", value=0.3, position=_pt(35.0, 0.0)
        ),
    ]
    pool = build_connection_pool([], raw_sad, [], hidden_vertex_ids={8})
    ids = {t.cp_id for t in pool}
    assert ids == {"SAD_7"}


def test_build_connection_pool_includes_effective_saddles() -> None:
    """E-SAD dicts (world frame, no vertex) enter the pool as saddles."""
    eff = [
        {
            "type": "E-SAD",
            "id_label": "E-SAD_MC7",
            "position": list(_pt(25.0, 0.0)),
            "value": 0.4,
        },
        {
            "type": "E-MAX",
            "id_label": "E-MAX_MC9",
            "position": list(_pt(50.0, 0.0)),
            "value": 1.0,
        },
    ]
    pool = build_connection_pool([], [], eff, set())
    by_id = {t.cp_id: t.polarity for t in pool}
    assert by_id == {"E-SAD_MC7": "saddle"}  # E-MAX comes via basin centers


def test_build_connection_pool_maps_raw_saddles_to_world_frame() -> None:
    """Raw saddles are centred; the pool is world frame. Pd_20K at the origin
    hides this bug, so the offset is applied explicitly and checked."""
    raw_sad = [
        CriticalPoint(vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0))
    ]
    pool = build_connection_pool([], raw_sad, [], set(), center=CENTER)
    p = pool[0].position
    assert float(np.linalg.norm(p - np.array(CENTER))) == pytest.approx(R, rel=1e-9)


def test_build_connection_pool_dedups_by_id() -> None:
    """A saddle present in two sources is scored once."""
    raw_sad = [
        CriticalPoint(vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0))
    ]
    eff = [
        {
            "type": "E-SAD",
            "id_label": "SAD_7",
            "position": list(_pt(30.0, 0.0)),
            "value": 0.3,
        }
    ]
    pool = build_connection_pool([], raw_sad, eff, set())
    assert len([t for t in pool if t.cp_id == "SAD_7"]) == 1


# --- Mutual, purity, determinism ----------------------------------------------


def test_mutual_flags_exactly_one_claim_per_maximum() -> None:
    """The tightest claim on a maximum is the mutual one, and only one is."""
    seeds = [
        _seed("SAD_A", [_branch(0, 20.0, 0.0)], position=_pt(20.0, 0.0)),
        _seed("SAD_B", [_branch(0, 20.0, 0.0)], position=_pt(60.0, 0.0)),
    ]
    pool = _pool(_max("TARGET", 20.0, 0.0))
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    connected = [c for c in conns if c.status == "connected"]
    assert len(connected) >= 1
    mutual = [c for c in conns if c.mutual]
    assert len(mutual) == len({c.max_id for c in connected})
    assert all(c.status == "connected" for c in mutual)


def test_mark_mutual_preserves_blocker_fields() -> None:
    """mark_mutual rebuilds records; it must not drop the occlusion fields."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("MIN_BLOCK", "minimum", 15.0, 0.0)],
    )
    conns = mark_mutual(connect_seed_to_maxima([seed], pool, R, PITCH))
    assert conns[0].blocker_id == "MIN_BLOCK"
    assert conns[0].blocker_polarity == "minimum"


def test_arc_points_lie_on_the_sphere_world_frame() -> None:
    """Every rendered sample sits at radius R from the nucleus, in world frame."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    pool = _pool(_max("T", 20.0, 35.0, center=CENTER))
    conns = connect_seed_to_maxima([seed], pool, R, PITCH, center=CENTER)
    arc = conns[0].arc_points
    assert len(arc) == ARC_SAMPLES + 1
    for p in arc:
        assert float(np.linalg.norm(np.array(p) - np.array(CENTER))) == pytest.approx(
            R, rel=1e-9
        )


def test_arc_is_a_great_circle() -> None:
    """All samples share one plane through the centre: zero torsion."""
    seed = _seed("SAD_1", [_branch(0, 40.0, 15.0)])
    conns = connect_seed_to_maxima([seed], _pool(_max("T", 40.0, 15.0)), R, PITCH)
    pts = np.array(conns[0].arc_points)
    normal = np.cross(pts[1] - pts[0], pts[-1] - pts[0])
    normal /= float(np.linalg.norm(normal))
    assert np.max(np.abs(pts @ normal)) < 1e-9


def test_world_frame_and_centred_runs_agree() -> None:
    """Offsetting the atom changes no metric -- the frame contract, pinned."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    pool_local = _pool(_max("T", 20.0, 35.0), _max("OFF", 12.0, 120.0))
    pool_world = _pool(
        _max("T", 20.0, 35.0, center=CENTER),
        _max("OFF", 12.0, 120.0, center=CENTER),
    )
    a = connect_seed_to_maxima([seed], pool_local, R, PITCH)
    b = connect_seed_to_maxima([seed], pool_world, R, PITCH, center=CENTER)
    assert a[0].max_id == b[0].max_id
    assert a[0].cross_track_deg == pytest.approx(b[0].cross_track_deg, abs=1e-9)
    assert a[0].along_track_deg == pytest.approx(b[0].along_track_deg, abs=1e-9)
    for pa, pb in zip(a[0].arc_points, b[0].arc_points, strict=True):
        assert np.allclose(np.array(pb) - np.array(pa), np.array(CENTER), atol=1e-9)


def test_result_is_independent_of_input_order() -> None:
    """Shuffling the pool and the seeds cannot change any claim."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0), _branch(1, 25.0, 120.0)])
    seed2 = _seed("SAD_2", [_branch(0, 30.0, 240.0)], position=_pt(5.0, 240.0))
    pool = _pool(
        _max("A", 20.0, 2.0),
        _max("B", 25.0, 121.0),
        _max("C", 30.0, 241.0),
        _max("D", 12.0, 60.0),
    )
    reference = connect_seed_to_maxima([seed, seed2], pool, R, PITCH)

    rng = random.Random(7)
    for _ in range(8):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        seeds = [seed, seed2][:]
        rng.shuffle(seeds)
        got = connect_seed_to_maxima(seeds, shuffled, R, PITCH)
        key = {
            (c.saddle_id, c.branch_index): (c.max_id, c.status, c.mutual) for c in got
        }
        want = {
            (c.saddle_id, c.branch_index): (c.max_id, c.status, c.mutual)
            for c in reference
        }
        assert key == want


def test_direction_modes_are_both_selectable() -> None:
    """Both modes run; the mode is recorded on every connection."""
    seed = _seed("SAD_1", [_branch(0, 30.0, 20.0)])
    pool = _pool(_max("T", 30.0, 20.0))
    modes: tuple[DirectionMode, ...] = ("chord", "port_tangent")
    assert set(modes) == set(DIRECTION_MODES)
    for mode in modes:
        conns = connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode=mode)
        assert conns[0].direction_mode == mode
        assert conns[0].max_id == "T"

    with pytest.raises(ValueError):
        connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode="bogus")  # type: ignore[arg-type]


def test_both_modes_share_the_same_great_circle() -> None:
    """The measured identity: same plane, same cross-track, shifted along."""
    seed = _seed("SAD_1", [_branch(0, 22.0, 47.0)])
    branch = seed.ascent_branches[0]
    chord = connection_ray(seed, branch, "chord", R)
    port = connection_ray(seed, branch, "port_tangent", R)
    assert chord is not None and port is not None
    assert abs(float(np.dot(chord.pole_unit, port.pole_unit))) == pytest.approx(
        1.0, abs=1e-12
    )
    assert not np.allclose(chord.origin_unit, port.origin_unit)

    pool = _pool(_max("T", 40.0, 48.0), _max("OFF", 40.0, 100.0))
    a = connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode="chord")
    b = connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode="port_tangent")
    assert a[0].max_id == b[0].max_id
    assert a[0].cross_track_deg is not None and b[0].cross_track_deg is not None
    assert a[0].cross_track_deg == pytest.approx(b[0].cross_track_deg, abs=1e-9)


def test_degenerate_seed_position_yields_no_target() -> None:
    """A saddle at the nucleus has no radial, hence no ray."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)], position=(0.0, 0.0, 0.0))
    assert (
        connect_seed_to_maxima([seed], _pool(_max("T", 20.0, 0.0)), R, PITCH)[0].status
        == "no_target"
    )


def test_empty_inputs() -> None:
    """No seeds or no targets produce an empty table, not an exception."""
    assert connect_seed_to_maxima([], [], R, PITCH) == []
    seed = _seed("SAD_1", [_branch(0, 20.0, 0.0)])
    assert connect_seed_to_maxima([seed], [], R, PITCH)[0].status == "no_target"


def test_summarize_connections_counts_all_three_statuses() -> None:
    """The summary separates connected / occluded / no_target, all finite."""
    seed = _seed(
        "SAD_1",
        [
            _branch(0, 20.0, 0.0),  # -> MAX (connected)
            _branch(1, 20.0, 90.0),  # -> MIN on that meridian (occluded)
            _branch(2, 20.0, 200.0),  # nothing in sector (no_target)
        ],
    )
    pool = _pool(
        _max("M0", 25.0, 1.0),
        extra=[_target("MIN90", "minimum", 25.0, 90.0)],
    )
    stats = summarize_connections(connect_seed_to_maxima([seed], pool, R, PITCH))
    assert stats["n_branches"] == 3
    assert stats["n_connected"] == 1
    assert stats["n_occluded"] == 1
    assert stats["n_no_target"] == 1
    assert stats["n_blocked_by_minimum"] == 1
    assert stats["n_blocked_by_saddle"] == 0
    assert stats["direction_mode"] == "chord"
    for key, value in stats.items():
        if isinstance(value, float):
            assert math.isfinite(value), key

    empty = summarize_connections([])
    assert empty["n_branches"] == 0
    assert empty["n_occluded"] == 0
    assert empty["mean_cross_track_deg"] == 0.0


def test_verified_flag_tracks_index_agrees() -> None:
    """Provisional (k_mismatch) seeds stay distinguishable in the table."""
    good = _seed("SAD_1", [_branch(0, 20.0, 0.0)], index_agrees=True)
    bad = _seed("SAD_2", [_branch(0, 20.0, 0.0)], index_agrees=False)
    conns = connect_seed_to_maxima([good, bad], _pool(_max("T", 20.0, 0.0)), R, PITCH)
    assert [c.verified for c in conns] == [True, False]
    assert summarize_connections(conns)["n_unverified"] == 1


def test_connection_statuses_are_the_three_verdicts() -> None:
    assert set(CONNECTION_STATUSES) == {"connected", "occluded", "no_target"}


def test_lifted_arc_endpoints_stay_radial() -> None:
    """radial_offset on the emitted arc is a true lift in the centred frame."""
    seed = _seed("SAD_1", [_branch(0, 20.0, 35.0)])
    conns = connect_seed_to_maxima([seed], _pool(_max("T", 20.0, 35.0)), R, PITCH)
    lifted = [radial_offset(p, 1.02) for p in conns[0].arc_points]
    assert all(
        float(np.linalg.norm(np.array(p))) == pytest.approx(R * 1.02, rel=1e-12)
        for p in lifted
    )
