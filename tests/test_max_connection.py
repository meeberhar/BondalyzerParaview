"""Unit tests for the Maximum Connection subroutine (Basin Locator Step 3).

The rule under test: an ascent branch launches from its PORT on the saddle's
excision boundary, follows that great circle outward, and claims the FIRST
critical point whose excision DISC it enters, inside the forward sector. First
hit a maximum -> connect; a minimum or saddle -> occluded.

Two design points the fixtures encode, both learned from real fields:

- The ray starts at the port, not the saddle centre. The disc inside the
  boundary is the degenerate core whose field is meaningless, so nothing in it
  is a candidate. A small ring radius (RING_DEG) is used so a target's
  along-track is (colatitude - RING_DEG) on the meridian.
- A critical point is a DISC of radius angular_radius_rad, not a point. A great
  circle that passes outside the disc never hits it. This is the MC18 case: a
  branch was blocked by a minimum at cross-track 7.4 deg with a 1.6 deg disc --
  the ray sails past it, so it is not an obstacle, and the maximum at cross 0.4
  with a 2.9 deg disc is the true first hit.

Every fixture is a unit sphere with the saddle at the north pole, so a branch at
azimuth phi is the meridian phi. The frame tests run the same geometry offset
from the origin and require identical answers, because Pd_20K sits at the origin
and hides any centred/world mixup.
"""

import math
import random

import numpy as np
import pytest

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import (
    CONE_PITCH,
    CONNECTION_STATUSES,
    DIRECTION_MODES,
    ConnectionTarget,
    DirectionMode,
    basin_centers_to_targets,
    build_connection_pool,
    connect_seed_to_maxima,
    connection_ray,
    disc_entry_angle,
    mark_mutual,
    summarize_connections,
    target_frame_angles,
)
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.sphere_frame import radial_offset, to_global

R = 1.0
PITCH = math.radians(2.0)
CENTER = (2.0, -1.5, 0.8)
# Excision ring radius used by the fixtures: small, so a target on the meridian
# at colatitude t has along-track (t - RING_DEG) from the port.
RING_DEG = 3.0
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
    az_deg: float,
    kind: str = "ascent",
    ring_deg: float = RING_DEG,
) -> LaunchBranch:
    """An ascent branch whose port sits at (ring_deg, az_deg) from the pole."""
    pos = np.array(_pt(ring_deg, az_deg), dtype=np.float64)
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
    ring_deg: float = RING_DEG,
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
        ring_radius_rad=math.radians(ring_deg),
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
    disc_deg: float = 0.0,
) -> ConnectionTarget:
    """A raw pool entry (a saddle/minimum) at a spherical position, with a disc."""
    return ConnectionTarget(
        cp_id=cp_id,
        polarity=polarity,
        position=np.array(_pt(colat_deg, az_deg), dtype=np.float64),
        value=value,
        angular_radius_rad=math.radians(disc_deg),
    )


def _pool(
    *maxima: BasinCenter,
    extra: list[ConnectionTarget] | None = None,
    disc_deg: float = 0.0,
) -> list[ConnectionTarget]:
    """Basin centers -> targets (with a uniform disc), plus extra targets."""
    out = basin_centers_to_targets(
        list(maxima), {m.center_id: math.radians(disc_deg) for m in maxima}
    )
    if extra:
        out.extend(extra)
    return out


# --- disc_entry_angle: the geometry primitive ---------------------------------


def test_disc_entry_is_the_first_touch_of_the_great_circle() -> None:
    """Entry = centre along-track minus the half-chord, and None when missed."""
    # Target dead on the ray (cross 0): entry is (along - rho).
    entry = disc_entry_angle(math.radians(30.0), 0.0, math.radians(5.0))
    assert entry is not None
    assert math.degrees(entry) == pytest.approx(25.0, abs=1e-9)

    # A great circle that passes outside the disc never hits it.
    assert (
        disc_entry_angle(math.radians(30.0), math.radians(8.0), math.radians(5.0))
        is None
    )

    # Exactly tangent: |cross| == rho grazes the disc, entry == along.
    tangent = disc_entry_angle(math.radians(30.0), math.radians(5.0), math.radians(5.0))
    assert tangent is not None
    assert math.degrees(tangent) == pytest.approx(30.0, abs=1e-9)

    # Zero disc radius degenerates to the centre point.
    point = disc_entry_angle(math.radians(30.0), math.radians(2.0), 0.0)
    assert point is not None
    assert math.degrees(point) == pytest.approx(30.0, abs=1e-9)


# --- ray origin: the port, not the saddle centre ------------------------------


def test_port_tangent_is_the_default_and_starts_at_the_port() -> None:
    """The default mode launches from the port on the excision boundary."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    import inspect

    sig = inspect.signature(connect_seed_to_maxima)
    assert sig.parameters["direction_mode"].default == "port_tangent"

    ray = connection_ray(seed, seed.ascent_branches[0], "port_tangent", R)
    assert ray is not None
    # Origin is the port at colatitude RING_DEG, not the pole.
    assert math.degrees(
        target_frame_angles(np.array(_pt(0.0, 0.0)), ray)[0]
    ) == pytest.approx(-RING_DEG, abs=1e-9)


def test_target_inside_the_ring_is_behind_the_port_origin() -> None:
    """A CP inside the excision boundary has along < 0 from the port.

    This is the mechanism that makes the degenerate core non-occluding: the ray
    starts at the boundary, so anything the ring swallowed is behind it.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    ray = connection_ray(seed, seed.ascent_branches[0], "port_tangent", R)
    assert ray is not None
    inside = np.array(_pt(RING_DEG * 0.5, 0.0))  # colatitude 1.5 < ring 3
    along, _ = target_frame_angles(inside, ray)
    assert along < 0.0


# --- first-hit on the disc ----------------------------------------------------


def test_first_hit_beats_farther_on_axis() -> None:
    """A nearer maximum on the ray wins over a far one with cross-track ~0.

    The MC109/MC8 case: a ray through a near maximum toward a far one has
    cross ~0 for both, so cross-track cannot separate them; along-track must.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_NEAR", 15.0, 0.001),
        _max("MAX_FAR", 30.0, 0.0),
        disc_deg=2.0,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].max_id == "MAX_NEAR"
    assert conns[0].status == "connected"
    # Entry is (colatitude - ring - disc) = 15 - 3 - 2 = 10.
    assert conns[0].along_track_deg == pytest.approx(10.0, abs=0.1)


def test_minimum_disc_on_the_ray_occludes_the_maximum_behind_it() -> None:
    """A minimum whose disc the ray enters blocks the maximum behind it."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max(
            "MAX_BEYOND",
            30.0,
            0.0,
        ),
        extra=[_target("MIN_BLOCK", "minimum", 15.0, 0.0, disc_deg=2.0)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "occluded"
    assert conns[0].max_id is None
    assert conns[0].blocker_id == "MIN_BLOCK"
    assert conns[0].blocker_polarity == "minimum"


def test_saddle_disc_on_the_ray_occludes() -> None:
    """A saddle whose disc the ray enters is a separatrix crossing: occluded."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("SAD_BLOCK", "saddle", 12.0, 0.0, disc_deg=1.5)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "occluded"
    assert conns[0].blocker_id == "SAD_BLOCK"
    assert conns[0].blocker_polarity == "saddle"


def test_off_ray_minimum_does_not_occlude_the_max_on_it() -> None:
    """The MC18 case: a minimum off the ray is not an obstacle.

    A minimum at cross-track 7.4 deg with a 1.6 deg disc: the great circle
    passes outside the disc, so it is not a candidate. The maximum at cross 0.4
    with a 2.9 deg disc is the true first hit, even though the minimum is nearer
    to the saddle.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_ON_AXIS", 45.0, 0.0),
        extra=[_target("MIN_OFF", "minimum", 15.0, 30.0, disc_deg=1.61)],
        disc_deg=2.91,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_ON_AXIS"
    assert conns[0].blocker_id is None


def test_first_hit_is_a_maximum_so_it_connects() -> None:
    """When the nearest disc the ray enters is a maximum, the branch connects."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_NEAR", 15.0, 0.0),
        extra=[_target("SAD_FAR", "saddle", 25.0, 0.0, disc_deg=2.0)],
        disc_deg=2.0,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_NEAR"


def test_source_saddle_is_not_its_own_target() -> None:
    """A branch never claims the saddle it launched from (skipped by id)."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_20", 20.0, 0.0),
        extra=[_target("SAD_1", "saddle", 10.0, 0.0, disc_deg=3.0)],
        disc_deg=2.0,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_20"


# --- the forward sector (cone) gate -------------------------------------------


def test_cone_gate_excludes_an_off_sector_near_target() -> None:
    """A nearer target outside the sector loses to a farther one inside it."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("OFF_SECTOR", 25.0, 60.0),
        _max("ON_AXIS", 30.0, 0.0),
        disc_deg=2.0,
    )
    assert CONE_DEG < 21.5
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].max_id == "ON_AXIS"


def test_cone_gate_with_nothing_inside_is_no_target() -> None:
    """A branch whose only forward disc is off-sector finds nothing."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("OFF_SECTOR", 25.0, 60.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "no_target"
    assert conns[0].blocker_id is None


def test_arc_cap_produces_no_target() -> None:
    """A maximum whose disc entry is past the cap is not claimed."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("FAR", 20.0, 0.0), disc_deg=0.0)
    # entry = 20 - 3 = 17 deg; cap of 5 pitches = 10 deg < 17.
    conns = connect_seed_to_maxima([seed], pool, R, PITCH, max_arc_pitch=5.0)
    assert conns[0].status == "no_target"
    assert conns[0].arc_points == ()
    assert connect_seed_to_maxima([seed], pool, R, PITCH)[0].max_id == "FAR"


def test_rear_hemisphere_is_never_claimed() -> None:
    """A maximum behind the port is not a forward hit."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("BEHIND", 160.0, 180.0), disc_deg=2.0)
    assert connect_seed_to_maxima([seed], pool, R, PITCH)[0].status == "no_target"


def test_opposite_branches_claim_opposite_maxima() -> None:
    """Two branches 180 deg apart must not collapse onto the same extremum."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(1, 180.0)])
    pool = _pool(_max("EAST", 25.0, 3.0), _max("WEST", 25.0, 177.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert [c.max_id for c in conns] == ["EAST", "WEST"]


def test_only_ascent_branches_are_connected() -> None:
    """Descent branches head for minima and are excluded from the table."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(0, 90.0, kind="descent")])
    pool = _pool(_max("A", 20.0, 0.0), _max("B", 20.0, 90.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert len(conns) == 1
    assert conns[0].max_id == "A"


# --- the two-part arc ---------------------------------------------------------


def test_arc_starts_at_the_port_and_ends_at_the_centre() -> None:
    """Leg 1 runs along the ray to the boundary; leg 2 to the centre."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("T", 30.0, 0.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    arc = np.array(conns[0].arc_points)
    # First point is the port (colatitude RING_DEG), not the pole.
    port = np.array(_pt(RING_DEG, 0.0))
    assert np.linalg.norm(arc[0] - port) < 1e-9
    # Last point is the maximum's centre.
    centre = np.array(_pt(30.0, 0.0))
    assert np.linalg.norm(arc[-1] - centre) < 1e-9
    # Every sample sits on the unit sphere.
    assert np.allclose(np.linalg.norm(arc, axis=1), 1.0, atol=1e-9)


def test_arc_points_lie_on_the_sphere_world_frame() -> None:
    """Every rendered sample sits at radius R from the nucleus, in world frame."""
    seed = _seed("SAD_1", [_branch(0, 35.0)])
    pool = _pool(_max("T", 30.0, 35.0, center=CENTER), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH, center=CENTER)
    for p in conns[0].arc_points:
        assert float(np.linalg.norm(np.array(p) - np.array(CENTER))) == pytest.approx(
            R, rel=1e-9
        )


# --- pool construction (the frame trap) ---------------------------------------


def test_build_connection_pool_merges_extrema_and_saddles() -> None:
    """The pool carries maxima, minima AND saddles with their disc radii."""
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
    pool = build_connection_pool(
        [*maxima, *minima],
        raw_sad,
        [],
        set(),
        ring_radius_by_vertex={7: math.radians(4.0)},
    )
    by_id = {t.cp_id: (t.polarity, t.angular_radius_rad) for t in pool}
    assert by_id["SAD_7"][0] == "saddle"
    assert math.degrees(by_id["SAD_7"][1]) == pytest.approx(4.0)


def test_build_connection_pool_excludes_hidden_raw_saddles() -> None:
    """A raw saddle collapsed into an effective entity is not visible."""
    raw_sad = [
        CriticalPoint(
            vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0)
        ),
        CriticalPoint(
            vertex_id=8, cp_type="saddle", value=0.3, position=_pt(35.0, 0.0)
        ),
    ]
    pool = build_connection_pool([], raw_sad, [], hidden_vertex_ids={8})
    assert {t.cp_id for t in pool} == {"SAD_7"}


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
    pool = build_connection_pool(
        [], [], eff, set(), ring_radius_by_id={"E-SAD_MC7": math.radians(8.0)}
    )
    by_id = {t.cp_id: t.polarity for t in pool}
    assert by_id == {"E-SAD_MC7": "saddle"}


def test_build_connection_pool_maps_raw_saddles_to_world_frame() -> None:
    """Raw saddles are centred; the pool is world frame. Checked with an offset."""
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


# --- mutual, purity, determinism ----------------------------------------------


def test_mutual_flags_exactly_one_claim_per_maximum() -> None:
    """The tightest claim on a maximum is the mutual one, and only one is."""
    seeds = [
        _seed("SAD_A", [_branch(0, 0.0)], position=_pt(20.0, 0.0)),
        _seed("SAD_B", [_branch(0, 0.0)], position=_pt(60.0, 0.0)),
    ]
    pool = _pool(_max("TARGET", 20.0, 0.0), disc_deg=2.0)
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    connected = [c for c in conns if c.status == "connected"]
    assert len(connected) >= 1
    mutual = [c for c in conns if c.mutual]
    assert len(mutual) == len({c.max_id for c in connected})


def test_mark_mutual_preserves_blocker_fields() -> None:
    """mark_mutual rebuilds records; it must not drop the occlusion fields."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("MIN_BLOCK", "minimum", 15.0, 0.0, disc_deg=2.0)],
    )
    conns = mark_mutual(connect_seed_to_maxima([seed], pool, R, PITCH))
    assert conns[0].blocker_id == "MIN_BLOCK"
    assert conns[0].blocker_polarity == "minimum"


def test_world_frame_and_centred_runs_agree() -> None:
    """Offsetting the atom changes no metric -- the frame contract, pinned."""
    seed = _seed("SAD_1", [_branch(0, 35.0)])
    pool_local = _pool(_max("T", 30.0, 35.0), _max("OFF", 12.0, 120.0), disc_deg=2.0)
    pool_world = _pool(
        _max("T", 30.0, 35.0, center=CENTER),
        _max("OFF", 12.0, 120.0, center=CENTER),
        disc_deg=2.0,
    )
    a = connect_seed_to_maxima([seed], pool_local, R, PITCH)
    b = connect_seed_to_maxima([seed], pool_world, R, PITCH, center=CENTER)
    assert a[0].max_id == b[0].max_id
    assert a[0].cross_track_deg == pytest.approx(b[0].cross_track_deg, abs=1e-9)
    assert a[0].along_track_deg == pytest.approx(b[0].along_track_deg, abs=1e-9)


def test_result_is_independent_of_input_order() -> None:
    """Shuffling the pool and the seeds cannot change any claim."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(1, 120.0)])
    seed2 = _seed("SAD_2", [_branch(0, 240.0)], position=_pt(5.0, 240.0))
    pool = _pool(
        _max("A", 20.0, 2.0),
        _max("B", 25.0, 121.0),
        _max("C", 30.0, 241.0),
        _max("D", 12.0, 60.0),
        disc_deg=2.0,
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
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    pool = _pool(_max("T", 30.0, 20.0), disc_deg=2.0)
    modes: tuple[DirectionMode, ...] = ("chord", "port_tangent")
    assert set(modes) == set(DIRECTION_MODES)
    for mode in modes:
        conns = connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode=mode)
        assert conns[0].direction_mode == mode
        assert conns[0].max_id == "T"

    with pytest.raises(ValueError):
        connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode="bogus")  # type: ignore[arg-type]


def test_degenerate_seed_position_yields_no_target() -> None:
    """A saddle at the nucleus has no radial, hence no ray."""
    seed = _seed("SAD_1", [_branch(0, 0.0)], position=(0.0, 0.0, 0.0))
    assert (
        connect_seed_to_maxima([seed], _pool(_max("T", 20.0, 0.0)), R, PITCH)[0].status
        == "no_target"
    )


def test_empty_inputs() -> None:
    """No seeds or no targets produce an empty table, not an exception."""
    assert connect_seed_to_maxima([], [], R, PITCH) == []
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    assert connect_seed_to_maxima([seed], [], R, PITCH)[0].status == "no_target"


def test_summarize_connections_counts_all_three_statuses() -> None:
    """The summary separates connected / occluded / no_target, all finite."""
    seed = _seed(
        "SAD_1",
        [_branch(0, 0.0), _branch(1, 90.0), _branch(2, 200.0)],
    )
    pool = _pool(
        _max("M0", 25.0, 1.0),
        extra=[_target("MIN90", "minimum", 25.0, 90.0, disc_deg=2.0)],
        disc_deg=2.0,
    )
    stats = summarize_connections(connect_seed_to_maxima([seed], pool, R, PITCH))
    assert stats["n_branches"] == 3
    assert stats["n_connected"] == 1
    assert stats["n_occluded"] == 1
    assert stats["n_no_target"] == 1
    assert stats["n_blocked_by_minimum"] == 1
    assert stats["n_blocked_by_saddle"] == 0
    for key, value in stats.items():
        if isinstance(value, float):
            assert math.isfinite(value), key

    empty = summarize_connections([])
    assert empty["n_branches"] == 0
    assert empty["n_occluded"] == 0


def test_verified_flag_tracks_index_agrees() -> None:
    """Provisional (k_mismatch) seeds stay distinguishable in the table."""
    good = _seed("SAD_1", [_branch(0, 0.0)], index_agrees=True)
    bad = _seed("SAD_2", [_branch(0, 0.0)], index_agrees=False)
    conns = connect_seed_to_maxima(
        [good, bad], _pool(_max("T", 20.0, 0.0), disc_deg=2.0), R, PITCH
    )
    assert [c.verified for c in conns] == [True, False]
    assert summarize_connections(conns)["n_unverified"] == 1


def test_connection_statuses_are_the_three_verdicts() -> None:
    assert set(CONNECTION_STATUSES) == {"connected", "occluded", "no_target"}


def test_lifted_arc_endpoints_stay_radial() -> None:
    """radial_offset on the emitted arc is a true lift in the centred frame."""
    seed = _seed("SAD_1", [_branch(0, 35.0)])
    conns = connect_seed_to_maxima(
        [seed], _pool(_max("T", 30.0, 35.0), disc_deg=2.0), R, PITCH
    )
    lifted = [radial_offset(p, 1.02) for p in conns[0].arc_points]
    assert all(
        float(np.linalg.norm(np.array(p))) == pytest.approx(R * 1.02, rel=1e-12)
        for p in lifted
    )
