"""Unit tests for corridor bridging in string_aggregator and the reduction-unit
partition in sphere_viewer.

The bridging rule under test is the one that matters on corrugated fields:
fragments of one ridge/valley corridor sit tip-to-tip with matching principal
axes, while two long strings crossing at 60 degrees overlap (negative gap).
Centroid separation cannot tell these apart -- for elongated clusters it is
dominated by the two half-lengths -- so the rule uses the uncovered gap plus
axis collinearity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    MicroCluster,
    generate_elliptical_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.sphere_viewer import build_reduction_units
from gba_topology2.src.string_aggregator import (
    aggregate_cluster_strings,
    cluster_directional_support_radius,
)

DELTA_THETA = 0.025
RADIUS = 1.0


def _ring(
    centroid: tuple[float, float, float],
    axis: tuple[float, float, float],
    semi_major: float,
    semi_minor: float,
) -> list[BoundaryRingPoint]:
    """Build an elliptical boundary ring for a synthetic string cluster."""
    return generate_elliptical_boundary_ring(
        centroid=centroid,
        sphere_radius=RADIUS,
        semi_major_rad=semi_major,
        semi_minor_rad=semi_minor,
        principal_axis_unit=axis,
        num_samples=24,
    )


def _string_cluster(
    cluster_id: str,
    centroid_unit: tuple[float, float, float],
    axis_unit: tuple[float, float, float],
    n_max: int,
    n_sad: int,
    semi_major: float,
    start_vid: int,
    semi_minor: float = 0.01,
) -> MicroCluster:
    """Build an elongated string cluster centred on a unit direction.

    Args:
        cluster_id: Identifier to assign.
        centroid_unit: Unit centroid direction.
        axis_unit: Unit principal (major) axis, tangent to the sphere.
        n_max: Number of maximum members.
        n_sad: Number of simple-saddle members.
        semi_major: Semi-major angular radius in radians.
        start_vid: First vertex id to allocate.
        semi_minor: Semi-minor angular radius in radians.

    Returns:
        A string-morphology MicroCluster.
    """
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    ax = np.array(axis_unit, dtype=np.float64)
    # Re-orthogonalise the axis against the centroid so offsets run along the
    # great circle, then renormalise. A zero axis leaves `ax` null and every
    # member collapses onto the centroid.
    ax = ax - float(np.dot(ax, c)) * c
    ax_norm = float(np.linalg.norm(ax))
    if ax_norm > 1e-12:
        ax = ax / ax_norm
    else:
        ax = np.zeros(3, dtype=np.float64)

    total = n_max + n_sad
    members: list[CriticalPoint] = []
    vid = start_vid
    for i, (cp_type, value) in enumerate(
        [("maximum", 1.0)] * n_max + [("saddle", 0.5)] * n_sad
    ):
        # Spread members across the full semi-major extent so the pooled
        # tangent-plane PCA sees a genuinely elongated set.
        frac = (i / (total - 1) - 0.5) if total > 1 else 0.0
        t = 2.0 * frac * semi_major
        p = math.cos(t) * c + math.sin(t) * ax
        p = p / np.linalg.norm(p)
        members.append(
            CriticalPoint(
                vertex_id=vid,
                cp_type=cp_type,
                value=value,
                position=(float(p[0]), float(p[1]), float(p[2])),
                multiplicity=1,
            )
        )
        vid += 1

    chi = n_max - n_sad
    centroid = (float(c[0] * RADIUS), float(c[1] * RADIUS), float(c[2] * RADIUS))
    return MicroCluster(
        cluster_id=cluster_id,
        members=members,
        centroid=centroid,
        centroid_unit=(float(c[0]), float(c[1]), float(c[2])),
        angular_radius_rad=semi_major,
        angular_radius_deg=math.degrees(semi_major),
        angular_diameter_rad=2.0 * semi_major,
        angular_diameter_deg=math.degrees(2.0 * semi_major),
        spatial_diameter=2.0 * semi_major * RADIUS,
        n_maxima=n_max,
        n_minima=0,
        n_saddles=n_sad,
        total_saddle_index=n_sad,
        local_euler_index=chi,
        is_multi_cp=len(members) > 1,
        boundary_ring=_ring(centroid, axis_unit, semi_major, semi_minor),
        morphology="string",
        aspect_ratio=8.0,
        principal_axis_unit=axis_unit,
        semi_major_rad=semi_major,
        semi_minor_rad=semi_minor,
    )


def _compact_cluster(
    cluster_id: str,
    centroid_unit: tuple[float, float, float],
    cp_type: str,
    radius: float,
    vid: int,
) -> MicroCluster:
    """Build a single-CP compact cluster at a unit direction.

    Mirrors what ``compute_micro_clusters`` produces for an isolated critical
    point: ``morphology == "compact"``, a null principal axis, and equal
    semi-major and semi-minor radii (a circular ring).

    Args:
        cluster_id: Identifier to assign.
        centroid_unit: Unit centroid direction.
        cp_type: ``"maximum"``, ``"minimum"`` or ``"saddle"``.
        radius: Circular ring angular radius in radians.
        vid: Vertex id for the single member.

    Returns:
        A compact-morphology MicroCluster with net index +1 for extrema.
    """
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    chi = 1 if cp_type in ("maximum", "minimum") else -1
    return MicroCluster(
        cluster_id=cluster_id,
        members=[
            CriticalPoint(
                vertex_id=vid,
                cp_type=cp_type,
                value=0.2 if cp_type == "minimum" else 1.0,
                position=(float(c[0]), float(c[1]), float(c[2])),
                multiplicity=1,
            )
        ],
        centroid=(float(c[0] * RADIUS), float(c[1] * RADIUS), float(c[2] * RADIUS)),
        centroid_unit=(float(c[0]), float(c[1]), float(c[2])),
        angular_radius_rad=radius,
        angular_radius_deg=math.degrees(radius),
        angular_diameter_rad=0.0,
        angular_diameter_deg=0.0,
        spatial_diameter=0.0,
        n_maxima=1 if cp_type == "maximum" else 0,
        n_minima=1 if cp_type == "minimum" else 0,
        n_saddles=1 if cp_type == "saddle" else 0,
        total_saddle_index=1 if cp_type == "saddle" else 0,
        local_euler_index=chi,
        is_multi_cp=False,
        boundary_ring=[],
        morphology="compact",
        aspect_ratio=1.0,
        semi_major_rad=radius,
        semi_minor_rad=radius,
    )


def _collinear_pair() -> list[MicroCluster]:
    """Two string fragments tip-to-tip along +x near the north pole.

    Each spans 0.05 rad semi-major; their centroids sit 0.115 rad apart, so the
    uncovered gap is 0.015 rad (0.6 delta_theta) and their axes are parallel.
    """
    a = _string_cluster(
        "MC1",
        centroid_unit=(math.sin(-0.065), 0.0, math.cos(-0.065)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.05,
        start_vid=0,
    )
    b = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.05), 0.0, math.cos(0.05)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.05,
        start_vid=100,
    )
    return [a, b]


def test_collinear_fragments_bridge() -> None:
    """Tip-to-tip fragments with parallel axes merge into one corridor."""
    clusters = _collinear_pair()

    macro, isolated = aggregate_cluster_strings(
        clusters=clusters,
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )

    assert len(macro) == 1
    assert isolated == []
    ms = macro[0]
    assert {c.cluster_id for c in ms.constituent_clusters} == {"MC1", "MC2"}
    # Index is summed, not averaged: (-1) + (-1) = -2.
    assert ms.local_euler_index == -2
    assert len(ms.all_members) == 18


def test_crossing_strings_do_not_bridge() -> None:
    """Two long strings crossing at 60 degrees must not merge.

    Their centroid separation is smaller than the collinear pair's, so a
    centroid-distance rule would merge them. The axis-collinearity test rejects
    them.
    """
    a = _string_cluster(
        "MC1",
        centroid_unit=(math.sin(0.02), 0.0, math.cos(0.02)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.30,
        start_vid=0,
    )
    ang = math.radians(60.0)
    b = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.02), 0.0, math.cos(0.02)),
        axis_unit=(math.cos(ang), math.sin(ang), 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.30,
        start_vid=100,
    )

    macro, isolated = aggregate_cluster_strings(
        clusters=[a, b],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )

    # No multi-cluster corridor is formed; both stay standalone.
    assert all(len(ms.constituent_clusters) == 1 for ms in macro)
    assert len(isolated) + sum(len(ms.constituent_clusters) for ms in macro) == 2


def test_non_collinear_fragments_rejected_when_ar_qualifies() -> None:
    """Collinearity is load-bearing: gap and pooled AR both pass, axes do not.

    A long string and a short one whose axes differ by 60 degrees are placed
    tip-to-tip. The gap test passes and the pooled aspect ratio is dominated by
    the long string, so the elongation gate passes too. Only the axis-angle test
    rejects the merge, which is what stops crossing corridors being fused.
    """
    long_a = _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=0,
    )
    ang = math.radians(60.0)
    short_b = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.35), 0.0, math.cos(0.35)),
        axis_unit=(math.cos(ang), math.sin(ang), 0.0),
        n_max=2,
        n_sad=3,
        semi_major=0.05,
        start_vid=100,
    )

    # Confirm the fixture really does satisfy the other two gates.
    gap = 0.35 - 0.30 - 0.05
    assert gap <= 3.0 * DELTA_THETA

    macro, _ = aggregate_cluster_strings(
        clusters=[long_a, short_b],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )

    assert all(len(ms.constituent_clusters) == 1 for ms in macro)

    # Same geometry with parallel axes must merge, proving the axis test is the
    # only difference.
    aligned = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.35), 0.0, math.cos(0.35)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=2,
        n_sad=3,
        semi_major=0.05,
        start_vid=100,
    )
    macro2, _ = aggregate_cluster_strings(
        clusters=[long_a, aligned],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )
    assert any(len(ms.constituent_clusters) == 2 for ms in macro2)


def test_wide_gap_does_not_bridge() -> None:
    """Collinear fragments separated by more than k_bridge stay separate."""
    near_a, _near_b = _collinear_pair()
    far = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.6), 0.0, math.cos(0.6)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.05,
        start_vid=100,
    )

    macro, isolated = aggregate_cluster_strings(
        clusters=[near_a, far],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )

    # In bridge_all mode every cluster seeds a chain, so unbridged strings come
    # back as one-constituent macro-strings rather than in `isolated`.
    assert all(len(ms.constituent_clusters) == 1 for ms in macro)
    assert isolated == []

    units, n_bridged = build_reduction_units(
        clusters=[near_a, far],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
        k_bridge=3.0,
    )
    assert n_bridged == 0
    assert len(units) == 2


def test_legacy_mode_unchanged_without_bridge_all() -> None:
    """With bridge_all off, a compact cluster never seeds or joins a chain."""
    compact = MicroCluster(
        cluster_id="MC9",
        members=[
            CriticalPoint(
                vertex_id=900,
                cp_type="maximum",
                value=1.0,
                position=(0.0, 0.0, 1.0),
                multiplicity=1,
            )
        ],
        centroid=(0.0, 0.0, 1.0),
        centroid_unit=(0.0, 0.0, 1.0),
        angular_radius_rad=0.02,
        angular_radius_deg=math.degrees(0.02),
        angular_diameter_rad=0.0,
        angular_diameter_deg=0.0,
        spatial_diameter=0.0,
        n_maxima=1,
        n_minima=0,
        n_saddles=0,
        total_saddle_index=0,
        local_euler_index=1,
        is_multi_cp=False,
        boundary_ring=[],
        morphology="compact",
        aspect_ratio=1.0,
    )

    macro, isolated = aggregate_cluster_strings(
        clusters=[compact],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        bridge_all=False,
    )

    assert macro == []
    assert [c.cluster_id for c in isolated] == ["MC9"]


def test_partition_is_disjoint_and_index_preserving() -> None:
    """Every cluster lands in exactly one unit, so summed index is preserved."""
    clusters = _collinear_pair()
    before = sum(c.local_euler_index for c in clusters)

    units, n_bridged = build_reduction_units(
        clusters=clusters,
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
        k_bridge=3.0,
    )

    origins = [oid for u in units for oid in u.origin_ids]
    assert len(origins) == len(set(origins)) == len(clusters)
    assert sum(u.local_euler_index for u in units) == before
    assert n_bridged == 2
    assert any(u.is_string for u in units)


def test_aggregation_off_is_identity_partition() -> None:
    """With aggregation off there is exactly one unit per cluster, unbridged."""
    clusters = _collinear_pair()

    units, n_bridged = build_reduction_units(
        clusters=clusters,
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=False,
    )

    assert n_bridged == 0
    assert [u.unit_id for u in units] == ["MC1", "MC2"]
    assert all(not u.is_string for u in units)


def test_lone_string_keeps_its_cluster_id() -> None:
    """A one-cluster macro-string is not a bridge and must keep its MC id."""
    solo = _string_cluster(
        "MC7",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.05,
        start_vid=0,
    )

    units, n_bridged = build_reduction_units(
        clusters=[solo],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
    )

    assert n_bridged == 0
    assert len(units) == 1
    assert units[0].unit_id == "MC7"
    assert units[0].is_string is False


def test_as_cluster_exposes_members_for_polarity_probing() -> None:
    """The MicroCluster adapter carries the merged members and index."""
    clusters = _collinear_pair()
    units, _ = build_reduction_units(
        clusters=clusters,
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
    )
    corridor = next(u for u in units if u.is_string)

    assert len(corridor.as_cluster.members) == 18
    assert corridor.as_cluster.local_euler_index == -2
    assert corridor.as_cluster.morphology == "string"
    assert corridor.as_cluster.cluster_id == corridor.unit_id


def test_composition_summary_reports_multiplicity() -> None:
    """The summary string matches the MicroCluster convention."""
    clusters = _collinear_pair()
    units, _ = build_reduction_units(
        clusters=clusters,
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
    )
    corridor = next(u for u in units if u.is_string)

    assert corridor.composition_summary == "8 Max, 10 Sad"


@pytest.mark.parametrize("aggregate", [True, False])
def test_empty_cluster_list(aggregate: bool) -> None:
    """Boundary case: no clusters yields no units and no bridging."""
    units, n_bridged = build_reduction_units(
        clusters=[],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=aggregate,
    )

    assert units == []
    assert n_bridged == 0


# ---------------------------------------------------------------------------
# Directional support radius
#
# The gap test must measure each cluster's boundary in the direction of the
# neighbour. Subtracting the scalar semi-major grants a long thin corridor a
# reach equal to its half-length in every direction, absorbing compact
# fragments that sit far off to the side.
# ---------------------------------------------------------------------------


def _long_corridor() -> MicroCluster:
    """A 17-CP chi=-1 corridor at the north pole, axis +x, a=0.30, b=0.01."""
    return _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=0,
        semi_minor=0.01,
    )


def test_support_radius_is_full_length_along_axis() -> None:
    """Toward a neighbour on the principal axis the reach is the semi-major."""
    cl = _long_corridor()
    assert cluster_directional_support_radius(
        cl, np.array([1.0, 0.0, 0.0])
    ) == pytest.approx(0.30)


def test_support_radius_is_narrow_transversely() -> None:
    """Toward a neighbour 90 degrees off-axis the reach collapses to semi-minor."""
    cl = _long_corridor()
    assert cluster_directional_support_radius(
        cl, np.array([0.0, 1.0, 0.0])
    ) == pytest.approx(0.01)


def test_support_radius_folds_axis_direction() -> None:
    """The principal axis is an undirected line, so -axis matches +axis."""
    cl = _long_corridor()
    fwd = cluster_directional_support_radius(cl, np.array([1.0, 0.0, 0.0]))
    back = cluster_directional_support_radius(cl, np.array([-1.0, 0.0, 0.0]))
    assert fwd == pytest.approx(back)


def test_support_radius_at_forty_five_degrees() -> None:
    """Midway between the axes the polar ellipse formula applies."""
    cl = _long_corridor()
    inv = 1.0 / math.sqrt(2.0)
    r = cluster_directional_support_radius(cl, np.array([inv, inv, 0.0]))
    assert r == pytest.approx(math.sqrt((0.30**2 + 0.01**2) / 2.0))


def test_support_radius_accepts_unnormalised_target() -> None:
    """Only the bearing matters, so a scaled target gives the same radius."""
    cl = _long_corridor()
    unit = cluster_directional_support_radius(cl, np.array([0.0, 1.0, 0.0]))
    scaled = cluster_directional_support_radius(cl, np.array([0.0, 7.5, 0.0]))
    assert scaled == pytest.approx(unit)


def test_support_radius_compact_cluster_is_isotropic() -> None:
    """A compact cluster has a == b, so its reach is a in every direction."""
    cl = _compact_cluster("MC9", (0.0, 0.0, 1.0), "minimum", radius=0.02, vid=900)
    for direction in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]:
        got = cluster_directional_support_radius(cl, np.array(direction))
        assert got == pytest.approx(0.02)


def test_support_radius_falls_back_on_degenerate_geometry() -> None:
    """Ambiguous bearings and null axes keep the conservative semi-major reach."""
    cl = _long_corridor()
    # Target sits at the cluster centroid: no bearing exists on the tangent plane.
    assert cluster_directional_support_radius(
        cl, np.array([0.0, 0.0, 1.0])
    ) == pytest.approx(0.30)
    # Null principal axis: no direction to measure against.
    null_axis = _string_cluster(
        "MC2",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(0.0, 0.0, 0.0),
        n_max=2,
        n_sad=3,
        semi_major=0.05,
        start_vid=500,
        semi_minor=0.01,
    )
    assert cluster_directional_support_radius(null_axis, np.array([0.0, 1.0, 0.0])) == (
        pytest.approx(0.05)
    )
    # Zero-extent cluster has no reach at all.
    flat = _string_cluster(
        "MC3",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=1,
        n_sad=2,
        semi_major=0.0,
        start_vid=700,
        semi_minor=0.0,
    )
    assert cluster_directional_support_radius(flat, np.array([1.0, 0.0, 0.0])) == 0.0


def test_perpendicular_compact_fragment_is_not_absorbed() -> None:
    """A lone minimum 90 degrees off a corridor axis must stay standalone.

    This is the Pd_20K curvedness failure at k_pitch=1.6: the 21-CP saddle
    corridor MC18 (chi = -1) swallowed a lone minimum sitting well to the side,
    cancelling the index to chi = 0 and annihilating a genuine saddle. The
    centroid separation (0.35) is smaller than the corridor half-length plus
    the bridge radius, so the isotropic semi-major subtraction absorbed it.
    """
    corridor = _long_corridor()
    off_side = _compact_cluster(
        "MC131",
        centroid_unit=(0.0, math.sin(0.35), math.cos(0.35)),
        cp_type="minimum",
        radius=0.02,
        vid=1310,
    )

    # The fixture reproduces the bug: the old isotropic gap would have linked.
    isotropic_gap = 0.35 - corridor.semi_major_rad - off_side.semi_major_rad
    assert isotropic_gap <= 3.0 * DELTA_THETA

    macro, isolated = aggregate_cluster_strings(
        clusters=[corridor, off_side],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )

    # The corridor returns as a one-constituent macro-string (it is a string,
    # so it seeds its own chain) and the rejected minimum comes back isolated.
    assert all(len(ms.constituent_clusters) == 1 for ms in macro)
    assert [c.cluster_id for c in isolated] == ["MC131"]

    units, n_bridged = build_reduction_units(
        clusters=[corridor, off_side],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
        k_bridge=3.0,
    )
    assert n_bridged == 0
    assert len(units) == 2
    # The corridor keeps its own index instead of being cancelled to chi = 0.
    corridor_unit = next(u for u in units if "MC1" in u.origin_ids)
    assert corridor_unit.local_euler_index == -1


def test_on_axis_compact_fragment_still_bridges() -> None:
    """The fix must not cost reach along the axis: a tip fragment still joins.

    Geometry is the perpendicular case rotated onto the corridor axis, so the
    only difference is the bearing. The directional radius returns the full
    semi-major there, the gap is 0.03 rad (1.2 delta_theta), and the merge
    proceeds.
    """
    corridor = _long_corridor()
    at_tip = _compact_cluster(
        "MC131",
        centroid_unit=(math.sin(0.35), 0.0, math.cos(0.35)),
        cp_type="minimum",
        radius=0.02,
        vid=1310,
    )

    macro, _ = aggregate_cluster_strings(
        clusters=[corridor, at_tip],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        k_bridge=3.0,
        aspect_ratio_threshold=2.0,
        bridge_all=True,
    )

    assert any(len(ms.constituent_clusters) == 2 for ms in macro)

    units, n_bridged = build_reduction_units(
        clusters=[corridor, at_tip],
        delta_theta_mesh_rad=DELTA_THETA,
        sphere_radius=RADIUS,
        aggregate_strings=True,
        k_bridge=3.0,
    )
    assert n_bridged == 2
    assert len(units) == 1
    assert units[0].local_euler_index == 0
