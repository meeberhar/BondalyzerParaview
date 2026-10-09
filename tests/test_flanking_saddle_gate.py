"""Unit tests for the flanking-saddle eligibility gate in catastrophe_classifier.

The gate (``is_flanking_saddle_candidate``) restricts the catastrophe flanking
pool to pure simple saddles (net index -1, no extrema). Its purpose is to stop
long composite saddle strings -- which also carry net index -1 -- from being
swallowed as flanking members, which strands them from their own effective
saddle reduction and corrupts the reported fold order.
"""

from __future__ import annotations

import math

import numpy as np

from gba_topology2.src.catastrophe_classifier import (
    classify_catastrophes,
    is_flanking_saddle_candidate,
)
from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint


def _make_composite_cluster(
    cluster_id: str,
    pos: tuple[float, float, float],
    n_max: int,
    n_min: int,
    n_sad: int,
    mult_last_saddle: int = 1,
) -> MicroCluster:
    """Build a MicroCluster with explicit extremum/saddle counts.

    Args:
        cluster_id: Identifier to assign.
        pos: Centroid position (also used for every member).
        n_max: Number of maximum members.
        n_min: Number of minimum members.
        n_sad: Number of saddle members.
        mult_last_saddle: Multiplicity of the final saddle. The real
            modified-Willmore strings reach chi = -1 this way: 12 Max + 2 Min
            against 14 saddles whose multiplicities sum to 15.

    Returns:
        A MicroCluster whose counts and net index match the requested
        composition.
    """
    members: list[CriticalPoint] = []
    base = abs(hash(cluster_id)) % 100000
    vid = base
    for cp_type, count in (("maximum", n_max), ("minimum", n_min)):
        for _ in range(count):
            members.append(
                CriticalPoint(
                    vertex_id=vid,
                    cp_type=cp_type,
                    value=1.0,
                    position=pos,
                    multiplicity=1,
                )
            )
            vid += 1
    for i in range(n_sad):
        mult = mult_last_saddle if i == n_sad - 1 else 1
        members.append(
            CriticalPoint(
                vertex_id=vid,
                cp_type="saddle",
                value=1.0,
                position=pos,
                multiplicity=mult,
            )
        )
        vid += 1

    total_saddle_index = (n_sad - 1) + mult_last_saddle if n_sad else 0

    p = np.array(pos, dtype=np.float64)
    norm = float(np.linalg.norm(p))
    unit = p / norm if norm > 1e-12 else p

    return MicroCluster(
        cluster_id=cluster_id,
        members=members,
        centroid=(float(pos[0]), float(pos[1]), float(pos[2])),
        centroid_unit=(float(unit[0]), float(unit[1]), float(unit[2])),
        angular_radius_rad=0.02,
        angular_radius_deg=math.degrees(0.02),
        angular_diameter_rad=0.0,
        angular_diameter_deg=0.0,
        spatial_diameter=0.0,
        n_maxima=n_max,
        n_minima=n_min,
        n_saddles=n_sad,
        total_saddle_index=total_saddle_index,
        local_euler_index=n_max + n_min - total_saddle_index,
        is_multi_cp=len(members) > 1,
    )


def _make_single_cluster(
    cluster_id: str,
    cp_type: str,
    pos: tuple[float, float, float],
) -> MicroCluster:
    """Build a single-CP MicroCluster of the requested type.

    Args:
        cluster_id: Identifier to assign.
        cp_type: One of ``"maximum"``, ``"minimum"``, ``"saddle"``.
        pos: CP position.

    Returns:
        A single-member MicroCluster.
    """
    if cp_type == "maximum":
        counts = (1, 0, 0)
    elif cp_type == "minimum":
        counts = (0, 1, 0)
    else:
        counts = (0, 0, 1)
    return _make_composite_cluster(cluster_id, pos, *counts)


# ---------------------------------------------------------------------------
# Predicate: admitted cases
# ---------------------------------------------------------------------------


def test_admits_pure_single_saddle() -> None:
    """A single simple saddle (chi = -1, no extrema) is a valid flanking member."""
    sad = _make_single_cluster("S1", "saddle", (0.0, 0.0, 1.0))

    assert sad.local_euler_index == -1
    assert is_flanking_saddle_candidate(sad) is True


def test_rejects_degenerate_double_saddle() -> None:
    """Two saddles with no extrema give chi = -2 and must be rejected.

    A flanking member must contribute exactly -1 for the constellation index to
    balance as 1 - k. A chi = -2 set (e.g. a multiplicity-2 saddle) would
    corrupt the reported fold order, so the gate guards the boundary from below.
    """
    pair = _make_composite_cluster("S2", (0.0, 0.0, 1.0), n_max=0, n_min=0, n_sad=2)

    assert pair.local_euler_index == -2
    assert is_flanking_saddle_candidate(pair) is False


# ---------------------------------------------------------------------------
# Predicate: rejected cases
# ---------------------------------------------------------------------------


def test_rejects_composite_saddle_string() -> None:
    """A saddle string with extrema and net chi = -1 is rejected.

    This is the modified-Willmore MC1 case: 12 Max + 2 Min against 14 saddles
    of summed multiplicity 15, giving chi = -1. Index arithmetic alone would
    admit it, so the extrema test is what excludes it.
    """
    string = _make_composite_cluster(
        "MC1", (0.0, 0.0, 1.0), n_max=12, n_min=2, n_sad=14, mult_last_saddle=2
    )

    assert string.local_euler_index == -1
    assert is_flanking_saddle_candidate(string) is False


def test_rejects_extrema_saddle_dipole() -> None:
    """A chi = 0 extrema/saddle dipole is not a saddle candidate."""
    dipole = _make_composite_cluster("D", (0.0, 0.0, 1.0), n_max=1, n_min=0, n_sad=1)

    assert dipole.local_euler_index == 0
    assert is_flanking_saddle_candidate(dipole) is False


def test_rejects_clusters_without_saddles() -> None:
    """Pure extrema carry no saddle and are never flanking candidates."""
    mx = _make_single_cluster("M", "maximum", (0.0, 0.0, 1.0))
    mn = _make_single_cluster("N", "minimum", (0.0, 0.0, 1.0))

    assert is_flanking_saddle_candidate(mx) is False
    assert is_flanking_saddle_candidate(mn) is False


def test_rejects_saddle_plus_minima_string() -> None:
    """A string of minima flanked by saddles (chi = -1) is rejected."""
    string = _make_composite_cluster(
        "MC9", (0.0, 0.0, 1.0), n_max=0, n_min=11, n_sad=12
    )

    assert string.local_euler_index == -1
    assert is_flanking_saddle_candidate(string) is False


# ---------------------------------------------------------------------------
# Integration: the gate changes what classify_catastrophes will assemble
# ---------------------------------------------------------------------------


def _constellation_ring(pos_offset: float) -> list[tuple[float, float, float]]:
    """Return three ring positions at 120 deg azimuths, offset from the pole.

    Args:
        pos_offset: Angular offset (radians) of the ring from the north pole.

    Returns:
        Three unit positions forming an equilateral ring around the pole.
    """
    ring: list[tuple[float, float, float]] = []
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        ring.append(
            (
                math.sin(pos_offset) * math.cos(phi),
                math.sin(pos_offset) * math.sin(phi),
                math.cos(pos_offset),
            )
        )
    return ring


def test_genuine_monkey_saddle_still_detected() -> None:
    """A +1 core ringed by three simple saddles is still a k=3 catastrophe."""
    core = _make_single_cluster("CORE", "maximum", (0.0, 0.0, 1.0))
    ring = _constellation_ring(0.1)
    flanking = [
        _make_single_cluster(f"SAD_{i}", "saddle", p) for i, p in enumerate(ring)
    ]
    clusters = [core, *flanking]

    cats, _ = classify_catastrophes(
        clusters=clusters,
        field_vals=np.array([0.0, 10.0]),
        delta_theta_mesh_rad=0.025,
        sphere_radius=1.0,
        barrier_tol=0.005,
        angular_tol_mult=4.0,
    )

    assert len(cats) == 1
    assert cats[0].catastrophe_type == "E-MSAD"
    assert cats[0].net_euler_index == -2
    assert len(cats[0].constituent_clusters) == 4


def test_composite_strings_are_not_swallowed() -> None:
    """Three chi = -1 composite strings must NOT form a catastrophe.

    Index arithmetic (1 + 3*(-1) = -2) matches the monkey-saddle template, so
    without the gate the strings are consumed as flanking saddles. With the gate
    they stay in the pool for standalone effective-saddle reduction.
    """
    core = _make_single_cluster("CORE", "maximum", (0.0, 0.0, 1.0))
    ring = _constellation_ring(0.1)
    flanking = [
        _make_composite_cluster(
            f"STR_{i}", p, n_max=12, n_min=2, n_sad=14, mult_last_saddle=2
        )
        for i, p in enumerate(ring)
    ]
    clusters = [core, *flanking]

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=np.array([0.0, 10.0]),
        delta_theta_mesh_rad=0.025,
        sphere_radius=1.0,
        barrier_tol=0.005,
        angular_tol_mult=4.0,
    )

    assert cats == []
    # Every string survives as an unassociated cluster, eligible for reduction.
    assert {cl.cluster_id for cl in unassoc} == {"CORE", "STR_0", "STR_1", "STR_2"}


def test_mixed_pool_uses_only_eligible_saddles() -> None:
    """With 3 eligible saddles and 3 composite strings present, the catastrophe
    forms from the eligible saddles and the strings are left alone."""
    core = _make_single_cluster("CORE", "maximum", (0.0, 0.0, 1.0))
    ring = _constellation_ring(0.1)
    saddles = [
        _make_single_cluster(f"SAD_{i}", "saddle", p) for i, p in enumerate(ring)
    ]
    strings = [
        _make_composite_cluster(
            f"STR_{i}", p, n_max=12, n_min=2, n_sad=14, mult_last_saddle=2
        )
        for i, p in enumerate(ring)
    ]
    clusters = [core, *saddles, *strings]

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=np.array([0.0, 10.0]),
        delta_theta_mesh_rad=0.025,
        sphere_radius=1.0,
        barrier_tol=0.005,
        angular_tol_mult=4.0,
    )

    assert len(cats) == 1
    member_ids = {cl.cluster_id for cl in cats[0].constituent_clusters}
    assert member_ids == {"CORE", "SAD_0", "SAD_1", "SAD_2"}
    assert not any(cid.startswith("STR_") for cid in member_ids)
    assert {cl.cluster_id for cl in unassoc} == {"STR_0", "STR_1", "STR_2"}
