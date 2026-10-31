"""Unit tests for catastrophe_classifier module."""

import math

import numpy as np
import pytest

from gba_topology2.src.catastrophe_classifier import (
    classify_catastrophes,
    is_bimodal_quad,
    split_quad_into_pairs,
)
from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint


def _make_cluster(
    cluster_id: str,
    cp_type: str,
    pos: tuple[float, float, float],
    value: float,
    multiplicity: int = 1,
) -> MicroCluster:
    """Helper to create a single-CP MicroCluster."""
    p_arr = np.array(pos, dtype=np.float64)
    norm = float(np.linalg.norm(p_arr))
    unit = p_arr / norm if norm > 1e-12 else p_arr

    cp = CriticalPoint(
        vertex_id=int(abs(hash(cluster_id)) % 10000),
        cp_type=cp_type,
        value=value,
        position=pos,
        multiplicity=multiplicity,
    )

    n_max = 1 if cp_type == "maximum" else 0
    n_min = 1 if cp_type == "minimum" else 0
    n_sad = 1 if cp_type == "saddle" else 0
    local_chi = n_max + n_min - (multiplicity if cp_type == "saddle" else 0)

    return MicroCluster(
        cluster_id=cluster_id,
        members=[cp],
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
        total_saddle_index=multiplicity if cp_type == "saddle" else 0,
        local_euler_index=local_chi,
        is_multi_cp=False,
    )


def test_3fold_monkey_saddle_spurious_unfolding() -> None:
    """Test detection of an unfolded monkey saddle (1 core max + 3 flanking saddles).

    Simulates numerical noise with negligible barrier depth (< barrier_tol) -> classified as fused E-MSAD.
    """
    # Core at North Pole (0, 0, 1) with value 1.0001
    core = _make_cluster("CORE_MAX", "maximum", (0.0, 0.0, 1.0), value=1.0001)

    # 3 flanking saddles separated by 6 degrees (0.1 rad) at 120 deg azimuths, values 1.0000
    r_ang = 0.1  # rad (~5.7 deg)
    saddles: list[MicroCluster] = []
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        pos = (
            math.sin(r_ang) * math.cos(phi),
            math.sin(r_ang) * math.sin(phi),
            math.cos(r_ang),
        )
        saddles.append(_make_cluster(f"SAD_{k}", "saddle", pos, value=1.0000))

    clusters = [core] + saddles
    field_vals = np.array(
        [0.0, 10.0]
    )  # field span = 10.0 => rel barrier ~ 1e-5 << 0.005

    delta_theta = 0.025  # ~1.43 deg, search window covers 0.1 rad easily

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=delta_theta,
        sphere_radius=1.0,
        barrier_tol=0.005,
        angular_tol_mult=8.0,  # 8 * 0.025 = 0.2 rad > span (0.17 rad) => spurious
    )

    assert len(cats) == 1
    assert len(unassoc) == 0

    cat = cats[0]
    assert cat.catastrophe_type == "E-MSAD"
    assert cat.fold_order == 3
    assert cat.net_euler_index == -2
    assert cat.is_fused
    assert cat.classification == "spurious_unfolding"
    assert len(cat.constituent_clusters) == 4
    assert len(cat.all_members) == 4


def test_3fold_monkey_saddle_physical_symmetry_breaking() -> None:
    """Test 3-fold constellation with real, large barrier depth -> classified as physical symmetry breaking."""
    core = _make_cluster("CORE_MAX", "maximum", (0.0, 0.0, 1.0), value=5.0)

    # 3 flanking saddles at 0.15 rad with value 1.0 (large barrier = 4.0)
    r_ang = 0.15
    saddles: list[MicroCluster] = []
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        pos = (
            math.sin(r_ang) * math.cos(phi),
            math.sin(r_ang) * math.sin(phi),
            math.cos(r_ang),
        )
        saddles.append(_make_cluster(f"SAD_{k}", "saddle", pos, value=1.0))

    clusters = [core] + saddles
    field_vals = np.array(
        [0.0, 10.0]
    )  # field span = 10.0 => rel barrier = 4.0 / 10.0 = 0.40 >> 0.005

    cats, _unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=0.02,
        sphere_radius=1.0,
        barrier_tol=0.005,
        angular_tol_mult=4.0,
    )

    assert len(cats) == 1
    cat = cats[0]
    assert cat.catastrophe_type == "E-MSAD"
    assert not cat.is_fused
    assert cat.classification == "physical_symmetry_breaking"
    assert cat.bifurcation_score > 0.8


def test_4fold_octupolar_saddle_detection() -> None:
    """Test 4-fold constellation (1 core max + 4 flanking saddles, net chi = -3)."""
    core = _make_cluster("CORE_MAX", "maximum", (0.0, 0.0, 1.0), value=1.0001)

    r_ang = 0.08
    saddles: list[MicroCluster] = []
    for k in range(4):
        phi = 2.0 * math.pi * k / 4.0
        pos = (
            math.sin(r_ang) * math.cos(phi),
            math.sin(r_ang) * math.sin(phi),
            math.cos(r_ang),
        )
        saddles.append(_make_cluster(f"SAD_{k}", "saddle", pos, value=1.0000))

    clusters = [core] + saddles
    field_vals = np.array([0.0, 1.0])

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=0.02,
        sphere_radius=1.0,
        barrier_tol=0.005,
    )

    assert len(cats) == 1
    assert len(unassoc) == 0

    cat = cats[0]
    assert cat.catastrophe_type == "E-4SAD"
    assert cat.fold_order == 4
    assert cat.net_euler_index == -3
    assert len(cat.constituent_clusters) == 5


def test_input_validation_catastrophe() -> None:
    """Test parameter validation in classify_catastrophes."""
    with pytest.raises(ValueError):
        classify_catastrophes([], np.zeros(2), delta_theta_mesh_rad=-0.1)

    with pytest.raises(ValueError):
        classify_catastrophes(
            [],
            np.zeros(2),
            delta_theta_mesh_rad=0.1,
            min_search_pitch=10.0,
            max_search_pitch=5.0,
        )

    res, unassoc = classify_catastrophes([], np.zeros(2), delta_theta_mesh_rad=0.1)
    assert res == []
    assert unassoc == []


def _sphere(polar_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """Unit-sphere point from polar angle (from +Z) and azimuth (from +X)."""
    t = math.radians(polar_deg)
    p = math.radians(azimuth_deg)
    return (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))


def test_is_bimodal_quad_flags_two_tight_pairs() -> None:
    """Four saddles forming two tight pairs are NOT one octupolar unfolding."""
    # The Pd.plt 'V (condensed)' geometry: pairs ~12 deg apart, ~63 deg between.
    saddles = [
        _make_cluster("S1", "saddle", (-0.1066, -0.8457, -0.5229), 1.0),
        _make_cluster("S2", "saddle", (-0.1066, -0.8457, 0.5229), 1.0),
        _make_cluster("S3", "saddle", (0.1066, -0.8457, -0.5229), 1.0),
        _make_cluster("S4", "saddle", (0.1066, -0.8457, 0.5229), 1.0),
    ]
    split = split_quad_into_pairs(saddles)
    assert split is not None
    _pairs, max_intra, min_inter = split
    assert max_intra == pytest.approx(12.24, abs=0.1)
    assert min_inter == pytest.approx(63.06, abs=0.5)
    assert is_bimodal_quad(saddles, pair_max_deg=16.64, bimodal_ratio=2.5)


def test_is_bimodal_quad_accepts_true_octupole() -> None:
    """Four saddles at ~90 deg cyclic spacing stay a valid k=4 constellation."""
    saddles = [
        _make_cluster("O1", "saddle", _sphere(45.0, 0.0), 1.0),
        _make_cluster("O2", "saddle", _sphere(45.0, 90.0), 1.0),
        _make_cluster("O3", "saddle", _sphere(45.0, 180.0), 1.0),
        _make_cluster("O4", "saddle", _sphere(45.0, 270.0), 1.0),
    ]
    split = split_quad_into_pairs(saddles)
    assert split is not None
    _pairs, max_intra, min_inter = split
    # A regular 4-fold set has comparable intra/inter gaps -> no bimodality.
    assert max_intra == pytest.approx(60.0, abs=0.5)
    assert min_inter == pytest.approx(60.0, abs=0.5)
    assert not is_bimodal_quad(saddles, pair_max_deg=16.64, bimodal_ratio=2.5)


def test_split_quad_into_pairs_requires_four() -> None:
    """The pairing helper is defined only on exactly four clusters."""
    assert split_quad_into_pairs([]) is None
    trio = [
        _make_cluster("T1", "saddle", _sphere(20.0, 0.0), 1.0),
        _make_cluster("T2", "saddle", _sphere(20.0, 120.0), 1.0),
        _make_cluster("T3", "saddle", _sphere(20.0, 240.0), 1.0),
    ]
    assert split_quad_into_pairs(trio) is None


def test_two_tight_pairs_become_two_monkey_saddles() -> None:
    """Two bound saddle pairs collapse into two E-MSADs, not one huge octupole.

    Reproduces Pd.plt 'V (condensed)' MC41/MC43 and MC42/MC44: each pair is a
    monkey saddle (chi = -1 + -1 = -2) with no extremum core, so the pair must
    be recognised directly and each must get its own tight boundary ring.
    """
    delta_theta = 0.0581  # ~3.33 deg, the Pd.plt mesh pitch
    saddles = [
        _make_cluster("MC41", "saddle", (-0.1066, -0.8457, -0.5229), 0.02029536),
        _make_cluster("MC42", "saddle", (-0.1066, -0.8457, 0.5229), 0.02029536),
        _make_cluster("MC43", "saddle", (0.1066, -0.8457, -0.5229), 0.02029536),
        _make_cluster("MC44", "saddle", (0.1066, -0.8457, 0.5229), 0.02029535),
    ]
    # A maximum far from the saddles; it must not absorb them into a quad.
    core = _make_cluster("CORE", "maximum", (0.0, 0.0, 1.0), 0.03)
    clusters = [core, *saddles]
    field_vals = np.array([0.0, 0.03])

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=delta_theta,
        sphere_radius=1.0,
    )

    assert len(cats) == 2
    assert all(c.catastrophe_type == "E-MSAD" for c in cats)
    assert all(c.fold_order == 3 for c in cats)
    assert all(c.net_euler_index == -2 for c in cats)
    assert all(c.is_fused for c in cats)

    member_ids = {
        frozenset(cl.cluster_id for cl in c.constituent_clusters) for c in cats
    }
    assert member_ids == {
        frozenset({"MC41", "MC43"}),
        frozenset({"MC42", "MC44"}),
    }

    # Each ring encloses only its own pair: radius ~ span/2 + pitch, not the
    # ~36 deg ring a 4-saddle octupole spanning both pairs would produce.
    for c in cats:
        assert c.angular_span_deg == pytest.approx(12.24, abs=0.2)
        assert c.angular_radius_deg == pytest.approx(9.45, abs=0.3)

    # The two rings are distinct and centred on their own pair.
    c0, c1 = (np.array(cats[0].centroid_unit), np.array(cats[1].centroid_unit))
    d = math.degrees(math.acos(float(np.clip(float(np.dot(c0, c1)), -1.0, 1.0))))
    assert d > 45.0

    # The far core is released, not consumed by a fabricated octupole.
    assert {cl.cluster_id for cl in unassoc} == {"CORE"}


def test_pairwise_channel_does_not_steal_core_flanking_saddles() -> None:
    """A tight saddle pair straddling a core belongs to that core.

    Two saddles on opposite sides of a core are close to each other and share a
    value, so separation and barrier alone would fuse them and strand the core.
    The coreless guard (nearest core distance vs pair separation) must reject
    the pair and leave the core-centred k=3 constellation intact.
    """
    delta_theta = 0.02
    core = _make_cluster("CORE", "maximum", _sphere(20.0, 90.0), 1.0)
    saddles = [
        _make_cluster("S0", "saddle", _sphere(15.0, 0.0), 0.9),
        _make_cluster("S1", "saddle", _sphere(15.0, 180.0), 0.9),
        _make_cluster("S2", "saddle", _sphere(15.0, 90.0), 0.9),
    ]
    clusters = [core, *saddles]
    field_vals = np.array([0.0, 1.0])

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=delta_theta,
        sphere_radius=1.0,
        max_search_pitch=25.0,
        pair_max_pitch=30.0,
    )
    assert len(cats) == 1
    assert cats[0].catastrophe_type == "E-MSAD"
    assert {cl.cluster_id for cl in cats[0].constituent_clusters} == {
        "CORE",
        "S0",
        "S1",
        "S2",
    }
    assert unassoc == []

    # A/B: disabling the coreless guard lets a saddle pair be stolen by the
    # coreless channel, so the core-centred constellation no longer forms and
    # the core is stranded. (Which pair wins is the greedy tightest-first
    # choice; the invariant under test is that the core loses its ring.)
    cats_off, unassoc_off = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=delta_theta,
        sphere_radius=1.0,
        max_search_pitch=25.0,
        pair_max_pitch=30.0,
        coreless_min_ratio=0.0,
    )
    assert not any(
        {cl.cluster_id for cl in c.constituent_clusters} == {"CORE", "S0", "S1", "S2"}
        for c in cats_off
    )
    assert "CORE" in {cl.cluster_id for cl in unassoc_off}


def test_core_sorting_by_saddle_proximity() -> None:
    """Test that cores are prioritized by proximity to their closest saddles.

    Prevents a distant core that appears earlier in the cluster list from
    reaching across the sphere and greedily stealing saddles from a close core.
    """
    # Core 1: distant from saddles (at (0.7, 0, 0.7) ~ 45 deg)
    p_dist = (math.sin(0.4), 0.0, math.cos(0.4))
    core_dist = _make_cluster("CORE_DISTANT", "maximum", p_dist, value=1.0)

    # Core 2: right at the center of the saddles (at North Pole)
    core_close = _make_cluster("CORE_CLOSE", "maximum", (0.0, 0.0, 1.0), value=1.0)

    # 3 saddles clustered tightly around North Pole (0.08 rad ~ 4.5 deg)
    r_ang = 0.08
    saddles: list[MicroCluster] = []
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        pos = (
            math.sin(r_ang) * math.cos(phi),
            math.sin(r_ang) * math.sin(phi),
            math.cos(r_ang),
        )
        saddles.append(_make_cluster(f"SAD_{k}", "saddle", pos, value=1.0))

    # Core_dist is placed first in the list, but core_close is much closer to the saddles
    clusters = [core_dist, core_close, *saddles]
    field_vals = np.array([0.0, 1.0])

    cats, unassoc = classify_catastrophes(
        clusters=clusters,
        field_vals=field_vals,
        delta_theta_mesh_rad=0.02,
        sphere_radius=1.0,
        max_search_pitch=25.0,  # wide enough that core_dist could reach saddles if evaluated first
    )

    assert len(cats) == 1
    # The close core must be chosen, not the distant one
    assert cats[0].constituent_clusters[0].cluster_id == "CORE_CLOSE"
    assert {c.cluster_id for c in cats[0].constituent_clusters} == {
        "CORE_CLOSE",
        "SAD_0",
        "SAD_1",
        "SAD_2",
    }
    assert {c.cluster_id for c in unassoc} == {"CORE_DISTANT"}
