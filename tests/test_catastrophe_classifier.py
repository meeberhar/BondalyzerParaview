"""Unit tests for catastrophe_classifier module."""

import math

import numpy as np
import pytest

from gba_topology2.src.catastrophe_classifier import (
    classify_catastrophes,
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
