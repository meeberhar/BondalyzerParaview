"""Unit tests for micro_cluster module."""

import math

import numpy as np
import pytest

from gba_topology2.src.micro_cluster import (
    compute_micro_clusters,
    generate_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint


def test_isolated_critical_points() -> None:
    """Test micro-clustering when all CPs are widely separated."""
    cps = [
        CriticalPoint(
            vertex_id=0,
            cp_type="maximum",
            value=1.0,
            position=(1.0, 0.0, 0.0),
        ),
        CriticalPoint(
            vertex_id=1,
            cp_type="minimum",
            value=-1.0,
            position=(-1.0, 0.0, 0.0),
        ),
    ]

    # delta_theta = 0.1 rad (~5.7 deg), k = 2.0 -> threshold = 0.2 rad.
    # Distance between (1, 0, 0) and (-1, 0, 0) is pi rad (~180 deg) >> 0.2 rad.
    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.1,
        k_pitch=2.0,
        sphere_radius=1.0,
    )

    assert len(clusters) == 2
    assert not any(cl.is_multi_cp for cl in clusters)
    assert clusters[0].members[0].vertex_id in (0, 1)
    assert clusters[1].members[0].vertex_id in (0, 1)
    assert len(clusters[0].boundary_ring) == 48


def test_clustered_close_points() -> None:
    """Test grouping of close CPs into a single micro-cluster."""
    # Two points separated by 3 degrees around the North pole
    ang1 = math.radians(1.0)
    ang2 = math.radians(4.0)

    p1 = (math.sin(ang1), 0.0, math.cos(ang1))
    p2 = (math.sin(ang2), 0.0, math.cos(ang2))
    p3 = (0.0, 1.0, 0.0)  # Distant point at equator

    cps = [
        CriticalPoint(vertex_id=10, cp_type="maximum", value=2.0, position=p1),
        CriticalPoint(vertex_id=11, cp_type="saddle", value=1.5, position=p2),
        CriticalPoint(vertex_id=20, cp_type="minimum", value=-2.0, position=p3),
    ]

    # delta_theta = 0.05 rad (~2.86 deg), k = 2.0 -> threshold = 0.10 rad (~5.73 deg)
    # Separation between p1 and p2 is ~3.0 deg <= 5.73 deg -> clustered!
    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.05,
        k_pitch=2.0,
        sphere_radius=1.0,
    )

    assert len(clusters) == 2
    # First cluster should be the multi-CP cluster
    mc_multi = clusters[0]
    assert mc_multi.is_multi_cp
    assert len(mc_multi.members) == 2
    assert {m.vertex_id for m in mc_multi.members} == {10, 11}
    assert mc_multi.n_maxima == 1
    assert mc_multi.n_saddles == 1
    assert mc_multi.n_minima == 0
    assert mc_multi.total_saddle_index == 1
    assert mc_multi.local_euler_index == 0  # 1 max - 1 saddle = 0
    assert mc_multi.composition_summary == "1 Max, 1 Sad"

    # Second cluster is isolated minimum
    mc_iso = clusters[1]
    assert not mc_iso.is_multi_cp
    assert mc_iso.members[0].vertex_id == 20
    assert mc_iso.local_euler_index == 1


def test_higher_order_saddle_cluster_accounting() -> None:
    """Test micro-cluster index accounting with higher-order saddles (e.g. MC7 scenario)."""
    # 4 Maxima and 1 Saddle with multiplicity 3 located close together
    ang = math.radians(1.0)
    p_center = (0.0, 0.0, 1.0)
    p_max1 = (math.sin(ang), 0.0, math.cos(ang))
    p_max2 = (-math.sin(ang), 0.0, math.cos(ang))
    p_max3 = (0.0, math.sin(ang), math.cos(ang))
    p_max4 = (0.0, -math.sin(ang), math.cos(ang))

    cps = [
        CriticalPoint(vertex_id=1, cp_type="maximum", value=2.0, position=p_max1),
        CriticalPoint(vertex_id=2, cp_type="maximum", value=2.0, position=p_max2),
        CriticalPoint(vertex_id=3, cp_type="maximum", value=2.0, position=p_max3),
        CriticalPoint(vertex_id=4, cp_type="maximum", value=2.0, position=p_max4),
        CriticalPoint(
            vertex_id=5,
            cp_type="saddle",
            value=1.0,
            position=p_center,
            multiplicity=3,
        ),
    ]

    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.05,
        k_pitch=2.0,
        sphere_radius=1.0,
    )

    assert len(clusters) == 1
    mc = clusters[0]
    assert mc.is_multi_cp
    assert len(mc.members) == 5
    assert mc.n_maxima == 4
    assert mc.n_saddles == 1
    assert mc.n_minima == 0
    assert mc.total_saddle_index == 3
    # 4 Max - 3 (saddle index) = +1
    assert mc.local_euler_index == 1
    assert mc.composition_summary == "4 Max, 1 Sad (mult 3)"


def test_boundary_ring_generation() -> None:
    """Test boundary ring sampling around a centroid."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=2.0,
        angular_radius_rad=math.radians(30.0),
        num_samples=16,
    )
    assert len(ring) == 16
    for pt in ring:
        # Distance from origin must equal sphere_radius
        norm_val = np.linalg.norm(np.array(pt.position))
        assert pytest.approx(norm_val, abs=1e-6) == 2.0
        # z-coordinate should be 2.0 * cos(30 deg) = 2.0 * sqrt(3)/2 = sqrt(3) ~ 1.73205
        assert pytest.approx(pt.position[2], abs=1e-6) == 2.0 * math.cos(
            math.radians(30.0)
        )


def test_micro_cluster_validation() -> None:
    """Test parameter validation in compute_micro_clusters."""
    with pytest.raises(ValueError):
        compute_micro_clusters([], delta_theta_mesh_rad=-0.1)

    with pytest.raises(ValueError):
        compute_micro_clusters([], delta_theta_mesh_rad=0.1, k_pitch=0.0)

    # Empty list returns empty list
    assert compute_micro_clusters([], delta_theta_mesh_rad=0.1) == []
