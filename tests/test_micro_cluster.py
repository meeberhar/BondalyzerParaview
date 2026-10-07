"""Unit tests for micro_cluster module."""

import math

import numpy as np
import pytest

from gba_topology2.src.micro_cluster import (
    compute_micro_clusters,
    compute_tangent_pca,
    generate_boundary_ring,
    generate_elliptical_boundary_ring,
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

    with pytest.raises(ValueError):
        compute_micro_clusters([], delta_theta_mesh_rad=0.1, aspect_ratio_threshold=0.5)

    # Empty list returns empty list
    assert compute_micro_clusters([], delta_theta_mesh_rad=0.1) == []


def test_string_morphology_and_elliptical_ring() -> None:
    """Test PCA aspect ratio and elliptical boundary generation for linear strings."""
    # Linear chain of 4 points along the x-direction around the North pole (0, 0, 1)
    angles_x = [-0.03, -0.01, 0.01, 0.03]  # in radians
    cps = [
        CriticalPoint(
            vertex_id=i,
            cp_type="maximum" if i % 2 == 0 else "saddle",
            value=float(i),
            position=(math.sin(ax), 0.0, math.cos(ax)),
        )
        for i, ax in enumerate(angles_x)
    ]

    # Cluster threshold large enough to link them: 0.025 * 2.0 = 0.05 rad
    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.025,
        k_pitch=2.0,
        sphere_radius=1.0,
        aspect_ratio_threshold=2.0,
    )

    assert len(clusters) == 1
    mc = clusters[0]
    assert mc.is_multi_cp
    assert mc.morphology == "string"
    assert mc.aspect_ratio > 2.0
    assert mc.semi_major_rad > mc.semi_minor_rad

    # Verify elliptical boundary ring points lie on the sphere
    ring = mc.boundary_ring
    assert len(ring) == 48
    for pt in ring:
        norm_val = np.linalg.norm(np.array(pt.position))
        assert pytest.approx(norm_val, abs=1e-6) == 1.0

    # Also test standalone generate_elliptical_boundary_ring and compute_tangent_pca
    pts_units = np.array(
        [[math.sin(ax), 0.0, math.cos(ax)] for ax in angles_x], dtype=np.float64
    )
    c_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ar, pa, a_rad, b_rad = compute_tangent_pca(pts_units, c_unit)
    assert ar > 2.0
    direct_ring = generate_elliptical_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        semi_major_rad=a_rad,
        semi_minor_rad=b_rad,
        principal_axis_unit=(float(pa[0]), float(pa[1]), float(pa[2])),
        num_samples=16,
    )
    assert len(direct_ring) == 16


def test_compact_morphology() -> None:
    """Test compact (Type 1) quasi-circular cluster classification."""
    # Symmetric cross around North pole (aspect ratio ~ 1.0)
    ang = 0.02
    cps = [
        CriticalPoint(
            vertex_id=1,
            cp_type="maximum",
            value=1.0,
            position=(math.sin(ang), 0.0, math.cos(ang)),
        ),
        CriticalPoint(
            vertex_id=2,
            cp_type="maximum",
            value=1.0,
            position=(-math.sin(ang), 0.0, math.cos(ang)),
        ),
        CriticalPoint(
            vertex_id=3,
            cp_type="maximum",
            value=1.0,
            position=(0.0, math.sin(ang), math.cos(ang)),
        ),
        CriticalPoint(
            vertex_id=4,
            cp_type="maximum",
            value=1.0,
            position=(0.0, -math.sin(ang), math.cos(ang)),
        ),
    ]

    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.025,
        k_pitch=2.0,
        sphere_radius=1.0,
        aspect_ratio_threshold=2.0,
    )

    assert len(clusters) == 1
    mc = clusters[0]
    assert mc.is_multi_cp
    assert mc.morphology == "compact"
    assert mc.aspect_ratio < 1.5
    assert pytest.approx(mc.semi_major_rad, abs=1e-6) == mc.semi_minor_rad
