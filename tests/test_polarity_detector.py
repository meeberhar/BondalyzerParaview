"""Unit tests for polarity_detector module."""

import math

import numpy as np
import pytest

from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.polarity_detector import (
    classify_cluster_polarity,
    effective_reduction_decision,
    evaluate_spherical_radial_slope,
)


def _generate_synthetic_sphere_points(
    num_points: int = 500, radius: float = 1.0
) -> np.ndarray:
    """Generate approximately uniform points on a sphere via Fibonacci lattice."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = math.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def test_evaluate_spherical_radial_slope_peak() -> None:
    """Test radial slope evaluation for an analytical Gaussian peak at North pole."""
    pts = _generate_synthetic_sphere_points(1000, radius=1.0)
    # Peak at North pole (0, 0, 1): f(theta) = cos(theta) (descends with theta away from pole)
    z = pts[:, 2]
    f_vals = z  # Maximum at z=1 (theta=0), decreases as theta increases

    slope, conf = evaluate_spherical_radial_slope(
        centroid_unit=(0.0, 0.0, 1.0),
        mesh_pts=pts,
        f_vals=f_vals,
        inner_radius_rad=0.05,
        outer_radius_rad=0.25,
        sphere_radius=1.0,
    )

    # For f = cos(theta), df/d(theta) = -sin(theta) < 0
    assert slope < 0.0
    assert conf >= 0.7


def test_evaluate_spherical_radial_slope_basin() -> None:
    """Test radial slope evaluation for an analytical basin at South pole."""
    pts = _generate_synthetic_sphere_points(1000, radius=1.0)
    # Basin at South pole (0, 0, -1): f = z (minimum at z=-1, increases as theta from South pole increases)
    z = pts[:, 2]
    f_vals = z

    slope, conf = evaluate_spherical_radial_slope(
        centroid_unit=(0.0, 0.0, -1.0),
        mesh_pts=pts,
        f_vals=f_vals,
        inner_radius_rad=0.05,
        outer_radius_rad=0.25,
        sphere_radius=1.0,
    )

    # Away from South pole, z increases, so df/d(theta) > 0
    assert slope > 0.0
    assert conf >= 0.7


def test_classify_cluster_polarity_peak() -> None:
    """Test classifying a composite micro-cluster with net index +1 as E-MAX."""
    pts = _generate_synthetic_sphere_points(1000, radius=1.0)
    # Field has a peak at North Pole
    f_vals = pts[:, 2] ** 2 * np.sign(pts[:, 2])

    member_cps = [
        CriticalPoint(
            vertex_id=0, cp_type="maximum", value=1.0, position=(0.0, 0.0, 1.0)
        ),
        CriticalPoint(
            vertex_id=1, cp_type="saddle", value=0.95, position=(0.02, 0.0, 0.9998)
        ),
        CriticalPoint(
            vertex_id=2, cp_type="maximum", value=0.98, position=(0.04, 0.0, 0.9992)
        ),
    ]

    cluster = MicroCluster(
        cluster_id="MC_TEST_PEAK",
        members=member_cps,
        centroid=(0.02, 0.0, 0.9998),
        centroid_unit=(0.02, 0.0, 0.9998),
        angular_radius_rad=0.05,
        angular_radius_deg=math.degrees(0.05),
        angular_diameter_rad=0.04,
        angular_diameter_deg=math.degrees(0.04),
        spatial_diameter=0.04,
        n_maxima=2,
        n_minima=0,
        n_saddles=1,
        total_saddle_index=1,
        local_euler_index=1,  # 2 - 1 = +1
        is_multi_cp=True,
    )

    eff = classify_cluster_polarity(
        cluster=cluster,
        mesh_pts=pts,
        f_vals=f_vals,
        delta_theta_mesh_rad=0.02,
        sphere_radius=1.0,
    )

    assert eff.extremum_type == "E-MAX"
    assert eff.radial_slope < 0.0
    assert eff.effective_value == 1.0
    assert eff.confidence >= 0.7


def test_classify_cluster_polarity_basin() -> None:
    """Test classifying a composite micro-cluster with net index +1 as E-MIN."""
    pts = _generate_synthetic_sphere_points(1000, radius=1.0)
    # Field has a basin at South Pole: f(z) = (z + 1)^2 - 1
    f_vals = pts[:, 2]

    member_cps = [
        CriticalPoint(
            vertex_id=10, cp_type="minimum", value=-1.0, position=(0.0, 0.0, -1.0)
        ),
        CriticalPoint(
            vertex_id=11, cp_type="saddle", value=-0.95, position=(0.02, 0.0, -0.9998)
        ),
        CriticalPoint(
            vertex_id=12, cp_type="minimum", value=-0.98, position=(0.04, 0.0, -0.9992)
        ),
    ]

    cluster = MicroCluster(
        cluster_id="MC_TEST_BASIN",
        members=member_cps,
        centroid=(0.02, 0.0, -0.9998),
        centroid_unit=(0.02, 0.0, -0.9998),
        angular_radius_rad=0.05,
        angular_radius_deg=math.degrees(0.05),
        angular_diameter_rad=0.04,
        angular_diameter_deg=math.degrees(0.04),
        spatial_diameter=0.04,
        n_maxima=0,
        n_minima=2,
        n_saddles=1,
        total_saddle_index=1,
        local_euler_index=1,  # 2 - 1 = +1
        is_multi_cp=True,
    )

    eff = classify_cluster_polarity(
        cluster=cluster,
        mesh_pts=pts,
        f_vals=f_vals,
        delta_theta_mesh_rad=0.02,
        sphere_radius=1.0,
    )

    assert eff.extremum_type == "E-MIN"
    assert eff.radial_slope > 0.0
    assert eff.effective_value == -1.0
    assert eff.confidence >= 0.7


def test_effective_reduction_decision_chi_zero_annihilates() -> None:
    """A chi = 0 set is topologically neutral and must reduce to nothing."""
    assert effective_reduction_decision(0) == "E-NONE"
    # Polarity is irrelevant for a neutral set.
    assert effective_reduction_decision(0, "E-MIN") == "E-NONE"


def test_effective_reduction_decision_chi_positive_uses_polarity() -> None:
    """A chi = +1 set reduces to the polarity-resolved effective extremum."""
    assert effective_reduction_decision(1, "E-MAX") == "E-MAX"
    assert effective_reduction_decision(1, "E-MIN") == "E-MIN"


def test_effective_reduction_decision_chi_negative_is_saddle() -> None:
    """A chi <= -1 set reduces to an effective saddle carrying that index."""
    assert effective_reduction_decision(-1) == "E-SAD"
    assert effective_reduction_decision(-2) == "E-SAD"
    assert effective_reduction_decision(-7) == "E-SAD"


def test_effective_reduction_decision_chi_ge_two_irreducible() -> None:
    """A chi >= +2 set cannot be carried by one effective CP and stays expanded."""
    assert effective_reduction_decision(2) == "IRREDUCIBLE"
    assert effective_reduction_decision(5, "E-MIN") == "IRREDUCIBLE"


def test_effective_reduction_decision_rejects_bad_polarity() -> None:
    """An invalid polarity is rejected when the index requires it."""
    with pytest.raises(ValueError):
        effective_reduction_decision(1, "E-SAD")


def test_effective_reduction_decision_fold_order_relation() -> None:
    """The k = 1 - chi fold-order relation is well-defined for saddle reductions."""
    for chi in (-1, -2, -3):
        assert effective_reduction_decision(chi) == "E-SAD"
        assert 1 - chi >= 2
