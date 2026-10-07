"""Unit tests for excision_boundary module."""

import math

import numpy as np
import pytest

from gba_topology2.src.excision_boundary import (
    extract_boundary_ports,
    interpolate_scalar_at_points,
    profile_boundary_loop,
)
from gba_topology2.src.micro_cluster import (
    generate_boundary_ring,
)


def _generate_fibonacci_sphere(
    num_points: int = 1200, radius: float = 1.0
) -> np.ndarray:
    """Generate uniform points on sphere."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = math.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def test_2fold_simple_saddle_ports() -> None:
    """Test 2-fold simple saddle field f = x^2 - y^2: must yield 2 valleys, 2 ridges, chi = -1."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(15.0),
        num_samples=72,
    )

    # Around north pole, f(phi) ~ cos(2*phi): has 2 maxima (phi = 0, pi) and 2 minima (phi = pi/2, 3pi/2)
    # Generate profile directly:
    vals = [math.cos(2.0 * rp.phi_rad) for rp in ring]

    valleys, ridges = extract_boundary_ports(vals, ring)
    assert len(valleys) == 2
    assert len(ridges) == 2

    # Check angular spacing: valleys should be spaced ~180 deg apart
    val_diff = abs(valleys[1].phi_deg - valleys[0].phi_deg)
    assert pytest.approx(val_diff, abs=10.0) == 180.0

    # Implied index: chi = 1 - 2 = -1
    chi = 1 - len(valleys)
    assert chi == -1


def test_3fold_monkey_saddle_ports() -> None:
    """Test 3-fold monkey saddle field f ~ cos(3*phi): must yield 3 valleys, 3 ridges, chi = -2."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(20.0),
        num_samples=72,
    )

    # 3-fold profile
    vals = [math.cos(3.0 * rp.phi_rad) for rp in ring]

    valleys, ridges = extract_boundary_ports(vals, ring)
    assert len(valleys) == 3
    assert len(ridges) == 3

    chi = 1 - len(valleys)
    assert chi == -2


def test_4fold_octupolar_cross_ports() -> None:
    """Test 4-fold octupolar cross field f ~ cos(4*phi): must yield 4 valleys, 4 ridges, chi = -3."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(25.0),
        num_samples=96,
    )

    # 4-fold profile
    vals = [math.cos(4.0 * rp.phi_rad) for rp in ring]

    valleys, ridges = extract_boundary_ports(vals, ring)
    assert len(valleys) == 4
    assert len(ridges) == 4

    chi = 1 - len(valleys)
    assert chi == -3


def test_profile_boundary_loop_with_mesh_interpolation() -> None:
    """Test end-to-end boundary loop profiling with IDW mesh interpolation."""
    pts = _generate_fibonacci_sphere(num_points=1500, radius=1.0)
    # Quadrupole field f(x, y, z) = x^2 - y^2
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2

    # Ring around North pole (0, 0, 1)
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(18.0),
        num_samples=72,
    )

    result = profile_boundary_loop(
        ring_points=ring,
        mesh_pts=pts,
        f_vals=f_vals,
    )

    assert result.is_topologically_balanced
    assert result.fold_order == 2
    assert result.implied_euler_index == -1
    assert len(result.valley_ports) == 2
    assert len(result.ridge_ports) == 2


def test_interpolate_scalar_exact_match() -> None:
    """Test IDW interpolation when query point exactly matches a mesh vertex."""
    pts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    f_vals = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    res = interpolate_scalar_at_points(
        query_pts=pts,
        mesh_pts=pts,
        f_vals=f_vals,
        k_neighbors=2,
    )

    assert np.allclose(res, f_vals, atol=1e-5)


def test_empty_boundary_loop() -> None:
    """Test empty ring input handles gracefully."""
    res = profile_boundary_loop([], np.zeros((3, 3)), np.zeros(3))
    assert res.fold_order == 0
    assert res.implied_euler_index == 1
    assert res.valley_ports == []
    assert res.ridge_ports == []
