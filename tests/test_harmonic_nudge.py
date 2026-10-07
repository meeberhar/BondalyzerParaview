"""Unit tests for harmonic_nudge module."""

import math

import numpy as np
import pytest

from gba_topology2.src.excision_boundary import BoundaryPort
from gba_topology2.src.harmonic_nudge import (
    compute_k_fold_symmetry_energy,
    harmonic_nudge_critical_point,
    perfect_boundary_ports,
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


def test_extremum_circular_variance_energy() -> None:
    """Test symmetry energy for an extremum (k=0) on sphere."""
    pts = _generate_fibonacci_sphere(num_points=1000, radius=1.0)
    # Perfectly symmetric peak at North pole: f(z) = z
    f_vals = pts[:, 2]

    # At North pole, level curves are circles, so variance along probe ring should be ~ 0
    e_center = compute_k_fold_symmetry_energy(
        center_pos=(0.0, 0.0, 1.0),
        mesh_pts=pts,
        f_vals=f_vals,
        fold_order=0,
        probe_radius_rad=math.radians(10.0),
    )

    # Offset from North pole should have higher variance
    e_offset = compute_k_fold_symmetry_energy(
        center_pos=(0.2, 0.0, math.sqrt(1.0 - 0.04)),
        mesh_pts=pts,
        f_vals=f_vals,
        fold_order=0,
        probe_radius_rad=math.radians(10.0),
    )

    assert e_center < e_offset


def test_2fold_saddle_energy() -> None:
    """Test symmetry energy for 2-fold quadrupole saddle f = x^2 - y^2."""
    pts = _generate_fibonacci_sphere(num_points=1200, radius=1.0)
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2

    # Center at North pole (0, 0, 1) has pure 2-fold quadrupole symmetry
    e_center = compute_k_fold_symmetry_energy(
        center_pos=(0.0, 0.0, 1.0),
        mesh_pts=pts,
        f_vals=f_vals,
        fold_order=2,
        probe_radius_rad=math.radians(15.0),
    )

    # Off-center position has phase distortion / mixed harmonics
    e_offset = compute_k_fold_symmetry_energy(
        center_pos=(0.25, 0.1, math.sqrt(1.0 - 0.25**2 - 0.1**2)),
        mesh_pts=pts,
        f_vals=f_vals,
        fold_order=2,
        probe_radius_rad=math.radians(15.0),
    )

    assert e_center < e_offset


def test_harmonic_nudge_centers_perturbed_extremum() -> None:
    """Test that harmonic_nudge nudges a slightly perturbed extremum toward the true center."""
    pts = _generate_fibonacci_sphere(num_points=1500, radius=1.0)
    # Peak at North pole (0, 0, 1): f = z
    f_vals = pts[:, 2]

    # Perturbed initial position ~ 0.03 rad (1.7 deg) away from North pole
    ang_err = 0.03
    p_init = (math.sin(ang_err), 0.0, math.cos(ang_err))

    res = harmonic_nudge_critical_point(
        initial_pos=p_init,
        mesh_pts=pts,
        f_vals=f_vals,
        fold_order=0,
        delta_theta_mesh_rad=0.03,
        sphere_radius=1.0,
        max_displacement_pitch=2.0,
        max_iterations=12,
    )

    assert res.final_energy <= res.initial_energy
    # Optimized position should be closer to North pole (higher z) than initial position
    assert res.optimized_position[2] >= res.initial_position[2]


def test_perfect_boundary_ports_spacing() -> None:
    """Test boundary port angle perfection to ideal 3-fold (120 deg) spacing."""
    # 3 ports with slight angular noise: ~0, ~115, ~245 deg
    raw_ports = [
        BoundaryPort("V1", "valley", math.radians(2.0), 2.0, (1.0, 0.0, 0.0), 0.5),
        BoundaryPort("V2", "valley", math.radians(115.0), 115.0, (0.0, 1.0, 0.0), 0.5),
        BoundaryPort("V3", "valley", math.radians(245.0), 245.0, (-1.0, 0.0, 0.0), 0.5),
    ]

    perf = perfect_boundary_ports(raw_ports, fold_order=3)
    assert len(perf) == 3

    # Check that consecutive perfected ports are spaced exactly 120 degrees apart
    diff1 = (perf[1].phi_deg - perf[0].phi_deg) % 360.0
    diff2 = (perf[2].phi_deg - perf[1].phi_deg) % 360.0
    diff3 = (perf[0].phi_deg - perf[2].phi_deg) % 360.0

    assert pytest.approx(diff1, abs=1e-4) == 120.0
    assert pytest.approx(diff2, abs=1e-4) == 120.0
    assert pytest.approx(diff3, abs=1e-4) == 120.0
