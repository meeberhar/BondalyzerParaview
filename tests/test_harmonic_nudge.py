"""Unit tests for harmonic_nudge module."""

import math

import numpy as np
import pytest

from gba_topology2.src.excision_boundary import BoundaryPort
from gba_topology2.src.harmonic_nudge import (
    compute_k_fold_symmetry_energy,
    harmonic_nudge_critical_point,
    perfect_boundary_ports,
    reposition_ports_on_ring,
)
from gba_topology2.src.micro_cluster import BoundaryRingPoint, generate_boundary_ring


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


def test_reposition_ports_round_trips_ring_samples() -> None:
    """Test that repositioning a port at a ring sample's phi recovers that sample position."""
    centroid = (0.2, -0.3, 0.9)
    radius = 1.0
    ang_rad = math.radians(12.0)
    ring = generate_boundary_ring(
        centroid=centroid,
        sphere_radius=radius,
        angular_radius_rad=ang_rad,
        num_samples=48,
    )

    # Ports taken directly from ring samples at phi = 0, 120, 240 deg
    wanted_phi = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    ports: list[BoundaryPort] = []
    reference: list[tuple[float, float, float]] = []
    for i, phi in enumerate(wanted_phi):
        rp = min(ring, key=lambda r: abs(r.phi_rad - phi))
        ports.append(
            BoundaryPort(
                f"V{i}",
                "valley",
                rp.phi_rad,
                math.degrees(rp.phi_rad),
                rp.position,
                0.1,
            )
        )
        reference.append(rp.position)

    out = reposition_ports_on_ring(ports, ring, sphere_radius=radius)
    assert len(out) == 3
    for got, ref in zip(out, reference):
        dist = float(np.linalg.norm(np.array(got.position) - np.array(ref)))
        assert dist < 1e-6


def test_reposition_ports_preserve_perfect_spacing_on_ring() -> None:
    """Test perfected + repositioned ports are equispaced in 3D and lie on the ring."""
    centroid = (0.0, 0.0, 1.0)
    radius = 1.0
    ang_rad = math.radians(15.0)
    ring = generate_boundary_ring(
        centroid=centroid,
        sphere_radius=radius,
        angular_radius_rad=ang_rad,
        num_samples=72,
    )

    # Noisy port angles around a 3-fold constellation
    noisy = [10.0, 118.0, 244.0]
    ports = [
        BoundaryPort(
            f"R{i}",
            "ridge",
            math.radians(a),
            a,
            (math.cos(math.radians(a)), math.sin(math.radians(a)), 1.0),
            0.2,
        )
        for i, a in enumerate(noisy)
    ]

    perfected = perfect_boundary_ports(ports, fold_order=3)
    placed = reposition_ports_on_ring(perfected, ring, sphere_radius=radius)
    assert len(placed) == 3

    c = np.array(centroid, dtype=float)
    c_unit = c / np.linalg.norm(c)
    for p in placed:
        p_unit = np.array(p.position, dtype=float) / np.linalg.norm(p.position)
        # Every port must sit on the small circle at the ring's angular radius
        assert (
            pytest.approx(
                float(np.arccos(np.clip(p_unit @ c_unit, -1.0, 1.0))), abs=1e-6
            )
            == ang_rad
        )

    # Perfected azimuthal angles must be equispaced by 120 deg
    for i in range(3):
        dphi = (placed[(i + 1) % 3].phi_deg - placed[i].phi_deg) % 360.0
        assert pytest.approx(dphi, abs=1e-3) == 120.0

    # On a small circle, equal azimuthal spacing implies equal 3D chord spacing
    chords = []
    for i in range(3):
        a = np.array(placed[i].position, dtype=float)
        b = np.array(placed[(i + 1) % 3].position, dtype=float)
        chords.append(float(np.linalg.norm(a - b)))
    assert pytest.approx(chords[0], abs=1e-6) == chords[1]
    assert pytest.approx(chords[1], abs=1e-6) == chords[2]


def test_reposition_ports_reinterpolates_scalar() -> None:
    """Test scalar values are re-interpolated from the mesh at the new port positions."""
    pts = _generate_fibonacci_sphere(num_points=4000, radius=1.0)
    f_vals = pts[:, 2]  # f = z

    centroid = (0.0, 0.0, 1.0)
    ang_rad = math.radians(20.0)
    ring = generate_boundary_ring(
        centroid=centroid,
        sphere_radius=1.0,
        angular_radius_rad=ang_rad,
        num_samples=48,
    )
    ports = [
        BoundaryPort("V0", "valley", 0.0, 0.0, (1.0, 0.0, 0.0), -999.0),
    ]
    out = reposition_ports_on_ring(
        ports, ring, sphere_radius=1.0, f_vals=f_vals, mesh_pts=pts
    )
    assert len(out) == 1
    # Ring sits at a constant polar angle, so f = z = cos(20 deg) everywhere on it.
    # IDW over a discrete mesh carries a small interpolation error, so the sentinel
    # value -999.0 must be replaced by a close approximation of cos(20 deg).
    assert out[0].scalar_value > -900.0
    assert pytest.approx(out[0].scalar_value, abs=2e-3) == math.cos(ang_rad)


def test_reposition_ports_edge_cases() -> None:
    """Test empty input and insufficient ring samples."""
    assert reposition_ports_on_ring([], [], sphere_radius=1.0) == []

    port = BoundaryPort("V0", "valley", 0.0, 0.0, (1.0, 0.0, 0.0), 0.0)
    short_ring = [
        BoundaryRingPoint(phi_rad=0.0, position=(1.0, 0.0, 0.0)),
        BoundaryRingPoint(phi_rad=1.0, position=(0.0, 1.0, 0.0)),
    ]
    with pytest.raises(ValueError):
        reposition_ports_on_ring([port], short_ring, sphere_radius=1.0)
