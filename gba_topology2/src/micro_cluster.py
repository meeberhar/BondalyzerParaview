"""Micro-Cluster Identification Subroutine.

Groups raw critical points into micro-clusters based on angular separation
scaled by mesh pitch:
    theta_ij <= k * delta_theta_mesh

Computes for each cluster:
- Centroid projected onto the sphere of radius R
- Member critical point IDs and types
- Angular span and spatial diameter
- Initial boundary ring vertices on the sphere
- Local Euler index: chi_micro = N_max + N_min - sum(saddle_multiplicities)
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "BoundaryRingPoint",
    "MicroCluster",
    "compute_micro_clusters",
    "generate_boundary_ring",
]


@dataclass(frozen=True)
class BoundaryRingPoint:
    """A point along the circular boundary ring on the sphere."""

    phi_rad: float
    position: tuple[float, float, float]


@dataclass(frozen=True)
class MicroCluster:
    """Represents a micro-cluster of one or more critical points."""

    cluster_id: str
    members: list[CriticalPoint]
    centroid: tuple[float, float, float]
    centroid_unit: tuple[float, float, float]
    angular_radius_rad: float
    angular_radius_deg: float
    angular_diameter_rad: float
    angular_diameter_deg: float
    spatial_diameter: float
    n_maxima: int
    n_minima: int
    n_saddles: int
    total_saddle_index: int = 0
    local_euler_index: int = 0
    is_multi_cp: bool = False
    boundary_ring: list[BoundaryRingPoint] = field(default_factory=list)

    @property
    def composition_summary(self) -> str:
        """Human-readable composition string, e.g. '2 Max, 1 Sad' or '4 Max, 1 Sad (mult 3)'."""
        parts: list[str] = []
        if self.n_maxima > 0:
            parts.append(f"{self.n_maxima} Max")
        if self.n_saddles > 0:
            if self.total_saddle_index > self.n_saddles:
                parts.append(f"{self.n_saddles} Sad (mult {self.total_saddle_index})")
            else:
                parts.append(f"{self.n_saddles} Sad")
        if self.n_minima > 0:
            parts.append(f"{self.n_minima} Min")
        return ", ".join(parts) if parts else "0 CPs"


def generate_boundary_ring(
    centroid: tuple[float, float, float],
    sphere_radius: float,
    angular_radius_rad: float,
    num_samples: int = 48,
) -> list[BoundaryRingPoint]:
    """Generate sample coordinates for a circular boundary ring on the sphere.

    Args:
        centroid: (x, y, z) coordinates of cluster centroid.
        sphere_radius: Radius R of the sphere.
        angular_radius_rad: Angular radius of the boundary ring in radians.
        num_samples: Number of circumferential sample points.

    Returns:
        List of BoundaryRingPoint objects forming the closed loop.
    """
    c_arr = np.array(centroid, dtype=np.float64)
    c_norm = float(np.linalg.norm(c_arr))
    if c_norm < 1e-12:
        c_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        c_unit = c_arr / c_norm

    # Find tangent orthonormal vectors t1, t2
    if abs(float(c_unit[2])) < 0.9:
        v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    t1 = np.cross(c_unit, v_temp)
    norm_t1 = float(np.linalg.norm(t1))
    if norm_t1 < 1e-12:
        v_temp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = np.cross(c_unit, v_temp)
        norm_t1 = float(np.linalg.norm(t1))
    t1 /= norm_t1
    t2 = np.cross(c_unit, t1)
    norm_t2 = float(np.linalg.norm(t2))
    if norm_t2 > 1e-12:
        t2 /= norm_t2

    cos_a = math.cos(angular_radius_rad)
    sin_a = math.sin(angular_radius_rad)

    ring: list[BoundaryRingPoint] = []
    for i in range(num_samples):
        phi = 2.0 * math.pi * float(i) / float(num_samples)
        dir_vec = cos_a * c_unit + sin_a * (math.cos(phi) * t1 + math.sin(phi) * t2)
        dir_vec /= np.linalg.norm(dir_vec)
        p = dir_vec * sphere_radius
        ring.append(
            BoundaryRingPoint(
                phi_rad=phi,
                position=(float(p[0]), float(p[1]), float(p[2])),
            )
        )
    return ring


def compute_micro_clusters(
    cps: list[CriticalPoint],
    delta_theta_mesh_rad: float,
    k_pitch: float = 2.0,
    sphere_radius: float = 1.0,
    min_boundary_margin_deg: float = 1.5,
    num_boundary_points: int = 48,
) -> list[MicroCluster]:
    """Cluster critical points within distance threshold d <= k * delta_theta_mesh.

    Args:
        cps: List of input CriticalPoint objects.
        delta_theta_mesh_rad: Mesh pitch delta_theta in radians.
        k_pitch: Multiplier for mesh pitch to define clustering radius (default 2.0).
        sphere_radius: Sphere radius R.
        min_boundary_margin_deg: Minimum angular boundary margin (default 1.5 deg).
        num_boundary_points: Sample points for boundary ring (default 48).

    Returns:
        List of MicroCluster objects sorted by member count descending.

    Raises:
        ValueError: If delta_theta_mesh_rad <= 0 or k_pitch <= 0.
    """
    if delta_theta_mesh_rad <= 0:
        raise ValueError(
            f"delta_theta_mesh_rad must be positive, got {delta_theta_mesh_rad}"
        )
    if k_pitch <= 0:
        raise ValueError(f"k_pitch must be positive, got {k_pitch}")
    if not cps:
        return []

    n = len(cps)
    threshold_rad = k_pitch * delta_theta_mesh_rad

    positions: npt.NDArray[np.float64] = np.array(
        [cp.position for cp in cps], dtype=np.float64
    )
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit_positions = positions / norms

    # Build adjacency based on angular distance <= threshold_rad
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = float(
                np.clip(np.dot(unit_positions[i], unit_positions[j]), -1.0, 1.0)
            )
            ang_ij = math.acos(cos_ij)
            if ang_ij <= threshold_rad:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components search
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(n):
        if i not in visited:
            comp: list[int] = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(comp)

    # Sort components: multi-element first, then largest first, then by earliest index
    components.sort(key=lambda comp: (-len(comp), min(comp)))

    min_margin_rad = math.radians(min_boundary_margin_deg)
    clusters: list[MicroCluster] = []

    for cl_idx, comp in enumerate(components):
        members = [cps[i] for i in comp]
        comp_positions = positions[comp]
        comp_units = unit_positions[comp]

        # Centroid projected onto sphere
        mean_unit = np.mean(comp_units, axis=0)
        norm_mean = float(np.linalg.norm(mean_unit))
        if norm_mean > 1e-8:
            centroid_unit = mean_unit / norm_mean
            centroid = centroid_unit * sphere_radius
        else:
            centroid_unit = comp_units[0]
            centroid = comp_positions[0]

        # Compute maximum pairwise angular distance and chord distance
        max_ang = 0.0
        max_dist = 0.0
        for i_idx in range(len(comp)):
            for j_idx in range(i_idx + 1, len(comp)):
                d = float(np.linalg.norm(comp_positions[i_idx] - comp_positions[j_idx]))
                cos_v = float(
                    np.clip(
                        np.dot(comp_units[i_idx], comp_units[j_idx]),
                        -1.0,
                        1.0,
                    )
                )
                ang = math.acos(cos_v)
                max_dist = max(max_dist, d)
                max_ang = max(max_ang, ang)

        # Angular radius from centroid to furthest member
        max_rad_from_c = 0.0
        for i_idx in range(len(comp)):
            cos_c = float(np.clip(np.dot(centroid_unit, comp_units[i_idx]), -1.0, 1.0))
            ang_c = math.acos(cos_c)
            max_rad_from_c = max(max_rad_from_c, ang_c)

        # Ring angular radius includes boundary margin
        ring_radius_rad = max(max_rad_from_c + min_margin_rad, threshold_rad / 2.0)
        ring_radius_deg = float(math.degrees(ring_radius_rad))

        n_max = sum(1 for m in members if m.cp_type == "maximum")
        n_min = sum(1 for m in members if m.cp_type == "minimum")
        n_sad = sum(1 for m in members if m.cp_type == "saddle")
        total_saddle_index = sum(
            m.multiplicity for m in members if m.cp_type == "saddle"
        )
        local_chi = n_max + n_min - total_saddle_index

        b_ring = generate_boundary_ring(
            centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            sphere_radius=sphere_radius,
            angular_radius_rad=ring_radius_rad,
            num_samples=num_boundary_points,
        )

        clusters.append(
            MicroCluster(
                cluster_id=f"MC{cl_idx + 1}",
                members=members,
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                centroid_unit=(
                    float(centroid_unit[0]),
                    float(centroid_unit[1]),
                    float(centroid_unit[2]),
                ),
                angular_radius_rad=ring_radius_rad,
                angular_radius_deg=ring_radius_deg,
                angular_diameter_rad=max_ang,
                angular_diameter_deg=float(math.degrees(max_ang)),
                spatial_diameter=max_dist,
                n_maxima=n_max,
                n_minima=n_min,
                n_saddles=n_sad,
                total_saddle_index=total_saddle_index,
                local_euler_index=local_chi,
                is_multi_cp=len(members) > 1,
                boundary_ring=b_ring,
            )
        )

    return clusters
