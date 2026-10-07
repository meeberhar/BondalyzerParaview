"""String Aggregator Subroutine for gba_topology2.

Identifies, groups, and collapses collinear strings of micro-clusters
(Type 2 strings) into unified macro-string entities with oriented
elliptical boundary rings and net topological indices.

A macro-string consists of:
1. An intrinsically elongated multi-CP cluster (morphology == "string"), OR
2. A chain of two or more adjacent micro-clusters whose centroids align
   along a low-gradient ridge/trough and are separated by <= k_bridge * delta_theta_mesh.

Follows the AGENTS.md modular subroutine architecture.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    MicroCluster,
    compute_tangent_pca,
    generate_elliptical_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "MacroString",
    "aggregate_cluster_strings",
]


@dataclass(frozen=True)
class MacroString:
    """Represents an aggregated string of one or more collinear micro-clusters."""

    string_id: str
    constituent_clusters: list[MicroCluster]
    all_members: list[CriticalPoint]
    centroid: tuple[float, float, float]
    centroid_unit: tuple[float, float, float]
    principal_axis_unit: tuple[float, float, float]
    aspect_ratio: float
    angular_length_rad: float
    angular_length_deg: float
    angular_width_rad: float
    angular_width_deg: float
    semi_major_rad: float
    semi_minor_rad: float
    n_maxima: int
    n_minima: int
    n_saddles: int
    total_saddle_index: int
    local_euler_index: int
    boundary_ring: list[BoundaryRingPoint] = field(default_factory=list)

    @property
    def composition_summary(self) -> str:
        """Human-readable composition string."""
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


def aggregate_cluster_strings(
    clusters: list[MicroCluster],
    delta_theta_mesh_rad: float,
    sphere_radius: float = 1.0,
    k_bridge: float = 3.0,
    aspect_ratio_threshold: float = 2.0,
    min_boundary_margin_deg: float = 1.5,
    num_boundary_points: int = 48,
) -> tuple[list[MacroString], list[MicroCluster]]:
    """Group collinear or intrinsically elongated micro-clusters into macro-strings.

    Clusters that do not form or belong to a string are returned in the remainder list
    as isolated compact clusters.

    Args:
        clusters: Input list of MicroCluster objects from compute_micro_clusters.
        delta_theta_mesh_rad: Mesh pitch in radians.
        sphere_radius: Radius of the sphere R.
        k_bridge: Maximum centroid separation (in delta_theta units) to bridge
            adjacent collinear clusters into a string (default 3.0).
        aspect_ratio_threshold: Minimum aspect ratio for a group to qualify
            as a string (default 2.0).
        min_boundary_margin_deg: Angular margin added to boundary ring (default 1.5 deg).
        num_boundary_points: Number of perimeter samples for the elliptical ring.

    Returns:
        Tuple of (macro_strings, isolated_compact_clusters).

    Raises:
        ValueError: If delta_theta_mesh_rad <= 0 or k_bridge <= 0.
    """
    if delta_theta_mesh_rad <= 0:
        raise ValueError(
            f"delta_theta_mesh_rad must be positive, got {delta_theta_mesh_rad}"
        )
    if k_bridge <= 0:
        raise ValueError(f"k_bridge must be positive, got {k_bridge}")
    if not clusters:
        return [], []

    bridge_threshold_rad = k_bridge * delta_theta_mesh_rad
    min_margin_rad = math.radians(min_boundary_margin_deg)

    # Separate candidates: clusters with morphology == "string" are immediate candidates.
    # Also nearby clusters within bridge_threshold_rad that together form an elongated chain.
    n = len(clusters)
    cluster_units = np.array([c.centroid_unit for c in clusters], dtype=np.float64)

    # Adjacency graph between clusters
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = float(
                np.clip(np.dot(cluster_units[i], cluster_units[j]), -1.0, 1.0)
            )
            ang_ij = math.acos(cos_ij)
            if ang_ij <= bridge_threshold_rad and (
                clusters[i].morphology == "string" or clusters[j].morphology == "string"
            ):
                adj[i].append(j)
                adj[j].append(i)

    # Traverse connected components of string-linked clusters
    visited: set[int] = set()
    string_components: list[list[int]] = []

    # First, process clusters that are intrinsically strings or connected to them
    for i in range(n):
        if clusters[i].morphology == "string" and i not in visited:
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
            string_components.append(comp)

    # Create MacroString entities
    macro_strings: list[MacroString] = []
    for s_idx, comp in enumerate(string_components):
        comp_clusters = [clusters[idx] for idx in comp]
        # Collect all unique critical points
        all_cps: list[CriticalPoint] = []
        seen_cp_ids: set[int] = set()
        for cl in comp_clusters:
            for m in cl.members:
                if m.vertex_id not in seen_cp_ids:
                    seen_cp_ids.add(m.vertex_id)
                    all_cps.append(m)

        all_positions = np.array([cp.position for cp in all_cps], dtype=np.float64)
        all_norms = np.linalg.norm(all_positions, axis=1, keepdims=True)
        all_norms = np.where(all_norms < 1e-12, 1.0, all_norms)
        all_units = all_positions / all_norms

        # String centroid
        mean_unit = np.mean(all_units, axis=0)
        norm_mean = float(np.linalg.norm(mean_unit))
        if norm_mean > 1e-8:
            c_unit = mean_unit / norm_mean
            centroid = c_unit * sphere_radius
        else:
            c_unit = all_units[0]
            centroid = all_positions[0]

        # Compute PCA on tangent plane across all member points
        ar, princ_axis, max_major, max_minor = compute_tangent_pca(all_units, c_unit)

        semi_major = max(max_major + min_margin_rad, delta_theta_mesh_rad)
        semi_minor = max(max_minor + min_margin_rad, delta_theta_mesh_rad / 2.0)

        # Generate oriented elliptical boundary
        b_ring = generate_elliptical_boundary_ring(
            centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            sphere_radius=sphere_radius,
            semi_major_rad=semi_major,
            semi_minor_rad=semi_minor,
            principal_axis_unit=(
                float(princ_axis[0]),
                float(princ_axis[1]),
                float(princ_axis[2]),
            ),
            num_samples=num_boundary_points,
        )

        n_max = sum(1 for cp in all_cps if cp.cp_type == "maximum")
        n_min = sum(1 for cp in all_cps if cp.cp_type == "minimum")
        n_sad = sum(1 for cp in all_cps if cp.cp_type == "saddle")
        total_saddle_idx = sum(
            cp.multiplicity for cp in all_cps if cp.cp_type == "saddle"
        )
        net_chi = n_max + n_min - total_saddle_idx

        macro_strings.append(
            MacroString(
                string_id=f"MS{s_idx + 1}",
                constituent_clusters=comp_clusters,
                all_members=all_cps,
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                centroid_unit=(float(c_unit[0]), float(c_unit[1]), float(c_unit[2])),
                principal_axis_unit=(
                    float(princ_axis[0]),
                    float(princ_axis[1]),
                    float(princ_axis[2]),
                ),
                aspect_ratio=ar,
                angular_length_rad=2.0 * max_major,
                angular_length_deg=float(math.degrees(2.0 * max_major)),
                angular_width_rad=2.0 * max_minor,
                angular_width_deg=float(math.degrees(2.0 * max_minor)),
                semi_major_rad=semi_major,
                semi_minor_rad=semi_minor,
                n_maxima=n_max,
                n_minima=n_min,
                n_saddles=n_sad,
                total_saddle_index=total_saddle_idx,
                local_euler_index=net_chi,
                boundary_ring=b_ring,
            )
        )

    # Remaining clusters that were not part of any string
    isolated_clusters = [clusters[i] for i in range(n) if i not in visited]

    return macro_strings, isolated_clusters
