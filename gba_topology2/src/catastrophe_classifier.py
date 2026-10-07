"""Catastrophe Classifier Subroutine for gba_topology2.

Identifies, groups, and classifies multi-pole catastrophe unfoldings in the
meso-scale angular window:
- 3-fold (k = 3): Monkey Saddles (E-MSAD, chi = -2, 6 ports)
- 4-fold (k = 4): Octupolar / Cross Saddles (E-4SAD, chi = -3, 8 ports)

Distinguishes between:
1. Spurious Unfolding (Numerical Noise): tiny barrier depth and tight span.
   -> Fused into a single effective catastrophe entity (E-MSAD / E-4SAD).
2. Physical Symmetry Breaking: real, finite scalar barrier and resolved separation.
   -> Preserved as distinct physical simple saddles with a bifurcation confidence score.

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Scaled relative to delta_theta_mesh_rad.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    MicroCluster,
    generate_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "CatastropheCandidate",
    "EffectiveCatastrophe",
    "classify_catastrophes",
    "evaluate_constellation_metrics",
]


@dataclass(frozen=True)
class CatastropheCandidate:
    """Represents a potential multi-pole catastrophe constellation."""

    core_cluster: MicroCluster
    flanking_saddles: list[MicroCluster]
    fold_order: int  # k = 3 (monkey saddle) or k = 4 (octupole)
    net_euler_index: int  # chi = 1 - k (-2 for k=3, -3 for k=4)
    angular_span_rad: float
    angular_span_deg: float
    barrier_depth: float  # max |f_i - f_j| across members
    relative_barrier_depth: float  # barrier_depth / field_span
    angular_symmetry_std_deg: float  # deviation from 360/k spacing around core


@dataclass(frozen=True)
class EffectiveCatastrophe:
    """Represents a classified catastrophe entity."""

    entity_id: str
    catastrophe_type: str  # "E-MSAD" (k=3) or "E-4SAD" (k=4)
    classification: str  # "spurious_unfolding" or "physical_symmetry_breaking"
    is_fused: bool
    fold_order: int
    net_euler_index: int
    centroid: tuple[float, float, float]
    centroid_unit: tuple[float, float, float]
    angular_radius_rad: float
    angular_radius_deg: float
    barrier_depth: float
    relative_barrier_depth: float
    angular_span_deg: float
    bifurcation_score: float  # [0.0, 1.0] confidence in real symmetry breaking
    constituent_clusters: list[MicroCluster]
    all_members: list[CriticalPoint]
    boundary_ring: list[BoundaryRingPoint] = field(default_factory=list)


def evaluate_constellation_metrics(
    core: MicroCluster,
    flanking_saddles: list[MicroCluster],
    fold_order: int,
    field_span: float,
) -> tuple[float, float, float, float]:
    """Evaluate geometric span, barrier depth, relative barrier, and angular symmetry deviation.

    Args:
        core: Central core micro-cluster.
        flanking_saddles: List of flanking saddle micro-clusters.
        fold_order: Expected symmetry fold order (k=3 or k=4).
        field_span: Global or local scalar field span ptp(f).

    Returns:
        Tuple of (max_span_rad, barrier_depth, relative_barrier, angular_std_deg).
    """
    all_clusters = [core] + flanking_saddles
    all_units = np.array([cl.centroid_unit for cl in all_clusters], dtype=np.float64)

    # Maximum pairwise angular distance
    max_span_rad = 0.0
    for i in range(len(all_clusters)):
        for j in range(i + 1, len(all_clusters)):
            cos_ij = float(np.clip(np.dot(all_units[i], all_units[j]), -1.0, 1.0))
            max_span_rad = max(max_span_rad, math.acos(cos_ij))

    # Barrier depth across member critical points
    all_cps: list[CriticalPoint] = []
    for cl in all_clusters:
        all_cps.extend(cl.members)

    vals = [cp.value for cp in all_cps]
    barrier_depth = float(max(vals) - min(vals)) if vals else 0.0
    rel_barrier = barrier_depth / max(field_span, 1e-12)

    # Evaluate cyclic angular spacing of flanking saddles projected on core tangent plane
    core_u = np.array(core.centroid_unit, dtype=np.float64)
    if abs(float(core_u[2])) < 0.9:
        v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(core_u, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(core_u, t1)
    t2 /= np.linalg.norm(t2)

    azimuths: list[float] = []
    for s in flanking_saddles:
        s_u = np.array(s.centroid_unit, dtype=np.float64)
        cos_s = float(np.clip(np.dot(core_u, s_u), -1.0, 1.0))
        proj = s_u - cos_s * core_u
        norm_proj = float(np.linalg.norm(proj))
        if norm_proj > 1e-12:
            proj /= norm_proj
            phi = math.atan2(float(np.dot(proj, t2)), float(np.dot(proj, t1)))
            azimuths.append(phi % (2.0 * math.pi))

    azimuths.sort()
    diffs_deg: list[float] = []
    ideal_diff_deg = 360.0 / float(fold_order)
    for i in range(len(azimuths)):
        next_phi = azimuths[(i + 1) % len(azimuths)]
        if next_phi < azimuths[i]:
            next_phi += 2.0 * math.pi
        diff_deg = math.degrees(next_phi - azimuths[i])
        diffs_deg.append(abs(diff_deg - ideal_diff_deg))

    angular_std_deg = float(np.mean(diffs_deg)) if diffs_deg else 0.0

    return max_span_rad, barrier_depth, rel_barrier, angular_std_deg


def classify_catastrophes(
    clusters: list[MicroCluster],
    field_vals: npt.NDArray[np.floating[Any]],
    delta_theta_mesh_rad: float,
    sphere_radius: float = 1.0,
    min_search_pitch: float = 2.0,
    max_search_pitch: float = 12.0,
    barrier_tol: float = 0.005,
    angular_tol_mult: float = 4.0,
    num_boundary_points: int = 48,
) -> tuple[list[EffectiveCatastrophe], list[MicroCluster]]:
    """Identify and classify 3-fold and 4-fold catastrophe constellations.

    Distinguishes spurious discretization unfoldings (fused into E-MSAD/E-4SAD)
    from genuine physical symmetry breaking based on barrier depth and angular span.

    Args:
        clusters: List of input MicroCluster objects.
        field_vals: (N,) array of scalar field values.
        delta_theta_mesh_rad: Mesh pitch delta_theta in radians.
        sphere_radius: Sphere radius R.
        min_search_pitch: Minimum angular search radius (in pitch units, default 2.0).
        max_search_pitch: Maximum angular search radius (in pitch units, default 12.0).
        barrier_tol: Threshold relative barrier depth (default 0.005 = 0.5% dynamic range).
        angular_tol_mult: Threshold angular span multiplier (in delta_theta units, default 4.0).
        num_boundary_points: Number of perimeter samples for boundary rings.

    Returns:
        Tuple of (effective_catastrophes, unassociated_clusters).
    """
    if delta_theta_mesh_rad <= 0:
        raise ValueError(
            f"delta_theta_mesh_rad must be positive, got {delta_theta_mesh_rad}"
        )
    if min_search_pitch >= max_search_pitch:
        raise ValueError("min_search_pitch must be < max_search_pitch")
    if not clusters:
        return [], []

    field_span = float(np.ptp(field_vals)) if len(field_vals) > 0 else 1.0
    field_span = max(field_span, 1e-12)

    min_search_rad = min_search_pitch * delta_theta_mesh_rad
    max_search_rad = max_search_pitch * delta_theta_mesh_rad
    angular_tol_rad = angular_tol_mult * delta_theta_mesh_rad

    # Candidate cores: clusters with positive index (e.g. chi = +1 or +2)
    # Flanking candidates: clusters with saddles (n_saddles > 0)
    saddle_clusters = [cl for cl in clusters if cl.n_saddles > 0]
    core_clusters = [cl for cl in clusters if cl.local_euler_index > 0]

    used_cluster_ids: set[str] = set()
    catastrophes: list[EffectiveCatastrophe] = []

    # Search around each candidate core
    for core in core_clusters:
        if core.cluster_id in used_cluster_ids:
            continue

        core_u = np.array(core.centroid_unit, dtype=np.float64)

        # Find nearby flanking saddle clusters within [min_search_rad, max_search_rad]
        flanking: list[tuple[float, MicroCluster]] = []
        for s in saddle_clusters:
            if s.cluster_id == core.cluster_id or s.cluster_id in used_cluster_ids:
                continue
            s_u = np.array(s.centroid_unit, dtype=np.float64)
            cos_ang = float(np.clip(np.dot(core_u, s_u), -1.0, 1.0))
            ang = math.acos(cos_ang)
            if min_search_rad <= ang <= max_search_rad:
                flanking.append((ang, s))

        flanking.sort(key=lambda x: x[0])
        candidate_saddles = [s for _, s in flanking]

        # Check for 4-fold (Octupolar, k=4, target net index = -3) FIRST if >= 4 saddles,
        # then 3-fold (Monkey Saddle, k=3, target net index = -2).
        # Prioritizing the higher symmetry / larger candidate set prevents a 4-fold
        # configuration from being prematurely captured as an incomplete 3-fold set.
        found_fold: int | None = None
        selected_flanking: list[MicroCluster] = []

        # Try k = 4
        if len(candidate_saddles) >= 4:
            for i in range(len(candidate_saddles)):
                for j in range(i + 1, len(candidate_saddles)):
                    for k_idx in range(j + 1, len(candidate_saddles)):
                        for l_idx in range(k_idx + 1, len(candidate_saddles)):
                            quad = [
                                candidate_saddles[i],
                                candidate_saddles[j],
                                candidate_saddles[k_idx],
                                candidate_saddles[l_idx],
                            ]
                            net_idx = core.local_euler_index + sum(
                                s.local_euler_index for s in quad
                            )
                            if net_idx == -3:
                                found_fold = 4
                                selected_flanking = quad
                                break
                        if found_fold:
                            break
                    if found_fold:
                        break
                if found_fold:
                    break

        # Try k = 3 if not matched k = 4
        if not found_fold and len(candidate_saddles) >= 3:
            for i in range(len(candidate_saddles)):
                for j in range(i + 1, len(candidate_saddles)):
                    for k_idx in range(j + 1, len(candidate_saddles)):
                        trio = [
                            candidate_saddles[i],
                            candidate_saddles[j],
                            candidate_saddles[k_idx],
                        ]
                        net_idx = core.local_euler_index + sum(
                            s.local_euler_index for s in trio
                        )
                        if net_idx == -2:
                            found_fold = 3
                            selected_flanking = trio
                            break
                    if found_fold:
                        break
                if found_fold:
                    break

        if found_fold is not None and selected_flanking:
            # Compute sensitivity metrics
            span_rad, barrier_depth, rel_barrier, _ang_std = (
                evaluate_constellation_metrics(
                    core=core,
                    flanking_saddles=selected_flanking,
                    fold_order=found_fold,
                    field_span=field_span,
                )
            )

            # Bifurcation confidence score in [0.0, 1.0]
            # High barrier and wide separation => high confidence in physical symmetry breaking
            score_barrier = min(1.0, rel_barrier / max(barrier_tol * 2.0, 1e-6))
            score_span = min(1.0, span_rad / max(angular_tol_rad * 2.0, 1e-6))
            bifurcation_score = 0.5 * score_barrier + 0.5 * score_span

            # Decision logic:
            # Spurious if barrier < barrier_tol and span < angular_tol_rad
            is_spurious = (rel_barrier < barrier_tol) or (span_rad < angular_tol_rad)
            classification = (
                "spurious_unfolding" if is_spurious else "physical_symmetry_breaking"
            )
            is_fused = is_spurious

            # Constellation member critical points
            constellation_clusters = [core] + selected_flanking
            all_cps: list[CriticalPoint] = []
            for cl in constellation_clusters:
                all_cps.extend(cl.members)

            # Constellation centroid
            all_units = np.array([cp.position for cp in all_cps], dtype=np.float64)
            norms = np.linalg.norm(all_units, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            all_units /= norms
            mean_u = np.mean(all_units, axis=0)
            norm_m = float(np.linalg.norm(mean_u))
            c_unit = mean_u / norm_m if norm_m > 1e-12 else all_units[0]
            centroid = c_unit * sphere_radius

            cat_type = "E-MSAD" if found_fold == 3 else "E-4SAD"
            net_chi = 1 - found_fold

            # Generate boundary ring enclosing the constellation
            ring_rad = max(span_rad / 2.0 + delta_theta_mesh_rad, delta_theta_mesh_rad)
            b_ring = generate_boundary_ring(
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                sphere_radius=sphere_radius,
                angular_radius_rad=ring_rad,
                num_samples=num_boundary_points,
            )

            catastrophes.append(
                EffectiveCatastrophe(
                    entity_id=f"CAT_{len(catastrophes) + 1}",
                    catastrophe_type=cat_type,
                    classification=classification,
                    is_fused=is_fused,
                    fold_order=found_fold,
                    net_euler_index=net_chi,
                    centroid=(
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                    ),
                    centroid_unit=(
                        float(c_unit[0]),
                        float(c_unit[1]),
                        float(c_unit[2]),
                    ),
                    angular_radius_rad=ring_rad,
                    angular_radius_deg=float(math.degrees(ring_rad)),
                    barrier_depth=barrier_depth,
                    relative_barrier_depth=rel_barrier,
                    angular_span_deg=float(math.degrees(span_rad)),
                    bifurcation_score=bifurcation_score,
                    constituent_clusters=constellation_clusters,
                    all_members=all_cps,
                    boundary_ring=b_ring,
                )
            )

            # Mark constituent clusters as used
            for cl in constellation_clusters:
                used_cluster_ids.add(cl.cluster_id)

    # Remaining unassociated clusters
    unassociated = [cl for cl in clusters if cl.cluster_id not in used_cluster_ids]

    return catastrophes, unassociated
