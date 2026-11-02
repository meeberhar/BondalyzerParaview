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

Flanking members are restricted to pure simple saddles (net index -1, no
extrema) by is_flanking_saddle_candidate, so composite saddle strings that
happen to carry net index -1 are never consumed as flanking members.

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Scaled relative to delta_theta_mesh_rad.
"""

import itertools
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
    "angular_separation_deg",
    "classify_catastrophes",
    "evaluate_constellation_metrics",
    "find_saddle_pairs",
    "is_bimodal_quad",
    "is_flanking_saddle_candidate",
    "split_quad_into_pairs",
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
    core: MicroCluster | None,
    flanking_saddles: list[MicroCluster],
    fold_order: int,
    field_span: float,
) -> tuple[float, float, float, float]:
    """Evaluate geometric span, barrier depth, relative barrier, and angular symmetry deviation.

    Args:
        core: Central core micro-cluster, or ``None`` for a coreless constellation
            (a saddle-saddle pair, whose azimuths are taken about its own centroid).
        flanking_saddles: List of flanking saddle micro-clusters.
        fold_order: Expected symmetry fold order (k=3 or k=4).
        field_span: Global or local scalar field span ptp(f).

    Returns:
        Tuple of (max_span_rad, barrier_depth, relative_barrier, angular_std_deg).
    """
    all_clusters = ([core] if core is not None else []) + flanking_saddles
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
    if core is not None:
        core_u = np.array(core.centroid_unit, dtype=np.float64)
    else:
        core_u = np.mean(all_units, axis=0)
        norm_u = float(np.linalg.norm(core_u))
        core_u = core_u / norm_u if norm_u > 1e-12 else np.array([0.0, 0.0, 1.0])
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


def constellation_span_deg(members: list[MicroCluster]) -> float:
    """Largest centroid-to-centroid angle (degrees) across a set of clusters.

    Used to rank candidate constellations so the tightest one wins.

    Args:
        members: The clusters forming the candidate constellation.

    Returns:
        Maximum pairwise angular separation in degrees (0.0 for fewer than two).
    """
    span = 0.0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            span = max(span, angular_separation_deg(members[i], members[j]))
    return float(span)


def angular_separation_deg(a: MicroCluster, b: MicroCluster) -> float:
    """Great-circle angle in degrees between two cluster centroids."""
    ua = np.array(a.centroid_unit, dtype=np.float64)
    ub = np.array(b.centroid_unit, dtype=np.float64)
    na = float(np.linalg.norm(ua))
    nb = float(np.linalg.norm(ub))
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    cos_ab = float(np.clip(float(np.dot(ua, ub)) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_ab)))


def split_quad_into_pairs(
    saddles: list[MicroCluster],
) -> tuple[list[tuple[MicroCluster, MicroCluster]], float, float] | None:
    """Partition four saddles into two pairs minimising the total intra-pair angle.

    A genuine octupolar (k = 4) unfolding places its four saddles at ~90 deg
    cyclic spacing, so every pairing has four comparable separations. Two
    independent monkey-saddle pairs place two saddles close together and the
    other two close together, so the best pairing has a small intra-pair angle
    and a large gap between the groups.

    Args:
        saddles: Exactly four saddle micro-clusters.

    Returns:
        Tuple of (pairs, max_intra_pair_deg, min_inter_pair_deg), or ``None``
        when ``saddles`` does not hold exactly four clusters.
    """
    if len(saddles) != 4:
        return None

    partitions = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    best: tuple[float, tuple[tuple[int, int], tuple[int, int]]] | None = None
    for part in partitions:
        (i0, i1), (j0, j1) = part
        total = angular_separation_deg(
            saddles[i0], saddles[i1]
        ) + angular_separation_deg(saddles[j0], saddles[j1])
        if best is None or total < best[0]:
            best = (total, part)
    if best is None:
        return None

    (_total, (pa, pb)) = best
    pairs = [(saddles[pa[0]], saddles[pa[1]]), (saddles[pb[0]], saddles[pb[1]])]
    max_intra = max(
        angular_separation_deg(pairs[0][0], pairs[0][1]),
        angular_separation_deg(pairs[1][0], pairs[1][1]),
    )
    min_inter = min(angular_separation_deg(a, b) for a in pairs[0] for b in pairs[1])
    return pairs, float(max_intra), float(min_inter)


def is_bimodal_quad(
    saddles: list[MicroCluster],
    pair_max_deg: float,
    bimodal_ratio: float,
) -> bool:
    """True when four saddles are two tight pairs rather than one 4-fold set.

    Args:
        saddles: The four candidate flanking saddles of a k = 4 constellation.
        pair_max_deg: Largest intra-pair angle still accepted as a bound pair.
        bimodal_ratio: Required ``min_inter_pair_deg / max_intra_pair_deg``.

    Returns:
        True when the set splits into two pairs that are each tight and
        mutually well separated -- i.e. not a single octupolar unfolding.
    """
    split = split_quad_into_pairs(saddles)
    if split is None:
        return False
    _pairs, max_intra, min_inter = split
    if max_intra > pair_max_deg:
        return False
    return min_inter >= bimodal_ratio * max_intra


def has_intervening_extremum(
    a: MicroCluster,
    b: MicroCluster,
    extrema: list[MicroCluster],
    margin_deg: float = 1.5,
) -> bool:
    """True when an extremum cluster sits on the geodesic between two saddles.

    An unfolded monkey saddle pair is joined by a separatrix with no extremum
    between the two poles; a real extremum on the midpoint means the saddles
    belong to separate structures and must not be fused.

    Args:
        a: First saddle cluster.
        b: Second saddle cluster.
        extrema: Clusters carrying at least one maximum or minimum.
        margin_deg: Angular slack added to half the pair separation.

    Returns:
        True when some extremum lies within ``sep/2 + margin_deg`` of the
        pair midpoint.
    """
    ua = np.array(a.centroid_unit, dtype=np.float64)
    ub = np.array(b.centroid_unit, dtype=np.float64)
    mid = ua + ub
    norm = float(np.linalg.norm(mid))
    if norm < 1e-12:
        return False
    mid /= norm
    sep = angular_separation_deg(a, b)
    limit = sep / 2.0 + margin_deg
    for ext in extrema:
        ue = np.array(ext.centroid_unit, dtype=np.float64)
        nu = float(np.linalg.norm(ue))
        if nu < 1e-12:
            continue
        d = math.degrees(
            math.acos(float(np.clip(float(np.dot(mid, ue)) / nu, -1.0, 1.0)))
        )
        if d < limit:
            return True
    return False


def nearest_core_distance_deg(
    a: MicroCluster,
    b: MicroCluster,
    cores: list[MicroCluster],
) -> float:
    """Angular distance from the pair midpoint to the nearest positive core.

    Args:
        a: First saddle cluster.
        b: Second saddle cluster.
        cores: Clusters with a positive local Euler index (extremum cores).

    Returns:
        Minimum great-circle angle in degrees between the pair midpoint and any
        core centroid, or ``180.0`` when there are no cores.
    """
    ua = np.array(a.centroid_unit, dtype=np.float64)
    ub = np.array(b.centroid_unit, dtype=np.float64)
    mid = ua + ub
    norm = float(np.linalg.norm(mid))
    if norm < 1e-12:
        return 180.0
    mid /= norm
    best = 180.0
    for core in cores:
        uc = np.array(core.centroid_unit, dtype=np.float64)
        nu = float(np.linalg.norm(uc))
        if nu < 1e-12:
            continue
        d = math.degrees(math.acos(float(np.clip(float(np.dot(mid, uc)) / nu, -1, 1))))
        best = min(best, d)
    return best


def find_saddle_pairs(
    saddle_clusters: list[MicroCluster],
    extrema: list[MicroCluster],
    cores: list[MicroCluster],
    field_span: float,
    pair_max_pitch: float,
    delta_theta_mesh_rad: float,
    barrier_tol: float,
    coreless_min_ratio: float,
    pair_max_deg_floor: float = 16.0,
) -> list[tuple[MicroCluster, MicroCluster]]:
    """Detect bound simple-saddle pairs that unfold a single monkey saddle.

    A monkey saddle (chi = -2) unfolds in two ways. On a smooth extremum it
    splits into a core (chi = +1) plus three saddles. On a flat ridge/valley
    there is no extremum to serve as a core, and it splits into **two** simple
    saddles whose indices sum to -2 directly. The core-centred search cannot
    see the second case, so those pairs are found here.

    A pair is accepted when the two saddles are close (separated by at most
    ``max(pair_max_pitch * delta_theta, pair_max_deg_floor)``), carry nearly the
    same scalar value (relative barrier below ``barrier_tol`` -- the signature of
    one critical point split by the mesh), have no extremum on the geodesic
    between them, and are **coreless**: the nearest positive core sits farther
    than ``coreless_min_ratio`` times the pair separation. The last test is what
    keeps this channel from stealing saddles that a core-centred constellation
    already explains -- a pair whose midpoint is only ~1 pair-width from a core
    is that core's flanking saddle, not a free-standing monkey saddle.

    Pairs are returned greedily in order of increasing separation, so each
    saddle is consumed by its nearest eligible partner.

    Args:
        saddle_clusters: Flanking-eligible simple-saddle clusters.
        extrema: Clusters carrying at least one maximum or minimum.
        cores: Positive-index clusters (candidate constellation cores).
        field_span: Scalar span ptp(f) used to normalise the barrier.
        pair_max_pitch: Maximum pair separation in mesh pitches.
        delta_theta_mesh_rad: Mesh pitch in radians.
        barrier_tol: Maximum relative value difference across the pair.
        coreless_min_ratio: Minimum ``nearest_core_dist / pair_sep`` for the
            pair to count as coreless.
        pair_max_deg_floor: Floor on the maximum pair separation in degrees
            (default 16.0) ensuring fine meshes can capture physical saddle
            unfoldings.

    Returns:
        Disjoint (saddle, saddle) pairs, tightest first.
    """
    pair_max_deg = max(
        pair_max_pitch * math.degrees(delta_theta_mesh_rad), pair_max_deg_floor
    )
    candidates: list[tuple[float, MicroCluster, MicroCluster]] = []
    for i in range(len(saddle_clusters)):
        for j in range(i + 1, len(saddle_clusters)):
            a = saddle_clusters[i]
            b = saddle_clusters[j]
            sep = angular_separation_deg(a, b)
            if sep > pair_max_deg:
                continue
            va = min(cp.value for cp in a.members)
            vb = min(cp.value for cp in b.members)
            ma = max(cp.value for cp in a.members)
            mb = max(cp.value for cp in b.members)
            lo = min(va, vb)
            hi = max(ma, mb)
            rel_barrier = (hi - lo) / max(field_span, 1e-12)
            if rel_barrier > barrier_tol:
                continue
            if has_intervening_extremum(a, b, extrema):
                continue
            if sep <= 1e-9:
                continue
            core_d = nearest_core_distance_deg(a, b, cores)
            if core_d < coreless_min_ratio * sep:
                continue
            candidates.append((sep, a, b))

    candidates.sort(key=lambda x: x[0])
    used: set[str] = set()
    pairs: list[tuple[MicroCluster, MicroCluster]] = []
    for _sep, a, b in candidates:
        if a.cluster_id in used or b.cluster_id in used:
            continue
        used.add(a.cluster_id)
        used.add(b.cluster_id)
        pairs.append((a, b))
    return pairs


def build_catastrophe(
    core: MicroCluster | None,
    flanking_saddles: list[MicroCluster],
    fold_order: int,
    entity_id: str,
    field_span: float,
    delta_theta_mesh_rad: float,
    sphere_radius: float,
    barrier_tol: float,
    angular_tol_rad: float,
    num_boundary_points: int,
) -> EffectiveCatastrophe:
    """Assemble one EffectiveCatastrophe from a core (or none) plus its saddles.

    Shared by the core-centred constellations and the coreless saddle-saddle
    pairs so both produce identical geometry, metrics and classification.

    Args:
        core: Extremum core cluster, or ``None`` for a saddle-saddle pair.
        flanking_saddles: The constellation's saddle members.
        fold_order: Symmetry order k (3 for monkey, 4 for octupolar).
        entity_id: Identifier to assign.
        field_span: Scalar span ptp(f).
        delta_theta_mesh_rad: Mesh pitch in radians.
        sphere_radius: Sphere radius R.
        barrier_tol: Relative barrier threshold.
        angular_tol_rad: Angular span threshold in radians.
        num_boundary_points: Perimeter samples for the boundary ring.

    Returns:
        The populated ``EffectiveCatastrophe``.
    """
    span_rad, barrier_depth, rel_barrier, _ang_std = evaluate_constellation_metrics(
        core=core,
        flanking_saddles=flanking_saddles,
        fold_order=fold_order,
        field_span=field_span,
    )

    score_barrier = min(1.0, rel_barrier / max(barrier_tol * 2.0, 1e-6))
    score_span = min(1.0, span_rad / max(angular_tol_rad * 2.0, 1e-6))
    bifurcation_score = 0.5 * score_barrier + 0.5 * score_span

    is_spurious = (rel_barrier < barrier_tol) or (span_rad < angular_tol_rad)
    classification = (
        "spurious_unfolding" if is_spurious else "physical_symmetry_breaking"
    )
    is_fused = is_spurious

    constellation_clusters = ([core] if core is not None else []) + flanking_saddles
    all_cps: list[CriticalPoint] = []
    for cl in constellation_clusters:
        all_cps.extend(cl.members)

    all_units = np.array([cp.position for cp in all_cps], dtype=np.float64)
    norms = np.linalg.norm(all_units, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    all_units /= norms
    mean_u = np.mean(all_units, axis=0)
    norm_m = float(np.linalg.norm(mean_u))
    c_unit = mean_u / norm_m if norm_m > 1e-12 else all_units[0]
    centroid = c_unit * sphere_radius

    cat_type = "E-MSAD" if fold_order == 3 else "E-4SAD"
    if core is not None:
        net_chi = core.local_euler_index + sum(
            cl.local_euler_index for cl in flanking_saddles
        )
    else:
        # A saddle-saddle pair carries the summed index of its members (-2),
        # which equals the monkey-saddle index 1 - k for k = 3.
        net_chi = sum(cl.local_euler_index for cl in flanking_saddles)

    ring_rad = max(span_rad / 2.0 + delta_theta_mesh_rad, delta_theta_mesh_rad)
    b_ring = generate_boundary_ring(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        sphere_radius=sphere_radius,
        angular_radius_rad=ring_rad,
        num_samples=num_boundary_points,
    )

    return EffectiveCatastrophe(
        entity_id=entity_id,
        catastrophe_type=cat_type,
        classification=classification,
        is_fused=is_fused,
        fold_order=fold_order,
        net_euler_index=net_chi,
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        centroid_unit=(float(c_unit[0]), float(c_unit[1]), float(c_unit[2])),
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
    pair_max_pitch: float = 5.0,
    bimodal_ratio: float = 2.5,
    coreless_min_ratio: float = 1.8,
    pair_max_deg_floor: float = 16.0,
) -> tuple[list[EffectiveCatastrophe], list[MicroCluster]]:
    """Identify and classify 3-fold and 4-fold catastrophe constellations.

    Distinguishes spurious discretization unfoldings (fused into E-MSAD/E-4SAD)
    from genuine physical symmetry breaking based on barrier depth and angular span.

    Two unfolding channels are searched, bound pairs first:

    1. **Saddle-saddle pairs** (``find_saddle_pairs``): a monkey saddle on a flat
       ridge splits into two simple saddles with no extremum between them, so
       ``chi = -1 + -1 = -2`` is reached without a core. These are claimed first
       because a core-centred search cannot see them and would otherwise absorb
       the pair into a larger, wrong constellation.
    2. **Core-centred constellations**: an extremum core (``chi = +1``) ringed by
       k saddles. A 4-fold candidate whose saddles are really two tight pairs is
       rejected by ``is_bimodal_quad`` so each pair resolves on its own.

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
        pair_max_pitch: Maximum saddle-saddle pair separation in pitch units
            (default 5.0) for the coreless monkey-saddle channel.
        bimodal_ratio: Required ratio of inter-pair gap to intra-pair angle
            (default 2.5) before a 4-saddle set is treated as two pairs and
            refused as a single octupole.
        coreless_min_ratio: Minimum ``nearest_core_dist / pair_sep`` (default
            1.8) for a saddle pair to be claimed as a free-standing monkey
            saddle rather than a core's flanking saddles.
        pair_max_deg_floor: Floor on the maximum pair separation in degrees
            (default 16.0) ensuring fine meshes can capture physical saddle
            unfoldings.

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
    # Flanking candidates: pure simple saddles (net index -1, no extrema).
    # The eligibility gate keeps composite saddle strings out of the flanking
    # pool -- see is_flanking_saddle_candidate for the index-balance argument.
    saddle_clusters = [cl for cl in clusters if is_flanking_saddle_candidate(cl)]
    core_clusters = [cl for cl in clusters if cl.local_euler_index > 0]
    extrema_clusters = [cl for cl in clusters if cl.n_maxima > 0 or cl.n_minima > 0]

    used_cluster_ids: set[str] = set()
    catastrophes: list[EffectiveCatastrophe] = []

    # Phase 1 -- coreless saddle-saddle monkey saddles (chi = -1 + -1 = -2).
    # Claimed before any core search so a tight pair cannot be scattered across
    # two core-centred constellations.
    for a, b in find_saddle_pairs(
        saddle_clusters=saddle_clusters,
        extrema=extrema_clusters,
        cores=core_clusters,
        field_span=field_span,
        pair_max_pitch=pair_max_pitch,
        delta_theta_mesh_rad=delta_theta_mesh_rad,
        barrier_tol=barrier_tol,
        coreless_min_ratio=coreless_min_ratio,
        pair_max_deg_floor=pair_max_deg_floor,
    ):
        cat = build_catastrophe(
            core=None,
            flanking_saddles=[a, b],
            fold_order=3,
            entity_id=f"CAT_{len(catastrophes) + 1}",
            field_span=field_span,
            delta_theta_mesh_rad=delta_theta_mesh_rad,
            sphere_radius=sphere_radius,
            barrier_tol=barrier_tol,
            angular_tol_rad=angular_tol_rad,
            num_boundary_points=num_boundary_points,
        )
        catastrophes.append(cat)
        used_cluster_ids.add(a.cluster_id)
        used_cluster_ids.add(b.cluster_id)

    # Pre-evaluate flanking candidates and sort cores by proximity to their closest saddles.
    # Sorting ensures each core claims its nearest flanking saddles rather than allowing distant
    # cores earlier in the list to reach across the sphere and grab saddles that belong to local
    # constellations.
    core_candidates: list[tuple[float, MicroCluster, list[MicroCluster]]] = []
    for core in core_clusters:
        core_u = np.array(core.centroid_unit, dtype=np.float64)
        flanking: list[tuple[float, MicroCluster]] = []
        for s in saddle_clusters:
            if s.cluster_id == core.cluster_id:
                continue
            s_u = np.array(s.centroid_unit, dtype=np.float64)
            cos_ang = float(np.clip(np.dot(core_u, s_u), -1.0, 1.0))
            ang = math.acos(cos_ang)
            if min_search_rad <= ang <= max_search_rad:
                flanking.append((ang, s))
        flanking.sort(key=lambda x: x[0])
        cand_saddles = [s for _, s in flanking]
        if cand_saddles:
            avg_proximity = sum(a for a, _ in flanking[:3]) / min(len(flanking), 3)
            core_candidates.append((avg_proximity, core, cand_saddles))

    core_candidates.sort(key=lambda x: x[0])

    pair_max_deg = max(
        pair_max_pitch * math.degrees(delta_theta_mesh_rad), pair_max_deg_floor
    )

    # Phase 2 -- core-centred constellations, searched in order of proximity to
    # each core's saddles
    for _, core, candidate_saddles in core_candidates:
        if core.cluster_id in used_cluster_ids:
            continue

        avail_saddles = [
            s for s in candidate_saddles if s.cluster_id not in used_cluster_ids
        ]

        # Check for 4-fold (Octupolar, k=4, target net index = -3) FIRST if >= 4 saddles,
        # then 3-fold (Monkey Saddle, k=3, target net index = -2).
        # Prioritizing the higher symmetry / larger candidate set prevents a 4-fold
        # configuration from being prematurely captured as an incomplete 3-fold set.
        # Within each fold order the TIGHTEST index-balancing combination wins: a
        # catastrophe is a local structure, so the constellation with the smallest
        # angular span is the real one. Taking the first match instead lets a core
        # reach across the sphere and bind distant saddles into one oversized ring.
        found_fold: int | None = None
        selected_flanking: list[MicroCluster] = []

        # Try k = 4
        if len(avail_saddles) >= 4:
            best: tuple[float, list[MicroCluster]] | None = None
            for quad in itertools.combinations(avail_saddles, 4):
                members = list(quad)
                net_idx = core.local_euler_index + sum(
                    s.local_euler_index for s in members
                )
                if net_idx != -3:
                    continue
                if is_bimodal_quad(members, pair_max_deg, bimodal_ratio):
                    continue
                span = constellation_span_deg([core, *members])
                if best is None or span < best[0]:
                    best = (span, members)
            if best is not None:
                found_fold = 4
                selected_flanking = best[1]

        # Try k = 3 if not matched k = 4
        if not found_fold and len(avail_saddles) >= 3:
            best3: tuple[float, list[MicroCluster]] | None = None
            for trio in itertools.combinations(avail_saddles, 3):
                members = list(trio)
                net_idx = core.local_euler_index + sum(
                    s.local_euler_index for s in members
                )
                if net_idx != -2:
                    continue
                span = constellation_span_deg([core, *members])
                if best3 is None or span < best3[0]:
                    best3 = (span, members)
            if best3 is not None:
                found_fold = 3
                selected_flanking = best3[1]

        if found_fold is not None and selected_flanking:
            cat = build_catastrophe(
                core=core,
                flanking_saddles=selected_flanking,
                fold_order=found_fold,
                entity_id=f"CAT_{len(catastrophes) + 1}",
                field_span=field_span,
                delta_theta_mesh_rad=delta_theta_mesh_rad,
                sphere_radius=sphere_radius,
                barrier_tol=barrier_tol,
                angular_tol_rad=angular_tol_rad,
                num_boundary_points=num_boundary_points,
            )
            catastrophes.append(cat)

            # Mark constituent clusters as used
            for cl in cat.constituent_clusters:
                used_cluster_ids.add(cl.cluster_id)

    # Remaining unassociated clusters
    unassociated = [cl for cl in clusters if cl.cluster_id not in used_cluster_ids]

    return catastrophes, unassociated


def is_flanking_saddle_candidate(cluster: MicroCluster) -> bool:
    """Decide whether a micro-cluster may serve as a flanking saddle.

    A k-fold catastrophe is an extremum core (chi = +1) ringed by k *simple*
    saddles, so the constellation index balances as

        chi_core + sum_i chi_saddle_i = 1 + k * (-1) = 1 - k.

    Every flanking member must therefore contribute exactly -1 and must be a
    *pure* saddle set. This is the topological statement of "simple saddle",
    and it is what keeps composite structures out of the flanking pool.

    That distinction matters on corrugated fields such as the modified Willmore
    energy, where a long ridge/valley string carries a net index of -1 (e.g.
    12 Max + 2 Min - 14 Sad) while spanning a large arc. Index arithmetic alone
    cannot tell it apart from a simple saddle, so the string is swallowed as a
    flanking member: the constellation then claims a set that is itself a
    composite saddle, the reported fold order is wrong, and the string is
    stranded from its own effective-saddle reduction.

    Requiring zero extrema plus net index -1 admits exactly the simple-saddle
    clusters and rejects:

    - saddle strings and other composite sets carrying extrema (any n_maxima or
      n_minima > 0),
    - degenerate multi-saddle clusters (chi <= -2, e.g. a multiplicity-2 saddle),
    - topologically neutral extrema/saddle dipoles (chi = 0).

    Args:
        cluster: Candidate micro-cluster from the clustering stage.

    Returns:
        True when the cluster is a pure saddle set of net index -1 and may
        flank a catastrophe core.
    """
    if cluster.n_saddles <= 0:
        return False
    if cluster.n_maxima > 0 or cluster.n_minima > 0:
        return False
    return cluster.local_euler_index == -1
