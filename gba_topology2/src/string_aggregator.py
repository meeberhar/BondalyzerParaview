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
import numpy.typing as npt

from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    MicroCluster,
    compute_tangent_pca,
    generate_elliptical_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "BridgeLink",
    "GroupGeometry",
    "MacroString",
    "aggregate_cluster_strings",
    "cluster_directional_support_radius",
    "compute_bridge_links",
    "compute_group_geometry",
    "is_degenerate_group",
    "select_elongated_core",
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


def cluster_directional_support_radius(
    cluster: MicroCluster,
    target_unit: npt.NDArray[np.float64],
) -> float:
    """Angular radius of a cluster's fitted boundary toward a target direction.

    Evaluates the polar form of the cluster's boundary ellipse,

        r(phi) = sqrt((a cos phi)^2 + (b sin phi)^2),

    at the bearing of ``target_unit`` measured on the tangent plane at the
    cluster centroid, where ``a`` and ``b`` are the cluster's semi-major and
    semi-minor angular radii and ``phi`` is folded to [0, 90] degrees because
    the principal axis is an undirected line.

    Why this exists: the bridging gap test subtracts the extent each cluster
    already sweeps from the centroid separation. Subtracting the scalar
    semi-major does so isotropically, which turns a long thin corridor into a
    disk of radius ``a`` and lets it swallow compact fragments sitting far off
    to the side. Evaluating the boundary in the direction of the neighbour
    keeps the reach elliptical: full length along the axis, narrow width
    sideways. A compact cluster has ``a == b`` and returns ``a`` unchanged, so
    isotropic fragments behave exactly as before.

    Args:
        cluster: The cluster whose boundary is evaluated.
        target_unit: (3,) direction of the neighbour, as a unit vector in the
            centred algorithm frame. Need not be normalised.

    Returns:
        Angular radius in radians toward the target. Falls back to the
        semi-major radius whenever the bearing is undefined (target at the
        centroid or antipodal to it) or the axis is degenerate, which is the
        conservative choice: it never shrinks the reach on ambiguous geometry.
    """
    a = float(cluster.semi_major_rad)
    b = float(cluster.semi_minor_rad)
    if a <= 0.0:
        return 0.0
    if b <= 0.0:
        return a
    if abs(a - b) <= 1e-12:
        return a

    c = np.array(cluster.centroid_unit, dtype=np.float64)
    c_norm = float(np.linalg.norm(c))
    if c_norm < 1e-12:
        return a
    c = c / c_norm

    t = np.asarray(target_unit, dtype=np.float64)
    bearing = t - float(np.dot(t, c)) * c
    bearing_norm = float(np.linalg.norm(bearing))
    if bearing_norm < 1e-12:
        return a

    axis = np.array(cluster.principal_axis_unit, dtype=np.float64)
    axis = axis - float(np.dot(axis, c)) * c
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return a

    cos_phi = abs(float(np.dot(bearing / bearing_norm, axis / axis_norm)))
    cos2 = min(max(cos_phi * cos_phi, 0.0), 1.0)
    sin2 = max(0.0, 1.0 - cos2)
    return math.sqrt((a * a) * cos2 + (b * b) * sin2)


@dataclass(frozen=True)
class BridgeLink:
    """One candidate link between two micro-clusters in the bridging graph.

    Attributes:
        index_a: Index of the first cluster in the input list.
        index_b: Index of the second cluster in the input list.
        cluster_a: Id of the first cluster.
        cluster_b: Id of the second cluster.
        gap_rad: Uncovered tip-to-tip gap between the two fitted boundaries.
        axis_angle_deg: Angle between the two principal axes, folded to
            [0, 90] degrees, or None when the test does not apply (either
            cluster is compact, so its axis carries no meaningful direction).
        is_linked: True when the pair is joined in the bridging graph.
        reason: Short human-readable explanation of the verdict.
    """

    index_a: int
    index_b: int
    cluster_a: str
    cluster_b: str
    gap_rad: float
    axis_angle_deg: float | None
    is_linked: bool
    reason: str


def _axis_pairwise_angle(c1: MicroCluster, c2: MicroCluster) -> float | None:
    """Angle between two principal axes, folded to [0, 90] degrees.

    Args:
        c1: First cluster.
        c2: Second cluster.

    Returns:
        The angle in degrees, or None when the test does not apply: either
        cluster is compact, in which case its axis is noise (a near-isotropic
        point cloud has no meaningful major direction) and a compact fragment
        can extend a corridor from any direction.
    """
    if c1.morphology != "string" or c2.morphology != "string":
        return None
    a = np.array(c1.principal_axis_unit, dtype=np.float64)
    b = np.array(c2.principal_axis_unit, dtype=np.float64)
    if float(np.linalg.norm(a)) < 1e-9 or float(np.linalg.norm(b)) < 1e-9:
        return None
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cos_v = abs(float(np.clip(float(np.dot(a, b)), -1.0, 1.0)))
    return math.degrees(math.acos(cos_v))


def compute_bridge_links(
    clusters: list[MicroCluster],
    delta_theta_mesh_rad: float,
    k_bridge: float,
    bridge_all: bool = True,
    axis_angle_tol_deg: float = 30.0,
) -> list[BridgeLink]:
    """Evaluate every candidate link between micro-clusters.

    A pair is linked when the **uncovered gap** between the two fitted
    boundaries is at most ``k_bridge`` delta_theta units *and* their principal
    axes are within ``axis_angle_tol_deg`` of each other. The gap is measured
    tip-to-tip using each cluster's directional support radius (see
    :func:`cluster_directional_support_radius`), not the scalar semi-major, so
    a long thin corridor reaches far along its axis and only narrowly to the
    side.

    The function is pure and side-effect free: it reports the verdict and the
    reason for every pair, which is what makes the bridging decision
    auditable. The aggregator consumes the ``is_linked`` flags to build its
    graph; the diagnostic runner consumes the whole list to explain *why* a
    given configuration bridges or does not.

    Args:
        clusters: Micro-clusters to consider.
        delta_theta_mesh_rad: Mesh pitch in radians.
        k_bridge: Maximum uncovered gap, in delta_theta units.
        bridge_all: Use the gap + axis-collinearity rule (True) or the legacy
            centroid-separation rule (False).
        axis_angle_tol_deg: Maximum principal-axis difference, in degrees, for
            two clusters to count as collinear.

    Returns:
        One :class:`BridgeLink` per unordered pair, in ``(i, j)`` index order.

    Raises:
        ValueError: If delta_theta_mesh_rad <= 0 or k_bridge <= 0.
    """
    if delta_theta_mesh_rad <= 0:
        raise ValueError(
            f"delta_theta_mesh_rad must be positive, got {delta_theta_mesh_rad}"
        )
    if k_bridge <= 0:
        raise ValueError(f"k_bridge must be positive, got {k_bridge}")

    bridge_threshold_rad = k_bridge * delta_theta_mesh_rad
    axis_tol_rad = math.radians(axis_angle_tol_deg)
    n = len(clusters)
    cluster_units = np.array([c.centroid_unit for c in clusters], dtype=np.float64)

    links: list[BridgeLink] = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = float(
                np.clip(np.dot(cluster_units[i], cluster_units[j]), -1.0, 1.0)
            )
            ang_ij = math.acos(cos_ij)
            axis_ang: float | None = None
            if bridge_all:
                reach_i = cluster_directional_support_radius(
                    clusters[i], cluster_units[j]
                )
                reach_j = cluster_directional_support_radius(
                    clusters[j], cluster_units[i]
                )
                gap_rad = ang_ij - reach_i - reach_j
                close = gap_rad <= bridge_threshold_rad
                axis_ang = _axis_pairwise_angle(clusters[i], clusters[j])
                collinear = axis_ang is None or math.radians(axis_ang) <= axis_tol_rad
                linked = close and collinear
                if linked:
                    reason = "linked"
                elif not close:
                    reason = (
                        f"gap {math.degrees(gap_rad):.2f} deg exceeds threshold "
                        f"{math.degrees(bridge_threshold_rad):.2f} deg"
                    )
                else:
                    reason = (
                        f"axis angle {axis_ang:.2f} deg exceeds tolerance "
                        f"{axis_angle_tol_deg:.2f} deg"
                    )
            else:
                gap_rad = ang_ij
                linked = ang_ij <= bridge_threshold_rad and (
                    clusters[i].morphology == "string"
                    or clusters[j].morphology == "string"
                )
                reason = "linked" if linked else "legacy rule: not linked"
            links.append(
                BridgeLink(
                    index_a=i,
                    index_b=j,
                    cluster_a=clusters[i].cluster_id,
                    cluster_b=clusters[j].cluster_id,
                    gap_rad=float(gap_rad),
                    axis_angle_deg=axis_ang,
                    is_linked=bool(linked),
                    reason=reason,
                )
            )
    return links


@dataclass(frozen=True)
class GroupGeometry:
    """Pooled geometry of a set of micro-clusters treated as one group.

    Attributes:
        aspect_ratio: Tangent-plane anisotropy (lambda_1 / lambda_2) of the
            pooled member set, the quantity the elongation gate tests.
        centroid_unit: (3,) unit direction of the pooled centroid.
        centroid: (3,) pooled centroid on the sphere.
        principal_axis_unit: (3,) unit principal (major) axis.
        max_major_rad: Half-extent along the major axis, in radians.
        max_minor_rad: Half-extent along the minor axis, in radians.
        members: De-duplicated member critical points, in cluster order.
    """

    aspect_ratio: float
    centroid_unit: tuple[float, float, float]
    centroid: tuple[float, float, float]
    principal_axis_unit: tuple[float, float, float]
    max_major_rad: float
    max_minor_rad: float
    members: list[CriticalPoint]


def compute_group_geometry(
    clusters: list[MicroCluster],
    sphere_radius: float = 1.0,
) -> GroupGeometry:
    """Pool the geometry of a set of micro-clusters into a single group.

    Every member critical point of every cluster is projected onto the unit
    sphere; the group's centroid is the normalised mean of those directions and
    the anisotropy is the tangent-plane PCA of the member set about that
    centroid.

    Args:
        clusters: Clusters to pool. An empty list yields a degenerate,
            zero-extent geometry.
        sphere_radius: Sphere radius R, used to place the centroid.

    Returns:
        The pooled :class:`GroupGeometry`.
    """
    all_cps: list[CriticalPoint] = []
    seen_cp_ids: set[int] = set()
    for cl in clusters:
        for m in cl.members:
            if m.vertex_id not in seen_cp_ids:
                seen_cp_ids.add(m.vertex_id)
                all_cps.append(m)

    if not all_cps:
        return GroupGeometry(
            aspect_ratio=1.0,
            centroid_unit=(0.0, 0.0, 1.0),
            centroid=(0.0, 0.0, sphere_radius),
            principal_axis_unit=(1.0, 0.0, 0.0),
            max_major_rad=0.0,
            max_minor_rad=0.0,
            members=[],
        )

    all_positions = np.array([cp.position for cp in all_cps], dtype=np.float64)
    all_norms = np.linalg.norm(all_positions, axis=1, keepdims=True)
    all_norms = np.where(all_norms < 1e-12, 1.0, all_norms)
    all_units = all_positions / all_norms

    mean_unit = np.mean(all_units, axis=0)
    norm_mean = float(np.linalg.norm(mean_unit))
    if norm_mean > 1e-8:
        c_unit = mean_unit / norm_mean
        centroid = c_unit * sphere_radius
    else:
        c_unit = all_units[0]
        centroid = all_positions[0]

    ar, princ_axis, max_major, max_minor = compute_tangent_pca(all_units, c_unit)

    return GroupGeometry(
        aspect_ratio=float(ar),
        centroid_unit=(float(c_unit[0]), float(c_unit[1]), float(c_unit[2])),
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        principal_axis_unit=(
            float(princ_axis[0]),
            float(princ_axis[1]),
            float(princ_axis[2]),
        ),
        max_major_rad=float(max_major),
        max_minor_rad=float(max_minor),
        members=all_cps,
    )


def is_degenerate_group(geometry: GroupGeometry) -> bool:
    """Report whether a group's shape is determined by its member set.

    The tangent-plane PCA of a member set is only meaningful once the set
    spans the plane. With fewer than three points the covariance is rank
    deficient: two points give a zero minor eigenvalue, so the aspect ratio
    is a division-by-zero artefact (astronomically large) rather than a
    measurement. Such a group is *round* in the only sense the gate cares
    about, so it must not be treated as an elongated corridor.

    Args:
        geometry: The pooled group geometry.

    Returns:
        True when the group's shape is degenerate.
    """
    return len(geometry.members) < 3


def select_elongated_core(
    component: list[MicroCluster],
    aspect_ratio_threshold: float,
    sphere_radius: float = 1.0,
    adjacency: dict[str, set[str]] | None = None,
) -> tuple[list[MicroCluster], list[MicroCluster]]:
    """Split a bridged component into an elongated core and rejected satellites.

    The elongation gate is a *pooled* property, so it is not decomposable: a
    single off-axis satellite that joins a valid corridor widens the pooled
    minor axis enough to push the group below ``aspect_ratio_threshold``.
    Gating the whole component then annihilates the corridor as well, which
    makes the response to ``k_bridge`` non-monotonic -- widening the bridge
    radius destroys structure that a narrower radius had already resolved.

    This function makes the gate *decomposable*. When the component as a whole
    is not elongated enough, the member whose removal most restores the
    anisotropy is peeled off as a satellite, and the test is repeated on the
    remainder. The surviving core is the largest sub-corridor that still
    qualifies; the peeled members are handed back so the caller can keep them
    as standalone clusters.

    A peel is only accepted when the remainder stays **connected** in the
    bridging graph. Without that requirement the routine could split a
    genuine corridor in half, promoting two halves that each look elongated
    while destroying the corridor that spanned them. Connectivity is checked
    against ``adjacency`` when it is supplied; without it the peel is accepted
    unconditionally, which is the correct fallback for a component that is a
    single pair.

    Args:
        component: Clusters linked into one component of the bridging graph.
        aspect_ratio_threshold: Minimum pooled aspect ratio for a group to
            qualify as a corridor.
        sphere_radius: Sphere radius R.
        adjacency: Optional map of cluster id to the ids it links to, used to
            keep the surviving core connected.

    Returns:
        Tuple of (core, satellites). ``core`` is empty when no sub-group
        qualifies, in which case ``satellites`` is the whole component.
    """
    if len(component) <= 1:
        # A lone cluster needs no peeling: it is either an intrinsically
        # elongated string (promoted by the caller's own check) or a compact
        # fragment (rejected by it). Either way the decision belongs to the
        # caller, so the component passes through untouched.
        return list(component), []

    keep = list(component)
    satellites: list[MicroCluster] = []

    while len(keep) >= 2:
        pooled = compute_group_geometry(keep, sphere_radius)
        if (
            not is_degenerate_group(pooled)
            and pooled.aspect_ratio >= aspect_ratio_threshold
        ):
            return keep, satellites

        best: tuple[float, MicroCluster] | None = None
        for cand in keep:
            trial = [c for c in keep if c.cluster_id != cand.cluster_id]
            if not trial:
                continue
            if adjacency is not None and not _is_connected(
                [c.cluster_id for c in trial], adjacency
            ):
                continue
            trial_geom = compute_group_geometry(trial, sphere_radius)
            # A degenerate trial has no meaningful anisotropy, so it cannot be
            # preferred on the strength of a division-by-zero ratio.
            if is_degenerate_group(trial_geom):
                continue
            ar = trial_geom.aspect_ratio
            if best is None or ar > best[0]:
                best = (ar, cand)
        if best is None:
            break
        satellites.append(best[1])
        keep = [c for c in keep if c.cluster_id != best[1].cluster_id]

    # Nothing in the component qualifies: the whole component is rejected and
    # its members return as standalone clusters, exactly as before.
    return [], list(component)


def _is_connected(node_ids: list[str], adjacency: dict[str, set[str]]) -> bool:
    """Report whether a set of nodes forms one connected subgraph.

    Args:
        node_ids: The node ids to test.
        adjacency: Map of node id to the ids it links to.

    Returns:
        True when every node is reachable from every other one. A single node
        is trivially connected.
    """
    if len(node_ids) <= 1:
        return True
    members = set(node_ids)
    seen = {node_ids[0]}
    stack = [node_ids[0]]
    while stack:
        curr = stack.pop()
        for nbr in adjacency.get(curr, ()):
            if nbr in members and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(members)


def aggregate_cluster_strings(
    clusters: list[MicroCluster],
    delta_theta_mesh_rad: float,
    sphere_radius: float = 1.0,
    k_bridge: float = 3.0,
    aspect_ratio_threshold: float = 2.0,
    min_boundary_margin_deg: float = 1.5,
    num_boundary_points: int = 48,
    bridge_all: bool = False,
    axis_angle_tol_deg: float = 30.0,
) -> tuple[list[MacroString], list[MicroCluster]]:
    """Group collinear or intrinsically elongated micro-clusters into macro-strings.

    Clusters that do not form or belong to a string are returned in the remainder list
    as isolated compact clusters.

    Two modes are offered:

    - ``bridge_all=False`` (default, legacy): only clusters already classified as
      strings seed a chain, and a neighbour joins only if it is itself a string.
    - ``bridge_all=True``: every cluster seeds a chain. Two clusters link when
      the **uncovered gap** between their ends is at most ``k_bridge``
      delta_theta units *and* their principal axes are within
      ``axis_angle_tol_deg`` of each other. Qualification then moves to the
      pooled group: a multi-cluster group becomes a macro-string only when its
      own tangent-plane aspect ratio reaches ``aspect_ratio_threshold``, which
      is what keeps isotropic blobs of critical points from being fused. A lone
      compact cluster is never promoted.

      Measuring the gap rather than centroid separation matters: for elongated
      clusters the centroid distance is dominated by the two half-lengths, so
      two collinear fragments sitting tip-to-tip look far apart, while two long
      strings crossing at 60 degrees look close. On the modified Willmore energy
      the fragments of one corridor sit 0.9 delta_theta apart with axes 12
      degrees out, whereas crossing strings overlap (negative gap) with axes 60
      degrees out. The gap test alone would merge the crossings; the axis test
      alone would not separate the collinear pair. Both are required.

      Each cluster's contribution to the gap is its boundary radius **in the
      direction of the neighbour** (see :func:`cluster_directional_support_radius`)
      rather than its scalar semi-major radius. Subtracting the semi-major
      isotropically grants a long thin corridor a reach equal to its half-length
      in every direction, so a compact extremum sitting well off to the side --
      90 degrees from the corridor axis, outside the fitted ellipse -- is
      absorbed and flips the corridor's net index. On the curvedness field of
      Pd_20K at k_pitch=1.6 a lone minimum was absorbed into the 21-CP saddle
      corridor MC18 this way, cancelling chi = -1 to chi = 0 and annihilating a
      genuine saddle. The directional radius keeps the reach elliptical: full
      length along the axis, narrow width sideways. Compact clusters have a == b
      and are unaffected.

    Args:
        clusters: Input list of MicroCluster objects from compute_micro_clusters.
        delta_theta_mesh_rad: Mesh pitch in radians.
        sphere_radius: Radius of the sphere R.
        k_bridge: Maximum centroid separation (in delta_theta units) to bridge
            adjacent collinear clusters into a string (default 3.0). In
            ``bridge_all`` mode this bounds the directional end-to-end gap
            instead.
        aspect_ratio_threshold: Minimum aspect ratio for a group to qualify
            as a string (default 2.0).
        min_boundary_margin_deg: Angular margin added to boundary ring (default 1.5 deg).
        num_boundary_points: Number of perimeter samples for the elliptical ring.
        bridge_all: Bridge compact fragments too, qualifying groups on their
            pooled aspect ratio (default False).
        axis_angle_tol_deg: Maximum angle (degrees) between two clusters'
            principal axes for them to count as collinear in ``bridge_all``
            mode (default 30.0). Clusters with a degenerate axis (compact
            morphology) skip this test, since a point can extend a corridor
            from any direction.

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

    min_margin_rad = math.radians(min_boundary_margin_deg)

    # Separate candidates: clusters with morphology == "string" are immediate candidates.
    # Also nearby clusters within bridge_threshold_rad that together form an elongated chain.
    n = len(clusters)

    # Adjacency graph between clusters. The link evaluation lives in
    # compute_bridge_links so the diagnostic runner can report the verdict and
    # the reason for every pair without re-implementing the rule.
    links = compute_bridge_links(
        clusters=clusters,
        delta_theta_mesh_rad=delta_theta_mesh_rad,
        k_bridge=k_bridge,
        bridge_all=bridge_all,
        axis_angle_tol_deg=axis_angle_tol_deg,
    )
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for link in links:
        if link.is_linked:
            adj[link.index_a].append(link.index_b)
            adj[link.index_b].append(link.index_a)

    # Traverse connected components of string-linked clusters
    visited: set[int] = set()
    string_components: list[list[int]] = []

    # Seed from intrinsically elongated clusters, or from every cluster when
    # compact fragments are eligible for bridging.
    for i in range(n):
        if (bridge_all or clusters[i].morphology == "string") and i not in visited:
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
    deferred: list[int] = []
    for comp in string_components:
        comp_clusters = [clusters[idx] for idx in comp]

        # The elongation gate is a pooled property and therefore not
        # decomposable: one off-axis satellite joining a valid corridor widens
        # the pooled minor axis enough to sink the whole group below the
        # threshold, which would annihilate the corridor too. When the
        # component as a whole fails, peel off the member whose removal most
        # restores the anisotropy and retest the remainder, so a valid
        # sub-corridor survives instead of the whole component being deferred.
        comp_adjacency: dict[str, set[str]] = {
            clusters[idx].cluster_id: {
                clusters[nbr].cluster_id for nbr in adj[idx] if nbr in comp
            }
            for idx in comp
        }
        core_clusters, satellites = select_elongated_core(
            component=comp_clusters,
            aspect_ratio_threshold=aspect_ratio_threshold,
            sphere_radius=sphere_radius,
            adjacency=comp_adjacency,
        )

        if not core_clusters:
            # No sub-group qualifies: every member returns as a standalone
            # cluster, exactly as the legacy all-or-nothing gate did.
            deferred.extend(comp)
            continue

        if satellites:
            # The peeled satellites are not part of the corridor. They return
            # as standalone clusters, so their index is preserved and they are
            # still eligible for standalone reduction.
            deferred.extend(
                idx
                for idx in comp
                if clusters[idx].cluster_id in {s.cluster_id for s in satellites}
            )

        geom = compute_group_geometry(core_clusters, sphere_radius)
        all_cps = geom.members
        c_unit = geom.centroid_unit
        centroid = geom.centroid
        ar = geom.aspect_ratio
        princ_axis = geom.principal_axis_unit
        max_major = geom.max_major_rad
        max_minor = geom.max_minor_rad

        # Qualification: a bridged group must be genuinely elongated, so pooled
        # anisotropy is the gate. A single cluster is promoted only when the
        # clustering stage already labelled it a string, which preserves the
        # legacy behaviour and stops isotropic blobs from being fused. A group
        # whose shape is degenerate (fewer than three points) has no meaningful
        # anisotropy, so it never qualifies on the pooled ratio.
        is_elongated_group = (
            len(core_clusters) > 1
            and not is_degenerate_group(geom)
            and ar >= aspect_ratio_threshold
        )
        is_intrinsic_string = (
            len(core_clusters) == 1 and core_clusters[0].morphology == "string"
        )
        if not (is_elongated_group or is_intrinsic_string):
            deferred.extend(comp)
            continue

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
                string_id=f"MS{len(macro_strings) + 1}",
                constituent_clusters=list(core_clusters),
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

    # Clusters that never joined a string, plus groups that failed the
    # elongation test, remain available as standalone reduction units.
    rejected = set(deferred)
    isolated_clusters = [
        clusters[i] for i in range(n) if i not in visited or i in rejected
    ]

    return macro_strings, isolated_clusters
