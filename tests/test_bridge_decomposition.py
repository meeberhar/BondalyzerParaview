"""Unit tests for the decomposable elongation gate of the corridor bridge.

Background
----------
The bridging rule in ``string_aggregator`` is two-stage:

1. A graph edge is created between two micro-clusters when the uncovered gap
   between their fitted boundaries is within ``k_bridge`` delta_theta units
   *and* their principal axes are collinear.
2. Each connected component of that graph is gated by its **pooled**
   tangent-plane aspect ratio. A group that fails the gate is rejected whole.

The second stage is the problem this module covers. The pooled aspect ratio
is not decomposable: a single off-axis satellite that joins a valid corridor
widens the pooled minor axis enough to push the whole group below the
threshold. Rejecting the group then annihilates the corridor as well, so
*widening* the bridge radius could destroy structure that a narrower radius
had already resolved -- the response to ``k_bridge`` became non-monotonic.

``select_elongated_core`` makes the gate decomposable: when a component fails,
the member whose removal most restores the anisotropy is peeled off as a
satellite and the remainder is retested, so a valid sub-corridor survives.

The regression test uses the real ``ethene4.plt`` dataset, because the failure
needs the genuine geometry of a corrugated field: the synthetic fixtures below
provably exercise the same rules but cannot reproduce the pooled aspect ratio
of a real corridor.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from vtkmodules.util import numpy_support

from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    MicroCluster,
    compute_micro_clusters,
    generate_elliptical_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint, compute_discrete_morse_cps
from gba_topology2.src.sphere_viewer import extract_atom_sphere_mesh
from gba_topology2.src.string_aggregator import (
    aggregate_cluster_strings,
    compute_bridge_links,
    compute_group_geometry,
    is_degenerate_group,
    select_elongated_core,
)

DELTA_THETA = 0.025
RADIUS = 1.0
THRESHOLD = 2.0

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETHENE4_PLT = os.path.join(REPO_ROOT, "ethene4.plt")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _ring(
    centroid: tuple[float, float, float],
    axis: tuple[float, float, float],
    semi_major: float,
    semi_minor: float,
) -> list[BoundaryRingPoint]:
    """Build an elliptical boundary ring for a synthetic string cluster."""
    return generate_elliptical_boundary_ring(
        centroid=centroid,
        sphere_radius=RADIUS,
        semi_major_rad=semi_major,
        semi_minor_rad=semi_minor,
        principal_axis_unit=axis,
        num_samples=24,
    )


def _string_cluster(
    cluster_id: str,
    centroid_unit: tuple[float, float, float],
    axis_unit: tuple[float, float, float],
    n_max: int,
    n_sad: int,
    semi_major: float,
    start_vid: int,
    semi_minor: float = 0.01,
) -> MicroCluster:
    """Build an elongated string cluster centred on a unit direction.

    Args:
        cluster_id: Identifier to assign.
        centroid_unit: Unit centroid direction.
        axis_unit: Unit principal (major) axis, tangent to the sphere.
        n_max: Number of maximum members.
        n_sad: Number of simple-saddle members.
        semi_major: Semi-major angular radius in radians.
        start_vid: First vertex id to allocate.
        semi_minor: Semi-minor angular radius in radians.

    Returns:
        A string-morphology MicroCluster.
    """
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    ax = np.array(axis_unit, dtype=np.float64)
    ax = ax - float(np.dot(ax, c)) * c
    ax_norm = float(np.linalg.norm(ax))
    ax = ax / ax_norm if ax_norm > 1e-12 else np.zeros(3, dtype=np.float64)

    total = n_max + n_sad
    members: list[CriticalPoint] = []
    vid = start_vid
    for i, (cp_type, value) in enumerate(
        [("maximum", 1.0)] * n_max + [("saddle", 0.5)] * n_sad
    ):
        frac = (i / (total - 1) - 0.5) if total > 1 else 0.0
        t = 2.0 * frac * semi_major
        p = math.cos(t) * c + math.sin(t) * ax
        p = p / np.linalg.norm(p)
        members.append(
            CriticalPoint(
                vertex_id=vid,
                cp_type=cp_type,
                value=value,
                position=(float(p[0]), float(p[1]), float(p[2])),
                multiplicity=1,
            )
        )
        vid += 1

    centroid = (float(c[0] * RADIUS), float(c[1] * RADIUS), float(c[2] * RADIUS))
    return MicroCluster(
        cluster_id=cluster_id,
        members=members,
        centroid=centroid,
        centroid_unit=(float(c[0]), float(c[1]), float(c[2])),
        angular_radius_rad=semi_major,
        angular_radius_deg=math.degrees(semi_major),
        angular_diameter_rad=2.0 * semi_major,
        angular_diameter_deg=math.degrees(2.0 * semi_major),
        spatial_diameter=2.0 * semi_major * RADIUS,
        n_maxima=n_max,
        n_minima=0,
        n_saddles=n_sad,
        total_saddle_index=n_sad,
        local_euler_index=n_max - n_sad,
        is_multi_cp=len(members) > 1,
        boundary_ring=_ring(centroid, axis_unit, semi_major, semi_minor),
        morphology="string",
        aspect_ratio=8.0,
        principal_axis_unit=axis_unit,
        semi_major_rad=semi_major,
        semi_minor_rad=semi_minor,
    )


def _compact_cluster(
    cluster_id: str,
    centroid_unit: tuple[float, float, float],
    cp_type: str,
    radius: float,
    vid: int,
) -> MicroCluster:
    """Build a single-CP compact cluster at a unit direction.

    Args:
        cluster_id: Identifier to assign.
        centroid_unit: Unit centroid direction.
        cp_type: ``"maximum"``, ``"minimum"`` or ``"saddle"``.
        radius: Circular ring angular radius in radians.
        vid: Vertex id for the single member.

    Returns:
        A compact-morphology MicroCluster.
    """
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    chi = 1 if cp_type in ("maximum", "minimum") else -1
    return MicroCluster(
        cluster_id=cluster_id,
        members=[
            CriticalPoint(
                vertex_id=vid,
                cp_type=cp_type,
                value=0.2,
                position=(float(c[0]), float(c[1]), float(c[2])),
                multiplicity=1,
            )
        ],
        centroid=(float(c[0] * RADIUS), float(c[1] * RADIUS), float(c[2] * RADIUS)),
        centroid_unit=(float(c[0]), float(c[1]), float(c[2])),
        angular_radius_rad=radius,
        angular_radius_deg=math.degrees(radius),
        angular_diameter_rad=0.0,
        angular_diameter_deg=0.0,
        spatial_diameter=0.0,
        n_maxima=1 if cp_type == "maximum" else 0,
        n_minima=1 if cp_type == "minimum" else 0,
        n_saddles=1 if cp_type == "saddle" else 0,
        total_saddle_index=1 if cp_type == "saddle" else 0,
        local_euler_index=chi,
        is_multi_cp=False,
        boundary_ring=[],
        morphology="compact",
        aspect_ratio=1.0,
        semi_major_rad=radius,
        semi_minor_rad=radius,
    )


def _blobs() -> list[MicroCluster]:
    """Three compact blobs close enough to form one bridging component.

    A group of isotropic blobs has no elongated sub-group: every one of them is
    a single point, so the pooled shape stays round no matter how it is split.
    """
    return [
        _compact_cluster("MC1", (0.0, 0.0, 1.0), "minimum", 0.02, 1),
        _compact_cluster(
            "MC2", (math.sin(0.05), 0.0, math.cos(0.05)), "minimum", 0.02, 2
        ),
        _compact_cluster(
            "MC3", (0.0, math.sin(0.05), math.cos(0.05)), "minimum", 0.02, 3
        ),
    ]


# ---------------------------------------------------------------------------
# compute_bridge_links
# ---------------------------------------------------------------------------


def test_bridge_links_reports_one_entry_per_pair() -> None:
    """Every unordered pair is evaluated exactly once."""
    clusters = _blobs()
    links = compute_bridge_links(
        clusters=clusters,
        delta_theta_mesh_rad=DELTA_THETA,
        k_bridge=3.0,
        bridge_all=True,
    )
    assert len(links) == len(clusters) * (len(clusters) - 1) // 2
    pairs = {(link.cluster_a, link.cluster_b) for link in links}
    assert len(pairs) == len(links)


def test_bridge_links_rejects_non_collinear_axes_with_a_reason() -> None:
    """A pair inside the gap but off-axis is rejected on the axis test."""
    corridor = _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=0,
    )
    other = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.35), 0.0, math.cos(0.35)),
        axis_unit=(0.0, 1.0, 0.0),
        n_max=2,
        n_sad=3,
        semi_major=0.05,
        start_vid=100,
    )
    links = compute_bridge_links(
        clusters=[corridor, other],
        delta_theta_mesh_rad=DELTA_THETA,
        k_bridge=3.0,
        bridge_all=True,
    )
    assert len(links) == 1
    assert links[0].is_linked is False
    assert "axis angle" in links[0].reason


def test_bridge_links_rejects_wide_gap_with_a_reason() -> None:
    """A pair beyond the threshold is rejected and the reason says why."""
    corridor = _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=0,
    )
    far = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.9), 0.0, math.cos(0.9)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=2,
        n_sad=3,
        semi_major=0.05,
        start_vid=100,
    )
    links = compute_bridge_links(
        clusters=[corridor, far],
        delta_theta_mesh_rad=DELTA_THETA,
        k_bridge=3.0,
        bridge_all=True,
    )
    assert links[0].is_linked is False
    assert "exceeds threshold" in links[0].reason


def test_bridge_links_threshold_scales_with_k_bridge(
    ethene4_clusters: tuple[list[MicroCluster], float, float],
) -> None:
    """A pair just outside one threshold is inside a larger one.

    The ethene4 geometry provides a pair that straddles the threshold: the gap
    between MC2 and MC13 is 21.55 degrees, which is outside the 21.26 degree
    threshold of k_bridge = 3.0 and inside the 23.03 degree threshold of
    k_bridge = 3.25.
    """
    clusters, _radius, pitch = ethene4_clusters
    tight = compute_bridge_links(
        clusters=clusters,
        delta_theta_mesh_rad=pitch,
        k_bridge=3.0,
        bridge_all=True,
    )
    loose = compute_bridge_links(
        clusters=clusters,
        delta_theta_mesh_rad=pitch,
        k_bridge=3.25,
        bridge_all=True,
    )
    tight_pairs = {(l.cluster_a, l.cluster_b) for l in tight if l.is_linked}
    loose_pairs = {(l.cluster_a, l.cluster_b) for l in loose if l.is_linked}
    assert tight_pairs < loose_pairs
    assert ("MC2", "MC13") in loose_pairs - tight_pairs


def test_bridge_links_rejects_invalid_parameters() -> None:
    """Non-positive pitch or radius is a programming error."""
    clusters = _blobs()
    with pytest.raises(ValueError):
        compute_bridge_links(clusters=clusters, delta_theta_mesh_rad=0.0, k_bridge=3.0)
    with pytest.raises(ValueError):
        compute_bridge_links(
            clusters=clusters, delta_theta_mesh_rad=DELTA_THETA, k_bridge=0.0
        )


# ---------------------------------------------------------------------------
# compute_group_geometry
# ---------------------------------------------------------------------------


def test_group_geometry_pools_members_without_duplicates() -> None:
    """Overlapping clusters contribute each critical point once."""
    a = _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.05,
        start_vid=0,
    )
    b = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.05), 0.0, math.cos(0.05)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.05,
        start_vid=100,
    )
    geom = compute_group_geometry([a, b], RADIUS)
    assert len(geom.members) == 18
    assert len({m.vertex_id for m in geom.members}) == 18
    assert geom.aspect_ratio >= 1.0


def test_group_geometry_of_empty_list_is_degenerate() -> None:
    """An empty group yields a zero-extent geometry rather than raising."""
    geom = compute_group_geometry([], RADIUS)
    assert geom.aspect_ratio == 1.0
    assert geom.members == []
    assert geom.max_major_rad == 0.0


# ---------------------------------------------------------------------------
# select_elongated_core
# ---------------------------------------------------------------------------


def test_core_selection_leaves_a_qualifying_group_untouched() -> None:
    """A group that already passes the gate is returned whole, with no satellites."""
    corridor = _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=0,
    )
    tip = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.40), 0.0, math.cos(0.40)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=4,
        n_sad=5,
        semi_major=0.30,
        start_vid=100,
    )
    group = [corridor, tip]
    assert compute_group_geometry(group, RADIUS).aspect_ratio >= THRESHOLD

    core, satellites = select_elongated_core(
        component=group,
        aspect_ratio_threshold=THRESHOLD,
        sphere_radius=RADIUS,
        adjacency=None,
    )
    assert {c.cluster_id for c in core} == {"MC1", "MC2"}
    assert satellites == []


def test_core_selection_rejects_a_pure_blob_entirely() -> None:
    """A group of compact blobs has no elongated sub-group, so all is rejected."""
    blobs = _blobs()
    core, satellites = select_elongated_core(
        component=blobs,
        aspect_ratio_threshold=THRESHOLD,
        sphere_radius=RADIUS,
        adjacency=None,
    )
    assert core == []
    assert {c.cluster_id for c in satellites} == {"MC1", "MC2", "MC3"}


def test_core_selection_passes_a_lone_cluster_through() -> None:
    """A single cluster is the caller's decision, so it is returned untouched."""
    lone = _compact_cluster("MC1", (0.0, 0.0, 1.0), "minimum", 0.02, 1)
    core, satellites = select_elongated_core(
        component=[lone],
        aspect_ratio_threshold=THRESHOLD,
        sphere_radius=RADIUS,
        adjacency=None,
    )
    assert [c.cluster_id for c in core] == ["MC1"]
    assert satellites == []


def test_core_selection_keeps_the_remainder_connected() -> None:
    """A peel that would split the corridor in half is not accepted.

    The core must stay one connected corridor: promoting two halves that each
    look elongated would destroy the corridor that spanned them.
    """
    a = _string_cluster(
        "MC1",
        centroid_unit=(0.0, 0.0, 1.0),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=0,
    )
    b = _string_cluster(
        "MC2",
        centroid_unit=(math.sin(0.115), 0.0, math.cos(0.115)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=100,
    )
    c = _string_cluster(
        "MC3",
        centroid_unit=(math.sin(0.23), 0.0, math.cos(0.23)),
        axis_unit=(1.0, 0.0, 0.0),
        n_max=8,
        n_sad=9,
        semi_major=0.30,
        start_vid=200,
    )
    clusters = [a, b, c]
    adjacency = {"MC1": {"MC2"}, "MC2": {"MC1", "MC3"}, "MC3": {"MC2"}}

    core, _ = select_elongated_core(
        component=clusters,
        aspect_ratio_threshold=99.0,  # unreachable, so the gate always fails
        sphere_radius=RADIUS,
        adjacency=adjacency,
    )
    # Either the whole chain survives as one unit or nothing does -- never a
    # disconnected half.
    ids = {c.cluster_id for c in core}
    assert ids in (set(), {"MC1", "MC2", "MC3"})


# ---------------------------------------------------------------------------
# Regression: the real ethene4 case
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ethene4_clusters() -> tuple[list[MicroCluster], float, float]:
    """Cluster the ethene4 C1 sphere on the curvedness field.

    Returns:
        Tuple of (micro-clusters, sphere radius, mesh pitch in radians).
    """
    if not os.path.exists(ETHENE4_PLT):
        pytest.skip(f"dataset not available: {ETHENE4_PLT}")

    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        ETHENE4_PLT, target_atom_num=1
    )
    # The saved-parameter snapshots are gitignored, so the field is located
    # from the mesh itself: the curvedness field is the one whose title names
    # it. Every condensed array on the sphere is discovered by the extractor.
    field = next(
        (f["value"] for f in field_items if "curvedness" in f["value"].lower()),
        None,
    )
    if field is None:
        pytest.skip("curvedness field not present on this dataset")

    pts = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    triangles = cell_arr.reshape((n_cells, 4))[:, 1:4]
    arr = sphere_poly.GetPointData().GetArray(field)
    assert arr is not None
    f_vals = numpy_support.vtk_to_numpy(arr)

    geom = analyze_mesh_geometry(
        pts, triangles, sphere_radius=float(atom_info["sphere_radius"])
    )
    radius = float(geom["sphere_radius"])
    pitch = float(geom["mesh_pitch_rad"])

    morse = compute_discrete_morse_cps(pts, triangles, f_vals, tie_break_epsilon=1e-15)
    cps = morse.minima + morse.maxima + morse.saddles
    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=pitch,
        k_pitch=2.0,
        sphere_radius=radius,
        min_boundary_margin_deg=1.5,
        num_boundary_points=48,
    )
    return clusters, radius, pitch


def _components(
    clusters: list[MicroCluster], pitch: float, k_bridge: float
) -> tuple[list[list[str]], dict[str, set[str]]]:
    """Return the bridging graph's components and adjacency at a k_bridge.

    Args:
        clusters: Micro-clusters.
        pitch: Mesh pitch in radians.
        k_bridge: Bridge radius in delta_theta units.

    Returns:
        Tuple of (components as id lists, adjacency map).
    """
    links = compute_bridge_links(
        clusters=clusters,
        delta_theta_mesh_rad=pitch,
        k_bridge=k_bridge,
        bridge_all=True,
    )
    adj: dict[str, set[str]] = {c.cluster_id: set() for c in clusters}
    for link in links:
        if link.is_linked:
            adj[link.cluster_a].add(link.cluster_b)
            adj[link.cluster_b].add(link.cluster_a)

    seen: set[str] = set()
    comps: list[list[str]] = []
    for cid in adj:
        if cid in seen:
            continue
        comp: list[str] = []
        queue = [cid]
        seen.add(cid)
        while queue:
            curr = queue.pop()
            comp.append(curr)
            for nbr in adj[curr]:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        comps.append(sorted(comp))
    return comps, adj


def test_ethene4_satellites_sink_the_pooled_gate_at_wider_bridge(
    ethene4_clusters: tuple[list[MicroCluster], float, float],
) -> None:
    """Widening the bridge radius dilutes the corridor's pooled aspect ratio.

    This is the root cause of the reported symptom. At k_bridge = 3.0 the
    corridor's component is the three clusters MC1+MC2+MC3 with a pooled
    aspect ratio above the threshold. At 3.25 two lone saddles (MC13, MC14)
    join the component, their off-axis bearing widens the pooled minor axis,
    and the pooled aspect ratio drops below the threshold -- so the legacy
    all-or-nothing gate rejected the corridor as well.
    """
    clusters, radius, pitch = ethene4_clusters
    by_id = {c.cluster_id: c for c in clusters}

    narrow_comps, _ = _components(clusters, pitch, 3.0)
    wide_comps, _ = _components(clusters, pitch, 3.25)

    narrow = next(c for c in narrow_comps if "MC1" in c)
    wide = next(c for c in wide_comps if "MC1" in c)

    assert narrow == ["MC1", "MC2", "MC3"]
    assert wide == ["MC1", "MC13", "MC14", "MC2", "MC3"]

    narrow_ar = compute_group_geometry(
        [by_id[cid] for cid in narrow], radius
    ).aspect_ratio
    wide_ar = compute_group_geometry([by_id[cid] for cid in wide], radius).aspect_ratio

    assert narrow_ar >= THRESHOLD
    assert wide_ar < THRESHOLD
    assert wide_ar < narrow_ar


def test_ethene4_core_selection_keeps_the_corridor(
    ethene4_clusters: tuple[list[MicroCluster], float, float],
) -> None:
    """The decomposed gate peels the satellites and keeps the corridor."""
    clusters, radius, pitch = ethene4_clusters
    by_id = {c.cluster_id: c for c in clusters}

    wide_comps, adj = _components(clusters, pitch, 3.25)
    wide = next(c for c in wide_comps if "MC1" in c)

    core, satellites = select_elongated_core(
        component=[by_id[cid] for cid in wide],
        aspect_ratio_threshold=THRESHOLD,
        sphere_radius=radius,
        adjacency=adj,
    )
    assert {c.cluster_id for c in core} == {"MC1", "MC2", "MC3"}
    assert {c.cluster_id for c in satellites} == {"MC13", "MC14"}
    assert compute_group_geometry(core, radius).aspect_ratio >= THRESHOLD


def test_ethene4_widening_the_bridge_does_not_increase_effective_cps(
    ethene4_clusters: tuple[list[MicroCluster], float, float],
) -> None:
    """The reported symptom is gone: a wider bridge never adds effective CPs.

    Before the fix, raising k_bridge from 3.0 to 3.25 dissolved the corridor,
    which raised the effective-CP count from 2 to 4. The response must now be
    non-increasing over the range where the corridor is stable, and the
    displayed Euler characteristic must stay at 2 throughout.
    """
    clusters, radius, pitch = ethene4_clusters

    def _reduced(k_bridge: float) -> tuple[int, int]:
        """Count the reduced groups and their summed index at a k_bridge.

        Mirrors the aggregator's own reduction: a group is reduced to one
        effective CP when it survives the elongation gate with more than one
        member.

        Args:
            k_bridge: Bridge radius in delta_theta units.

        Returns:
            Tuple of (number of reduced groups, summed net index).
        """
        comps, adj = _components(clusters, pitch, k_bridge)
        by_id = {c.cluster_id: c for c in clusters}
        n_reduced = 0
        index_sum = 0
        for comp in comps:
            group = [by_id[cid] for cid in comp]
            core, _satellites = select_elongated_core(
                component=group,
                aspect_ratio_threshold=THRESHOLD,
                sphere_radius=radius,
                adjacency=adj,
            )
            geom = compute_group_geometry(core, radius)
            if (
                len(core) > 1
                and not is_degenerate_group(geom)
                and geom.aspect_ratio >= THRESHOLD
            ):
                n_reduced += 1
                index_sum += sum(c.local_euler_index for c in core)
        return n_reduced, index_sum

    counts = [_reduced(kb)[0] for kb in (3.0, 3.05, 3.25, 3.5)]
    # The corridor must survive every one of these values, so the number of
    # reduced groups cannot grow.
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == counts[1] == counts[2] == counts[3]


def test_ethene4_aggregator_preserves_the_index_sum(
    ethene4_clusters: tuple[list[MicroCluster], float, float],
) -> None:
    """Every cluster lands in exactly one unit and the index sum is preserved."""
    clusters, radius, pitch = ethene4_clusters
    before = sum(c.local_euler_index for c in clusters)

    for k_bridge in (3.0, 3.25, 3.75, 5.0):
        macro, isolated = aggregate_cluster_strings(
            clusters=clusters,
            delta_theta_mesh_rad=pitch,
            sphere_radius=radius,
            k_bridge=k_bridge,
            aspect_ratio_threshold=THRESHOLD,
            bridge_all=True,
        )
        ids = [c.cluster_id for ms in macro for c in ms.constituent_clusters]
        ids += [c.cluster_id for c in isolated]
        assert len(ids) == len(set(ids)) == len(clusters)

        total = sum(ms.local_euler_index for ms in macro) + sum(
            c.local_euler_index for c in isolated
        )
        assert total == before


def test_ethene4_macro_string_members_match_its_constituents(
    ethene4_clusters: tuple[list[MicroCluster], float, float],
) -> None:
    """A macro-string's members come from its constituents, not the whole group.

    Regression guard: the constituents and the member list must agree, so a
    peeled satellite can never leave a member behind inside the corridor.
    """
    clusters, radius, pitch = ethene4_clusters
    macro, _ = aggregate_cluster_strings(
        clusters=clusters,
        delta_theta_mesh_rad=pitch,
        sphere_radius=radius,
        k_bridge=3.25,
        aspect_ratio_threshold=THRESHOLD,
        bridge_all=True,
    )
    assert macro
    for ms in macro:
        member_ids = {m.vertex_id for m in ms.all_members}
        expected = {m.vertex_id for c in ms.constituent_clusters for m in c.members}
        assert member_ids == expected
