#!/usr/bin/env python3
"""
Topological Analysis Engine for Spherical Triangle Meshes.

Implements global level-set topological analysis:
1. Discrete Morse theory / Banchoff local link classification (Euler characteristic check)
2. Sublevel and Superlevel Merge Trees (Join/Split Trees)
3. Topological persistence simplification and persistence-diagram pair tracking
4. Geodesic / spatial clustering of critical points on the 2-sphere
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx


class DisjointSet:
    """Disjoint-set (Union-Find) data structure with representative tracking."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rep = list(range(n))

    def find(self, i: int) -> int:
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i: int, j: int, values: np.ndarray, keep_min: bool = True) -> Tuple[int, Optional[int]]:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            rep_i = self.rep[root_i]
            rep_j = self.rep[root_j]
            v_i = values[rep_i]
            v_j = values[rep_j]
            
            # Decide which representative survives
            if (keep_min and v_i <= v_j) or (not keep_min and v_i >= v_j):
                self.parent[root_j] = root_i
                return root_i, rep_j  # rep_j is killed by rep_i
            else:
                self.parent[root_i] = root_j
                return root_j, rep_i  # rep_i is killed by rep_j
        return root_i, None


def compute_discrete_morse_cps(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Classify vertices on a closed triangulated 2-sphere into local Discrete Morse critical points:
    - Minima: Lower link has 0 connected components
    - Maxima: Upper link has 0 connected components
    - Regular: Lower link has 1 component and Upper link has 1 component
    - Saddles: Lower link has k >= 2 connected components (multiplicity k - 1)
    """
    n_pts = len(pts)
    v_tris = [[] for _ in range(n_pts)]
    for t_idx, tri in enumerate(triangles):
        v_tris[tri[0]].append(t_idx)
        v_tris[tri[1]].append(t_idx)
        v_tris[tri[2]].append(t_idx)

    raw_minima = []
    raw_maxima = []
    raw_saddles = []

    for v in range(n_pts):
        fv = f_vals[v]
        # Build cyclic link graph around v
        link_edges = []
        for t_idx in v_tris[v]:
            tri = triangles[t_idx]
            others = [int(x) for x in tri if x != v]
            if len(others) == 2:
                link_edges.append((others[0], others[1]))

        if not link_edges:
            continue

        g_link = nx.Graph()
        g_link.add_edges_from(link_edges)

        # Lower link L-(v): nodes with f < f(v)
        lower_nodes = [u for u in g_link.nodes() if f_vals[u] < fv]
        g_lower = g_link.subgraph(lower_nodes)
        n_lower = nx.number_connected_components(g_lower)

        # Upper link L+(v): nodes with f > f(v)
        upper_nodes = [u for u in g_link.nodes() if f_vals[u] > fv]
        g_upper = g_link.subgraph(upper_nodes)
        n_upper = nx.number_connected_components(g_upper)

        pos = pts[v]
        cp_record = {
            "vertex_id": int(v),
            "value": float(fv),
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        }

        if n_lower == 0 and n_upper == 1:
            cp_record["type"] = "minimum"
            raw_minima.append(cp_record)
        elif n_upper == 0 and n_lower == 1:
            cp_record["type"] = "maximum"
            raw_maxima.append(cp_record)
        elif n_lower >= 2:
            cp_record["type"] = "saddle"
            cp_record["multiplicity"] = int(n_lower - 1)
            raw_saddles.append(cp_record)

    return {
        "minima": raw_minima,
        "maxima": raw_maxima,
        "saddles": raw_saddles,
    }


def compute_merge_trees_and_persistence(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute sublevel and superlevel merge trees (Join/Split trees) to determine
    the global topological persistence of every Minimum, Maximum, and Saddle point.
    """
    n_pts = len(pts)
    v_adj = [set() for _ in range(n_pts)]
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v_adj[i0].update([i1, i2])
        v_adj[i1].update([i0, i2])
        v_adj[i2].update([i0, i1])

    # 1. SUBLEVEL MERGE TREE (Minima -> Saddles persistence pairs)
    order_asc = np.argsort(f_vals)
    uf_sub = DisjointSet(n_pts)
    active_sub = set()
    min_pairs = {}
    global_min_v = int(order_asc[0])

    for v in order_asc:
        v = int(v)
        active_sub.add(v)
        active_nbrs = [u for u in v_adj[v] if u in active_sub]
        nbr_roots = set(uf_sub.find(u) for u in active_nbrs)

        if len(nbr_roots) == 0:
            # Local minimum born at v
            pass
        elif len(nbr_roots) == 1:
            # Component growth
            r = next(iter(nbr_roots))
            uf_sub.parent[v] = r
        else:
            # Saddle point merging components
            roots_list = list(nbr_roots)
            base_root = roots_list[0]
            uf_sub.parent[v] = base_root
            for other_root in roots_list[1:]:
                survivor, killed_min = uf_sub.union(base_root, other_root, f_vals, keep_min=True)
                if killed_min is not None:
                    pers = float(f_vals[v] - f_vals[killed_min])
                    min_pairs[killed_min] = {
                        "cp_vertex": killed_min,
                        "paired_saddle": v,
                        "birth": float(f_vals[killed_min]),
                        "death": float(f_vals[v]),
                        "persistence": pers,
                    }
                base_root = survivor

    # Global minimum has infinite / max persistence across sphere range
    span = float(f_vals.max() - f_vals.min())
    min_pairs[global_min_v] = {
        "cp_vertex": global_min_v,
        "paired_saddle": -1,
        "birth": float(f_vals[global_min_v]),
        "death": float(f_vals.max()),
        "persistence": span,
    }

    # 2. SUPERLEVEL MERGE TREE (Maxima -> Saddles persistence pairs)
    order_desc = np.argsort(-f_vals)
    uf_sup = DisjointSet(n_pts)
    active_sup = set()
    max_pairs = {}
    global_max_v = int(order_desc[0])

    for v in order_desc:
        v = int(v)
        active_sup.add(v)
        active_nbrs = [u for u in v_adj[v] if u in active_sup]
        nbr_roots = set(uf_sup.find(u) for u in active_nbrs)

        if len(nbr_roots) == 0:
            # Local maximum born at v
            pass
        elif len(nbr_roots) == 1:
            r = next(iter(nbr_roots))
            uf_sup.parent[v] = r
        else:
            roots_list = list(nbr_roots)
            base_root = roots_list[0]
            uf_sup.parent[v] = base_root
            for other_root in roots_list[1:]:
                survivor, killed_max = uf_sup.union(base_root, other_root, f_vals, keep_min=False)
                if killed_max is not None:
                    pers = float(f_vals[killed_max] - f_vals[v])
                    max_pairs[killed_max] = {
                        "cp_vertex": killed_max,
                        "paired_saddle": v,
                        "birth": float(f_vals[killed_max]),
                        "death": float(f_vals[v]),
                        "persistence": pers,
                    }
                base_root = survivor

    max_pairs[global_max_v] = {
        "cp_vertex": global_max_v,
        "paired_saddle": -1,
        "birth": float(f_vals[global_max_v]),
        "death": float(f_vals.min()),
        "persistence": span,
    }

    return {
        "min_pairs": min_pairs,
        "max_pairs": max_pairs,
        "span": span,
    }


def analyze_spherical_topology(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
    persistence_threshold_pct: float = 1.0,
    cluster_radius: float = 0.15,
    cluster_angle_deg: Optional[float] = None,
    sphere_radius: Optional[float] = None,
    enable_harmonic_nudge: bool = False,
) -> Dict[str, Any]:
    """
    Full global topological pipeline on 2-sphere scalar field:
    1. Discrete Morse classification
    2. Sublevel and Superlevel Merge Tree persistence evaluation
    3. User-controlled persistence simplification filtering
    4. Mixed-type CP spatial clustering and composition analysis (angular or chord metric)
    5. Euler characteristic verification: N_max + N_min - N_saddles = 2
    """
    raw_cps = compute_discrete_morse_cps(pts, triangles, f_vals)
    trees = compute_merge_trees_and_persistence(pts, triangles, f_vals)

    min_pairs = trees["min_pairs"]
    max_pairs = trees["max_pairs"]
    span = max(trees["span"], 1e-12)
    threshold = (persistence_threshold_pct / 100.0) * span

    # Filter minima based on persistence
    filtered_minima = []
    active_saddle_ids = set()

    for m in raw_cps["minima"]:
        vid = m["vertex_id"]
        pair_info = min_pairs.get(vid, {"persistence": 0.0, "paired_saddle": -1})
        pers = pair_info["persistence"]
        m_copy = dict(m)
        m_copy["persistence"] = pers
        m_copy["persistence_pct"] = (pers / span) * 100.0
        m_copy["paired_saddle"] = pair_info.get("paired_saddle", -1)
        if pers >= threshold:
            filtered_minima.append(m_copy)
            if pair_info.get("paired_saddle", -1) >= 0:
                active_saddle_ids.add(pair_info["paired_saddle"])

    # Filter maxima based on persistence
    filtered_maxima = []
    for mx in raw_cps["maxima"]:
        vid = mx["vertex_id"]
        pair_info = max_pairs.get(vid, {"persistence": 0.0, "paired_saddle": -1})
        pers = pair_info["persistence"]
        mx_copy = dict(mx)
        mx_copy["persistence"] = pers
        mx_copy["persistence_pct"] = (pers / span) * 100.0
        mx_copy["paired_saddle"] = pair_info.get("paired_saddle", -1)
        if pers >= threshold:
            filtered_maxima.append(mx_copy)
            if pair_info.get("paired_saddle", -1) >= 0:
                active_saddle_ids.add(pair_info["paired_saddle"])

    # Filter saddles associated with persisting components
    filtered_saddles = []
    for s in raw_cps["saddles"]:
        vid = s["vertex_id"]
        s_copy = dict(s)
        assoc_pers = 0.0
        for p in min_pairs.values():
            if p.get("paired_saddle") == vid:
                assoc_pers = max(assoc_pers, p["persistence"])
        for p in max_pairs.values():
            if p.get("paired_saddle") == vid:
                assoc_pers = max(assoc_pers, p["persistence"])

        s_copy["persistence"] = assoc_pers
        s_copy["persistence_pct"] = (assoc_pers / span) * 100.0

        if vid in active_saddle_ids or assoc_pers >= threshold or persistence_threshold_pct == 0:
            filtered_saddles.append(s_copy)

    # Compute mean sphere radius from vertices if not provided
    if sphere_radius is None:
        sphere_radius = float(np.mean(np.linalg.norm(pts, axis=1)))

    final_minima = filtered_minima
    final_maxima = filtered_maxima
    final_saddles = filtered_saddles

    # Refine positions of local minima and maxima using harmonic circular contour variance minimization
    if enable_harmonic_nudge:
        # Refine Minima
        all_min_positions = [np.array(m["position"]) for m in final_minima]
        for idx_m, m in enumerate(final_minima):
            p_orig = np.array(m["position"])
            other_positions = [all_min_positions[j] for j in range(len(all_min_positions)) if j != idx_m]
            p_ref, n_deg = refine_extremum_harmonic_centroid(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                initial_pos=p_orig,
                sphere_radius=sphere_radius,
                sample_radius_deg=2.0,
                max_nudge_deg=1.5,
                other_extrema_positions=other_positions,
                min_separation_deg=2.0,
            )
            if n_deg > 0.05:
                m["position"] = p_ref.tolist()
                m["harmonic_nudge_deg"] = n_deg
                all_min_positions[idx_m] = p_ref

        # Refine Maxima
        all_max_positions = [np.array(mx["position"]) for mx in final_maxima]
        for idx_mx, mx in enumerate(final_maxima):
            p_orig = np.array(mx["position"])
            other_positions = [all_max_positions[j] for j in range(len(all_max_positions)) if j != idx_mx]
            p_ref, n_deg = refine_extremum_harmonic_centroid(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                initial_pos=p_orig,
                sphere_radius=sphere_radius,
                sample_radius_deg=2.0,
                max_nudge_deg=1.5,
                other_extrema_positions=other_positions,
                min_separation_deg=2.0,
            )
            if n_deg > 0.05:
                mx["position"] = p_ref.tolist()
                mx["harmonic_nudge_deg"] = n_deg
                all_max_positions[idx_mx] = p_ref

    # Assign sequential labels (MIN1, MAX1, SAD1)
    for idx, m in enumerate(sorted(final_minima, key=lambda x: x["value"])):
        m["id_label"] = f"MIN{idx + 1}"
    for idx, mx in enumerate(sorted(final_maxima, key=lambda x: -x["value"])):
        mx["id_label"] = f"MAX{idx + 1}"
    for idx, s in enumerate(sorted(final_saddles, key=lambda x: x["value"])):
        s["id_label"] = f"SAD{idx + 1}"

    n_min = len(final_minima)
    n_max = len(final_maxima)
    n_sad = len(final_saddles)
    euler_val = n_max + n_min - n_sad
    euler_valid = (euler_val == 2)

    # Compute mixed-type critical point clusters for composition analysis
    all_final_cps = []
    for m in final_minima:
        c = dict(m)
        c["type"] = "minimum"
        all_final_cps.append(c)
    for mx in final_maxima:
        c = dict(mx)
        c["type"] = "maximum"
        all_final_cps.append(c)
    for s in final_saddles:
        c = dict(s)
        c["type"] = "saddle"
        all_final_cps.append(c)

    clusters = compute_cp_clusters(
        all_final_cps,
        cluster_radius=cluster_radius,
        cluster_angle_deg=cluster_angle_deg,
        sphere_radius=sphere_radius,
    )

    return {
        "minima": final_minima,
        "maxima": final_maxima,
        "saddles": final_saddles,
        "clusters": clusters,
        "raw_counts": {
            "minima": len(raw_cps["minima"]),
            "maxima": len(raw_cps["maxima"]),
            "saddles": len(raw_cps["saddles"]),
            "euler": len(raw_cps["maxima"]) + len(raw_cps["minima"]) - sum(s.get("multiplicity", 1) for s in raw_cps["saddles"]),
        },
        "filtered_counts": {
            "minima": n_min,
            "maxima": n_max,
            "saddles": n_sad,
            "euler": euler_val,
            "euler_valid": euler_valid,
        },
        "span": span,
        "persistence_threshold": threshold,
        "sphere_radius": sphere_radius,
    }


def compute_cp_clusters(
    cps: List[Dict[str, Any]],
    cluster_radius: float = 0.15,
    cluster_angle_deg: Optional[float] = None,
    sphere_radius: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Cluster critical points of all types within an angular separation (degrees)
    or spatial chord radius.
    Computes cluster composition, net topological index (Euler contribution), centroid,
    and angular/spatial diameter.

    Returns a list of cluster dictionaries sorted by member count descending.
    """
    if not cps:
        return []

    n = len(cps)
    positions = np.array([c["position"] for c in cps], dtype=float)
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit_positions = positions / norms

    if sphere_radius is None:
        sphere_radius = float(np.mean(np.linalg.norm(positions, axis=1)))

    # Determine angular threshold
    if cluster_angle_deg is not None:
        target_angle_rad = math.radians(max(cluster_angle_deg, 1e-4))
    else:
        # Convert chord distance to angle on sphere of given radius
        ratio = min(max(cluster_radius / (2.0 * max(sphere_radius, 1e-6)), 0.0), 1.0)
        target_angle_rad = 2.0 * math.asin(ratio)

    # Build adjacency graph based on angular distance <= target_angle_rad
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            cos_val = np.clip(np.dot(unit_positions[i], unit_positions[j]), -1.0, 1.0)
            ang_ij = math.acos(cos_val)
            if ang_ij <= target_angle_rad:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    # Sort components so multi-element clusters come first, then largest first
    components.sort(key=lambda comp: (-len(comp), min(comp)))

    clusters = []
    for cl_idx, comp in enumerate(components):
        members = [cps[i] for i in comp]
        comp_positions = np.array([m["position"] for m in members])
        comp_units = unit_positions[comp]

        # Centroid projected to mean sphere radius
        mean_unit = np.mean(comp_units, axis=0)
        norm_mean = np.linalg.norm(mean_unit)
        if norm_mean > 1e-8:
            centroid_unit = mean_unit / norm_mean
            centroid = centroid_unit * sphere_radius
        else:
            centroid_unit = comp_units[0]
            centroid = comp_positions[0]

        # Calculate max angular distance and chord diameter
        max_ang = 0.0
        max_dist = 0.0
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                d = float(np.linalg.norm(comp_positions[i] - comp_positions[j]))
                cos_v = np.clip(np.dot(comp_units[i], comp_units[j]), -1.0, 1.0)
                ang = math.acos(cos_v)
                if d > max_dist:
                    max_dist = d
                if ang > max_ang:
                    max_ang = ang

        # Angular radius from centroid to furthest member
        max_rad_from_centroid = 0.0
        for i in range(len(comp)):
            cos_c = np.clip(np.dot(centroid_unit, comp_units[i]), -1.0, 1.0)
            ang_c = math.acos(cos_c)
            if ang_c > max_rad_from_centroid:
                max_rad_from_centroid = ang_c

        # Minimum margin for single point or small cluster
        angular_radius_deg = math.degrees(max(max_rad_from_centroid, math.radians(2.5)))

        # Composition counts
        n_max = sum(1 for m in members if m.get("type") == "maximum")
        n_min = sum(1 for m in members if m.get("type") == "minimum")
        n_sad = sum(1 for m in members if m.get("type") == "saddle")
        net_index = n_max + n_min - n_sad

        # Format summary string
        comp_parts = []
        if n_max > 0:
            comp_parts.append(f"{n_max} Max")
        if n_sad > 0:
            comp_parts.append(f"{n_sad} Sad")
        if n_min > 0:
            comp_parts.append(f"{n_min} Min")
        comp_summary = ", ".join(comp_parts) if comp_parts else "0 CPs"

        clusters.append({
            "id": f"CL{cl_idx + 1}",
            "size": len(members),
            "n_max": n_max,
            "n_min": n_min,
            "n_sad": n_sad,
            "net_index": net_index,
            "comp_summary": comp_summary,
            "centroid": centroid.tolist(),
            "centroid_unit": centroid_unit.tolist(),
            "radius": max(max_dist / 2.0, 0.04),
            "angular_radius_deg": angular_radius_deg,
            "angular_diameter_deg": math.degrees(max_ang),
            "diameter": max_dist,
            "members": members,
            "member_ids": [m.get("id_label", str(m.get("vertex_id"))) for m in members],
            "is_multicluster": len(members) > 1,
        })

    return clusters


def compute_effective_critical_points(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
    standard_cluster_angle_deg: float = 10.0,
    monkey_saddle_fuse_angle_deg: float = 16.0,
    sphere_radius: Optional[float] = None,
    enable_harmonic_nudge: bool = False,
) -> Dict[str, Any]:
    """
    Compute Effective Critical Points (ECPs) based on persistence stability:
    1. Base level analysis (lowest persistence setting = 0.0, zero geodesic merge distance)
       groups all raw CPs into two classes: 'isolated' (single-point clusters) and 'clustered' (multi-point clusters).
    2. Isolated CPs: Fixed directly from base detection (lowest persistence setting).
    3. Clustered CPs: Track the net Euler index (chi) of surviving CPs in each cluster cone
       as persistence threshold is swept.
       - If cluster chi is stable/unchanged across persistence simplification,
         an Effective CP with that characteristic (Max if chi=+1, Min if chi=+1 & min-dominant,
         Saddle if chi=-1, Monkey Saddle if chi=-2) is placed at the geometric centroid of the cluster.
    4. Second-Pass Saddle Fusion (Monkey Saddle resolution):
       - If two simple effective saddles (E-SAD) lie within monkey_saddle_fuse_angle_deg with no intervening
         extrema, they are recognized as the unfolded components of a single monkey saddle and fused into an E-MSAD (chi=-2).
    """
    if sphere_radius is None:
        sphere_radius = float(np.mean(np.linalg.norm(pts, axis=1)))

    # 1. Base analysis at lowest persistence (0.0)
    base_res = analyze_spherical_topology(
        pts,
        triangles,
        f_vals,
        persistence_threshold_pct=0.0,
        cluster_angle_deg=standard_cluster_angle_deg,
        sphere_radius=sphere_radius,
        enable_harmonic_nudge=enable_harmonic_nudge,
    )
    base_clusters = base_res["clusters"]

    # 2. Persistence sweep across thresholds (e.g. 0.0% to 5.0%)
    tau_samples = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    sweep_results = []
    for tau in tau_samples:
        r = analyze_spherical_topology(
            pts,
            triangles,
            f_vals,
            persistence_threshold_pct=tau,
            cluster_angle_deg=standard_cluster_angle_deg,
            sphere_radius=sphere_radius,
        )
        sweep_results.append((tau, r))

    # All base critical points for domain expansion queries
    all_base_cps = base_res["minima"] + base_res["maxima"] + base_res["saddles"]

    # Pre-process clusters: for complex multi-saddle / high-order clusters (|chi| > 1 or chi <= -2),
    # adaptively expand the domain boundary outwards until the enclosed net Euler characteristic resolves to +/-1.
    absorbed_cluster_ids = set()
    expanded_cluster_specs = []

    for cl in base_clusters:
        cl_id = cl["id"]
        if cl_id in absorbed_cluster_ids:
            continue

        is_multi = cl.get("is_multicluster", False)
        centroid = np.array(cl["centroid"])
        centroid_unit = np.array(cl["centroid_unit"])
        r_deg = float(cl.get("angular_radius_deg", standard_cluster_angle_deg))
        members = list(cl.get("members", []))
        base_chi = cl["n_max"] + cl["n_min"] - cl["n_sad"]

        # If this is a complex unfolded catastrophe hub (|base_chi| > 1 or base_chi <= -2)
        if is_multi and (abs(base_chi) > 1 or base_chi <= -2):
            curr_r = r_deg
            max_r = min(30.0, r_deg + 18.0)
            step_r = 1.0
            found_stable = False
            best_members = members
            best_r = curr_r
            best_chi = base_chi

            while curr_r <= max_r:
                # Find all base CPs within angular distance curr_r of the symmetry centroid
                enclosed_cps = []
                for p in all_base_cps:
                    p_pos = np.array(p["position"])
                    u = p_pos / (np.linalg.norm(p_pos) + 1e-12)
                    dot_v = np.clip(np.dot(centroid_unit, u), -1.0, 1.0)
                    ang = math.degrees(math.acos(dot_v))
                    if ang <= curr_r + 0.5:
                        enclosed_cps.append(p)

                n_mx = sum(1 for x in enclosed_cps if x["type"] == "maximum")
                n_mn = sum(1 for x in enclosed_cps if x["type"] == "minimum")
                n_sd = sum(1 for x in enclosed_cps if x["type"] == "saddle")
                enclosed_chi = n_mx + n_mn - n_sd

                if abs(enclosed_chi) == 1:
                    best_members = enclosed_cps
                    best_r = curr_r
                    best_chi = enclosed_chi
                    found_stable = True
                    break

                curr_r += step_r

            if found_stable:
                # Mark all other base clusters whose members or vertices are subsumed in this domain as absorbed
                absorbed_vids = {int(m.get("vertex_id", -1)) for m in best_members if int(m.get("vertex_id", -1)) >= 0}
                for other_cl in base_clusters:
                    if other_cl["id"] == cl_id:
                        continue
                    # Check if other cluster's centroid is inside our resolved domain
                    other_c = np.array(other_cl["centroid"])
                    other_u = other_c / (np.linalg.norm(other_c) + 1e-12)
                    d_ang = math.degrees(math.acos(np.clip(np.dot(centroid_unit, other_u), -1.0, 1.0)))
                    if d_ang <= best_r + 0.5:
                        absorbed_cluster_ids.add(other_cl["id"])
                    else:
                        # Also check if any member vertices overlap
                        other_vids = {int(m.get("vertex_id", -1)) for m in other_cl.get("members", []) if int(m.get("vertex_id", -1)) >= 0}
                        if other_vids and other_vids.issubset(absorbed_vids):
                            absorbed_cluster_ids.add(other_cl["id"])

                # Update cluster properties
                cl["members"] = best_members
                cl["angular_radius_deg"] = best_r
                cl["n_max"] = sum(1 for x in best_members if x["type"] == "maximum")
                cl["n_min"] = sum(1 for x in best_members if x["type"] == "minimum")
                cl["n_sad"] = sum(1 for x in best_members if x["type"] == "saddle")
                cl["net_index"] = best_chi
                comp_parts = []
                if cl["n_max"] > 0:
                    comp_parts.append(f"{cl['n_max']} Max")
                if cl["n_sad"] > 0:
                    comp_parts.append(f"{cl['n_sad']} Sad")
                if cl["n_min"] > 0:
                    comp_parts.append(f"{cl['n_min']} Min")
                cl["comp_summary"] = ", ".join(comp_parts)
                cl["expanded_domain"] = True
                cl["is_multicluster"] = True

    effective_maxima = []
    effective_minima = []
    effective_saddles = []
    effective_cps_summary = []

    for cl in base_clusters:
        cl_id = cl["id"]
        if cl_id in absorbed_cluster_ids:
            continue

        is_multi = cl.get("is_multicluster", False) or cl.get("expanded_domain", False)
        centroid = np.array(cl["centroid"])
        centroid_unit = np.array(cl["centroid_unit"])
        r_deg = cl.get("angular_radius_deg", standard_cluster_angle_deg)
        members = cl.get("members", [])

        if not is_multi:
            # CLASS 1: ISOLATED CP
            # Corresponds to the lowest persistence setting; fixed at its detected location
            m0 = members[0]
            cp_type = m0["type"]
            val = float(m0["value"])
            pos = list(m0["position"])
            vid = int(m0.get("vertex_id", -1))
            pers_pct = float(m0.get("persistence_pct", 100.0))

            ecp_entry = {
                "category": "isolated",
                "cluster_id": cl_id,
                "type": cp_type,
                "value": val,
                "position": pos,
                "vertex_id": vid,
                "persistence_pct": pers_pct,
                "source": "isolated (fixed at base)",
                "chi": 1 if cp_type in ("maximum", "minimum") else -1,
                "num_base_members": 1,
            }

            if cp_type == "maximum":
                effective_maxima.append(ecp_entry)
            elif cp_type == "minimum":
                effective_minima.append(ecp_entry)
            else:
                effective_saddles.append(ecp_entry)
            effective_cps_summary.append(ecp_entry)

        else:
            # CLASS 2: CLUSTERED CPs
            # Track cluster chi across persistence sweep
            chi_history = []
            comp_history = []

            for tau, r_swp in sweep_results:
                surv_list = r_swp["maxima"] + r_swp["minima"] + r_swp["saddles"]
                in_cone = []
                for p in surv_list:
                    p_pos = np.array(p["position"])
                    u = p_pos / (np.linalg.norm(p_pos) + 1e-12)
                    dot_v = np.clip(np.dot(centroid_unit, u), -1.0, 1.0)
                    ang = math.degrees(math.acos(dot_v))
                    if ang <= r_deg + 1.5:
                        in_cone.append(p)

                n_mx = sum(1 for x in in_cone if x["type"] == "maximum")
                n_mn = sum(1 for x in in_cone if x["type"] == "minimum")
                n_sd = sum(1 for x in in_cone if x["type"] == "saddle")
                chi_val = n_mx + n_mn - n_sd
                chi_history.append(chi_val)
                comp_history.append((n_mx, n_mn, n_sd))

            # Determine dominant stable chi
            # Filter non-zero chi occurrences in moderate persistence range
            mod_chis = [c for c in chi_history[:6] if c != 0]
            if not mod_chis:
                # If all smoothed steps yielded 0, respect the base cluster's intrinsic Euler index
                base_chi = cl["n_max"] + cl["n_min"] - cl["n_sad"]
                stable_chi = base_chi
            else:
                # Most frequent non-zero chi across low-to-mid persistence
                vals, counts = np.unique(mod_chis, return_counts=True)
                stable_chi = int(vals[np.argmax(counts)])

            # Mean scalar value of cluster members
            mean_val = float(np.mean([m["value"] for m in members]))
            max_pers = float(max((m.get("persistence_pct", 0.0) for m in members), default=0.0))

            if stable_chi == 1:
                # Differentiate Maximum vs Minimum via Boundary Flux Principle (Macroscopic Normal Derivative)
                # Sample scalar field on the outer cluster boundary loop to determine true basin vs peak slope
                sample_r = max(float(r_deg) + 1.5, 4.0)
                _, _, b_vals = sample_circle_field_values(
                    pts=pts,
                    triangles=triangles,
                    f_vals=f_vals,
                    centroid=centroid,
                    angular_radius_deg=sample_r,
                    num_samples=32,
                    sphere_radius=sphere_radius,
                )
                mean_boundary_val = float(np.mean(b_vals))

                # Identify highest-persistence / core member extremum values
                min_members_vals = [m["value"] for m in members if m.get("type") == "minimum"]
                max_members_vals = [m["value"] for m in members if m.get("type") == "maximum"]

                # Compare boundary value to the core extremum values
                # If there's a strong minimum inside whose basin rises to the boundary -> MINIMUM
                # If there's a strong maximum inside whose peak drops to the boundary -> MAXIMUM
                delta_from_mean = mean_boundary_val - mean_val
                
                # If both min and max members exist, check relative depth/height against boundary
                if min_members_vals and max_members_vals:
                    core_min = min(min_members_vals)
                    core_max = max(max_members_vals)
                    min_depth = mean_boundary_val - core_min  # positive if boundary is higher than min
                    max_height = core_max - mean_boundary_val # positive if boundary is lower than max
                    if min_depth > max_height:
                        delta_flux = 1.0  # Dominant basin (MINIMUM)
                    else:
                        delta_flux = -1.0 # Dominant peak (MAXIMUM)
                elif min_members_vals and not max_members_vals:
                    delta_flux = 1.0
                elif max_members_vals and not min_members_vals:
                    delta_flux = -1.0
                else:
                    delta_flux = delta_from_mean

                # If boundary values are higher than centroid/core, it is an ascending basin -> MINIMUM
                # If boundary values are lower than centroid/core, it is a descending peak -> MAXIMUM
                if delta_flux > 0:
                    ecp_type = "minimum"
                    effective_minima.append({
                        "category": "clustered",
                        "cluster_id": cl_id,
                        "type": ecp_type,
                        "value": min(min_members_vals) if min_members_vals else mean_val,
                        "position": centroid.tolist(),
                        "vertex_id": -1,
                        "persistence_pct": max_pers,
                        "source": f"cluster centroid ({cl['comp_summary']} -> stable chi=+1 flux-min)",
                        "chi": 1,
                        "num_base_members": len(members),
                    })
                elif delta_flux < 0:
                    ecp_type = "maximum"
                    effective_maxima.append({
                        "category": "clustered",
                        "cluster_id": cl_id,
                        "type": ecp_type,
                        "value": max(max_members_vals) if max_members_vals else mean_val,
                        "position": centroid.tolist(),
                        "vertex_id": -1,
                        "persistence_pct": max_pers,
                        "source": f"cluster centroid ({cl['comp_summary']} -> stable chi=+1 flux-max)",
                        "chi": 1,
                        "num_base_members": len(members),
                    })
                else:
                    # Fallback to member count only if flux is exactly zero
                    if cl["n_max"] >= cl["n_min"]:
                        ecp_type = "maximum"
                        effective_maxima.append({
                            "category": "clustered",
                            "cluster_id": cl_id,
                            "type": ecp_type,
                            "value": mean_val,
                            "position": centroid.tolist(),
                            "vertex_id": -1,
                            "persistence_pct": max_pers,
                            "source": f"cluster centroid ({cl['comp_summary']} -> stable chi=+1)",
                            "chi": 1,
                            "num_base_members": len(members),
                        })
                    else:
                        ecp_type = "minimum"
                        effective_minima.append({
                            "category": "clustered",
                            "cluster_id": cl_id,
                            "type": ecp_type,
                            "value": mean_val,
                            "position": centroid.tolist(),
                            "vertex_id": -1,
                            "persistence_pct": max_pers,
                            "source": f"cluster centroid ({cl['comp_summary']} -> stable chi=+1)",
                            "chi": 1,
                            "num_base_members": len(members),
                        })
                effective_cps_summary.append(effective_maxima[-1] if ecp_type == "maximum" else effective_minima[-1])

            elif stable_chi == -1:
                ecp_type = "saddle"
                effective_saddles.append({
                    "category": "clustered",
                    "cluster_id": cl_id,
                    "type": ecp_type,
                    "value": mean_val,
                    "position": centroid.tolist(),
                    "vertex_id": -1,
                    "persistence_pct": max_pers,
                    "source": f"cluster centroid ({cl['comp_summary']} -> stable chi=-1)",
                    "chi": -1,
                    "num_base_members": len(members),
                })
                effective_cps_summary.append(effective_saddles[-1])

            elif stable_chi <= -2:
                # Higher-order Monkey Saddle / Multi-Saddle (multiplicity >= 2, index <= -2)
                ecp_type = "monkey_saddle"
                effective_saddles.append({
                    "category": "clustered",
                    "cluster_id": cl_id,
                    "type": ecp_type,
                    "value": mean_val,
                    "position": centroid.tolist(),
                    "vertex_id": -1,
                    "persistence_pct": max_pers,
                    "source": f"cluster centroid ({cl['comp_summary']} -> stable chi={stable_chi} multi-saddle)",
                    "chi": stable_chi,
                    "num_base_members": len(members),
                })
                effective_cps_summary.append(effective_saddles[-1])

            elif stable_chi == 0:
                # True neutral / dipole pair with zero base index
                pass
            else:
                # Multi-index positive complex (e.g. chi >= +2)
                if stable_chi > 0:
                    effective_maxima.append({
                        "category": "clustered",
                        "cluster_id": cl_id,
                        "type": "maximum",
                        "value": mean_val,
                        "position": centroid.tolist(),
                        "vertex_id": -1,
                        "persistence_pct": max_pers,
                        "source": f"cluster centroid ({cl['comp_summary']} -> complex chi={stable_chi})",
                        "chi": stable_chi,
                        "num_base_members": len(members),
                    })

    # Second-Pass Saddle Fusion (Monkey Saddle resolution based on proximity sensitivity)
    # If two simple effective saddles lie within monkey_saddle_fuse_angle_deg without an intervening extremum,
    # fuse them into an Effective Monkey Saddle (E-MSAD, chi=-2).
    if monkey_saddle_fuse_angle_deg > 0 and len(effective_saddles) >= 2:
        fused_saddles = []
        visited_saddles = set()
        fuse_rad = math.radians(monkey_saddle_fuse_angle_deg)

        for i in range(len(effective_saddles)):
            if i in visited_saddles:
                continue
            s1 = effective_saddles[i]
            p1 = np.array(s1["position"])
            u1 = p1 / (np.linalg.norm(p1) + 1e-12)

            pair_match = None
            for j in range(i + 1, len(effective_saddles)):
                if j in visited_saddles:
                    continue
                s2 = effective_saddles[j]
                if s1.get("type") == "monkey_saddle" or s2.get("type") == "monkey_saddle":
                    continue
                p2 = np.array(s2["position"])
                u2 = p2 / (np.linalg.norm(p2) + 1e-12)
                ang_ij = math.acos(np.clip(np.dot(u1, u2), -1.0, 1.0))

                if ang_ij <= fuse_rad:
                    # Check if there is an intervening strong extremum directly between them
                    midpoint = (p1 + p2) / 2.0
                    u_mid = midpoint / (np.linalg.norm(midpoint) + 1e-12)
                    has_intervening_ext = False
                    for ext in (effective_maxima + effective_minima):
                        p_ext = np.array(ext["position"])
                        u_ext = p_ext / (np.linalg.norm(p_ext) + 1e-12)
                        ang_ext = math.acos(np.clip(np.dot(u_mid, u_ext), -1.0, 1.0))
                        if ang_ext < (ang_ij / 2.0 + math.radians(1.5)):
                            has_intervening_ext = True
                            break

                    if not has_intervening_ext:
                        pair_match = (j, s2, ang_ij)
                        break

            if pair_match is not None:
                j_idx, s2, ang_sep = pair_match
                visited_saddles.add(i)
                visited_saddles.add(j_idx)
                p2 = np.array(s2["position"])
                # Centroid of the two unfolded saddles on the sphere
                fused_pos = (p1 + p2) / 2.0
                fused_pos = (fused_pos / np.linalg.norm(fused_pos)) * sphere_radius
                mean_v = (s1["value"] + s2["value"]) / 2.0
                max_p = max(s1.get("persistence_pct", 0.0), s2.get("persistence_pct", 0.0))
                num_m = s1.get("num_base_members", 1) + s2.get("num_base_members", 1)

                fused_saddles.append({
                    "category": "clustered",
                    "cluster_id": f"{s1.get('cluster_id')}+{s2.get('cluster_id')}",
                    "type": "monkey_saddle",
                    "value": mean_v,
                    "position": fused_pos.tolist(),
                    "vertex_id": -1,
                    "persistence_pct": max_p,
                    "source": f"fused saddle pair (sep={math.degrees(ang_sep):.1f}° <= {monkey_saddle_fuse_angle_deg:.1f}° -> monkey saddle chi=-2)",
                    "chi": -2,
                    "num_base_members": num_m,
                })
            else:
                visited_saddles.add(i)
                fused_saddles.append(s1)

        effective_saddles = fused_saddles

    # Refine positions of all Effective Minima & Maxima
    if enable_harmonic_nudge:
        # Effective Minima
        all_emin_positions = [np.array(mn["position"]) for mn in effective_minima]
        for idx_mn, mn in enumerate(effective_minima):
            p_orig = np.array(mn["position"])
            other_positions = [all_emin_positions[j] for j in range(len(all_emin_positions)) if j != idx_mn]
            p_ref, n_deg = refine_extremum_harmonic_centroid(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                initial_pos=p_orig,
                sphere_radius=sphere_radius,
                sample_radius_deg=2.0,
                max_nudge_deg=1.5,
                other_extrema_positions=other_positions,
                min_separation_deg=2.0,
            )
            if n_deg > 0.05:
                mn["position"] = p_ref.tolist()
                mn["source"] += f" | harmonic minimum nudge={n_deg:.2f}°"
                all_emin_positions[idx_mn] = p_ref

        # Effective Maxima
        all_emax_positions = [np.array(mx["position"]) for mx in effective_maxima]
        for idx_mx, mx in enumerate(effective_maxima):
            p_orig = np.array(mx["position"])
            other_positions = [all_emax_positions[j] for j in range(len(all_emax_positions)) if j != idx_mx]
            p_ref, n_deg = refine_extremum_harmonic_centroid(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                initial_pos=p_orig,
                sphere_radius=sphere_radius,
                sample_radius_deg=2.0,
                max_nudge_deg=1.5,
                other_extrema_positions=other_positions,
                min_separation_deg=2.0,
            )
            if n_deg > 0.05:
                mx["position"] = p_ref.tolist()
                mx["source"] += f" | harmonic maximum nudge={n_deg:.2f}°"
                all_emax_positions[idx_mx] = p_ref

    # Assign sequential labels for Effective CPs
    for idx, mx in enumerate(sorted(effective_maxima, key=lambda x: -x["value"])):
        mx["id_label"] = f"E-MAX{idx + 1}"
        mx["badge_color"] = "error"
        mx["type_title"] = "Effective Maximum"

    for idx, mn in enumerate(sorted(effective_minima, key=lambda x: x["value"])):
        mn["id_label"] = f"E-MIN{idx + 1}"
        mn["badge_color"] = "info"
        mn["type_title"] = "Effective Minimum"

    for idx, sd in enumerate(sorted(effective_saddles, key=lambda x: x["value"])):
        is_monkey = (sd.get("type") == "monkey_saddle" or sd.get("chi") == -2)
        if is_monkey:
            sd["id_label"] = f"E-MSAD{idx + 1}"
            sd["badge_color"] = "purple-accent-3"
            sd["type_title"] = "Effective Monkey Saddle"
        else:
            sd["id_label"] = f"E-SAD{idx + 1}"
            sd["badge_color"] = "success"
            sd["type_title"] = "Effective Saddle"

    effective_cps_summary = effective_maxima + effective_minima + effective_saddles
    n_emax = len(effective_maxima)
    n_emin = len(effective_minima)
    # Sum effective index contribution: monkey saddle contributes -2
    index_sum = sum(mx.get("chi", 1) for mx in effective_maxima) + sum(mn.get("chi", 1) for mn in effective_minima) + sum(sd.get("chi", -1) for sd in effective_saddles)

    # 4. Adaptive Excision Boundary Sizing and Port Extraction
    excision_regions = compute_adaptive_excision_regions(
        pts=pts,
        triangles=triangles,
        f_vals=f_vals,
        effective_cps=effective_cps_summary,
        sphere_radius=sphere_radius,
        enable_harmonic_nudge=enable_harmonic_nudge,
    )

    return {
        "maxima": effective_maxima,
        "minima": effective_minima,
        "saddles": effective_saddles,
        "all_effective_cps": effective_maxima + effective_minima + effective_saddles,
        "excision_regions": excision_regions,
        "counts": {
            "maxima": n_emax,
            "minima": n_emin,
            "saddles": len([s for s in effective_saddles if s.get("type") != "monkey_saddle"]),
            "monkey_saddles": len([s for s in effective_saddles if s.get("type") == "monkey_saddle"]),
            "euler": index_sum,
            "euler_valid": (index_sum == 2),
            "isolated_count": len([c for c in base_clusters if not c.get("is_multicluster")]),
            "clustered_count": len([c for c in base_clusters if c.get("is_multicluster")]),
        },
        "base_clusters": base_clusters,
        "standard_cluster_angle_deg": standard_cluster_angle_deg,
    }


def sample_circle_field_values(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
    centroid: np.ndarray,
    angular_radius_deg: float,
    num_samples: int = 64,
    sphere_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample scalar field values along a circular boundary loop on the sphere.
    Uses inverse distance weighting (IDW) interpolation from nearby mesh vertices.

    Returns:
        angles_rad: 1D array of sample angles [0, 2pi)
        sample_pts: (N, 3) 3D coordinates on sphere
        sampled_f: 1D array of interpolated scalar values
    """
    if sphere_radius is None:
        sphere_radius = float(np.mean(np.linalg.norm(pts, axis=1)))

    c_norm = np.linalg.norm(centroid)
    c_unit = centroid / (c_norm + 1e-12)

    # Tangent orthonormal basis (t1, t2)
    if abs(c_unit[0]) < 0.8 and abs(c_unit[1]) < 0.8:
        v_temp = np.array([1.0, 0.0, 0.0])
    else:
        v_temp = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(c_unit, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(c_unit, t1)
    t2 /= np.linalg.norm(t2)

    ang_rad = math.radians(max(angular_radius_deg, 0.1))
    cos_a = math.cos(ang_rad)
    sin_a = math.sin(ang_rad)

    angles_rad = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=False)
    sample_pts = np.zeros((num_samples, 3), dtype=float)

    for i, phi in enumerate(angles_rad):
        dir_vec = cos_a * c_unit + sin_a * (math.cos(phi) * t1 + math.sin(phi) * t2)
        dir_vec /= np.linalg.norm(dir_vec)
        sample_pts[i] = dir_vec * sphere_radius

    # Fast KDTree or spatial IDW from closest mesh vertices
    sampled_f = np.zeros(num_samples, dtype=float)
    # Filter mesh vertices within a bounding angular cone to speed up IDW
    dot_all = np.dot(pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12), c_unit)
    candidate_indices = np.where(dot_all >= math.cos(math.radians(angular_radius_deg + 12.0)))[0]
    if len(candidate_indices) < 3:
        candidate_indices = np.arange(len(pts))

    cand_pts = pts[candidate_indices]
    cand_f = f_vals[candidate_indices]

    for i in range(num_samples):
        diffs = cand_pts - sample_pts[i]
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 1e-7:
            sampled_f[i] = cand_f[min_idx]
        else:
            # 4 nearest neighbors IDW
            k = min(4, len(dists))
            k_indices = np.argpartition(dists, k)[:k]
            k_dists = dists[k_indices]
            weights = 1.0 / (k_dists ** 2)
            sampled_f[i] = np.sum(weights * cand_f[k_indices]) / np.sum(weights)

    return angles_rad, sample_pts, sampled_f


def find_boundary_extrema_ports(
    sample_pts: np.ndarray,
    sampled_f: np.ndarray,
) -> Dict[str, Any]:
    """
    Find 1D local maxima (Ridge Ports) and local minima (Valley Ports) along the circular boundary.
    Handles cyclic boundary conditions.

    Returns dictionary with lists of maxima ports, minima ports, and total count.
    """
    n = len(sampled_f)
    if n < 4:
        return {"maxima_ports": [], "minima_ports": [], "num_maxima": 0, "num_minima": 0, "is_alternating": False}

    # Smooth sampled_f slightly with circular moving average to eliminate sub-mesh discretization flutter
    smoothed_f = np.zeros(n, dtype=float)
    for i in range(n):
        smoothed_f[i] = 0.25 * sampled_f[(i - 1) % n] + 0.50 * sampled_f[i] + 0.25 * sampled_f[(i + 1) % n]

    maxima_ports = []
    minima_ports = []
    extrema_sequence = []

    for i in range(n):
        prev_val = smoothed_f[(i - 1) % n]
        curr_val = smoothed_f[i]
        next_val = smoothed_f[(i + 1) % n]

        if curr_val > prev_val and curr_val >= next_val:
            p_pos = sample_pts[i].tolist()
            entry = {"index": i, "value": float(curr_val), "position": p_pos, "port_type": "ridge_max"}
            maxima_ports.append(entry)
            extrema_sequence.append(("max", i))
        elif curr_val < prev_val and curr_val <= next_val:
            p_pos = sample_pts[i].tolist()
            entry = {"index": i, "value": float(curr_val), "position": p_pos, "port_type": "valley_min"}
            minima_ports.append(entry)
            extrema_sequence.append(("min", i))

    # Check strict cyclic alternation (max -> min -> max -> min ...)
    is_alternating = True
    if len(extrema_sequence) >= 2:
        for k in range(len(extrema_sequence)):
            curr_type = extrema_sequence[k][0]
            next_type = extrema_sequence[(k + 1) % len(extrema_sequence)][0]
            if curr_type == next_type:
                is_alternating = False
                break
    else:
        is_alternating = False

    return {
        "maxima_ports": maxima_ports,
        "minima_ports": minima_ports,
        "num_maxima": len(maxima_ports),
        "num_minima": len(minima_ports),
        "is_alternating": is_alternating,
        "total_extrema": len(maxima_ports) + len(minima_ports),
    }


def compute_harmonic_port_energy(
    extrema_ports_info: Dict[str, Any],
    target_count: int = 6,
    target_spacing_deg: float = 60.0,
) -> float:
    """
    Compute the eccentricity energy E(c) = sum (Delta_theta_k - target_spacing)^2
    measuring how evenly spaced the boundary ports are along the boundary loop.
    A lower energy indicates the boundary circle is centered directly over the true harmonic singularity.
    """
    all_ports = extrema_ports_info.get("maxima_ports", []) + extrema_ports_info.get("minima_ports", [])
    if len(all_ports) != target_count or not extrema_ports_info.get("is_alternating", False):
        # Penalty for missing or extra non-alternating ports
        return 1e6

    # Sort ports by index / circular sample angle
    all_ports_sorted = sorted(all_ports, key=lambda p: p["index"])
    n_samples = 64
    angles_deg = np.array([p["index"] * (360.0 / n_samples) for p in all_ports_sorted])

    # Compute adjacent circular angular intervals
    diffs = np.zeros(target_count)
    for k in range(target_count):
        d = (angles_deg[(k + 1) % target_count] - angles_deg[k]) % 360.0
        diffs[k] = d

    return float(np.sum((diffs - target_spacing_deg) ** 2))


def refine_monkey_saddle_harmonic_centroid(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
    initial_centroid: np.ndarray,
    optimal_radius_deg: float,
    sphere_radius: float,
    max_nudge_deg: float = 4.0,
    search_grid_steps: int = 7,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Refine the 3D position of an effective monkey saddle by nudging its center on the sphere surface
    to minimize the eccentricity energy of the 6 alternating boundary ports (3 ridges + 3 valleys).

    Returns:
        refined_centroid: (3,) numpy array on sphere
        best_ports_info: Dict containing the optimized boundary ports
    """
    c0 = np.array(initial_centroid)
    c0_norm = np.linalg.norm(c0)
    c0_unit = c0 / (c0_norm + 1e-12)

    # Tangent orthonormal vectors t1, t2
    if abs(c0_unit[0]) < 0.8 and abs(c0_unit[1]) < 0.8:
        v_temp = np.array([1.0, 0.0, 0.0])
    else:
        v_temp = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(c0_unit, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(c0_unit, t1)
    t2 /= np.linalg.norm(t2)

    # Evaluate initial candidate
    _, sample_pts0, sampled_f0 = sample_circle_field_values(
        pts=pts,
        triangles=triangles,
        f_vals=f_vals,
        centroid=c0_unit * sphere_radius,
        angular_radius_deg=optimal_radius_deg,
        num_samples=64,
        sphere_radius=sphere_radius,
    )
    ports0 = find_boundary_extrema_ports(sample_pts0, sampled_f0)
    best_energy = compute_harmonic_port_energy(ports0, target_count=6, target_spacing_deg=60.0)
    best_unit = c0_unit.copy()
    best_ports = ports0

    # 2D search in tangent plane within max_nudge_deg cone
    nudge_rad_max = math.radians(max_nudge_deg)
    u_vals = np.linspace(-nudge_rad_max, nudge_rad_max, search_grid_steps)
    v_vals = np.linspace(-nudge_rad_max, nudge_rad_max, search_grid_steps)

    for du in u_vals:
        for dv in v_vals:
            if du ** 2 + dv ** 2 > nudge_rad_max ** 2:
                continue
            cand_vec = c0_unit + du * t1 + dv * t2
            cand_unit = cand_vec / np.linalg.norm(cand_vec)

            _, s_pts, s_f = sample_circle_field_values(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                centroid=cand_unit * sphere_radius,
                angular_radius_deg=optimal_radius_deg,
                num_samples=64,
                sphere_radius=sphere_radius,
            )
            cand_ports = find_boundary_extrema_ports(s_pts, s_f)
            cand_energy = compute_harmonic_port_energy(cand_ports, target_count=6, target_spacing_deg=60.0)

            # Preference for lower eccentricity + closer to initial centroid in case of tie
            if cand_energy < best_energy - 1e-4:
                best_energy = cand_energy
                best_unit = cand_unit
                best_ports = cand_ports

    refined_centroid = best_unit * sphere_radius
    return refined_centroid, best_ports


def refine_extremum_harmonic_centroid(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
    initial_pos: np.ndarray,
    sphere_radius: float,
    sample_radius_deg: float = 2.0,
    max_nudge_deg: float = 1.5,
    search_grid_steps: int = 9,
    other_extrema_positions: Optional[List[np.ndarray]] = None,
    min_separation_deg: float = 2.5,
) -> Tuple[np.ndarray, float]:
    """
    Refine the position of a local extremum (minimum or maximum, isolated or effective) by nudging
    its center on the sphere to minimize the circular contour scalar variance Var(f|_{partial D(c, r)}).
    In an elliptical or asymmetric basin/peak, this pulls the discrete vertex extremum to the true continuous
    geometric/harmonic center of the surrounding level curves while respecting topological separation.

    Returns:
        refined_pos: (3,) numpy array on sphere
        nudge_deg: float angular nudge applied (degrees)
    """
    c0 = np.array(initial_pos)
    c0_norm = np.linalg.norm(c0)
    c0_unit = c0 / (c0_norm + 1e-12)

    # Tangent orthonormal vectors t1, t2
    if abs(c0_unit[0]) < 0.8 and abs(c0_unit[1]) < 0.8:
        v_temp = np.array([1.0, 0.0, 0.0])
    else:
        v_temp = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(c0_unit, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(c0_unit, t1)
    t2 /= np.linalg.norm(t2)

    # Initial variance at c0
    _, _, sampled_f0 = sample_circle_field_values(
        pts=pts,
        triangles=triangles,
        f_vals=f_vals,
        centroid=c0_unit * sphere_radius,
        angular_radius_deg=sample_radius_deg,
        num_samples=48,
        sphere_radius=sphere_radius,
    )
    best_var = float(np.var(sampled_f0))
    best_unit = c0_unit.copy()

    # 2D search in tangent plane
    nudge_rad_max = math.radians(max_nudge_deg)
    u_vals = np.linspace(-nudge_rad_max, nudge_rad_max, search_grid_steps)
    v_vals = np.linspace(-nudge_rad_max, nudge_rad_max, search_grid_steps)

    for du in u_vals:
        for dv in v_vals:
            if du ** 2 + dv ** 2 > nudge_rad_max ** 2:
                continue
            cand_vec = c0_unit + du * t1 + dv * t2
            cand_unit = cand_vec / np.linalg.norm(cand_vec)

            # Check separation constraint against other extrema to prevent collapse
            if other_extrema_positions is not None:
                too_close = False
                for other_pos in other_extrema_positions:
                    other_u = other_pos / (np.linalg.norm(other_pos) + 1e-12)
                    dot_o = np.clip(np.dot(cand_unit, other_u), -1.0, 1.0)
                    ang_o = math.degrees(math.acos(dot_o))
                    if ang_o < min_separation_deg:
                        too_close = True
                        break
                if too_close:
                    continue

            _, _, s_f = sample_circle_field_values(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                centroid=cand_unit * sphere_radius,
                angular_radius_deg=sample_radius_deg,
                num_samples=48,
                sphere_radius=sphere_radius,
            )
            cand_var = float(np.var(s_f))

            if cand_var < best_var - 1e-9:
                best_var = cand_var
                best_unit = cand_unit

    refined_pos = best_unit * sphere_radius
    cos_nudge = np.clip(np.dot(c0_unit, best_unit), -1.0, 1.0)
    nudge_deg = math.degrees(math.acos(cos_nudge))
    return refined_pos, nudge_deg


def compute_adaptive_excision_regions(
    pts: np.ndarray,
    triangles: np.ndarray,
    f_vals: np.ndarray,
    effective_cps: List[Dict[str, Any]],
    sphere_radius: float,
    min_search_deg: float = 3.0,
    max_search_deg: float = 24.0,
    step_deg: float = 0.5,
    enable_harmonic_nudge: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each clustered Effective Critical Point, dynamically expand the circular boundary radius r
    until the 1D boundary scalar profile achieves the pure topological port signature:
      - Simple Saddle (E-SAD): exactly 2 Maxima (Ridge Ports) and 2 Minima (Valley Ports) -> 4 alternating crossings.
      - Monkey Saddle (E-MSAD): exactly 3 Maxima (Ridge Ports) and 3 Minima (Valley Ports) -> 6 alternating crossings.
      - Extremum (E-MAX / E-MIN): monotonic / single boundary peak/valley.

    Returns the list of optimal excision regions with exact boundary ports and circle geometry.
    """
    excision_regions = []

    for ecp in effective_cps:
        if ecp.get("category") != "clustered":
            continue

        centroid = np.array(ecp["position"])
        ecp_type = ecp.get("type", "saddle")
        chi = ecp.get("chi", -1)
        ecp_id = ecp.get("id_label", "ECP")

        # Target number of boundary peaks (ridges) and valleys based on topological index
        if ecp_type == "monkey_saddle" or chi == -2:
            target_maxima = 3
            target_minima = 3
        elif ecp_type == "saddle" or chi == -1:
            target_maxima = 2
            target_minima = 2
        else:
            target_maxima = 1
            target_minima = 1

        optimal_radius_deg = None
        optimal_ports = None

        r_sweep = np.arange(min_search_deg, max_search_deg + step_deg, step_deg)
        for r_deg in r_sweep:
            angles_rad, sample_pts, sampled_f = sample_circle_field_values(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                centroid=centroid,
                angular_radius_deg=float(r_deg),
                num_samples=64,
                sphere_radius=sphere_radius,
            )
            ports_info = find_boundary_extrema_ports(sample_pts, sampled_f)

            if (
                ports_info["num_maxima"] == target_maxima
                and ports_info["num_minima"] == target_minima
                and ports_info["is_alternating"]
            ):
                optimal_radius_deg = float(r_deg)
                optimal_ports = ports_info
                break

        # If strict target was not found in sweep, use fallback radius from cluster base
        if optimal_radius_deg is None:
            optimal_radius_deg = float(max(min_search_deg, 12.0))
            angles_rad, sample_pts, sampled_f = sample_circle_field_values(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                centroid=centroid,
                angular_radius_deg=optimal_radius_deg,
                num_samples=64,
                sphere_radius=sphere_radius,
            )
            optimal_ports = find_boundary_extrema_ports(sample_pts, sampled_f)

        # Harmonic Centroid Refinement for Monkey Saddles
        final_centroid = centroid
        if enable_harmonic_nudge and (ecp_type == "monkey_saddle" or chi == -2):
            refined_c, refined_ports = refine_monkey_saddle_harmonic_centroid(
                pts=pts,
                triangles=triangles,
                f_vals=f_vals,
                initial_centroid=centroid,
                optimal_radius_deg=optimal_radius_deg,
                sphere_radius=sphere_radius,
                max_nudge_deg=3.5,
                search_grid_steps=9,
            )
            final_centroid = refined_c
            optimal_ports = refined_ports
            # Update the ECP position in-place
            ecp["position"] = refined_c.tolist()
            # Calculate nudge distance
            nudge_deg = math.degrees(math.acos(np.clip(np.dot(centroid / np.linalg.norm(centroid), refined_c / np.linalg.norm(refined_c)), -1.0, 1.0)))
            ecp["source"] += f" | harmonic port nudge={nudge_deg:.2f}°"

        excision_regions.append({
            "ecp_id": ecp_id,
            "ecp_type": ecp_type,
            "chi": chi,
            "centroid": final_centroid.tolist(),
            "optimal_radius_deg": optimal_radius_deg,
            "arc_radius_angstrom": float(sphere_radius * math.radians(optimal_radius_deg)),
            "target_signature": f"{target_maxima} Max, {target_minima} Min",
            "found_signature": f"{optimal_ports['num_maxima']} Max, {optimal_ports['num_minima']} Min",
            "is_pure_signature": bool(optimal_ports["num_maxima"] == target_maxima and optimal_ports["num_minima"] == target_minima),
            "maxima_ports": optimal_ports["maxima_ports"],
            "minima_ports": optimal_ports["minima_ports"],
            "all_ports": optimal_ports["maxima_ports"] + optimal_ports["minima_ports"],
        })

    return excision_regions
