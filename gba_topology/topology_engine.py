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
    geodesic_min_dist: float = 0.05,
) -> Dict[str, Any]:
    """
    Full global topological pipeline on 2-sphere scalar field:
    1. Discrete Morse classification
    2. Sublevel and Superlevel Merge Tree persistence evaluation
    3. User-controlled persistence simplification filtering
    4. Geodesic distance clustering of close degenerate CPs
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
        # Include if it was the pairing death saddle for an accepted min or max
        # or if its raw multiplicity requires it
        s_copy = dict(s)
        # Compute maximum persistence associated with this saddle
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

    # Spatial / geodesic clustering: merge any CPs of same type within geodesic_min_dist
    def cluster_cps(cp_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not cp_list or geodesic_min_dist <= 0:
            return cp_list
        clustered = []
        visited = set()
        for i, c1 in enumerate(cp_list):
            if i in visited:
                continue
            group = [c1]
            visited.add(i)
            p1 = np.array(c1["position"])
            for j, c2 in enumerate(cp_list):
                if j in visited:
                    continue
                p2 = np.array(c2["position"])
                # Euclidean chord distance on sphere ~ geodesic arc distance
                dist = float(np.linalg.norm(p1 - p2))
                if dist <= geodesic_min_dist:
                    group.append(c2)
                    visited.add(j)
            # Pick representative with highest persistence
            best = max(group, key=lambda x: x.get("persistence", 0.0))
            clustered.append(best)
        return clustered

    final_minima = cluster_cps(filtered_minima)
    final_maxima = cluster_cps(filtered_maxima)
    final_saddles = cluster_cps(filtered_saddles)

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

    return {
        "minima": final_minima,
        "maxima": final_maxima,
        "saddles": final_saddles,
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
    }
