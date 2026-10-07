"""Standalone verification of the Step 6 effective-CP reduction & nudge logic.

Replicates the decision logic added to sphere_viewer.update_topology() so the
four assessment toggles can be exercised headlessly on real data.
"""

import math
import os
import sys

import numpy as np

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from vtkmodules.util import numpy_support

from gba_topology2.src.catastrophe_classifier import classify_catastrophes
from gba_topology2.src.excision_boundary import profile_boundary_loop
from gba_topology2.src.harmonic_nudge import (
    harmonic_nudge_critical_point,
    perfect_boundary_ports,
    reposition_ports_on_ring,
)
from gba_topology2.src.micro_cluster import compute_micro_clusters
from gba_topology2.src.morse_detector import compute_discrete_morse_cps
from gba_topology2.src.polarity_detector import classify_cluster_polarity
from gba_topology2.src.sphere_viewer import extract_atom_sphere_mesh

PLT = os.path.join(workspace_root, "Pd_20K.plt")
FIELD = "V (condensed)"

sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(PLT, target_atom_num=1)
print(f"Atom: {atom_info['atom_name']}  nodes={atom_info['num_nodes']}")
print(f"Fields: {[f['value'] for f in field_items][:6]} ...")

pts_np = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
n_cells = sphere_poly.GetNumberOfPolys()
cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
triangles_np = cell_arr.reshape((n_cells, 4))[:, 1:4]

from gba_topology2.src.mesh_geometry import analyze_mesh_geometry

geom = analyze_mesh_geometry(pts_np, triangles_np)
sphere_radius = geom["sphere_radius"]
delta_theta_rad = geom["mesh_pitch_rad"]
print(
    f"sphere_radius={sphere_radius:.6f}  mesh_pitch={math.degrees(delta_theta_rad):.4f} deg"
)

arr = sphere_poly.GetPointData().GetArray(FIELD)
assert arr is not None
f_vals = numpy_support.vtk_to_numpy(arr)

morse = compute_discrete_morse_cps(
    pts_np, triangles_np, f_vals, tie_break_epsilon=1e-15
)
print(
    f"\nRAW MORSE: max={len(morse.maxima)} min={len(morse.minima)} "
    f"sad={len(morse.saddles)} chi={morse.euler_characteristic} valid={morse.is_valid_euler}"
)

all_raw = morse.minima + morse.maxima + morse.saddles
clusters = compute_micro_clusters(
    cps=all_raw,
    delta_theta_mesh_rad=delta_theta_rad,
    k_pitch=2.0,
    sphere_radius=sphere_radius,
    min_boundary_margin_deg=1.5,
    num_boundary_points=48,
)
n_multi = sum(1 for c in clusters if c.is_multi_cp)
print(f"CLUSTERS: {len(clusters)} total, {n_multi} multi-CP")

cats, _ = classify_catastrophes(
    clusters=clusters,
    field_vals=f_vals,
    delta_theta_mesh_rad=delta_theta_rad,
    sphere_radius=sphere_radius,
    barrier_tol=0.005,
    angular_tol_mult=4.0,
)
print(f"CATASTROPHES: {len(cats)}")


def run(reduce_cats: bool, reduce_cl: bool, nudge: bool, perfect: bool) -> dict:
    hidden: set[int] = set()
    effs: list[dict] = []
    nudge_degs: list[float] = []
    n_perfected = 0
    cat_cluster_ids: set[str] = set()

    for cat in cats:
        b_res = profile_boundary_loop(
            ring_points=cat.boundary_ring, mesh_pts=pts_np, f_vals=f_vals
        )
        vp, rp = b_res.valley_ports, b_res.ridge_ports
        if perfect:
            vp = reposition_ports_on_ring(
                perfect_boundary_ports(vp, cat.fold_order),
                cat.boundary_ring,
                sphere_radius,
                f_vals,
                pts_np,
            )
            rp = reposition_ports_on_ring(
                perfect_boundary_ports(rp, cat.fold_order),
                cat.boundary_ring,
                sphere_radius,
                f_vals,
                pts_np,
            )
            n_perfected += len(vp) + len(rp)
            # Verify ports lie on the ring's small circle
            c = np.array(cat.centroid, dtype=float)
            cu = c / np.linalg.norm(c)
            for p in list(vp) + list(rp):
                pu = np.array(p.position) / np.linalg.norm(p.position)
                ang = math.acos(float(np.clip(pu @ cu, -1.0, 1.0)))
                assert abs(ang - cat.angular_radius_rad) < 1e-3, (
                    f"port off ring: {math.degrees(ang)} vs {math.degrees(cat.angular_radius_rad)}"
                )

        for cl in cat.constituent_clusters:
            cat_cluster_ids.add(cl.cluster_id)

        nd = None
        if reduce_cats and nudge:
            probe = max(
                1.5,
                cat.angular_span_deg / 2.0 / max(math.degrees(delta_theta_rad), 1e-9),
            )
            nd = harmonic_nudge_critical_point(
                initial_pos=cat.centroid,
                mesh_pts=pts_np,
                f_vals=f_vals,
                fold_order=cat.fold_order,
                delta_theta_mesh_rad=delta_theta_rad,
                sphere_radius=sphere_radius,
                max_displacement_pitch=1.5,
                probe_radius_pitch=probe,
            )
        if reduce_cats:
            pos = nd.optimized_position if (nudge and nd) else cat.centroid
            if nudge and nd:
                nudge_degs.append(nd.displacement_ang_deg)
            effs.append(
                {
                    "id": f"E-SAD_{cat.entity_id}",
                    "type": "E-SAD",
                    "pos": pos,
                    "index": cat.net_euler_index,
                    "n_members": len(cat.all_members),
                    "energy_drop": (nd.initial_energy - nd.final_energy) if nd else 0.0,
                }
            )
            for m in cat.all_members:
                hidden.add(m.vertex_id)

    n_cl_red = 0
    if reduce_cl:
        for cl in clusters:
            if not cl.is_multi_cp or cl.cluster_id in cat_cluster_ids:
                continue
            eff = classify_cluster_polarity(
                cluster=cl,
                mesh_pts=pts_np,
                f_vals=f_vals,
                delta_theta_mesh_rad=delta_theta_rad,
                sphere_radius=sphere_radius,
            )
            nd2 = None
            if nudge:
                nd2 = harmonic_nudge_critical_point(
                    initial_pos=cl.centroid,
                    mesh_pts=pts_np,
                    f_vals=f_vals,
                    fold_order=0,
                    delta_theta_mesh_rad=delta_theta_rad,
                    sphere_radius=sphere_radius,
                    max_displacement_pitch=1.5,
                    probe_radius_pitch=max(
                        1.5,
                        cl.angular_radius_deg
                        / max(math.degrees(delta_theta_rad), 1e-9),
                    ),
                )
            pos = nd2.optimized_position if nudge and nd2 else cl.centroid
            if nudge and nd2:
                nudge_degs.append(nd2.displacement_ang_deg)
            effs.append(
                {
                    "id": f"{eff.extremum_type}_{cl.cluster_id}",
                    "type": eff.extremum_type,
                    "pos": pos,
                    "index": eff.local_euler_index,
                    "n_members": len(cl.members),
                    "energy_drop": (nd2.initial_energy - nd2.final_energy)
                    if nd2
                    else 0.0,
                }
            )
            for m in cl.members:
                hidden.add(m.vertex_id)
            n_cl_red += 1

    n_max = sum(1 for m in morse.maxima if m.vertex_id not in hidden)
    n_min = sum(1 for m in morse.minima if m.vertex_id not in hidden)
    n_sad = sum(1 for m in morse.saddles if m.vertex_id not in hidden)
    # Multiplicity-weighted index sum: invariant under set reduction
    index_sum = 0
    for m in morse.maxima:
        if m.vertex_id not in hidden:
            index_sum += 1
    for m in morse.minima:
        if m.vertex_id not in hidden:
            index_sum += 1
    for m in morse.saddles:
        if m.vertex_id not in hidden:
            index_sum -= m.multiplicity
    for e in effs:
        if e["type"] == "E-MAX":
            n_max += 1
        elif e["type"] == "E-MIN":
            n_min += 1
        else:
            n_sad += 1
        index_sum += e["index"]

    return {
        "hidden": len(hidden),
        "n_eff": len(effs),
        "n_cats_red": len(cats) if reduce_cats else 0,
        "n_cl_red": n_cl_red,
        "max": n_max,
        "min": n_min,
        "sad": n_sad,
        "chi": index_sum,
        "n_nudged": len(nudge_degs) if nudge else 0,
        "mean_nudge": (sum(nudge_degs) / len(nudge_degs))
        if (nudge and nudge_degs)
        else 0.0,
        "max_nudge": max(nudge_degs) if (nudge and nudge_degs) else 0.0,
        "n_perfected": n_perfected if perfect else 0,
        "effs": effs,
    }


cases = [
    ("base (all off)", False, False, False, False),
    ("reduce cats", True, False, False, False),
    ("reduce clusters", False, True, False, False),
    ("reduce both", True, True, False, False),
    ("reduce cats + nudge", True, False, True, False),
    ("reduce both + nudge", True, True, True, False),
    ("perfect ports", False, False, False, True),
    ("ALL ON", True, True, True, True),
]

for name, rc, rl, nu, pf in cases:
    r = run(rc, rl, nu, pf)
    print(
        f"\n--- {name} ---\n"
        f"  hidden={r['hidden']} eff={r['n_eff']} catsRed={r['n_cats_red']} clustRed={r['n_cl_red']}\n"
        f"  displayed: max={r['max']} min={r['min']} sad={r['sad']} chi={r['chi']}\n"
        f"  nudged={r['n_nudged']} mean={r['mean_nudge']:.3f}deg max={r['max_nudge']:.3f}deg"
        f"  portsPerfected={r['n_perfected']}"
    )
    if rc and r["effs"]:
        e0 = r["effs"][0]
        print(
            f"  first ECP: {e0['id']} index={e0['index']} members={e0['n_members']} "
            f"pos=({e0['pos'][0]:.4f},{e0['pos'][1]:.4f},{e0['pos'][2]:.4f})"
        )

# Sanity: base state must reproduce raw counts and chi=2
base = run(False, False, False, False)
assert base["hidden"] == 0 and base["n_eff"] == 0
assert base["max"] == len(morse.maxima)
assert base["sad"] == len(morse.saddles)
assert base["chi"] == 2, base["chi"]

# Reducing must strictly reduce the number of displayed CPs
both = run(True, True, False, False)
assert both["hidden"] > 0
assert (both["max"] + both["min"] + both["sad"]) < (
    base["max"] + base["min"] + base["sad"]
)

# Nudge must be a descent method: energy must not increase
nudged = run(True, True, True, False)
assert all(e["energy_drop"] >= -1e-12 for e in nudged["effs"]), [
    e["energy_drop"] for e in nudged["effs"]
]

print("\nALL ASSERTIONS PASSED")
