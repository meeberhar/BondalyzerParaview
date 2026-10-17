"""Headless verification of the sphere_viewer "Save Parameters" export.

Replicates the save_parameters trigger body on ethene4.plt atom 1 with the
canonical DEFAULT_UI_STATE values, writes the JSON to the default path, and
prints it for inspection.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from vtkmodules.util import numpy_support

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.morse_detector import compute_discrete_morse_cps
from gba_topology2.src.sphere_viewer import (
    DEFAULT_UI_STATE,
    default_params_path,
    extract_atom_sphere_mesh,
    save_params_to_dict,
)

PLT = os.path.join(workspace_root, "ethene4.plt")
ATOM = 1

sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(PLT, ATOM)
print(f"Atom: {atom_info['atom_name']}  nodes={atom_info['num_nodes']}")

pts = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
n_cells = sphere_poly.GetNumberOfPolys()
cells = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
tris = cells.reshape((n_cells, 4))[:, 1:4]
geom = analyze_mesh_geometry(pts, tris, sphere_radius=float(atom_info["sphere_radius"]))

arr = sphere_poly.GetPointData().GetArray(field_items[0]["value"])
f_vals: np.ndarray = numpy_support.vtk_to_numpy(arr)
morse = compute_discrete_morse_cps(pts, tris, f_vals, tie_break_epsilon=1e-15)

# Seed the state exactly as run_sphere_viewer does at startup, then let the
# first pipeline pass fill the metric dicts (defaults are off, so display
# counts equal raw counts and reduction stats stay at zero).
values: dict[str, Any] = dict(DEFAULT_UI_STATE)
values["selected_field"] = field_items[0]["value"]
values["geom_info"] = {
    "num_vertices": geom["num_vertices"],
    "num_triangles": geom["num_triangles"],
    "num_edges": geom["num_edges"],
    "sphere_radius": float(geom["sphere_radius"]),
    "avg_edge_length": float(geom["avg_edge_length"]),
    "mesh_pitch_deg": float(geom["mesh_pitch_deg"]),
    "center": list(atom_info["center"]),
    "center_offset": float(atom_info["center_offset"]),
    "radius_source": str(atom_info["radius_source"]),
}
values["morse_counts"] = {
    "minima": len(morse.minima),
    "maxima": len(morse.maxima),
    "saddles": len(morse.saddles),
    "euler": morse.euler_characteristic,
    "valid": morse.is_valid_euler,
}
values["display_counts"] = dict(values["morse_counts"])
values["reduction_stats"] = {
    "n_catastrophes_reduced": 0,
    "n_clusters_reduced": 0,
    "n_clusters_annihilated": 0,
    "n_clusters_irreducible": 0,
    "n_clusters_bridged": 0,
    "n_effective_cps": 0,
    "n_raw_cps_hidden": 0,
    "n_nudged": 0,
    "max_nudge_deg": 0.0,
    "mean_nudge_deg": 0.0,
    "n_ports_perfected": 0,
}
values["launch_stats"] = {
    "n_saddles_seeded": 0,
    "n_ascent": 0,
    "n_descent": 0,
    "n_index_mismatch": 0,
    "max_fold_order": 0,
}
values["selected_cp"] = None
values["selected_cluster"] = None
values["selected_catastrophe"] = None
values["selected_basin_center"] = None

state = SimpleNamespace(**values)
payload = save_params_to_dict(
    state,
    plt_path=PLT,
    atom_info={**atom_info, "field_items": field_items},
    atom_num=ATOM,
)
out_path = default_params_path(PLT, ATOM)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")

print(f"[verify] Parameters saved to: {out_path}")
print(json.dumps(payload, indent=2, sort_keys=True))
sys.exit(0)
