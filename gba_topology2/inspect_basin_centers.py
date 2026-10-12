"""Read-only diagnostic: verify basin center coordinates lie on the atom sphere.

For every BasinCenter produced by ``basin_center.extract_basin_centers`` this
reports the distance from the nuclear centre. A correct basin center sits at
~1.0 R. A value near 0 R means the point is still in the centred (local) frame
while the sphere is drawn in the world frame -- the "halo far from the sphere"
symptom.

Run with:
    uv run python gba_topology2/inspect_basin_centers.py ethene3.plt -a 1
    uv run python gba_topology2/inspect_basin_centers.py Pd_20K.plt -a 1
"""

import argparse
import os
import sys

import numpy as np
from vtkmodules.util import numpy_support

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.src.basin_center import extract_basin_centers
from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.morse_detector import compute_discrete_morse_cps
from gba_topology2.src.sphere_viewer import extract_atom_sphere_mesh


def main() -> None:
    parser = argparse.ArgumentParser(prog="inspect_basin_centers.py")
    parser.add_argument("plt_file", nargs="?", default="Pd_20K.plt")
    parser.add_argument("-a", "--atom", type=int, default=1)
    parser.add_argument("-f", "--field", type=str, default=None)
    args, _unknown = parser.parse_known_args()

    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        args.plt_file, target_atom_num=args.atom
    )
    center = np.asarray(atom_info["center"], dtype=np.float64)
    R = float(atom_info["sphere_radius"])

    pts_np = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    tris = cell_arr.reshape((n_cells, 4))[:, 1:4]
    geom = analyze_mesh_geometry(pts_np, tris, sphere_radius=R)

    field = args.field or field_items[0]["value"]
    arr = sphere_poly.GetPointData().GetArray(field)
    f_vals = numpy_support.vtk_to_numpy(arr)

    print(f"file        : {args.plt_file}")
    print(f"atom        : {atom_info['atom_name']}")
    print(f"field       : {field}")
    print(f"center      : ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"|center|    : {np.linalg.norm(center):.4f} A")
    print(f"R           : {R:.6f} A")
    print(f"pitch       : {geom['mesh_pitch_deg']:.4f} deg")
    print()

    morse = compute_discrete_morse_cps(pts_np, tris, f_vals, tie_break_epsilon=1e-15)
    all_raw = morse.minima + morse.maxima + morse.saddles

    # Case A: no reduction -- all basin centers come from raw CPs, nucleus passed
    centers_a = extract_basin_centers(
        all_raw, effective_cps=None, hidden_vertex_ids=set(), center=tuple(center)
    )
    _report("A: raw only, nucleus passed (fixed)", centers_a, center, R)

    # Case A_bad: the pre-fix call, nucleus omitted -- raw positions stay in the
    # centred frame while the sphere is drawn in the world frame.
    centers_bad = extract_basin_centers(
        all_raw, effective_cps=None, hidden_vertex_ids=set(), center=None
    )
    _report("A_bad: raw only, nucleus omitted (the bug)", centers_bad, center, R)

    # Case B: simulate an effective E-MAX built the way the viewer builds it
    # (position mapped to world via to_global(center)).
    from gba_topology2.src.sphere_frame import to_global

    if morse.maxima:
        m0 = morse.maxima[0]
        eff_dicts = [
            {
                "id_label": "E-MAX_MC99",
                "type": "E-MAX",
                "position": list(to_global(m0.position, center)),
                "value": m0.value,
                "origin_id": "MC99",
                "net_index": 1,
                "member_count": 1,
            }
        ]
        centers_b = extract_basin_centers(
            all_raw,
            effective_cps=eff_dicts,
            hidden_vertex_ids={m0.vertex_id},
            center=tuple(center),
        )
        _report("B: world-frame E-MAX + raw, nucleus passed", centers_b, center, R)

        eff_only = [bc for bc in centers_b if bc.is_effective]
        for bc in eff_only:
            p = np.asarray(bc.position, dtype=np.float64)
            print(
                f"    [eff] {bc.center_id:<14} d/R={np.linalg.norm(p - center) / R:.4f}"
            )
        print()


def _report(label: str, centers: list, center: np.ndarray, R: float) -> None:
    print(f"--- {label}: {len(centers)} basin centers ---")
    if not centers:
        print("  (none)")
        print()
        return
    ratios = []
    for bc in centers:
        p = np.asarray(bc.position, dtype=np.float64)
        d = float(np.linalg.norm(p - center))
        ratios.append(d / R)
    ratios_arr = np.asarray(ratios)
    print(
        f"  ||p - center|| / R : min={ratios_arr.min():.4f} "
        f"median={np.median(ratios_arr):.4f} max={ratios_arr.max():.4f}"
    )
    for bc in centers[:4]:
        p = np.asarray(bc.position, dtype=np.float64)
        d = float(np.linalg.norm(p - center))
        print(
            f"    {bc.center_id:<14} {bc.polarity:<8} "
            f"pos=({p[0]:8.4f},{p[1]:8.4f},{p[2]:8.4f})  d/R={d / R:.4f}"
        )
    print()


if __name__ == "__main__":
    main()
