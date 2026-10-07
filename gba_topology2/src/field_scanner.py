"""Field Scanner Subroutine for gba_topology2.

Scans all condensed scalar fields of an AtomSphereData mesh to detect
adjacent same-type critical points (Max-Max or Min-Min sharing a mesh edge, d_G = 1).
Reports the field name, enclosing micro-cluster, vertex IDs, and values.

NOTE ON PIPELINE ARCHITECTURE (Step 3: Extrema Fusion):
    In the standard production pipeline, Simulation of Simplicity (SoS) tie-breaking
    (tie_break_epsilon=1e-15) in morse_detector.py breaks plateau ties deterministically
    (yielding chi = 2) without producing adjacent same-type extrema (d_G = 1).
    Downstream, Harmonic Nudge (Step 8) optimizes effective critical point positions.

    Consequently, an active runtime scan/fusion pass is bypassed in the default pipeline.
    This module serves as the standalone diagnostic tool and reference implementation
    should explicit pre-clustering plateau fusion (d_G = 1) be integrated upstream of
    micro_cluster.py in future iterations.

Run with:
    uv run python -m gba_topology2.src.field_scanner Pd_20K.plt
or:
    uv run python gba_topology2/src/field_scanner.py Pd_20K.plt
"""

import argparse
import os
import sys
from typing import Any

import networkx as nx
import numpy as np
from vtkmodules.util import numpy_support

# Ensure workspace root is in sys.path
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.micro_cluster import compute_micro_clusters
from gba_topology2.src.morse_detector import compute_discrete_morse_cps
from gba_topology2.src.sphere_viewer import extract_atom_sphere_mesh

__all__ = ["scan_fields_for_close_extrema"]


def scan_fields_for_close_extrema(
    plt_path: str,
    target_atom_num: int = 1,
    tie_break_epsilon: float = 0.0,
    k_pitch: float = 2.0,
) -> list[dict[str, Any]]:
    """Scan all condensed fields for adjacent (d_G = 1) same-type extrema.

    Args:
        plt_path: Path to the .plt file.
        target_atom_num: 1-based index of the target atom.
        tie_break_epsilon: Epsilon for Simulation of Simplicity tie-breaking (default 0.0).
        k_pitch: Multiplier for mesh pitch to define micro-clusters (default 2.0).

    Returns:
        List of detected adjacency event dictionaries.
    """
    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        plt_path, target_atom_num
    )

    pts = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    triangles = cell_arr.reshape((n_cells, 4))[:, 1:4]

    geom = analyze_mesh_geometry(pts, triangles)
    delta_theta_rad = float(geom["mesh_pitch_rad"])
    delta_theta_deg = float(geom["mesh_pitch_deg"])
    sphere_radius = float(geom["sphere_radius"])

    # Build 1-skeleton mesh graph to evaluate direct edge adjacency (d_G = 1)
    mesh_graph = nx.Graph()
    for tri in triangles:
        mesh_graph.add_edge(int(tri[0]), int(tri[1]))
        mesh_graph.add_edge(int(tri[1]), int(tri[2]))
        mesh_graph.add_edge(int(tri[2]), int(tri[0]))

    pd = sphere_poly.GetPointData()
    detected_events: list[dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(
        f"SCANNING DATASET: {os.path.basename(plt_path)} | Atom #{target_atom_num} ({atom_info.get('atom_type', 'Pd')})"
    )
    print(
        f"Mesh: {len(pts)} vertices, {len(triangles)} triangles, R = {sphere_radius:.4f} Å, δθ = {delta_theta_deg:.3f}°"
    )
    print(f"Tie-break epsilon: {tie_break_epsilon:g}")
    print("=" * 90)

    header = f"{'Field Name':<32} | {'Max':>4} | {'Min':>4} | {'Sad':>4} | {'χ':>3} | {'Max-Max(d=1)':>12} | {'Min-Min(d=1)':>12}"
    print(header)
    print("-" * len(header))

    for item in field_items:
        field_name = item["value"]
        arr = pd.GetArray(field_name)
        if arr is None:
            continue
        f_vals = numpy_support.vtk_to_numpy(arr).astype(np.float64)

        # Detect CPs and construct micro-clusters
        morse_res = compute_discrete_morse_cps(
            pts, triangles, f_vals, tie_break_epsilon=tie_break_epsilon
        )
        all_cps = morse_res.minima + morse_res.maxima + morse_res.saddles
        clusters = compute_micro_clusters(
            cps=all_cps,
            delta_theta_mesh_rad=delta_theta_rad,
            k_pitch=k_pitch,
            sphere_radius=sphere_radius,
        )

        # Map vertex_id -> cluster_id
        v_to_cluster: dict[int, str] = {}
        for cl in clusters:
            for m in cl.members:
                v_to_cluster[m.vertex_id] = cl.cluster_id

        # Scan for adjacent Max-Max pairs (d_G = 1)
        max_adj_count = 0
        for i, cp1 in enumerate(morse_res.maxima):
            for cp2 in morse_res.maxima[i + 1 :]:
                if mesh_graph.has_edge(cp1.vertex_id, cp2.vertex_id):
                    max_adj_count += 1
                    detected_events.append(
                        {
                            "field": field_name,
                            "type": "Max-Max",
                            "v1": cp1.vertex_id,
                            "v2": cp2.vertex_id,
                            "val1": cp1.value,
                            "val2": cp2.value,
                            "cluster_id": v_to_cluster.get(cp1.vertex_id, "None"),
                            "cluster_id_v2": v_to_cluster.get(cp2.vertex_id, "None"),
                        }
                    )

        # Scan for adjacent Min-Min pairs (d_G = 1)
        min_adj_count = 0
        for i, cp1 in enumerate(morse_res.minima):
            for cp2 in morse_res.minima[i + 1 :]:
                if mesh_graph.has_edge(cp1.vertex_id, cp2.vertex_id):
                    min_adj_count += 1
                    detected_events.append(
                        {
                            "field": field_name,
                            "type": "Min-Min",
                            "v1": cp1.vertex_id,
                            "v2": cp2.vertex_id,
                            "val1": cp1.value,
                            "val2": cp2.value,
                            "cluster_id": v_to_cluster.get(cp1.vertex_id, "None"),
                            "cluster_id_v2": v_to_cluster.get(cp2.vertex_id, "None"),
                        }
                    )

        print(
            f"{field_name:<32} | {len(morse_res.maxima):>4} | {len(morse_res.minima):>4} | "
            f"{len(morse_res.saddles):>4} | {morse_res.euler_characteristic:>3} | "
            f"{max_adj_count:>12} | {min_adj_count:>12}"
        )

    print("-" * len(header))

    if detected_events:
        print(
            f"\n[!] Detected {len(detected_events)} adjacent same-type CP pairs (d_G = 1):\n"
        )
        for ev in detected_events:
            print(
                f"  • Field: '{ev['field']}' | Cluster: {ev['cluster_id']} | Type: {ev['type']} | "
                f"Vertices: {ev['v1']} (val={ev['val1']:.6g}) <--> {ev['v2']} (val={ev['val2']:.6g})"
            )
    else:
        print(
            "\nNo adjacent same-type CPs (d_G = 1) detected across any scanned fields."
        )

    return detected_events


def main() -> None:
    """CLI entrypoint for field scanner."""
    parser = argparse.ArgumentParser(
        description="Scan fields for adjacent same-type critical points."
    )
    parser.add_argument("plt_file", help="Path to input .plt file (e.g. Pd_20K.plt)")
    parser.add_argument(
        "--atom",
        type=int,
        default=1,
        help="Target atom index (1-based, default 1)",
    )
    parser.add_argument(
        "--tie-break",
        type=float,
        default=0.0,
        help="Tie-break epsilon (0.0 for raw, 1e-15 for SoS)",
    )
    parser.add_argument(
        "--k-pitch",
        type=float,
        default=2.0,
        help="Micro-cluster pitch multiplier (default 2.0)",
    )

    args = parser.parse_args()
    scan_fields_for_close_extrema(
        plt_path=args.plt_file,
        target_atom_num=args.atom,
        tie_break_epsilon=args.tie_break,
        k_pitch=args.k_pitch,
    )


if __name__ == "__main__":
    main()
