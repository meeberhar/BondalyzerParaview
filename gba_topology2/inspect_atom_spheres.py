"""Read-only inspection of AtomSphereData sphere frames in a Tecplot .plt file.

For every analysed atom sphere this reports:
- the zone metadata radius (Aux SphereRadius) and the mesh size;
- the vertex centroid, i.e. the true nuclear position x_center;
- the offset of that centroid from the world origin;
- the naive radius mean(||x||) used by the pre-centering pipeline;
- the true radius mean(||x - x_center||) and its agreement with the metadata;
- the angular mesh pitch delta_theta from BOTH radii, the quantity that
  corrupts micro-clustering when the atom sits off the world origin.

Run with:
    uv run python gba_topology2/inspect_atom_spheres.py ethene3.plt
    uv run python gba_topology2/inspect_atom_spheres.py Pd_20K.plt
    uv run python gba_topology2/inspect_atom_spheres.py ethene3.plt --atom 3
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkPolyData

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from BondalyzerParaView.plt_gba_to_vtm import extract_gba_zones_from_plt
from gba_topology2.src.mesh_geometry import (
    compute_average_edge_length,
    compute_mesh_pitch,
    extract_unique_edges,
)

__all__ = ["SphereFrame", "inspect_atom_spheres"]


@dataclass(frozen=True)
class SphereFrame:
    """Measured geometry of one AtomSphereData zone.

    Attributes:
        atom_number: Value of the zone's Aux AtomNumber.
        atom_symbol: Value of the zone's Aux AtomType.
        zone_name: Tecplot zone name.
        num_nodes: Vertex count of the sphere mesh.
        num_triangles: Triangle count of the sphere mesh.
        meta_radius: Aux SphereRadius, or None when the zone omits it.
        center: Vertex centroid (the true nuclear position), global frame.
        center_offset: ||center||, the distance from the world origin.
        naive_radius: mean(||x||), the radius the current code computes.
        fitted_radius: mean(||x - center||), the true sphere radius.
        radial_spread: max - min of ||x - center||, a sphericity check.
        avg_edge_length: Mean edge length over the unique mesh edges.
        pitch_naive_deg: delta_theta from avg_edge_length and naive_radius.
        pitch_true_deg: delta_theta from avg_edge_length and fitted_radius.
    """

    atom_number: int
    atom_symbol: str
    zone_name: str
    num_nodes: int
    num_triangles: int
    meta_radius: float | None
    center: tuple[float, float, float]
    center_offset: float
    naive_radius: float
    fitted_radius: float
    radial_spread: float
    avg_edge_length: float
    pitch_naive_deg: float
    pitch_true_deg: float


def _parse_meta_radius(aux: dict[str, str]) -> float | None:
    """Read the zone's Aux SphereRadius as a positive float.

    Args:
        aux: Zone auxiliary key/value map from the PLT header.

    Returns:
        The metadata radius, or None when absent, unparsable, or non-positive.
    """
    raw = aux.get("SphereRadius")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def inspect_atom_spheres(
    plt_path: str,
    target_atom_num: int | None = None,
) -> list[SphereFrame]:
    """Measure the sphere frame of every AtomSphereData zone in a .plt file.

    Args:
        plt_path: Path to the Tecplot .plt file.
        target_atom_num: When given, restrict the scan to that atom number.

    Returns:
        One SphereFrame per matching zone, ordered by atom number.

    Raises:
        FileNotFoundError: If plt_path does not exist.
        RuntimeError: If the file carries no AtomSphereData zones.
    """
    abs_plt = os.path.abspath(plt_path)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"PLT file not found: {abs_plt}")

    gba_mb, gba_meta = extract_gba_zones_from_plt(
        abs_plt,
        output_vtm=None,
        include_sphere_patches=False,
        include_surfaces=False,
        include_atom_spheres=True,
    )

    frames: list[SphereFrame] = []
    for entry in gba_meta:
        if entry.get("zone_type") != "AtomSphereData":
            continue
        block_idx = entry.get("block_index")
        if block_idx is None:
            continue

        atom_number = int(str(entry.get("atom_number", 0)) or 0)
        if target_atom_num is not None and atom_number != target_atom_num:
            continue

        poly = gba_mb.GetBlock(int(block_idx))
        if not isinstance(poly, vtkPolyData) or poly.GetNumberOfPoints() == 0:
            continue

        pts: npt.NDArray[np.float64] = numpy_support.vtk_to_numpy(
            poly.GetPoints().GetData()
        )
        n_cells = int(poly.GetNumberOfPolys())
        if n_cells == 0:
            continue
        cell_arr = numpy_support.vtk_to_numpy(poly.GetPolys().GetData())
        triangles: npt.NDArray[np.int32] = cell_arr.reshape((n_cells, 4))[:, 1:4]

        center = pts.mean(axis=0)
        rel = pts - center
        rel_norms = np.linalg.norm(rel, axis=1)
        naive_radius = float(np.mean(np.linalg.norm(pts, axis=1)))
        fitted_radius = float(np.mean(rel_norms))

        edges = extract_unique_edges(triangles)
        avg_edge = compute_average_edge_length(pts, edges)
        _, pitch_naive_deg = compute_mesh_pitch(avg_edge, naive_radius)
        _, pitch_true_deg = compute_mesh_pitch(avg_edge, fitted_radius)

        frames.append(
            SphereFrame(
                atom_number=atom_number,
                atom_symbol=str(entry.get("atom_type", "?")),
                zone_name=str(entry.get("name", "")),
                num_nodes=int(poly.GetNumberOfPoints()),
                num_triangles=n_cells,
                meta_radius=_parse_meta_radius(dict(entry.get("aux", {}))),
                center=(float(center[0]), float(center[1]), float(center[2])),
                center_offset=float(np.linalg.norm(center)),
                naive_radius=naive_radius,
                fitted_radius=fitted_radius,
                radial_spread=float(rel_norms.max() - rel_norms.min()),
                avg_edge_length=avg_edge,
                pitch_naive_deg=pitch_naive_deg,
                pitch_true_deg=pitch_true_deg,
            )
        )

    if not frames:
        raise RuntimeError(f"No AtomSphereData zones found in {plt_path}")
    return sorted(frames, key=lambda f: f.atom_number)


def _report(plt_path: str, frames: list[SphereFrame]) -> None:
    """Print a per-atom diagnostic block and a summary table.

    Args:
        plt_path: The inspected file, used for the banner.
        frames: Measured frames to report.
    """
    name = os.path.basename(plt_path)
    print("=" * 96)
    print(f"ATOM SPHERE FRAMES: {name}  ({len(frames)} sphere(s))")
    print("=" * 96)

    for f in frames:
        if f.meta_radius is not None:
            meta = f"{f.meta_radius:.6f}"
            delta = abs(f.meta_radius - f.fitted_radius) / f.fitted_radius * 100.0
            delta_txt = f"(metadata delta {delta:.3f}%)"
        else:
            meta = " (absent)"
            delta_txt = "(metadata delta    n/a)"
        infl = f.naive_radius / f.fitted_radius
        dist = f.pitch_naive_deg / f.pitch_true_deg
        print(
            f"\nAtom {f.atom_number:>2}  {f.atom_symbol:<2}  '{f.zone_name}'\n"
            f"  nodes {f.num_nodes}   triangles {f.num_triangles}\n"
            f"  center (x_center)       = "
            f"({f.center[0]:+.6f}, {f.center[1]:+.6f}, {f.center[2]:+.6f})\n"
            f"  ||x_center|| (offset)   = {f.center_offset:.6f} A\n"
            f"  Aux SphereRadius        = {meta} A\n"
            f"  mean(||x - x_center||)  = {f.fitted_radius:.6f} A   {delta_txt}\n"
            f"  mean(||x||) naive R     = {f.naive_radius:.6f} A"
            f"   (inflation x{infl:.3f})\n"
            f"  radial spread           = {f.radial_spread:.6f} A\n"
            f"  avg edge length         = {f.avg_edge_length:.6f} A\n"
            f"  delta_theta naive       = {f.pitch_naive_deg:.4f} deg\n"
            f"  delta_theta true        = {f.pitch_true_deg:.4f} deg"
            f"   (distortion x{dist:.3f})"
        )

    print("\n" + "-" * 96)
    print(
        f"{'atom':>4} {'sym':>3} {'offset':>9} {'R_meta':>9} {'R_fit':>9} "
        f"{'R_naive':>9} {'infl':>7} {'dTh_nai':>8} {'dTh_tru':>8} {'dist':>7}"
    )
    for f in frames:
        meta_val = f.meta_radius if f.meta_radius is not None else float("nan")
        print(
            f"{f.atom_number:>4} {f.atom_symbol:>3} {f.center_offset:>9.5f} "
            f"{meta_val:>9.5f} {f.fitted_radius:>9.5f} {f.naive_radius:>9.5f} "
            f"{f.naive_radius / f.fitted_radius:>6.2f}x "
            f"{f.pitch_naive_deg:>8.4f} {f.pitch_true_deg:>8.4f} "
            f"{f.pitch_naive_deg / f.pitch_true_deg:>6.2f}x"
        )
    print("=" * 96)


def main() -> None:
    """CLI entrypoint for the atom sphere frame inspector."""
    parser = argparse.ArgumentParser(
        prog="inspect_atom_spheres.py",
        description="Report the true center, radius and pitch of each "
        "AtomSphereData zone in a Tecplot .plt file.",
    )
    parser.add_argument("plt_file", help="Input Tecplot .plt file")
    parser.add_argument(
        "-a",
        "--atom",
        type=int,
        default=None,
        help="Restrict the scan to a single atom number",
    )
    args, _unknown = parser.parse_known_args()

    frames = inspect_atom_spheres(args.plt_file, target_atom_num=args.atom)
    _report(args.plt_file, frames)


if __name__ == "__main__":
    main()
