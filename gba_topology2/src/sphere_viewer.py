"""Sphere Viewer Subroutine for gba_topology2.

Step 2 Interactive Viewer:
- Displays atom sphere, raw critical points (minima, maxima, saddles), and micro-cluster boundary rings.
- Interactive 3D Picking: Clicking directly on a cluster or CP in the 3D viewport or sidebar table
  immediately opens its inspector card displaying all constituent CPs, coordinates, scalar values,
  angular radius, and local Euler index.
- Clean separation from algorithmic subroutines: imports mesh_geometry, morse_detector, and micro_cluster.

Run with:
    uv run python -m gba_topology2.src.sphere_viewer Pd_20K.plt
or:
    uv run python gba_topology2/src/sphere_viewer.py Pd_20K.plt
"""

import argparse
import math
import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

# VTK Imports
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkColorTransferFunction,
    vtkCoordinate,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

# Trame Imports
try:
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import html
    from trame.widgets import vuetify3 as v3
    from trame.widgets.vtk import VtkRemoteView
except ImportError:
    from trame.app import get_server  # type: ignore
    from trame.ui.vuetify import SinglePageWithDrawerLayout  # type: ignore
    from trame.widgets import html  # type: ignore
    from trame.widgets import vuetify as v3  # type: ignore
    from trame.widgets.vtk import VtkRemoteView  # type: ignore

# Add workspace parent dir so imports resolve cleanly
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from BondalyzerParaView.plt_gba_to_vtm import extract_gba_zones_from_plt
from BondalyzerParaView.trame_viewer import get_display_title, get_robust_scalar_bounds
from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.micro_cluster import MicroCluster, compute_micro_clusters
from gba_topology2.src.morse_detector import CriticalPoint, compute_discrete_morse_cps

__all__ = ["extract_atom_sphere_mesh", "run_sphere_viewer"]


def extract_atom_sphere_mesh(
    plt_path: str, target_atom_num: int = 1
) -> tuple[vtkPolyData, list[dict[str, str]], dict[str, Any]]:
    """Extract the AtomSphereData mesh and condensed scalar fields from a .plt file.

    Args:
        plt_path: Path to the .plt file.
        target_atom_num: Atom number (1-based index).

    Returns:
        Tuple of (sphere_poly, field_items, atom_info).
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

    sphere_poly: vtkPolyData | None = None
    atom_symbol = "C"

    for entry in gba_meta:
        ztype = entry.get("zone_type")
        block_idx = entry.get("block_index")
        if block_idx is None:
            continue
        poly = gba_mb.GetBlock(int(block_idx))
        if not isinstance(poly, vtkPolyData) or poly.GetNumberOfPoints() == 0:
            continue

        atom_nr = entry.get("atom_number", 1)
        try:
            atom_nr = int(atom_nr)
        except (ValueError, TypeError):
            atom_nr = 1

        if atom_nr != target_atom_num:
            continue

        if ztype == "AtomSphereData":
            sphere_poly = poly
            atom_symbol = entry.get("atom_type", "C")
            break

    if sphere_poly is None:
        raise RuntimeError(
            f"Could not find AtomSphereData for Atom #{target_atom_num} in {plt_path}"
        )

    # Discover condensed surface fields
    pd = sphere_poly.GetPointData()
    discovered_fields: list[str] = []
    for i in range(pd.GetNumberOfArrays()):
        aname = pd.GetArrayName(i)
        if not aname:
            continue
        lower_a = aname.lower().strip()
        is_condensed = "(condensed)" in lower_a or lower_a in (
            "v",
            "v (condensed)",
            "î±",
            "α",
            "alpha",
            "trajectory parameter",
        )
        is_ignored = aname in (
            "X",
            "Y",
            "Z",
            "RGBColor",
            "Normals",
            "atomic_number",
            "AtomicNumber",
        )
        if is_condensed and not is_ignored and aname not in discovered_fields:
            discovered_fields.append(aname)

    if not discovered_fields:
        discovered_fields = ["Electron Density (condensed)"]

    field_items = [
        {"title": get_display_title(f), "value": f} for f in discovered_fields
    ]
    atom_nodes = int(sphere_poly.GetNumberOfPoints())
    num_tris = int(sphere_poly.GetNumberOfPolys())

    pts_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    mean_r = float(np.mean(np.linalg.norm(pts_arr, axis=1))) if atom_nodes > 0 else 1.0

    atom_info = {
        "atom_number": target_atom_num,
        "atom_symbol": atom_symbol,
        "atom_name": f"{atom_symbol}{target_atom_num}",
        "num_nodes": atom_nodes,
        "num_triangles": num_tris,
        "sphere_radius": mean_r,
    }

    return sphere_poly, field_items, atom_info


def build_discrete_colormap(
    n_colors: int, f_min: float, f_max: float
) -> vtkColorTransferFunction:
    """Build smooth Viridis transfer function."""
    ctf = vtkColorTransferFunction()
    anchors = [
        (0.00, 0.267, 0.004, 0.329),  # dark purple
        (0.25, 0.190, 0.407, 0.556),  # dark blue
        (0.50, 0.127, 0.566, 0.550),  # teal
        (0.75, 0.369, 0.788, 0.382),  # bright green
        (1.00, 0.993, 0.906, 0.143),  # bright yellow
    ]
    n_steps = max(2, min(n_colors, 64))
    for step in range(n_steps):
        frac = step / (n_steps - 1)
        for a_i in range(len(anchors) - 1):
            f0, r0, g0, b0 = anchors[a_i]
            f1, r1, g1, b1 = anchors[a_i + 1]
            if f0 <= frac <= f1:
                t = (frac - f0) / (f1 - f0) if f1 > f0 else 0.0
                r = r0 + t * (r1 - r0)
                g = g0 + t * (g1 - g0)
                b = b0 + t * (b1 - b0)
                break
        else:
            r, g, b = anchors[-1][1:]

        val = f_min + frac * (f_max - f_min)
        ctf.AddRGBPoint(val, r, g, b)
    return ctf


def run_sphere_viewer(
    plt_file: str,
    atom_num: int = 1,
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Launch Step 2 sphere viewer application for a given atom and PLT file."""
    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        plt_file, target_atom_num=atom_num
    )

    pts_np: npt.NDArray[np.float64] = numpy_support.vtk_to_numpy(
        sphere_poly.GetPoints().GetData()
    )
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    triangles_np: npt.NDArray[np.int32] = cell_arr.reshape((n_cells, 4))[:, 1:4]

    geom_info = analyze_mesh_geometry(pts_np, triangles_np)
    sphere_radius = geom_info["sphere_radius"]
    delta_theta_rad = geom_info["mesh_pitch_rad"]

    # Renderer and Window
    renderer = vtkRenderer()
    renderer.SetBackground(0.12, 0.13, 0.16)
    renderer.SetBackground2(0.20, 0.22, 0.26)
    renderer.SetGradientBackground(True)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 750)
    render_window.SetWindowName(
        f"gba_topology2 - Sphere Viewer: {atom_info['atom_name']}"
    )
    render_window.SetOffScreenRendering(1)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.Initialize()

    # Orientation axes
    axes_actor = vtkAxesActor()
    axes_actor.SetShaftTypeToCylinder()
    axes_actor.SetCylinderRadius(0.03)
    axes_actor.SetTotalLength(1.0, 1.0, 1.0)
    axes_actor.SetNormalizedShaftLength(0.75, 0.75, 0.75)
    axes_actor.SetNormalizedTipLength(0.25, 0.25, 0.25)
    for cap_actor in (
        axes_actor.GetXAxisCaptionActor2D(),
        axes_actor.GetYAxisCaptionActor2D(),
        axes_actor.GetZAxisCaptionActor2D(),
    ):
        cap_actor.GetTextActor().SetTextScaleModeToNone()
        cap_prop = cap_actor.GetCaptionTextProperty()
        cap_prop.SetFontSize(16)
        cap_prop.BoldOn()
        cap_prop.ShadowOn()

    orientation_widget = vtkOrientationMarkerWidget()
    orientation_widget.SetOrientationMarker(axes_actor)
    orientation_widget.SetInteractor(interactor)
    orientation_widget.SetViewport(0.72, 0.72, 0.98, 0.98)
    orientation_widget.SetEnabled(1)
    orientation_widget.InteractiveOff()

    # 1. Sphere Surface Flood Actor
    flood_mapper = vtkPolyDataMapper()
    flood_mapper.SetInputData(sphere_poly)
    flood_mapper.SetScalarModeToUsePointFieldData()
    flood_mapper.ScalarVisibilityOn()

    flood_actor = vtkActor()
    flood_actor.SetMapper(flood_mapper)
    flood_actor.GetProperty().SetAmbient(0.35)
    flood_actor.GetProperty().SetDiffuse(0.75)
    flood_actor.GetProperty().SetSpecular(0.40)
    flood_actor.GetProperty().SetSpecularPower(30)
    renderer.AddActor(flood_actor)

    # 2. Sphere Isocontours Actor
    contour_filter = vtkContourFilter()
    contour_filter.SetInputData(sphere_poly)

    contour_mapper = vtkPolyDataMapper()
    contour_mapper.SetInputConnection(contour_filter.GetOutputPort())
    contour_mapper.ScalarVisibilityOff()

    contour_actor = vtkActor()
    contour_actor.SetMapper(contour_mapper)
    contour_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    contour_actor.GetProperty().SetLineWidth(1.8)
    contour_actor.GetProperty().SetLighting(False)
    renderer.AddActor(contour_actor)

    # 3. Sphere Wireframe Actor
    sphere_wire_mapper = vtkPolyDataMapper()
    sphere_wire_mapper.SetInputData(sphere_poly)
    sphere_wire_mapper.ScalarVisibilityOff()

    sphere_wire_actor = vtkActor()
    sphere_wire_actor.SetMapper(sphere_wire_mapper)
    sphere_wire_actor.GetProperty().SetRepresentationToWireframe()
    sphere_wire_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
    sphere_wire_actor.GetProperty().SetOpacity(0.20)
    sphere_wire_actor.GetProperty().SetLineWidth(1.0)
    sphere_wire_actor.SetVisibility(False)
    renderer.AddActor(sphere_wire_actor)

    # 4. Critical Point Glyphs
    # Maxima (Red Spheres)
    maxima_pts = vtkPoints()
    maxima_poly = vtkPolyData()
    maxima_poly.SetPoints(maxima_pts)

    max_sphere_src = vtkSphereSource()
    max_sphere_src.SetRadius(0.045)
    max_sphere_src.SetThetaResolution(16)
    max_sphere_src.SetPhiResolution(16)

    max_glyph = vtkGlyph3D()
    max_glyph.SetSourceConnection(max_sphere_src.GetOutputPort())
    max_glyph.SetInputData(maxima_poly)
    max_glyph.ScalingOff()

    max_mapper = vtkPolyDataMapper()
    max_mapper.SetInputConnection(max_glyph.GetOutputPort())
    max_mapper.ScalarVisibilityOff()

    max_actor = vtkActor()
    max_actor.SetMapper(max_mapper)
    max_actor.GetProperty().SetColor(0.95, 0.15, 0.15)
    max_actor.GetProperty().SetAmbient(0.6)
    max_actor.GetProperty().SetDiffuse(0.6)
    renderer.AddActor(max_actor)

    # Minima (Blue Spheres)
    minima_pts = vtkPoints()
    minima_poly = vtkPolyData()
    minima_poly.SetPoints(minima_pts)

    min_sphere_src = vtkSphereSource()
    min_sphere_src.SetRadius(0.045)
    min_sphere_src.SetThetaResolution(16)
    min_sphere_src.SetPhiResolution(16)

    min_glyph = vtkGlyph3D()
    min_glyph.SetSourceConnection(min_sphere_src.GetOutputPort())
    min_glyph.SetInputData(minima_poly)
    min_glyph.ScalingOff()

    min_mapper = vtkPolyDataMapper()
    min_mapper.SetInputConnection(min_glyph.GetOutputPort())
    min_mapper.ScalarVisibilityOff()

    min_actor = vtkActor()
    min_actor.SetMapper(min_mapper)
    min_actor.GetProperty().SetColor(0.15, 0.55, 0.95)
    min_actor.GetProperty().SetAmbient(0.6)
    min_actor.GetProperty().SetDiffuse(0.6)
    renderer.AddActor(min_actor)

    # Saddles (Green Cubes)
    saddles_pts = vtkPoints()
    saddles_poly = vtkPolyData()
    saddles_poly.SetPoints(saddles_pts)

    sad_cube_src = vtkCubeSource()
    sad_cube_src.SetXLength(0.065)
    sad_cube_src.SetYLength(0.065)
    sad_cube_src.SetZLength(0.065)

    sad_glyph = vtkGlyph3D()
    sad_glyph.SetSourceConnection(sad_cube_src.GetOutputPort())
    sad_glyph.SetInputData(saddles_poly)
    sad_glyph.ScalingOff()

    sad_mapper = vtkPolyDataMapper()
    sad_mapper.SetInputConnection(sad_glyph.GetOutputPort())
    sad_mapper.ScalarVisibilityOff()

    sad_actor = vtkActor()
    sad_actor.SetMapper(sad_mapper)
    sad_actor.GetProperty().SetColor(0.20, 0.85, 0.35)
    sad_actor.GetProperty().SetAmbient(0.6)
    sad_actor.GetProperty().SetDiffuse(0.6)
    renderer.AddActor(sad_actor)

    # 5. Micro-Cluster Boundary Rings (Tubes on sphere surface)
    cluster_rings_poly = vtkPolyData()
    cluster_rings_tubes = vtkTubeFilter()
    cluster_rings_tubes.SetInputData(cluster_rings_poly)
    cluster_rings_tubes.SetRadius(0.010)
    cluster_rings_tubes.SetNumberOfSides(12)
    cluster_rings_tubes.CappingOn()

    cluster_rings_mapper = vtkPolyDataMapper()
    cluster_rings_mapper.SetInputConnection(cluster_rings_tubes.GetOutputPort())
    cluster_rings_mapper.ScalarVisibilityOff()

    cluster_rings_actor = vtkActor()
    cluster_rings_actor.SetMapper(cluster_rings_mapper)
    cluster_rings_actor.GetProperty().SetColor(0.05, 0.05, 0.05)  # Crisp black rings
    cluster_rings_actor.GetProperty().SetAmbient(0.9)
    cluster_rings_actor.GetProperty().SetDiffuse(0.1)
    renderer.AddActor(cluster_rings_actor)

    # 6. Selection Highlight Actor
    highlight_src = vtkSphereSource()
    highlight_src.SetRadius(0.08)
    highlight_src.SetThetaResolution(18)
    highlight_src.SetPhiResolution(18)

    highlight_mapper = vtkPolyDataMapper()
    highlight_mapper.SetInputConnection(highlight_src.GetOutputPort())

    highlight_actor = vtkActor()
    highlight_actor.SetMapper(highlight_mapper)
    highlight_actor.GetProperty().SetColor(1.0, 0.95, 0.1)  # Vivid yellow
    highlight_actor.GetProperty().SetRepresentationToWireframe()
    highlight_actor.GetProperty().SetLineWidth(3.0)
    highlight_actor.SetVisibility(False)
    renderer.AddActor(highlight_actor)

    # 7. Cluster Member Highlight Rings (Cyan wireframe spheres)
    cluster_members_pts = vtkPoints()
    cluster_members_poly = vtkPolyData()
    cluster_members_poly.SetPoints(cluster_members_pts)

    cluster_member_src = vtkSphereSource()
    cluster_member_src.SetRadius(0.065)
    cluster_member_src.SetThetaResolution(16)
    cluster_member_src.SetPhiResolution(16)

    cluster_members_glyph = vtkGlyph3D()
    cluster_members_glyph.SetSourceConnection(cluster_member_src.GetOutputPort())
    cluster_members_glyph.SetInputData(cluster_members_poly)
    cluster_members_glyph.ScalingOff()

    cluster_members_mapper = vtkPolyDataMapper()
    cluster_members_mapper.SetInputConnection(cluster_members_glyph.GetOutputPort())
    cluster_members_mapper.ScalarVisibilityOff()

    cluster_members_actor = vtkActor()
    cluster_members_actor.SetMapper(cluster_members_mapper)
    cluster_members_actor.GetProperty().SetColor(0.2, 0.9, 1.0)  # Cyan
    cluster_members_actor.GetProperty().SetRepresentationToWireframe()
    cluster_members_actor.GetProperty().SetLineWidth(2.5)
    cluster_members_actor.SetVisibility(False)
    renderer.AddActor(cluster_members_actor)

    renderer.ResetCamera()

    # Trame Server Setup
    server = get_server("sphere_viewer_step2")
    state, ctrl = server.state, server.controller

    def request_view_update() -> None:
        if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
            try:
                ctrl.view_update()
            except (RuntimeError, Exception):
                pass

    # Initial state
    default_field = field_items[0]["value"] if field_items else "Electron Density"
    state.plt_file = os.path.basename(plt_file)
    state.atom_info = atom_info
    state.geom_info = {
        "num_vertices": geom_info["num_vertices"],
        "num_triangles": geom_info["num_triangles"],
        "num_edges": geom_info["num_edges"],
        "sphere_radius": float(geom_info["sphere_radius"]),
        "avg_edge_length": float(geom_info["avg_edge_length"]),
        "mesh_pitch_deg": float(geom_info["mesh_pitch_deg"]),
    }
    state.field_items = field_items
    state.selected_field = default_field
    state.show_flood = True
    state.show_contours = True
    state.show_sphere_wire = False
    state.num_contours = 15

    # Step 1 & 2 Topological State
    state.show_cps = True
    state.show_maxima = True
    state.show_minima = True
    state.show_saddles = True
    state.k_pitch = 2.0  # Threshold multiplier k * delta_theta_mesh
    state.cp_glyph_scale = 1.0
    state.show_rings = True

    # Computed lists & selection
    state.morse_counts = {
        "minima": 0,
        "maxima": 0,
        "saddles": 0,
        "euler": 2,
        "valid": True,
    }
    state.all_cps_list = []
    state.clusters_list = []
    state.selected_cp = None
    state.selected_cluster = None

    # Cached in memory for 3D picking ray-cast
    runtime_cps: list[CriticalPoint] = []
    runtime_clusters: list[MicroCluster] = []

    def update_topology() -> None:
        nonlocal runtime_cps, runtime_clusters
        cur_field = state.selected_field
        pd = sphere_poly.GetPointData()
        arr = pd.GetArray(cur_field)
        if not arr:
            return

        f_vals_np: npt.NDArray[np.float64] = numpy_support.vtk_to_numpy(arr)

        # 1. Run Morse Detector (Subroutine 2)
        morse_res = compute_discrete_morse_cps(
            pts_np,
            triangles_np,
            f_vals_np,
            tie_break_epsilon=1e-15,
        )

        state.morse_counts = {
            "minima": len(morse_res.minima),
            "maxima": len(morse_res.maxima),
            "saddles": len(morse_res.saddles),
            "euler": morse_res.euler_characteristic,
            "valid": morse_res.is_valid_euler,
        }

        # Update Glyph Points
        maxima_pts.Reset()
        for mx in morse_res.maxima:
            p = mx.position
            maxima_pts.InsertNextPoint(p[0], p[1], p[2])
        maxima_poly.SetPoints(maxima_pts)
        maxima_poly.Modified()
        max_actor.SetVisibility(bool(state.show_cps and state.show_maxima))

        minima_pts.Reset()
        for mn in morse_res.minima:
            p = mn.position
            minima_pts.InsertNextPoint(p[0], p[1], p[2])
        minima_poly.SetPoints(minima_pts)
        minima_poly.Modified()
        min_actor.SetVisibility(bool(state.show_cps and state.show_minima))

        saddles_pts.Reset()
        for sd in morse_res.saddles:
            p = sd.position
            saddles_pts.InsertNextPoint(p[0], p[1], p[2])
        saddles_poly.SetPoints(saddles_pts)
        saddles_poly.Modified()
        sad_actor.SetVisibility(bool(state.show_cps and state.show_saddles))

        # 2. Run Micro-Cluster (Subroutine 3)
        all_raw_cps = morse_res.minima + morse_res.maxima + morse_res.saddles
        runtime_cps = all_raw_cps
        k_val = float(state.k_pitch)
        clusters = compute_micro_clusters(
            cps=all_raw_cps,
            delta_theta_mesh_rad=delta_theta_rad,
            k_pitch=k_val,
            sphere_radius=sphere_radius,
            min_boundary_margin_deg=1.5,
            num_boundary_points=48,
        )
        runtime_clusters = clusters

        # Build VTK geometry for cluster boundary rings
        ring_pts = vtkPoints()
        ring_lines = vtkCellArray()
        pt_counter = 0

        multi_clusters = [cl for cl in clusters if cl.is_multi_cp]
        for cl in multi_clusters:
            ring = cl.boundary_ring
            n_r = len(ring)
            if n_r < 3:
                continue

            for rp in ring:
                # Offset boundary ring slightly outwards to prevent z-fighting with sphere
                norm_p = math.sqrt(
                    rp.position[0] ** 2 + rp.position[1] ** 2 + rp.position[2] ** 2
                )
                factor = 1.008 if norm_p > 1e-6 else 1.0
                ring_pts.InsertNextPoint(
                    rp.position[0] * factor,
                    rp.position[1] * factor,
                    rp.position[2] * factor,
                )

            ring_lines.InsertNextCell(n_r + 1)
            for k in range(n_r):
                ring_lines.InsertCellPoint(pt_counter + k)
            ring_lines.InsertCellPoint(pt_counter)  # Close the ring
            pt_counter += n_r

        cluster_rings_poly.SetPoints(ring_pts)
        cluster_rings_poly.SetLines(ring_lines)
        cluster_rings_poly.Modified()
        cluster_rings_actor.SetVisibility(bool(state.show_rings and pt_counter > 0))

        # Flatten CP list for UI
        cps_ui: list[dict[str, Any]] = []
        for mx in morse_res.maxima:
            cps_ui.append(
                {
                    "vertex_id": mx.vertex_id,
                    "type": "maximum",
                    "type_title": "Maximum",
                    "badge_color": "error",
                    "value": mx.value,
                    "position": list(mx.position),
                    "multiplicity": mx.multiplicity,
                    "id_label": f"MAX_{mx.vertex_id}",
                }
            )
        for mn in morse_res.minima:
            cps_ui.append(
                {
                    "vertex_id": mn.vertex_id,
                    "type": "minimum",
                    "type_title": "Minimum",
                    "badge_color": "info",
                    "value": mn.value,
                    "position": list(mn.position),
                    "multiplicity": mn.multiplicity,
                    "id_label": f"MIN_{mn.vertex_id}",
                }
            )
        for sd in morse_res.saddles:
            cps_ui.append(
                {
                    "vertex_id": sd.vertex_id,
                    "type": "saddle",
                    "type_title": "Saddle",
                    "badge_color": "success",
                    "value": sd.value,
                    "position": list(sd.position),
                    "multiplicity": sd.multiplicity,
                    "id_label": f"SAD_{sd.vertex_id}",
                }
            )

        state.all_cps_list = sorted(cps_ui, key=lambda x: -x["value"])

        # Format clusters list for UI
        clusters_ui: list[dict[str, Any]] = []
        for cl in clusters:
            clusters_ui.append(
                {
                    "id": cl.cluster_id,
                    "size": len(cl.members),
                    "composition_summary": cl.composition_summary,
                    "local_euler_index": cl.local_euler_index,
                    "is_multi_cp": cl.is_multi_cp,
                    "angular_radius_deg": cl.angular_radius_deg,
                    "angular_diameter_deg": cl.angular_diameter_deg,
                    "spatial_diameter": cl.spatial_diameter,
                    "centroid": list(cl.centroid),
                    "centroid_unit": list(cl.centroid_unit),
                    "members": [
                        {
                            "vertex_id": m.vertex_id,
                            "type": m.cp_type,
                            "value": m.value,
                            "position": list(m.position),
                            "id_label": f"{m.cp_type[:3].upper()}_{m.vertex_id}",
                        }
                        for m in cl.members
                    ],
                }
            )
        state.clusters_list = clusters_ui

    def update_field() -> None:
        cur_field = state.selected_field
        pd = sphere_poly.GetPointData()
        arr = pd.GetArray(cur_field)
        if arr:
            f_min, f_max = get_robust_scalar_bounds(arr, lower_pct=2.0, upper_pct=98.0)
            n_levels = max(2, int(state.num_contours))

            if state.show_flood:
                ctf = build_discrete_colormap(n_levels, f_min, f_max)
                flood_mapper.SelectColorArray(cur_field)
                flood_mapper.SetLookupTable(ctf)
                flood_mapper.SetScalarRange(f_min, f_max)
                flood_actor.SetVisibility(True)
            else:
                flood_actor.SetVisibility(False)

            if state.show_contours:
                contour_filter.SetInputArrayToProcess(0, 0, 0, 0, cur_field)
                step_lin = (f_max - f_min) / (n_levels - 1)
                contour_filter.SetNumberOfContours(n_levels)
                for i in range(n_levels):
                    contour_filter.SetValue(i, f_min + i * step_lin)
                contour_filter.Update()
                contour_actor.SetVisibility(True)
            else:
                contour_actor.SetVisibility(False)

        sphere_wire_actor.SetVisibility(bool(state.show_sphere_wire))
        update_topology()
        render_window.Render()
        request_view_update()

    def select_cp(cp_item: dict[str, Any] | None) -> None:
        state.selected_cp = cp_item
        if cp_item is None:
            highlight_actor.SetVisibility(False)
        else:
            p = cp_item["position"]
            scale = float(state.cp_glyph_scale)
            highlight_src.SetRadius(0.075 * scale)
            highlight_src.SetCenter(p[0], p[1], p[2])
            highlight_src.Update()
            highlight_actor.SetVisibility(True)
        render_window.Render()
        request_view_update()

    def select_cluster(cluster_item: dict[str, Any] | None) -> None:
        state.selected_cluster = cluster_item
        if cluster_item is None:
            cluster_members_pts.Reset()
            cluster_members_poly.SetPoints(cluster_members_pts)
            cluster_members_poly.Modified()
            cluster_members_actor.SetVisibility(False)
            highlight_actor.SetVisibility(False)
        else:
            c_pos = cluster_item["centroid"]
            highlight_src.SetRadius(
                max(float(cluster_item.get("spatial_diameter", 0.08)) * 0.75, 0.08)
            )
            highlight_src.SetCenter(c_pos[0], c_pos[1], c_pos[2])
            highlight_src.Update()
            highlight_actor.SetVisibility(True)

            # Highlight all individual member CPs in cyan rings
            cluster_members_pts.Reset()
            for mem in cluster_item.get("members", []):
                m_pos = mem["position"]
                cluster_members_pts.InsertNextPoint(m_pos[0], m_pos[1], m_pos[2])
            cluster_members_poly.SetPoints(cluster_members_pts)
            cluster_members_poly.Modified()
            cluster_members_actor.SetVisibility(
                len(cluster_item.get("members", [])) > 0
            )

        render_window.Render()
        request_view_update()

    # 3D Direct Picking Ray-Caster
    picker = vtkCellPicker()
    picker.SetTolerance(0.02)
    world_coord = vtkCoordinate()
    world_coord.SetCoordinateSystemToWorld()

    @ctrl.add("on_scene_click")
    def on_scene_click(
        click_x: float | None = None,
        click_y: float | None = None,
        client_w: float | None = None,
        client_h: float | None = None,
    ) -> None:
        try:
            if click_x is None or click_y is None:
                return

            rw_size = render_window.GetSize()
            w, h = rw_size[0], rw_size[1]
            cw = float(client_w) if client_w else float(w)
            ch = float(client_h) if client_h else float(h)
            norm_x = float(click_x) / cw
            norm_y = float(click_y) / ch
            disp_x = norm_x * w
            disp_y = (1.0 - norm_y) * h

            # Ray-pick in 3D world coordinates
            picker.Pick(disp_x, disp_y, 0, renderer)
            picked_actor = picker.GetActor()

            if picked_actor is not None:
                pick_pos = picker.GetPickPosition()
                # Check closest cluster centroid
                min_cluster_dist = float("inf")
                best_cluster: MicroCluster | None = None
                for cl in runtime_clusters:
                    c_pos = cl.centroid
                    d = math.sqrt(
                        (c_pos[0] - pick_pos[0]) ** 2
                        + (c_pos[1] - pick_pos[1]) ** 2
                        + (c_pos[2] - pick_pos[2]) ** 2
                    )
                    if d < min_cluster_dist:
                        min_cluster_dist = d
                        best_cluster = cl

                # Check closest CP
                min_cp_dist = float("inf")
                best_cp: CriticalPoint | None = None
                for cp in runtime_cps:
                    p = cp.position
                    d = math.sqrt(
                        (p[0] - pick_pos[0]) ** 2
                        + (p[1] - pick_pos[1]) ** 2
                        + (p[2] - pick_pos[2]) ** 2
                    )
                    if d < min_cp_dist:
                        min_cp_dist = d
                        best_cp = cp

                # If clicked close to a cluster centroid (within spatial radius or 0.15 A)
                if best_cluster is not None and min_cluster_dist <= max(
                    best_cluster.spatial_diameter, 0.15
                ):
                    for cl_dict in state.clusters_list:
                        if cl_dict["id"] == best_cluster.cluster_id:
                            select_cluster(cl_dict)
                            return

                # Otherwise check CP
                if best_cp is not None and min_cp_dist <= 0.15:
                    for cp_dict in state.all_cps_list:
                        if cp_dict["vertex_id"] == best_cp.vertex_id:
                            select_cp(cp_dict)
                            return

            # Screen-space proximity fallback
            min_screen_dist = float("inf")
            screen_cluster_dict: dict[str, Any] | None = None
            for cl_dict in state.clusters_list:
                c_pos = cl_dict["centroid"]
                world_coord.SetValue(c_pos[0], c_pos[1], c_pos[2])
                item_disp = world_coord.GetComputedDisplayValue(renderer)
                sdx = item_disp[0] - disp_x
                sdy = item_disp[1] - disp_y
                dist = math.sqrt(sdx * sdx + sdy * sdy)
                if dist < min_screen_dist:
                    min_screen_dist = dist
                    screen_cluster_dict = cl_dict

            if screen_cluster_dict is not None and min_screen_dist <= 35.0:
                select_cluster(screen_cluster_dict)
                return

            # Clear selection if background clicked
            select_cluster(None)
            select_cp(None)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"[sphere_viewer] Picking error: {e}")

    @state.change(
        "selected_field",
        "show_flood",
        "show_contours",
        "show_sphere_wire",
        "num_contours",
        "show_cps",
        "show_maxima",
        "show_minima",
        "show_saddles",
        "k_pitch",
        "cp_glyph_scale",
        "show_rings",
    )
    def on_param_change(**kwargs: Any) -> None:
        update_field()

    @ctrl.add("select_cp_from_list")
    def select_cp_from_list(cp_id: str) -> None:
        for cp in state.all_cps_list:
            if cp.get("id_label") == cp_id:
                select_cp(cp)
                return

    @ctrl.add("clear_cp_selection")
    def clear_cp_selection() -> None:
        select_cp(None)

    @ctrl.add("select_cluster_from_list")
    def select_cluster_from_list(cl_id: str) -> None:
        for cl in state.clusters_list:
            if cl.get("id") == cl_id:
                select_cluster(cl)
                return

    @ctrl.add("clear_cluster_selection")
    def clear_cluster_selection() -> None:
        select_cluster(None)

    @ctrl.add("reset_camera")
    def reset_camera() -> None:
        renderer.ResetCamera()
        render_window.Render()
        request_view_update()

    # Initial pipeline update
    update_field()

    # UI Layout
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text(f"gba_topology2: Atom Sphere ({atom_info['atom_name']})")
        layout.drawer.width = 400

        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-3"):
                # Atom & Mesh Geometry Card (Step 1 metrics)
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center"
                        ):
                            v3.VIcon("mdi-atom", classes="mr-2", color="success")
                            html.Span(f"Atom: {atom_info['atom_name']}")
                        v3.VCardSubtitle(f"File: {state.plt_file}")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Mesh Vertices",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.num_vertices }}",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Mesh Triangles",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.num_triangles }}",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Avg Edge Length (l_e)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.avg_edge_length.toFixed(4) }} Å",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Mesh Pitch (δθ)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.mesh_pitch_deg.toFixed(2) }}°",
                                    classes="text-body-2 font-weight-bold text-success",
                                )

                # Field Controls
                with v3.VCard(elevation=2, classes="mb-3"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold"):
                            html.Span("Condensed Field & Mesh")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-3 pb-2"):
                        v3.VSelect(
                            label="Select Field",
                            items=("field_items",),
                            v_model=("selected_field",),
                            density="compact",
                            variant="outlined",
                            classes="mb-2",
                        )

                        v3.VSwitch(
                            label="Surface Color Flood",
                            v_model=("show_flood",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        v3.VSwitch(
                            label="Isocontour Lines",
                            v_model=("show_contours",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        v3.VSwitch(
                            label="Show Mesh Wireframe",
                            v_model=("show_sphere_wire",),
                            density="compact",
                            color="secondary",
                            hide_details=True,
                        )

                # Step 1: Discrete Morse Detection
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 1: Morse Detection (τ = 0.0)")
                            v3.VChip(
                                "Euler χ={{ morse_counts.euler }}",
                                size="x-small",
                                color="success"
                                if ("morse_counts.valid", True)
                                else "error",
                                classes="font-weight-bold",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Display Critical Points",
                            v_model=("show_cps",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        with html.Div(v_if="show_cps"):
                            with v3.VRow(dense=True, classes="align-center mb-1"):
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Max (Red)",
                                        v_model=("show_maxima",),
                                        density="compact",
                                        color="error",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Min (Blue)",
                                        v_model=("show_minima",),
                                        density="compact",
                                        color="info",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Sad (Green)",
                                        v_model=("show_saddles",),
                                        density="compact",
                                        color="success",
                                        hide_details=True,
                                    )

                            with html.Div(classes="mt-2 pa-2 rounded surface"):
                                with v3.VRow(
                                    dense=True, classes="text-caption text-center"
                                ):
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Maxima", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ morse_counts.maxima }}",
                                            classes="text-subtitle-2 font-weight-bold text-error",
                                        )
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Minima", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ morse_counts.minima }}",
                                            classes="text-subtitle-2 font-weight-bold text-info",
                                        )
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Saddles", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ morse_counts.saddles }}",
                                            classes="text-subtitle-2 font-weight-bold text-success",
                                        )

                # Step 2: Micro-Clustering & Interactive 3D Picking
                with v3.VCard(elevation=2, classes="mb-3"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 2: Micro-Clusters")
                            v3.VChip(
                                "{{ clusters_list.filter(c => c.is_multi_cp).length }} multi-CP",
                                size="x-small",
                                color="warning",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Show Cluster Boundary Rings",
                            v_model=("show_rings",),
                            density="compact",
                            color="warning",
                            hide_details=True,
                            classes="mb-1",
                        )

                        with html.Div(
                            classes="d-flex justify-space-between align-center mt-2"
                        ):
                            html.Div(
                                "Cluster Pitch Multiplier (k · δθ)",
                                classes="text-caption font-weight-bold text-medium-emphasis",
                            )
                            html.Div(
                                "{{ Number(k_pitch).toFixed(1) }}x ({{ (k_pitch * geom_info.mesh_pitch_deg).toFixed(1) }}°)",
                                classes="text-caption font-weight-bold text-warning",
                            )

                        v3.VSlider(
                            min=1.0,
                            max=5.0,
                            step=0.2,
                            v_model=("k_pitch",),
                            density="compact",
                            thumb_label=False,
                            color="warning",
                            classes="mt-1",
                        )

                        # Clusters table
                        html.Div(
                            "Micro-Clusters (Click in 3D or list to inspect)",
                            classes="text-caption font-weight-bold text-medium-emphasis mt-2 mb-1",
                        )
                        with v3.VTable(
                            density="compact",
                            classes="elevation-0",
                            style="max-height: 180px; overflow-y: auto;",
                        ):
                            with html.Thead():
                                with html.Tr():
                                    html.Th(
                                        "ID",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "Composition",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "χ",
                                        classes="text-right text-caption font-weight-bold",
                                    )
                            with html.Tbody():
                                with html.Tr(
                                    v_for="cl in clusters_list",
                                    key="cl.id",
                                    click=(ctrl.select_cluster_from_list, "[cl.id]"),
                                    classes="cursor-pointer",
                                ):
                                    html.Td(
                                        "{{ cl.id }} ({{ cl.size }})",
                                        classes="text-caption font-weight-bold",
                                    )
                                    with html.Td():
                                        v3.VChip(
                                            "{{ cl.composition_summary }}",
                                            size="x-small",
                                            color="amber-darken-3"
                                            if ("cl.is_multi_cp", True)
                                            else "default",
                                        )
                                    html.Td(
                                        "{{ cl.local_euler_index >= 0 ? '+' + cl.local_euler_index : cl.local_euler_index }}",
                                        classes="text-right text-caption font-mono",
                                    )

                # Inspector Card: Selected Micro-Cluster
                with v3.VCard(
                    v_if="selected_cluster",
                    elevation=3,
                    classes="mb-3 border-warning",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-chart-bubble", classes="mr-2", color="warning"
                                )
                                html.Span(
                                    "Cluster {{ selected_cluster.id }} ({{ selected_cluster.size }} CPs)"
                                )
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_cluster_selection,
                            )
                        v3.VCardSubtitle(
                            "Composition: {{ selected_cluster.composition_summary }} | Net Index: {{ selected_cluster.local_euler_index >= 0 ? '+' + selected_cluster.local_euler_index : selected_cluster.local_euler_index }}"
                        )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Angular Radius / Diam",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cluster.angular_radius_deg.toFixed(1) }}° / {{ selected_cluster.angular_diameter_deg.toFixed(1) }}°",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Spatial Diam",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cluster.spatial_diameter.toFixed(4) }} Å",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div(
                                    "Centroid Unit Vector (X, Y, Z)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "({{ selected_cluster.centroid_unit[0].toFixed(3) }}, {{ selected_cluster.centroid_unit[1].toFixed(3) }}, {{ selected_cluster.centroid_unit[2].toFixed(3) }})",
                                    classes="text-caption font-mono",
                                )

                        html.Div(
                            "Constituent Critical Points",
                            classes="text-caption font-weight-bold text-medium-emphasis mt-2 mb-1",
                        )
                        with v3.VList(density="compact", classes="pa-0 bg-transparent"):
                            with v3.VListItem(
                                v_for="mem in selected_cluster.members",
                                key="mem.id_label",
                                click=(ctrl.select_cp_from_list, "[mem.id_label]"),
                                classes="px-1 py-0 cursor-pointer",
                            ):
                                with html.Template(v_slot_prepend=True):
                                    v3.VChip(
                                        "{{ mem.id_label }}",
                                        size="x-small",
                                        color="primary",
                                        classes="mr-2",
                                    )
                                html.Span(
                                    "{{ mem.type }} | Val: {{ mem.value.toFixed(5) }}",
                                    classes="text-caption",
                                )

                # Inspector Card: Selected Critical Point
                with v3.VCard(
                    v_if="selected_cp",
                    elevation=3,
                    classes="mb-3 border-primary",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-crosshairs-gps",
                                    classes="mr-2",
                                    color="amber-darken-2",
                                )
                                html.Span("{{ selected_cp.id_label }}")
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_cp_selection,
                            )
                        v3.VCardSubtitle("Vertex Index: {{ selected_cp.vertex_id }}")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Type", classes="text-caption text-medium-emphasis"
                                )
                                v3.VChip(
                                    "{{ selected_cp.type_title }}",
                                    size="x-small",
                                    color=("selected_cp.badge_color",),
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Scalar Value",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cp.value.toFixed(6) }}",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div(
                                    "Coordinates (X, Y, Z)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "({{ selected_cp.position[0].toFixed(4) }}, {{ selected_cp.position[1].toFixed(4) }}, {{ selected_cp.position[2].toFixed(4) }})",
                                    classes="text-body-2 font-mono",
                                )

        # Toolbar
        with layout.toolbar:
            v3.VSpacer()
            v3.VBtn(
                "Reset Camera",
                prepend_icon="mdi-camera-flip-outline",
                click=ctrl.reset_camera,
                variant="tonal",
                density="compact",
                color="primary",
            )

        # 3D Viewport with direct clicking
        with layout.content:
            with html.Div(
                style="position: relative; width: 100%; height: 100%; cursor: pointer;",
                click=(
                    ctrl.on_scene_click,
                    "[$event.offsetX, $event.offsetY, $event.currentTarget.clientWidth, $event.currentTarget.clientHeight]",
                ),
            ):
                view = VtkRemoteView(render_window, interactive_ratio=1.0)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    print(
        f"\n[gba_topology2] Starting Sphere Viewer for {atom_info['atom_name']} from {plt_file}"
    )
    server.start(port=port, open_browser=open_browser)


def main() -> None:
    """CLI entrypoint for sphere_viewer."""
    parser = argparse.ArgumentParser(
        prog="sphere_viewer.py",
        description="gba_topology2 Step 2: Isolated Atom Sphere & Micro-Cluster Interactive Viewer",
    )
    parser.add_argument(
        "plt_file",
        nargs="?",
        default="Pd_20K.plt",
        help="Input Tecplot .plt file (default: Pd_20K.plt)",
    )
    parser.add_argument(
        "-a",
        "--atom",
        type=int,
        default=1,
        help="Target atom number to isolate (default: 1 for Pd1)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=None, help="Trame web server port"
    )
    parser.add_argument(
        "--server", action="store_true", help="Run without auto-opening browser"
    )
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], args.plt_file] + unknown
    run_sphere_viewer(
        args.plt_file,
        atom_num=args.atom,
        port=args.port,
        open_browser=not args.server,
    )


if __name__ == "__main__":
    main()
