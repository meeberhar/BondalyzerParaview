#!/usr/bin/env python3
"""
Stage 0: Isolated Atom Sphere Condensed Field Visualizer.

Standalone module for inspecting condensed scalar fields mapped onto an isolated
atomic bounding sphere (e.g., C1), including:
1. Continuous / stepped colormap flood
2. Selectable level-set isocontours
3. Reference baseline basin boundaries extracted from the PLT file

Run with:
    uv run python -m gba_topology.stage0_sphere_view ethene4.plt
or:
    uv run python gba_topology/stage0_sphere_view.py ethene4.plt
"""

import os
import sys
import math
import argparse
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# VTK Imports
import vtk
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray, vtkStringArray, vtkIdList
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkTriangle, vtkCellArray
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkConeSource, vtkCubeSource
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkFeatureEdges, vtkTubeFilter, vtkPolyDataNormals, vtkGlyph3D
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkPolyDataMapper,
    vtkActor,
    vtkColorTransferFunction,
    vtkCoordinate,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.util import numpy_support

# Add workspace parent dir to sys.path so we can import plt_gba_to_vtm
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from bondalyzer_viewer.plt_gba_to_vtm import extract_gba_zones_from_plt
from gba_topology.topology_engine import analyze_spherical_topology
from bondalyzer_viewer.trame_viewer import (
    normalize_field_name,
    matches_field,
    get_display_title,
    get_robust_scalar_bounds,
    get_field_slider_config,
    order_primary_secondary,
    GBA_DISTINCT_PALETTE,
    assign_neighbor_aware_basin_colors,
)

# Trame Imports
try:
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import vuetify3 as v3
    from trame.widgets import html
    from trame.widgets.vtk import VtkRemoteView
    TRAME_AVAILABLE = True
except ImportError:
    try:
        from trame.app import get_server
        from trame.ui.vuetify import SinglePageWithDrawerLayout
        from trame.widgets import vuetify as v3
        from trame.widgets import html
        from trame.widgets.vtk import VtkRemoteView
        TRAME_AVAILABLE = True
    except ImportError:
        TRAME_AVAILABLE = False


def extract_single_atom_sphere_data(plt_path: str, target_atom_num: int = 1):
    """
    Extract the AtomSphereData mesh and reference CondensedBasinSphere patches for a specific atom.
    Returns (sphere_poly, basin_patches_list, available_fields_list, atom_info_dict).
    """
    abs_plt = os.path.abspath(plt_path)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"PLT file not found: {abs_plt}")

    gba_mb, gba_meta = extract_gba_zones_from_plt(
        abs_plt,
        output_vtm=None,
        include_sphere_patches=True,
        include_surfaces=False,
        include_atom_spheres=True,
    )

    sphere_poly = None
    basin_patches = []
    atom_symbol = "C"

    for entry in gba_meta:
        ztype = entry.get("zone_type")
        block_idx = entry.get("block_index")
        poly = gba_mb.GetBlock(block_idx)
        if not poly or poly.GetNumberOfPoints() == 0:
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
        elif ztype == "CondensedBasinSphere":
            basin_patches.append({
                "meta": entry,
                "poly": poly,
            })

    if sphere_poly is None:
        raise RuntimeError(f"Could not find AtomSphereData for Atom #{target_atom_num} in {plt_path}")

    # Discover available condensed fields on PointData (strictly exclude 3D volumetric fields)
    pd = sphere_poly.GetPointData()
    discovered_fields = []
    for i in range(pd.GetNumberOfArrays()):
        aname = pd.GetArrayName(i)
        if not aname:
            continue
        lower_a = aname.lower().strip()
        # Strictly keep only condensed surface fields
        is_condensed = (
            "(condensed)" in lower_a
            or lower_a in ("v", "v (condensed)", "î±", "α", "alpha", "trajectory parameter")
        )
        is_ignored = aname in ("X", "Y", "Z", "RGBColor", "Normals", "atomic_number", "AtomicNumber")

        if is_condensed and not is_ignored:
            if aname not in discovered_fields:
                discovered_fields.append(aname)

    if not discovered_fields:
        discovered_fields = ["Electron Density (condensed)", "Ï\x81 mean curvature (condensed)", "Ï\x81 Gaussian curvature (condensed)"]

    ordered_fields = order_primary_secondary(discovered_fields)
    field_items = [{"title": get_display_title(f), "value": f} for f in ordered_fields]

    atom_info = {
        "atom_number": target_atom_num,
        "atom_symbol": atom_symbol,
        "atom_name": f"{atom_symbol}{target_atom_num}",
        "num_nodes": sphere_poly.GetNumberOfPoints(),
        "num_triangles": sphere_poly.GetNumberOfPolys(),
        "total_basins": len(basin_patches),
    }

    return sphere_poly, basin_patches, field_items, atom_info


def build_discrete_colormap(n_colors: int, f_min: float, f_max: float, is_log: bool):
    """Build stepped or smooth Viridis transfer function."""
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

        if is_log:
            pos_min = max(f_min, 1e-4)
            pos_max = max(f_max, pos_min * 10.0)
            val = float(10 ** (math.log10(pos_min) + frac * (math.log10(pos_max) - math.log10(pos_min))))
        else:
            val = f_min + frac * (f_max - f_min)

        ctf.AddRGBPoint(val, r, g, b)
    return ctf


def create_stage0_pipeline(sphere_poly: vtkPolyData, basin_patches: List[Dict[str, Any]]):
    """
    Construct Stage 0 isolated sphere rendering pipeline with color flood, contours, and reference basin boundaries.
    """
    renderer = vtkRenderer()
    renderer.SetBackground(0.12, 0.13, 0.16)
    renderer.SetBackground2(0.20, 0.22, 0.26)
    renderer.SetGradientBackground(True)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 750)
    render_window.SetWindowName("Stage 0: Isolated Atom Sphere Field Sandbox")
    render_window.SetOffScreenRendering(1)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.Initialize()

    # Orientation axes in upper right
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
    contour_actor.GetProperty().SetLineWidth(2.0)
    contour_actor.GetProperty().SetLighting(False)
    renderer.AddActor(contour_actor)

    # 3. Sphere Boundary Wireframe Actor
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

    # 4. Reference Basin Boundaries (Extracted from PLT CondensedBasinSphere patches)
    ref_boundary_actors = []
    for patch in basin_patches:
        poly_p = patch["poly"]
        edges = vtkFeatureEdges()
        edges.SetInputData(poly_p)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.ManifoldEdgesOff()

        tuber = vtkTubeFilter()
        tuber.SetInputConnection(edges.GetOutputPort())
        tuber.SetRadius(0.010)
        tuber.SetNumberOfSides(12)
        tuber.CappingOn()

        b_mapper = vtkPolyDataMapper()
        b_mapper.SetInputConnection(tuber.GetOutputPort())
        b_mapper.ScalarVisibilityOff()

        b_actor = vtkActor()
        b_actor.SetMapper(b_mapper)
        # Default bright cyan/gold for reference boundaries
        b_actor.GetProperty().SetColor(1.0, 0.85, 0.10)
        b_actor.GetProperty().SetAmbient(0.8)
        b_actor.GetProperty().SetDiffuse(0.2)
        b_actor.SetVisibility(True)
        renderer.AddActor(b_actor)

        ref_boundary_actors.append({
            "actor": b_actor,
            "meta": patch["meta"],
        })

    # 5. Critical Point Glyphs: Maxima (Red Cones/Spheres), Minima (Blue Spheres), Saddles (Yellow Cubes/Crosses)
    # 5A. Maxima Glyphs (Vivid Red Spheres)
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
    max_actor.GetProperty().SetColor(0.95, 0.15, 0.15)  # Vivid Red
    max_actor.GetProperty().SetAmbient(0.6)
    max_actor.GetProperty().SetDiffuse(0.6)
    max_actor.GetProperty().SetSpecular(0.5)
    max_actor.GetProperty().SetSpecularPower(30)
    max_actor.SetVisibility(True)
    renderer.AddActor(max_actor)

    # 5B. Minima Glyphs (Bright Blue Spheres)
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
    min_actor.GetProperty().SetColor(0.15, 0.55, 0.95)  # Bright Blue
    min_actor.GetProperty().SetAmbient(0.6)
    min_actor.GetProperty().SetDiffuse(0.6)
    min_actor.GetProperty().SetSpecular(0.5)
    min_actor.GetProperty().SetSpecularPower(30)
    min_actor.SetVisibility(True)
    renderer.AddActor(min_actor)

    # 5C. Saddle Glyphs (Bright Gold/Green Cubes)
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
    sad_actor.GetProperty().SetColor(0.20, 0.85, 0.35)  # Emerald / Bright Green
    sad_actor.GetProperty().SetAmbient(0.6)
    sad_actor.GetProperty().SetDiffuse(0.6)
    sad_actor.GetProperty().SetSpecular(0.5)
    sad_actor.GetProperty().SetSpecularPower(30)
    sad_actor.SetVisibility(True)
    renderer.AddActor(sad_actor)

    # 5D. Selected CP Highlight Wireframe Ring
    cp_highlight_src = vtkSphereSource()
    cp_highlight_src.SetRadius(0.075)
    cp_highlight_src.SetThetaResolution(16)
    cp_highlight_src.SetPhiResolution(16)

    cp_highlight_mapper = vtkPolyDataMapper()
    cp_highlight_mapper.SetInputConnection(cp_highlight_src.GetOutputPort())

    cp_highlight_actor = vtkActor()
    cp_highlight_actor.SetMapper(cp_highlight_mapper)
    cp_highlight_actor.GetProperty().SetColor(1.0, 0.95, 0.1)  # Vivid yellow highlight
    cp_highlight_actor.GetProperty().SetRepresentationToWireframe()
    cp_highlight_actor.GetProperty().SetLineWidth(3.0)
    cp_highlight_actor.SetVisibility(False)
    renderer.AddActor(cp_highlight_actor)

    renderer.ResetCamera()
    return (
        renderer,
        render_window,
        flood_mapper,
        flood_actor,
        contour_filter,
        contour_actor,
        sphere_wire_actor,
        ref_boundary_actors,
        maxima_pts,
        maxima_poly,
        max_actor,
        minima_pts,
        minima_poly,
        min_actor,
        saddles_pts,
        saddles_poly,
        sad_actor,
        cp_highlight_src,
        cp_highlight_actor,
        orientation_widget,
    )


def run_stage0_app(plt_file: str, atom_num: int = 1, port: Optional[int] = None, open_browser: bool = True):
    """
    Launch Stage 0 interactive Trame application for isolated atom sphere analysis.
    """
    (
        sphere_poly,
        basin_patches,
        field_items,
        atom_info,
    ) = extract_single_atom_sphere_data(plt_file, target_atom_num=atom_num)

    (
        renderer,
        render_window,
        flood_mapper,
        flood_actor,
        contour_filter,
        contour_actor,
        sphere_wire_actor,
        ref_boundary_actors,
        maxima_pts,
        maxima_poly,
        max_actor,
        minima_pts,
        minima_poly,
        min_actor,
        saddles_pts,
        saddles_poly,
        sad_actor,
        cp_highlight_src,
        cp_highlight_actor,
        orientation_widget,
    ) = create_stage0_pipeline(sphere_poly, basin_patches)

    server = get_server("stage0_sphere_sandbox")
    state, ctrl = server.state, server.controller

    def request_view_update():
        if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
            try:
                ctrl.view_update()
            except Exception:
                pass

    # Extract NumPy geometry for topological engine
    pts_np = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    polys = sphere_poly.GetPolys()
    polys.InitTraversal()
    id_list = vtkIdList()
    triangles_list = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            triangles_list.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
    triangles_np = np.array(triangles_list, dtype=np.int32)

    # Initial state
    default_field = field_items[0]["value"] if field_items else "Electron Density"
    state.plt_file = os.path.basename(plt_file)
    state.atom_info = atom_info
    state.field_items = field_items
    state.selected_field = default_field
    state.show_flood = True
    state.show_contours = True
    state.show_sphere_wire = False
    state.show_ref_boundaries = False
    state.show_min_boundaries = True
    state.show_max_boundaries = True
    state.num_contours = 15
    state.scale_type = "linear"
    state.active_ref_boundary_count = len(ref_boundary_actors)

    # Stage 1 Critical Point State
    state.show_cps = True
    state.show_maxima = True
    state.show_minima = True
    state.show_saddles = True
    state.persistence_threshold_pct = 1.0
    state.geodesic_min_dist = 0.08
    state.cp_results = {}
    state.selected_cp = None
    state.all_cps_list = []

    def update_critical_points():
        cur_field = state.selected_field
        pd = sphere_poly.GetPointData()

        target_arr_name = None
        for i in range(pd.GetNumberOfArrays()):
            aname = pd.GetArrayName(i)
            if aname and aname == cur_field:
                target_arr_name = aname
                break
        if not target_arr_name:
            for i in range(pd.GetNumberOfArrays()):
                aname = pd.GetArrayName(i)
                if aname and matches_field(aname, cur_field):
                    target_arr_name = aname
                    break

        if not target_arr_name or not pd.HasArray(target_arr_name):
            max_actor.SetVisibility(False)
            min_actor.SetVisibility(False)
            sad_actor.SetVisibility(False)
            state.cp_results = {}
            state.all_cps_list = []
            return

        f_vals_np = numpy_support.vtk_to_numpy(pd.GetArray(target_arr_name))
        res = analyze_spherical_topology(
            pts_np,
            triangles_np,
            f_vals_np,
            persistence_threshold_pct=float(state.persistence_threshold_pct),
            geodesic_min_dist=float(state.geodesic_min_dist),
        )

        state.cp_results = res

        # Update VTK point containers for glyphs
        # 1. Maxima
        maxima_pts.Reset()
        for mx in res["maxima"]:
            p = mx["position"]
            maxima_pts.InsertNextPoint(p[0], p[1], p[2])
        maxima_poly.SetPoints(maxima_pts)
        maxima_poly.Modified()
        max_actor.SetVisibility(bool(state.show_cps and state.show_maxima and len(res["maxima"]) > 0))

        # 2. Minima
        minima_pts.Reset()
        for m in res["minima"]:
            p = m["position"]
            minima_pts.InsertNextPoint(p[0], p[1], p[2])
        minima_poly.SetPoints(minima_pts)
        minima_poly.Modified()
        min_actor.SetVisibility(bool(state.show_cps and state.show_minima and len(res["minima"]) > 0))

        # 3. Saddles
        saddles_pts.Reset()
        for s in res["saddles"]:
            p = s["position"]
            saddles_pts.InsertNextPoint(p[0], p[1], p[2])
        saddles_poly.SetPoints(saddles_pts)
        saddles_poly.Modified()
        sad_actor.SetVisibility(bool(state.show_cps and state.show_saddles and len(res["saddles"]) > 0))

        # Flatten CP list for UI table
        combined = []
        for mx in res["maxima"]:
            combined.append({**mx, "badge_color": "error", "type_title": "Maximum"})
        for m in res["minima"]:
            combined.append({**m, "badge_color": "info", "type_title": "Minimum"})
        for s in res["saddles"]:
            combined.append({**s, "badge_color": "success", "type_title": "Saddle"})

        state.all_cps_list = sorted(combined, key=lambda x: -x.get("persistence", 0.0))

    def update_field_visualization():
        cur_field = state.selected_field
        pd = sphere_poly.GetPointData()

        # Resolve field array on sphere
        target_arr_name = None
        for i in range(pd.GetNumberOfArrays()):
            aname = pd.GetArrayName(i)
            if aname and aname == cur_field:
                target_arr_name = aname
                break
        if not target_arr_name:
            for i in range(pd.GetNumberOfArrays()):
                aname = pd.GetArrayName(i)
                if aname and matches_field(aname, cur_field):
                    target_arr_name = aname
                    break

        if target_arr_name and pd.HasArray(target_arr_name):
            arr = pd.GetArray(target_arr_name)
            f_min, f_max = get_robust_scalar_bounds(arr, lower_pct=2.0, upper_pct=98.0)
            n_levels = max(2, int(state.num_contours))
            is_log = (state.scale_type == "log")

            # 1. Update Color Flood
            if state.show_flood:
                ctf = build_discrete_colormap(n_levels, f_min, f_max, is_log)
                flood_mapper.SelectColorArray(target_arr_name)
                flood_mapper.SetLookupTable(ctf)
                flood_mapper.SetScalarRange(f_min, f_max)
                flood_actor.SetVisibility(True)
            else:
                flood_actor.SetVisibility(False)

            # 2. Update Contours
            if state.show_contours:
                contour_filter.SetInputArrayToProcess(0, 0, 0, 0, target_arr_name)
                if is_log:
                    pos_min = max(f_min, 1e-4)
                    pos_max = max(f_max, pos_min * 10.0)
                    log_vals = [
                        float(10 ** (math.log10(pos_min) + i * (math.log10(pos_max) - math.log10(pos_min)) / (n_levels - 1)))
                        for i in range(n_levels)
                    ]
                    contour_filter.SetNumberOfContours(n_levels)
                    for i, val in enumerate(log_vals):
                        contour_filter.SetValue(i, val)
                else:
                    step_lin = (f_max - f_min) / (n_levels - 1)
                    contour_filter.SetNumberOfContours(n_levels)
                    for i in range(n_levels):
                        contour_filter.SetValue(i, f_min + i * step_lin)
                contour_filter.Update()
                contour_actor.SetVisibility(True)
            else:
                contour_actor.SetVisibility(False)
        else:
            flood_actor.SetVisibility(False)
            contour_actor.SetVisibility(False)

        # 3. Sphere wireframe
        sphere_wire_actor.SetVisibility(bool(state.show_sphere_wire))

        # 4. Reference Basin Boundaries
        show_ref = bool(state.show_ref_boundaries)
        show_min = bool(state.show_min_boundaries)
        show_max = bool(state.show_max_boundaries)
        vis_count = 0

        for r_entry in ref_boundary_actors:
            meta = r_entry["meta"]
            reg = meta.get("region_type", "")
            fn = meta.get("function_name", "")

            is_match = matches_field(fn, cur_field)
            is_min = "minimum" in reg
            is_max = "maximum" in reg

            vis = False
            if show_ref and is_match:
                if (is_min and show_min) or (is_max and show_max):
                    vis = True

            r_entry["actor"].SetVisibility(vis)
            if vis:
                vis_count += 1

        state.active_ref_boundary_count = vis_count
        update_critical_points()
        render_window.Render()
        request_view_update()

    def select_cp_item(item: Optional[Dict[str, Any]]):
        """Highlight a clicked critical point in 3D and update state."""
        state.selected_cp = item
        if item is None or cp_highlight_actor is None or cp_highlight_src is None:
            if cp_highlight_actor is not None:
                cp_highlight_actor.SetVisibility(False)
        else:
            p = item["position"]
            cp_highlight_src.SetCenter(p[0], p[1], p[2])
            cp_highlight_src.Update()
            cp_highlight_actor.SetVisibility(True)
        render_window.Render()
        request_view_update()

    @state.change(
        "selected_field",
        "show_flood",
        "show_contours",
        "show_sphere_wire",
        "show_ref_boundaries",
        "show_min_boundaries",
        "show_max_boundaries",
        "num_contours",
        "scale_type",
        "show_cps",
        "show_maxima",
        "show_minima",
        "show_saddles",
        "persistence_threshold_pct",
        "geodesic_min_dist",
    )
    def on_param_change(**kwargs):
        update_field_visualization()

    @ctrl.add("select_cp_from_list")
    def select_cp_from_list(cp_label):
        for cp in state.all_cps_list:
            if cp.get("id_label") == cp_label:
                select_cp_item(cp)
                return

    @ctrl.add("clear_cp_selection")
    def clear_cp_selection():
        select_cp_item(None)

    @ctrl.add("reset_camera")
    def reset_camera():
        renderer.ResetCamera()
        render_window.Render()
        request_view_update()

    # Initial pipeline setup
    update_field_visualization()

    # Build UI Layout
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text(f"Stage 0: Isolated Atom Sphere ({atom_info['atom_name']})")
        layout.drawer.width = 380

        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-3"):

                # Atom Summary Card
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center"):
                            v3.VIcon("mdi-atom", classes="mr-2", color="success")
                            html.Span(f"Atom: {atom_info['atom_name']}")
                        v3.VCardSubtitle(f"File: {state.plt_file}")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div("Mesh Triangles", classes="text-caption text-medium-emphasis")
                                html.Div("{{ atom_info.num_triangles }}", classes="text-body-2 font-weight-bold")
                            with v3.VCol(cols=6):
                                html.Div("Mesh Vertices", classes="text-caption text-medium-emphasis")
                                html.Div("{{ atom_info.num_nodes }}", classes="text-body-2 font-weight-bold")

                # Field Selection
                with v3.VCard(elevation=2, classes="mb-3"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold"):
                            html.Span("1. Condensed Field on Sphere")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-3 pb-2"):
                        with v3.VSelect(
                            label="Select Condensed Field",
                            items=("field_items",),
                            v_model=("selected_field",),
                            density="compact",
                            variant="outlined",
                            classes="mb-2",
                        ):
                            with html.Template(v_slot_prepend_inner=True):
                                html.Span("F[ρ]", classes="font-italic font-weight-bold text-success mr-1")

                        # Color Flood Switch
                        v3.VSwitch(
                            label="Surface Color Flood",
                            v_model=("show_flood",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        # Wireframe Switch
                        v3.VSwitch(
                            label="Show Sphere Mesh Wireframe",
                            v_model=("show_sphere_wire",),
                            density="compact",
                            color="secondary",
                            hide_details=True,
                        )

                # Isocontours Configuration
                with v3.VCard(elevation=2, classes="mb-3"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold"):
                            html.Span("2. Level-Set Isocontours")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-3 pb-2"):
                        v3.VSwitch(
                            label="Enable Isocontour Lines",
                            v_model=("show_contours",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-2",
                        )

                        with html.Div(v_if="show_contours"):
                            html.Div("Contour Scaling Mode", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                            with v3.VBtnToggle(
                                v_model=("scale_type", "linear"),
                                density="compact",
                                color="primary",
                                mandatory=True,
                                classes="mb-2 d-flex justify-center",
                            ):
                                v3.VBtn("Linear", value="linear", size="small", prepend_icon="mdi-ruler")
                                v3.VBtn("Logarithmic", value="log", size="small", prepend_icon="mdi-math-log")

                            with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                html.Div("Contour Levels", classes="text-caption font-weight-bold text-medium-emphasis")
                                html.Div("{{ num_contours }} lines", classes="text-caption font-weight-bold text-primary")

                            v3.VSlider(
                                min=3,
                                max=40,
                                step=1,
                                v_model=("num_contours",),
                                density="compact",
                                thumb_label=False,
                                color="primary",
                                classes="mt-1",
                            )

                # Reference Basin Boundaries Card
                with v3.VCard(elevation=2, classes="mb-3"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"):
                            html.Span("3. Reference Basin Boundaries")
                            v3.VChip("{{ active_ref_boundary_count }} visible", size="x-small", color="warning")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Show Existing Basin Boundaries",
                            v_model=("show_ref_boundaries",),
                            density="compact",
                            color="warning",
                            hide_details=True,
                            classes="mb-1",
                        )
                        with html.Div(v_if="show_ref_boundaries"):
                            v3.VSwitch(
                                label="Minimum Basin Boundaries",
                                v_model=("show_min_boundaries",),
                                density="compact",
                                color="primary",
                                hide_details=True,
                                classes="mb-1",
                            )
                            v3.VSwitch(
                                label="Maximum Basin Boundaries",
                                v_model=("show_max_boundaries",),
                                density="compact",
                                color="secondary",
                                hide_details=True,
                            )

                # =============================================================
                # STAGE 1: GLOBAL TOPOLOGICAL CRITICAL POINTS & PERSISTENCE
                # =============================================================
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"):
                            html.Span("4. Stage 1: Critical Points (Morse)")
                            with html.Div(v_if="cp_results && cp_results.filtered_counts"):
                                v3.VChip(
                                    "Euler χ={{ cp_results.filtered_counts.euler }}",
                                    size="x-small",
                                    color="success" if ("cp_results.filtered_counts.euler_valid", True) else "error",
                                    classes="font-weight-bold",
                                )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Display Critical Point Glyphs",
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
                                        label="Maxima (Red)",
                                        v_model=("show_maxima",),
                                        density="compact",
                                        color="error",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Minima (Blue)",
                                        v_model=("show_minima",),
                                        density="compact",
                                        color="info",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Saddles (Green)",
                                        v_model=("show_saddles",),
                                        density="compact",
                                        color="success",
                                        hide_details=True,
                                    )

                            v3.VDivider(classes="my-2")

                            # Persistence Threshold Slider
                            with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                html.Div("Persistence Simplification Threshold", classes="text-caption font-weight-bold text-medium-emphasis")
                                html.Div("{{ Number(persistence_threshold_pct).toFixed(1) }}%", classes="text-caption font-weight-bold text-primary")

                            v3.VSlider(
                                min=0.0,
                                max=10.0,
                                step=0.1,
                                v_model=("persistence_threshold_pct",),
                                density="compact",
                                thumb_label=False,
                                color="primary",
                                classes="mt-1",
                            )

                            # Geodesic Clustering Distance Slider
                            with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                html.Div("Geodesic Min Separation", classes="text-caption font-weight-bold text-medium-emphasis")
                                html.Div("{{ Number(geodesic_min_dist).toFixed(3) }} Å", classes="text-caption font-weight-bold text-primary")

                            v3.VSlider(
                                min=0.0,
                                max=0.30,
                                step=0.01,
                                v_model=("geodesic_min_dist",),
                                density="compact",
                                thumb_label=False,
                                color="primary",
                                classes="mt-1",
                            )

                            # Critical Point Counts Summary
                            with html.Div(v_if="cp_results && cp_results.filtered_counts", classes="mt-2 pa-2 rounded surface"):
                                with v3.VRow(dense=True, classes="text-caption text-center"):
                                    with v3.VCol(cols=4):
                                        html.Div("Maxima", classes="text-medium-emphasis")
                                        html.Div("{{ cp_results.filtered_counts.maxima }}", classes="text-subtitle-2 font-weight-bold text-error")
                                    with v3.VCol(cols=4):
                                        html.Div("Minima", classes="text-medium-emphasis")
                                        html.Div("{{ cp_results.filtered_counts.minima }}", classes="text-subtitle-2 font-weight-bold text-info")
                                    with v3.VCol(cols=4):
                                        html.Div("Saddles", classes="text-medium-emphasis")
                                        html.Div("{{ cp_results.filtered_counts.saddles }}", classes="text-subtitle-2 font-weight-bold text-success")

                            # Critical Points Table / List
                            with html.Div(v_if="all_cps_list && all_cps_list.length > 0", classes="mt-2"):
                                html.Div("Detected Critical Points (click to highlight)", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                with v3.VTable(density="compact", classes="elevation-0", style="max-height: 200px; overflow-y: auto;"):
                                    with html.Thead():
                                        with html.Tr():
                                            html.Th("ID", classes="text-left text-caption font-weight-bold")
                                            html.Th("Type", classes="text-left text-caption font-weight-bold")
                                            html.Th("Value", classes="text-right text-caption font-weight-bold")
                                            html.Th("Pers %", classes="text-right text-caption font-weight-bold")
                                    with html.Tbody():
                                        with html.Tr(
                                            v_for="cp in all_cps_list",
                                            key="cp.id_label",
                                            click=(ctrl.select_cp_from_list, "[cp.id_label]"),
                                            classes="cursor-pointer",
                                        ):
                                            html.Td("{{ cp.id_label }}", classes="text-caption font-weight-bold")
                                            with html.Td():
                                                v3.VChip("{{ cp.type_title }}", size="x-small", color=("cp.badge_color",))
                                            html.Td("{{ cp.value.toFixed(5) }}", classes="text-right text-caption font-mono")
                                            html.Td("{{ cp.persistence_pct.toFixed(1) }}%", classes="text-right text-caption font-weight-bold text-primary")

                # Selected CP Inspection Card
                with v3.VCard(
                    v_if="selected_cp",
                    elevation=3,
                    classes="mb-3 border-primary",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon("mdi-crosshairs-gps", classes="mr-2", color="amber-darken-2")
                                html.Span("{{ selected_cp.id_label }} ({{ selected_cp.type_title }})")
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
                                html.Div("Scalar Value", classes="text-caption text-medium-emphasis")
                                html.Div("{{ selected_cp.value.toFixed(6) }}", classes="text-body-2 font-weight-bold")
                            with v3.VCol(cols=6):
                                html.Div("Persistence", classes="text-caption text-medium-emphasis")
                                html.Div("{{ selected_cp.persistence.toFixed(6) }} ({{ selected_cp.persistence_pct.toFixed(2) }}%)", classes="text-body-2 font-weight-bold text-primary")

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div("Coordinates (X, Y, Z)", classes="text-caption text-medium-emphasis")
                                html.Div("({{ selected_cp.position[0].toFixed(4) }}, {{ selected_cp.position[1].toFixed(4) }}, {{ selected_cp.position[2].toFixed(4) }})", classes="text-body-2 font-mono")

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

        # 3D Viewport
        with layout.content:
            with html.Div(style="position: relative; width: 100%; height: 100%;"):
                view = VtkRemoteView(render_window, interactive_ratio=1.0)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    print(f"\n[Stage 0 Sandbox] Starting Isolated Atom Sphere Viewer for {atom_info['atom_name']} from {plt_file}")
    server.start(port=port, open_browser=open_browser)


def main():
    parser = argparse.ArgumentParser(
        prog="stage0_sphere_view.py",
        description="Stage 0: Isolated Atom Sphere Condensed Field Visualizer for topological basin analysis.",
    )
    parser.add_argument("plt_file", nargs="?", default="ethene4.plt", help="Input Tecplot .plt file (default: ethene4.plt)")
    parser.add_argument("-a", "--atom", type=int, default=1, help="Target atom number to isolate (default: 1 for C1)")
    parser.add_argument("-p", "--port", type=int, default=None, help="Trame web server port")
    parser.add_argument("--server", action="store_true", help="Run without auto-opening browser")
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], args.plt_file] + unknown
    run_stage0_app(args.plt_file, atom_num=args.atom, port=args.port, open_browser=not args.server)


if __name__ == "__main__":
    main()
