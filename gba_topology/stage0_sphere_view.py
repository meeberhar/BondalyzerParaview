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

# Add workspace parent dir to sys.path so we can import modules
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from BondalyzerParaView.plt_gba_to_vtm import extract_gba_zones_from_plt
    from BondalyzerParaView.trame_viewer import (
        normalize_field_name,
        matches_field,
        get_display_title,
        get_robust_scalar_bounds,
        get_field_slider_config,
        order_primary_secondary,
        GBA_DISTINCT_PALETTE,
        assign_neighbor_aware_basin_colors,
    )
except ImportError:
    from bondalyzer_viewer.plt_gba_to_vtm import extract_gba_zones_from_plt
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
from gba_topology.topology_engine import analyze_spherical_topology, compute_effective_critical_points

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

    atom_nodes = sphere_poly.GetNumberOfPoints()
    pts_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    mean_r = float(np.mean(np.linalg.norm(pts_arr, axis=1))) if atom_nodes > 0 else 1.0

    atom_info = {
        "atom_number": target_atom_num,
        "atom_symbol": atom_symbol,
        "atom_name": f"{atom_symbol}{target_atom_num}",
        "num_nodes": atom_nodes,
        "num_triangles": sphere_poly.GetNumberOfPolys(),
        "total_basins": len(basin_patches),
        "sphere_radius": mean_r,
    }

    return sphere_poly, basin_patches, field_items, atom_info


def build_spherical_circle_polydata(
    centroid: np.ndarray,
    sphere_radius: float,
    angular_radius_deg: float,
    num_pts: int = 64,
    offset_factor: float = 1.006,
) -> vtkPolyData:
    """
    Generate a 3D closed polygonal line circle conformally lying on the sphere surface.
    Offset slightly outwards by offset_factor to prevent z-fighting with the sphere surface.
    """
    c_norm = np.linalg.norm(centroid)
    if c_norm < 1e-8:
        c_unit = np.array([0.0, 0.0, 1.0])
    else:
        c_unit = centroid / c_norm

    # Find two orthonormal vectors tangent to c_unit
    if abs(c_unit[0]) < 0.8 and abs(c_unit[1]) < 0.8:
        v_temp = np.array([1.0, 0.0, 0.0])
    else:
        v_temp = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(c_unit, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(c_unit, t1)
    t2 /= np.linalg.norm(t2)

    ang_rad = math.radians(max(angular_radius_deg, 0.2))
    sin_a = math.sin(ang_rad)
    cos_a = math.cos(ang_rad)

    r_effective = sphere_radius * offset_factor

    pts = vtkPoints()
    cells = vtkCellArray()
    cells.InsertNextCell(num_pts + 1)

    for i in range(num_pts):
        phi = 2.0 * math.pi * i / num_pts
        dir_vec = cos_a * c_unit + sin_a * (math.cos(phi) * t1 + math.sin(phi) * t2)
        dir_vec /= np.linalg.norm(dir_vec)
        p = dir_vec * r_effective
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        cells.InsertCellPoint(i)

    cells.InsertCellPoint(0)  # Close the loop

    circle_pd = vtkPolyData()
    circle_pd.SetPoints(pts)
    circle_pd.SetLines(cells)
    return circle_pd


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

    # 5E. Multi-CP Cluster Outline Circles (Thick Black Outlines conforming to sphere)
    cluster_circles_poly = vtkPolyData()
    cluster_circles_tubes = vtkTubeFilter()
    cluster_circles_tubes.SetInputData(cluster_circles_poly)
    cluster_circles_tubes.SetRadius(0.012)
    cluster_circles_tubes.SetNumberOfSides(12)
    cluster_circles_tubes.CappingOn()

    cluster_circles_mapper = vtkPolyDataMapper()
    cluster_circles_mapper.SetInputConnection(cluster_circles_tubes.GetOutputPort())
    cluster_circles_mapper.ScalarVisibilityOff()

    cluster_actor = vtkActor()
    cluster_actor.SetMapper(cluster_circles_mapper)
    cluster_actor.GetProperty().SetColor(0.05, 0.05, 0.05)  # Solid Black
    cluster_actor.GetProperty().SetAmbient(0.9)
    cluster_actor.GetProperty().SetDiffuse(0.1)
    cluster_actor.SetVisibility(False)
    renderer.AddActor(cluster_actor)

    # 5F. Selected Cluster Member Highlight Actors (Multiple cyan rings for all members)
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
    cluster_members_actor.GetProperty().SetColor(0.2, 0.9, 1.0)  # Cyan highlight rings
    cluster_members_actor.GetProperty().SetRepresentationToWireframe()
    cluster_members_actor.GetProperty().SetLineWidth(2.5)
    cluster_members_actor.SetVisibility(False)
    renderer.AddActor(cluster_members_actor)

    # 6. Effective Critical Point (ECP) Glyphs (Distinct slightly larger with white diamond/halo border)
    # 6A. Effective Maxima (Red spheres with halo)
    ecp_max_pts = vtkPoints()
    ecp_max_poly = vtkPolyData()
    ecp_max_poly.SetPoints(ecp_max_pts)

    ecp_max_src = vtkSphereSource()
    ecp_max_src.SetRadius(0.060)
    ecp_max_src.SetThetaResolution(18)
    ecp_max_src.SetPhiResolution(18)

    ecp_max_glyph = vtkGlyph3D()
    ecp_max_glyph.SetSourceConnection(ecp_max_src.GetOutputPort())
    ecp_max_glyph.SetInputData(ecp_max_poly)
    ecp_max_glyph.ScalingOff()

    ecp_max_mapper = vtkPolyDataMapper()
    ecp_max_mapper.SetInputConnection(ecp_max_glyph.GetOutputPort())
    ecp_max_mapper.ScalarVisibilityOff()

    ecp_max_actor = vtkActor()
    ecp_max_actor.SetMapper(ecp_max_mapper)
    ecp_max_actor.GetProperty().SetColor(1.0, 0.10, 0.10)
    ecp_max_actor.GetProperty().SetAmbient(0.7)
    ecp_max_actor.GetProperty().SetDiffuse(0.5)
    ecp_max_actor.SetVisibility(False)
    renderer.AddActor(ecp_max_actor)

    # 6B. Effective Minima (Blue spheres with halo)
    ecp_min_pts = vtkPoints()
    ecp_min_poly = vtkPolyData()
    ecp_min_poly.SetPoints(ecp_min_pts)

    ecp_min_src = vtkSphereSource()
    ecp_min_src.SetRadius(0.060)
    ecp_min_src.SetThetaResolution(18)
    ecp_min_src.SetPhiResolution(18)

    ecp_min_glyph = vtkGlyph3D()
    ecp_min_glyph.SetSourceConnection(ecp_min_src.GetOutputPort())
    ecp_min_glyph.SetInputData(ecp_min_poly)
    ecp_min_glyph.ScalingOff()

    ecp_min_mapper = vtkPolyDataMapper()
    ecp_min_mapper.SetInputConnection(ecp_min_glyph.GetOutputPort())
    ecp_min_mapper.ScalarVisibilityOff()

    ecp_min_actor = vtkActor()
    ecp_min_actor.SetMapper(ecp_min_mapper)
    ecp_min_actor.GetProperty().SetColor(0.10, 0.50, 1.0)
    ecp_min_actor.GetProperty().SetAmbient(0.7)
    ecp_min_actor.GetProperty().SetDiffuse(0.5)
    ecp_min_actor.SetVisibility(False)
    renderer.AddActor(ecp_min_actor)

    # 6C. Effective Saddles (Green cubes with halo)
    ecp_sad_pts = vtkPoints()
    ecp_sad_poly = vtkPolyData()
    ecp_sad_poly.SetPoints(ecp_sad_pts)

    ecp_sad_src = vtkCubeSource()
    ecp_sad_src.SetXLength(0.085)
    ecp_sad_src.SetYLength(0.085)
    ecp_sad_src.SetZLength(0.085)

    ecp_sad_glyph = vtkGlyph3D()
    ecp_sad_glyph.SetSourceConnection(ecp_sad_src.GetOutputPort())
    ecp_sad_glyph.SetInputData(ecp_sad_poly)
    ecp_sad_glyph.ScalingOff()

    ecp_sad_mapper = vtkPolyDataMapper()
    ecp_sad_mapper.SetInputConnection(ecp_sad_glyph.GetOutputPort())
    ecp_sad_mapper.ScalarVisibilityOff()

    ecp_sad_actor = vtkActor()
    ecp_sad_actor.SetMapper(ecp_sad_mapper)
    ecp_sad_actor.GetProperty().SetColor(0.10, 0.90, 0.35)
    ecp_sad_actor.GetProperty().SetAmbient(0.7)
    ecp_sad_actor.GetProperty().SetDiffuse(0.5)
    ecp_sad_actor.SetVisibility(False)
    renderer.AddActor(ecp_sad_actor)

    # 7. Excision Boundary Ports (Ridge Maxima = Orange Spheres, Valley Minima = Cyan Spheres)
    # 7A. Ridge Maxima Ports (Reduced size ~50%)
    port_max_pts = vtkPoints()
    port_max_poly = vtkPolyData()
    port_max_poly.SetPoints(port_max_pts)

    port_max_src = vtkSphereSource()
    port_max_src.SetRadius(0.018)  # Reduced ~50%
    port_max_src.SetThetaResolution(14)
    port_max_src.SetPhiResolution(14)

    port_max_glyph = vtkGlyph3D()
    port_max_glyph.SetSourceConnection(port_max_src.GetOutputPort())
    port_max_glyph.SetInputData(port_max_poly)
    port_max_glyph.ScalingOff()

    port_max_mapper = vtkPolyDataMapper()
    port_max_mapper.SetInputConnection(port_max_glyph.GetOutputPort())
    port_max_mapper.ScalarVisibilityOff()

    port_max_actor = vtkActor()
    port_max_actor.SetMapper(port_max_mapper)
    port_max_actor.GetProperty().SetColor(1.0, 0.55, 0.0)  # Orange
    port_max_actor.GetProperty().SetAmbient(0.8)
    port_max_actor.GetProperty().SetDiffuse(0.4)
    port_max_actor.SetVisibility(False)
    renderer.AddActor(port_max_actor)

    # 7B. Valley Minima Ports (Reduced size ~50%)
    port_min_pts = vtkPoints()
    port_min_poly = vtkPolyData()
    port_min_poly.SetPoints(port_min_pts)

    port_min_src = vtkSphereSource()
    port_min_src.SetRadius(0.018)  # Reduced ~50%
    port_min_src.SetThetaResolution(14)
    port_min_src.SetPhiResolution(14)

    port_min_glyph = vtkGlyph3D()
    port_min_glyph.SetSourceConnection(port_min_src.GetOutputPort())
    port_min_glyph.SetInputData(port_min_poly)
    port_min_glyph.ScalingOff()

    port_min_mapper = vtkPolyDataMapper()
    port_min_mapper.SetInputConnection(port_min_glyph.GetOutputPort())
    port_min_mapper.ScalarVisibilityOff()

    port_min_actor = vtkActor()
    port_min_actor.SetMapper(port_min_mapper)
    port_min_actor.GetProperty().SetColor(0.0, 0.90, 0.90)  # Cyan
    port_min_actor.GetProperty().SetAmbient(0.8)
    port_min_actor.GetProperty().SetDiffuse(0.4)
    port_min_actor.SetVisibility(False)
    renderer.AddActor(port_min_actor)

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
        max_sphere_src,
        min_sphere_src,
        sad_cube_src,
        cluster_circles_poly,
        cluster_circles_tubes,
        cluster_actor,
        cluster_members_pts,
        cluster_members_poly,
        cluster_member_src,
        cluster_members_actor,
        ecp_max_pts,
        ecp_max_poly,
        ecp_max_src,
        ecp_max_actor,
        ecp_min_pts,
        ecp_min_poly,
        ecp_min_src,
        ecp_min_actor,
        ecp_sad_pts,
        ecp_sad_poly,
        ecp_sad_src,
        ecp_sad_actor,
        port_max_pts,
        port_max_poly,
        port_max_src,
        port_max_actor,
        port_min_pts,
        port_min_poly,
        port_min_src,
        port_min_actor,
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
        max_sphere_src,
        min_sphere_src,
        sad_cube_src,
        cluster_circles_poly,
        cluster_circles_tubes,
        cluster_actor,
        cluster_members_pts,
        cluster_members_poly,
        cluster_member_src,
        cluster_members_actor,
        ecp_max_pts,
        ecp_max_poly,
        ecp_max_src,
        ecp_max_actor,
        ecp_min_pts,
        ecp_min_poly,
        ecp_min_src,
        ecp_min_actor,
        ecp_sad_pts,
        ecp_sad_poly,
        ecp_sad_src,
        ecp_sad_actor,
        port_max_pts,
        port_max_poly,
        port_max_src,
        port_max_actor,
        port_min_pts,
        port_min_poly,
        port_min_src,
        port_min_actor,
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
    sphere_radius = atom_info.get("sphere_radius", 1.0)

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

    # Stage 1 Critical Point State (Angular Clustering & Separation)
    state.show_cps = True
    state.show_maxima = True
    state.show_minima = True
    state.show_saddles = True
    state.persistence_threshold_pct = 1.0
    state.cluster_radius_deg = 10.0  # Cluster recognition radius (degrees)
    state.monkey_saddle_fuse_deg = 16.0  # Saddle-saddle fusion sensitivity angle (degrees)
    state.cp_glyph_scale = 1.0
    state.show_cluster_halos = True
    state.enable_harmonic_nudge = False  # Toggle for harmonic contour variance nudging of extrema and monkey saddles (default Off for fast browsing)
    state.show_effective_cps = False  # Toggle for viewing Effective Critical Point structure
    state.show_boundary_ports = True   # Toggle for viewing 1D Boundary Ports (Ridge Max / Valley Min)
    state.cp_results = {}
    state.effective_cp_results = {}
    state.excision_regions_list = []
    state.selected_cp = None
    state.selected_cluster = None
    state.all_cps_list = []
    state.clusters_list = []
    state.effective_cps_list = []

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
            cluster_actor.SetVisibility(False)
            cluster_members_actor.SetVisibility(False)
            state.cp_results = {}
            state.all_cps_list = []
            state.clusters_list = []
            return

        clust_deg = float(state.cluster_radius_deg)

        f_vals_np = numpy_support.vtk_to_numpy(pd.GetArray(target_arr_name))
        res = analyze_spherical_topology(
            pts_np,
            triangles_np,
            f_vals_np,
            persistence_threshold_pct=float(state.persistence_threshold_pct),
            cluster_angle_deg=clust_deg,
            sphere_radius=sphere_radius,
            enable_harmonic_nudge=bool(state.enable_harmonic_nudge),
        )

        # Apply glyph scaling
        scale = float(state.cp_glyph_scale)
        max_sphere_src.SetRadius(0.045 * scale)
        max_sphere_src.Update()
        min_sphere_src.SetRadius(0.045 * scale)
        min_sphere_src.Update()
        sad_cube_src.SetXLength(0.065 * scale)
        sad_cube_src.SetYLength(0.065 * scale)
        sad_cube_src.SetZLength(0.065 * scale)
        sad_cube_src.Update()
        cluster_member_src.SetRadius(0.060 * scale)
        cluster_member_src.Update()
        ecp_max_src.SetRadius(0.060 * scale)
        ecp_max_src.Update()
        ecp_min_src.SetRadius(0.060 * scale)
        ecp_min_src.Update()
        ecp_sad_src.SetXLength(0.085 * scale)
        ecp_sad_src.SetYLength(0.085 * scale)
        ecp_sad_src.SetZLength(0.085 * scale)
        ecp_sad_src.Update()

        # Compute Effective Critical Points & Excision Regions
        fuse_deg = float(state.monkey_saddle_fuse_deg)
        ecp_res = compute_effective_critical_points(
            pts_np,
            triangles_np,
            f_vals_np,
            standard_cluster_angle_deg=clust_deg,
            monkey_saddle_fuse_angle_deg=fuse_deg,
            sphere_radius=sphere_radius,
            enable_harmonic_nudge=bool(state.enable_harmonic_nudge),
        )
        state.effective_cp_results = ecp_res
        state.effective_cps_list = ecp_res.get("all_effective_cps", [])
        excision_regions = ecp_res.get("excision_regions", [])
        state.excision_regions_list = excision_regions

        is_eff_mode = bool(state.show_effective_cps)

        if is_eff_mode:
            # Hide standard CPs and display Effective CPs
            max_actor.SetVisibility(False)
            min_actor.SetVisibility(False)
            sad_actor.SetVisibility(False)

            # Update Effective Maxima
            ecp_max_pts.Reset()
            for mx in ecp_res["maxima"]:
                p = mx["position"]
                ecp_max_pts.InsertNextPoint(p[0], p[1], p[2])
            ecp_max_poly.SetPoints(ecp_max_pts)
            ecp_max_poly.Modified()
            ecp_max_actor.SetVisibility(bool(state.show_cps and state.show_maxima and len(ecp_res["maxima"]) > 0))

            # Update Effective Minima
            ecp_min_pts.Reset()
            for mn in ecp_res["minima"]:
                p = mn["position"]
                ecp_min_pts.InsertNextPoint(p[0], p[1], p[2])
            ecp_min_poly.SetPoints(ecp_min_pts)
            ecp_min_poly.Modified()
            ecp_min_actor.SetVisibility(bool(state.show_cps and state.show_minima and len(ecp_res["minima"]) > 0))

            # Update Effective Saddles
            ecp_sad_pts.Reset()
            for sd in ecp_res["saddles"]:
                p = sd["position"]
                ecp_sad_pts.InsertNextPoint(p[0], p[1], p[2])
            ecp_sad_poly.SetPoints(ecp_sad_pts)
            ecp_sad_poly.Modified()
            ecp_sad_actor.SetVisibility(bool(state.show_cps and state.show_saddles and len(ecp_res["saddles"]) > 0))

        else:
            # Hide Effective CP actors and show standard CPs
            ecp_max_actor.SetVisibility(False)
            ecp_min_actor.SetVisibility(False)
            ecp_sad_actor.SetVisibility(False)

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

        # 4. Multi-CP Cluster / Excision Circles & Boundary Ports
        circle_append_pts = vtkPoints()
        circle_append_lines = vtkCellArray()
        pt_counter = 0

        # Update Boundary Extrema Ports
        port_max_pts.Reset()
        port_min_pts.Reset()

        if is_eff_mode and excision_regions:
            # In Effective Mode, draw circles at optimal adaptive excision radius and render boundary ports
            for exc in excision_regions:
                c_pos = np.array(exc["centroid"])
                r_opt_deg = float(exc.get("optimal_radius_deg", 10.0))
                single_circle = build_spherical_circle_polydata(
                    centroid=c_pos,
                    sphere_radius=sphere_radius,
                    angular_radius_deg=r_opt_deg,
                    num_pts=64,
                    offset_factor=1.008,
                )
                c_pts = single_circle.GetPoints()
                c_lines = single_circle.GetLines()
                n_c = c_pts.GetNumberOfPoints()

                for i in range(n_c):
                    p = c_pts.GetPoint(i)
                    circle_append_pts.InsertNextPoint(p[0], p[1], p[2])

                c_lines.InitTraversal()
                c_idlist = vtkIdList()
                while c_lines.GetNextCell(c_idlist):
                    circle_append_lines.InsertNextCell(c_idlist.GetNumberOfIds())
                    for k in range(c_idlist.GetNumberOfIds()):
                        circle_append_lines.InsertCellPoint(pt_counter + c_idlist.GetId(k))

                pt_counter += n_c

                # Add Boundary Maxima (Ridge Ports)
                for pm in exc.get("maxima_ports", []):
                    pos = pm["position"]
                    port_max_pts.InsertNextPoint(pos[0], pos[1], pos[2])

                # Add Boundary Minima (Valley Ports)
                for pmn in exc.get("minima_ports", []):
                    pos = pmn["position"]
                    port_min_pts.InsertNextPoint(pos[0], pos[1], pos[2])

        else:
            # Standard clustering circles
            clusters = res.get("clusters", [])
            multi_clusters = [cl for cl in clusters if cl.get("is_multicluster")]
            display_circle_deg = max(clust_deg / 2.0, 2.5)

            for cl in multi_clusters:
                c_pos = np.array(cl["centroid"])
                cl_ang_r = max(cl.get("angular_radius_deg", display_circle_deg) * 1.12, display_circle_deg)
                single_circle = build_spherical_circle_polydata(
                    centroid=c_pos,
                    sphere_radius=sphere_radius,
                    angular_radius_deg=cl_ang_r,
                    num_pts=48,
                    offset_factor=1.008,
                )
                c_pts = single_circle.GetPoints()
                c_lines = single_circle.GetLines()
                n_c = c_pts.GetNumberOfPoints()

                for i in range(n_c):
                    p = c_pts.GetPoint(i)
                    circle_append_pts.InsertNextPoint(p[0], p[1], p[2])

                c_lines.InitTraversal()
                c_idlist = vtkIdList()
                while c_lines.GetNextCell(c_idlist):
                    circle_append_lines.InsertNextCell(c_idlist.GetNumberOfIds())
                    for k in range(c_idlist.GetNumberOfIds()):
                        circle_append_lines.InsertCellPoint(pt_counter + c_idlist.GetId(k))

                pt_counter += n_c

        cluster_circles_poly.SetPoints(circle_append_pts)
        cluster_circles_poly.SetLines(circle_append_lines)
        cluster_circles_poly.Modified()
        cluster_circles_tubes.SetRadius(0.010 * scale)
        cluster_circles_tubes.Modified()
        cluster_actor.SetVisibility(bool(state.show_cps and state.show_cluster_halos and pt_counter > 0))

        port_max_poly.SetPoints(port_max_pts)
        port_max_poly.Modified()
        port_min_poly.SetPoints(port_min_pts)
        port_min_poly.Modified()

        show_ports = bool(is_eff_mode and state.show_cps and state.show_boundary_ports)
        port_max_actor.SetVisibility(show_ports and port_max_pts.GetNumberOfPoints() > 0)
        port_min_actor.SetVisibility(show_ports and port_min_pts.GetNumberOfPoints() > 0)

        # Flatten CP list for UI table
        combined = []
        for mx in res["maxima"]:
            combined.append({**mx, "badge_color": "error", "type_title": "Maximum"})
        for m in res["minima"]:
            combined.append({**m, "badge_color": "info", "type_title": "Minimum"})
        for s in res["saddles"]:
            combined.append({**s, "badge_color": "success", "type_title": "Saddle"})

        clusters_all = res.get("clusters", [])
        state.all_cps_list = sorted(combined, key=lambda x: -x.get("persistence", 0.0))
        state.clusters_list = clusters_all
        state.cp_results = res

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
            scale = float(state.cp_glyph_scale)
            cp_highlight_src.SetRadius(0.075 * scale)
            cp_highlight_src.SetCenter(p[0], p[1], p[2])
            cp_highlight_src.Update()
            cp_highlight_actor.SetVisibility(True)
        render_window.Render()
        request_view_update()

    def select_cluster_item(cluster: Optional[Dict[str, Any]]):
        """Highlight a clicked cluster in 3D and update state."""
        state.selected_cluster = cluster
        if cluster is None:
            cluster_members_pts.Reset()
            cluster_members_poly.SetPoints(cluster_members_pts)
            cluster_members_poly.Modified()
            cluster_members_actor.SetVisibility(False)
            if cp_highlight_actor is not None:
                cp_highlight_actor.SetVisibility(False)
        else:
            # Highlight cluster centroid
            c_pos = cluster["centroid"]
            cp_highlight_src.SetRadius(max(float(cluster.get("radius", 0.08)) * 1.25, 0.08))
            cp_highlight_src.SetCenter(c_pos[0], c_pos[1], c_pos[2])
            cp_highlight_src.Update()
            cp_highlight_actor.SetVisibility(True)

            # Highlight all individual member CPs in cyan rings
            cluster_members_pts.Reset()
            for mem in cluster.get("members", []):
                m_pos = mem["position"]
                cluster_members_pts.InsertNextPoint(m_pos[0], m_pos[1], m_pos[2])
            cluster_members_poly.SetPoints(cluster_members_pts)
            cluster_members_poly.Modified()
            cluster_members_actor.SetVisibility(len(cluster.get("members", [])) > 0)

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
        "cluster_radius_deg",
        "monkey_saddle_fuse_deg",
        "cp_glyph_scale",
        "show_cluster_halos",
        "enable_harmonic_nudge",
        "show_effective_cps",
        "show_boundary_ports",
    )
    def on_param_change(**kwargs):
        update_field_visualization()

    @ctrl.add("select_cp_from_list")
    def select_cp_from_list(cp_label):
        # Check standard CPs
        for cp in state.all_cps_list:
            if cp.get("id_label") == cp_label:
                select_cp_item(cp)
                return
        # Check Effective CPs
        for ecp in state.effective_cps_list:
            if ecp.get("id_label") == cp_label:
                select_cp_item(ecp)
                return

    @ctrl.add("clear_cp_selection")
    def clear_cp_selection():
        select_cp_item(None)

    @ctrl.add("select_cluster_from_list")
    def select_cluster_from_list(cl_id):
        for cl in state.clusters_list:
            if cl.get("id") == cl_id:
                select_cluster_item(cl)
                return

    @ctrl.add("clear_cluster_selection")
    def clear_cluster_selection():
        select_cluster_item(None)

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

                            # Cluster Grouping Distance Threshold Slider (Angular radius in degrees)
                            with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                html.Div("Cluster Recognition Radius", classes="text-caption font-weight-bold text-medium-emphasis")
                                html.Div("{{ Number(cluster_radius_deg).toFixed(1) }}° (~{{ (atom_info.sphere_radius * Math.PI * cluster_radius_deg / 180.0).toFixed(3) }} Å)", classes="text-caption font-weight-bold text-warning")

                            v3.VSlider(
                                min=1.0,
                                max=30.0,
                                step=0.5,
                                v_model=("cluster_radius_deg",),
                                density="compact",
                                thumb_label=False,
                                color="warning",
                                classes="mt-1",
                            )

                            # Monkey Saddle Fusion Sensitivity Angle Slider
                            with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                html.Div("Monkey Saddle Fusion Sensitivity", classes="text-caption font-weight-bold text-medium-emphasis")
                                html.Div("{{ Number(monkey_saddle_fuse_deg).toFixed(1) }}° (~{{ (atom_info.sphere_radius * Math.PI * monkey_saddle_fuse_deg / 180.0).toFixed(3) }} Å)", classes="text-caption font-weight-bold text-purple-lighten-2")

                            v3.VSlider(
                                min=0.0,
                                max=35.0,
                                step=0.5,
                                v_model=("monkey_saddle_fuse_deg",),
                                density="compact",
                                thumb_label=False,
                                color="purple-accent-3",
                                classes="mt-1",
                            )

                            # Glyph Scale Slider
                            with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                html.Div("Glyph Size Scale", classes="text-caption font-weight-bold text-medium-emphasis")
                                html.Div("{{ Number(cp_glyph_scale).toFixed(2) }}x", classes="text-caption font-weight-bold text-primary")

                            v3.VSlider(
                                min=0.20,
                                max=2.00,
                                step=0.05,
                                v_model=("cp_glyph_scale",),
                                density="compact",
                                thumb_label=False,
                                color="primary",
                                classes="mt-1",
                            )

                            # Multi-CP Cluster Halos Switch
                            v3.VSwitch(
                                label="Show Multi-CP Cluster Outline Circles",
                                v_model=("show_cluster_halos",),
                                density="compact",
                                color="warning",
                                hide_details=True,
                                classes="mt-1 mb-1",
                            )

                            # Harmonic Contour Centroid Nudge Switch
                            v3.VSwitch(
                                label="Enable Harmonic Centroid Nudge",
                                v_model=("enable_harmonic_nudge",),
                                density="compact",
                                color="teal-accent-3",
                                hide_details=True,
                                classes="mt-1 mb-1",
                            )

                            v3.VDivider(classes="my-2")

                            # Effective Critical Point Structure Toggle
                            with v3.VCard(elevation=1, classes="pa-2 mb-2 bg-grey-darken-4 border-primary"):
                                with html.Div(classes="d-flex justify-space-between align-center"):
                                    html.Div("Effective CP Structure Mode", classes="text-caption font-weight-bold text-amber")
                                    v3.VChip("{{ effective_cps_list.length }} E-CPs", size="x-small", color="amber-darken-3")

                                v3.VSwitch(
                                    label="View Effective Critical Points (ECPs)",
                                    v_model=("show_effective_cps",),
                                    density="compact",
                                    color="amber",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                with html.Div(v_if="show_effective_cps", classes="mt-1"):
                                    v3.VSwitch(
                                        label="Show Excision Boundary Ports",
                                        v_model=("show_boundary_ports",),
                                        density="compact",
                                        color="cyan",
                                        hide_details=True,
                                        classes="mb-1",
                                    )
                                    html.Div(
                                        "Red dots = Ridge Maxima (Ascending Paths Out) | Blue dots = Valley Minima (Descending Paths Out)",
                                        classes="text-caption text-medium-emphasis",
                                        style="font-size: 11px;",
                                    )

                            # Critical Point Counts Summary (Dynamic based on Effective Mode)
                            with html.Div(v_if="!show_effective_cps && cp_results && cp_results.filtered_counts", classes="mt-2 pa-2 rounded surface"):
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

                            with html.Div(v_if="show_effective_cps && effective_cp_results && effective_cp_results.counts", classes="mt-2 pa-2 rounded surface bg-amber-darken-4"):
                                with v3.VRow(dense=True, classes="text-caption text-center"):
                                    with v3.VCol(cols=3):
                                        html.Div("Eff. Max", classes="text-white text-caption")
                                        html.Div("{{ effective_cp_results.counts.maxima }}", classes="text-subtitle-2 font-weight-bold text-white")
                                    with v3.VCol(cols=3):
                                        html.Div("Eff. Min", classes="text-white text-caption")
                                        html.Div("{{ effective_cp_results.counts.minima }}", classes="text-subtitle-2 font-weight-bold text-white")
                                    with v3.VCol(cols=3):
                                        html.Div("Eff. Sad", classes="text-white text-caption")
                                        html.Div("{{ effective_cp_results.counts.saddles }}", classes="text-subtitle-2 font-weight-bold text-white")
                                    with v3.VCol(cols=3):
                                        html.Div("Monkey", classes="text-white text-caption")
                                        html.Div("{{ effective_cp_results.counts.monkey_saddles }}", classes="text-subtitle-2 font-weight-bold text-purple-lighten-2")

                            # Effective Critical Points Table (when Effective Mode is ON)
                            with html.Div(v_if="show_effective_cps && effective_cps_list && effective_cps_list.length > 0", classes="mt-3"):
                                html.Div("Effective Critical Points (click to highlight)", classes="text-caption font-weight-bold text-amber mb-1")
                                with v3.VTable(density="compact", classes="elevation-0", style="max-height: 200px; overflow-y: auto;"):
                                    with html.Thead():
                                        with html.Tr():
                                            html.Th("E-ID", classes="text-left text-caption font-weight-bold")
                                            html.Th("Type", classes="text-left text-caption font-weight-bold")
                                            html.Th("Source", classes="text-left text-caption font-weight-bold")
                                            html.Th("Value", classes="text-right text-caption font-weight-bold")
                                    with html.Tbody():
                                        with html.Tr(
                                            v_for="ecp in effective_cps_list",
                                            key="ecp.id_label",
                                            click=(ctrl.select_cp_from_list, "[ecp.id_label]"),
                                            classes="cursor-pointer",
                                        ):
                                            html.Td("{{ ecp.id_label }}", classes="text-caption font-weight-bold text-amber")
                                            with html.Td():
                                                v3.VChip("{{ ecp.type_title }}", size="x-small", color=("ecp.badge_color",))
                                            with html.Td():
                                                v3.VChip("{{ ecp.category }}", size="x-small", color="warning" if ("ecp.category === 'clustered'", True) else "info")
                                            html.Td("{{ ecp.value.toFixed(5) }}", classes="text-right text-caption font-mono")

                            # Excision Regions & Boundary Ports Breakdown (when Effective Mode is ON)
                            with html.Div(v_if="show_effective_cps && excision_regions_list && excision_regions_list.length > 0", classes="mt-3"):
                                html.Div("Adaptive Excision Boundaries & Ports", classes="text-caption font-weight-bold text-cyan mb-1")
                                with v3.VTable(density="compact", classes="elevation-0", style="max-height: 180px; overflow-y: auto;"):
                                    with html.Thead():
                                        with html.Tr():
                                            html.Th("Region", classes="text-left text-caption font-weight-bold")
                                            html.Th("Radius (θ)", classes="text-left text-caption font-weight-bold")
                                            html.Th("Boundary Ports", classes="text-right text-caption font-weight-bold")
                                    with html.Tbody():
                                        with html.Tr(
                                            v_for="exc in excision_regions_list",
                                            key="exc.ecp_id",
                                        ):
                                            html.Td("{{ exc.ecp_id }} (χ={{ exc.chi }})", classes="text-caption font-weight-bold text-cyan")
                                            html.Td("{{ exc.optimal_radius_deg.toFixed(1) }}° (~{{ exc.arc_radius_angstrom.toFixed(3) }} Å)", classes="text-caption font-mono")
                                            with html.Td(classes="text-right"):
                                                v3.VChip("{{ exc.found_signature }}", size="x-small", color="cyan" if ("exc.is_pure_signature", True) else "warning")

                            # Standard CP Clusters Table & Composition Breakdown (when Effective Mode is OFF)
                            with html.Div(v_if="!show_effective_cps && clusters_list && clusters_list.length > 0", classes="mt-3"):
                                with html.Div(classes="d-flex justify-space-between align-center mb-1"):
                                    html.Div("CP Clusters & Composition", classes="text-caption font-weight-bold text-medium-emphasis")
                                    v3.VChip("{{ clusters_list.filter(c => c.is_multicluster).length }} multi-CP", size="x-small", color="warning")

                                with v3.VTable(density="compact", classes="elevation-0", style="max-height: 180px; overflow-y: auto;"):
                                    with html.Thead():
                                        with html.Tr():
                                            html.Th("Cluster", classes="text-left text-caption font-weight-bold")
                                            html.Th("Composition", classes="text-left text-caption font-weight-bold")
                                            html.Th("Net χ", classes="text-right text-caption font-weight-bold")
                                    with html.Tbody():
                                        with html.Tr(
                                            v_for="cl in clusters_list",
                                            key="cl.id",
                                            click=(ctrl.select_cluster_from_list, "[cl.id]"),
                                            classes="cursor-pointer",
                                        ):
                                            html.Td("{{ cl.id }} ({{ cl.size }} pts)", classes="text-caption font-weight-bold")
                                            with html.Td():
                                                v3.VChip("{{ cl.comp_summary }}", size="x-small", color="amber-darken-3" if ("cl.is_multicluster", True) else "default")
                                            html.Td("{{ cl.net_index >= 0 ? '+' + cl.net_index : cl.net_index }}", classes="text-right text-caption font-weight-bold font-mono")

                            # Critical Points Table / List (when Effective Mode is OFF)
                            with html.Div(v_if="!show_effective_cps && all_cps_list && all_cps_list.length > 0", classes="mt-3"):
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

                # Selected Cluster Composition Inspector Card
                with v3.VCard(
                    v_if="selected_cluster",
                    elevation=3,
                    classes="mb-3 border-warning",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon("mdi-chart-bubble", classes="mr-2", color="warning")
                                html.Span("Cluster {{ selected_cluster.id }} ({{ selected_cluster.size }} CPs)")
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_cluster_selection,
                            )
                        v3.VCardSubtitle("Composition: {{ selected_cluster.comp_summary }} | Net Index: {{ selected_cluster.net_index >= 0 ? '+' + selected_cluster.net_index : selected_cluster.net_index }}")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div("Angular Radius / Diam", classes="text-caption text-medium-emphasis")
                                html.Div("{{ selected_cluster.angular_radius_deg.toFixed(1) }}° / {{ selected_cluster.angular_diameter_deg.toFixed(1) }}°", classes="text-body-2 font-weight-bold")
                            with v3.VCol(cols=6):
                                html.Div("Arc Span (~Å)", classes="text-caption text-medium-emphasis")
                                html.Div("{{ selected_cluster.diameter.toFixed(4) }} Å", classes="text-body-2 font-weight-bold")

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div("Centroid Unit Vector (X, Y, Z)", classes="text-caption text-medium-emphasis")
                                html.Div("({{ selected_cluster.centroid_unit[0].toFixed(3) }}, {{ selected_cluster.centroid_unit[1].toFixed(3) }}, {{ selected_cluster.centroid_unit[2].toFixed(3) }})", classes="text-caption font-mono")

                        # List constituent CPs in this cluster
                        html.Div("Constituent Critical Points", classes="text-caption font-weight-bold text-medium-emphasis mt-2 mb-1")
                        with v3.VList(density="compact", classes="pa-0 bg-transparent"):
                            with v3.VListItem(
                                v_for="mem in selected_cluster.members",
                                key="mem.id_label",
                                click=(ctrl.select_cp_from_list, "[mem.id_label]"),
                                classes="px-1 py-0 cursor-pointer",
                            ):
                                with html.Template(v_slot_prepend=True):
                                    v3.VChip("{{ mem.id_label }}", size="x-small", color="primary", classes="mr-2")
                                html.Span("{{ mem.type }} | Val: {{ mem.value.toFixed(5) }} | Pers: {{ mem.persistence_pct.toFixed(1) }}%", classes="text-caption")

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
