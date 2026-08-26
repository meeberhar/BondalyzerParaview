#!/usr/bin/env python3
"""
Prototype Trame Viewer for Bondalyzer .vtm Datasets.

Renders the 1D molecular skeleton from .vtm files:
- Atoms rendered as 3D sphere glyphs with element/RGB colors
- Inferred bonds & bond paths rendered as 3D cylindrical tubes
- Critical points rendered as distinct point markers

Run with:
    /Applications/ParaView-6.2.0-RC1.app/Contents/bin/pvpython trame_viewer.py [ethene_1d_zones.vtm]
or with standard python if trame and vtk are installed:
    python3 trame_viewer.py [ethene_1d_zones.vtm]
"""

import os
import sys
from typing import Optional

# VTK Imports
import vtk
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataReader
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkPolyDataMapper,
    vtkActor,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

# Trame Imports
try:
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import vuetify3 as v3
    from trame.widgets.vtk import VtkRemoteView
    TRAME_AVAILABLE = True
except ImportError:
    try:
        from trame.app import get_server
        from trame.ui.vuetify import SinglePageLayout
        from trame.widgets import vuetify as v3
        from trame.widgets.vtk import VtkRemoteView
        TRAME_AVAILABLE = True
    except ImportError:
        TRAME_AVAILABLE = False


def create_visualization_pipeline(vtm_path: str):
    """
    Build a standard VTK rendering pipeline from the .vtm file.
    Returns (renderer, render_window, actors_dict).
    """
    if not os.path.exists(vtm_path):
        raise FileNotFoundError(f"VTM file not found: {vtm_path}")

    # 1. Read MultiBlock dataset
    reader = vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_path)
    reader.Update()
    mb = reader.GetOutput()

    renderer = vtkRenderer()
    renderer.SetBackground(0.12, 0.13, 0.16)  # Dark chemist canvas background
    renderer.SetBackground2(0.20, 0.22, 0.26)
    renderer.SetGradientBackground(True)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 750)
    render_window.SetWindowName("Bondalyzer Molecule Viewer")

    actors = {}

    num_blocks = mb.GetNumberOfBlocks()
    for b in range(num_blocks):
        block_name = mb.GetMetaData(b).Get(mb.NAME()) if mb.GetMetaData(b) else f"Block_{b}"
        poly = mb.GetBlock(b)
        if not poly or poly.GetNumberOfPoints() == 0:
            continue

        n_pts = poly.GetNumberOfPoints()
        n_lines = poly.GetNumberOfLines()

        # Check if line/path or point cloud
        is_line = (n_lines > 0)

        if is_line:
            # -------------------------------------------------------------
            # Line / Path Block: Apply 3D Cylindrical Tube Filter
            # -------------------------------------------------------------
            tuber = vtkTubeFilter()
            tuber.SetInputData(poly)
            tuber.SetNumberOfSides(16)
            tuber.CappingOn()

            if "bond path" in block_name.lower() or "path" in block_name.lower():
                tuber.SetRadius(0.025)  # Thinner for gradient paths
            else:
                tuber.SetRadius(0.045)  # Standard bond tube radius
            tuber.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tuber.GetOutputPort())

            if poly.GetPointData().HasArray("RGBColor"):
                mapper.SetColorModeToDirectScalars()
                mapper.ScalarVisibilityOn()
                poly.GetPointData().SetActiveScalars("RGBColor")
            else:
                mapper.ScalarVisibilityOff()

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetSpecular(0.4)
            actor.GetProperty().SetSpecularPower(30)
            renderer.AddActor(actor)
            actors[block_name] = actor

        else:
            # -------------------------------------------------------------
            # Point Set (Atoms or Critical Points): Apply Sphere Glyphs
            # -------------------------------------------------------------
            sphere_source = vtkSphereSource()
            sphere_source.SetThetaResolution(24)
            sphere_source.SetPhiResolution(24)

            glyph = vtkGlyph3D()
            glyph.SetSourceConnection(sphere_source.GetOutputPort())
            glyph.SetInputData(poly)
            glyph.ScalingOn()
            glyph.SetScaleModeToDataScalingOff()

            is_atom = "atom" in block_name.lower()
            if is_atom:
                if "(c)" in block_name.lower():
                    glyph.SetScaleFactor(0.35)  # Carbon radius
                elif "(h)" in block_name.lower():
                    glyph.SetScaleFactor(0.25)  # Hydrogen radius
                else:
                    glyph.SetScaleFactor(0.32)
            else:
                # Critical Points
                glyph.SetScaleFactor(0.12)

            glyph.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())

            if poly.GetPointData().HasArray("RGBColor"):
                mapper.SetColorModeToDirectScalars()
                mapper.ScalarVisibilityOn()
                poly.GetPointData().SetActiveScalars("RGBColor")
            else:
                mapper.ScalarVisibilityOff()

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetSpecular(0.6)
            actor.GetProperty().SetSpecularPower(50)
            renderer.AddActor(actor)
            actors[block_name] = actor

    renderer.ResetCamera()
    return renderer, render_window, actors


def run_trame_app(vtm_path: str, server_name: str = "bondalyzer_viewer"):
    """
    Launch the Trame-based interactive viewer application.
    """
    renderer, render_window, actors = create_visualization_pipeline(vtm_path)

    server = get_server(server_name)
    state, ctrl = server.state, server.controller

    # Initial state
    state.vtm_file = os.path.basename(vtm_path)
    state.num_blocks = len(actors)
    state.block_names = list(actors.keys())

    @ctrl.add("reset_camera")
    def reset_camera():
        renderer.ResetCamera()
        render_window.Render()
        ctrl.view_update()

    # Build UI Layout
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Bondalyzer Molecule Viewer")

        with layout.toolbar:
            v3.VSpacer()
            v3.VBtn(
                "Reset View",
                prepend_icon="mdi-camera-flip-outline",
                click=ctrl.reset_camera,
                variant="tonal",
                density="compact",
                color="primary",
            )

        with layout.content:
            with v3.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
                view = VtkRemoteView(render_window, interactive_ratio=1.0)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    print(f"\n[Bondalyzer] Starting Trame application for: {vtm_path}")
    print(f"[Bondalyzer] Loaded {len(actors)} rendered blocks.")
    server.start()


def run_native_vtk_window(vtm_path: str):
    """
    Fallback native VTK interactive window if trame is not installed in the python environment.
    """
    print(f"\n[Bondalyzer] Trame not found. Launching native VTK window for: {vtm_path}")
    renderer, render_window, actors = create_visualization_pipeline(vtm_path)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    render_window.Render()
    interactor.Start()


def main():
    default_vtm = "ethene_1d_zones.vtm"
    vtm_file = sys.argv[1] if len(sys.argv) > 1 else default_vtm

    if not os.path.exists(vtm_file):
        print(f"File '{vtm_file}' not found. Please provide a valid .vtm file path.")
        sys.exit(1)

    if TRAME_AVAILABLE:
        run_trame_app(vtm_file)
    else:
        run_native_vtk_window(vtm_file)


if __name__ == "__main__":
    main()
