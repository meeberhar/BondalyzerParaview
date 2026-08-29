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
import math
from typing import Optional, Dict, Any, List, Tuple

# VTK Imports
import vtk
import vtkmodules.vtkRenderingOpenGL2  # Ensure OpenGL2 backend is properly initialized
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataReader
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkPolyDataMapper,
    vtkActor,
    vtkCellPicker,
    vtkCoordinate,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

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


# -----------------------------------------------------------------------------
# Covalent Radii in Angstroms (Cordero et al., Dalton Trans. 2008, 2832-2838)
# Elements 1 through 96 (H to Cm)
# -----------------------------------------------------------------------------
COVALENT_RADII = {
    # Period 1
    "H": 0.31, "He": 0.28,
    # Period 2
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    # Period 3
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    # Period 4
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "Kr": 1.16,
    # Period 5
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46,
    "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
    "I": 1.39, "Xe": 1.40,
    # Period 6 & Lanthanides
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
    # Period 7 & Actinides
    "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87,
    "Am": 1.80, "Cm": 1.69,
}

# Standard ball-and-stick display scale factor applied to covalent radii (0.42 * 2.5 = 1.05)
BALL_AND_STICK_SCALE = 1.05


def get_covalent_radius(element_symbol: str, default: float = 0.75) -> float:
    """Retrieve covalent radius in Angstroms for an element symbol."""
    return COVALENT_RADII.get(element_symbol.strip(), default)


def parse_dataset_metadata(mb) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract high-level molecule metadata, catalog of atoms, and catalog of critical points
    from the MultiBlock dataset without duplicates.
    """
    atoms = []
    element_counts = {}
    bonds = 0
    bond_paths = 0
    critical_points = []
    cp_counts = {"bond": 0, "ring": 0, "cage": 0, "nuclear": 0}

    num_blocks = mb.GetNumberOfBlocks()
    for b in range(num_blocks):
        poly = mb.GetBlock(b)
        if not poly:
            continue
        name = mb.GetMetaData(b).Get(mb.NAME()) if mb.GetMetaData(b) else f"Block_{b}"
        n_pts = poly.GetNumberOfPoints()
        n_lines = poly.GetNumberOfLines()
        fd = poly.GetFieldData()
        pd = poly.GetPointData()

        # 1. ATOMS
        if n_lines == 0 and "atom" in name.lower():
            atom_type = "X"
            if fd.HasArray("Aux_AtomType"):
                atom_type = str(fd.GetAbstractArray("Aux_AtomType").GetValue(0))
            elif "(" in name and ")" in name:
                atom_type = name.split("(")[1].split(")")[0].strip()

            for i in range(n_pts):
                pt = poly.GetPoint(i)
                at_num = 0
                if pd.HasArray("atomic_number"):
                    at_num = int(pd.GetArray("atomic_number").GetTuple1(i))

                density = None
                if pd.HasArray("Electron Density"):
                    density = float(pd.GetArray("Electron Density").GetTuple1(i))

                atom_id = len(atoms) + 1
                atom_name = f"{atom_type}{atom_id}"
                element_counts[atom_type] = element_counts.get(atom_type, 0) + 1

                atoms.append({
                    "id": atom_id,
                    "name": atom_name,
                    "block": name,
                    "element": atom_type,
                    "atomic_number": at_num,
                    "position": [round(pt[0], 4), round(pt[1], 4), round(pt[2], 4)],
                    "raw_pos": pt,
                    "electron_density": f"{density:.4f}" if density is not None else "N/A",
                })

        # 2. BOND PATHS (Gradient paths)
        elif n_lines > 0 and "bond path" in name.lower():
            bond_paths += 1

        # 3. INFERRED BONDS (Connectivity mesh)
        elif n_lines > 0:
            bonds += 1

        # 4. TOPOLOGICAL CRITICAL POINTS (Only read specific sub-zones to prevent double counting)
        elif n_lines == 0 and ("_cp" in name.lower() or "criticalpoints" in str(fd.GetAbstractArray("Aux_ZoneType").GetValue(0) if fd.HasArray("Aux_ZoneType") else "").lower()):
            # Skip the composite master block "Critical Points"
            if name.strip().lower() == "critical points":
                continue
            # Skip "nuclear_cp" since atoms are already represented as atomic nuclei
            if name.strip().lower() == "nuclear_cp":
                continue

            cp_type = "Critical Point"
            signature = "(3, -1)"
            badge_color = "red"
            cp_key = "bond"

            if "bond" in name.lower():
                cp_type = "Bond CP"
                signature = "(3, -1)"
                badge_color = "error"
                cp_key = "bond"
            elif "ring" in name.lower():
                cp_type = "Ring CP"
                signature = "(3, +1)"
                badge_color = "success"
                cp_key = "ring"
            elif "cage" in name.lower():
                cp_type = "Cage CP"
                signature = "(3, +3)"
                badge_color = "info"
                cp_key = "cage"
            elif "nuclear" in name.lower():
                cp_type = "Nuclear CP"
                signature = "(3, -3)"
                badge_color = "warning"
                cp_key = "nuclear"

            for i in range(n_pts):
                pt = poly.GetPoint(i)
                density = None
                if pd.HasArray("Electron Density"):
                    density = float(pd.GetArray("Electron Density").GetTuple1(i))

                cp_id = len(critical_points) + 1
                cp_counts[cp_key] += 1

                critical_points.append({
                    "id": cp_id,
                    "name": f"{cp_type} #{cp_counts[cp_key]}",
                    "type": cp_type,
                    "signature": signature,
                    "badge_color": badge_color,
                    "block": name,
                    "position": [round(pt[0], 4), round(pt[1], 4), round(pt[2], 4)],
                    "raw_pos": pt,
                    "electron_density": f"{density:.4f}" if density is not None else "N/A",
                })

    # Format chemical formula (e.g., C2H4)
    formula_parts = []
    if "C" in element_counts:
        c_cnt = element_counts["C"]
        formula_parts.append(f"C{c_cnt if c_cnt > 1 else ''}")
    if "H" in element_counts:
        h_cnt = element_counts["H"]
        formula_parts.append(f"H{h_cnt if h_cnt > 1 else ''}")
    for el in sorted(element_counts.keys()):
        if el not in ("C", "H"):
            cnt = element_counts[el]
            formula_parts.append(f"{el}{cnt if cnt > 1 else ''}")
    formula = "".join(formula_parts) if formula_parts else "N/A"

    # Canonical order for SCA global fields
    # Ordered by user request:
    # 1. Electron Density (ρ)
    # 2. Kinetic Energy (K) (when present)
    # 3. Volume (V)
    # 4. Mean Curvature (H)
    # 5. Gaussian Curvature (K)
    # 6. Shape Index (S)
    # 7. Curvedness (C) - (RMS curvature dropped as equivalent)
    # 8. Willmore Energy (H^2)
    # 9. Modified Willmore Energy (H^2 - K)
    # Any other fields appended, and sign-change fields placed at the very end.
    priority_patterns = [
        ("electron density", "Electron Density"),
        ("kinetic", "Kinetic Energy"),
        ("v", "Volume (V)"),
        ("mean curvature", "Mean Curvature (H)"),
        ("gaussian curvature", "Gaussian Curvature (K)"),
        ("shape index", "Shape Index (S)"),
        ("curvedness", "Curvedness (C)"),
        ("willmore energy", "Willmore Energy (H²)"),
        ("modified willmore", "Modified Willmore Energy (H²−K)"),
    ]

    # Collect non-condensed global fields from the blocks.
    # Exclude:
    # - (condensed) fields
    # - RMS curvature (dropped as duplicate of curvedness)
    # - Positive/negative mean curvature (dropped as duplicate of mean curvature)
    # - Sign-change fields (atom-specific, moved to GBA tools)
    # - Spatial coordinates, RGB colors, atomic numbers, normals
    raw_global_fields = set()
    for b in range(num_blocks):
        poly = mb.GetBlock(b)
        if not poly:
            continue
        pd = poly.GetPointData()
        for i in range(pd.GetNumberOfArrays()):
            aname = pd.GetArrayName(i)
            if not aname:
                continue
            lower_a = aname.lower()
            if (
                "(condensed)" not in lower_a
                and "rms" not in lower_a
                and "positive mean curvature" not in lower_a
                and "negative mean curvature" not in lower_a
                and "sign change" not in lower_a
                and aname not in ("X", "Y", "Z", "RGBColor", "atomic_number", "Atomic Numbers", "AtomicNumber", "Normals")
            ):
                raw_global_fields.add(aname)

    def get_display_title(raw_name: str) -> str:
        lower = raw_name.lower()
        if "electron density" in lower:
            return "Electron Density (ρ)"
        if "kinetic" in lower:
            return "Kinetic Energy (K)"
        if lower == "v":
            return "Volume (V)"
        if "mean curvature" in lower:
            return "Mean Curvature (H)"
        if "gaussian curvature" in lower:
            return "Gaussian Curvature (K)"
        if "shape index" in lower:
            return "Shape Index (S)"
        if "curvedness" in lower:
            return "Curvedness (C)"
        if "willmore energy" in lower and "modified" not in lower:
            return "Willmore Energy (H²)"
        if "modified willmore" in lower:
            return "Modified Willmore Energy (H²−K)"
        if lower == "î±" or lower == "α" or "alpha" in lower:
            return "Trajectory Parameter (α)"
        # Clean any remaining mojibake 'Ï\x81' or 'Ï '
        cleaned = raw_name.replace("Ï\x81", "ρ").replace("Ï ", "ρ ").replace("Î±", "α")
        return cleaned

    def get_priority_rank(f_name: str) -> int:
        lower = f_name.lower()
        if "electron density" in lower:
            return 1
        if "kinetic" in lower:
            return 2
        if lower == "v":
            return 3
        if "mean curvature" in lower:
            return 4
        if "gaussian curvature" in lower:
            return 5
        if "shape index" in lower:
            return 6
        if "curvedness" in lower:
            return 7
        if "willmore energy" in lower and "modified" not in lower:
            return 8
        if "modified willmore" in lower:
            return 9
        return 50  # Other fields in between

    sorted_raw_fields = sorted(raw_global_fields, key=lambda f: (get_priority_rank(f), f))

    # Build list of items for Vuetify VSelect: [{title: 'Mean Curvature (H)', value: 'Ï\x81 mean curvature'}, ...]
    global_field_items = [
        {"title": get_display_title(f), "value": f}
        for f in sorted_raw_fields
    ]

    # Build list of 11 condensed fields for GBA Tools matching SCA naming/order
    gba_condensed_field_items = [
        {"title": "Electron Density (ρ)", "value": "Electron Density"},
        {"title": "Volume (V)", "value": "V"},
        {"title": "Mean Curvature (H)", "value": "Mean Curvature"},
        {"title": "Positive Mean Curvature (H⁺)", "value": "Positive Mean Curvature"},
        {"title": "Negative Mean Curvature (H⁻)", "value": "Negative Mean Curvature"},
        {"title": "Gaussian Curvature (K)", "value": "Gaussian Curvature"},
        {"title": "Shape Index (S)", "value": "Shape Index"},
        {"title": "Curvedness (C)", "value": "Curvedness"},
        {"title": "Willmore Energy (H²)", "value": "Willmore Energy"},
        {"title": "Modified Willmore Energy (H²−K)", "value": "Modified Willmore Energy"},
        {"title": "RMS Curvature", "value": "RMS Curvature"},
    ]

    molecule_info = {
        "formula": formula,
        "title": f"Ethene ({formula})",
        "total_atoms": len(atoms),
        "element_counts": element_counts,
        "bonds": bonds,
        "bond_paths": bond_paths,
        "bond_cps": cp_counts["bond"],
        "ring_cps": cp_counts["ring"],
        "cage_cps": cp_counts["cage"],
        "total_cps": cp_counts["bond"] + cp_counts["ring"] + cp_counts["cage"],
        "num_blocks": num_blocks,
        "global_fields": global_field_items,
        "selected_global_field": sorted_raw_fields[0] if sorted_raw_fields else "Electron Density",
        "gba_atoms_list": ["C1"],
        "gba_fields": gba_condensed_field_items,
    }

    return molecule_info, atoms, critical_points


def create_visualization_pipeline(vtm_path: str):
    """
    Build a standard VTK rendering pipeline from the .vtm file.
    Returns (renderer, render_window, actors_dict, molecule_info, atoms_list, cp_list, highlight_actor, highlight_source).
    """
    if not os.path.exists(vtm_path):
        raise FileNotFoundError(f"VTM file not found: {vtm_path}")

    # 1. Read MultiBlock dataset
    reader = vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_path)
    reader.Update()
    mb = reader.GetOutput()

    molecule_info, atoms, critical_points = parse_dataset_metadata(mb)

    renderer = vtkRenderer()
    renderer.SetBackground(0.12, 0.13, 0.16)  # Dark chemist canvas background
    renderer.SetBackground2(0.20, 0.22, 0.26)
    renderer.SetGradientBackground(True)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 750)
    render_window.SetWindowName("Bondalyzer Molecule Viewer")
    # Hide native desktop window on macOS (render purely offscreen for web streaming)
    render_window.SetOffScreenRendering(1)

    # Attach and initialize interactor with TrackballCamera style
    # (Required by VTK / vtkWebApplication on macOS/Cocoa to handle mouse interaction events without segfaulting)
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.Initialize()

    actors = {}

    num_blocks = mb.GetNumberOfBlocks()
    for b in range(num_blocks):
        block_name = mb.GetMetaData(b).Get(mb.NAME()) if mb.GetMetaData(b) else f"Block_{b}"
        poly = mb.GetBlock(b)
        if not poly or poly.GetNumberOfPoints() == 0:
            continue

        # Skip composite master "Critical Points" block to prevent double rendering & z-fighting
        if block_name.strip().lower() == "critical points":
            continue
        # Skip "nuclear_cp" block since atoms already represent nuclei
        if block_name.strip().lower() == "nuclear_cp":
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
            mapper.ScalarVisibilityOff()  # Use actor material properties

            # Extract color
            arr = poly.GetPointData().GetArray("RGBColor")
            r_val, g_val, b_val = (0.7, 0.7, 0.7)
            if arr and arr.GetNumberOfTuples() > 0:
                t = arr.GetTuple(0)
                r_val, g_val, b_val = t[0] / 255.0, t[1] / 255.0, t[2] / 255.0

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(r_val, g_val, b_val)
            actor.GetProperty().SetAmbient(0.35)
            actor.GetProperty().SetDiffuse(0.75)
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
                # Extract element symbol from FieldData or block name
                element = "C"
                if poly.GetFieldData().HasArray("Aux_AtomType"):
                    element = str(poly.GetFieldData().GetAbstractArray("Aux_AtomType").GetValue(0))
                elif "(" in block_name and ")" in block_name:
                    element = block_name.split("(")[1].split(")")[0].strip()

                r_cov = get_covalent_radius(element, default=0.75)
                glyph.SetScaleFactor(r_cov * BALL_AND_STICK_SCALE)
            elif "bond" in block_name.lower() or "bcp" in block_name.lower():
                # Bond Critical Points: clearly visible sphere
                glyph.SetScaleFactor(0.18)
            elif "ring" in block_name.lower() or "rcp" in block_name.lower():
                # Ring Critical Points
                glyph.SetScaleFactor(0.18)
            elif "cage" in block_name.lower() or "ccp" in block_name.lower():
                # Cage Critical Points
                glyph.SetScaleFactor(0.18)
            else:
                glyph.SetScaleFactor(0.14)

            glyph.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOff()  # Use actor material properties

            # Extract color
            arr = poly.GetPointData().GetArray("RGBColor")
            r_val, g_val, b_val = (0.8, 0.8, 0.8)
            if arr and arr.GetNumberOfTuples() > 0:
                t = arr.GetTuple(0)
                r_val, g_val, b_val = t[0] / 255.0, t[1] / 255.0, t[2] / 255.0

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(r_val, g_val, b_val)
            actor.GetProperty().SetAmbient(0.35)
            actor.GetProperty().SetDiffuse(0.75)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(40)
            renderer.AddActor(actor)
            actors[block_name] = actor

    # Create Selection Highlight Wireframe Actor (initially hidden)
    highlight_source = vtkSphereSource()
    highlight_source.SetThetaResolution(20)
    highlight_source.SetPhiResolution(20)
    highlight_source.SetRadius(0.42)
    highlight_source.Update()

    highlight_mapper = vtkPolyDataMapper()
    highlight_mapper.SetInputConnection(highlight_source.GetOutputPort())

    highlight_actor = vtkActor()
    highlight_actor.SetMapper(highlight_mapper)
    highlight_actor.GetProperty().SetColor(1.0, 0.85, 0.1)  # Vivid yellow highlight
    highlight_actor.GetProperty().SetRepresentationToWireframe()
    highlight_actor.GetProperty().SetLineWidth(2.5)
    highlight_actor.SetVisibility(False)
    renderer.AddActor(highlight_actor)

    renderer.ResetCamera()
    return renderer, render_window, actors, molecule_info, atoms, critical_points, highlight_actor, highlight_source


def run_trame_app(vtm_path: str, server_name: str = "bondalyzer_viewer"):
    """
    Launch the Trame-based interactive viewer application with metadata drawer and atom/CP picking.
    """
    (
        renderer,
        render_window,
        actors,
        molecule_info,
        atoms,
        critical_points,
        highlight_actor,
        highlight_source,
    ) = create_visualization_pipeline(vtm_path)

    server = get_server(server_name)
    state, ctrl = server.state, server.controller

    # Initial state
    state.vtm_file = os.path.basename(vtm_path)
    state.num_blocks = len(actors)
    state.block_names = list(actors.keys())
    state.molecule_info = molecule_info
    state.atoms_list = atoms
    state.cps_list = critical_points
    state.selected_item = None
    state.active_nav_mode = "overview"  # 'overview', 'sca', 'gba'
    state.sca_visualization_mode = "cutplane"  # 'cutplane' or 'isosurface'
    state.selected_global_field = molecule_info["selected_global_field"]
    state.selected_gba_atom_id = 1
    state.selected_condensed_field = "Electron Density"
    state.drawer_open = True

    # Cell picker and world coordinate projector for interactive 3D picking
    picker = vtkCellPicker()
    picker.SetTolerance(0.02)
    world_coord = vtkCoordinate()
    world_coord.SetCoordinateSystemToWorld()

    def select_item(item: Optional[Dict[str, Any]]):
        """Helper to highlight an atom or critical point and update UI state."""
        state.selected_item = item
        if item is None:
            highlight_actor.SetVisibility(False)
        else:
            raw_pos = item["raw_pos"]
            highlight_source.SetCenter(raw_pos[0], raw_pos[1], raw_pos[2])
            
            # Set highlight sphere radius based on item type
            # Note: vtkSphereSource default radius is 0.5, so vtkGlyph3D renders spheres
            # with actual radius = 0.5 * glyph_scale_factor.
            # To make the highlight wireframe exactly 1.2x the rendered sphere:
            # highlight_radius = (0.5 * glyph_scale_factor) * 1.20 = glyph_scale_factor * 0.60
            if "element" in item:
                el = item["element"]
                r_cov = get_covalent_radius(el, default=0.75)
                glyph_scale = r_cov * BALL_AND_STICK_SCALE
                highlight_source.SetRadius(glyph_scale * 0.5 * 1.20)
            else:
                # Critical Point (CP scale factor is 0.18)
                highlight_source.SetRadius(0.18 * 0.5 * 1.20)

            highlight_source.Update()
            highlight_actor.SetVisibility(True)

        render_window.Render()
        if hasattr(ctrl, "view_update"):
            ctrl.view_update()

    @ctrl.add("on_scene_click")
    def on_scene_click(click_x, click_y, client_w, client_h):
        """Handle 3D picking when user clicks in the 3D viewport."""
        if click_x is None or click_y is None:
            return

        rw_size = render_window.GetSize()
        w, h = rw_size[0], rw_size[1]
        
        # Convert client pixel coordinates (0,0 at top-left) to VTK display coordinates (0,0 at bottom-left)
        cw = float(client_w) if client_w else float(w)
        ch = float(client_h) if client_h else float(h)
        
        norm_x = float(click_x) / cw
        norm_y = float(click_y) / ch
        
        disp_x = norm_x * w
        disp_y = (1.0 - norm_y) * h

        # 1. First strategy: Exact 3D ray picking
        picker.Pick(disp_x, disp_y, 0, renderer)
        picked_actor = picker.GetActor()

        best_candidate = None
        min_world_dist = float("inf")

        if picked_actor is not None:
            pick_pos = picker.GetPickPosition()
            all_features = atoms + critical_points
            for item in all_features:
                raw_pos = item["raw_pos"]
                dx = raw_pos[0] - pick_pos[0]
                dy = raw_pos[1] - pick_pos[1]
                dz = raw_pos[2] - pick_pos[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < min_world_dist:
                    min_world_dist = dist
                    best_candidate = item

            # If ray intersection hit close to an atom or CP sphere center
            threshold = 1.35
            if best_candidate is not None and min_world_dist <= threshold:
                select_item(best_candidate)
                return

        # 2. Second strategy: Projected 2D screen-space distance
        # Ensures smooth picking even when clicking near edges of perspective-projected spheres
        min_screen_dist = float("inf")
        screen_candidate = None
        all_features = atoms + critical_points

        for item in all_features:
            raw_pos = item["raw_pos"]
            world_coord.SetValue(raw_pos[0], raw_pos[1], raw_pos[2])
            item_disp = world_coord.GetComputedDisplayValue(renderer)
            sdx = item_disp[0] - disp_x
            sdy = item_disp[1] - disp_y
            screen_dist = math.sqrt(sdx * sdx + sdy * sdy)
            if screen_dist < min_screen_dist:
                min_screen_dist = screen_dist
                screen_candidate = item

        # Maximum screen click tolerance in pixels (e.g., 40px around the sphere)
        if screen_candidate is not None and min_screen_dist <= 40.0:
            select_item(screen_candidate)
            return

        # If clicked outside or on empty space, deselect
        select_item(None)

    @ctrl.add("select_atom_from_list")
    def select_atom_from_list(atom_id):
        """Select an atom directly from the table/list in the drawer."""
        for a in atoms:
            if a["id"] == atom_id:
                select_item(a)
                return

    @ctrl.add("select_cp_from_list")
    def select_cp_from_list(cp_id):
        """Select a critical point directly from the table/list in the drawer."""
        for cp in critical_points:
            if cp["id"] == cp_id:
                select_item(cp)
                return

    @ctrl.add("clear_selection")
    def clear_selection():
        select_item(None)

    @ctrl.add("reset_camera")
    def reset_camera():
        renderer.ResetCamera()
        render_window.Render()
        if hasattr(ctrl, "view_update"):
            ctrl.view_update()

    # Build UI Layout with Collapsible Side Drawer
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Bondalyzer Molecule Viewer")
        layout.drawer.width = 410

        # --- DRAWER (Molecule Info & Feature Inspector) ---
        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-3"):

                # Primary Navigation Tabs: Overview, SCA Tools, GBA Tools
                with v3.VTabs(
                    v_model=("active_nav_mode", "overview"),
                    density="compact",
                    color="primary",
                    grow=True,
                    classes="mb-3 rounded elevation-1",
                ):
                    v3.VTab("Overview", value="overview", prepend_icon="mdi-molecule")
                    v3.VTab("SCA Tools", value="sca", prepend_icon="mdi-layers-outline")
                    v3.VTab("GBA Tools", value="gba", prepend_icon="mdi-chart-bubble")

                # =====================================================================
                # TAB 1: MOLECULE OVERVIEW & SKELETON
                # =====================================================================
                with v3.VWindow(v_model=("active_nav_mode", "overview")):
                    with v3.VWindowItem(value="overview"):

                        # 1. Molecule Summary Card
                        with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center"):
                                    v3.VIcon("mdi-molecule", classes="mr-2", color="primary")
                                    html.Span("{{ molecule_info.title }}")
                                v3.VCardSubtitle("Dataset: {{ vtm_file }}")

                            v3.VDivider()
                            with v3.VCardText(classes="pt-2 pb-2"):
                                with v3.VRow(dense=True):
                                    with v3.VCol(cols=6):
                                        html.Div("Formula", classes="text-caption text-medium-emphasis")
                                        html.Div("{{ molecule_info.formula }}", classes="text-h6 font-weight-bold text-primary")
                                    with v3.VCol(cols=6):
                                        html.Div("Total Atoms", classes="text-caption text-medium-emphasis")
                                        html.Div("{{ molecule_info.total_atoms }}", classes="text-h6 font-weight-bold")

                                with v3.VRow(dense=True, classes="mt-1"):
                                    with v3.VCol(cols=4):
                                        html.Div("Inferred Bonds", classes="text-caption text-medium-emphasis")
                                        html.Div("{{ molecule_info.bonds }}", classes="text-body-2 font-weight-medium")
                                    with v3.VCol(cols=4):
                                        html.Div("Bond Paths", classes="text-caption text-medium-emphasis")
                                        html.Div("{{ molecule_info.bond_paths }}", classes="text-body-2 font-weight-medium")
                                    with v3.VCol(cols=4):
                                        html.Div("Bond CPs (3,-1)", classes="text-caption text-medium-emphasis")
                                        html.Div("{{ molecule_info.bond_cps }}", classes="text-body-2 font-weight-bold text-error")

                        # 2. Selected Feature Details Card (Appears on click/selection)
                        with v3.VCard(
                            v_if="selected_item",
                            elevation=3,
                            classes="mb-3 border-primary",
                            color="surface",
                        ):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"):
                                    with html.Div(classes="d-flex align-center"):
                                        v3.VIcon("mdi-crosshairs-gps", classes="mr-2", color="amber-darken-2")
                                        html.Span("{{ selected_item.name }}")
                                    v3.VBtn(
                                        icon="mdi-close",
                                        variant="text",
                                        density="compact",
                                        click=ctrl.clear_selection,
                                    )
                                v3.VCardSubtitle("Zone: {{ selected_item.block }}")

                            v3.VDivider()
                            with v3.VCardText(classes="pt-2"):
                                with v3.VList(density="compact", lines=False, classes="pa-0"):
                                    with v3.VListItem(v_if="selected_item.element", classes="px-0"):
                                        v3.VListItemTitle("Element & Atomic Number")
                                        v3.VListItemSubtitle("{{ selected_item.element }} (Z = {{ selected_item.atomic_number }})")

                                    with v3.VListItem(v_if="selected_item.type", classes="px-0"):
                                        v3.VListItemTitle("Classification & Signature")
                                        v3.VListItemSubtitle("{{ selected_item.type }} {{ selected_item.signature }}")

                                    with v3.VListItem(classes="px-0"):
                                        v3.VListItemTitle("Position (X, Y, Z)")
                                        v3.VListItemSubtitle("({{ selected_item.position[0] }}, {{ selected_item.position[1] }}, {{ selected_item.position[2] }})")

                                    with v3.VListItem(classes="px-0"):
                                        v3.VListItemTitle("Electron Density (ρ)")
                                        v3.VListItemSubtitle("{{ selected_item.electron_density }} a.u.")

                        # Prompt if no item selected
                        with v3.VAlert(
                            v_if="!selected_item",
                            type="info",
                            variant="tonal",
                            density="compact",
                            classes="mb-3 text-caption",
                        ):
                            html.Div("Click any atom or critical point sphere in the 3D view to inspect its properties.")

                        # 3. Atoms List
                        with v3.VCard(elevation=1):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"):
                                    html.Span("Atoms in Dataset")
                                    v3.VChip("{{ atoms_list.length }} atoms", size="x-small", color="primary")

                            v3.VDivider()
                            with v3.VList(density="compact", nav=True, classes="py-0", max_height="250px"):
                                with v3.VListItem(
                                    v_for="atom in atoms_list",
                                    key="atom.id",
                                    classes="my-1",
                                    click=(ctrl.select_atom_from_list, "[atom.id]"),
                                ):
                                    with html.Template(v_slot_prepend=True):
                                        v3.VChip(
                                            "{{ atom.element }}",
                                            size="x-small",
                                            color="primary",
                                            classes="mr-2 font-weight-bold",
                                        )
                                    v3.VListItemTitle("{{ atom.name }} (Z={{ atom.atomic_number }})")
                                    v3.VListItemSubtitle("Pos: ({{ atom.position[0] }}, {{ atom.position[1] }}, {{ atom.position[2] }})")

                    # =================================================================
                    # TAB 2: SCA TOOLS (Scalar Field Analysis)
                    # =================================================================
                    with v3.VWindowItem(value="sca"):
                        with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center"):
                                    v3.VIcon("mdi-layers-outline", classes="mr-2", color="primary")
                                    html.Span("Scalar Field Analysis (SCA)")
                                v3.VCardSubtitle("Cutplane contours & isosurfaces")

                            v3.VDivider()
                            with v3.VCardText(classes="pt-3 pb-2"):
                                # 1. Global 3D Scalar Field Selector
                                with v3.VSelect(
                                    label="Select 3D Scalar Field",
                                    items=("molecule_info.global_fields",),
                                    v_model=("selected_global_field",),
                                    density="compact",
                                    variant="outlined",
                                    classes="mb-3",
                                ):
                                    with html.Template(v_slot_prepend_inner=True):
                                        html.Span("f(ρ)", classes="font-italic font-weight-bold text-primary mr-1", style="font-size: 0.95rem; line-height: 1;")

                                # 2. Visualization Mode Toggle: Cutplane vs Isosurface
                                html.Div("Visualization Mode", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                with v3.VBtnToggle(
                                    v_model=("sca_visualization_mode", "cutplane"),
                                    density="compact",
                                    color="primary",
                                    mandatory=True,
                                    classes="mb-3 d-flex justify-center",
                                ):
                                    v3.VBtn("Cutplane Contours", value="cutplane", size="small", prepend_icon="mdi-vector-square")
                                    v3.VBtn("Isosurfaces", value="isosurface", size="small", prepend_icon="mdi-blur-radial")

                                # 3A. Cutplane Contour Controls
                                with html.Div(v_if="sca_visualization_mode === 'cutplane'"):
                                    html.Div("Slice Plane Orientation", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                    with v3.VBtnToggle(density="compact", color="primary", mandatory=True, classes="mb-3 d-flex justify-center"):
                                        v3.VBtn("XY Plane", value="XY", size="small")
                                        v3.VBtn("XZ Plane", value="XZ", size="small")
                                        v3.VBtn("YZ Plane", value="YZ", size="small")

                                    html.Div("Slice Position Offset", classes="text-caption font-weight-bold text-medium-emphasis")
                                    v3.VSlider(
                                        min=-3.0,
                                        max=3.0,
                                        step=0.05,
                                        density="compact",
                                        thumb_label="always",
                                        color="primary",
                                        classes="mt-1",
                                    )

                                    v3.VDivider(classes="my-2")

                                    with v3.VRow(dense=True, classes="align-center"):
                                        with v3.VCol(cols=6):
                                            v3.VSwitch(
                                                label="Show Contours",
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                            )
                                        with v3.VCol(cols=6):
                                            v3.VSwitch(
                                                label="Color Flood",
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                            )

                                # 3B. Isosurface Controls
                                with html.Div(v_if="sca_visualization_mode === 'isosurface'"):
                                    html.Div("Isosurface Value", classes="text-caption font-weight-bold text-medium-emphasis")
                                    v3.VSlider(
                                        min=0.0,
                                        max=1.0,
                                        step=0.01,
                                        density="compact",
                                        thumb_label="always",
                                        color="primary",
                                        classes="mt-1",
                                    )

                                    html.Div("Number of Isosurface Levels", classes="text-caption font-weight-bold text-medium-emphasis mt-2")
                                    v3.VSlider(
                                        min=1,
                                        max=20,
                                        step=1,
                                        density="compact",
                                        thumb_label="always",
                                        color="primary",
                                        classes="mt-1",
                                    )

                                    v3.VDivider(classes="my-2")

                                    v3.VSwitch(
                                        label="Enable Opacity / Transparency",
                                        density="compact",
                                        color="primary",
                                        hide_details=True,
                                    )

                    # =================================================================
                    # TAB 3: GBA TOOLS (Atomic Basin Analysis)
                    # =================================================================
                    with v3.VWindowItem(value="gba"):
                        with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center"):
                                    v3.VIcon("mdi-chart-bubble", classes="mr-2", color="success")
                                    html.Span("GBA Tools")
                                v3.VCardSubtitle("Gradient bundle basin analysis")

                            v3.VDivider()
                            with v3.VCardText(classes="pt-3 pb-2"):
                                v3.VSelect(
                                    label="Select GBA Atom",
                                    items=("molecule_info.gba_atoms_list",),
                                    model_value="C1",
                                    density="compact",
                                    variant="outlined",
                                    prepend_inner_icon="mdi-atom",
                                    classes="mb-2",
                                )

                                with v3.VSelect(
                                    label="Select Condensed Field",
                                    items=("molecule_info.gba_fields",),
                                    v_model=("selected_condensed_field",),
                                    density="compact",
                                    variant="outlined",
                                    classes="mb-3",
                                ):
                                    with html.Template(v_slot_prepend_inner=True):
                                        html.Span("F[ρ]", classes="font-italic font-weight-bold text-success mr-1", style="font-size: 0.95rem; line-height: 1;")

                        with v3.VCard(elevation=1):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center"):
                                    v3.VIcon("mdi-eye-outline", classes="mr-2")
                                    html.Span("Basin Surfaces")

                            v3.VDivider()
                            with v3.VCardText(classes="pt-2"):
                                v3.VSwitch(
                                    label="Show Atom Sphere Boundary",
                                    model_value=True,
                                    density="compact",
                                    color="primary",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    label="Show Minimum Basin Surfaces",
                                    model_value=True,
                                    density="compact",
                                    color="primary",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    label="Show Maximum Basin Surfaces",
                                    model_value=False,
                                    density="compact",
                                    color="secondary",
                                    hide_details=True,
                                )

        # --- TOOLBAR ---
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

        # --- 3D VIEWPORT ---
        with layout.content:
            with html.Div(
                style="position: relative; width: 100%; height: 100%; cursor: pointer;",
                click=(
                    ctrl.on_scene_click,
                    "[$event.offsetX, $event.offsetY, $event.currentTarget.clientWidth, $event.currentTarget.clientHeight]",
                ),
            ):
                view = VtkRemoteView(
                    render_window,
                    interactive_ratio=1.0,
                )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    print(f"\n[Bondalyzer] Starting Trame application for: {vtm_path}")
    print(f"[Bondalyzer] Loaded {len(actors)} rendered blocks.")
    print(f"[Bondalyzer] Formula: {molecule_info['formula']} ({len(atoms)} atoms, {len(critical_points)} critical points)")
    server.start()


def run_native_vtk_window(vtm_path: str):
    """
    Fallback native VTK interactive window if trame is not installed in the python environment.
    """
    print(f"\n[Bondalyzer] Trame not found. Launching native VTK window for: {vtm_path}")
    (
        renderer,
        render_window,
        actors,
        molecule_info,
        atoms,
        critical_points,
        highlight_actor,
        highlight_source,
    ) = create_visualization_pipeline(vtm_path)

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
