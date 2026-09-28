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
import re
import sys
import math
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# VTK Imports
import vtk
import vtkmodules.vtkRenderingOpenGL2  # Ensure OpenGL2 backend is properly initialized
from vtkmodules.vtkIOXML import (
    vtkXMLMultiBlockDataReader,
    vtkXMLImageDataReader,
    vtkXMLRectilinearGridReader,
    vtkXMLStructuredGridReader,
)
from vtkmodules.vtkFiltersCore import (
    vtkGlyph3D,
    vtkTubeFilter,
    vtkFlyingEdges3D,
    vtkContourFilter,
    vtkCutter,
    vtkFeatureEdges,
    vtkClipPolyData,
)
from vtkmodules.vtkImagingCore import vtkExtractVOI
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPlanes
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkPolyDataMapper,
    vtkActor,
    vtkProperty,
    vtkCellPicker,
    vtkCoordinate,
    vtkColorTransferFunction,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

try:
    from plt_gba_to_vtm import (
        extract_gba_zones_from_plt,
        convert_1d_zones_to_vtm,
        convert_zone0_to_vtk,
        build_pyramidal_wedge_polydata,
    )
except ImportError:
    try:
        from BondalyzerParaView.plt_gba_to_vtm import (
            extract_gba_zones_from_plt,
            convert_1d_zones_to_vtm,
            convert_zone0_to_vtk,
            build_pyramidal_wedge_polydata,
        )
    except ImportError:
        from bondalyzer_viewer.plt_gba_to_vtm import (
            extract_gba_zones_from_plt,
            convert_1d_zones_to_vtm,
            convert_zone0_to_vtk,
            build_pyramidal_wedge_polydata,
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


# Supported dataset extensions for file selection
DATASET_EXTENSIONS = (".vtm", ".plt", ".vti", ".vtr", ".vtp")


def open_native_file_dialog(initial_dir: Optional[str] = None) -> Optional[str]:
    """
    Open a native OS file selection dialog (macOS, Windows, Linux) via subprocess,
    returning the chosen file path or None if cancelled/unavailable.
    Zero external GUI library dependencies (safe in pvpython and standard python).
    """
    init_path = os.path.abspath(initial_dir or os.getcwd())
    if not os.path.isdir(init_path):
        init_path = os.path.dirname(init_path) or os.getcwd()

    # 1. macOS (osascript / Cocoa AppleScript dialog)
    if sys.platform == "darwin":
        try:
            osa_script = (
                f'set defaultPath to POSIX file "{init_path}"\n'
                'try\n'
                '    set chosenFile to choose file with prompt "Select Bondalyzer Dataset (.plt or .vtm):" '
                'of type {"vtm", "plt", "vti", "vtr", "vtp"} default location defaultPath\n'
                '    return POSIX path of chosenFile\n'
                'on error\n'
                '    return ""\n'
                'end try'
            )
            res = subprocess.run(
                ["osascript", "-e", osa_script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = res.stdout.strip()
            return out if out and os.path.exists(out) else None
        except Exception as e:
            print(f"[Bondalyzer] Native macOS file dialog notice: {e}")

    # 2. Windows (PowerShell / .NET OpenFileDialog)
    elif sys.platform == "win32":
        try:
            ps_script = (
                'Add-Type -AssemblyName System.Windows.Forms;'
                '$f = New-Object System.Windows.Forms.OpenFileDialog;'
                '$f.Title = "Select Bondalyzer Dataset";'
                f'$f.InitialDirectory = "{init_path}";'
                '$f.Filter = "Bondalyzer Datasets (*.vtm;*.plt;*.vti;*.vtr;*.vtp)|*.vtm;*.plt;*.vti;*.vtr;*.vtp|All Files (*.*)|*.*";'
                'if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }'
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = res.stdout.strip()
            return out if out and os.path.exists(out) else None
        except Exception as e:
            print(f"[Bondalyzer] Native Windows file dialog notice: {e}")

    # 3. Linux (zenity or kdialog)
    elif sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            try:
                res = subprocess.run(
                    [
                        "zenity",
                        "--file-selection",
                        "--title=Select Bondalyzer Dataset",
                        f"--filename={init_path}/",
                        '--file-filter=Bondalyzer Datasets | *.vtm *.plt *.vti *.vtr *.vtp',
                        '--file-filter=All Files | *',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                out = res.stdout.strip()
                return out if out and os.path.exists(out) else None
            except Exception as e:
                print(f"[Bondalyzer] zenity dialog notice: {e}")
        elif shutil.which("kdialog"):
            try:
                res = subprocess.run(
                    [
                        "kdialog",
                        "--title",
                        "Select Bondalyzer Dataset",
                        "--getopenfilename",
                        init_path,
                        "*.vtm *.plt *.vti *.vtr *.vtp|Bondalyzer Datasets",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                out = res.stdout.strip()
                return out if out and os.path.exists(out) else None
            except Exception as e:
                print(f"[Bondalyzer] kdialog notice: {e}")

    # 4. Fallback: tkinter.filedialog if available in python environment
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            initialdir=init_path,
            title="Select Bondalyzer Dataset",
            filetypes=[
                ("Bondalyzer Datasets", "*.vtm;*.plt;*.vti;*.vtr;*.vtp"),
                ("Tecplot PLT", "*.plt"),
                ("VTK MultiBlock", "*.vtm"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return chosen if chosen and os.path.exists(chosen) else None
    except Exception:
        pass

    return None


def list_server_directory(dir_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan a directory on the server for the in-app Vuetify file browser modal.
    Returns path metadata, parent path, and entries list.
    """
    target = os.path.abspath(dir_path or os.getcwd())
    if not os.path.exists(target):
        target = os.getcwd()
    if not os.path.isdir(target):
        target = os.path.dirname(target) or os.getcwd()

    parent = os.path.dirname(target)
    if parent == target:
        parent = None

    entries = []
    try:
        with os.scandir(target) as it:
            for entry in it:
                # Skip hidden files
                if entry.name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir(follow_symlinks=True)
                    ext = os.path.splitext(entry.name)[1].lower() if not is_dir else ""
                    is_supported = is_dir or (ext in DATASET_EXTENSIONS)
                    stat = entry.stat()
                    size_str = ""
                    if not is_dir:
                        s_bytes = stat.st_size
                        if s_bytes < 1024:
                            size_str = f"{s_bytes} B"
                        elif s_bytes < 1024 * 1024:
                            size_str = f"{s_bytes / 1024:.1f} KB"
                        else:
                            size_str = f"{s_bytes / (1024 * 1024):.1f} MB"

                    entries.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": is_dir,
                        "ext": ext,
                        "is_supported": is_supported,
                        "size": size_str,
                    })
                except OSError:
                    continue
    except OSError as e:
        print(f"[Bondalyzer] Error scanning directory {target}: {e}")

    # Sort: folders first (alphabetical), then supported datasets, then others
    entries.sort(key=lambda x: (not x["is_dir"], not x["is_supported"], x["name"].lower()))

    # Build breadcrumb segments
    p = Path(target)
    parts = []
    cum = Path(p.anchor)
    parts.append({"name": str(p.anchor) or "/", "path": str(cum)})
    for part in p.parts:
        if part in (p.anchor, "/", ""):
            continue
        cum = cum / part
        parts.append({"name": part, "path": str(cum)})

    return {
        "current_path": target,
        "parent_path": parent,
        "breadcrumbs": parts,
        "entries": entries,
    }


def get_covalent_radius(element_symbol: str, default: float = 0.75) -> float:
    """Retrieve covalent radius in Angstroms for an element symbol."""
    return COVALENT_RADII.get(element_symbol.strip(), default)


# Physical slider ranges (min, max, default, step) for chemically meaningful isosurfaces
# Computed from the molecular bonding region (where ρ > 0.001 a.u.)
FIELD_ISOSURFACE_RANGES = {
    "electron density": (0.001, 0.50, 0.05, 0.005),
    "kinetic": (0.01, 10.0, 1.0, 0.1),
    "modified willmore": (0.001, 1.50, 0.05, 0.01),
    "willmore energy": (0.01, 3.50, 0.20, 0.02),
    "shape index": (0.00, 1.00, 0.70, 0.01),
    "curvedness": (0.05, 2.50, 0.50, 0.02),
    "gaussian curvature": (-1.00, 3.00, 0.10, 0.02),
    "mean curvature": (-0.50, 2.00, 0.30, 0.02),
    "v": (0.0, 1.0, 0.5, 0.01),
    "trajectory parameter": (0.0, 1.0, 0.5, 0.01),
}


# -----------------------------------------------------------------------------
# Crystal Structure & Wigner-Seitz Masking Functions (Periodic Calculations)
# -----------------------------------------------------------------------------
# TODO (Bondalyzer Exporter - Tim):
# In future Bondalyzer updates, write crystal periodicity and lattice vectors
# directly into Tecplot dataset auxiliary metadata (Marker 799.0f):
#   - Aux_Periodic: "True" | "False"
#   - Aux_Lattice_A: "ax ay az"
#   - Aux_Lattice_B: "bx by bz"
#   - Aux_Lattice_C: "cx cy cz"
#   - Aux_UnitCell_Origin: "x0 y0 z0"
# In the meantime, this viewer checks for a companion .run / .in file (e.g. Pd.run)
# or dataset FieldData to automatically extract the lattice and construct the
# exact Wigner-Seitz mask for periodic crystals.
# -----------------------------------------------------------------------------

def parse_companion_run_lattice(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse crystal lattice vectors and atom positions from an electronic structure
    input file (.run / .in / .ams) companion to the dataset.
    Lattice vectors in .run files are specified in Angstroms and converted to Bohr
    (1 Å = 1.8897261246 Bohr) to match PLT dataset coordinate space.
    """
    if not file_path:
        return None

    ANGSTROM_TO_BOHR = 1.88972612462577

    base_path = os.path.abspath(file_path)
    dir_name = os.path.dirname(base_path)
    stem = os.path.splitext(os.path.basename(base_path))[0].replace("_1d_zones", "").replace("_zone0", "")

    candidates = [
        os.path.join(dir_name, f"{stem}.run"),
        os.path.join(dir_name, f"{stem}.in"),
        os.path.join(dir_name, f"{stem}.cell"),
        os.path.join(os.getcwd(), f"{stem}.run"),
    ]

    run_file = None
    for cand in candidates:
        if os.path.exists(cand):
            run_file = cand
            break

    if not run_file:
        return None

    lattice_vecs = []
    atoms = []
    try:
        with open(run_file, "r", errors="replace") as f:
            lines = f.readlines()

        in_lattice = False
        in_atoms = False

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            if lower.startswith("lattice"):
                in_lattice = True
                continue
            elif in_lattice:
                if lower.startswith("end") or lower.startswith("bondorders") or lower.startswith("engine"):
                    in_lattice = False
                else:
                    parts = stripped.split()
                    if len(parts) >= 3:
                        try:
                            vec = [float(parts[0]) * ANGSTROM_TO_BOHR, float(parts[1]) * ANGSTROM_TO_BOHR, float(parts[2]) * ANGSTROM_TO_BOHR]
                            lattice_vecs.append(vec)
                        except ValueError:
                            pass

            if lower.startswith("atoms"):
                in_atoms = True
                continue
            elif in_atoms:
                if lower.startswith("end") or lower.startswith("lattice") or lower.startswith("bondorders"):
                    in_atoms = False
                else:
                    parts = stripped.split()
                    if len(parts) >= 4:
                        el = parts[0]
                        try:
                            pos = [float(parts[1]) * ANGSTROM_TO_BOHR, float(parts[2]) * ANGSTROM_TO_BOHR, float(parts[3]) * ANGSTROM_TO_BOHR]
                            atoms.append({"element": el, "position": pos})
                        except ValueError:
                            pass

        if len(lattice_vecs) == 3:
            origin = atoms[0]["position"] if atoms else [0.0, 0.0, 0.0]
            return {
                "is_periodic": True,
                "source": os.path.basename(run_file),
                "lattice_a": np.array(lattice_vecs[0], dtype=np.float64),
                "lattice_b": np.array(lattice_vecs[1], dtype=np.float64),
                "lattice_c": np.array(lattice_vecs[2], dtype=np.float64),
                "origin": np.array(origin, dtype=np.float64),
                "atoms": atoms,
            }
    except Exception as e:
        print(f"[Bondalyzer] Warning parsing companion run file '{run_file}': {e}")

    return None


def extract_crystal_lattice_info(dataset_path: str, volume_grid=None, mb=None, atoms=None) -> Optional[Dict[str, Any]]:
    """
    Determine whether a dataset represents a periodic crystal and return lattice parameters.
    Lattice vectors are expressed in Bohr to match dataset coordinate space.
    Checks (in priority):
      1. Dataset FieldData embedded by converters (Aux_Periodic / Aux_Lattice_A/B/C)
      2. Companion electronic structure input file (.run / .in)
      3. Hard-coded dev fallback for Pd.plt
    """
    ANGSTROM_TO_BOHR = 1.88972612462577
    lattice_info = None

    # 1. Check FieldData on volume_grid or multiblock
    target_fd = None
    if volume_grid is not None and volume_grid.GetFieldData():
        target_fd = volume_grid.GetFieldData()

    if target_fd and target_fd.HasArray("Aux_Periodic"):
        p_val = str(target_fd.GetAbstractArray("Aux_Periodic").GetValue(0)).strip().lower()
        if p_val in ("true", "1", "yes"):
            try:
                def parse_vec(arr_name: str, def_v: List[float]) -> np.ndarray:
                    if target_fd.HasArray(arr_name):
                        v_str = str(target_fd.GetAbstractArray(arr_name).GetValue(0)).strip()
                        return np.array([float(x) for x in v_str.split()[:3]], dtype=np.float64)
                    return np.array(def_v, dtype=np.float64)

                la = parse_vec("Aux_Lattice_A", [1.0, 0.0, 0.0])
                lb = parse_vec("Aux_Lattice_B", [0.0, 1.0, 0.0])
                lc = parse_vec("Aux_Lattice_C", [0.0, 0.0, 1.0])
                orig = parse_vec("Aux_UnitCell_Origin", [0.0, 0.0, 0.0])
                lattice_info = {
                    "is_periodic": True,
                    "source": "PLT FieldData",
                    "lattice_a": la,
                    "lattice_b": lb,
                    "lattice_c": lc,
                    "origin": orig,
                    "atoms": [],
                }
            except Exception:
                pass

    # 2. Check companion .run file
    if lattice_info is None:
        lattice_info = parse_companion_run_lattice(dataset_path)

    # 3. Development fallback for Pd.plt if run file is missing
    if lattice_info is None:
        bname = os.path.basename(dataset_path or "").lower()
        if "pd" in bname:
            lattice_info = {
                "is_periodic": True,
                "source": "Dev default (Pd distorted FCC)",
                "lattice_a": np.array([2.7506 * ANGSTROM_TO_BOHR, 0.0, 0.0], dtype=np.float64),
                "lattice_b": np.array([0.0, 2.7506 * ANGSTROM_TO_BOHR, 0.0], dtype=np.float64),
                "lattice_c": np.array([1.3753 * ANGSTROM_TO_BOHR, 1.3753 * ANGSTROM_TO_BOHR, 1.9450 * ANGSTROM_TO_BOHR], dtype=np.float64),
                "origin": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "atoms": [{"element": "Pd", "position": [0.0, 0.0, 0.0]}],
            }

    if lattice_info is None or not lattice_info.get("is_periodic", False):
        return None

    # Origin center (atom position in dataset)
    if atoms and len(atoms) > 0 and "raw_pos" in atoms[0]:
        lattice_info["origin"] = np.array(atoms[0]["raw_pos"], dtype=np.float64)

    return lattice_info


def build_wigner_seitz_planes(lattice_info: Dict[str, Any]) -> Tuple[vtkPlanes, Optional[vtkActor]]:
    """
    Construct a vtkPlanes convex implicit function representing the Wigner-Seitz cell
    of the Bravais lattice directly from lattice translation vectors (in Bohr),
    along with a wireframe polydata actor for 3D boundary rendering.
    """
    a1 = lattice_info["lattice_a"]
    a2 = lattice_info["lattice_b"]
    a3 = lattice_info["lattice_c"]
    center = lattice_info.get("origin", np.array([0.0, 0.0, 0.0], dtype=np.float64))

    # Generate 26 surrounding lattice translation vectors R = n1*a1 + n2*a2 + n3*a3
    trans_vectors = []
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            for n3 in (-1, 0, 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                R = n1 * a1 + n2 * a2 + n3 * a3
                d2 = float(np.dot(R, R))
                trans_vectors.append((d2, R))

    trans_vectors.sort(key=lambda x: x[0])
    min_d2 = trans_vectors[0][0]
    cutoff_d2 = min_d2 * 2.25

    plane_normals = []
    plane_points = []

    for d2, R in trans_vectors:
        if d2 > cutoff_d2:
            break
        norm = math.sqrt(d2)
        if norm < 1e-6:
            continue
        n_hat = R / norm
        p_mid = center + 0.5 * R
        plane_normals.append(n_hat)
        plane_points.append(p_mid)

    # Build vtkPlanes implicit function for clipping
    vtk_pts = vtk.vtkPoints()
    vtk_nrms = vtk.vtkDoubleArray()
    vtk_nrms.SetNumberOfComponents(3)
    vtk_nrms.SetName("Normals")

    for p, n in zip(plane_points, plane_normals):
        vtk_pts.InsertNextPoint(p[0], p[1], p[2])
        vtk_nrms.InsertNextTuple3(n[0], n[1], n[2])

    ws_planes = vtkPlanes()
    ws_planes.SetPoints(vtk_pts)
    ws_planes.SetNormals(vtk_nrms)

    # Build 3D Wigner-Seitz Polyhedron wireframe actor via exact half-space intersections
    ws_actor = None
    try:
        num_p = len(plane_normals)
        raw_vertices = []
        d_vals = [float(np.dot(plane_normals[k], plane_points[k])) for k in range(num_p)]
        r_max_sq = min_d2 * 3.5

        for i in range(num_p):
            for j in range(i + 1, num_p):
                for k in range(j + 1, num_p):
                    M = np.array([plane_normals[i], plane_normals[j], plane_normals[k]], dtype=np.float64)
                    if abs(np.linalg.det(M)) < 1e-5:
                        continue
                    b = np.array([d_vals[i], d_vals[j], d_vals[k]], dtype=np.float64)
                    try:
                        v = np.linalg.solve(M, b)
                    except np.linalg.LinAlgError:
                        continue

                    dist_to_center_sq = float(np.sum((v - center) ** 2))
                    if dist_to_center_sq > r_max_sq:
                        continue

                    inside = True
                    for m in range(num_p):
                        if np.dot(plane_normals[m], v) > d_vals[m] + 1e-4:
                            inside = False
                            break
                    if inside:
                        raw_vertices.append(v)

        unique_vertices = []
        for v in raw_vertices:
            if not any(np.allclose(v, uv, atol=1e-3) for uv in unique_vertices):
                unique_vertices.append(v)

        if len(unique_vertices) >= 4:
            poly_points = vtk.vtkPoints()
            for uv in unique_vertices:
                poly_points.InsertNextPoint(uv[0], uv[1], uv[2])

            cloud_poly = vtk.vtkPolyData()
            cloud_poly.SetPoints(poly_points)

            delaunay = vtk.vtkDelaunay3D()
            delaunay.SetInputData(cloud_poly)
            delaunay.Update()

            geom_filter = vtk.vtkDataSetSurfaceFilter()
            geom_filter.SetInputConnection(delaunay.GetOutputPort())
            geom_filter.Update()

            feat_edges = vtkFeatureEdges()
            feat_edges.SetInputConnection(geom_filter.GetOutputPort())
            feat_edges.BoundaryEdgesOn()
            feat_edges.FeatureEdgesOn()
            feat_edges.ManifoldEdgesOff()
            feat_edges.NonManifoldEdgesOff()
            feat_edges.SetFeatureAngle(15.0)

            tuber = vtkTubeFilter()
            tuber.SetInputConnection(feat_edges.GetOutputPort())
            tuber.SetRadius(0.022)
            tuber.SetNumberOfSides(16)
            tuber.CappingOn()
            tuber.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tuber.GetOutputPort())
            mapper.ScalarVisibilityOff()

            ws_actor = vtkActor()
            ws_actor.SetMapper(mapper)
            ws_actor.GetProperty().SetColor(1.0, 0.84, 0.0)  # Distinct Gold Wireframe
            ws_actor.GetProperty().SetAmbient(0.85)
            ws_actor.GetProperty().SetDiffuse(0.35)
            ws_actor.GetProperty().SetSpecular(0.50)
            ws_actor.GetProperty().SetSpecularPower(30)
            ws_actor.SetVisibility(False)
    except Exception as e:
        print(f"[Bondalyzer] Notice building WS wireframe: {e}")

    return ws_planes, ws_actor


# -----------------------------------------------------------------------------
# Display-title configuration tables
# -----------------------------------------------------------------------------
# Ordered suffix lookup: the FIRST key contained (case-insensitive) in the
# title-cased variable name wins, so more-specific keys must come BEFORE
# generic ones (e.g. 'modified willmore energy' before 'willmore energy').
# Map a key to "" to suppress a suffix for matching names.
FIELD_SUFFIX_LABELS: List[Tuple[str, str]] = [
    ("modified willmore energy", " (H²−K)"),
    ("willmore energy", " (H²)"),
    # Guard: sign-change descriptors are distinct fields, no (H) suffix
    ("mean curvature sign change", ""),
    ("positive mean curvature", " (H⁺)"),
    ("negative mean curvature", " (H⁻)"),
    ("mean curvature", " (H)"),
    ("gaussian curvature", " (K)"),
    ("shape index", " (S)"),
    ("curvedness", " (C)"),
    ("rms curvature", ""),
    ("electron density", " (ρ)"),
    ("kinetic energy density", " (τ)"),
    ("kinetic energy", " (K)"),
    ("volume", " (V)"),
    ("trajectory parameter", " (α)"),
]


def get_field_slider_config(field_name: str, raw_range: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """
    Return (min_val, max_val, default_val, step) for a scalar field,
    using domain-specific molecular ranges when available.
    """
    lower = field_name.lower()
    
    # Check whole-word 'v' or exact matches first to prevent letter 'v' matching inside 'curvature'
    if lower == "v" or lower.strip() == "v":
        return FIELD_ISOSURFACE_RANGES["v"]
    if lower == "î±" or lower == "α" or "alpha" in lower or "trajectory" in lower:
        return FIELD_ISOSURFACE_RANGES["trajectory parameter"]

    for key, (f_min, f_max, f_val, f_step) in FIELD_ISOSURFACE_RANGES.items():
        if key != "v" and key != "trajectory parameter" and key in lower:
            return f_min, f_max, f_val, f_step

    # Fallback to bounded raw range
    r_min, r_max = raw_range
    if r_max > r_min:
        f_min = max(r_min, -100.0)
        f_max = min(r_max, 100.0)
        f_val = f_min + (f_max - f_min) * 0.10
        f_step = (f_max - f_min) / 100.0
        return round(f_min, 4), round(f_max, 4), round(f_val, 4), round(f_step, 4)

    return 0.0, 1.0, 0.5, 0.01


def get_robust_scalar_bounds(arr, lower_pct: float = 2.0, upper_pct: float = 98.0) -> Tuple[float, float]:
    """
    Compute robust min/max bounds for a scalar array on a surface using percentiles
    to prevent single-point numerical singularity/asymptote spikes from distorting colormaps.
    """
    if arr is None or arr.GetNumberOfTuples() == 0:
        return (0.0, 1.0)

    try:
        from vtkmodules.util import numpy_support
        np_arr = numpy_support.vtk_to_numpy(arr)
        # Filter non-finite values if any
        valid = np_arr[np.isfinite(np_arr)]
        if len(valid) == 0:
            return arr.GetRange()
        
        q_low = float(np.percentile(valid, lower_pct))
        q_high = float(np.percentile(valid, upper_pct))

        # If constant or inverted, fallback to full range
        if q_high <= q_low:
            full_r = arr.GetRange()
            return full_r[0], full_r[1]

        return q_low, q_high
    except Exception:
        return arr.GetRange()


def normalize_field_name(name: str) -> str:
    """
    Clean and normalize scalar/condensed field names for robust comparison across
    PLT variable names, PointData arrays, and zone auxiliary FunctionName metadata.
    """
    if not name:
        return ""
    # Standardize Greek / mojibake characters
    s = str(name).replace("Ï\x81", "ρ").replace("Ï ", "ρ ").replace("Î±", "α")
    s = s.strip().lower()
    # Strip auxiliary qualifiers
    s = s.replace("(condensed)", "").replace("(3d)", "").replace("(sca)", "").replace("(gba)", "").strip()
    # Strip leading density prefixes ('ρ ', 'rho ')
    if s.startswith("ρ ") or s.startswith("ρ-"):
        s = s[2:].strip()
    elif s.startswith("rho ") or s.startswith("rho-"):
        s = s[4:].strip()
    return s


def matches_field(source_name: str, target_name: str) -> bool:
    """
    Check if a source field name (from PointData or zone FunctionName) matches a target field.
    Handles exact names, normalized representations, and canonical aliases.
    """
    if not source_name or not target_name:
        return False
    # 1. Exact string match
    if source_name == target_name or source_name.strip() == target_name.strip():
        return True

    # 2. Normalized string comparison
    norm_src = normalize_field_name(source_name)
    norm_tgt = normalize_field_name(target_name)

    if norm_src == norm_tgt and norm_src != "":
        return True

    # 3. Canonical aliases and mappings
    # Electron density
    if norm_tgt in ("ρ", "electron density", "density", "scf density"):
        return norm_src in ("ρ", "electron density", "density", "scf density", "")
    # Volume
    if norm_tgt in ("v", "volume"):
        return norm_src in ("v", "volume")
    # Kinetic energy
    if "kinetic" in norm_tgt:
        return "kinetic" in norm_src
    # Sign-change arc fraction / distance are distinct fields from plain mean curvature
    if "sign change" in norm_tgt:
        return "sign change" in norm_src and (
            ("arc fraction" in norm_tgt) == ("arc fraction" in norm_src)
        )
    # Curvatures & energies
    if "positive mean curvature" in norm_tgt:
        return "positive mean curvature" in norm_src or norm_src == "h+"
    if "negative mean curvature" in norm_tgt:
        return "negative mean curvature" in norm_src or norm_src == "h-"
    if "mean curvature" in norm_tgt and "positive" not in norm_tgt and "negative" not in norm_tgt and "sign change" not in norm_tgt:
        return (
            "mean curvature" in norm_src
            and "positive" not in norm_src
            and "negative" not in norm_src
            and "sign change" not in norm_src
        )
    if "gaussian curvature" in norm_tgt:
        return "gaussian curvature" in norm_src
    if "shape index" in norm_tgt:
        return "shape index" in norm_src
    if "curvedness" in norm_tgt:
        return "curvedness" in norm_src
    if "modified willmore" in norm_tgt:
        return "modified willmore" in norm_src
    if "willmore energy" in norm_tgt and "modified" not in norm_tgt:
        return "willmore energy" in norm_src and "modified" not in norm_src
    if "rms curvature" in norm_tgt or norm_tgt == "rms":
        return "rms curvature" in norm_src or "rms" in norm_src
    if norm_tgt in ("α", "trajectory parameter", "trajectory"):
        return norm_src in ("α", "trajectory parameter", "trajectory")

    return norm_tgt in norm_src or norm_src in norm_tgt


# Rich discrete palette of maximally distinct colors for neighboring basin coloring
GBA_DISTINCT_PALETTE: List[Tuple[float, float, float]] = [
    (0.89, 0.10, 0.11),  # Red
    (0.12, 0.47, 0.71),  # Blue
    (0.20, 0.63, 0.17),  # Green
    (1.00, 0.50, 0.00),  # Orange
    (0.42, 0.24, 0.60),  # Purple
    (1.00, 0.85, 0.10),  # Yellow-Gold
    (0.65, 0.34, 0.16),  # Brown
    (0.97, 0.51, 0.75),  # Pink
    (0.00, 0.75, 0.75),  # Cyan
    (0.70, 0.87, 0.54),  # Light Green
    (0.69, 0.71, 0.17),  # Olive
    (0.60, 0.60, 0.60),  # Gray
    (0.18, 0.80, 0.44),  # Emerald
    (0.91, 0.30, 0.24),  # Coral
    (0.20, 0.20, 0.75),  # Royal Blue
    (0.95, 0.77, 0.06),  # Amber
    (0.55, 0.00, 0.55),  # Magenta
    (0.00, 0.50, 0.50),  # Dark Teal
]


def assign_neighbor_aware_basin_colors(patches: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
    """
    Greedy graph coloring with distance-based penalty to ensure neighboring
    and physically adjacent basin patches receive contrasting distinct colors.
    """
    n = len(patches)
    if n == 0:
        return []
    if n == 1:
        return [GBA_DISTINCT_PALETTE[0]]

    # Compute bounding boxes, point sets, and centroids for adjacency/proximity
    patch_pts = []
    patch_centers = []
    for p in patches:
        poly = p.get("poly")
        pts_set = set()
        c_x, c_y, c_z = 0.0, 0.0, 0.0
        n_pts = 0
        if poly is not None and poly.GetNumberOfPoints() > 0:
            n_pts = poly.GetNumberOfPoints()
            for i in range(n_pts):
                pt = poly.GetPoint(i)
                pts_set.add((round(pt[0], 3), round(pt[1], 3), round(pt[2], 3)))
                c_x += pt[0]
                c_y += pt[1]
                c_z += pt[2]
            if n_pts > 0:
                c_x /= n_pts
                c_y /= n_pts
                c_z /= n_pts
        patch_pts.append(pts_set)
        patch_centers.append((c_x, c_y, c_z))

    # Build adjacency matrix (patches sharing boundary vertices or in close proximity)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            shared = patch_pts[i].intersection(patch_pts[j])
            # Connected if they share vertices on the sphere
            if len(shared) > 0:
                adj[i].add(j)
                adj[j].add(i)
            else:
                # Or if centroids are very close (neighboring small basins)
                ci, cj = patch_centers[i], patch_centers[j]
                d2 = (ci[0] - cj[0])**2 + (ci[1] - cj[1])**2 + (ci[2] - cj[2])**2
                if d2 < 0.60:
                    adj[i].add(j)
                    adj[j].add(i)

    # Greedy graph coloring ordered by degree
    assigned_color_idx: Dict[int, int] = {}
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)

    num_palette_colors = len(GBA_DISTINCT_PALETTE)

    for node in order:
        neighbor_colors = {assigned_color_idx[nbr] for nbr in adj[node] if nbr in assigned_color_idx}
        
        # Pick first available palette color not used by any adjacent neighbor
        chosen = None
        for c_idx in range(num_palette_colors):
            if c_idx not in neighbor_colors:
                chosen = c_idx
                break
        
        if chosen is None:
            # If all palette colors are used by neighbors, find the color with minimum neighbor frequency
            freq = {c_idx: 0 for c_idx in range(num_palette_colors)}
            for nbr in adj[node]:
                if nbr in assigned_color_idx:
                    freq[assigned_color_idx[nbr]] += 1
            min_count = min(freq.values())
            chosen = next(c_k for c_k, count in freq.items() if count == min_count)

        assigned_color_idx[node] = chosen

    return [GBA_DISTINCT_PALETTE[assigned_color_idx[i] % num_palette_colors] for i in range(n)]


def _clean_display_name(raw_name: str) -> str:
    """
    Clean a raw PLT/VTK variable name for display: fix Greek mojibake and strip
    parenthetical qualifiers such as ' (condensed)'.
    """
    s = str(raw_name or "").strip()
    s = s.replace("Ï\x81", "ρ").replace("Ï ", "ρ ").replace("Î±", "α")
    s = re.sub(r"\s*\(\s*(condensed|3d|sca|gba)\s*\)", "", s, flags=re.IGNORECASE).strip()

    # If the raw name is literally 'ρ', 'rho', or starts with 'ρ '/'rho ' without other text, expand to Electron Density
    if s in ("ρ", "rho", "Ï\x81") or s.lower() in ("ρ", "rho"):
        return "Electron Density"
    if s.lower().startswith("ρ ") or s.lower().startswith("rho "):
        # e.g. "ρ mean curvature" -> "Mean Curvature"
        remainder = s[2:].strip() if s.lower().startswith("ρ ") else s[4:].strip()
        return remainder

    return s.strip()


def _title_case_display(name: str) -> str:
    """
    Title-case a cleaned field name for display, preserving Greek letters
    (str.title() would uppercase 'ρ' -> 'Ρ') and acronyms such as 'ELF'.
    """
    words = []
    for word in name.split():
        first = word[0]
        if not first.isascii() or any(c.isupper() for c in word[1:]):
            words.append(word)
        else:
            words.append(first.upper() + word[1:].lower())
    return " ".join(words)


def get_display_title(raw_name: str) -> str:
    """
    Build a human-friendly display title from a raw variable name:
    clean mojibake, drop the '(condensed)' qualifier, title-case the name,
    and append a symbol suffix from FIELD_SUFFIX_LABELS when one matches.
    (Leading f(ρ)/F[ρ] functional badges are rendered separately in the UI.)
    """
    cleaned = _clean_display_name(raw_name)
    if not cleaned:
        return ""

    base = _title_case_display(cleaned)

    base_lower = base.lower()
    for key, label in FIELD_SUFFIX_LABELS:
        if key in base_lower:
            return f"{base}{label}"

    return base


# Keywords that mark a variable as a "secondary" (curvature-based isosurface)
# field. Secondary fields are pushed to the END of the field-selection
# dropdowns (SCA Tools / GBA Tools), while all other ("primary") variables
# keep their original order at the front of the list.
# Note: 'î±' is the mojibake form of 'α' (UTF-8 bytes decoded as Latin-1)
# that arrives from the PLT converters, matching get_field_slider_config().
SECONDARY_FIELD_KEYWORDS = ("curvature", "shape index", "willmore", "alpha", "α", "î±", 'curvedness')


def is_secondary_field(f_name: str) -> bool:
    """Return True if the variable name matches a secondary (curvature-based) keyword."""
    lower = f_name.lower()
    return any(kw in lower for kw in SECONDARY_FIELD_KEYWORDS)


def order_primary_secondary(names: List[str]) -> List[str]:
    """
    Stable partition of field names: primary (non-curvature) fields first,
    secondary (curvature-based) fields last, original order preserved within
    each group.
    """
    primary = [n for n in names if not is_secondary_field(n)]
    secondary = [n for n in names if is_secondary_field(n)]
    return primary + secondary


# When True (set via --force-convert), cached .vtm/.vts/.vti/.vtr conversions are always regenerated.
FORCE_CONVERT = False


def is_output_stale(out_path: str, src_path: Optional[str]) -> bool:
    """
    Return True if the converted output file is missing, or the source .plt is
    newer than it (meaning cached VTK outputs predate the current PLT/metadata).
    """
    if FORCE_CONVERT:
        return True
    if not os.path.exists(out_path):
        return True
    if not src_path or not os.path.exists(src_path):
        return False
    try:
        return os.path.getmtime(src_path) > os.path.getmtime(out_path)
    except OSError:
        return False


def read_variable_types_from_field_data(dataset) -> Dict[str, List[str]]:
    """
    Read per-variable categorization embedded as FieldData by the converters
    (parallel arrays: VariableNames[i] with Aux_VariableType[i]).
    Returns a dict mapping lower-cased VariableType -> list of variable names,
    or an empty dict when no embedded metadata is present.
    """
    type_map: Dict[str, List[str]] = {}
    if dataset is None:
        return type_map

    fd = dataset.GetFieldData()
    if fd is None or not fd.HasArray("VariableNames") or not fd.HasArray("Aux_VariableType"):
        return type_map

    names_arr = fd.GetAbstractArray("VariableNames")
    types_arr = fd.GetAbstractArray("Aux_VariableType")
    n = min(names_arr.GetNumberOfValues(), types_arr.GetNumberOfValues())

    def as_str(arr, i: int) -> str:
        val = arr.GetValue(i)
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)

    for i in range(n):
        vname = as_str(names_arr, i).strip()
        vtype = as_str(types_arr, i).strip()
        if not vname or not vtype:
            continue
        type_map.setdefault(vtype.lower(), []).append(vname)
    return type_map


def parse_dataset_metadata(mb, volume_grid=None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract high-level molecule metadata, catalog of atoms, and catalog of critical points
    from the MultiBlock dataset without duplicates.
    Filters global 3D scalar fields and GBA condensed fields using the VariableType
    metadata embedded as FieldData on the converted volume grid.
    """
    atoms = []
    element_counts = {}
    bonds = 0
    bond_paths = 0
    ring_paths = 0
    cage_paths = 0
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

        # 3. RING & CAGE PATHS
        elif n_lines > 0 and "ring" in name.lower():
            ring_paths += 1
        elif n_lines > 0 and "cage" in name.lower():
            cage_paths += 1

        # 4. INFERRED BONDS (Connectivity mesh)
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

    # Read VariableType categorization embedded as FieldData on the converted
    # volume grid by the PLT converters (VariableNames / Aux_VariableType):
    # - Scalar 3D fields: VariableType in ('Scaler3DField', 'Scalar3DField')
    # - Condensed fields: VariableType == 'DGBCondensedField'
    var_type_map = read_variable_types_from_field_data(volume_grid)
    if not var_type_map and volume_grid is not None:
        print("[Bondalyzer] Warning: Volume grid carries no embedded VariableType FieldData. "
              "Field lists will use fallbacks. Delete cached *_zone0.vts/.vti/.vtr (or touch the .plt) "
              "to regenerate with metadata.")

    def vars_of_types(types: Tuple[str, ...]) -> List[str]:
        out = []
        for t in types:
            out.extend(var_type_map.get(t, []))
        return out

    # 1. POPULATE 3D SCALAR FIELDS (for SCA Tools)
    # Filter for VariableType in ('Scaler3DField', 'Scalar3DField') (case-insensitive)
    scalar_3d_vars = vars_of_types(("scaler3dfield", "scalar3dfield"))

    if scalar_3d_vars:
        raw_global_fields = scalar_3d_vars
    else:
        # Fallback to collecting from block arrays if PLT VariableType aux is unavailable
        # (dict preserves discovery order for the primary/secondary partitioning below)
        raw_global_fields = {}
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
                    raw_global_fields[aname] = None

    # Primary (non-curvature) fields first, curvature-based isosurface
    # functions last; original order preserved within each group.
    sorted_raw_fields = order_primary_secondary(list(raw_global_fields))

    # Build list of items for Vuetify VSelect: [{title: 'Mean Curvature (H)', value: 'Ï\x81 mean curvature'}, ...]
    global_field_items = [
        {"title": get_display_title(f), "value": f}
        for f in sorted_raw_fields
    ]

    # 2. POPULATE CONDENSED FIELDS (for GBA Tools)
    # Filter for VariableType == 'DGBCondensedField' (case-insensitive)
    condensed_vars = vars_of_types(("dgbcondensedfield",))

    if condensed_vars:
        sorted_condensed = order_primary_secondary(condensed_vars)
        gba_condensed_field_items = [
            {"title": get_display_title(f), "value": f}
            for f in sorted_condensed
        ]
    else:
        # Fallback canonical list of condensed fields if PLT VariableType aux is unavailable.
        # Display titles are derived from the canonical name via get_display_title().
        fallback_condensed = [
            "Electron Density",
            "V",
            "Mean Curvature",
            "Positive Mean Curvature",
            "Negative Mean Curvature",
            "Gaussian Curvature",
            "Shape Index",
            "Curvedness",
            "Willmore Energy",
            "Modified Willmore Energy",
            "RMS Curvature",
        ]
        gba_condensed_field_items = [
            {"title": get_display_title(name), "value": name}
            for name in order_primary_secondary(fallback_condensed)
        ]

    default_condensed = gba_condensed_field_items[0]["value"] if gba_condensed_field_items else "Electron Density"

    system_title = f"{formula} Solid" if (len(atoms) > 0 and atoms[0].get("element") not in ("H", "C", "N", "O") and len(atoms) == 1) else f"Molecule ({formula})"
    if formula == "C2H4":
        system_title = "Ethene (C2H4)"

    default_gba_atoms = [a["name"] for a in atoms] if atoms else ["C1"]

    molecule_info = {
        "formula": formula,
        "title": system_title,
        "total_atoms": len(atoms),
        "element_counts": element_counts,
        "bonds": bonds,
        "bond_paths": bond_paths,
        "ring_paths": ring_paths,
        "cage_paths": cage_paths,
        "bond_cps": cp_counts["bond"],
        "ring_cps": cp_counts["ring"],
        "cage_cps": cp_counts["cage"],
        "total_cps": cp_counts["bond"] + cp_counts["ring"] + cp_counts["cage"],
        "num_blocks": num_blocks,
        "global_fields": global_field_items,
        "selected_global_field": sorted_raw_fields[0] if sorted_raw_fields else "Electron Density",
        "gba_atoms_list": default_gba_atoms,
        "gba_fields": gba_condensed_field_items,
        "selected_condensed_field": default_condensed,
    }

    return molecule_info, atoms, critical_points


def resolve_plt_for(vtm_path: str) -> Optional[str]:
    """Find the companion .plt used to (re)generate a .vtm file."""
    cand = vtm_path.replace("_1d_zones.vtm", ".plt").replace(".vtm", ".plt")
    if os.path.exists(cand):
        return cand
    base = os.path.splitext(os.path.basename(vtm_path))[0].replace("_1d_zones", "")
    if os.path.exists(f"{base}.plt"):
        return f"{base}.plt"
    for fb in ("ethene4.plt", "ethene2.plt", "ethene.plt"):
        if os.path.exists(fb):
            return fb
    return None


def prepare_dataset_file(file_path: str) -> str:
    """
    Dynamically resolve and convert a .plt or .vtm file into an active .vtm dataset.
    Converts on-demand if the target .vtm is missing or stale relative to the source .plt.
    Returns the absolute or valid path to the resulting .vtm file.
    """
    if not file_path:
        raise ValueError("No file path provided")

    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Case 1: User selected a .plt file directly
    if file_path.endswith(".plt"):
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        vtm_target = os.path.join(base_dir, f"{base_name}_1d_zones.vtm")
        if is_output_stale(vtm_target, file_path):
            reason = "Forced regeneration" if FORCE_CONVERT else (
                f"'{vtm_target}' missing or older than '{file_path}'"
            )
            print(f"[Bondalyzer] Dynamically generating '{vtm_target}' from '{file_path}'... ({reason})")
            convert_1d_zones_to_vtm(file_path, output_file=vtm_target)
        return vtm_target

    # Case 2: User selected a .vtm file
    elif file_path.endswith(".vtm"):
        plt_fallback = resolve_plt_for(file_path)
        if plt_fallback and is_output_stale(file_path, plt_fallback):
            reason = "Forced regeneration" if FORCE_CONVERT else (
                "missing" if not os.path.exists(file_path) else f"older than '{plt_fallback}'"
            )
            print(f"[Bondalyzer] Dynamically regenerating '{file_path}' from '{plt_fallback}'... ({reason})")
            convert_1d_zones_to_vtm(plt_fallback, output_file=file_path)
        return file_path

    return file_path


def load_volume_grid(vtm_path: str):
    """
    Locate (and generate on demand / when stale) the Zone 0 volume grid
    associated with a .vtm file, returning the loaded vtk dataset or None.
    Supports Uniform (.vti), Rectilinear (.vtr), and general Curvilinear/Rotated (.vts).
    """
    base_no_ext = os.path.splitext(os.path.basename(vtm_path))[0].replace("_1d_zones", "")
    vti_candidate = os.path.splitext(vtm_path)[0].replace("_1d_zones", "_zone0") + ".vti"
    if not os.path.exists(vti_candidate):
        vti_candidate = os.path.join(os.path.dirname(vtm_path), f"{base_no_ext}_zone0.vti")
    vtr_candidate = os.path.splitext(vti_candidate)[0] + ".vtr"
    vts_candidate = os.path.splitext(vti_candidate)[0] + ".vts"

    # Resolve the companion .plt used for on-demand generation
    plt_for_vol = vtm_path.replace("_1d_zones.vtm", ".plt").replace(".vtm", ".plt")
    if not os.path.exists(plt_for_vol):
        candidate_plt = f"{base_no_ext}.plt"
        if os.path.exists(candidate_plt):
            plt_for_vol = candidate_plt
        elif os.path.exists("ethene4.plt"):
            plt_for_vol = "ethene4.plt"
        elif os.path.exists("ethene2.plt"):
            plt_for_vol = "ethene2.plt"
        elif os.path.exists("ethene.plt"):
            plt_for_vol = "ethene.plt"

    # Generate grid on demand when missing, forced, or stale relative to the .plt
    has_any_grid = os.path.exists(vti_candidate) or os.path.exists(vtr_candidate) or os.path.exists(vts_candidate)
    
    if not has_any_grid:
        if os.path.exists(plt_for_vol):
            try:
                print(f"[Bondalyzer] Generating volume grid from '{plt_for_vol}'...")
                convert_zone0_to_vtk(plt_for_vol, grid_type="auto")
            except Exception as e:
                print(f"[Bondalyzer] Warning: Could not generate volume grid: {e}")
    else:
        for cand in (vts_candidate, vti_candidate, vtr_candidate):
            if os.path.exists(cand) and is_output_stale(cand, plt_for_vol):
                if os.path.exists(plt_for_vol):
                    try:
                        print(f"[Bondalyzer] Volume grid '{cand}' is older than '{plt_for_vol}'. Regenerating...")
                        convert_zone0_to_vtk(plt_for_vol, grid_type="auto")
                    except Exception as e:
                        print(f"[Bondalyzer] Warning: Could not regenerate volume grid: {e}")
                break

    volume_grid = None
    if os.path.exists(vts_candidate):
        vts_reader = vtkXMLStructuredGridReader()
        vts_reader.SetFileName(vts_candidate)
        vts_reader.Update()
        volume_grid = vts_reader.GetOutput()
    elif os.path.exists(vti_candidate):
        vti_reader = vtkXMLImageDataReader()
        vti_reader.SetFileName(vti_candidate)
        vti_reader.Update()
        volume_grid = vti_reader.GetOutput()
    elif os.path.exists(vtr_candidate):
        vtr_reader = vtkXMLRectilinearGridReader()
        vtr_reader.SetFileName(vtr_candidate)
        vtr_reader.Update()
        volume_grid = vtr_reader.GetOutput()
    return volume_grid


def init_vtk_context():
    """
    Initialize persistent VTK renderer, render window, interactor, and orientation axes.
    These persist throughout the entire lifecycle of the viewer server.
    """
    renderer = vtkRenderer()
    renderer.TwoSidedLightingOn()
    renderer.SetBackground(0.12, 0.13, 0.16)  # Dark chemist canvas background
    renderer.SetBackground2(0.20, 0.22, 0.26)
    renderer.SetGradientBackground(True)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 750)
    render_window.SetWindowName("Bondalyzer Molecule Viewer")
    # Hide native desktop window on macOS (render purely offscreen for web streaming)
    render_window.SetOffScreenRendering(1)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.Initialize()

    # Orientation Marker (Labeled 3D Coordinate Axes in Upper-Right Corner)
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
        cap_prop.SetFontSize(18)
        cap_prop.BoldOn()
        cap_prop.ShadowOn()

    orientation_widget = vtkOrientationMarkerWidget()
    orientation_widget.SetOrientationMarker(axes_actor)
    orientation_widget.SetInteractor(interactor)
    orientation_widget.SetViewport(0.68, 0.68, 0.98, 0.98)
    orientation_widget.SetEnabled(1)
    orientation_widget.InteractiveOff()

    return renderer, render_window, interactor, orientation_widget


def clear_pipeline_dataset(pipeline_data: Dict[str, Any], renderer: vtkRenderer):
    """
    Remove all dataset actors from the renderer and reset pipeline data structures.
    """
    # 1. Remove skeleton actors
    for actor in pipeline_data.get("actors_raw", {}).values():
        if actor:
            renderer.RemoveActor(actor)
    for actor in pipeline_data.get("actors_clipped", {}).values():
        if actor and actor not in pipeline_data.get("actors_raw", {}).values():
            renderer.RemoveActor(actor)
    pipeline_data["actors"] = {}
    pipeline_data["actors_raw"] = {}
    pipeline_data["actors_clipped"] = {}
    pipeline_data["actor_categories"] = {}

    # 2. Remove highlight actor
    if pipeline_data.get("highlight_actor"):
        renderer.RemoveActor(pipeline_data["highlight_actor"])
        pipeline_data["highlight_actor"] = None

    # 3. Remove SCA actors
    for key in ("iso_actor", "cut_actor", "contour_actor", "ws_actor"):
        if pipeline_data.get(key):
            renderer.RemoveActor(pipeline_data[key])
            pipeline_data[key] = None

    # 4. Remove GBA actors
    for p in pipeline_data.get("gba_patch_actors", []):
        if p.get("actor"):
            renderer.RemoveActor(p["actor"])
    pipeline_data["gba_patch_actors"] = []

    for key in (
        "gba_sphere_actor",
        "gba_sphere_solid_actor",
        "gba_nucleus_actor",
        "gba_atom_flood_actor",
        "gba_atom_contour_actor",
        "gba_highlight_actor",
        "gba_wedge_actor",
        "gba_wedge_edge_actor",
    ):
        if pipeline_data.get(key):
            renderer.RemoveActor(pipeline_data[key])
            pipeline_data[key] = None

    pipeline_data["gba_surface_blocks"] = []
    pipeline_data["atoms"] = []
    pipeline_data["critical_points"] = []
    pipeline_data["volume_grid"] = None
    pipeline_data["atom_glyph_entries"] = []
    pipeline_data["gba_sphere_radius"] = 1.00


def populate_dataset_pipeline(vtm_path: str, renderer: vtkRenderer, pipeline_data: Dict[str, Any]):
    """
    Read dataset from vtm_path, parse metadata, build VTK filters & actors,
    and register them into the renderer and pipeline_data container.
    """
    clear_pipeline_dataset(pipeline_data, renderer)

    if not vtm_path or not os.path.exists(vtm_path):
        return

    # 1. Read MultiBlock dataset
    reader = vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_path)
    reader.Update()
    mb = reader.GetOutput()

    # 2. Load Zone 0 volume grid
    volume_grid = load_volume_grid(vtm_path)
    pipeline_data["volume_grid"] = volume_grid

    molecule_info, atoms, critical_points = parse_dataset_metadata(mb, volume_grid=volume_grid)
    pipeline_data["molecule_info"] = molecule_info
    pipeline_data["atoms"] = atoms
    pipeline_data["critical_points"] = critical_points

    # Extract crystal periodicity & lattice vectors (from .run / PLT Aux / Dev fallback)
    lattice_info = extract_crystal_lattice_info(
        vtm_path,
        volume_grid=volume_grid,
        mb=mb,
        atoms=atoms,
    )
    pipeline_data["lattice_info"] = lattice_info
    is_periodic = lattice_info is not None and lattice_info.get("is_periodic", False)

    ws_planes = None
    ws_actor = None
    if is_periodic:
        ws_planes, ws_actor = build_wigner_seitz_planes(lattice_info)
        if ws_actor is not None:
            renderer.AddActor(ws_actor)
            pipeline_data["ws_actor"] = ws_actor

    actors_raw = {}
    actors_clipped = {}
    actor_categories = {}
    atom_glyph_entries = []

    # Extract plane equations (normals and midpoints) for fast in-cell testing
    ws_plane_normals = []
    ws_plane_dvals = []
    if ws_planes is not None:
        pts = ws_planes.GetPoints()
        nrms = ws_planes.GetNormals()
        if pts and nrms:
            n_ws = pts.GetNumberOfPoints()
            for i in range(n_ws):
                p_i = pts.GetPoint(i)
                n_i = nrms.GetTuple3(i)
                ws_plane_normals.append(np.array(n_i, dtype=np.float64))
                ws_plane_dvals.append(float(np.dot(np.array(n_i, dtype=np.float64), np.array(p_i, dtype=np.float64))))

    def is_point_inside_ws(pt: Tuple[float, float, float]) -> bool:
        """Check if a 3D point is inside all WS bounding half-spaces (within tolerance)."""
        if not ws_plane_normals:
            return True
        p_arr = np.array(pt, dtype=np.float64)
        for n_i, d_i in zip(ws_plane_normals, ws_plane_dvals):
            if np.dot(n_i, p_arr) > d_i + 1e-4:
                return False
        return True

    def filter_poly_to_ws(poly_in: Any, is_line_poly: bool) -> Any:
        """Construct a trimmed vtkPolyData containing only points/cells inside the WS cell."""
        if not ws_plane_normals:
            return poly_in

        n_p = poly_in.GetNumberOfPoints()
        if n_p == 0:
            return poly_in

        inside_mask = [is_point_inside_ws(poly_in.GetPoint(i)) for i in range(n_p)]
        if all(inside_mask):
            return poly_in
        if not any(inside_mask):
            empty_poly = vtk.vtkPolyData()
            empty_poly.SetPoints(vtk.vtkPoints())
            return empty_poly

        out_poly = vtk.vtkPolyData()
        out_pts = vtk.vtkPoints()

        if is_line_poly:
            # For 1D trajectories, keep points that are inside
            for old_idx, inside in enumerate(inside_mask):
                if inside:
                    pt = poly_in.GetPoint(old_idx)
                    out_pts.InsertNextPoint(pt[0], pt[1], pt[2])

            out_poly.SetPoints(out_pts)
            if out_pts.GetNumberOfPoints() > 1:
                lines_ca = vtk.vtkCellArray()
                polyline = vtk.vtkPolyLine()
                polyline.GetPointIds().SetNumberOfIds(out_pts.GetNumberOfPoints())
                for i in range(out_pts.GetNumberOfPoints()):
                    polyline.GetPointIds().SetId(i, i)
                lines_ca.InsertNextCell(polyline)
                out_poly.SetLines(lines_ca)
        else:
            # Discrete critical points / atoms
            verts_ca = vtk.vtkCellArray()
            for old_idx, inside in enumerate(inside_mask):
                if inside:
                    pt = poly_in.GetPoint(old_idx)
                    new_idx = out_pts.InsertNextPoint(pt[0], pt[1], pt[2])
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, new_idx)
                    verts_ca.InsertNextCell(vertex)

            out_poly.SetPoints(out_pts)
            out_poly.SetVerts(verts_ca)

        return out_poly

    num_blocks = mb.GetNumberOfBlocks()
    for b in range(num_blocks):
        block_name = mb.GetMetaData(b).Get(mb.NAME()) if mb.GetMetaData(b) else f"Block_{b}"
        poly = mb.GetBlock(b)
        if not poly or poly.GetNumberOfPoints() == 0:
            continue

        if block_name.strip().lower() == "critical points" or block_name.strip().lower() == "nuclear_cp":
            continue

        n_pts = poly.GetNumberOfPoints()
        n_lines = poly.GetNumberOfLines()
        is_line = (n_lines > 0)
        is_atom = "atom" in block_name.lower()

        # Build clipped polydata for periodic WS cell
        poly_clipped = filter_poly_to_ws(poly, is_line) if (ws_planes is not None and not is_atom) else poly

        # Determine color
        arr = poly.GetPointData().GetArray("RGBColor")
        r_val, g_val, b_val = (0.7, 0.7, 0.7) if is_line else (0.8, 0.8, 0.8)
        if arr and arr.GetNumberOfTuples() > 0:
            t = arr.GetTuple(0)
            r_val, g_val, b_val = t[0] / 255.0, t[1] / 255.0, t[2] / 255.0

        if is_line:
            r_tube = 0.025 if ("bond path" in block_name.lower() or "path" in block_name.lower()) else 0.045

            # 1. Unclipped tube actor
            tuber_raw = vtkTubeFilter()
            tuber_raw.SetInputData(poly)
            tuber_raw.SetNumberOfSides(16)
            tuber_raw.CappingOn()
            tuber_raw.SetRadius(r_tube)
            tuber_raw.Update()

            mapper_raw = vtkPolyDataMapper()
            mapper_raw.SetInputConnection(tuber_raw.GetOutputPort())
            mapper_raw.ScalarVisibilityOff()

            actor_raw = vtkActor()
            actor_raw.SetMapper(mapper_raw)
            actor_raw.GetProperty().SetColor(r_val, g_val, b_val)
            actor_raw.GetProperty().SetAmbient(0.35)
            actor_raw.GetProperty().SetDiffuse(0.75)
            actor_raw.GetProperty().SetSpecular(0.4)
            actor_raw.GetProperty().SetSpecularPower(30)
            renderer.AddActor(actor_raw)
            actors_raw[block_name] = actor_raw

            # 2. Clipped tube actor
            if ws_planes is not None and not is_atom:
                tuber_clipped = vtkTubeFilter()
                tuber_clipped.SetInputData(poly_clipped)
                tuber_clipped.SetNumberOfSides(16)
                tuber_clipped.CappingOn()
                tuber_clipped.SetRadius(r_tube)
                tuber_clipped.Update()

                mapper_clipped = vtkPolyDataMapper()
                mapper_clipped.SetInputConnection(tuber_clipped.GetOutputPort())
                mapper_clipped.ScalarVisibilityOff()

                actor_clipped = vtkActor()
                actor_clipped.SetMapper(mapper_clipped)
                actor_clipped.GetProperty().SetColor(r_val, g_val, b_val)
                actor_clipped.GetProperty().SetAmbient(0.35)
                actor_clipped.GetProperty().SetDiffuse(0.75)
                actor_clipped.GetProperty().SetSpecular(0.4)
                actor_clipped.GetProperty().SetSpecularPower(30)
                actor_clipped.SetVisibility(False)
                renderer.AddActor(actor_clipped)
                actors_clipped[block_name] = actor_clipped
            else:
                actors_clipped[block_name] = actor_raw

            # Classify line/path actor category
            b_lower = block_name.lower()
            if "bond path" in b_lower:
                actor_categories[block_name] = "bond_paths"
            elif "ring" in b_lower:
                actor_categories[block_name] = "ring_paths"
            elif "cage" in b_lower:
                actor_categories[block_name] = "cage_paths"
            elif "path" in b_lower:
                actor_categories[block_name] = "bond_paths"
            else:
                actor_categories[block_name] = "inferred_bonds"

        else:
            sphere_source = vtkSphereSource()
            sphere_source.SetThetaResolution(24)
            sphere_source.SetPhiResolution(24)

            if is_atom:
                element = "C"
                if poly.GetFieldData().HasArray("Aux_AtomType"):
                    element = str(poly.GetFieldData().GetAbstractArray("Aux_AtomType").GetValue(0))
                elif "(" in block_name and ")" in block_name:
                    element = block_name.split("(")[1].split(")")[0].strip()

                r_cov = get_covalent_radius(element, default=0.75)
                # Ball-and-stick scale factor: vtkSphereSource default radius is 0.5 (diameter 1.0)
                s_factor = 2.0 * r_cov * BALL_AND_STICK_SCALE
            elif "bond" in block_name.lower() or "bcp" in block_name.lower():
                s_factor = 0.18
            elif "ring" in block_name.lower() or "rcp" in block_name.lower():
                s_factor = 0.18
            elif "cage" in block_name.lower() or "ccp" in block_name.lower():
                s_factor = 0.18
            else:
                s_factor = 0.14

            # 1. Unclipped glyph actor
            glyph_raw = vtkGlyph3D()
            glyph_raw.SetSourceConnection(sphere_source.GetOutputPort())
            glyph_raw.SetInputData(poly)
            glyph_raw.ScalingOn()
            glyph_raw.SetScaleModeToDataScalingOff()
            glyph_raw.SetScaleFactor(s_factor)
            glyph_raw.Update()

            mapper_raw = vtkPolyDataMapper()
            mapper_raw.SetInputConnection(glyph_raw.GetOutputPort())
            mapper_raw.ScalarVisibilityOff()

            actor_raw = vtkActor()
            actor_raw.SetMapper(mapper_raw)
            actor_raw.GetProperty().SetColor(r_val, g_val, b_val)
            actor_raw.GetProperty().SetAmbient(0.35)
            actor_raw.GetProperty().SetDiffuse(0.75)
            actor_raw.GetProperty().SetSpecular(0.5)
            actor_raw.GetProperty().SetSpecularPower(40)
            renderer.AddActor(actor_raw)
            actors_raw[block_name] = actor_raw

            if is_atom:
                atom_glyph_entries.append({
                    "glyph_raw": glyph_raw,
                    "glyph_clipped": glyph_clipped if (ws_planes is not None and not is_atom) else None,
                    "covalent_scale": s_factor,
                })

            # 2. Clipped glyph actor
            if ws_planes is not None and not is_atom:
                glyph_clipped = vtkGlyph3D()
                glyph_clipped.SetSourceConnection(sphere_source.GetOutputPort())
                glyph_clipped.SetInputData(poly_clipped)
                glyph_clipped.ScalingOn()
                glyph_clipped.SetScaleModeToDataScalingOff()
                glyph_clipped.SetScaleFactor(s_factor)
                glyph_clipped.Update()

                mapper_clipped = vtkPolyDataMapper()
                mapper_clipped.SetInputConnection(glyph_clipped.GetOutputPort())
                mapper_clipped.ScalarVisibilityOff()

                actor_clipped = vtkActor()
                actor_clipped.SetMapper(mapper_clipped)
                actor_clipped.GetProperty().SetColor(r_val, g_val, b_val)
                actor_clipped.GetProperty().SetAmbient(0.35)
                actor_clipped.GetProperty().SetDiffuse(0.75)
                actor_clipped.GetProperty().SetSpecular(0.5)
                actor_clipped.GetProperty().SetSpecularPower(40)
                actor_clipped.SetVisibility(False)
                renderer.AddActor(actor_clipped)
                actors_clipped[block_name] = actor_clipped
            else:
                actors_clipped[block_name] = actor_raw

            # Classify point/sphere actor category
            b_lower = block_name.lower()
            if is_atom or "nuclear" in b_lower:
                actor_categories[block_name] = "atoms"
            elif "bond" in b_lower or "bcp" in b_lower:
                actor_categories[block_name] = "bond_cps"
            elif "ring" in b_lower or "rcp" in b_lower:
                actor_categories[block_name] = "ring_cps"
            elif "cage" in b_lower or "ccp" in b_lower:
                actor_categories[block_name] = "cage_cps"
            else:
                actor_categories[block_name] = "cps"

    pipeline_data["actors"] = actors_raw
    pipeline_data["actors_raw"] = actors_raw
    pipeline_data["actors_clipped"] = actors_clipped
    pipeline_data["actor_categories"] = actor_categories
    pipeline_data["atom_glyph_entries"] = atom_glyph_entries

    # Selection Highlight Actor
    highlight_source = vtkSphereSource()
    highlight_source.SetThetaResolution(20)
    highlight_source.SetPhiResolution(20)
    highlight_source.SetRadius(0.42)
    highlight_source.Update()

    highlight_mapper = vtkPolyDataMapper()
    highlight_mapper.SetInputConnection(highlight_source.GetOutputPort())

    highlight_actor = vtkActor()
    highlight_actor.SetMapper(highlight_mapper)
    highlight_actor.GetProperty().SetColor(1.0, 0.85, 0.1)
    highlight_actor.GetProperty().SetRepresentationToWireframe()
    highlight_actor.GetProperty().SetLineWidth(2.5)
    highlight_actor.SetVisibility(False)
    renderer.AddActor(highlight_actor)

    pipeline_data["highlight_source"] = highlight_source
    pipeline_data["highlight_actor"] = highlight_actor

    # 3D Volume Grid, Isosurface & Cutplane Filters (SCA Tools)
    if volume_grid is not None:
        active_grid = volume_grid
        atom_positions = [a["raw_pos"] for a in atoms if "raw_pos" in a]

        # Only apply molecular VOI trimming for non-periodic molecules with multiple atoms
        # that have low edge density, to avoid truncating periodic crystals or unit cells.
        if not is_periodic and len(atom_positions) > 1 and volume_grid.GetPointData().HasArray("Electron Density"):
            dims = volume_grid.GetDimensions()
            dens_arr = volume_grid.GetPointData().GetArray("Electron Density")
            nx, ny, nz = dims
            sample_ids = [
                0, nx - 1, (ny - 1) * nx, (ny - 1) * nx + (nx - 1),
                (nz - 1) * nx * ny, (nz - 1) * nx * ny + (nx - 1),
                (nz - 1) * nx * ny + (ny - 1) * nx, (nz - 1) * nx * ny + (ny - 1) * nx + (nx - 1),
                nx // 2, (ny // 2) * nx, ((nz // 2) * ny + ny // 2) * nx,
            ]
            max_edge_dens = max(float(dens_arr.GetTuple1(pid)) for pid in sample_ids if 0 <= pid < dens_arr.GetNumberOfTuples())

            if max_edge_dens < 0.001 and volume_grid.IsA("vtkImageData"):
                pad = 2.0
                xs = [p[0] for p in atom_positions]
                ys = [p[1] for p in atom_positions]
                zs = [p[2] for p in atom_positions]
                crop_box = [
                    min(xs) - pad, max(xs) + pad,
                    min(ys) - pad, max(ys) + pad,
                    min(zs) - pad, max(zs) + pad,
                ]
                origin = volume_grid.GetOrigin()
                spacing = volume_grid.GetSpacing()
                imin = max(0, int((crop_box[0] - origin[0]) / spacing[0]))
                imax = min(nx - 1, int(math.ceil((crop_box[1] - origin[0]) / spacing[0])))
                jmin = max(0, int((crop_box[2] - origin[1]) / spacing[1]))
                jmax = min(ny - 1, int(math.ceil((crop_box[3] - origin[1]) / spacing[1])))
                kmin = max(0, int((crop_box[4] - origin[2]) / spacing[2]))
                kmax = min(nz - 1, int(math.ceil((crop_box[5] - origin[2]) / spacing[2])))

                extract_voi = vtkExtractVOI()
                extract_voi.SetInputData(volume_grid)
                extract_voi.SetVOI(imin, imax, jmin, jmax, kmin, kmax)
                extract_voi.Update()
                active_grid = extract_voi.GetOutput()

        # Isosurface Filter: vtkFlyingEdges3D for vtkImageData, vtkContourFilter for vtkStructuredGrid / vtkRectilinearGrid
        if active_grid.IsA("vtkImageData"):
            iso_filter = vtkFlyingEdges3D()
        else:
            iso_filter = vtkContourFilter()
        iso_filter.SetInputData(active_grid)
        iso_filter.SetInputArrayToProcess(0, 0, 0, 0, "Electron Density")
        iso_filter.SetValue(0, 0.05)
        if hasattr(iso_filter, "ComputeNormalsOn"):
            iso_filter.ComputeNormalsOn()
        iso_filter.Update()

        ws_clipper = None
        if is_periodic and ws_planes is not None:
            ws_clipper = vtkClipPolyData()
            ws_clipper.SetInputConnection(iso_filter.GetOutputPort())
            ws_clipper.SetClipFunction(ws_planes)
            ws_clipper.InsideOutOn()

        pipeline_data["ws_planes"] = ws_planes
        pipeline_data["ws_clipper"] = ws_clipper

        # Isosurface Mapper: connects to ws_clipper if periodic masking active, otherwise iso_filter
        iso_mapper = vtkPolyDataMapper()
        if ws_clipper is not None:
            iso_mapper.SetInputConnection(ws_clipper.GetOutputPort())
        else:
            iso_mapper.SetInputConnection(iso_filter.GetOutputPort())
        iso_mapper.ScalarVisibilityOff()

        iso_actor = vtkActor()
        iso_actor.SetMapper(iso_mapper)
        iso_prop = iso_actor.GetProperty()
        iso_prop.SetColor(0.20, 0.90, 1.0)
        iso_prop.SetAmbient(0.45)
        iso_prop.SetDiffuse(0.90)
        iso_prop.SetSpecular(0.50)
        iso_prop.SetSpecularPower(35)
        iso_prop.SetOpacity(0.70)

        # Explicit backface property so interior/reverse-normal faces are identically lit
        back_prop = vtkProperty()
        back_prop.DeepCopy(iso_prop)
        iso_actor.SetBackfaceProperty(back_prop)

        iso_actor.SetVisibility(False)
        renderer.AddActor(iso_actor)

        # Planar Cutplane
        cut_plane = vtkPlane()
        cut_plane.SetOrigin(0.0, 0.0, 0.0)
        cut_plane.SetNormal(0.0, 0.0, 1.0)

        cutter = vtkCutter()
        cutter.SetInputData(active_grid)
        cutter.SetCutFunction(cut_plane)
        cutter.Update()

        color_tf = vtkColorTransferFunction()
        color_tf.AddRGBPoint(0.0, 0.267, 0.004, 0.329)
        color_tf.AddRGBPoint(0.25, 0.190, 0.407, 0.556)
        color_tf.AddRGBPoint(0.50, 0.127, 0.566, 0.550)
        color_tf.AddRGBPoint(0.75, 0.369, 0.788, 0.382)
        color_tf.AddRGBPoint(1.00, 0.993, 0.906, 0.143)

        cut_mapper = vtkPolyDataMapper()
        cut_mapper.SetInputConnection(cutter.GetOutputPort())
        cut_mapper.SetLookupTable(color_tf)
        cut_mapper.SelectColorArray("Electron Density")
        cut_mapper.SetScalarModeToUsePointFieldData()
        cut_mapper.SetScalarRange(0.001, 0.50)
        cut_mapper.ScalarVisibilityOn()

        cut_actor = vtkActor()
        cut_actor.SetMapper(cut_mapper)
        cut_actor.GetProperty().SetOpacity(0.85)
        cut_actor.SetVisibility(False)
        renderer.AddActor(cut_actor)

        # Cutplane Contours
        contour_filter = vtkContourFilter()
        contour_filter.SetInputConnection(cutter.GetOutputPort())
        contour_filter.SetInputArrayToProcess(0, 0, 0, 0, "Electron Density")
        contour_filter.GenerateValues(15, 0.001, 0.50)
        contour_filter.Update()

        contour_mapper = vtkPolyDataMapper()
        contour_mapper.SetInputConnection(contour_filter.GetOutputPort())
        contour_mapper.ScalarVisibilityOff()

        contour_actor = vtkActor()
        contour_actor.SetMapper(contour_mapper)
        contour_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        contour_actor.GetProperty().SetLineWidth(2.0)
        contour_actor.GetProperty().SetLighting(False)
        contour_actor.SetVisibility(False)
        renderer.AddActor(contour_actor)

        pipeline_data["iso_filter"] = iso_filter
        pipeline_data["iso_mapper"] = iso_mapper
        pipeline_data["iso_actor"] = iso_actor
        pipeline_data["cut_plane"] = cut_plane
        pipeline_data["cutter"] = cutter
        pipeline_data["cut_mapper"] = cut_mapper
        pipeline_data["cut_actor"] = cut_actor
        pipeline_data["contour_filter"] = contour_filter
        pipeline_data["contour_actor"] = contour_actor
        pipeline_data["color_tf"] = color_tf

    # GBA Basin Sphere Patches & Atom Sphere Boundary Setup
    gba_patch_actors = []
    gba_surface_blocks = []
    plt_candidate = vtm_path.replace("_1d_zones.vtm", ".plt").replace(".vtm", ".plt")
    if not os.path.exists(plt_candidate):
        base_no_ext = os.path.splitext(os.path.basename(vtm_path))[0].replace("_1d_zones", "")
        if os.path.exists(f"{base_no_ext}.plt"):
            plt_candidate = f"{base_no_ext}.plt"
        elif os.path.exists("ethene4.plt"):
            plt_candidate = "ethene4.plt"
        elif os.path.exists("ethene2.plt"):
            plt_candidate = "ethene2.plt"
        elif os.path.exists("ethene.plt"):
            plt_candidate = "ethene.plt"

    if os.path.exists(plt_candidate):
        try:
            gba_mb, gba_meta = extract_gba_zones_from_plt(
                plt_candidate,
                output_vtm=None,
                include_sphere_patches=True,
                include_surfaces=True,
                include_atom_spheres=True,
            )

            for entry in gba_meta:
                ztype = entry.get("zone_type")
                poly_b = gba_mb.GetBlock(entry["block_index"])
                if not poly_b:
                    continue

                if ztype == "CondensedBasinSurface":
                    gba_surface_blocks.append({"poly": poly_b, "meta": entry})

                elif ztype == "AtomSphereData":
                    pipeline_data["gba_sphere_poly"] = poly_b

                    # Extract SphereRadius from zone aux metadata
                    sr_aux = entry.get("aux", {}).get("SphereRadius")
                    if sr_aux:
                        try:
                            sr_val = float(sr_aux)
                            pipeline_data["gba_sphere_radius"] = sr_val
                        except ValueError:
                            pass

                    mapper = vtkPolyDataMapper()
                    mapper.SetInputData(poly_b)
                    mapper.ScalarVisibilityOff()

                    actor = vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetRepresentationToWireframe()
                    actor.GetProperty().SetColor(0.85, 0.85, 0.85)
                    actor.GetProperty().SetOpacity(0.35)
                    actor.GetProperty().SetLineWidth(1.2)
                    actor.SetVisibility(False)
                    renderer.AddActor(actor)
                    pipeline_data["gba_sphere_actor"] = actor

                    # Solid GBA Sphere Base Actor (substitutes for the ball-and-stick sphere in GBA tools)
                    solid_mapper = vtkPolyDataMapper()
                    solid_mapper.SetInputData(poly_b)
                    solid_mapper.ScalarVisibilityOff()

                    solid_actor = vtkActor()
                    solid_actor.SetMapper(solid_mapper)
                    solid_actor.GetProperty().SetColor(0.75, 0.75, 0.78)  # Neutral atom surface color
                    solid_actor.GetProperty().SetAmbient(0.35)
                    solid_actor.GetProperty().SetDiffuse(0.75)
                    solid_actor.GetProperty().SetSpecular(0.40)
                    solid_actor.GetProperty().SetSpecularPower(30)
                    solid_actor.SetVisibility(False)
                    renderer.AddActor(solid_actor)
                    pipeline_data["gba_sphere_solid_actor"] = solid_actor

                    # Small nucleus sphere marker at the center
                    nucleus_src = vtkSphereSource()
                    nucleus_src.SetRadius(0.10)
                    nucleus_src.SetThetaResolution(16)
                    nucleus_src.SetPhiResolution(16)
                    nucleus_bnds = poly_b.GetBounds()
                    nucleus_center = [(nucleus_bnds[0] + nucleus_bnds[1]) * 0.5, (nucleus_bnds[2] + nucleus_bnds[3]) * 0.5, (nucleus_bnds[4] + nucleus_bnds[5]) * 0.5]
                    pipeline_data["gba_nucleus_center"] = nucleus_center
                    nucleus_src.SetCenter(nucleus_center[0], nucleus_center[1], nucleus_center[2])
                    nucleus_src.Update()

                    nucleus_mapper = vtkPolyDataMapper()
                    nucleus_mapper.SetInputConnection(nucleus_src.GetOutputPort())
                    nucleus_mapper.ScalarVisibilityOff()

                    nucleus_actor = vtkActor()
                    nucleus_actor.SetMapper(nucleus_mapper)
                    nucleus_actor.GetProperty().SetColor(0.4, 0.4, 0.4)
                    nucleus_actor.GetProperty().SetAmbient(0.4)
                    nucleus_actor.GetProperty().SetDiffuse(0.7)
                    nucleus_actor.SetVisibility(False)
                    renderer.AddActor(nucleus_actor)
                    pipeline_data["gba_nucleus_actor"] = nucleus_actor

                    # Atom Surface Flood
                    gba_atom_flood_mapper = vtkPolyDataMapper()
                    gba_atom_flood_mapper.SetInputData(poly_b)
                    gba_atom_flood_mapper.SetScalarModeToUsePointFieldData()
                    gba_atom_flood_mapper.ScalarVisibilityOn()

                    gba_atom_flood_actor = vtkActor()
                    gba_atom_flood_actor.SetMapper(gba_atom_flood_mapper)
                    gba_atom_flood_actor.GetProperty().SetAmbient(0.35)
                    gba_atom_flood_actor.GetProperty().SetDiffuse(0.75)
                    gba_atom_flood_actor.GetProperty().SetSpecular(0.40)
                    gba_atom_flood_actor.GetProperty().SetSpecularPower(30)
                    gba_atom_flood_actor.SetVisibility(False)
                    renderer.AddActor(gba_atom_flood_actor)
                    pipeline_data["gba_atom_flood_mapper"] = gba_atom_flood_mapper
                    pipeline_data["gba_atom_flood_actor"] = gba_atom_flood_actor

                    # Atom Surface Contours
                    gba_atom_contour_filter = vtkContourFilter()
                    gba_atom_contour_filter.SetInputData(poly_b)

                    gba_atom_contour_mapper = vtkPolyDataMapper()
                    gba_atom_contour_mapper.SetInputConnection(gba_atom_contour_filter.GetOutputPort())
                    gba_atom_contour_mapper.ScalarVisibilityOff()

                    gba_atom_contour_actor = vtkActor()
                    gba_atom_contour_actor.SetMapper(gba_atom_contour_mapper)
                    gba_atom_contour_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                    gba_atom_contour_actor.GetProperty().SetLineWidth(2.0)
                    gba_atom_contour_actor.GetProperty().SetLighting(False)
                    gba_atom_contour_actor.SetVisibility(False)
                    renderer.AddActor(gba_atom_contour_actor)
                    pipeline_data["gba_atom_contour_filter"] = gba_atom_contour_filter
                    pipeline_data["gba_atom_contour_actor"] = gba_atom_contour_actor

                elif ztype == "CondensedBasinSphere":
                    mapper = vtkPolyDataMapper()
                    mapper.SetInputData(poly_b)
                    mapper.ScalarVisibilityOff()

                    actor = vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetAmbient(0.85)
                    actor.GetProperty().SetDiffuse(0.35)
                    actor.GetProperty().SetSpecular(0.20)
                    actor.GetProperty().SetSpecularPower(20)
                    actor.SetVisibility(False)
                    renderer.AddActor(actor)

                    gba_patch_actors.append({
                        "actor": actor,
                        "meta": entry,
                        "poly": poly_b,
                    })

            # Basin patch boundary highlight actor
            gba_highlight_edges = vtkFeatureEdges()
            gba_highlight_edges.BoundaryEdgesOn()
            gba_highlight_edges.FeatureEdgesOff()
            gba_highlight_edges.NonManifoldEdgesOff()
            gba_highlight_edges.ManifoldEdgesOff()

            gba_highlight_tuber = vtkTubeFilter()
            gba_highlight_tuber.SetInputConnection(gba_highlight_edges.GetOutputPort())
            gba_highlight_tuber.SetRadius(0.012)
            gba_highlight_tuber.SetNumberOfSides(12)
            gba_highlight_tuber.CappingOn()

            gba_highlight_mapper = vtkPolyDataMapper()
            gba_highlight_mapper.SetInputConnection(gba_highlight_tuber.GetOutputPort())
            gba_highlight_mapper.ScalarVisibilityOff()

            gba_highlight_actor = vtkActor()
            gba_highlight_actor.SetMapper(gba_highlight_mapper)
            gba_highlight_actor.GetProperty().SetColor(1.0, 0.95, 0.20)
            gba_highlight_actor.GetProperty().SetAmbient(0.8)
            gba_highlight_actor.GetProperty().SetDiffuse(0.2)
            gba_highlight_actor.SetVisibility(False)
            renderer.AddActor(gba_highlight_actor)

            # 3D Basin Wedge Actors
            gba_wedge_mapper = vtkPolyDataMapper()
            gba_wedge_mapper.ScalarVisibilityOff()

            gba_wedge_actor = vtkActor()
            gba_wedge_actor.SetMapper(gba_wedge_mapper)
            gba_wedge_actor.GetProperty().SetColor(0.25, 0.70, 0.95)
            gba_wedge_actor.GetProperty().SetOpacity(0.55)
            gba_wedge_actor.GetProperty().SetAmbient(0.40)
            gba_wedge_actor.GetProperty().SetDiffuse(0.70)
            gba_wedge_actor.GetProperty().SetSpecular(0.40)
            gba_wedge_actor.GetProperty().SetSpecularPower(30)
            gba_wedge_actor.SetVisibility(False)
            renderer.AddActor(gba_wedge_actor)

            # Wedge wireframe outline
            gba_wedge_edges = vtkFeatureEdges()
            gba_wedge_edges.BoundaryEdgesOn()
            gba_wedge_edges.FeatureEdgesOn()
            gba_wedge_edges.SetFeatureAngle(30.0)

            gba_wedge_edge_mapper = vtkPolyDataMapper()
            gba_wedge_edge_mapper.SetInputConnection(gba_wedge_edges.GetOutputPort())
            gba_wedge_edge_mapper.ScalarVisibilityOff()

            gba_wedge_edge_actor = vtkActor()
            gba_wedge_edge_actor.SetMapper(gba_wedge_edge_mapper)
            gba_wedge_edge_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            gba_wedge_edge_actor.GetProperty().SetLineWidth(1.8)
            gba_wedge_edge_actor.GetProperty().SetLighting(False)
            gba_wedge_edge_actor.SetVisibility(False)
            renderer.AddActor(gba_wedge_edge_actor)

            pipeline_data["gba_highlight_edges"] = gba_highlight_edges
            pipeline_data["gba_highlight_actor"] = gba_highlight_actor
            pipeline_data["gba_wedge_mapper"] = gba_wedge_mapper
            pipeline_data["gba_wedge_actor"] = gba_wedge_actor
            pipeline_data["gba_wedge_edges"] = gba_wedge_edges
            pipeline_data["gba_wedge_edge_actor"] = gba_wedge_edge_actor

            # Dynamically extract all unique atoms with GBA data from gba_meta
            discovered_gba_atoms = []
            for entry in gba_meta:
                ztype = entry.get("zone_type")
                if ztype in ("AtomSphereData", "CondensedBasinSphere", "CondensedBasinSurface"):
                    a_type = entry.get("atom_type", "").strip()
                    a_num = entry.get("atom_number", "")
                    if a_type or a_num:
                        atom_id_str = f"{a_type}{a_num}"
                        if atom_id_str not in discovered_gba_atoms:
                            discovered_gba_atoms.append(atom_id_str)

            if discovered_gba_atoms:
                pipeline_data["molecule_info"]["gba_atoms_list"] = discovered_gba_atoms

            print(f"[Bondalyzer] Loaded {len(gba_patch_actors)} GBA basin patches from {plt_candidate} (GBA atoms: {pipeline_data['molecule_info'].get('gba_atoms_list', [])})")
        except Exception as e:
            print(f"[Bondalyzer] Warning: Could not load GBA patches from {plt_candidate}: {e}")

    pipeline_data["gba_patch_actors"] = gba_patch_actors
    pipeline_data["gba_surface_blocks"] = gba_surface_blocks
    renderer.ResetCamera()


def create_visualization_pipeline(vtm_path: str):
    """
    Build a standard VTK rendering pipeline from the .vtm file.
    Legacy helper for native VTK window fallback.
    """
    renderer, render_window, interactor, orientation_widget = init_vtk_context()
    pipeline_data = {}
    populate_dataset_pipeline(vtm_path, renderer, pipeline_data)

    return (
        renderer,
        render_window,
        pipeline_data.get("actors", {}),
        pipeline_data.get("molecule_info", {}),
        pipeline_data.get("atoms", []),
        pipeline_data.get("critical_points", []),
        pipeline_data.get("highlight_actor"),
        pipeline_data.get("highlight_source"),
        pipeline_data.get("volume_grid"),
        pipeline_data.get("iso_filter"),
        pipeline_data.get("iso_actor"),
        pipeline_data.get("cut_plane"),
        pipeline_data.get("cutter"),
        pipeline_data.get("cut_mapper"),
        pipeline_data.get("cut_actor"),
        pipeline_data.get("contour_filter"),
        pipeline_data.get("contour_actor"),
        pipeline_data.get("color_tf"),
        pipeline_data.get("gba_sphere_actor"),
        pipeline_data.get("gba_patch_actors", []),
        pipeline_data.get("gba_sphere_poly"),
        pipeline_data.get("gba_atom_flood_mapper"),
        pipeline_data.get("gba_atom_flood_actor"),
        pipeline_data.get("gba_atom_contour_filter"),
        None,
        pipeline_data.get("gba_atom_contour_actor"),
        pipeline_data.get("gba_highlight_edges"),
        pipeline_data.get("gba_highlight_actor"),
        pipeline_data.get("gba_wedge_mapper"),
        pipeline_data.get("gba_wedge_actor"),
        pipeline_data.get("gba_wedge_edges"),
        pipeline_data.get("gba_wedge_edge_actor"),
        pipeline_data.get("gba_surface_blocks", []),
        orientation_widget,
    )


def run_trame_app(vtm_path: Optional[str] = None, server_name: str = "bondalyzer_viewer", port: Optional[int] = None, open_browser: bool = True):
    """
    Launch the Trame-based interactive viewer application with metadata drawer,
    atom/CP picking, dynamic Open File dialogs, and hot-swapping dataset pipeline.
    """
    renderer, render_window, interactor, orientation_widget = init_vtk_context()
    pipeline_data: Dict[str, Any] = {}

    if vtm_path and os.path.exists(vtm_path):
        populate_dataset_pipeline(vtm_path, renderer, pipeline_data)

    server = get_server(server_name)
    state, ctrl = server.state, server.controller

    def request_view_update():
        """Safely trigger remote view update if view has been mounted and bound."""
        if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
            try:
                ctrl.view_update()
            except Exception:
                pass

    active_gba_patch_poly = None

    # Determine default initial field states
    has_initial_data = bool(vtm_path and os.path.exists(vtm_path) and pipeline_data.get("molecule_info"))
    mol_info = pipeline_data.get("molecule_info", {
        "formula": "-",
        "title": "No Dataset Loaded",
        "total_atoms": 0,
        "element_counts": {},
        "bonds": 0,
        "bond_paths": 0,
        "ring_paths": 0,
        "cage_paths": 0,
        "bond_cps": 0,
        "ring_cps": 0,
        "cage_cps": 0,
        "total_cps": 0,
        "num_blocks": 0,
        "global_fields": [],
        "selected_global_field": "Electron Density",
        "gba_atoms_list": ["C1"],
        "gba_fields": [],
        "selected_condensed_field": "Electron Density",
    })

    default_field = mol_info.get("selected_global_field", "Electron Density")
    raw_rng = (0.001, 1.0)
    vol_grid = pipeline_data.get("volume_grid")
    if vol_grid is not None and vol_grid.GetPointData().HasArray(default_field):
        raw_rng = vol_grid.GetPointData().GetArray(default_field).GetRange()
    init_min, init_max, init_val, init_step = get_field_slider_config(default_field, raw_rng)

    lat_info = pipeline_data.get("lattice_info")
    is_periodic_init = bool(lat_info is not None and lat_info.get("is_periodic", False))
    init_gba_atoms = mol_info.get("gba_atoms_list", ["C1"])
    init_gba_atom = init_gba_atoms[0] if init_gba_atoms else "C1"

    # Initial Trame state
    state.has_dataset = has_initial_data
    state.is_periodic = is_periodic_init
    state.clip_to_wigner_seitz = is_periodic_init
    state.show_ws_boundary = is_periodic_init
    state.overview_clip_to_ws = is_periodic_init
    state.overview_show_ws_boundary = is_periodic_init
    state.atom_scale_factor = 1.00
    state.vtm_file = os.path.basename(vtm_path) if vtm_path else ""
    state.current_file_path = vtm_path or ""
    state.num_blocks = len(pipeline_data.get("actors", {}))
    state.block_names = list(pipeline_data.get("actors", {}).keys())
    state.molecule_info = mol_info
    state.atoms_list = pipeline_data.get("atoms", [])
    state.cps_list = pipeline_data.get("critical_points", [])
    state.selected_item = None
    state.active_nav_mode = "overview"
    state.show_inferred_bonds = True
    state.show_bond_paths = True
    state.show_ring_paths = True
    state.show_cage_paths = True
    state.show_bond_cps = True
    state.show_ring_cps = True
    state.show_cage_cps = True
    state.sca_visualization_mode = "cutplane"
    state.selected_global_field = default_field
    state.selected_gba_atom_id = init_gba_atom
    state.selected_condensed_field = mol_info.get("selected_condensed_field", "Electron Density")
    state.selected_gba_basin = None
    state.gba_active_basins_list = []
    state.gba_wedge_opacity = 0.55
    state.gba_show_wedge_edges = True
    state.gba_visualization_mode = "basins"
    state.gba_show_sphere_boundary = True
    state.gba_show_min_basins = True
    state.gba_show_max_basins = False
    state.gba_active_basins_count = 0
    state.gba_show_contours = True
    state.gba_show_flood = True
    state.gba_scale_type = "linear"
    state.gba_num_contours = 15
    state.drawer_open = True

    # In-App File Browser State
    state.file_dialog_modal_open = False
    initial_browser_data = list_server_directory(os.getcwd())
    state.browser_current_path = initial_browser_data["current_path"]
    state.browser_parent_path = initial_browser_data["parent_path"]
    state.browser_breadcrumbs = initial_browser_data["breadcrumbs"]
    state.browser_entries = initial_browser_data["entries"]
    state.browser_selected_path = ""
    state.browser_filter_text = ""

    # Isosurface interactive state
    state.iso_enabled = False
    state.iso_value = init_val
    state.iso_min = init_min
    state.iso_max = init_max
    state.iso_step = init_step
    state.iso_opacity = 0.65
    state.has_volume_data = (vol_grid is not None)

    # Cutplane interactive state
    state.cut_enabled = False
    state.cut_orientation = "XY"
    state.cut_offset = 0.00
    state.cut_offset_min = -5.0
    state.cut_offset_max = 5.0
    state.cut_offset_step = 0.05
    state.cut_show_contours = True
    state.cut_show_flood = True
    state.cut_scale_type = "log" if "electron density" in default_field.lower() else "linear"
    state.cut_num_contours = 15
    state.flood_num_colors = 15

    # Colormap building function with discrete steps
    def build_discrete_colormap(n_colors: int, f_min: float, f_max: float, is_log: bool):
        ctf = vtkColorTransferFunction()
        anchors = [
            (0.00, 0.267, 0.004, 0.329),
            (0.25, 0.190, 0.407, 0.556),
            (0.50, 0.127, 0.566, 0.550),
            (0.75, 0.369, 0.788, 0.382),
            (1.00, 0.993, 0.906, 0.143),
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

    def update_skeleton_visibility():
        actors_raw = pipeline_data.get("actors_raw", {})
        actors_clipped = pipeline_data.get("actors_clipped", {})
        actor_categories = pipeline_data.get("actor_categories", {})
        atom_glyph_entries = pipeline_data.get("atom_glyph_entries", [])
        ws_actor = pipeline_data.get("ws_actor")
        show_inferred = bool(state.show_inferred_bonds)
        show_bpaths = bool(state.show_bond_paths)
        show_rpaths = bool(state.show_ring_paths)
        show_cpaths = bool(state.show_cage_paths)
        show_bcps = bool(state.show_bond_cps)
        show_rcps = bool(state.show_ring_cps)
        show_ccps = bool(state.show_cage_cps)
        clip_skeleton = bool(state.is_periodic and state.overview_clip_to_ws)
        is_gba = (state.active_nav_mode == "gba")
        atom_scale = float(state.atom_scale_factor if state.atom_scale_factor is not None else 1.0)

        # In Viewer/SCA mode, maintain empirical covalent scaling on the Ball-and-Stick glyph multiplied by atom_scale_factor
        if not is_gba:
            for entry in atom_glyph_entries:
                eff_scale = entry["covalent_scale"] * atom_scale
                g_raw = entry.get("glyph_raw")
                if g_raw:
                    g_raw.SetScaleFactor(eff_scale)
                    g_raw.Update()
                g_clipped = entry.get("glyph_clipped")
                if g_clipped:
                    g_clipped.SetScaleFactor(eff_scale)
                    g_clipped.Update()

        for block_name, raw_actor in actors_raw.items():
            clipped_actor = actors_clipped.get(block_name, raw_actor)
            cat = actor_categories.get(block_name, "")

            if is_gba:
                # While in GBA Tools, hide all 3D ball-and-stick glyphs & critical points
                if cat in ("atoms", "bond_cps", "ring_cps", "cage_cps", "cps"):
                    is_cat_visible = False
                elif cat == "inferred_bonds":
                    is_cat_visible = show_inferred
                elif cat == "bond_paths":
                    is_cat_visible = show_bpaths
                elif cat == "ring_paths":
                    is_cat_visible = show_rpaths
                elif cat == "cage_paths":
                    is_cat_visible = show_cpaths
                else:
                    is_cat_visible = False
            else:
                # Restore full visibility and opacity when in Viewer / SCA Tools
                if cat == "atoms":
                    raw_actor.GetProperty().SetOpacity(1.0)
                    raw_actor.GetProperty().SetRepresentationToSurface()
                    is_cat_visible = (atom_scale > 0.001)
                elif cat == "inferred_bonds":
                    is_cat_visible = show_inferred
                elif cat == "bond_paths":
                    is_cat_visible = show_bpaths
                elif cat == "ring_paths":
                    is_cat_visible = show_rpaths
                elif cat == "cage_paths":
                    is_cat_visible = show_cpaths
                elif cat == "bond_cps":
                    is_cat_visible = show_bcps
                elif cat == "ring_cps":
                    is_cat_visible = show_rcps
                elif cat == "cage_cps":
                    is_cat_visible = show_ccps
                else:
                    is_cat_visible = True

            if clip_skeleton:
                raw_actor.SetVisibility(False)
                clipped_actor.SetVisibility(is_cat_visible)
            else:
                clipped_actor.SetVisibility(False)
                raw_actor.SetVisibility(is_cat_visible)

        # Viewer Wigner-Seitz wireframe visibility (when in viewer mode)
        if ws_actor is not None and state.active_nav_mode == "overview":
            ws_actor.SetVisibility(bool(state.is_periodic and state.overview_show_ws_boundary))

        render_window.Render()
        request_view_update()

    def update_isosurface():
        iso_filter = pipeline_data.get("iso_filter")
        iso_mapper = pipeline_data.get("iso_mapper")
        iso_actor = pipeline_data.get("iso_actor")
        ws_clipper = pipeline_data.get("ws_clipper")
        ws_actor = pipeline_data.get("ws_actor")
        volume_grid = pipeline_data.get("volume_grid")
        if iso_filter is None or iso_actor is None or volume_grid is None:
            return

        if state.active_nav_mode != "sca" or state.sca_visualization_mode != "isosurface" or not state.iso_enabled:
            iso_actor.SetVisibility(False)
            if ws_actor is not None:
                ws_actor.SetVisibility(False)
        else:
            cur_field = state.selected_global_field
            if volume_grid.GetPointData().HasArray(cur_field):
                iso_filter.SetInputArrayToProcess(0, 0, 0, 0, cur_field)
                iso_filter.SetValue(0, float(state.iso_value))
                iso_filter.Update()

                # Dynamic Wigner-Seitz clipping connection
                clip_active = bool(state.is_periodic and state.clip_to_wigner_seitz and ws_clipper is not None)
                if iso_mapper is not None:
                    if clip_active:
                        ws_clipper.Update()
                        iso_mapper.SetInputConnection(ws_clipper.GetOutputPort())
                    else:
                        iso_mapper.SetInputConnection(iso_filter.GetOutputPort())

                iso_actor.GetProperty().SetOpacity(float(state.iso_opacity))
                iso_actor.SetVisibility(True)

                if ws_actor is not None:
                    ws_actor.SetVisibility(bool(state.is_periodic and state.show_ws_boundary))
            else:
                iso_actor.SetVisibility(False)
                if ws_actor is not None:
                    ws_actor.SetVisibility(False)

        render_window.Render()
        request_view_update()

    def update_cutplane():
        cut_plane = pipeline_data.get("cut_plane")
        cutter = pipeline_data.get("cutter")
        cut_actor = pipeline_data.get("cut_actor")
        cut_mapper = pipeline_data.get("cut_mapper")
        contour_filter = pipeline_data.get("contour_filter")
        contour_actor = pipeline_data.get("contour_actor")
        volume_grid = pipeline_data.get("volume_grid")

        if cut_plane is None or cutter is None or cut_actor is None or contour_filter is None or contour_actor is None or volume_grid is None:
            return

        if state.active_nav_mode != "sca" or state.sca_visualization_mode != "cutplane" or not state.cut_enabled:
            cut_actor.SetVisibility(False)
            contour_actor.SetVisibility(False)
            render_window.Render()
            request_view_update()
            return

        orient = state.cut_orientation
        offset = float(state.cut_offset)
        if orient == "XY":
            cut_plane.SetNormal(0.0, 0.0, 1.0)
            cut_plane.SetOrigin(0.0, 0.0, offset)
        elif orient == "XZ":
            cut_plane.SetNormal(0.0, 1.0, 0.0)
            cut_plane.SetOrigin(0.0, offset, 0.0)
        elif orient == "YZ":
            cut_plane.SetNormal(1.0, 0.0, 0.0)
            cut_plane.SetOrigin(offset, 0.0, 0.0)

        cur_field = state.selected_global_field
        if volume_grid.GetPointData().HasArray(cur_field):
            cutter.Update()
            f_min, f_max, _, _ = get_field_slider_config(cur_field, (0, 1))
            n_levels = max(2, int(state.cut_num_contours))
            is_log = (state.cut_scale_type == "log")

            if state.cut_show_flood:
                stepped_ctf = build_discrete_colormap(n_levels, f_min, f_max, is_log)
                cut_mapper.SelectColorArray(cur_field)
                cut_mapper.SetLookupTable(stepped_ctf)
                cut_mapper.SetScalarRange(f_min, f_max)
                cut_actor.SetVisibility(True)
            else:
                cut_actor.SetVisibility(False)

            if state.cut_show_contours:
                contour_filter.SetInputArrayToProcess(0, 0, 0, 0, cur_field)
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
            cut_actor.SetVisibility(False)
            contour_actor.SetVisibility(False)

        render_window.Render()
        request_view_update()

    def update_gba_patches():
        is_gba = (state.active_nav_mode == "gba")
        gba_sphere_actor = pipeline_data.get("gba_sphere_actor")
        gba_sphere_solid_actor = pipeline_data.get("gba_sphere_solid_actor")
        gba_nucleus_actor = pipeline_data.get("gba_nucleus_actor")
        gba_atom_flood_actor = pipeline_data.get("gba_atom_flood_actor")
        gba_atom_flood_mapper = pipeline_data.get("gba_atom_flood_mapper")
        gba_atom_contour_actor = pipeline_data.get("gba_atom_contour_actor")
        gba_atom_contour_filter = pipeline_data.get("gba_atom_contour_filter")
        gba_highlight_actor = pipeline_data.get("gba_highlight_actor")
        gba_wedge_actor = pipeline_data.get("gba_wedge_actor")
        gba_wedge_edge_actor = pipeline_data.get("gba_wedge_edge_actor")
        gba_patch_actors = pipeline_data.get("gba_patch_actors", [])
        gba_sphere_poly = pipeline_data.get("gba_sphere_poly")

        if not is_gba:
            if gba_sphere_actor is not None:
                gba_sphere_actor.SetVisibility(False)
            if gba_sphere_solid_actor is not None:
                gba_sphere_solid_actor.SetVisibility(False)
            if gba_nucleus_actor is not None:
                gba_nucleus_actor.SetVisibility(False)
            if gba_atom_flood_actor is not None:
                gba_atom_flood_actor.SetVisibility(False)
            if gba_atom_contour_actor is not None:
                gba_atom_contour_actor.SetVisibility(False)
            if gba_highlight_actor is not None:
                gba_highlight_actor.SetVisibility(False)
            if gba_wedge_actor is not None:
                gba_wedge_actor.SetVisibility(False)
            if gba_wedge_edge_actor is not None:
                gba_wedge_edge_actor.SetVisibility(False)
            for p in gba_patch_actors:
                p["actor"].SetVisibility(False)
            render_window.Render()
            request_view_update()
            return

        if gba_nucleus_actor is not None:
            gba_nucleus_actor.SetVisibility(True)

        vis_mode = state.gba_visualization_mode

        if vis_mode == "contours":
            for p in gba_patch_actors:
                p["actor"].SetVisibility(False)
            if gba_sphere_solid_actor is not None:
                gba_sphere_solid_actor.SetVisibility(False)

            if gba_sphere_actor is not None:
                gba_sphere_actor.SetVisibility(bool(state.gba_show_sphere_boundary))

            if gba_sphere_poly is not None and gba_atom_flood_actor is not None and gba_atom_contour_actor is not None:
                pd = gba_sphere_poly.GetPointData()
                sel_field = state.selected_condensed_field
                sel_is_condensed = "(condensed)" in (sel_field or "").lower()

                def condensed_parity(arr_name: str) -> bool:
                    return ("(condensed)" in arr_name.lower()) == sel_is_condensed

                target_field = None
                for arr_i in range(pd.GetNumberOfArrays()):
                    arr_name = pd.GetArrayName(arr_i)
                    if arr_name and arr_name == sel_field:
                        target_field = arr_name
                        break

                if target_field is None:
                    sel_norm = normalize_field_name(sel_field)
                    for arr_i in range(pd.GetNumberOfArrays()):
                        arr_name = pd.GetArrayName(arr_i)
                        if arr_name and condensed_parity(arr_name) and normalize_field_name(arr_name) == sel_norm:
                            target_field = arr_name
                            break

                if target_field is None:
                    for arr_i in range(pd.GetNumberOfArrays()):
                        arr_name = pd.GetArrayName(arr_i)
                        if arr_name and condensed_parity(arr_name) and matches_field(arr_name, sel_field):
                            target_field = arr_name
                            break

                if target_field is not None:
                    arr = pd.GetArray(target_field)
                    f_min, f_max = get_robust_scalar_bounds(arr, lower_pct=2.0, upper_pct=98.0)
                    n_levels = max(2, int(state.gba_num_contours))
                    is_log = (state.gba_scale_type == "log")

                    if state.gba_show_flood:
                        stepped_ctf = build_discrete_colormap(n_levels, f_min, f_max, is_log)
                        gba_atom_flood_mapper.SelectColorArray(target_field)
                        gba_atom_flood_mapper.SetLookupTable(stepped_ctf)
                        gba_atom_flood_mapper.SetScalarRange(f_min, f_max)
                        gba_atom_flood_actor.SetVisibility(True)
                    else:
                        gba_atom_flood_actor.SetVisibility(False)

                    if state.gba_show_contours:
                        gba_atom_contour_filter.SetInputArrayToProcess(0, 0, 0, 0, target_field)
                        if is_log:
                            pos_min = max(f_min, 1e-4)
                            pos_max = max(f_max, pos_min * 10.0)
                            log_vals = [
                                float(10 ** (math.log10(pos_min) + i * (math.log10(pos_max) - math.log10(pos_min)) / (n_levels - 1)))
                                for i in range(n_levels)
                            ]
                            gba_atom_contour_filter.SetNumberOfContours(n_levels)
                            for i, val in enumerate(log_vals):
                                gba_atom_contour_filter.SetValue(i, val)
                        else:
                            step_lin = (f_max - f_min) / (n_levels - 1)
                            gba_atom_contour_filter.SetNumberOfContours(n_levels)
                            for i in range(n_levels):
                                gba_atom_contour_filter.SetValue(i, f_min + i * step_lin)

                        gba_atom_contour_filter.Update()
                        gba_atom_contour_actor.SetVisibility(True)
                    else:
                        gba_atom_contour_actor.SetVisibility(False)
                else:
                    gba_atom_flood_actor.SetVisibility(False)
                    gba_atom_contour_actor.SetVisibility(False)

            state.gba_active_basins_count = 0
            render_window.Render()
            request_view_update()
            return

        if gba_atom_flood_actor is not None:
            gba_atom_flood_actor.SetVisibility(False)
        if gba_atom_contour_actor is not None:
            gba_atom_contour_actor.SetVisibility(False)

        if gba_sphere_actor is not None:
            gba_sphere_actor.SetVisibility(bool(state.gba_show_sphere_boundary))
        if gba_sphere_solid_actor is not None:
            gba_sphere_solid_actor.SetVisibility(True)

        sel_field = state.selected_condensed_field
        show_min = bool(state.gba_show_min_basins)
        show_max = bool(state.gba_show_max_basins)

        matching_patches = []
        for p in gba_patch_actors:
            meta = p["meta"]
            fn = meta.get("function_name", "")
            region = meta.get("region_type", "")
            is_match_field = matches_field(fn, sel_field)
            is_min = "minimum" in region
            is_max = "maximum" in region

            if is_match_field and ((is_min and show_min) or (is_max and show_max)):
                matching_patches.append(p)
            else:
                p["actor"].SetVisibility(False)

        assigned_colors = assign_neighbor_aware_basin_colors(matching_patches)
        active_basins_info = []
        for i, p in enumerate(matching_patches):
            meta = p["meta"]
            b_idx = meta.get("basin_index", i)
            c = assigned_colors[i] if i < len(assigned_colors) else (0.20, 0.60, 0.86)

            p["actor"].GetProperty().SetColor(*c)
            p["actor"].SetVisibility(True)

            raw_totals = meta.get("integrated_totals", [])
            display_totals = [
                {
                    "name": get_display_title(item.get("name", "")),
                    "value": item.get("value", 0.0),
                    "formatted": item.get("formatted", ""),
                }
                for item in raw_totals
            ]

            active_basins_info.append({
                "basin_index": b_idx,
                "atom_number": meta.get("atom_number", 1),
                "atom_type": meta.get("atom_type", "C"),
                "function_name": get_display_title(meta.get("function_name", "")),
                "region_type": meta.get("region_type", "minimum"),
                "num_triangles": meta.get("num_triangles", 0),
                "num_nodes": meta.get("num_nodes", 0),
                "integrated_totals": display_totals,
                "color": [c[0], c[1], c[2]],
                "color_hex": f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}",
            })

        state.gba_active_basins_count = len(matching_patches)
        state.gba_active_basins_list = active_basins_info
        render_window.Render()
        request_view_update()

    def update_gba_wedge_geometry():
        nonlocal active_gba_patch_poly
        gba_wedge_actor = pipeline_data.get("gba_wedge_actor")
        gba_wedge_mapper = pipeline_data.get("gba_wedge_mapper")
        gba_wedge_edges = pipeline_data.get("gba_wedge_edges")
        gba_wedge_edge_actor = pipeline_data.get("gba_wedge_edge_actor")
        gba_surface_blocks = pipeline_data.get("gba_surface_blocks", [])
        atoms = pipeline_data.get("atoms", [])

        if gba_wedge_actor is None or gba_wedge_mapper is None:
            return

        basin_entry = state.selected_gba_basin
        is_gba = (state.active_nav_mode == "gba")

        if not is_gba or basin_entry is None or active_gba_patch_poly is None:
            gba_wedge_actor.SetVisibility(False)
            if gba_wedge_edge_actor is not None:
                gba_wedge_edge_actor.SetVisibility(False)
            render_window.Render()
            request_view_update()
            return

        patch_poly = active_gba_patch_poly
        if patch_poly.GetNumberOfPoints() == 0:
            gba_wedge_actor.SetVisibility(False)
            if gba_wedge_edge_actor is not None:
                gba_wedge_edge_actor.SetVisibility(False)
            render_window.Render()
            request_view_update()
            return

        atom_num = basin_entry.get("atom_number", 1)
        atom_type = basin_entry.get("atom_type", "")
        basin_idx = basin_entry.get("basin_index", 0)

        # 1. Prefer true geometric center of the GBA sphere mesh if available
        if "gba_nucleus_center" in pipeline_data and pipeline_data["gba_nucleus_center"] is not None:
            nucleus_pos = list(pipeline_data["gba_nucleus_center"])
        elif "gba_sphere_poly" in pipeline_data and pipeline_data["gba_sphere_poly"] is not None:
            bnds = pipeline_data["gba_sphere_poly"].GetBounds()
            nucleus_pos = [(bnds[0] + bnds[1]) * 0.5, (bnds[2] + bnds[3]) * 0.5, (bnds[4] + bnds[5]) * 0.5]
        else:
            nucleus_pos = [0.0, 0.0, 0.0]
            # Match atom by ID or element name (e.g. C1, Pd1, etc.)
            matched = False
            for a in atoms:
                if a.get("id") == atom_num:
                    nucleus_pos = a["raw_pos"]
                    matched = True
                    break
                if atom_type and a.get("name") == f"{atom_type}{atom_num}":
                    nucleus_pos = a["raw_pos"]
                    matched = True
                    break
            if not matched and atoms:
                # Fallback to closest atom or first atom if available
                nucleus_pos = atoms[0]["raw_pos"]

        matching_surface = None
        basin_func = basin_entry.get("function_name", "")
        basin_region = str(basin_entry.get("region_type", "")).strip().lower()
        for s in gba_surface_blocks:
            s_meta = s.get("meta", {})
            try:
                s_idx = int(s_meta.get("basin_index", -1))
                if s_idx == int(basin_idx):
                    s_fn = s_meta.get("function_name", "")
                    s_reg = str(s_meta.get("region_type", "")).strip().lower()
                    if basin_region and s_reg and (basin_region in s_reg or s_reg in basin_region):
                        if not basin_func or not s_fn or matches_field(s_fn, basin_func):
                            matching_surface = s["poly"]
                            break
                    elif not basin_region:
                        if not basin_func or not s_fn or matches_field(s_fn, basin_func):
                            matching_surface = s["poly"]
                            break
            except (ValueError, TypeError):
                pass

        if matching_surface is not None and matching_surface.GetNumberOfPoints() > 0:
            wedge_poly = matching_surface
        else:
            wedge_poly = build_pyramidal_wedge_polydata(patch_poly, (nucleus_pos[0], nucleus_pos[1], nucleus_pos[2]))

        gba_wedge_mapper.SetInputData(wedge_poly)
        gba_wedge_mapper.Update()

        if gba_wedge_edges is not None and gba_wedge_edge_actor is not None:
            gba_wedge_edges.SetInputData(wedge_poly)
            gba_wedge_edges.Update()
            show_edges = bool(state.gba_show_wedge_edges)
            gba_wedge_edge_actor.SetVisibility(show_edges)

        patch_color = basin_entry.get("color", [0.25, 0.70, 0.95])
        gba_wedge_actor.GetProperty().SetColor(*patch_color)
        gba_wedge_actor.GetProperty().SetOpacity(float(state.gba_wedge_opacity))
        gba_wedge_actor.SetVisibility(True)

        render_window.Render()
        request_view_update()

    def select_gba_basin(basin_entry: Optional[Dict[str, Any]], patch_poly: Optional[Any] = None):
        nonlocal active_gba_patch_poly
        state.selected_gba_basin = basin_entry
        active_gba_patch_poly = patch_poly
        gba_highlight_actor = pipeline_data.get("gba_highlight_actor")
        gba_highlight_edges = pipeline_data.get("gba_highlight_edges")
        gba_wedge_actor = pipeline_data.get("gba_wedge_actor")
        gba_wedge_edge_actor = pipeline_data.get("gba_wedge_edge_actor")

        if basin_entry is None or patch_poly is None or gba_highlight_actor is None or gba_highlight_edges is None:
            if gba_highlight_actor is not None:
                gba_highlight_actor.SetVisibility(False)
            if gba_wedge_actor is not None:
                gba_wedge_actor.SetVisibility(False)
            if gba_wedge_edge_actor is not None:
                gba_wedge_edge_actor.SetVisibility(False)
        else:
            gba_highlight_edges.SetInputData(patch_poly)
            gba_highlight_edges.Update()
            gba_highlight_actor.SetVisibility(True)
            update_gba_wedge_geometry()

        render_window.Render()
        request_view_update()

    def select_item(item: Optional[Dict[str, Any]]):
        state.selected_item = item
        highlight_actor = pipeline_data.get("highlight_actor")
        highlight_source = pipeline_data.get("highlight_source")
        if highlight_actor is None or highlight_source is None:
            return

        if item is None:
            highlight_actor.SetVisibility(False)
        else:
            raw_pos = item["raw_pos"]
            highlight_source.SetCenter(raw_pos[0], raw_pos[1], raw_pos[2])
            if "element" in item:
                el = item["element"]
                r_cov = get_covalent_radius(el, default=0.75)
                glyph_scale = r_cov * BALL_AND_STICK_SCALE
                highlight_source.SetRadius(glyph_scale * 0.5 * 1.20)
            else:
                highlight_source.SetRadius(0.18 * 0.5 * 1.20)

            highlight_source.Update()
            highlight_actor.SetVisibility(True)

        render_window.Render()
        request_view_update()

    def load_dataset_into_viewer(target_file_path: str):
        """
        Dynamically load or hot-swap a .plt or .vtm dataset into the running viewer.
        """
        if not target_file_path:
            return

        try:
            resolved_vtm = prepare_dataset_file(target_file_path)
            populate_dataset_pipeline(resolved_vtm, renderer, pipeline_data)

            mol_info = pipeline_data.get("molecule_info", {})
            vol_grid = pipeline_data.get("volume_grid")
            atoms = pipeline_data.get("atoms", [])
            critical_points = pipeline_data.get("critical_points", [])
            actors = pipeline_data.get("actors", {})

            def_field = mol_info.get("selected_global_field", "Electron Density")
            raw_r = (0.001, 1.0)
            if vol_grid is not None and vol_grid.GetPointData().HasArray(def_field):
                raw_r = vol_grid.GetPointData().GetArray(def_field).GetRange()
            f_min, f_max, f_val, f_step = get_field_slider_config(def_field, raw_r)

            lat_info = pipeline_data.get("lattice_info")
            is_periodic_load = bool(lat_info is not None and lat_info.get("is_periodic", False))

            state.has_dataset = True
            state.is_periodic = is_periodic_load
            state.clip_to_wigner_seitz = is_periodic_load
            state.show_ws_boundary = is_periodic_load
            state.overview_clip_to_ws = is_periodic_load
            state.overview_show_ws_boundary = is_periodic_load
            state.vtm_file = os.path.basename(resolved_vtm)
            state.current_file_path = resolved_vtm
            state.num_blocks = len(actors)
            state.block_names = list(actors.keys())
            state.molecule_info = mol_info
            state.atoms_list = atoms
            state.cps_list = critical_points
            state.selected_item = None
            state.selected_gba_basin = None
            state.selected_global_field = def_field
            gba_atoms_avail = mol_info.get("gba_atoms_list", [])
            state.selected_gba_atom_id = gba_atoms_avail[0] if gba_atoms_avail else "C1"
            state.selected_condensed_field = mol_info.get("selected_condensed_field", "Electron Density")
            state.iso_min = f_min
            state.iso_max = f_max
            state.iso_value = f_val
            state.iso_step = f_step
            state.has_volume_data = (vol_grid is not None)

            update_skeleton_visibility()
            update_isosurface()
            update_cutplane()
            update_gba_patches()
            update_gba_wedge_geometry()
            select_item(None)
            select_gba_basin(None)

            renderer.ResetCamera()
            render_window.Render()
            request_view_update()
            print(f"[Bondalyzer] Successfully loaded dataset: {resolved_vtm}")
        except Exception as e:
            print(f"[Bondalyzer] Error loading dataset '{target_file_path}': {e}")

    @ctrl.add("open_file_dialog")
    def open_file_dialog():
        """
        Trigger the cross-platform file selection. First attempts native OS dialog.
        If native dialog is unavailable or runs in a headless environment, opens the in-app file browser modal.
        """
        init_dir = os.path.dirname(state.current_file_path) if state.current_file_path else os.getcwd()
        selected = open_native_file_dialog(init_dir)
        if selected:
            load_dataset_into_viewer(selected)
        elif selected is None and not (sys.platform == "darwin" or sys.platform == "win32" or shutil.which("zenity") or shutil.which("kdialog")):
            # Open web-based in-app file browser modal as fallback
            ctrl.open_in_app_browser()

    @ctrl.add("open_in_app_browser")
    def open_in_app_browser():
        """Explicitly open the in-app server file browser modal."""
        cur_dir = os.path.dirname(state.current_file_path) if state.current_file_path else os.getcwd()
        b_data = list_server_directory(cur_dir)
        state.browser_current_path = b_data["current_path"]
        state.browser_parent_path = b_data["parent_path"]
        state.browser_breadcrumbs = b_data["breadcrumbs"]
        state.browser_entries = b_data["entries"]
        state.browser_selected_path = ""
        state.file_dialog_modal_open = True

    @ctrl.add("browser_navigate")
    def browser_navigate(target_dir):
        """Navigate to directory inside the in-app file browser modal."""
        b_data = list_server_directory(target_dir)
        state.browser_current_path = b_data["current_path"]
        state.browser_parent_path = b_data["parent_path"]
        state.browser_breadcrumbs = b_data["breadcrumbs"]
        state.browser_entries = b_data["entries"]
        state.browser_selected_path = ""

    @ctrl.add("browser_select_entry")
    def browser_select_entry(entry_path, is_dir):
        """Handle entry selection or folder expansion in in-app file browser."""
        if is_dir:
            browser_navigate(entry_path)
        else:
            state.browser_selected_path = entry_path

    @ctrl.add("browser_confirm_load")
    def browser_confirm_load():
        """Confirm selection from in-app file browser modal and load dataset."""
        sel = state.browser_selected_path
        if sel and os.path.exists(sel):
            state.file_dialog_modal_open = False
            load_dataset_into_viewer(sel)

    @ctrl.add("browser_cancel")
    def browser_cancel():
        """Cancel and dismiss in-app file browser modal."""
        state.file_dialog_modal_open = False

    @state.change("active_nav_mode", "sca_visualization_mode")
    def on_nav_mode_change(active_nav_mode=None, **kwargs):
        if active_nav_mode != "overview":
            select_item(None)
        if active_nav_mode != "gba":
            select_gba_basin(None)
        update_skeleton_visibility()
        update_isosurface()
        update_cutplane()
        update_gba_patches()
        update_gba_wedge_geometry()

    @state.change("gba_visualization_mode", "selected_condensed_field", "gba_show_sphere_boundary", "gba_show_min_basins", "gba_show_max_basins", "selected_gba_atom_id", "gba_show_contours", "gba_show_flood", "gba_scale_type", "gba_num_contours")
    def on_gba_param_change(**kwargs):
        update_skeleton_visibility()
        update_gba_patches()

    @state.change("gba_wedge_opacity", "gba_show_wedge_edges")
    def on_wedge_param_change(**kwargs):
        update_gba_wedge_geometry()

    @state.change(
        "show_inferred_bonds",
        "show_bond_paths",
        "show_ring_paths",
        "show_cage_paths",
        "show_bond_cps",
        "show_ring_cps",
        "show_cage_cps",
        "overview_clip_to_ws",
        "overview_show_ws_boundary",
        "atom_scale_factor",
    )
    def on_skeleton_visibility_change(**kwargs):
        update_skeleton_visibility()

    update_skeleton_visibility()
    update_isosurface()
    update_cutplane()
    update_gba_patches()
    update_gba_wedge_geometry()

    @state.change("iso_enabled", "iso_value", "iso_opacity", "clip_to_wigner_seitz", "show_ws_boundary")
    def on_iso_param_change(**kwargs):
        update_isosurface()

    @state.change("cut_enabled", "cut_orientation", "cut_offset", "cut_show_contours", "cut_show_flood", "cut_scale_type", "cut_num_contours")
    def on_cut_param_change(**kwargs):
        update_cutplane()

    @state.change("selected_global_field")
    def on_field_change(selected_global_field, **kwargs):
        volume_grid = pipeline_data.get("volume_grid")
        if volume_grid is not None and volume_grid.GetPointData().HasArray(selected_global_field):
            raw_r = volume_grid.GetPointData().GetArray(selected_global_field).GetRange()
            f_min, f_max, f_val, f_step = get_field_slider_config(selected_global_field, raw_r)
            state.iso_min = f_min
            state.iso_max = f_max
            state.iso_value = f_val
            state.iso_step = f_step
            if "electron density" in selected_global_field.lower() or "willmore" in selected_global_field.lower():
                state.cut_scale_type = "log"
            else:
                state.cut_scale_type = "linear"
            update_isosurface()
            update_cutplane()

    picker = vtkCellPicker()
    picker.SetTolerance(0.02)
    world_coord = vtkCoordinate()
    world_coord.SetCoordinateSystemToWorld()

    @ctrl.add("on_scene_click")
    def on_scene_click(click_x=None, click_y=None, client_w=None, client_h=None):
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

            gba_patch_actors = pipeline_data.get("gba_patch_actors", [])
            gba_wedge_actor = pipeline_data.get("gba_wedge_actor")
            atoms = pipeline_data.get("atoms", [])
            critical_points = pipeline_data.get("critical_points", [])

            if state.active_nav_mode == "gba":
                if state.gba_visualization_mode != "basins":
                    return

                visible_patches = [p for p in gba_patch_actors if p["actor"].GetVisibility()]
                if not visible_patches:
                    return

                picker.Pick(disp_x, disp_y, 0, renderer)
                picked_actor = picker.GetActor()

                if picked_actor is not None:
                    if gba_wedge_actor is not None and picked_actor == gba_wedge_actor:
                        select_gba_basin(None)
                        return

                    for patch in visible_patches:
                        if patch["actor"] == picked_actor:
                            meta = patch["meta"]
                            patch_poly = patch["poly"]
                            patch_color = patch["actor"].GetProperty().GetColor()
                            basin_idx = meta.get("basin_index", 0)
                            region_type = meta.get("region_type", "minimum")

                            if (
                                state.selected_gba_basin
                                and str(state.selected_gba_basin.get("basin_index", "")) == str(basin_idx)
                                and str(state.selected_gba_basin.get("region_type", "")).lower() == str(region_type).lower()
                            ):
                                select_gba_basin(None)
                                return

                            basin_info = {
                                "basin_index": basin_idx,
                                "atom_number": meta.get("atom_number", 1),
                                "atom_type": meta.get("atom_type", "C"),
                                "function_name": get_display_title(meta.get("function_name", "")),
                                "region_type": region_type,
                                "num_triangles": meta.get("num_triangles", 0),
                                "num_nodes": meta.get("num_nodes", 0),
                                "integrated_totals": [
                                    {
                                        "name": get_display_title(item.get("name", "")),
                                        "value": item.get("value", 0.0),
                                        "formatted": item.get("formatted", ""),
                                    }
                                    for item in meta.get("integrated_totals", [])
                                ],
                                "color": [patch_color[0], patch_color[1], patch_color[2]],
                            }
                            select_gba_basin(basin_info, patch_poly=patch_poly)
                            return

                min_screen_dist = float("inf")
                best_patch = None
                for patch in visible_patches:
                    poly = patch["poly"]
                    if poly and poly.GetNumberOfPoints() > 0:
                        bnds = poly.GetBounds()
                        cx = (bnds[0] + bnds[1]) * 0.5
                        cy = (bnds[2] + bnds[3]) * 0.5
                        cz = (bnds[4] + bnds[5]) * 0.5
                        world_coord.SetValue(cx, cy, cz)
                        disp_val = world_coord.GetComputedDisplayValue(renderer)
                        sdx = disp_val[0] - disp_x
                        sdy = disp_val[1] - disp_y
                        s_dist = math.sqrt(sdx * sdx + sdy * sdy)
                        if s_dist < min_screen_dist:
                            min_screen_dist = s_dist
                            best_patch = patch

                if best_patch is not None and min_screen_dist <= 60.0:
                    meta = best_patch["meta"]
                    basin_idx = meta.get("basin_index", 0)
                    region_type = meta.get("region_type", "minimum")
                    if (
                        state.selected_gba_basin
                        and str(state.selected_gba_basin.get("basin_index", "")) == str(basin_idx)
                        and str(state.selected_gba_basin.get("region_type", "")).lower() == str(region_type).lower()
                    ):
                        select_gba_basin(None)
                        return

                    patch_color = best_patch["actor"].GetProperty().GetColor()
                    basin_info = {
                        "basin_index": basin_idx,
                        "atom_number": meta.get("atom_number", 1),
                        "atom_type": meta.get("atom_type", "C"),
                        "function_name": get_display_title(meta.get("function_name", "")),
                        "region_type": region_type,
                        "num_triangles": meta.get("num_triangles", 0),
                        "num_nodes": meta.get("num_nodes", 0),
                        "integrated_totals": [
                            {
                                "name": get_display_title(item.get("name", "")),
                                "value": item.get("value", 0.0),
                                "formatted": item.get("formatted", ""),
                            }
                            for item in meta.get("integrated_totals", [])
                        ],
                        "color": [patch_color[0], patch_color[1], patch_color[2]],
                    }
                    select_gba_basin(basin_info, patch_poly=best_patch["poly"])
                    return
                return

            if state.active_nav_mode != "overview":
                return

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

                threshold = 1.35
                if best_candidate is not None and min_world_dist <= threshold:
                    select_item(best_candidate)
                    return

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

            if screen_candidate is not None and min_screen_dist <= 40.0:
                select_item(screen_candidate)
                return

            select_item(None)
        except Exception as e:
            print(f"[Bondalyzer] Picking notice: {e}")

    @ctrl.add("select_atom_from_list")
    def select_atom_from_list(atom_id):
        for a in pipeline_data.get("atoms", []):
            if a["id"] == atom_id:
                select_item(a)
                return

    @ctrl.add("select_cp_from_list")
    def select_cp_from_list(cp_id):
        for cp in pipeline_data.get("critical_points", []):
            if cp["id"] == cp_id:
                select_item(cp)
                return

    @ctrl.add("select_gba_basin_from_list")
    def select_gba_basin_from_list(basin_idx, region_type=None):
        if (
            state.selected_gba_basin
            and str(state.selected_gba_basin.get("basin_index", "")) == str(basin_idx)
            and (region_type is None or str(state.selected_gba_basin.get("region_type", "")).lower() == str(region_type).lower())
        ):
            select_gba_basin(None)
            return

        sel_field = state.selected_condensed_field
        target_reg = str(region_type).strip().lower() if region_type else ""

        # First search among visible/matching patches for the active field and region type
        for patch in pipeline_data.get("gba_patch_actors", []):
            meta = patch["meta"]
            p_reg = str(meta.get("region_type", "")).strip().lower()
            if (
                str(meta.get("basin_index", "")) == str(basin_idx)
                and matches_field(meta.get("function_name", ""), sel_field)
                and (not target_reg or target_reg in p_reg or p_reg in target_reg)
            ):
                patch_poly = patch["poly"]
                patch_color = patch["actor"].GetProperty().GetColor()
                basin_info = {
                    "basin_index": meta.get("basin_index", 0),
                    "atom_number": meta.get("atom_number", 1),
                    "atom_type": meta.get("atom_type", "C"),
                    "function_name": get_display_title(meta.get("function_name", "")),
                    "region_type": meta.get("region_type", "minimum"),
                    "num_triangles": meta.get("num_triangles", 0),
                    "num_nodes": meta.get("num_nodes", 0),
                    "integrated_totals": [
                        {
                            "name": get_display_title(item.get("name", "")),
                            "value": item.get("value", 0.0),
                            "formatted": item.get("formatted", ""),
                        }
                        for item in meta.get("integrated_totals", [])
                    ],
                    "color": [patch_color[0], patch_color[1], patch_color[2]],
                }
                select_gba_basin(basin_info, patch_poly=patch_poly)
                return

        # Fallback to any patch matching the basin index and region type
        for patch in pipeline_data.get("gba_patch_actors", []):
            meta = patch["meta"]
            p_reg = str(meta.get("region_type", "")).strip().lower()
            if (
                str(meta.get("basin_index", "")) == str(basin_idx)
                and (not target_reg or target_reg in p_reg or p_reg in target_reg)
            ):
                patch_poly = patch["poly"]
                patch_color = patch["actor"].GetProperty().GetColor()
                basin_info = {
                    "basin_index": meta.get("basin_index", 0),
                    "atom_number": meta.get("atom_number", 1),
                    "atom_type": meta.get("atom_type", "C"),
                    "function_name": get_display_title(meta.get("function_name", "")),
                    "region_type": meta.get("region_type", "minimum"),
                    "num_triangles": meta.get("num_triangles", 0),
                    "num_nodes": meta.get("num_nodes", 0),
                    "integrated_totals": [
                        {
                            "name": get_display_title(item.get("name", "")),
                            "value": item.get("value", 0.0),
                            "formatted": item.get("formatted", ""),
                        }
                        for item in meta.get("integrated_totals", [])
                    ],
                    "color": [patch_color[0], patch_color[1], patch_color[2]],
                }
                select_gba_basin(basin_info, patch_poly=patch_poly)
                return

    @ctrl.add("clear_selection")
    def clear_selection():
        select_item(None)

    @ctrl.add("clear_gba_basin_selection")
    def clear_gba_basin_selection():
        select_gba_basin(None)

    @ctrl.add("reset_camera")
    def reset_camera():
        renderer.ResetCamera()
        render_window.Render()
        request_view_update()

    # Build UI Layout with Collapsible Side Drawer
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Bondalyzer Molecule Viewer")
        layout.drawer.width = 410

        # --- DRAWER (Molecule Info & Feature Inspector) ---
        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-3"):

                # Primary Navigation Tabs: Viewer, SCA Tools, GBA Tools
                with v3.VTabs(
                    v_model=("active_nav_mode", "overview"),
                    density="compact",
                    color="primary",
                    grow=True,
                    classes="mb-3 rounded elevation-1",
                ):
                    v3.VTab("Viewer", value="overview", prepend_icon="mdi-molecule")
                    v3.VTab("SCA Tools", value="sca", prepend_icon="mdi-layers-outline", disabled=("!has_dataset",))
                    v3.VTab("GBA Tools", value="gba", prepend_icon="mdi-chart-bubble", disabled=("!has_dataset",))

                # =====================================================================
                # TAB 1: MOLECULE VIEWER & SKELETON
                # =====================================================================
                with v3.VWindow(v_model=("active_nav_mode", "overview")):
                    with v3.VWindowItem(value="overview"):

                        # Empty dataset placeholder card
                        with v3.VCard(
                            v_if="!has_dataset",
                            elevation=2,
                            classes="mb-3 pa-4 text-center",
                            color="surface-variant",
                        ):
                            v3.VIcon("mdi-folder-open-outline", size="48", color="primary", classes="mb-2")
                            html.Div("No Dataset Loaded", classes="text-h6 font-weight-bold mb-1")
                            html.Div(
                                "Open a Tecplot .plt or VTK .vtm file to visualize molecular geometry, critical points, and gradient bundles.",
                                classes="text-caption text-medium-emphasis mb-3",
                            )
                            with v3.VBtn(
                                "Open Dataset",
                                prepend_icon="mdi-folder-open",
                                color="primary",
                                variant="elevated",
                                click=ctrl.open_file_dialog,
                            ):
                                pass

                        # 1. Molecule Summary Card (shown when dataset loaded)
                        with v3.VCard(v_if="has_dataset", elevation=2, classes="mb-3", color="surface-variant"):
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

                        # Display Elements / Framework Visibility Controls
                        with v3.VCard(v_if="has_dataset", elevation=1, classes="mb-3"):
                            with v3.VCardItem():
                                with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"):
                                    with html.Div(classes="d-flex align-center"):
                                        v3.VIcon("mdi-eye-outline", classes="mr-2", color="primary")
                                        html.Span("Display Elements")
                            v3.VDivider()
                            with v3.VCardText(classes="pt-2 pb-2"):
                                html.Div("1D Paths & Bonds", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                v3.VSwitch(
                                    v_if="molecule_info.bonds > 0",
                                    label="Inferred Bonds (Connectivity)",
                                    v_model=("show_inferred_bonds", True),
                                    density="compact",
                                    color="primary",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    v_if="molecule_info.bond_paths > 0",
                                    label="Bond Paths (3, -1)",
                                    v_model=("show_bond_paths", True),
                                    density="compact",
                                    color="info",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    v_if="molecule_info.ring_paths > 0",
                                    label="Ring Paths (3, +1)",
                                    v_model=("show_ring_paths", True),
                                    density="compact",
                                    color="success",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    v_if="molecule_info.cage_paths > 0",
                                    label="Cage Paths (3, +3)",
                                    v_model=("show_cage_paths", True),
                                    density="compact",
                                    color="warning",
                                    hide_details=True,
                                    classes="mb-1",
                                )

                                v3.VDivider(v_if="molecule_info.total_cps > 0", classes="my-2")
                                html.Div(v_if="molecule_info.total_cps > 0", children=["Critical Points"], classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                v3.VSwitch(
                                    v_if="molecule_info.bond_cps > 0",
                                    label="Bond CPs (3, -1)",
                                    v_model=("show_bond_cps", True),
                                    density="compact",
                                    color="error",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    v_if="molecule_info.ring_cps > 0",
                                    label="Ring CPs (3, +1)",
                                    v_model=("show_ring_cps", True),
                                    density="compact",
                                    color="success",
                                    hide_details=True,
                                    classes="mb-1",
                                )
                                v3.VSwitch(
                                    v_if="molecule_info.cage_cps > 0",
                                    label="Cage CPs (3, +3)",
                                    v_model=("show_cage_cps", True),
                                    density="compact",
                                    color="info",
                                    hide_details=True,
                                )

                                # Atom Size Slider
                                with html.Div(classes="mt-3"):
                                    v3.VDivider(classes="my-2")
                                    with html.Div(classes="d-flex justify-space-between align-center mb-1"):
                                        html.Div("Atom Sphere Size", classes="text-caption font-weight-bold text-medium-emphasis")
                                        html.Div("{{ Math.round((Number(atom_scale_factor) || 0) * 100) }}%", classes="text-caption font-weight-bold text-primary")
                                    v3.VSlider(
                                        min=0.0,
                                        max=2.0,
                                        step=0.05,
                                        v_model=("atom_scale_factor", 1.0),
                                        density="compact",
                                        thumb_label=False,
                                        color="primary",
                                        hide_details=True,
                                    )

                                # Periodic Crystal Wigner-Seitz Masking Controls in Viewer
                                with html.Div(v_if="is_periodic", classes="mt-2"):
                                    v3.VDivider(classes="my-2")
                                    html.Div("Crystal Cell Boundary (Periodic Solid)", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                    v3.VSwitch(
                                        label="Mask Framework to WS Cell",
                                        v_model=("overview_clip_to_ws", True),
                                        density="compact",
                                        color="primary",
                                        hide_details=True,
                                        classes="mb-1",
                                    )
                                    v3.VSwitch(
                                        label="Show Wigner-Seitz Wireframe",
                                        v_model=("overview_show_ws_boundary", True),
                                        density="compact",
                                        color="amber-darken-2",
                                        hide_details=True,
                                    )

                        # 2. Selected Feature Details Card (Appears on click/selection)
                        with v3.VCard(
                            v_if="has_dataset && selected_item",
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
                            v_if="has_dataset && !selected_item",
                            type="info",
                            variant="tonal",
                            density="compact",
                            classes="mb-3 text-caption",
                        ):
                            html.Div("Click any atom or critical point sphere in the 3D view to inspect its properties.")

                        # 3. Atoms List
                        with v3.VCard(v_if="has_dataset", elevation=1):
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

                                # Cutplane Controls
                                with html.Div(v_if="sca_visualization_mode === 'cutplane'"):
                                    with v3.VRow(dense=True, classes="align-center mb-2"):
                                        with v3.VCol(cols=12):
                                            v3.VSwitch(
                                                label="Enable Cutplane",
                                                v_model=("cut_enabled",),
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                            )

                                    with html.Div(v_if="cut_enabled"):
                                        html.Div("Slice Plane Orientation", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                        with v3.VBtnToggle(
                                            v_model=("cut_orientation", "XY"),
                                            density="compact",
                                            color="primary",
                                            mandatory=True,
                                            classes="mb-3 d-flex justify-center",
                                        ):
                                            v3.VBtn("XY Plane", value="XY", size="small")
                                            v3.VBtn("XZ Plane", value="XZ", size="small")
                                            v3.VBtn("YZ Plane", value="YZ", size="small")

                                        with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                            html.Div("Slice Position Offset", classes="text-caption font-weight-bold text-medium-emphasis")
                                            html.Div("{{ (Number(cut_offset) || 0).toFixed(2) }} Å", classes="text-caption font-weight-bold text-primary")

                                        v3.VSlider(
                                            min=("cut_offset_min",),
                                            max=("cut_offset_max",),
                                            step=("cut_offset_step",),
                                            v_model=("cut_offset",),
                                            density="compact",
                                            thumb_label="always",
                                            color="primary",
                                            classes="mt-1",
                                        )

                                        v3.VDivider(classes="my-2")

                                        html.Div("Contour Scaling Mode", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                        with v3.VBtnToggle(
                                            v_model=("cut_scale_type", "log"),
                                            density="compact",
                                            color="primary",
                                            mandatory=True,
                                            classes="mb-3 d-flex justify-center",
                                        ):
                                            v3.VBtn("Linear Contours", value="linear", size="small", prepend_icon="mdi-ruler")
                                            v3.VBtn("Logarithmic Contours", value="log", size="small", prepend_icon="mdi-math-log")

                                        with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                            html.Div("Number of Contour Lines", classes="text-caption font-weight-bold text-medium-emphasis")
                                            html.Div("{{ cut_num_contours }} levels", classes="text-caption font-weight-bold text-primary")

                                        v3.VSlider(
                                            min=3,
                                            max=40,
                                            step=1,
                                            v_model=("cut_num_contours",),
                                            density="compact",
                                            thumb_label=False,
                                            color="primary",
                                            classes="mt-1 mb-2",
                                        )

                                        v3.VDivider(classes="my-2")

                                        with v3.VRow(dense=True, classes="align-center"):
                                            with v3.VCol(cols=6):
                                                v3.VSwitch(
                                                    label="Contour Lines",
                                                    v_model=("cut_show_contours",),
                                                    density="compact",
                                                    color="primary",
                                                    hide_details=True,
                                                )
                                            with v3.VCol(cols=6):
                                                v3.VSwitch(
                                                    label="Color Flood",
                                                    v_model=("cut_show_flood",),
                                                    density="compact",
                                                    color="primary",
                                                    hide_details=True,
                                                )

                                    with v3.VAlert(
                                        v_if="!has_volume_data",
                                        type="warning",
                                        variant="tonal",
                                        density="compact",
                                        classes="text-caption mt-2",
                                    ):
                                        html.Div("Volume data file (*_zone0.vts / .vti / .vtr) not detected.")

                                # Isosurface Controls
                                with html.Div(v_if="sca_visualization_mode === 'isosurface'"):
                                    with v3.VRow(dense=True, classes="align-center mb-2"):
                                        with v3.VCol(cols=12):
                                            v3.VSwitch(
                                                label="Enable Isosurface",
                                                v_model=("iso_enabled",),
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                            )

                                    with html.Div(v_if="iso_enabled"):
                                        with html.Div(classes="d-flex justify-space-between align-center mt-2"):
                                            html.Div("Isosurface Value (Isovalue)", classes="text-caption font-weight-bold text-medium-emphasis")
                                            html.Div("{{ (Number(iso_value) || 0).toFixed(4) }}", classes="text-caption font-weight-bold text-primary")

                                        v3.VSlider(
                                            min=("iso_min",),
                                            max=("iso_max",),
                                            step=("iso_step",),
                                            v_model=("iso_value",),
                                            density="compact",
                                            thumb_label="always",
                                            color="primary",
                                            classes="mt-1",
                                        )

                                        with html.Div(classes="d-flex justify-space-between align-center mt-2"):
                                            html.Div("Surface Opacity", classes="text-caption font-weight-bold text-medium-emphasis")
                                            html.Div("{{ Math.round((Number(iso_opacity) || 0) * 100) }}%", classes="text-caption font-weight-bold")

                                        v3.VSlider(
                                            min=0.05,
                                            max=1.0,
                                            step=0.05,
                                            v_model=("iso_opacity",),
                                            density="compact",
                                            thumb_label=False,
                                            color="primary",
                                            classes="mt-1",
                                        )

                                        # Periodic Crystal Wigner-Seitz Masking Controls
                                        with html.Div(v_if="is_periodic", classes="mt-3"):
                                            v3.VDivider(classes="mb-2")
                                            html.Div("Crystal Cell Boundary (Periodic Solid)", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                            v3.VSwitch(
                                                label="Mask to Wigner-Seitz Cell",
                                                v_model=("clip_to_wigner_seitz", True),
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                                classes="mb-1",
                                            )
                                            v3.VSwitch(
                                                label="Show Wigner-Seitz Wireframe",
                                                v_model=("show_ws_boundary", True),
                                                density="compact",
                                                color="amber-darken-2",
                                                hide_details=True,
                                            )

                                    with v3.VAlert(
                                        v_if="!has_volume_data",
                                        type="warning",
                                        variant="tonal",
                                        density="compact",
                                        classes="text-caption mt-2",
                                    ):
                                        html.Div("Volume data file (*_zone0.vts / .vti / .vtr) not detected.")

                                # Global Atom Display Scaling Controls in SCA Tools
                                v3.VDivider(classes="my-3")
                                with html.Div(classes="mb-1"):
                                    with html.Div(classes="d-flex justify-space-between align-center mb-1"):
                                        with html.Div(classes="d-flex align-center"):
                                            v3.VIcon("mdi-atom", size="small", classes="mr-1", color="primary")
                                            html.Span("Atom Sphere Size", classes="text-caption font-weight-bold text-medium-emphasis")
                                        html.Div("{{ Math.round((Number(atom_scale_factor) || 0) * 100) }}%", classes="text-caption font-weight-bold text-primary")
                                    v3.VSlider(
                                        min=0.0,
                                        max=2.0,
                                        step=0.05,
                                        v_model=("atom_scale_factor", 1.0),
                                        density="compact",
                                        thumb_label=False,
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
                                v3.VCardSubtitle("Gradient bundle basin & surface analysis")

                            v3.VDivider()
                            with v3.VCardText(classes="pt-3 pb-2"):
                                v3.VSelect(
                                    label="Select GBA Atom",
                                    items=("molecule_info.gba_atoms_list",),
                                    v_model=("selected_gba_atom_id",),
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
                                    classes="mb-2",
                                ):
                                    with html.Template(v_slot_prepend_inner=True):
                                        html.Span("F[ρ]", classes="font-italic font-weight-bold text-success mr-1", style="font-size: 0.95rem; line-height: 1;")

                                html.Div("Representation Mode", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                with v3.VBtnToggle(
                                    v_model=("gba_visualization_mode", "basins"),
                                    density="compact",
                                    color="success",
                                    mandatory=True,
                                    classes="mb-1 d-flex justify-center",
                                ):
                                    v3.VBtn("Basin Patches", value="basins", size="small", prepend_icon="mdi-chart-bubble")
                                    v3.VBtn("Atom Contours", value="contours", size="small", prepend_icon="mdi-texture")

                        # Basin Patches Controls
                        with html.Div(v_if="gba_visualization_mode === 'basins'"):
                            with v3.VCard(elevation=1):
                                with v3.VCardItem():
                                    with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"):
                                        with html.Div(classes="d-flex align-center"):
                                            v3.VIcon("mdi-eye-outline", classes="mr-2")
                                            html.Span("Basin Surfaces")
                                        v3.VChip("{{ gba_active_basins_count }} active", size="x-small", color="success")

                                v3.VDivider()
                                with v3.VCardText(classes="pt-2"):
                                    v3.VSwitch(
                                        label="Show Atom Sphere Boundary",
                                        v_model=("gba_show_sphere_boundary",),
                                        density="compact",
                                        color="primary",
                                        hide_details=True,
                                        classes="mb-1",
                                    )
                                    v3.VSwitch(
                                        label="Show Minimum Basin Patches",
                                        v_model=("gba_show_min_basins",),
                                        density="compact",
                                        color="primary",
                                        hide_details=True,
                                        classes="mb-1",
                                    )
                                    v3.VSwitch(
                                        label="Show Maximum Basin Patches",
                                        v_model=("gba_show_max_basins",),
                                        density="compact",
                                        color="secondary",
                                        hide_details=True,
                                    )

                                    with html.Div(v_if="gba_active_basins_list && gba_active_basins_list.length > 0", classes="mt-3"):
                                        v3.VDivider(classes="mb-2")
                                        html.Div("Active Basins (click to view wedge)", classes="text-caption font-weight-bold text-medium-emphasis mb-2")
                                        with html.Div(classes="d-flex flex-wrap ga-1", style="max-height: 180px; overflow-y: auto;"):
                                            with v3.VChip(
                                                v_for="b in gba_active_basins_list",
                                                key="`${b.region_type}_${b.basin_index}`",
                                                click=(ctrl.select_gba_basin_from_list, "[b.basin_index, b.region_type]"),
                                                size="small",
                                                classes="ma-1 font-weight-bold",
                                                variant="elevated",
                                                style=("`background-color: ${b.color_hex}; color: #ffffff; cursor: pointer;`",),
                                            ):
                                                v3.VIcon("mdi-chart-arc", size="x-small", classes="mr-1")
                                                html.Span("Basin {{ b.basin_index }}")

                            with v3.VCard(
                                v_if="selected_gba_basin",
                                elevation=3,
                                classes="mt-3 border-success",
                                color="surface",
                            ):
                                with v3.VCardItem():
                                    with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"):
                                        with html.Div(classes="d-flex align-center"):
                                            v3.VIcon("mdi-chart-arc", classes="mr-2", color="success")
                                            html.Span("Basin #{{ selected_gba_basin.basin_index }}")
                                        v3.VBtn(
                                            icon="mdi-close",
                                            variant="text",
                                            density="compact",
                                            click=ctrl.clear_gba_basin_selection,
                                        )
                                    v3.VCardSubtitle("Atom: {{ selected_gba_basin.atom_type }}{{ selected_gba_basin.atom_number }} | {{ selected_gba_basin.function_name }}")

                                v3.VDivider()
                                with v3.VCardText(classes="pt-2 pb-2"):
                                    with v3.VRow(dense=True, classes="mb-1"):
                                        with v3.VCol(cols=6):
                                            html.Div("Region Type", classes="text-caption text-medium-emphasis")
                                            v3.VChip(
                                                "{{ selected_gba_basin.region_type }}",
                                                size="x-small",
                                                color="primary",
                                                classes="font-weight-bold text-uppercase",
                                            )
                                        with v3.VCol(cols=6):
                                            html.Div("Triangles / Nodes", classes="text-caption text-medium-emphasis")
                                            html.Div("{{ selected_gba_basin.num_triangles }} / {{ selected_gba_basin.num_nodes }}", classes="text-body-2 font-weight-bold")

                                    v3.VDivider(classes="my-2")

                                    html.Div("Integrated Basin Quantities", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                    with v3.VTable(density="compact", classes="elevation-0"):
                                        with html.Thead():
                                            with html.Tr():
                                                html.Th("Condensed Property", classes="text-left text-caption font-weight-bold")
                                                html.Th("Integral Total", classes="text-right text-caption font-weight-bold")
                                        with html.Tbody():
                                            with html.Tr(v_for="item in selected_gba_basin.integrated_totals", key="item.name"):
                                                html.Td("{{ item.name }}", classes="text-caption")
                                                html.Td("{{ item.formatted }}", classes="text-right text-caption font-weight-bold font-italic text-primary")

                                    v3.VDivider(classes="my-3")

                                    html.Div("3D Basin Wedge Controls", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                    with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                        html.Div("Wedge Opacity", classes="text-caption text-medium-emphasis")
                                        html.Div("{{ Math.round((Number(gba_wedge_opacity) || 0) * 100) }}%", classes="text-caption font-weight-bold")

                                    v3.VSlider(
                                        min=0.10,
                                        max=1.0,
                                        step=0.05,
                                        v_model=("gba_wedge_opacity",),
                                        density="compact",
                                        thumb_label=False,
                                        color="primary",
                                        classes="mt-1 mb-1",
                                    )

                                    v3.VSwitch(
                                        label="Show Lateral Feature Edges",
                                        v_model=("gba_show_wedge_edges",),
                                        density="compact",
                                        color="primary",
                                        hide_details=True,
                                    )

                            with v3.VAlert(
                                v_if="!selected_gba_basin",
                                type="info",
                                variant="tonal",
                                density="compact",
                                classes="mt-3 text-caption",
                            ):
                                html.Div("Click any colored basin patch in the 3D view or in the list above to inspect its condensed quantities and generate the wedge.")

                        # Atom Surface Contours Controls
                        with html.Div(v_if="gba_visualization_mode === 'contours'"):
                            with v3.VCard(elevation=1):
                                with v3.VCardItem():
                                    with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold d-flex align-center"):
                                        v3.VIcon("mdi-texture", classes="mr-2")
                                        html.Span("Atom Surface Field")

                                v3.VDivider()
                                with v3.VCardText(classes="pt-2"):
                                    html.Div("Contour Scaling Mode", classes="text-caption font-weight-bold text-medium-emphasis mb-1")
                                    with v3.VBtnToggle(
                                        v_model=("gba_scale_type", "linear"),
                                        density="compact",
                                        color="primary",
                                        mandatory=True,
                                        classes="mb-3 d-flex justify-center",
                                    ):
                                        v3.VBtn("Linear Contours", value="linear", size="small", prepend_icon="mdi-ruler")
                                        v3.VBtn("Logarithmic Contours", value="log", size="small", prepend_icon="mdi-math-log")

                                    with html.Div(classes="d-flex justify-space-between align-center mt-1"):
                                        html.Div("Number of Contour Lines", classes="text-caption font-weight-bold text-medium-emphasis")
                                        html.Div("{{ gba_num_contours }} levels", classes="text-caption font-weight-bold text-primary")

                                    v3.VSlider(
                                        min=3,
                                        max=40,
                                        step=1,
                                        v_model=("gba_num_contours",),
                                        density="compact",
                                        thumb_label=False,
                                        color="primary",
                                        classes="mt-1 mb-2",
                                    )

                                    v3.VDivider(classes="my-2")

                                    with v3.VRow(dense=True, classes="align-center"):
                                        with v3.VCol(cols=6):
                                            v3.VSwitch(
                                                label="Contour Lines",
                                                v_model=("gba_show_contours",),
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                            )
                                        with v3.VCol(cols=6):
                                            v3.VSwitch(
                                                label="Color Flood",
                                                v_model=("gba_show_flood",),
                                                density="compact",
                                                color="primary",
                                                hide_details=True,
                                            )

                                    v3.VSwitch(
                                        label="Show Sphere Wireframe",
                                        v_model=("gba_show_sphere_boundary",),
                                        density="compact",
                                        color="secondary",
                                        hide_details=True,
                                        classes="mt-2",
                                    )

        # --- TOOLBAR ---
        with layout.toolbar:
            with v3.VBtn(
                "Open File",
                prepend_icon="mdi-folder-open",
                click=ctrl.open_file_dialog,
                variant="elevated",
                density="compact",
                color="primary",
                classes="mr-2 font-weight-bold",
            ):
                pass

            v3.VSpacer()
            v3.VBtn(
                "Reset View",
                prepend_icon="mdi-camera-flip-outline",
                click=ctrl.reset_camera,
                variant="tonal",
                density="compact",
                color="primary",
            )

        # --- IN-APP FILE BROWSER MODAL (CROSS-PLATFORM / REMOTE FALLBACK) ---
        with v3.VDialog(v_model=("file_dialog_modal_open", False), max_width="720px"):
            with v3.VCard():
                with v3.VCardItem():
                    with v3.VCardTitle(classes="text-h6 d-flex align-center"):
                        v3.VIcon("mdi-folder-open", color="primary", classes="mr-2")
                        html.Span("Open Dataset (.plt, .vtm, .vti)")
                    v3.VCardSubtitle("Browse files on the server filesystem")

                v3.VDivider()
                with v3.VCardText(classes="pa-3"):
                    # Breadcrumbs Path Bar
                    with html.Div(classes="d-flex align-center mb-2 px-2 py-1 bg-surface-variant rounded"):
                        v3.VIcon("mdi-folder-home", size="small", classes="mr-2 text-primary")
                        with html.Div(classes="d-flex flex-wrap ga-1 flex-grow-1 align-center"):
                            with html.Span(
                                v_for="crumb in browser_breadcrumbs",
                                key="crumb.path",
                                classes="d-inline-flex align-center text-caption",
                            ):
                                with v3.VBtn(
                                    variant="text",
                                    density="compact",
                                    size="small",
                                    click=(ctrl.browser_navigate, "[crumb.path]"),
                                    classes="pa-1 text-none",
                                ):
                                    html.Span("{{ crumb.name }}")
                                html.Span("/", classes="mx-1 text-disabled")

                    # Filter / Search Box
                    v3.VTextField(
                        v_model=("browser_filter_text", ""),
                        density="compact",
                        variant="outlined",
                        placeholder="Filter directory contents...",
                        prepend_inner_icon="mdi-magnify",
                        clearable=True,
                        hide_details=True,
                        classes="mb-2",
                    )

                    # Directory Entries List
                    with v3.VList(
                        density="compact",
                        nav=True,
                        classes="border rounded overflow-y-auto pa-0",
                        style="max-height: 320px; min-height: 200px;",
                    ):
                        # Parent directory row
                        with v3.VListItem(
                            v_if="browser_parent_path",
                            click=(ctrl.browser_navigate, "[browser_parent_path]"),
                            classes="border-b",
                        ):
                            with html.Template(v_slot_prepend=True):
                                v3.VIcon("mdi-folder-arrow-up", color="primary", classes="mr-2")
                            v3.VListItemTitle(".. (Parent Directory)", classes="font-weight-bold")

                        # Entries rows
                        with v3.VListItem(
                            v_for="entry in browser_entries.filter(e => !browser_filter_text || e.name.toLowerCase().includes(browser_filter_text.toLowerCase()))",
                            key="entry.path",
                            click=(ctrl.browser_select_entry, "[entry.path, entry.is_dir]"),
                            active=("browser_selected_path === entry.path",),
                            classes="py-1",
                        ):
                            with html.Template(v_slot_prepend=True):
                                v3.VIcon(
                                    v_if="entry.is_dir",
                                    icon="mdi-folder",
                                    color="amber-darken-2",
                                    classes="mr-2",
                                )
                                v3.VIcon(
                                    v_if="!entry.is_dir && entry.is_supported",
                                    icon="mdi-molecule",
                                    color="success",
                                    classes="mr-2",
                                )
                                v3.VIcon(
                                    v_if="!entry.is_dir && !entry.is_supported",
                                    icon="mdi-file-outline",
                                    color="medium-emphasis",
                                    classes="mr-2",
                                )
                            v3.VListItemTitle(
                                "{{ entry.name }}",
                                classes=("entry.is_supported && !entry.is_dir ? 'font-weight-bold text-success' : (entry.is_dir ? 'font-weight-medium' : 'text-medium-emphasis')",),
                            )
                            with html.Template(v_slot_append=True):
                                with v3.VChip(
                                    v_if="!entry.is_dir && entry.size",
                                    size="x-small",
                                    variant="tonal",
                                    color="secondary",
                                    classes="text-caption",
                                ):
                                    html.Span("{{ entry.size }}")

                    # Selected Path Preview
                    with html.Div(v_if="browser_selected_path", classes="mt-2 text-caption d-flex align-center text-primary font-weight-medium"):
                        v3.VIcon("mdi-check-circle", size="small", color="primary", classes="mr-1")
                        html.Span("Selected: {{ browser_selected_path }}")

                v3.VDivider()
                with v3.VCardActions(classes="pa-3"):
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        variant="text",
                        click=ctrl.browser_cancel,
                    )
                    v3.VBtn(
                        "Load Dataset",
                        color="primary",
                        variant="elevated",
                        disabled=("!browser_selected_path",),
                        click=ctrl.browser_confirm_load,
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

    if vtm_path:
        print(f"\n[Bondalyzer] Starting Trame application for: {vtm_path}")
        print(f"[Bondalyzer] Loaded {len(pipeline_data.get('actors', {}))} rendered blocks.")
        print(f"[Bondalyzer] Formula: {mol_info.get('formula', '-')} ({len(pipeline_data.get('atoms', []))} atoms, {len(pipeline_data.get('critical_points', []))} critical points)")
    else:
        print("\n[Bondalyzer] Starting Trame application in standby mode (No initial dataset specified).")
        print("[Bondalyzer] Use 'Open File' in the toolbar to select a .plt or .vtm dataset.")

    server.start(port=port, open_browser=open_browser)


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
        volume_grid,
        iso_filter,
        iso_actor,
    ) = create_visualization_pipeline(vtm_path)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    render_window.Render()
    interactor.Start()


def main():
    global FORCE_CONVERT

    parser = argparse.ArgumentParser(
        prog="trame_viewer.py",
        description="Bondalyzer Trame Viewer: render QTAIM/GBA datasets converted from Tecplot .plt files.",
    )
    parser.add_argument(
        "vtm_file",
        nargs="?",
        default=None,
        help="Input .vtm or .plt dataset (optional; if omitted, viewer starts ready to Open File).",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="Port for the Trame web server (default: trame-chosen).",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the web server without auto-opening the browser.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Always regenerate cached .vtm/.vts/.vti/.vtr conversions from the source .plt.",
    )
    # Unknown flags are forwarded to trame/wslink via sys.argv below.
    args, unknown = parser.parse_known_args()

    if args.force_convert:
        FORCE_CONVERT = True

    # Keep trame/wslink arg parsing clean: expose only the positional input file if provided.
    clean_argv = [sys.argv[0]]
    if args.vtm_file:
        clean_argv.append(args.vtm_file)
    sys.argv = clean_argv + unknown
    vtm_file = args.vtm_file

    def resolve_plt_for(vtm_path: str) -> Optional[str]:
        """Find the companion .plt used to (re)generate a .vtm file."""
        cand = vtm_path.replace("_1d_zones.vtm", ".plt").replace(".vtm", ".plt")
        if os.path.exists(cand):
            return cand
        base = os.path.splitext(os.path.basename(vtm_path))[0].replace("_1d_zones", "")
        if os.path.exists(f"{base}.plt"):
            return f"{base}.plt"
        for fb in ("ethene4.plt", "ethene2.plt", "ethene.plt"):
            if os.path.exists(fb):
                return fb
        return None

    # If the user supplied a .plt directly (e.g. `trame_viewer.py ethene2.plt`), convert target .vtm name
    if vtm_file:
        try:
            vtm_file = prepare_dataset_file(vtm_file)
        except Exception as e:
            print(f"Error preparing dataset '{vtm_file}': {e}")
            sys.exit(1)

    if TRAME_AVAILABLE:
        run_trame_app(vtm_file, port=args.port, open_browser=not args.server)
    else:
        if not vtm_file or not os.path.exists(vtm_file):
            print("Please provide a valid dataset file for native VTK window.")
            sys.exit(1)
        run_native_vtk_window(vtm_file)


if __name__ == "__main__":
    main()
