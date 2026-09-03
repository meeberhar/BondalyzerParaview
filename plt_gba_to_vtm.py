#!/usr/bin/env python3
"""
plt_gba_to_vtm.py - Unified Tecplot (.plt) Binary to VTK Conversion Engine.

Provides complete parsing and extraction of:
1. 3D Volume Grids (Zone 0 -> .vti / .vtr / .vts / vtkImageData / vtkRectilinearGrid)
2. 1D Molecular Skeletons (Atoms, Inferred Bonds, Critical Points, Gradient Paths -> .vtm)
3. 2D GBA Surface Meshes (CondensedBasinSphere patches, AtomSphereData, CondensedBasinSurface -> .vtm)

Self-contained module replacing standalone converter scripts for Bondalyzer ParaView / Trame.
"""

import os
import sys
import struct
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import vtk
    from vtkmodules.vtkCommonCore import (
        vtkPoints,
        vtkStringArray,
        vtkTypeInt32Array,
        vtkUnsignedCharArray,
        vtkFloatArray,
        vtkDoubleArray,
    )
    from vtkmodules.vtkCommonDataModel import (
        vtkPolyData,
        vtkCellArray,
        vtkPolyLine,
        vtkVertex,
        vtkTriangle,
        vtkImageData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
        vtkMultiBlockDataSet,
        vtkCompositeDataSet,
    )
    from vtkmodules.vtkIOXML import (
        vtkXMLMultiBlockDataWriter,
        vtkXMLImageDataWriter,
        vtkXMLRectilinearGridWriter,
        vtkXMLStructuredGridWriter,
    )
    from vtkmodules.vtkIOLegacy import (
        vtkStructuredPointsWriter,
        vtkRectilinearGridWriter as vtkLegacyRectilinearGridWriter,
        vtkStructuredGridWriter as vtkLegacyStructuredGridWriter,
    )
    from vtkmodules.util import numpy_support
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False


# Periodic table mapping for symbol -> atomic number
ELEMENT_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert a hex color string like '#909090' to (R, G, B) tuple of ints in [0, 255]."""
    s = hex_str.strip().lstrip("#")
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    elif len(s) == 3:
        return (int(s[0]*2, 16), int(s[1]*2, 16), int(s[2]*2, 16))
    return (255, 255, 255)


def read_int_str(f, endian: str) -> Optional[str]:
    """Read a null-terminated 32-bit integer string from Tecplot binary file."""
    chars = []
    while True:
        raw = f.read(4)
        if len(raw) < 4:
            return None
        c = struct.unpack(f"{endian}I", raw)[0]
        if c == 0:
            break
        try:
            chars.append(chr(c))
        except ValueError:
            chars.append("?")
    return "".join(chars)


def parse_tecplot_header(f, endian: str) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Parse complete header metadata up to data section marker 357.0f.
    Returns (file_info, var_names, zones, dataset_aux, variable_aux).
    """
    file_type = struct.unpack(f"{endian}I", f.read(4))[0]
    title = read_int_str(f, endian)

    num_vars = struct.unpack(f"{endian}I", f.read(4))[0]
    var_names = [read_int_str(f, endian) for _ in range(num_vars)]

    dataset_aux: Dict[str, str] = {}
    variable_aux: Dict[str, Dict[str, str]] = {v: {} for v in var_names if v is not None}
    zones: List[Dict[str, Any]] = []

    zone_type_map = {
        0: "ORDERED",
        1: "FELINESEG",
        2: "FETRIANGLE",
        3: "FEQUADRILATERAL",
        4: "FETETRAHEDRON",
        5: "FEBRICK",
        6: "FEPOLYGON",
        7: "FEPOLYHEDRON",
    }

    while True:
        raw = f.read(4)
        if len(raw) < 4:
            break
        marker = struct.unpack(f"{endian}f", raw)[0]

        # Zone Header Marker (299.0f)
        if abs(marker - 299.0) < 1e-4:
            zname = read_int_str(f, endian)
            parent_zone = struct.unpack(f"{endian}i", f.read(4))[0]
            strand_id = struct.unpack(f"{endian}i", f.read(4))[0]
            sol_time = struct.unpack(f"{endian}d", f.read(8))[0]
            color = struct.unpack(f"{endian}i", f.read(4))[0]
            zone_type = struct.unpack(f"{endian}i", f.read(4))[0]
            var_loc = struct.unpack(f"{endian}i", f.read(4))[0]
            if var_loc != 0:
                var_loc_arr = [struct.unpack(f"{endian}i", f.read(4))[0] for _ in range(num_vars)]
            else:
                var_loc_arr = [0] * num_vars

            raw_user_def = struct.unpack(f"{endian}i", f.read(4))[0]
            if raw_user_def != 0:
                pass

            misc = struct.unpack(f"{endian}i", f.read(4))[0]

            # Dimensions
            if zone_type == 0:  # ORDERED
                imax = struct.unpack(f"{endian}i", f.read(4))[0]
                jmax = struct.unpack(f"{endian}i", f.read(4))[0]
                kmax = struct.unpack(f"{endian}i", f.read(4))[0]
            elif zone_type in (1, 2, 3, 4, 5):  # Classic FE types
                imax = struct.unpack(f"{endian}i", f.read(4))[0]  # num nodes
                jmax = struct.unpack(f"{endian}i", f.read(4))[0]  # num elements
                _extra1 = struct.unpack(f"{endian}i", f.read(4))[0]
                _extra2 = struct.unpack(f"{endian}i", f.read(4))[0]
                _extra3 = struct.unpack(f"{endian}i", f.read(4))[0]
                kmax = 1
            else:
                imax = struct.unpack(f"{endian}i", f.read(4))[0]
                jmax = struct.unpack(f"{endian}i", f.read(4))[0]
                _extra1 = struct.unpack(f"{endian}i", f.read(4))[0]
                _extra2 = struct.unpack(f"{endian}i", f.read(4))[0]
                _extra3 = struct.unpack(f"{endian}i", f.read(4))[0]
                kmax = 1

            # Zone Aux Data
            zone_aux: Dict[str, str] = {}
            while True:
                raw_flag = f.read(4)
                if len(raw_flag) < 4:
                    break
                aux_flag = struct.unpack(f"{endian}i", raw_flag)[0]
                if aux_flag == 1:
                    aux_k = read_int_str(f, endian)
                    aux_t = struct.unpack(f"{endian}i", f.read(4))[0]
                    aux_v = read_int_str(f, endian)
                    if aux_k is not None:
                        zone_aux[aux_k] = aux_v or ""
                elif aux_flag == 0:
                    break
                else:
                    break

            zones.append({
                "index": len(zones),
                "name": zname,
                "zone_type_code": zone_type,
                "zone_type": zone_type_map.get(zone_type, f"TYPE_{zone_type}"),
                "imax": imax,
                "jmax": jmax,
                "kmax": kmax,
                "strand_id": strand_id,
                "parent_zone": parent_zone,
                "color": color,
                "solution_time": sol_time,
                "aux": zone_aux,
            })

        # Dataset Aux Data Marker (799.0f)
        elif abs(marker - 799.0) < 1e-4:
            aux_name = read_int_str(f, endian)
            aux_type = struct.unpack(f"{endian}i", f.read(4))[0]
            aux_val = read_int_str(f, endian)
            if aux_name is not None:
                dataset_aux[aux_name] = aux_val or ""

        # Variable Aux Data Marker (899.0f)
        elif abs(marker - 899.0) < 1e-4:
            var_idx = struct.unpack(f"{endian}i", f.read(4))[0]
            aux_name = read_int_str(f, endian)
            aux_type = struct.unpack(f"{endian}i", f.read(4))[0]
            aux_val = read_int_str(f, endian)
            if 0 <= var_idx < len(var_names):
                vname = var_names[var_idx]
                if vname and aux_name:
                    variable_aux[vname][aux_name] = aux_val or ""

        # Data Section Marker (357.0f)
        elif abs(marker - 357.0) < 1e-4:
            break

    file_info = {
        "file_type": file_type,
        "title": title,
        "num_vars": num_vars,
    }
    return file_info, var_names, zones, dataset_aux, variable_aux


def write_variable_aux_field_data(
    dataset,
    var_names: List[str],
    variable_aux: Dict[str, Dict[str, str]],
) -> None:
    """
    Embed per-variable Tecplot AUX metadata (e.g. VariableType) as FieldData on a
    VTK dataset. Writes parallel vtkStringArrays over the full dataset variable
    order (including coordinates):

      - VariableNames[i]  -> original PLT variable name
      - Aux_<Key>[i]      -> AUX value for that variable ('' when absent)

    Consumers can zip VariableNames with Aux_VariableType to categorize fields
    (e.g. 'Coordinate', 'Scalar3DField'/'Scaler3DField', 'DGBCondensedField').
    """
    if dataset is None or not var_names:
        return

    field_data = dataset.GetFieldData()

    names_arr = vtkStringArray()
    names_arr.SetName("VariableNames")
    for v in var_names:
        names_arr.InsertNextValue(str(v) if v is not None else "")
    field_data.AddArray(names_arr)

    # Union of all aux keys across variables (stable order by first appearance)
    aux_keys: List[str] = []
    for v in var_names:
        for k in (variable_aux or {}).get(v, {}).keys():
            if k not in aux_keys:
                aux_keys.append(k)

    for key in aux_keys:
        s_arr = vtkStringArray()
        s_arr.SetName(f"Aux_{key}")
        for v in var_names:
            val = (variable_aux or {}).get(v, {}).get(key, "")
            s_arr.InsertNextValue(str(val) if val else "")
        field_data.AddArray(s_arr)


def parse_zone_data(f, endian: str, zone_info: Dict[str, Any], var_names: List[str], shared_pool: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Read the data arrays for a single zone from the Tecplot binary stream.
    """
    num_vars = len(var_names)
    raw_mark = f.read(4)
    if len(raw_mark) < 4:
        raise EOFError("Unexpected end of file while looking for zone data marker.")
    zmark = struct.unpack(f"{endian}f", raw_mark)[0]
    if abs(zmark - 299.0) > 1e-4:
        raise ValueError(f"Zone {zone_info['index']} data marker is {zmark}, expected 299.0f")

    # Variable data formats: 1=Float32, 2=Float64, 3=Int32, 4=Int16, 5=UInt8
    var_formats = [struct.unpack(f"{endian}i", f.read(4))[0] for _ in range(num_vars)]

    # Passive variables
    has_passive = struct.unpack(f"{endian}i", f.read(4))[0]
    if has_passive != 0:
        passive_flags = [struct.unpack(f"{endian}i", f.read(4))[0] for _ in range(num_vars)]
    else:
        passive_flags = [0] * num_vars

    # Variable sharing
    has_sharing = struct.unpack(f"{endian}i", f.read(4))[0]
    if has_sharing != 0:
        sharing_flags = [struct.unpack(f"{endian}i", f.read(4))[0] for _ in range(num_vars)]
    else:
        sharing_flags = [0] * num_vars

    # Connectivity sharing (in TDV112: -1 or 0 means not shared; >0 is shared zone index)
    conn_share = struct.unpack(f"{endian}i", f.read(4))[0]

    # Min/Max values per variable
    min_max_list = []
    for i in range(num_vars):
        if sharing_flags[i] == 0:
            min_v = struct.unpack(f"{endian}d", f.read(8))[0]
            max_v = struct.unpack(f"{endian}d", f.read(8))[0]
            min_max_list.append((min_v, max_v))
        else:
            min_max_list.append(None)

    # Number of points in this zone
    if zone_info["zone_type"] == "ORDERED":
        n_points = zone_info["imax"] * zone_info["jmax"] * zone_info["kmax"]
    else:
        n_points = zone_info["imax"]  # num nodes

    dtype_map = {
        1: (f"{endian}f4", 4),
        2: (f"{endian}f8", 8),
        3: (f"{endian}i4", 4),
        4: (f"{endian}i2", 2),
        5: (f"{endian}u1", 1),
    }

    zone_arrays = {}
    for i in range(num_vars):
        vname = var_names[i]
        if passive_flags[i] != 0:
            continue

        if sharing_flags[i] != 0:
            # Shared from another zone
            src_zone_idx = sharing_flags[i] - 1  # 1-based index in Tecplot
            if (src_zone_idx, vname) in shared_pool:
                zone_arrays[vname] = shared_pool[(src_zone_idx, vname)]
            continue

        dt_str, itemsize = dtype_map.get(var_formats[i], (f"{endian}f4", 4))
        dt = np.dtype(dt_str)
        arr = np.fromfile(f, dtype=dt, count=n_points)
        if len(arr) != n_points:
            raise EOFError(f"Incomplete data read for zone {zone_info['index']}, var '{vname}'")

        zone_arrays[vname] = arr
        # Save to sharing pool
        shared_pool[(zone_info["index"], vname)] = arr

    # If classic FE zone, handle connectivity array (0-based in VTK, 1-based in Tecplot)
    nodes_per_elem_map = {
        "FELINESEG": 2,
        "FETRIANGLE": 3,
        "FEQUADRILATERAL": 4,
        "FETETRAHEDRON": 4,
        "FEBRICK": 8,
    }
    if zone_info["zone_type"] in nodes_per_elem_map:
        num_elems = zone_info["jmax"]
        nodes_per_elem = nodes_per_elem_map[zone_info["zone_type"]]
        if conn_share <= 0 and nodes_per_elem > 0:
            # 32-bit integer node connectivity array
            conn_arr = np.fromfile(f, dtype=f"{endian}i4", count=num_elems * nodes_per_elem)
            zone_arrays["__connectivity__"] = conn_arr

    return zone_arrays


# =============================================================================
# 1D SKELETON / MOLECULE EXTRACTION (Atoms, Bonds, CPs, Paths)
# =============================================================================

def build_1d_zone_polydata(zone_info: Dict[str, Any], var_names: List[str], zone_data: Dict[str, np.ndarray]) -> vtkPolyData:
    """
    Convert a 1D ordered zone to a vtkPolyData object:
    - Atom / Critical Point scatter points -> vtkVertex cells per point.
    - Bond / Gradient paths -> continuous vtkPolyLine cell.
    - Explicit 'atomic_number' PointData arrays.
    - RGBColor PointData array from AtomColor hex code.
    - Zone auxiliary metadata on FieldData.
    """
    poly = vtkPolyData()

    # Identify Coordinate variables (X, Y, Z)
    var_names_lower = [v.strip().lower() for v in var_names]
    x_idx = var_names_lower.index("x") if "x" in var_names_lower else 0
    y_idx = var_names_lower.index("y") if "y" in var_names_lower else 1
    z_idx = var_names_lower.index("z") if "z" in var_names_lower else 2

    x_name = var_names[x_idx]
    y_name = var_names[y_idx]
    z_name = var_names[z_idx]

    x_arr = zone_data[x_name].astype(np.float64)
    y_arr = zone_data[y_name].astype(np.float64)
    z_arr = zone_data[z_name].astype(np.float64)

    n_pts = len(x_arr)
    coords = np.column_stack((x_arr, y_arr, z_arr))

    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(coords, deep=1))
    poly.SetPoints(vtk_pts)

    aux = zone_info.get("aux", {})
    ztype_aux = aux.get("ZoneType", "")
    zstyle_aux = aux.get("ZoneStyle", "")
    path_type = aux.get("PathType", "")
    zname = zone_info.get("name", "")

    is_line_mesh = False
    if "bond" in zname.lower() or "path" in zname.lower() or "inferredbonds" in ztype_aux.lower() or "gradientpath" in ztype_aux.lower() or path_type != "":
        if n_pts > 1 and zstyle_aux.lower() != "scatter":
            is_line_mesh = True

    if is_line_mesh and n_pts > 1:
        lines_ca = vtkCellArray()
        polyline = vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n_pts)
        for i in range(n_pts):
            polyline.GetPointIds().SetId(i, i)
        lines_ca.InsertNextCell(polyline)
        poly.SetLines(lines_ca)
    else:
        verts_ca = vtkCellArray()
        for i in range(n_pts):
            vertex = vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            verts_ca.InsertNextCell(vertex)
        poly.SetVerts(verts_ca)

    coord_names = {x_name, y_name, z_name}
    for var_name, arr in zone_data.items():
        if var_name in coord_names or var_name.startswith("__"):
            continue
        clean_name = var_name.strip()
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=1)
        vtk_arr.SetName(clean_name)
        poly.GetPointData().AddArray(vtk_arr)

    atomic_number = 0
    if "AtomElementNumber" in aux:
        try:
            atomic_number = int(aux["AtomElementNumber"])
        except ValueError:
            pass

    if atomic_number == 0 and "AtomType" in aux:
        sym = aux["AtomType"].strip()
        if sym in ELEMENT_SYMBOL_TO_Z:
            atomic_number = ELEMENT_SYMBOL_TO_Z[sym]

    if atomic_number == 0:
        for sym, z_val in ELEMENT_SYMBOL_TO_Z.items():
            if f"({sym})" in zname or zname.startswith(f"Atoms ({sym})"):
                atomic_number = z_val
                break

    z_array = np.full(n_pts, atomic_number, dtype=np.int32)
    vtk_z1 = numpy_support.numpy_to_vtk(z_array, deep=1)
    vtk_z1.SetName("atomic_number")
    poly.GetPointData().AddArray(vtk_z1)

    vtk_z2 = numpy_support.numpy_to_vtk(z_array, deep=1)
    vtk_z2.SetName("Atomic Numbers")
    poly.GetPointData().AddArray(vtk_z2)

    vtk_z3 = numpy_support.numpy_to_vtk(z_array, deep=1)
    vtk_z3.SetName("AtomicNumber")
    poly.GetPointData().AddArray(vtk_z3)

    cp_type = aux.get("CriticalPointType", "").lower()
    z_type = aux.get("ZoneType", "").lower()

    if "AtomColor" in aux:
        rgb = hex_to_rgb(aux["AtomColor"])
    elif is_line_mesh:
        if "bond path" in zname.lower():
            rgb = (100, 200, 255)
        else:
            rgb = (180, 180, 180)
    elif "bond_cp" in zname.lower() or "bond" in cp_type or "bond" in z_type:
        rgb = (230, 30, 30)
    elif "ring_cp" in zname.lower() or "ring" in cp_type or "ring" in z_type:
        rgb = (30, 200, 30)
    elif "cage_cp" in zname.lower() or "cage" in cp_type or "cage" in z_type:
        rgb = (0, 220, 255)
    elif "nuclear_cp" in zname.lower() or "nuclear" in cp_type or "nuclear" in z_type:
        rgb = (255, 128, 0)
    elif "critical" in zname.lower() or "cp" in zname.lower():
        rgb = (255, 128, 0)
    else:
        rgb = (200, 200, 200)

    rgb_data = np.tile(np.array(rgb, dtype=np.uint8), (n_pts, 1))
    vtk_rgb = numpy_support.numpy_to_vtk(rgb_data, deep=1)
    vtk_rgb.SetName("RGBColor")
    poly.GetPointData().AddArray(vtk_rgb)

    field_data = poly.GetFieldData()
    name_arr = vtkStringArray()
    name_arr.SetName("ZoneName")
    name_arr.InsertNextValue(zname)
    field_data.AddArray(name_arr)

    for k, v in aux.items():
        s_arr = vtkStringArray()
        s_arr.SetName(f"Aux_{k}")
        s_arr.InsertNextValue(str(v))
        field_data.AddArray(s_arr)

    return poly


def convert_1d_zones_to_vtm(
    plt_file: str,
    output_file: Optional[str] = None,
    include_all_1d: bool = True,
) -> str:
    """
    Extract all 1D ordered zones from a Tecplot binary (.plt) file and save
    them into a single VTK MultiBlock (.vtm) dataset.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK Python module is required.")

    abs_plt = os.path.abspath(plt_file)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"Tecplot file not found: {abs_plt}")

    with open(abs_plt, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"#!TDV"):
            raise ValueError(f"Not a valid Tecplot binary file (Magic: {magic!r})")

        byte_order_flag = struct.unpack("<I", f.read(4))[0]
        endian = "<" if byte_order_flag == 1 else ">"

        file_info, var_names, zones, dataset_aux, variable_aux = parse_tecplot_header(f, endian)

        f.seek(0)
        file_bytes = f.read()
        marker_357_bytes = struct.pack(f"{endian}f", 357.0)
        data_sec_offset = file_bytes.find(marker_357_bytes)
        if data_sec_offset == -1:
            raise RuntimeError("Could not find data section marker (357.0) in file.")

        f.seek(data_sec_offset + 4)

        shared_pool: Dict[str, Any] = {}
        poly_blocks: List[Tuple[str, vtkPolyData, Dict[str, Any]]] = []

        for z in zones:
            zone_data = parse_zone_data(f, endian, z, var_names, shared_pool)
            is_1d_ordered = (z["zone_type"] == "ORDERED" and z["jmax"] == 1 and z["kmax"] == 1)
            if is_1d_ordered:
                poly = build_1d_zone_polydata(z, var_names, zone_data)
                poly_blocks.append((z["name"], poly, z))

    mb = vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(len(poly_blocks))

    for block_idx, (bname, poly, z_info) in enumerate(poly_blocks):
        mb.SetBlock(block_idx, poly)
        mb.GetMetaData(block_idx).Set(vtkCompositeDataSet.NAME(), bname)

    # Embed PLT variable AUX metadata (VariableNames / Aux_VariableType / ...) as FieldData
    write_variable_aux_field_data(mb, var_names, variable_aux)

    if output_file is None:
        base_name = os.path.splitext(os.path.basename(plt_file))[0]
        output_file = f"{base_name}_1d_zones.vtm"

    abs_out = os.path.abspath(output_file)
    writer = vtkXMLMultiBlockDataWriter()
    writer.SetFileName(abs_out)
    writer.SetInputData(mb)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.Write()

    return abs_out


# =============================================================================
# ZONE 0 (3D VOLUME) EXTRACTION (.vti / .vtr / .vts)
# =============================================================================

def convert_zone0_to_vtk(
    plt_file: str,
    output_file: Optional[str] = None,
    grid_type: str = "auto",
) -> str:
    """
    Extract Zone 0 from a Tecplot PLT file and write to an equivalent VTK volume file.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK Python module is required.")

    abs_plt = os.path.abspath(plt_file)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"Tecplot file not found: {abs_plt}")

    with open(abs_plt, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"#!TDV"):
            raise ValueError(f"Not a valid Tecplot binary file (Magic: {magic!r})")

        byte_order_flag = struct.unpack("<I", f.read(4))[0]
        endian = "<" if byte_order_flag == 1 else ">"

        file_info, var_names, zones, dataset_aux, variable_aux = parse_tecplot_header(f, endian)
        if not zones:
            raise RuntimeError("No zones found in Tecplot file header.")
        zone0_info = zones[0]

        f.seek(0)
        file_bytes = f.read()
        marker_357_bytes = struct.pack(f"{endian}f", 357.0)
        data_sec_offset = file_bytes.find(marker_357_bytes)
        if data_sec_offset == -1:
            raise RuntimeError("Could not find data section marker (357.0) in file.")

        f.seek(data_sec_offset + 4)
        shared_pool: Dict[str, Any] = {}
        var_data = parse_zone_data(f, endian, zone0_info, var_names, shared_pool)

    nx = zone0_info["imax"]
    ny = zone0_info["jmax"]
    nz = zone0_info["kmax"]

    var_names_lower = [v.strip().lower() for v in var_names]
    x_idx = var_names_lower.index("x") if "x" in var_names_lower else 0
    y_idx = var_names_lower.index("y") if "y" in var_names_lower else 1
    z_idx = var_names_lower.index("z") if "z" in var_names_lower else 2

    x_name = var_names[x_idx]
    y_name = var_names[y_idx]
    z_name = var_names[z_idx]

    x_arr = var_data[x_name]
    y_arr = var_data[y_name]
    z_arr = var_data[z_name]

    x_3d = x_arr.reshape((nz, ny, nx))
    y_3d = y_arr.reshape((nz, ny, nx))
    z_3d = z_arr.reshape((nz, ny, nx))

    x_coords_1d = x_3d[0, 0, :].astype(np.float64)
    y_coords_1d = y_3d[0, :, 0].astype(np.float64)
    z_coords_1d = z_3d[:, 0, 0].astype(np.float64)

    is_orthogonal = True
    if np.max(np.abs(x_3d - x_coords_1d[None, None, :])) > 1e-5:
        is_orthogonal = False
    if np.max(np.abs(y_3d - y_coords_1d[None, :, None])) > 1e-5:
        is_orthogonal = False
    if np.max(np.abs(z_3d - z_coords_1d[:, None, None])) > 1e-5:
        is_orthogonal = False

    is_uniform = False
    if is_orthogonal and nx > 1 and ny > 1 and nz > 1:
        dx = np.diff(x_coords_1d)
        dy = np.diff(y_coords_1d)
        dz = np.diff(z_coords_1d)
        if np.std(dx) < 1e-4 * np.mean(dx) and np.std(dy) < 1e-4 * np.mean(dy) and np.std(dz) < 1e-4 * np.mean(dz):
            is_uniform = True

    base_name = os.path.splitext(os.path.basename(plt_file))[0]
    if output_file is None:
        if grid_type == "image" or (grid_type == "auto" and is_uniform):
            ext = ".vti"
        elif grid_type == "rectilinear" or (grid_type == "auto" and is_orthogonal):
            ext = ".vtr"
        elif grid_type == "legacy":
            ext = ".vtk"
        else:
            ext = ".vts"
        output_file = f"{base_name}_zone0{ext}"

    out_ext = os.path.splitext(output_file)[1].lower()

    if out_ext == ".vti" or grid_type == "image":
        grid = vtkImageData()
        grid.SetDimensions(nx, ny, nz)
        origin = (float(x_coords_1d[0]), float(y_coords_1d[0]), float(z_coords_1d[0]))
        spacing = (
            float((x_coords_1d[-1] - x_coords_1d[0]) / (nx - 1)) if nx > 1 else 1.0,
            float((y_coords_1d[-1] - y_coords_1d[0]) / (ny - 1)) if ny > 1 else 1.0,
            float((z_coords_1d[-1] - z_coords_1d[0]) / (nz - 1)) if nz > 1 else 1.0,
        )
        grid.SetOrigin(*origin)
        grid.SetSpacing(*spacing)

    elif out_ext == ".vtr" or (grid_type in ("auto", "rectilinear") and is_orthogonal and out_ext != ".vts"):
        grid = vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetXCoordinates(numpy_support.numpy_to_vtk(x_coords_1d, deep=1))
        grid.SetYCoordinates(numpy_support.numpy_to_vtk(y_coords_1d, deep=1))
        grid.SetZCoordinates(numpy_support.numpy_to_vtk(z_coords_1d, deep=1))

    elif out_ext in (".vts", ".vtk") or grid_type == "structured":
        grid = vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        pts = vtkPoints()
        coords = np.column_stack((x_arr, y_arr, z_arr))
        pts.SetData(numpy_support.numpy_to_vtk(coords, deep=1))
        grid.SetPoints(pts)

    else:
        grid = vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetXCoordinates(numpy_support.numpy_to_vtk(x_coords_1d, deep=1))
        grid.SetYCoordinates(numpy_support.numpy_to_vtk(y_coords_1d, deep=1))
        grid.SetZCoordinates(numpy_support.numpy_to_vtk(z_coords_1d, deep=1))

    coord_names = {x_name, y_name, z_name}
    field_vars = {k: v for k, v in var_data.items() if k not in coord_names}

    for var_name, arr in field_vars.items():
        clean_name = var_name.strip()
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=1)
        vtk_arr.SetName(clean_name)
        grid.GetPointData().AddArray(vtk_arr)

    scalar_cand = None
    for cand in ["Electron Density", "SCF Density", "Density", "Rho"]:
        for name in field_vars.keys():
            if cand.lower() in name.lower():
                scalar_cand = name
                break
        if scalar_cand:
            break
    if not scalar_cand and field_vars:
        scalar_cand = list(field_vars.keys())[0]

    if scalar_cand:
        grid.GetPointData().SetActiveScalars(scalar_cand)

    # Embed PLT variable AUX metadata (VariableNames / Aux_VariableType / ...) as FieldData
    write_variable_aux_field_data(grid, var_names, variable_aux)

    abs_out = os.path.abspath(output_file)
    if out_ext == ".vti":
        writer = vtkXMLImageDataWriter()
    elif out_ext == ".vtr":
        writer = vtkXMLRectilinearGridWriter()
    elif out_ext == ".vts":
        writer = vtkXMLStructuredGridWriter()
    elif out_ext == ".vtk":
        if isinstance(grid, vtkImageData):
            writer = vtkStructuredPointsWriter()
        elif isinstance(grid, vtkRectilinearGrid):
            writer = vtkLegacyRectilinearGridWriter()
        else:
            writer = vtkLegacyStructuredGridWriter()
    else:
        writer = vtkXMLRectilinearGridWriter()

    writer.SetFileName(abs_out)
    writer.SetInputData(grid)
    writer.Write()

    return abs_out


# =============================================================================
# 2D GBA SURFACE / BASIN PATCH EXTRACTION
# =============================================================================

def build_fe_triangle_polydata(
    zone_info: Dict[str, Any],
    var_names: List[str],
    zone_data: Dict[str, np.ndarray],
    dataset_aux: Optional[Dict[str, str]] = None,
) -> vtkPolyData:
    """
    Convert a 2D FETRIANGLE zone to a vtkPolyData object:
    - Points: X, Y, Z
    - Polys: Triangle cells from __connectivity__
    - PointData: Scalar fields (Electron Density, curvatures, etc.)
    - FieldData: Zone metadata and integrated property totals.
    """
    poly = vtkPolyData()

    # Identify Coordinate variables (X, Y, Z)
    var_names_lower = [v.strip().lower() for v in var_names]
    x_idx = var_names_lower.index("x") if "x" in var_names_lower else 0
    y_idx = var_names_lower.index("y") if "y" in var_names_lower else 1
    z_idx = var_names_lower.index("z") if "z" in var_names_lower else 2

    x_name = var_names[x_idx]
    y_name = var_names[y_idx]
    z_name = var_names[z_idx]

    x_arr = zone_data[x_name].astype(np.float64)
    y_arr = zone_data[y_name].astype(np.float64)
    z_arr = zone_data[z_name].astype(np.float64)

    n_pts = len(x_arr)
    coords = np.column_stack((x_arr, y_arr, z_arr))

    # Points
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(coords, deep=1))
    poly.SetPoints(vtk_pts)

    # Polys / Triangle connectivity (1-based in Tecplot, convert to 0-based in VTK)
    conn = zone_data.get("__connectivity__")
    if conn is not None and len(conn) > 0:
        # Reshape to (num_triangles, 3)
        tri_conn = conn.reshape(-1, 3)
        # Check if 1-based indexing (min value >= 1)
        if tri_conn.min() >= 1:
            tri_conn = tri_conn - 1

        n_tri = len(tri_conn)
        # Create VTK cell array efficiently
        # Format: [3, id0, id1, id2, 3, id0, id1, id2, ...]
        cell_array_np = np.empty((n_tri, 4), dtype=np.int64)
        cell_array_np[:, 0] = 3
        cell_array_np[:, 1:] = tri_conn.astype(np.int64)

        cells = vtkCellArray()
        vtk_conn = numpy_support.numpy_to_vtkIdTypeArray(cell_array_np.ravel(), deep=1)
        cells.SetCells(n_tri, vtk_conn)
        poly.SetPolys(cells)

    # Attach all scalar field variables to PointData
    coord_names = {x_name, y_name, z_name}
    for var_name, arr in zone_data.items():
        if var_name in coord_names or var_name.startswith("__"):
            continue
        clean_name = var_name.strip()
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=1)
        vtk_arr.SetName(clean_name)
        poly.GetPointData().AddArray(vtk_arr)

    # Auxiliary Metadata to FieldData
    aux = zone_info.get("aux", {})
    zname = zone_info.get("name", "")
    field_data = poly.GetFieldData()

    name_arr = vtkStringArray()
    name_arr.SetName("ZoneName")
    name_arr.InsertNextValue(zname)
    field_data.AddArray(name_arr)

    # Zone index
    zidx_arr = vtkTypeInt32Array()
    zidx_arr.SetName("ZoneIndex")
    zidx_arr.InsertNextValue(int(zone_info.get("index", 0)))
    field_data.AddArray(zidx_arr)

    # Attach all zone auxiliary key-values
    for k, v in aux.items():
        s_arr = vtkStringArray()
        s_arr.SetName(f"Aux_{k}")
        s_arr.InsertNextValue(str(v))
        field_data.AddArray(s_arr)

    # Parse numerical properties for quick access
    if "BasinIndex" in aux:
        try:
            b_arr = vtkTypeInt32Array()
            b_arr.SetName("BasinIndex")
            b_arr.InsertNextValue(int(aux["BasinIndex"]))
            field_data.AddArray(b_arr)
        except ValueError:
            pass

    if "AtomNumber" in aux:
        try:
            a_arr = vtkTypeInt32Array()
            a_arr.SetName("AtomNumber")
            a_arr.InsertNextValue(int(aux["AtomNumber"]))
            field_data.AddArray(a_arr)
        except ValueError:
            pass

    if "FunctionIndex" in aux:
        try:
            f_arr = vtkTypeInt32Array()
            f_arr.SetName("FunctionIndex")
            f_arr.InsertNextValue(int(aux["FunctionIndex"]))
            field_data.AddArray(f_arr)
        except ValueError:
            pass

    if "SphereRadius" in aux:
        try:
            r_arr = vtkDoubleArray()
            r_arr.SetName("SphereRadius")
            r_arr.InsertNextValue(float(aux["SphereRadius"]))
            field_data.AddArray(r_arr)
        except ValueError:
            pass

    # Parse Integrated Function Totals if present
    if "IntegratedFunctionTotals" in aux and "IntegratedFunctionNames" in aux:
        try:
            names = [n.strip() for n in aux["IntegratedFunctionNames"].split(",")]
            vals = [float(v.strip()) for v in aux["IntegratedFunctionTotals"].split(",")]
            for n, v in zip(names, vals):
                val_arr = vtkDoubleArray()
                val_arr.SetName(f"Total_{n}")
                val_arr.InsertNextValue(v)
                field_data.AddArray(val_arr)
        except Exception:
            pass

    return poly


def extract_gba_zones_from_plt(
    plt_file: str,
    output_vtm: Optional[str] = None,
    include_sphere_patches: bool = True,
    include_surfaces: bool = False,
    include_atom_spheres: bool = True,
) -> Tuple[vtkMultiBlockDataSet, List[Dict[str, Any]]]:
    """
    Parse a Tecplot .plt binary file and extract GBA surface/patch zones.
    Returns the populated vtkMultiBlockDataSet and block metadata list.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK Python module is required.")

    abs_plt = os.path.abspath(plt_file)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"Tecplot file not found: {abs_plt}")

    with open(abs_plt, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"#!TDV"):
            raise ValueError(f"Not a valid Tecplot binary file (Magic: {magic!r})")

        byte_order_flag = struct.unpack("<I", f.read(4))[0]
        endian = "<" if byte_order_flag == 1 else ">"

        file_info, var_names, zones, dataset_aux, variable_aux = parse_tecplot_header(f, endian)

        # Find 357.0f marker
        f.seek(0)
        file_bytes = f.read()
        marker_357_bytes = struct.pack(f"{endian}f", 357.0)
        data_sec_offset = file_bytes.find(marker_357_bytes)
        if data_sec_offset == -1:
            raise RuntimeError("Could not find data section marker (357.0) in file.")

        f.seek(data_sec_offset + 4)

        shared_pool: Dict[str, Any] = {}
        extracted_blocks: List[Tuple[str, vtkPolyData, Dict[str, Any]]] = []

        for z in zones:
            zone_data = parse_zone_data(f, endian, z, var_names, shared_pool)

            aux = z.get("aux", {})
            ztype = aux.get("ZoneType", "")

            should_extract = False
            if include_sphere_patches and ztype == "CondensedBasinSphere":
                should_extract = True
            elif include_atom_spheres and ztype == "AtomSphereData":
                should_extract = True
            elif include_surfaces and ztype == "CondensedBasinSurface":
                should_extract = True

            if should_extract:
                poly = build_fe_triangle_polydata(z, var_names, zone_data, dataset_aux)
                extracted_blocks.append((z["name"], poly, z))

    # Build vtkMultiBlockDataSet
    mb = vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(len(extracted_blocks))

    metadata_list = []
    for idx, (bname, poly, z_info) in enumerate(extracted_blocks):
        mb.SetBlock(idx, poly)
        mb.GetMetaData(idx).Set(vtkCompositeDataSet.NAME(), bname)
        meta_entry = {
            "block_index": idx,
            "zone_index": z_info.get("index"),
            "name": bname,
            "zone_type": z_info.get("aux", {}).get("ZoneType", ""),
            "atom_number": z_info.get("aux", {}).get("AtomNumber", ""),
            "atom_type": z_info.get("aux", {}).get("AtomType", ""),
            "function_name": z_info.get("aux", {}).get("FunctionName", ""),
            "function_index": z_info.get("aux", {}).get("FunctionIndex", ""),
            "region_type": z_info.get("aux", {}).get("RegionType", ""),
            "basin_index": z_info.get("aux", {}).get("BasinIndex", ""),
            "num_nodes": poly.GetNumberOfPoints(),
            "num_triangles": poly.GetNumberOfPolys(),
        }
        metadata_list.append(meta_entry)

    # Embed PLT variable AUX metadata (VariableNames / Aux_VariableType / ...) as FieldData
    write_variable_aux_field_data(mb, var_names, variable_aux)

    if output_vtm:
        abs_out = os.path.abspath(output_vtm)
        writer = vtkXMLMultiBlockDataWriter()
        writer.SetFileName(abs_out)
        writer.SetInputData(mb)
        writer.SetDataModeToAppended()
        writer.EncodeAppendedDataOff()
        writer.Write()

    return mb, metadata_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GBA basin patches and sphere data from Tecplot binary (.plt) to VTK MultiBlock (.vtm).")
    parser.add_argument("plt_file", help="Input Tecplot binary file (e.g. ethene.plt)")
    parser.add_argument("-o", "--output", help="Output .vtm filename (default: <base>_gba_patches.vtm)")
    parser.add_argument("--include-surfaces", action="store_true", help="Also extract CondensedBasinSurface zones")
    args = parser.parse_args()

    out_file = args.output
    if out_file is None:
        base = os.path.splitext(os.path.basename(args.plt_file))[0]
        out_file = f"{base}_gba_patches.vtm"

    mb, meta = extract_gba_zones_from_plt(
        args.plt_file,
        output_vtm=out_file,
        include_sphere_patches=True,
        include_surfaces=args.include_surfaces,
        include_atom_spheres=True,
    )
    print(f"Extracted {len(meta)} GBA blocks to {out_file}")
