#!/usr/bin/env python3
"""
Convert 1D ordered zones (Atoms, Inferred Bonds, Critical Points, Gradient/Bond Paths)
from a Tecplot binary (.plt) file into a single VTK MultiBlock (.vtm) dataset
for ParaView, preserving auxiliary metadata and formatting atomic numbers
specifically for ParaView's 'Convert to Molecule' filter.
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
        vtkMultiBlockDataSet,
        vtkCompositeDataSet,
    )
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter
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


def build_1d_zone_polydata(zone_info: Dict[str, Any], var_names: List[str], zone_data: Dict[str, np.ndarray]) -> vtkPolyData:
    """
    Convert a 1D ordered zone to a vtkPolyData object:
    - Atom / Critical Point scatter points -> vtkVertex cells per point.
    - Bond / Gradient paths -> continuous vtkPolyLine cell.
    - Explicit 'atomic_number' / 'Atomic Numbers' PointData arrays for ParaView's Convert to Molecule filter.
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

    # Points
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(coords, deep=1))
    poly.SetPoints(vtk_pts)

    # Determine topology: Lines vs Scatter Vertices
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
        # Continuous PolyLine spanning points 0 -> N-1
        lines_ca = vtkCellArray()
        polyline = vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n_pts)
        for i in range(n_pts):
            polyline.GetPointIds().SetId(i, i)
        lines_ca.InsertNextCell(polyline)
        poly.SetLines(lines_ca)
    else:
        # Discrete Vertex cells per point
        verts_ca = vtkCellArray()
        for i in range(n_pts):
            vertex = vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            verts_ca.InsertNextCell(vertex)
        poly.SetVerts(verts_ca)

    # Attach all scalar field variables to PointData
    coord_names = {x_name, y_name, z_name}
    for var_name, arr in zone_data.items():
        if var_name in coord_names or var_name.startswith("__"):
            continue
        clean_name = var_name.strip()
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=1)
        vtk_arr.SetName(clean_name)
        poly.GetPointData().AddArray(vtk_arr)

    # -------------------------------------------------------------------------
    # Atomic Number PointData Arrays for ParaView 'Convert to Molecule' filter
    # -------------------------------------------------------------------------
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

    # Fallback to parsing element name from zone name (e.g. 'Atoms (C)', 'C1', 'H3')
    if atomic_number == 0:
        for sym, z_val in ELEMENT_SYMBOL_TO_Z.items():
            if f"({sym})" in zname or zname.startswith(f"Atoms ({sym})"):
                atomic_number = z_val
                break

    # Always create explicit atomic number arrays on PointData so merged datasets
    # preserve consistent arrays across all points for ConvertIntoMolecule
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

    # -------------------------------------------------------------------------
    # RGB Color Array for direct ParaView coloring & standard QTAIM schemes
    # -------------------------------------------------------------------------
    cp_type = aux.get("CriticalPointType", "").lower()
    z_type = aux.get("ZoneType", "").lower()

    if "AtomColor" in aux:
        rgb = hex_to_rgb(aux["AtomColor"])
    elif is_line_mesh:
        if "bond path" in zname.lower():
            rgb = (100, 200, 255)  # cyan for gradient bond paths
        else:
            rgb = (180, 180, 180)  # gray for inferred bonds
    elif "bond_cp" in zname.lower() or "bond" in cp_type or "bond" in z_type:
        rgb = (230, 30, 30)        # Standard QTAIM Red for Bond CPs (3, -1)
    elif "ring_cp" in zname.lower() or "ring" in cp_type or "ring" in z_type:
        rgb = (30, 200, 30)        # Standard QTAIM Green for Ring CPs (3, +1)
    elif "cage_cp" in zname.lower() or "cage" in cp_type or "cage" in z_type:
        rgb = (0, 220, 255)        # Standard QTAIM Cyan for Cage CPs (3, +3)
    elif "nuclear_cp" in zname.lower() or "nuclear" in cp_type or "nuclear" in z_type:
        rgb = (255, 128, 0)        # Orange for Nuclear CPs (3, -3)
    elif "critical" in zname.lower() or "cp" in zname.lower():
        rgb = (255, 128, 0)        # General critical points
    else:
        rgb = (200, 200, 200)

    rgb_data = np.tile(np.array(rgb, dtype=np.uint8), (n_pts, 1))
    vtk_rgb = numpy_support.numpy_to_vtk(rgb_data, deep=1)
    vtk_rgb.SetName("RGBColor")
    poly.GetPointData().AddArray(vtk_rgb)

    # -------------------------------------------------------------------------
    # Auxiliary Metadata to FieldData
    # -------------------------------------------------------------------------
    field_data = poly.GetFieldData()

    # Zone Name
    name_arr = vtkStringArray()
    name_arr.SetName("ZoneName")
    name_arr.InsertNextValue(zname)
    field_data.AddArray(name_arr)

    # All aux key-values
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
    them into a single VTK MultiBlock (.vtm) dataset for ParaView.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK Python module is required. Use ParaView's pvpython or install vtk.")

    abs_plt = os.path.abspath(plt_file)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"Tecplot file not found: {abs_plt}")

    print(f"Reading Tecplot file: {abs_plt}")

    with open(abs_plt, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"#!TDV"):
            raise ValueError(f"Not a valid Tecplot binary file (Magic: {magic!r})")

        byte_order_flag = struct.unpack("<I", f.read(4))[0]
        endian = "<" if byte_order_flag == 1 else ">"

        file_info, var_names, zones, dataset_aux, variable_aux = parse_tecplot_header(f, endian)

        # Locate data section (357.0f)
        f.seek(0)
        file_bytes = f.read()
        marker_357_bytes = struct.pack(f"{endian}f", 357.0)
        data_sec_offset = file_bytes.find(marker_357_bytes)
        if data_sec_offset == -1:
            raise RuntimeError("Could not find data section marker (357.0) in file.")

        f.seek(data_sec_offset + 4)

        # Iterate over all zones in the data section
        shared_pool: Dict[str, Any] = {}
        poly_blocks: List[Tuple[str, vtkPolyData, Dict[str, Any]]] = []

        for z in zones:
            zone_data = parse_zone_data(f, endian, z, var_names, shared_pool)

            # Check if zone is a 1D Ordered zone (jmax == 1, kmax == 1)
            is_1d_ordered = (z["zone_type"] == "ORDERED" and z["jmax"] == 1 and z["kmax"] == 1)

            if is_1d_ordered:
                poly = build_1d_zone_polydata(z, var_names, zone_data)
                poly_blocks.append((z["name"], poly, z))

    print(f"Total Zones in PLT:   {len(zones)}")
    print(f"Extracted 1D Zones:   {len(poly_blocks)}")

    # Assemble vtkMultiBlockDataSet
    mb = vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(len(poly_blocks))

    for block_idx, (bname, poly, z_info) in enumerate(poly_blocks):
        mb.SetBlock(block_idx, poly)
        mb.GetMetaData(block_idx).Set(vtkCompositeDataSet.NAME(), bname)
        cell_type_str = "PolyLine" if poly.GetNumberOfLines() > 0 else "Vertices"
        print(f"  Block [{block_idx:02d}]: '{bname}' ({poly.GetNumberOfPoints()} pts, {cell_type_str})")

    # Determine output file path
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(plt_file))[0]
        output_file = f"{base_name}_1d_zones.vtm"

    abs_out = os.path.abspath(output_file)
    print(f"\nWriting VTK MultiBlock file to: {abs_out}")

    writer = vtkXMLMultiBlockDataWriter()
    writer.SetFileName(abs_out)
    writer.SetInputData(mb)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    success = writer.Write()

    if success == 1:
        print(f"Successfully wrote {len(poly_blocks)} 1D zones to {abs_out}")
    else:
        print(f"MultiBlock write returned status: {success}")

    return abs_out


def main():
    parser = argparse.ArgumentParser(
        description="Convert 1D ordered zones from a Tecplot (.plt) file to a VTK MultiBlock (.vtm) file for ParaView."
    )
    parser.add_argument("input_plt", nargs="?", default="ethene.plt", help="Path to input Tecplot .plt file")
    parser.add_argument("-o", "--output", default=None, help="Output .vtm file path")

    args = parser.parse_args()

    try:
        convert_1d_zones_to_vtm(args.input_plt, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
