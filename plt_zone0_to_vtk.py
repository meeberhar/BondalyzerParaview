#!/usr/bin/env python3
"""
Convert the first zone of a Tecplot binary (.plt) file to a VTK file
suitable for importing into ParaView.

Supports modern VTK XML formats (.vtr / .vti / .vts / .vtu) as well as legacy .vtk format.
"""

import os
import sys
import struct
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    try:
        import paraview
    except ImportError:
        pass
    import vtk
    from vtkmodules.vtkCommonCore import vtkPoints, vtkStringArray
    from vtkmodules.vtkCommonDataModel import (
        vtkImageData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
        vtkUnstructuredGrid,
    )
    from vtkmodules.vtkIOXML import (
        vtkXMLImageDataWriter,
        vtkXMLRectilinearGridWriter,
        vtkXMLStructuredGridWriter,
        vtkXMLUnstructuredGridWriter,
    )
    from vtkmodules.vtkIOLegacy import (
        vtkStructuredPointsWriter,
        vtkRectilinearGridWriter as vtkLegacyRectilinearGridWriter,
        vtkStructuredGridWriter as vtkLegacyStructuredGridWriter,
        vtkUnstructuredGridWriter as vtkLegacyUnstructuredGridWriter,
    )
    from vtkmodules.util import numpy_support
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False


def read_int_str(f, endian: str) -> Optional[str]:
    """Read a null-terminated 32-bit integer string from binary file."""
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


def write_variable_aux_field_data(
    dataset,
    var_names: List[str],
    variable_aux: Dict[str, Dict[str, str]],
) -> None:
    """
    Embed per-variable Tecplot AUX metadata (e.g. VariableType) as FieldData on a
    VTK dataset via parallel vtkStringArrays: VariableNames[i] plus Aux_<Key>[i].
    """
    if dataset is None or not var_names:
        return

    field_data = dataset.GetFieldData()

    names_arr = vtkStringArray()
    names_arr.SetName("VariableNames")
    for v in var_names:
        names_arr.InsertNextValue(str(v) if v is not None else "")
    field_data.AddArray(names_arr)

    aux_keys: List[str] = []
    for v in var_names:
        for k in (variable_aux or {}).get(v, {}):
            if k not in aux_keys:
                aux_keys.append(k)

    for key in aux_keys:
        s_arr = vtkStringArray()
        s_arr.SetName(f"Aux_{key}")
        for v in var_names:
            val = (variable_aux or {}).get(v, {}).get(key, "")
            s_arr.InsertNextValue(str(val) if val else "")
        field_data.AddArray(s_arr)


def parse_tecplot_zone0(filepath: str) -> Dict[str, Any]:
    """
    Parse header and extract Zone 0 data and variables from a Tecplot binary (.plt) file.
    """
    with open(filepath, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"#!TDV"):
            raise ValueError(f"Not a valid Tecplot binary file (Magic: {magic!r})")

        magic_str = magic.decode("latin1", errors="replace")
        byte_order_flag = struct.unpack("<I", f.read(4))[0]
        endian = "<" if byte_order_flag == 1 else ">"

        file_type = struct.unpack(f"{endian}I", f.read(4))[0]
        title = read_int_str(f, endian)

        # Number of variables and names
        num_vars = struct.unpack(f"{endian}I", f.read(4))[0]
        var_names = [read_int_str(f, endian) for _ in range(num_vars)]

        # Dataset auxiliary data, Zone 0 header, and variable auxiliary data
        dataset_aux = {}
        variable_aux: Dict[str, Dict[str, str]] = {v: {} for v in var_names if v is not None}
        zone0_info = None

        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            marker = struct.unpack(f"{endian}f", raw)[0]

            if abs(marker - 299.0) < 1e-4:
                # Zone header marker
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
                    pass  # User defined face neighbors if any

                misc = struct.unpack(f"{endian}i", f.read(4))[0]

                # Dimensions
                imax = struct.unpack(f"{endian}i", f.read(4))[0]
                jmax = 1
                kmax = 1
                if zone_type == 0:  # ORDERED
                    jmax = struct.unpack(f"{endian}i", f.read(4))[0]
                    kmax = struct.unpack(f"{endian}i", f.read(4))[0]
                elif zone_type in (1, 2, 3, 4, 5):  # Classic FE types in TDV112
                    jmax = struct.unpack(f"{endian}i", f.read(4))[0]  # num elements
                    # TDV112 classic FE headers contain 3 extra 32-bit integer fields
                    struct.unpack(f"{endian}i", f.read(4))[0]
                    struct.unpack(f"{endian}i", f.read(4))[0]
                    struct.unpack(f"{endian}i", f.read(4))[0]
                else:  # Polyhedral / Polygon FE types
                    jmax = struct.unpack(f"{endian}i", f.read(4))[0]
                    struct.unpack(f"{endian}i", f.read(4))[0]
                    struct.unpack(f"{endian}i", f.read(4))[0]
                    struct.unpack(f"{endian}i", f.read(4))[0]

                # Zone aux data
                zone_aux = {}
                while True:
                    aux_flag = struct.unpack(f"{endian}i", f.read(4))[0]
                    if aux_flag == 1:
                        aux_k = read_int_str(f, endian)
                        aux_t = struct.unpack(f"{endian}i", f.read(4))[0]
                        aux_v = read_int_str(f, endian)
                        if aux_k is not None:
                            zone_aux[aux_k] = aux_v
                    elif aux_flag == 0:
                        break
                    else:
                        break

                if zone0_info is None:
                    zone0_info = {
                        "name": zname,
                        "parent_zone": parent_zone,
                        "strand_id": strand_id,
                        "sol_time": sol_time,
                        "color": color,
                        "zone_type": zone_type,
                        "var_loc": var_loc_arr,
                        "imax": imax,
                        "jmax": jmax,
                        "kmax": kmax,
                        "aux": zone_aux,
                    }
                # Keep walking the header (to collect variable aux markers that
                # appear after zone headers) but only keep Zone 0 metadata.

            elif abs(marker - 799.0) < 1e-4:
                # Dataset aux data
                aux_name = read_int_str(f, endian)
                aux_type = struct.unpack(f"{endian}i", f.read(4))[0]
                aux_val = read_int_str(f, endian)
                if aux_name is not None:
                    dataset_aux[aux_name] = aux_val
            elif abs(marker - 899.0) < 1e-4:
                # Variable aux data (e.g. VariableType='DGBCondensedField')
                var_idx = struct.unpack(f"{endian}i", f.read(4))[0]
                aux_name = read_int_str(f, endian)
                aux_type = struct.unpack(f"{endian}i", f.read(4))[0]
                aux_val = read_int_str(f, endian)
                if 0 <= var_idx < len(var_names):
                    vname = var_names[var_idx]
                    if vname and aux_name:
                        variable_aux[vname][aux_name] = aux_val or ""
            elif abs(marker - 357.0) < 1e-4:
                # Data section marker
                break

        if zone0_info is None:
            raise RuntimeError("Could not find Zone 0 in Tecplot file header.")

        # Find data section marker 357.0f
        f.seek(0)
        file_bytes = f.read()
        marker_357_bytes = struct.pack(f"{endian}f", 357.0)
        data_sec_offset = file_bytes.find(marker_357_bytes)
        if data_sec_offset == -1:
            raise RuntimeError("Could not find data section marker (357.0) in file.")

        # Seek to start of Zone 0 data section
        f.seek(data_sec_offset + 4)
        zmark = struct.unpack(f"{endian}f", f.read(4))[0]
        if abs(zmark - 299.0) > 1e-4:
            raise RuntimeError(f"Unexpected zone data marker: {zmark} (expected 299.0)")

        # Variable data format list: 1=Float32, 2=Float64, 3=Int32, 4=Int16, 5=UInt8, 6=Bit
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

        # Connectivity sharing
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

        # Read array data for Zone 0
        nx = zone0_info["imax"]
        ny = zone0_info["jmax"]
        nz = zone0_info["kmax"]
        n_points = nx * ny * nz

        dtype_map = {
            1: f"{endian}f4",
            2: f"{endian}f8",
            3: f"{endian}i4",
            4: f"{endian}i2",
            5: f"{endian}u1",
        }

        variables_data = {}
        for i in range(num_vars):
            vname = var_names[i]
            if passive_flags[i] != 0:
                # Passive variable (no data stored)
                continue

            dt_str = dtype_map.get(var_formats[i], f"{endian}f4")
            dt = np.dtype(dt_str)
            arr = np.fromfile(f, dtype=dt, count=n_points)
            if len(arr) != n_points:
                raise RuntimeError(
                    f"Incomplete data read for variable '{vname}': expected {n_points}, got {len(arr)}"
                )
            variables_data[vname] = arr

        return {
            "format": magic_str,
            "title": title,
            "var_names": var_names,
            "zone": zone0_info,
            "variables_data": variables_data,
            "min_max": min_max_list,
            "dataset_aux": dataset_aux,
            "variable_aux": variable_aux,
        }


def convert_zone0_to_vtk(
    plt_file: str,
    output_file: Optional[str] = None,
    grid_type: str = "auto",
) -> str:
    """
    Extract Zone 0 from a Tecplot PLT file and write to an equivalent VTK file.
    
    Parameters:
    -----------
    plt_file: str
        Input Tecplot binary file.
    output_file: Optional[str]
        Output VTK file path (.vtr, .vti, .vts, .vtk). Defaults to <input_basename>_zone0.<ext>.
    grid_type: str
        Grid type to write: 'auto', 'rectilinear' (.vtr), 'image' (.vti), 'structured' (.vts), or 'legacy' (.vtk).
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK Python module is required. Ensure VTK / ParaView python is accessible.")

    abs_plt = os.path.abspath(plt_file)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"Tecplot file not found: {abs_plt}")

    print(f"Reading Zone 0 from: {abs_plt}")
    data = parse_tecplot_zone0(abs_plt)

    zone_info = data["zone"]
    var_names = data["var_names"]
    var_data = data["variables_data"]

    nx = zone_info["imax"]
    ny = zone_info["jmax"]
    nz = zone_info["kmax"]

    print(f"Zone Name:      '{zone_info['name']}'")
    print(f"Zone Type:      {zone_info['zone_type']} (ORDERED grid)")
    print(f"Dimensions:     {nx} x {ny} x {nz} ({nx * ny * nz:,} points)")
    print(f"Variables:      {len(var_data)} active variable(s)")

    # Identify Coordinate variables (X, Y, Z)
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

    # In Tecplot ordered grid: I varies fastest, then J, then K (C-order: shape = (nz, ny, nx))
    x_3d = x_arr.reshape((nz, ny, nx))
    y_3d = y_arr.reshape((nz, ny, nx))
    z_3d = z_arr.reshape((nz, ny, nx))

    # Check grid orthogonality / regular spacing
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

    # Check uniform spacing
    is_uniform = False
    if is_orthogonal and nx > 1 and ny > 1 and nz > 1:
        dx = np.diff(x_coords_1d)
        dy = np.diff(y_coords_1d)
        dz = np.diff(z_coords_1d)
        if np.std(dx) < 1e-4 * np.mean(dx) and np.std(dy) < 1e-4 * np.mean(dy) and np.std(dz) < 1e-4 * np.mean(dz):
            is_uniform = True

    # Determine default output format and grid class
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

    # Construct the VTK DataObject
    if out_ext == ".vti" or grid_type == "image":
        print(f"Building vtkImageData (Uniform Grid)...")
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
        print(f"Building vtkRectilinearGrid (Rectilinear Grid)...")
        grid = vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        vtk_x = numpy_support.numpy_to_vtk(x_coords_1d, deep=1)
        vtk_y = numpy_support.numpy_to_vtk(y_coords_1d, deep=1)
        vtk_z = numpy_support.numpy_to_vtk(z_coords_1d, deep=1)
        grid.SetXCoordinates(vtk_x)
        grid.SetYCoordinates(vtk_y)
        grid.SetZCoordinates(vtk_z)

    elif out_ext in (".vts", ".vtk") or grid_type == "structured":
        print(f"Building vtkStructuredGrid (Curvilinear / General Structured Grid)...")
        grid = vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        pts = vtkPoints()
        # Interleave coordinates: (N, 3)
        coords = np.column_stack((x_arr, y_arr, z_arr))
        vtk_pts = numpy_support.numpy_to_vtk(coords, deep=1)
        pts.SetData(vtk_pts)
        grid.SetPoints(pts)

    else:
        # Fallback to RectilinearGrid
        grid = vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetXCoordinates(numpy_support.numpy_to_vtk(x_coords_1d, deep=1))
        grid.SetYCoordinates(numpy_support.numpy_to_vtk(y_coords_1d, deep=1))
        grid.SetZCoordinates(numpy_support.numpy_to_vtk(z_coords_1d, deep=1))

    # Add scalar field arrays to PointData (excluding spatial coordinate variables X, Y, Z)
    coord_names = {x_name, y_name, z_name}
    field_vars = {k: v for k, v in var_data.items() if k not in coord_names}
    print(f"Attaching {len(field_vars)} field variable(s) to VTK PointData (excluding {list(coord_names)})...")

    for var_name, arr in field_vars.items():
        # Clean variable name for ParaView attribute handling
        clean_name = var_name.strip()
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=1)
        vtk_arr.SetName(clean_name)
        grid.GetPointData().AddArray(vtk_arr)

    # Set active scalars (e.g. Electron Density if present, else first non-coord variable)
    scalar_cand = None
    for cand in ["Electron Density", "SCF Density", "Density", "Rho"]:
        for name in field_vars.keys():
            if cand.lower() in name.lower():
                scalar_cand = name
                break
        if scalar_cand:
            break
    if not scalar_cand:
        if field_vars:
            scalar_cand = list(field_vars.keys())[0]

    if scalar_cand:
        grid.GetPointData().SetActiveScalars(scalar_cand)

    # Embed PLT variable AUX metadata (VariableNames / Aux_VariableType / ...) as FieldData
    write_variable_aux_field_data(grid, var_names, data.get("variable_aux", {}))

    # Write output file
    abs_out = os.path.abspath(output_file)
    print(f"Writing VTK output to: {abs_out}")

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
        # Default to XML rectilinear writer
        writer = vtkXMLRectilinearGridWriter()

    writer.SetFileName(abs_out)
    writer.SetInputData(grid)
    success = writer.Write()

    if success == 1:
        print(f"Successfully converted Zone 0 to: {abs_out}")
    else:
        print(f"Write operation returned status: {success}")

    return abs_out


def main():
    parser = argparse.ArgumentParser(
        description="Extract the first zone from a Tecplot binary (.plt) file and write an equivalent VTK file for ParaView."
    )
    parser.add_argument("input_plt", nargs="?", default="ethene.plt", help="Path to input Tecplot .plt file")
    parser.add_argument("-o", "--output", default=None, help="Output VTK file path (.vtr, .vti, .vts, .vtk)")
    parser.add_argument(
        "--grid-type",
        choices=["auto", "rectilinear", "image", "structured", "legacy"],
        default="auto",
        help="Grid representation type to export (default: auto)",
    )

    args = parser.parse_args()

    try:
        convert_zone0_to_vtk(args.input_plt, args.output, args.grid_type)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
