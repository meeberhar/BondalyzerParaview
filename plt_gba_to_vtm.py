#!/usr/bin/env python3
"""
plt_gba_to_vtm.py - Extract GBA (Gradient Bundle Analysis) 2D FE triangle zones
(CondensedBasinSphere patches, AtomSphereData, CondensedBasinSurface) from Tecplot
binary (.plt) files and save them as structured VTK MultiBlock datasets.

Each basin patch is saved as a vtkPolyData object with:
- Triangle connectivity (vtkCellArray of vtkTriangle cells)
- Coordinates (X, Y, Z) on Points
- All scalar field variables on PointData
- Clean auxiliary metadata (AtomNumber, FunctionName, RegionType, BasinIndex,
  IntegratedFunctionTotals, SphereRadius, etc.) on FieldData
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
        vtkTriangle,
        vtkMultiBlockDataSet,
        vtkCompositeDataSet,
    )
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter
    from vtkmodules.util import numpy_support
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False

from plt_1d_to_vtm import parse_tecplot_header, parse_zone_data


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
