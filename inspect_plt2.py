#!/usr/bin/env python3
"""
Inspect and extract metadata (Dataset, Variable, and Zone auxiliary data, 
zones, and variables) from Tecplot binary (.plt) files.
"""

import os
import sys
import struct
from typing import List, Dict, Any, Optional


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


def parse_tecplot_full_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parse the complete header of a Tecplot binary (.plt) file,
    extracting dataset info, variables, dataset aux data,
    variable aux data, and all zones with their aux data and dimensions.
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

        # Number of variables and variable names
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

            # -------------------------------------------------------------
            # Zone Header Marker (299.0f)
            # -------------------------------------------------------------
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
                elif zone_type in (1, 2, 3, 4, 5):  # Classic FE types in TDV112
                    imax = struct.unpack(f"{endian}i", f.read(4))[0]  # num nodes
                    jmax = struct.unpack(f"{endian}i", f.read(4))[0]  # num elements
                    # In TDV112 format, classic FE headers contain 3 extra 32-bit integer fields (e.g. face/cell metadata)
                    _extra1 = struct.unpack(f"{endian}i", f.read(4))[0]
                    _extra2 = struct.unpack(f"{endian}i", f.read(4))[0]
                    _extra3 = struct.unpack(f"{endian}i", f.read(4))[0]
                    kmax = 1
                else:  # Polyhedral / Polygon FE types
                    imax = struct.unpack(f"{endian}i", f.read(4))[0]
                    jmax = struct.unpack(f"{endian}i", f.read(4))[0]
                    _extra1 = struct.unpack(f"{endian}i", f.read(4))[0]
                    _extra2 = struct.unpack(f"{endian}i", f.read(4))[0]
                    _extra3 = struct.unpack(f"{endian}i", f.read(4))[0]
                    kmax = 1

                # Zone Auxiliary Data loop
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
                    "zone_type": zone_type_map.get(zone_type, f"TYPE_{zone_type}"),
                    "imax": imax,
                    "jmax": jmax,
                    "kmax": kmax,
                    "strand_id": strand_id,
                    "solution_time": sol_time,
                    "aux": zone_aux,
                })

            # -------------------------------------------------------------
            # Dataset Auxiliary Data Marker (799.0f)
            # -------------------------------------------------------------
            elif abs(marker - 799.0) < 1e-4:
                aux_name = read_int_str(f, endian)
                aux_type = struct.unpack(f"{endian}i", f.read(4))[0]
                aux_val = read_int_str(f, endian)
                if aux_name is not None:
                    dataset_aux[aux_name] = aux_val or ""

            # -------------------------------------------------------------
            # Variable Auxiliary Data Marker (899.0f)
            # -------------------------------------------------------------
            elif abs(marker - 899.0) < 1e-4:
                var_idx = struct.unpack(f"{endian}i", f.read(4))[0]
                aux_name = read_int_str(f, endian)
                aux_type = struct.unpack(f"{endian}i", f.read(4))[0]
                aux_val = read_int_str(f, endian)
                if 0 <= var_idx < len(var_names):
                    vname = var_names[var_idx]
                    if vname and aux_name:
                        variable_aux[vname][aux_name] = aux_val or ""

            # -------------------------------------------------------------
            # Data Section Marker (357.0f)
            # -------------------------------------------------------------
            elif abs(marker - 357.0) < 1e-4:
                break

    return {
        "format": magic_str,
        "title": title,
        "variables": var_names,
        "dataset_aux": dataset_aux,
        "variable_aux": variable_aux,
        "zones": zones,
    }


def display_plt_metadata(filepath: str, output_file: Optional[str] = None):
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        print(f"Error: File '{abs_path}' does not exist.", file=sys.stderr)
        return

    meta = parse_tecplot_full_metadata(abs_path)

    # Determine output file path if not explicitly provided
    if output_file is None:
        base_name = os.path.splitext(filepath)[0]
        output_file = f"{base_name}_metadata.txt"

    abs_out = os.path.abspath(output_file)

    with open(abs_out, "w", encoding="utf-8") as f_out:
        def out(text: str = ""):
            print(text)
            f_out.write(text + "\n")
            f_out.flush()

        out("=" * 80)
        out(f"FILE: {filepath}")
        out(f"Format: {meta['format']} | Title: '{meta['title']}'")
        out(f"Variables: {len(meta['variables'])} | Zones: {len(meta['zones'])}")
        out("=" * 80)

        # 1. Dataset Auxiliary Data
        out("\n" + "=" * 80)
        out(f"[DATASET AUXILIARY DATA] ({len(meta['dataset_aux'])} entries)")
        out("=" * 80)
        if meta["dataset_aux"]:
            for k, v in meta["dataset_aux"].items():
                out(f"  {k} = {v}")
        else:
            out("  (None)")

        # 2. Variables and Variable Auxiliary Data
        out("\n" + "=" * 80)
        out(f"[VARIABLES & VARIABLE AUX DATA] ({len(meta['variables'])} variables)")
        out("=" * 80)
        for i, vname in enumerate(meta["variables"], start=1):
            vaux = meta["variable_aux"].get(vname, {})
            out(f"  [{i:02d}] {vname}")
            if vaux:
                for ak, av in vaux.items():
                    out(f"       Aux: {ak} = {av}")

        # 3. Zones and Zone Auxiliary Data
        out("\n" + "=" * 80)
        out(f"[ZONES & ZONE AUX DATA] ({len(meta['zones'])} zones)")
        out("=" * 80)
        for z in meta["zones"]:
            dims = (
                f"Dimensions: {z['imax']} x {z['jmax']} x {z['kmax']}"
                if z["zone_type"] == "ORDERED"
                else f"Nodes: {z['imax']}, Elements: {z['jmax']}"
            )
            out(f"  Zone [{z['index']:03d}]: '{z['name']}' ({z['zone_type']}, {dims})")
            if z["aux"]:
                for ak, av in z["aux"].items():
                    out(f"       Aux: {ak} = {av}")
            else:
                out("       Aux: (None)")

    print(f"\nSaved metadata output to: {abs_out}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "ethene.plt"
    out_target = sys.argv[2] if len(sys.argv) > 2 else None
    display_plt_metadata(target, out_target)
