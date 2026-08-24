#!/usr/bin/env python3
"""
Inspect and extract metadata (zones and variables) from Tecplot binary (.plt) files
with support for both ParaView's native readers and Tecplot binary format parsing.
"""

import os
import sys
import struct
from typing import List, Dict, Any, Optional

try:
    import paraview.simple as pvs
    import paraview.servermanager as sm
    PARAVIEW_AVAILABLE = True
except ImportError:
    PARAVIEW_AVAILABLE = False


def read_tecplot_binary_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parse the header of a Tecplot binary (.plt) file (e.g. #!TDV112 / #!TDV107)
    to extract title, variable names, and zone names.
    """
    with open(filepath, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"#!TDV"):
            raise ValueError(f"Not a valid Tecplot binary file (Magic: {magic!r})")

        magic_str = magic.decode("latin1", errors="replace")
        byte_order_flag = struct.unpack("<I", f.read(4))[0]
        endian = "<" if byte_order_flag == 1 else ">"
        
        file_type = struct.unpack(f"{endian}I", f.read(4))[0]

        # Read null-terminated 32-bit int string (Title)
        title_chars = []
        while True:
            c = struct.unpack(f"{endian}I", f.read(4))[0]
            if c == 0:
                break
            title_chars.append(chr(c))
        title = "".join(title_chars)

        # Number of variables
        num_vars = struct.unpack(f"{endian}I", f.read(4))[0]
        variables = []
        for _ in range(num_vars):
            var_chars = []
            while True:
                c = struct.unpack(f"{endian}I", f.read(4))[0]
                if c == 0:
                    break
                var_chars.append(chr(c))
            variables.append("".join(var_chars))

        # Scan for zones (Marker 299.0f)
        zones = []
        while True:
            marker_bytes = f.read(4)
            if len(marker_bytes) < 4:
                break
            marker_val = struct.unpack(f"{endian}f", marker_bytes)[0]
            if abs(marker_val - 299.0) < 1e-4:  # Zone header marker
                z_chars = []
                while True:
                    c_bytes = f.read(4)
                    if len(c_bytes) < 4:
                        break
                    c = struct.unpack(f"{endian}I", c_bytes)[0]
                    if c == 0:
                        break
                    z_chars.append(chr(c))
                zone_name = "".join(z_chars)
                if zone_name:
                    zones.append(zone_name)
            elif abs(marker_val - 357.0) < 1e-4 or abs(marker_val - 399.0) < 1e-4:
                # End of zone headers / geometries / custom labels
                break

    return {
        "format": magic_str,
        "title": title,
        "num_variables": len(variables),
        "variables": variables,
        "num_zones": len(zones),
        "zones": zones,
    }


def open_plt_file(filepath: str):
    """
    Open and inspect a Tecplot PLT file, trying ParaView readers first
    and reading file metadata.
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    print("=" * 70)
    print(f"Opening Tecplot File: {filepath}")
    print(f"Absolute Path:       {abs_path}")
    print("=" * 70)

    # 1. Parse Tecplot header metadata directly
    meta = read_tecplot_binary_metadata(abs_path)
    print(f"\n[Tecplot Header Info]")
    print(f"Format Version : {meta['format']}")
    print(f"Dataset Title  : {meta['title']}")
    print(f"Total Variables: {meta['num_variables']}")
    print(f"Total Zones    : {meta['num_zones']}")

    print("\n" + "-" * 40)
    print(f"VARIABLES ({meta['num_variables']}):")
    print("-" * 40)
    for i, var in enumerate(meta["variables"], start=1):
        print(f"  [{i:02d}] {var}")

    print("\n" + "-" * 40)
    print(f"ZONES ({meta['num_zones']}):")
    print("-" * 40)
    for i, zone in enumerate(meta["zones"], start=1):
        print(f"  [{i:03d}] {zone}")

    # 2. Inspect with ParaView
    if PARAVIEW_AVAILABLE:
        print("\n" + "=" * 70)
        print("ParaView Integration Status:")
        print("=" * 70)
        print(f"ParaView Python Bindings: Available (ParaView version: {pvs.paraview.__version__})")
        try:
            reader = pvs.VisItTecplotBinaryReader(registrationName=os.path.basename(filepath), FileName=abs_path)
            print("ParaView VisItTecplotBinaryReader initialized successfully.")
        except Exception as e:
            print(f"ParaView reader note: {e}")
    else:
        print("\nParaView Python module is not loaded in current environment.")


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "ethene.plt"
    open_plt_file(target_file)
