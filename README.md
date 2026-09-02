# BondalyzerParaView

A modern Python toolkit for visualizing Quantum Theory of Atoms in Molecules (QTAIM) and Gradient Bundle Analysis (GBA) topological data from Tecplot binary format in the VTK/ParaView ecosystem.

## Features

- **Tecplot PLT File Conversion**: Convert QTAIM Tecplot binary data to standard VTK formats
- **Interactive 3D Visualization**: Browser-based viewer using Trame and VTK
- **Molecular Skeleton Analysis**: Visualize atoms, bonds, critical points, and bond paths
- **Scalar Field Analysis (SCA)**: Interactive cutplane contours and isosurfaces for 3D scalar fields
- **Gradient Bundle Analysis (GBA)**: Atomic basin patches and condensed field visualization
- **Chemistry-Centric Interface**: Specialized GUI designed for chemists, not generic 3D visualization

## Quick Start

### 1. Install Dependencies with `uv`

```bash
# Clone the repository
git clone <repository-url>
cd BondalyzerParaview

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Run the Viewer

```bash
python trame_viewer.py ethene4.plt
```

The application will start at: **http://localhost:8080/**

### 3. Browse the Interface

- **Overview Tab**: Explore molecular skeleton (atoms, bonds, critical points)
- **SCA Tools**: Analyze 3D scalar fields with interactive contours and isosurfaces
- **GBA Tools**: Visualize atomic basins and condensed field data
- **3D View**: Click any atom or critical point to inspect properties

## Files Overview

| File | Purpose |
|------|---------|
| `trame_viewer.py` | Main interactive visualization application |
| `plt_1d_to_vtm.py` | Convert 1D zones (molecular skeleton) to VTK MultiBlock format |
| `plt_zone0_to_vtk.py` | Convert 3D volume data to VTK ImageData/RectilinearGrid |
| `plt_gba_to_vtm.py` | Extract and convert GBA basin patches |
| `inspect_plt.py`, `inspect_plt2.py` | Tools for examining Tecplot file structure |

## Requirements

- **Python 3.10+**
- **uv** (recommended) or pip
- Dependencies: numpy, vtk, trame, trame-vuetify, trame-vtk (see `requirements.txt`)

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Data Format

Input: **Tecplot binary files** (`.plt`) containing:
- 3D volume grids with scalar fields (electron density, curvature, etc.)
- 1D point/line zones (atoms, bonds, critical points, gradient paths)
- 2D surface patches (atomic basins, finite element surfaces)

Output: **VTK formats** (`.vti`, `.vtr`, `.vtm`, `.vtp`)

## Architecture

The toolkit uses a two-stage pipeline:

1. **Conversion Stage** (Python + struct binary parsing)
   - Parse Tecplot TDV format headers and binary data
   - Restructure into standard VTK arrays and metadata
   - Save as portable XML-based VTK formats

2. **Visualization Stage** (Trame + VTK)
   - Load multi-block VTK datasets
   - Render molecular skeleton with element-aware styling
   - Provide interactive filtering and scalar field analysis

## Contributing

See [AGENTS.md](AGENTS.md) for architecture and development guidelines.

## License

[Include your license here]

## References

- **Tecplot Binary Format**: TDV112 specification
- **QTAIM**: Bader, R. F. W. (1990). *Atoms in Molecules: A Quantum Theory*
- **VTK**: https://www.vtk.org/
- **Trame**: https://kitware.github.io/trame/
