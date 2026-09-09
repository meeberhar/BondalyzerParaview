# Plan: Experimental Isolated Atom Sphere Field Sandbox (`atom_sphere_sandbox.py`)

## Objective
Create a standalone experimental script/tool (`atom_sphere_sandbox.py`) dedicated to visualizing condensed scalar fields on an isolated atomic sphere (C1) with color flood and contours, and providing a clean playground for custom basin-finding algorithms.

Each phase below is self-contained and produces an immediate, visually verifiable outcome in the running viewer:

---

## Phase 1: Interactive Basin Patch Picking & Live Condensed Property Inspector
**Visual Outcome**:
- In **GBA Tools**, clicking on any colored basin patch instantly highlights that patch with a distinct glowing border.
- A new **Selected Basin Card** appears in the GBA sidebar displaying the basin's ID, polarity (minimum/maximum), function name, and a table of its exact integrated condensed totals ($N(\Omega)$, volume, energy, etc.).
- A **Clear Selection** button and deselect-on-outside-click behavior.

**Key Changes**:
- Update `plt_gba_to_vtm.py` to structure `IntegratedFunctionTotals` / `IntegratedFunctionNames` into metadata dictionaries.
- Update `trame_viewer.py` with GBA ray-picking in `on_scene_click` (`state.active_nav_mode == "gba"`).
- Add the reactive **Selected Basin Inspector Card** in the GBA drawer.

---

## Phase 2: 3D Basin Wedge Rendering (PLT Surface Mesh & Geometric Pyramidal Wedge)
**Visual Outcome**:
- When a basin is clicked, its full 3D wedge geometry appears originating from the atomic nucleus and connecting to the spherical patch.
- A toggle in the GBA drawer allows switching between:
  1. **PLT Basin Surface Mesh**: Exact FE boundary surface from the `.plt` file.
  2. **Geometric Wedge to Nucleus**: Solid pyramidal radial wedge to the atomic center.
- Wedge opacity and wireframe/surface styling controls.

**Key Changes**:
- Update `extract_gba_zones_from_plt` to index `CondensedBasinSurface` zones.
- Add procedural radial wedge generator connecting patch perimeter to atom center.
- Add `gba_wedge_actor` with customizable opacity and rendering style.

---

## Phase 3: Floating Viewport Inset Dialog & Docking Controls
**Visual Outcome**:
- An interactive, semi-transparent **Floating Inset Window** hovers in the corner of the 3D viewport displaying condensed metrics side-by-side with the rotating 3D molecule.
- Has minimize/expand, dock-to-sidebar, and close controls so you can rotate the model and view data without opening the drawer.

**Key Changes**:
- Add floating overlay card in `trame_viewer.py` layout content area with Vuetify styling and responsive docking toggles.

## Update progress as you work
- Please use this file as a working todo list to track your progress.