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

---

## Phase 4: Morse Critical Point Spatial Clustering & Composition Analysis
**Visual Outcome**:
- Automatic grouping of closely spaced critical points into **CP Clusters** on the atomic sphere.
- **CP Clusters & Composition Table** in the sidebar detailing:
  - Cluster ID and member count
  - Composition signature (e.g. `4 Max, 3 Sad (Net: +1)`)
  - Net Poincaré-Hopf topological index $\chi_{\text{cluster}} = N_{\max} + N_{\min} - N_{\text{sad}}$
  - Cluster diameter and centroid coordinates
- **3D Cluster Halos**: Translucent wireframe halos surrounding multi-CP clusters.
- **Glyph Scale Slider**: Dynamic slider to shrink CP glyphs to visually separate dense clusters.
- **Cluster Inspector Card**: Inspect individual constituent critical points within the cluster.

**Key Changes**:
- Added `compute_cp_clusters` in `gba_topology/topology_engine.py` with spatial graph clustering and composition metrics.
- Added cluster wireframe halos, glyph size scaling, and reactive cluster inspector in `gba_topology/stage0_sphere_view.py`.

---

## Phase 5: Two-Class Hierarchy & Effective Critical Points (ECPs)
**Visual Outcome**:
- Clean reduction of raw, cluttered critical points into coherent **Effective Critical Points** (`E-MAX`, `E-MIN`, `E-SAD`, `E-MSAD`).
- **Two-Class Hierarchy**:
  1. **Class 1 (Isolated CPs)**: Single-point features fixed at lowest persistence ($\tau = 0.0\%$).
  2. **Class 2 (Clustered CPs)**: Multi-point clusters evaluated across a persistence sweep ($\tau = 0.0\% \dots 5.0\%$) to determine stable topological charge $\chi$, placed at cluster centroids.
- **Toggle View in UI**: Switch between viewing raw Morse CPs vs. resolved Effective CPs.

**Key Changes**:
- Implemented `compute_effective_critical_points` in `gba_topology/topology_engine.py`.
- Added VTK actors for Effective Maxima (Red), Effective Minima (Blue), Effective Saddles (Green), and Effective Monkey Saddles (Purple).

---

## Phase 6: Adaptive Domain Expansion & Monkey Saddle Resolution
**Visual Outcome**:
- Unfolded higher-order singularities (e.g. 4-fold or 3-fold monkey saddles with micro-wrinkles) dynamically expand their domain radius until the enclosed Euler characteristic reaches fundamental closure ($|\chi| = 1$ or $\chi = -2$).
- Avoids false isolated peripheral points by absorbing consumed micro-basins into the parent catastrophe hub.
- **Boundary Excision Loops**: Circular boundary profiles $\partial D(\mathbf{c}, r)$ with pure alternating launch ports:
  - Simple Saddle (`E-SAD`): 2 Ridge Maxima (Orange), 2 Valley Minima (Cyan) -> 4 alternating crossings.
  - Monkey Saddle (`E-MSAD`): 3 Ridge Maxima (Orange), 3 Valley Minima (Cyan) -> 6 alternating crossings.
- **Harmonic Centroid Nudging**:
  - `refine_extremum_harmonic_centroid`: Level-curve variance minimization $\min \text{Var}(f|_{\partial D})$ for extrema with inter-basin separation constraints.
  - `refine_monkey_saddle_harmonic_centroid`: Port eccentricity energy minimization $\min \sum (\Delta \theta_k - 60^\circ)^2$.
  - UI switch added ("Enable Harmonic Centroid Nudge", default: `False` for fast browsing).

---

## Phase 7: Boundary Flux Principle for Fine Mesh Invariance (Completed)
**Visual Outcome**:
- On high-density meshes (`Pd_20K.plt`), clusters containing mixed numerical ripple features (such as `2 Max, 1 Min, 1 Sad`) accurately classify as either an `E-MIN` or `E-MAX` based on the macroscopic normal derivative across the outer boundary loop.
- Core extremum values ($\min(f_{\text{members}})$ or $\max(f_{\text{members}})$) are compared against the outer boundary loop mean $\overline{f}_{\partial D}$, eliminating false maxima/minima flips caused by internal ripple counts.

**Key Changes**:
- Updated `compute_effective_critical_points` in `gba_topology/topology_engine.py` with multi-member depth/height boundary flux evaluation.
- Verified absence of duplicate glyph overlaps (e.g., half-red/half-blue z-fighting) via full vertex-ID subsumption in `absorbed_vids`.

## Update progress as you work
- Please use this file as a working todo list to track your progress.