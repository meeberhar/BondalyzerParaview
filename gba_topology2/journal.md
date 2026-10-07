# Development Journal: `gba_topology2`

This journal tracks atomic engineering units, algorithmic decisions, empirical findings, and validation milestones across the `gba_topology2` project.

---

## Standards Reference
All work follows the coding standards outlined in `/AGENTS.md`:
- Atomic units of work (minimal logical increments).
- Post-unit workflow gates:
  1. Unit tests in `pytest`.
  2. Formatting & linting via `ruff`, type checking via `mypy`.
  3. Docstrings and documentation entries in this journal.
- Package and execution orchestration via `uv`.
- Pre-commit enforcement hooks.

---

## Work Log

### [Entry 001] Setup Coding Standards & Tooling Infrastructure
- **Objective**: Establish development standards, directory layout, dev dependencies, and pre-commit automation for `gba_topology2`.
- **Changes**:
  - Authored root `/AGENTS.md` specifying:
    - Atomic units of work.
    - Required testing (`pytest`), formatting/linting (`ruff`), and type checking (`mypy`).
    - Documentation expectations (docstrings + `journal.md`).
    - Dedicated source structure under `gba_topology2/src/`.
    - Automated pre-commit hooks running before each commit.
  - Updated `pyproject.toml` with `[dependency-groups].dev` including `mypy`, `pre-commit`, `ruff`, and `pytest`.
  - Configured `.pre-commit-config.yaml` to enforce `ruff`, `ruff-format`, `mypy`, and `pytest` locally.
  - Installed git hooks via `pre-commit install`.
  - Created initial test structure `tests/test_sanity.py` and `gba_topology2/src/__init__.py`.
- **Validation**:
  - `uv run pre-commit run --all-files`: Passed.
  - `uv run ruff check gba_topology2 tests`: Passed.
  - `uv run ruff format --check gba_topology2 tests`: Passed.
  - `uv run mypy gba_topology2 tests`: Passed.
  - `uv run pytest`: 1 passed.

---

### [Entry 002] Step 1: Data Ingestion & Base Morse Detection (`mesh_geometry.py`, `morse_detector.py`)
- **Objective**: Implement discrete, modular subroutines for Step 1 of the `gba_topology2` roadmap:
  1. `mesh_geometry.py`: Calculate sphere radius $R$, average Euclidean edge length $\overline{\ell}_e$, and intrinsic angular mesh pitch $\delta\theta_{\text{mesh}} = 2\arcsin(\overline{\ell}_e / 2R)$.
  2. `morse_detector.py`: Pure Discrete Morse classification on spherical triangle meshes at $\tau = 0.0$ persistence. Classify vertices into local minima ($L^-(v) = \emptyset$), maxima ($L^+(v) = \emptyset$), and saddles ($L^-(v)$ has $k \ge 2$ connected components). Include Simulation of Simplicity (SoS) tie-breaking to guarantee integer Poincaré-Hopf theorem sum $N_{\max} + N_{\min} - N_{\text{sad}} = 2$.
- **Changes**:
  - Implemented `gba_topology2/src/mesh_geometry.py` with standalone functions: `compute_sphere_radius`, `extract_unique_edges`, `compute_average_edge_length`, `compute_mesh_pitch`, and `analyze_mesh_geometry`.
  - Implemented `gba_topology2/src/morse_detector.py` with `CriticalPoint`, `MorseDetectionResult`, and `compute_discrete_morse_cps`.
  - Added unit test suites `tests/test_mesh_geometry.py` and `tests/test_morse_detector.py` validating octahedron geometry, height function extrema, quadrupole saddle configurations, and dimension validation.
- **Empirical Findings on `Pd_20K.plt`**:
  - $N_{\text{vertices}} = 20,174$, $N_{\text{triangles}} = 40,344$, $\overline{\ell}_e \approx 0.0384$ Å, $\delta\theta_{\text{mesh}} \approx 1.61^\circ$.
  - Without SoS tie-breaking, float discretization plateaus on Electron Density yielded $\chi = 3$. With SoS relative perturbation ($10^{-15}$), $\chi = 2$ is verified across all condensed fields (`Electron Density`, `V`, `ρ mean curvature`, etc.).

---

### [Entry 003] Step 2: Micro-Cluster Identification & Interactive 3D Picking (`micro_cluster.py`, `sphere_viewer.py`)
- **Objective**: Implement modular micro-clustering scaled by $\delta\theta_{\text{mesh}}$ and create the Step 2 interactive Trame 3D viewer with direct ray-pick inspection on `Pd_20K.plt`.
- **Changes**:
  - Implemented `gba_topology2/src/micro_cluster.py` with `MicroCluster`, `BoundaryRingPoint`, `generate_boundary_ring`, and `compute_micro_clusters`.
    - Groups critical points within $\theta_{ij} \le k \cdot \delta\theta_{\text{mesh}}$.
    - Calculates projected spherical centroid, angular/spatial diameter, boundary ring vertices, and local Euler index $\chi_{\text{micro}} = N_{\max} + N_{\min} - \sum \text{mult}_{\text{sad}}$.
  - Implemented `tests/test_micro_cluster.py` testing isolated CPs, cluster grouping, boundary ring coordinates, and parameter validation.
  - Implemented `gba_topology2/src/sphere_viewer.py`:
    - Renders atom sphere with scalar flood, level-set isocontours, and mesh wireframe.
    - Renders Discrete Morse critical points (Red = Maxima, Blue = Minima, Green = Saddles).
    - Renders black boundary rings around multi-CP micro-clusters.
    - Interactive 3D Picking: Ray-picks clusters and critical points directly in 3D canvas and via sidebar table to highlight in 3D and display full inspector cards.
- **Validation**:
  - Pre-commit hook suite (`ruff check`, `ruff format`, `mypy`, `pytest`): 10 passed, 0 errors.
  - Successfully ran `sphere_viewer.py` on `Pd_20K.plt` verifying 47 micro-clusters (39 multi-CP) and live HTTP rendering.

---

### [Entry 004] Saddle Multiplicity Transparency & Index Accounting (`micro_cluster.py`, `sphere_viewer.py`)
- **Objective**: Clarify topological index accounting for clusters containing higher-order / multi-pass saddles (specifically resolving empirical observation on `Pd_20K.plt` Cluster MC7 on Shape Index where 4 Maxima + 1 Saddle gave net index $+1$).
- **Background & Root Cause**:
  - On a 2D triangulated mesh, a saddle vertex with $n_{\text{lower}}$ disconnected lower link components carries topological index $-(n_{\text{lower}} - 1) = -\mu$.
  - In Cluster MC7, the single saddle vertex has $n_{\text{lower}} = 4$ descending components, yielding multiplicity $\mu = 3$ (index $-3$).
  - Naive display of `"4 Max, 1 Sad"` led users to expect $4 - 1 = +3$, whereas discrete Morse theory rigorously dictates $4 + 0 - 3 = +1$.
- **Changes**:
  - `micro_cluster.py`: Added `total_saddle_index` to `MicroCluster`. Updated `composition_summary` to annotate multi-index saddles when `total_saddle_index > n_saddles` (e.g. `"4 Max, 1 Sad (mult 3)"`).
  - `sphere_viewer.py`: Exposed `total_saddle_index`, individual CP `multiplicity`, and `index_contrib` in the UI state. Added an explicit arithmetic index breakdown to the cluster inspector card (`+N_max + N_min - total_saddle_index = net_index`) and annotated multi-component saddles in the constituent critical point list.
  - `tests/test_micro_cluster.py`: Added `test_higher_order_saddle_cluster_accounting` verifying cluster serialization, local Euler index evaluation, and composition summary formatting with higher-order saddles.
- **Validation**:
  - Unit tests: 11 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 005] Step 3 Evaluation: Adjacent Extrema Scanning & Architecture Hook (`field_scanner.py`)
- **Objective**: Implement a dedicated diagnostic tool to scan all condensed scalar fields on `Pd_20K.plt` for adjacent same-type critical points ($d_G = 1$) to evaluate the necessity of Step 3 extrema fusion.
- **Implementation**:
  - Authored `gba_topology2/src/field_scanner.py` using 1-skeleton graph edge adjacency and Discrete Morse classification.
  - Ran scan across all 13 condensed scalar fields on Atom #1 of `Pd_20K.plt` with unperturbed floating-point data ($\epsilon = 0.0$) and Simulation of Simplicity ($\epsilon = 10^{-15}$).
- **Empirical Findings**:
  - With raw unperturbed data ($\epsilon = 0.0$), flat zero-curvature or valley plateaus produce rare adjacent same-type extrema (e.g. Min-Min at vertices `659` and `663` in MC29 on `Electron Density` and MC35 on `V`).
  - With default Simulation of Simplicity ($\epsilon = 10^{-15}$), infinitesimal tie-breaking resolves plateaus deterministically without splitting extrema, producing exactly 0 adjacent same-type extrema ($d_G = 1$) and preserving $\chi = 2$.
  - The resulting critical point is consistent within mesh edge spacing ($\sim 0.038$ Å) with a fused centroid, and downstream Step 8 (Harmonic Nudge) optimizes the physical location.
- **Architectural Decision**:
  - Documented extension hook in `sphere_viewer.py` right before micro-clustering.
  - Documented architectural role in `field_scanner.py` and `plan.md`.
  - Bypassed active Step 3 runtime fusion in favor of proceeding directly to Step 4.

---

### [Entry 006] Unit 1: Tangent PCA, Cluster Morphology & Elliptical Boundaries (`micro_cluster.py`)
- **Objective**: Implement local geometric shape characterization (PCA on sphere tangent plane) in `micro_cluster.py` to classify multi-CP clusters into compact (Type 1) vs elongated string (Type 2) morphologies and generate oriented elliptical boundary rings.
- **Changes**:
  - `micro_cluster.py`:
    - Implemented `compute_tangent_pca`: projects member unit vectors onto the tangent plane at the centroid, diagonalizes the 2D covariance tensor, and extracts aspect ratio $\alpha = \sqrt{\lambda_1 / \lambda_2}$ and principal 3D tangent vector.
    - Implemented `generate_elliptical_boundary_ring`: generates a closed oriented elliptical boundary ring parameterized by semi-major radius $a$ and semi-minor radius $b$ along the principal tangent axes.
    - Updated `MicroCluster` dataclass with `morphology` (`"compact"` vs `"string"`), `aspect_ratio`, `principal_axis_unit`, `semi_major_rad`, and `semi_minor_rad`.
    - Updated `compute_micro_clusters` to automatically fit elliptical rings when $\alpha \ge 2.0$ for multi-CP clusters.
  - `tests/test_micro_cluster.py`:
    - Added `test_string_morphology_and_elliptical_ring` and `test_compact_morphology`.
    - Added validation for `aspect_ratio_threshold >= 1.0`.
- **Validation**:
  - Unit tests: 7 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 007] Unit 2: Macro-String Aggregation (`string_aggregator.py`)
- **Objective**: Implement collinear string aggregation (`string_aggregator.py`) to chain intrinsically elongated Type 2 clusters and collinear neighboring clusters along low-field valleys into unified `MacroString` entities with oriented elliptical excision boundaries.
- **Changes**:
  - `string_aggregator.py`:
    - Implemented `MacroString` container capturing constituent clusters, merged critical point members, tangent PCA, aspect ratio, length, width, and oriented boundary ring.
    - Implemented `aggregate_cluster_strings`: bridges string-like micro-clusters within $k_{\text{bridge}} \cdot \delta\theta_{\text{mesh}}$, computes global tangent PCA across all points in the string, and fits an oriented elliptical boundary ring $(a, b, \vec{t}_1)$.
  - `tests/test_string_aggregator.py`:
    - Implemented unit tests validating string aggregation, conservation of constituent critical points, local Euler characteristic, spherical constraint on elliptical ring points, and exclusion of isotropic compact clusters.
- **Validation**:
  - Full test suite: 16 passed across all tests.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 008] Phase 1: Polarity Detection & Effective Extrema (`polarity_detector.py`)
- **Objective**: Implement outward radial slope analysis on the sphere to unambiguously classify composite micro-clusters and macro-strings with positive Euler index ($\chi_{\text{local}} = +1$) into either Mountain/Peak ($E\text{-MAX}$) or Basin/Pit ($E\text{-MIN}$).
- **Changes**:
  - `polarity_detector.py`:
    - Implemented `evaluate_spherical_radial_slope`: samples mesh vertices in an annular ring $[r_{\text{inner}}, r_{\text{outer}}]$ around the cluster centroid, computes the directional radial derivative $\partial f / \partial \theta$ via linear regression, and evaluates directional concordance confidence.
    - Implemented `classify_cluster_polarity`: probes terrain around a `MicroCluster`, assigns $E\text{-MAX}$ (if $\partial f / \partial \theta < 0$, field descends outward) or $E\text{-MIN}$ (if $\partial f / \partial \theta > 0$, field ascends outward), extracts the representative peak/pit value, and wraps into an `EffectiveExtremum` dataclass.
  - `tests/test_polarity_detector.py`:
    - Implemented unit tests for analytical spherical Gaussian peak ($df/d\theta < 0$), analytical basin ($df/d\theta > 0$), and composite micro-cluster classification (both peak and basin cases with concordance $\ge 0.7$).
- **Validation**:
  - Full test suite: 20 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 009] Phase 2: Catastrophe Classification & Sensitivity Metrics (`catastrophe_classifier.py`)
- **Objective**: Implement 3-fold (monkey saddle, $\chi = -2$) and 4-fold (octupolar cross, $\chi = -3$) catastrophe constellation identification in the meso-scale angular window, computing scalar barrier depth and angular span metrics to distinguish numerical noise (spurious unfolding) from genuine physical symmetry breaking.
- **Changes**:
  - `catastrophe_classifier.py`:
    - Implemented `evaluate_constellation_metrics`: computes pairwise angular span, barrier depth $\Delta f_{\text{internal}}$, relative barrier depth $\eta = \Delta f / \text{span}(f)$, and cyclic angular standard deviation across flanking saddles projected onto the tangent plane.
    - Implemented `classify_catastrophes`: searches around positive-index core clusters for candidate flanking saddles satisfying the topological index relation $\chi = 1 - k$ ($k \in \{4, 3\}$), prioritizes higher-order 4-fold constellations before 3-fold, computes a bifurcation confidence score $\mathcal{B} \in [0.0, 1.0]$, classifies constellations as `"spurious_unfolding"` (fused $E\text{-MSAD}$ / $E\text{-4SAD}$) vs `"physical_symmetry_breaking"` based on user-tunable `barrier_tol` and `angular_tol_mult`, and fits an enclosing boundary ring on the sphere.
  - `tests/test_catastrophe_classifier.py`:
    - Implemented unit tests for 3-fold spurious unfolding with negligible barrier (fused $E\text{-MSAD}$), 3-fold physical symmetry breaking with finite barrier ($\mathcal{B} > 0.8$, preserved separate saddles), 4-fold octupolar detection (5 clusters merging into $E\text{-4SAD}$, $\chi = -3$), and input validation.
- **Validation**:
  - Full test suite: 24 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 010] Phase 3: Excision Boundary Profiling & Port Extraction (`excision_boundary.py`)
- **Objective**: Implement closed 1D excision boundary loop profiling and discrete port extraction to detect alternating valley and ridge ports, independently verifying the topological index formula $\chi = 1 - k$ directly on the boundary.
- **Changes**:
  - `excision_boundary.py`:
    - Implemented `interpolate_scalar_at_points`: samples field values along arbitrary 3D boundary points using Inverse Distance Weighting (IDW) from sphere mesh vertices.
    - Implemented `extract_boundary_ports`: detects strict local 1D extrema on the closed periodic ring $f(\phi)$, filters peaks/valleys by topological prominence, and extracts `BoundaryPort` entities with type (`"valley"` vs `"ridge"`), azimuthal angles ($\phi$), 3D positions, and scalar values.
    - Implemented `profile_boundary_loop`: profiles boundary loops end-to-end, evaluates topological balance (equal valley and ridge counts), and computes the implied Euler index $\chi = 1 - k$.
  - `tests/test_excision_boundary.py`:
    - Implemented unit tests for 2-fold simple saddle fields (2 valleys, 2 ridges $\implies \chi = -1$), 3-fold monkey saddle fields (3 valleys, 3 ridges $\implies \chi = -2$), 4-fold octupolar cross fields (4 valleys, 4 ridges $\implies \chi = -3$), end-to-end mesh IDW interpolation, exact vertex interpolation, and empty loop validation.
- **Validation**:
  - Full test suite: 30 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 011] Phase 4: Harmonic Nudge Energy Minimization (`harmonic_nudge.py`)
- **Objective**: Implement symmetry energy functionals and gradient-free pattern optimization on the sphere to refine and center effective critical points and launch ports tailored to their fold order $k \in \{0, 2, 3, 4\}$.
- **Changes**:
  - `harmonic_nudge.py`:
    - Implemented `compute_k_fold_symmetry_energy`: evaluates circular variance for extrema ($k=0$), and Fourier spectral power penalty of non-$k$ harmonics for saddles ($k=2, 3, 4$).
    - Implemented `harmonic_nudge_critical_point`: performs spherical pattern search on the tangent plane with adaptive step halving and maximum displacement clamping scaled by $\delta\theta_{\text{mesh}}$.
    - Implemented `perfect_boundary_ports`: optimizes azimuthal angles of boundary ports toward ideal $360^\circ / k$ symmetry spacing using circular phase offset minimization.
  - `tests/test_harmonic_nudge.py`:
    - Implemented unit tests for circular variance extremum energy, 2-fold quadrupole saddle Fourier penalty, pattern search convergence on a perturbed extremum, and 3-fold ($120^\circ$) boundary port angle perfection.
- **Validation**:
  - Full test suite: 34 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 012] Phase 5: Interactive 3D Catastrophe & Boundary Port Visualization (`sphere_viewer.py`)
- **Objective**: Integrate catastrophe classification and boundary port extraction into the Trame 3D viewer (`sphere_viewer.py`), enabling interactive inspection of monkey saddles ($k=3$, $\chi = -2$), 4-fold cross saddles ($k=4$, $\chi = -3$), and their alternating valley/ridge ports directly on `Pd_20K.plt`.
- **Changes**:
  - `sphere_viewer.py`:
    - Added dedicated VTK pipelines for:
      - Catastrophe boundary rings (purple tubes on the sphere surface).
      - Boundary ports: Valley Ports (vivid orange spheres) and Ridge Ports (cyan spheres).
    - Added Step 5 interactive sidebar card:
      - Toggle switches for Catastrophe Rings and Boundary Ports.
      - User sliders for relative barrier depth tolerance (`cat_barrier_tol`, default $0.5\%$) and angular span tolerance (`cat_angular_tol_mult`, default $4.0\,\delta\theta_{\text{mesh}}$).
      - Catastrophes summary table displaying ID, Type (`E-MSAD`/`E-4SAD`), net index $\chi$, and classification (`Fused` vs `Split`).
    - Added Catastrophe Inspector card displaying:
      - Net index, fold order $k$, exact port breakdown ($k\text{V} + k\text{R}$).
      - Bifurcation confidence score ($\mathcal{B}$).
      - Scalar barrier depth, relative barrier percentage, and angular span in degrees.
      - Constituent micro-cluster IDs.
- **Empirical Validation on `Pd_20K.plt` (`V (condensed)`)**:
  - Identified all 8 expected monkey saddle constellations:
    - **`CAT_1`** through **`CAT_6`**: Octahedral/cubic symmetry group (including **`MC29`** + flanking saddles `MC44`, `MC53`, `MC56`, and **`MC33`** + flanking saddles `MC46`, `MC50`, `MC54`). Each possesses 3 Valley ports and 3 Ridge ports ($3\text{V} + 3\text{R}$) with net $\chi = -2$.
    - **`CAT_7`** & **`CAT_8`**: Symmetry-equivalent pole configurations (including **`MC41`** + `MC61`, `MC63`, `MC65`). Each resolves with $3\text{V} + 3\text{R}$, net $\chi = -2$, and is recognized as a spurious numerical unfolding ($\Delta f / \text{span}(f) < 0.5\%$) that fuses cleanly into an effective monkey saddle.
- **Validation**:
  - Full test suite: 34 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.

---

### [Entry 013] Step 6 Assessment Toggles: Effective Critical Points & Harmonic Nudge (`sphere_viewer.py`)
- **Objective**: Expose the multi-CP collapse and harmonic nudge stages as **independent** interactive switches in the viewer, so each algorithmic stage can be judged visually before we commit to a fully automated pipeline.
- **Design Decisions**:
  - **Four independent switches** (not a master switch) so stages can be isolated:
    1. `reduce_catastrophes` — collapse a catastrophe constellation (e.g. 1 core + 3 flanking saddles) into a single effective saddle `E-SAD`.
    2. `reduce_multi_clusters` — collapse a standalone multi-CP micro-cluster into a single `E-MAX`/`E-MIN` via `classify_cluster_polarity`.
    3. `nudge_cp_positions` — run `harmonic_nudge_critical_point` symmetry-energy pattern search on effective CPs.
    4. `perfect_port_angles` — run `perfect_boundary_ports` to enforce ideal $360^\circ / k$ port spacing.
  - **Standard glyphs, boundary retained**: effective CPs reuse the normal red/blue/green glyphs. The black cluster ring and purple catastrophe ring stay visible after collapse, so a lone glyph inside a ring reads unambiguously as "this set was collapsed".
  - **Nudge is opt-in and lazy**: the pattern search only executes when `nudge_cp_positions` is on, keeping the default interaction responsive.
- **Changes**:
  - `harmonic_nudge.py`:
    - Added `reposition_ports_on_ring`: `perfect_boundary_ports` only rewrites the azimuthal angle `phi_rad`, so ports were visually unchanged in 3D. This subroutine recovers the ring frame $(\hat{c}, \hat{t}_1, \hat{t}_2, a)$ from the boundary samples and projects perfected angles back onto the excision loop, re-interpolating scalars via IDW.
    - Added `_ring_frame` helper and a `probe_radius_pitch` argument to `harmonic_nudge_critical_point` so constellations spanning several pitches probe outside their own footprint (per the excision-radius rule in `micro_cluster`).
  - `sphere_viewer.py`:
    - Deferred CP glyph population to the end of `update_topology()` so collapsed members can be hidden while rings persist.
    - Added `raw_cp_to_ui` / `effective_cp_to_ui` builders producing a uniform CP list for the table, picking, and inspector.
    - Added `state.display_counts` with a **multiplicity-weighted** index sum, so the displayed $\chi$ is invariant under reduction (a multiplicity-2 saddle glyph carries index $-2$; an effective saddle carries the net index of its set).
    - Added Step 6 sidebar card: 4 switches, max-nudge slider, reduction/nudge/perfection stats, and an effective-CP table.
    - CP inspector now shows origin set, net index, pre-nudge center, displacement, iteration count, symmetry-energy drop, and the constituent CP list.
    - 3D picking resolves effective glyphs (world-space and screen-space fallback) and skips raw CPs hidden by reduction.
  - `tests/test_harmonic_nudge.py`: 4 new tests for `reposition_ports_on_ring` (ring round-trip, equispaced 3D placement, IDW re-interpolation, edge cases).
  - `gba_topology2/verify_step6_toggles.py`: headless harness exercising all 8 toggle permutations on real data.
- **Empirical Validation on `Pd_20K.plt` (`V (condensed)`, 20174 nodes, $\delta\theta = 1.606^\circ$)**:
  - Raw Morse: 70 Max / 136 Min / 200 Sad, $\chi = 2$. 66 clusters (34 multi-CP), 8 catastrophes.
  - `reduce_catastrophes`: 48 raw CPs $\to$ 8 effective saddles; displayed $\chi$ stays exactly $2$.
  - `reduce_multi_clusters`: 352 raw CPs $\to$ 28 effective extrema; $\chi$ stays $2$.
  - Both: 400 CPs $\to$ 36 effective CPs (22 Max / 12 Min / 8 Sad), $\chi = 2$.
  - `nudge_cp_positions`: catastrophe centroids move mean $0.843^\circ$, max $1.204^\circ$ ($\approx 0.5$–$0.75\,\delta\theta_{\text{mesh}}$); symmetry energy never increases (verified descent across all 36 effective CPs).
  - `perfect_port_angles`: all 48 ports re-projected onto their excision loops (verified to lie on the ring's small circle to $<10^{-3}$ rad).
- **Validation**:
  - Full test suite: 38 passed in `pytest`.
  - Format, lint, & typecheck: `ruff format`, `ruff check`, and `mypy` passing cleanly.
- **Open Questions / Next Steps**:
  - The nudge displacement of $\approx 0.5\,\delta\theta_{\text{mesh}}$ for catastrophes is meaningful and should be measured against the ideal $k$-fold port phase once ports are perfected jointly with positions.
  - Effective saddles currently inherit the constellation centroid as the pre-nudge seed; a barrier-weighted centroid may be a better seed than the unweighted one.
  - Ready to feed the perfected ports into Step 6 basin tracing (`basin_tracer.py`).

---

### [Entry 014] Chi-Gated Reduction: Annihilating Topologically Neutral Sets (`polarity_detector.py`, `sphere_viewer.py`)

**Date**: 2026-10-07

- **Problem (user-reported)**: With `reduce_multi_clusters` on, cluster MC10
  (2 Max, 8 Sad, 6 Min; net $\chi_{\text{local}} = 0$) was rendered as a red
  *maximum* glyph while honestly reporting net index $+0$. A glyph carrying zero
  index is a category error: extrema carry $+1$, saddles carry $\le -1$.
- **Root cause**: `classify_cluster_polarity` is a *binary* classifier by design
  (its contract covers $\chi = +1$ sets: radial slope $< 0 \Rightarrow$ E-MAX,
  $> 0 \Rightarrow$ E-MIN). The viewer's Step 5 reduction loop called it for
  **every** standalone multi-CP cluster with no index gate, so all 16 $\chi = 0$
  ridge strings (MC1-MC16, the 2 Max/8 Sad/6 Min periodic stripe family,
  AR $\approx 5.7$) were forced into an extremum glyph.
- **Topological contract adopted** (`effective_reduction_decision`):
  - $\chi = 0$  -> **annihilate**: hide all member glyphs, emit **no** effective
    CP, keep the boundary ring (ghost-grey annulment style) as the visual record.
  - $\chi = +1$ -> E-MAX / E-MIN via the polarity classifier (unchanged).
  - $\chi \le -1$ -> E-SAD carrying the net index, fold order $k = 1 - \chi$.
  - $\chi \ge +2$ -> **irreducible**: one effective CP carries at most $+1$, so
    the set stays expanded (counted, never silently collapsed).
- **Changes**:
  - `polarity_detector.py`: new pure `effective_reduction_decision(chi, polarity)`
    + `ReductionDecision` literal; exported.
  - `sphere_viewer.py`: gate the cluster reduction loop on the decision; new
    ghost-grey `annulled_rings_actor` (tube radius 0.014, opacity 0.85, offset
    1.014R) re-emitting annulled cluster boundaries; `is_annulled` /
    `is_irreducible` flags on cluster UI rows with table chips; Step 6 card stat
    lines; `n_clusters_annihilated` / `n_clusters_irreducible` in
    `reduction_stats`.
  - `verify_step6_toggles.py`: mirrors the gate; new assertions (no ECP carries
    index 0 or $\ge +2$; every standalone multi-CP cluster is reduced, annulled,
    or irreducible; annihilation hides members; $\chi = 2$ invariant).
  - `tests/test_polarity_detector.py`: 6 unit tests for the decision rule
    (all branches + invalid-polarity `ValueError`).
- **Empirical validation (Pd_20K, V (condensed), k=2.0)**: 28 standalone multi-CP
  clusters -> 12 reduced ($\chi=+1$: 6 E-MAX, 6 E-MIN), **16 annulled**
  ($\chi=0$), 0 irreducible. Displayed: 8 Max / 28 Min / 30 Sad, $\chi = 2$
  invariant preserved. With catastrophes too: 6/12/8 displayed, 20 ECPs, $\chi=2$.
- **MC32 note**: MC32 (1 mult-2 Sad + 3 Min, $\chi=+1$) is a CAT_4 constituent;
  the green saddle near it is `E-SAD_CAT_4` (net $\chi=-2$, correct monkey
  saddle). Standalone it reduces to E-MIN $\chi=+1$ (3-lobed basin), which is
  topologically sound. Open question for next session: whether a $\chi=+1$ basin
  containing a mult-2 monkey saddle should prefer the saddle glyph with a
  basin annotation.
- **Tooling lessons**:
  - `insert_edit_into_file` auto-generated three bogus tests in
    `test_polarity_detector.py` (calling the new function with a
    `classify_cluster_polarity` signature). Always `grep -n "^def test"` after
    inserting into test files; delete artifacts with `sed -i '' 'A,Bd'`.
  - trame widgets are **not callable** (`Div(...)(...)` -> TypeError); children
    must go through `with` blocks. `v_text` is unproven in this codebase - use
    mustache positional children.
  - `ruff format .` reformats 6 legacy files outside the pre-commit scope
    (`^(gba_topology2/|tests/)`); scope format commands to the touched files.
  - Headless trame: `state.flush()` does not fire `@state.change` handlers
    outside the server loop; use the verify harness for pipeline assertions and
    the `s.start` stub only for UI-build smoke tests.
