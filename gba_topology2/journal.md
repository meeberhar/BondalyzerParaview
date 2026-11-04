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

---

### [Entry 015] Catastrophe Claim Collision Fix & Reset to Defaults (`sphere_viewer.py`)

- **Reported symptom (Pd_20K, modified Willmore energy)**: on symmetric
  micro-clusters MC21 / MC22, enabling "Reduce Multi-CP Clusters" produced a
  saddle for one and *nothing* for the other; with both reduction toggles on,
  neither showed a saddle. The clusters are long noisy strings (25-28 CPs,
  ~12 Max / 13-14 Sad, $\chi = -1$, aspect ratio ~8) linking two maxima along
  one direction and two minima along the perpendicular direction.
- **Root cause**: `cat_cluster_ids` was built **unconditionally** from every
  detected catastrophe's constituents, and the standalone multi-cluster stage
  skipped any cluster in that set. When `reduce_catastrophes` was **off**, the
  catastrophe stage collapsed nothing, so a claimed cluster was reduced by
  *neither* stage: its member glyphs vanished and no effective CP was emitted.
  Ownership of a cluster must follow the stage that actually performs the
  collapse.
- **Fix**: extracted `catastrophe_claimed_cluster_ids(catastrophes,
  reduce_catastrophes)` as a module-level, unit-testable subroutine. It returns
  the empty set when catastrophe reduction is disabled, and the full set of
  constellation members when enabled. `verify_step6_toggles.py` mirrors the
  rule.
- **Reset to Defaults**: added `DEFAULT_UI_STATE` as the single canonical
  baseline, now consumed by *both* the initial state seeding (replacing 19
  literal assignments) and the new `ctrl.reset_defaults` action, so the two
  cannot drift. `selected_field` is deliberately excluded - the field is a data
  choice, not an analysis parameter. A `suppress_param_change` latch makes the
  19-key batched write rebuild the pipeline once instead of 19 times. Button
  added to the toolbar next to "Reset Camera".
- **Empirical validation (Pd_20K, modified Willmore energy, 314 Max / 67 Min /
  363 Sad, $\chi = 2$)**:

  | k_pitch | catastrophes | E-SAD (old rule) | E-SAD (new rule) | recovered |
  |---------|--------------|------------------|------------------|-----------|
  | 2.0     | 0            | 20               | 20               | +0        |
  | 2.4     | 6            | 8                | 26               | **+18**   |
  | 3.0     | 8            | 0                | 24               | **+24**   |

  At $k = 3.0$ the old rule collapsed **every** effective saddle to zero - the
  six 26-28 CP $\chi = -1$ strings (MC1-MC6, AR ~7.7-8.8) were all swallowed by
  catastrophes and stranded. The new rule restores all 24.
- **Key empirical finding for the string problem**: the long saddle strings are
  already detected as single $\chi = -1$ clusters at $k = 2.0$ (MC1-MC6,
  28 CPs, AR ~8.7, morphology `string`) and reduce correctly once the collision
  is fixed. Raising $k$ *merges* neighbouring strings together and triggers
  spurious catastrophe detection, which is what made the behaviour look
  erratic. The prescreen/aggregation work is therefore about **stabilising**
  these clusters against the catastrophe stage, not about merging them.
- **Tests**: `tests/test_sphere_viewer_reduction.py` (8 tests) - claim rule
  both toggle states, empty list, shared cluster, defaults coverage against the
  `@state.change` key list, pinned baseline values, non-mutation of the shared
  table. Suite: 53 passed. `ruff format` / `ruff check` / `mypy` clean.
- **Open next**: (a) ~~gate catastrophe *detection* so a $\chi \le -1$ string is
  not consumed as a flanking saddle~~ -> **done in Entry 016**;
  (b) quadrupolar prescreen (2 Max longitudinal + 2 Min transverse) to
  recognise these corridors explicitly and bridge fragments such as MC11+MC55;
  (c) wire `string_aggregator.py` into the pipeline.

---

### [Entry 016] Flanking-Saddle Eligibility Gate (`catastrophe_classifier.py`)

- **Symptom**: on the modified Willmore energy, raising `k_pitch` produced
  spurious catastrophes (0 at $k=2.0$ -> 6 at $2.4$ -> 8 at $3.0$) that
  swallowed the long $\chi = -1$ saddle strings MC1-MC6 (26-28 CPs, AR ~7.7-8.8).
  Symmetric clusters behaved asymmetrically because constellation assembly is
  order-dependent once strings enter the pool.
- **Root cause**: the flanking pool was `n_saddles > 0`. A $+1$ core plus three
  $\chi = -1$ strings sums to $1 + 3(-1) = -2$, which matches the monkey-saddle
  template exactly. Index arithmetic alone cannot distinguish a simple saddle
  from a composite string of the same net index.
- **Fix**: `is_flanking_saddle_candidate(cluster)` - admit a cluster only when it
  has saddles, **no extrema**, and net index exactly $-1$. This is the
  topological statement of "simple saddle" required by the index balance
  $\chi_{\text{core}} + \sum_i \chi_i = 1 - k$. It rejects composite strings
  (extrema present), degenerate multi-saddle sets ($\chi \le -2$), and neutral
  dipoles ($\chi = 0$).
- **Empirical A/B (gate off -> gate on)**:

  | field | k | cats off | cats on |
  |-------|---|----------|---------|
  | V (condensed) | 2.0 | 8 | **8** |
  | V (condensed) | 2.4 | 0 | 0 |
  | V (condensed) | 3.0 | 0 | 0 |
  | modified Willmore | 2.0 | 0 | 0 |
  | modified Willmore | 2.4 | 6 | **0** |
  | modified Willmore | 3.0 | 8 | **0** |

  Genuine catastrophes are untouched (all 8 monkey saddles on `V (condensed)`
  preserved, identical members). The $k=2.4$ drop on `V (condensed)` is
  **pre-existing** search-window geometry, not this gate - verified by A/B.
  Willmore E-SAD count is now stable across $k$: 20 / 26 / 24 / 24, with
  **0 strings swallowed** at every $k$.
- **Fixture lesson**: the real MC1 reaches $\chi = -1$ via a **multiplicity-2
  saddle** (12 Max + 2 Min, 14 saddles of summed multiplicity 15). A test
  fixture using 14 unit saddles gives $\chi = 0$ and silently tests the wrong
  branch. `total_saddle_index`, not `n_saddles`, drives `local_euler_index`.
- **Harness fix**: `verify_step6_toggles.py`'s standalone-pool assertion
  hard-coded the *old* unconditional claim rule on its right-hand side, so it
  failed once Entry 015's toggle-gated claim rule was in place. It now derives
  the pool from `catastrophe_claimed_cluster_ids`, keeping harness and viewer
  on one definition.
- **Tests**: `tests/test_flanking_saddle_gate.py` (9 tests) - predicate
  admit/reject branches plus three integration cases. Verified gate-sensitive:
  disabling the gate fails 2 of them. Suite: 62 passed.
- **Open next**: (b) quadrupolar prescreen for corridor recognition and
  fragment bridging (MC11+MC55); (c) wire `string_aggregator.py` in.

---

### [Entry 017] Corridor Bridging Into Reduction Units (`string_aggregator.py`, `sphere_viewer.py`)

- **Goal**: a ridge/valley corridor fragmented by the isotropic clustering
  radius must collapse to **one** effective saddle at its centre. Target case on
  the modified Willmore energy: MC32 + MC43 + MC34.
- **New abstraction**: `ReductionUnit` in `sphere_viewer.py` -- one set of CPs
  reduced to at most one effective CP, either a lone micro-cluster (`MC12`) or a
  bridged corridor (`MS1`). `build_reduction_units()` partitions clusters into
  units; `macro_string_to_cluster()` adapts a `MacroString` to the `MicroCluster`
  interface so `classify_cluster_polarity` and `harmonic_nudge` consume a
  corridor unchanged, with no parallel code path. The Step 6 loop iterates units.
- **Bridging rule (the substantive finding)**: centroid separation is the wrong
  metric. For elongated clusters it is dominated by the two half-lengths, so
  collinear fragments sitting tip-to-tip look far apart while long strings
  crossing at 60 degrees look close. `bridge_all` mode now links two clusters
  when the **uncovered gap** (`centroid_dist - semi_major_i - semi_major_j`) is
  within `k_bridge` delta_theta units **and** their principal axes are within
  30 degrees. Measured on Pd_20K modified Willmore at k_pitch=2.4:

  | pair | gap | axis angle | verdict |
  |------|-----|-----------|---------|
  | MC32-MC43 | 1.45 deg (0.9 dth) | 11.9 | merge |
  | MC43-MC34 | 1.45 deg | 11.9 | merge |
  | MC32-MC34 | 8.03 deg | 23.7 | not adjacent (MC43 is the bridge) |
  | MC21-MC22 | 13.09 deg | 90.0 | perpendicular, correctly separate |
  | MC17-MC19 (and 8 more) | -7.69 deg | 59.9 | crossing, must not merge |

  Gap alone would merge the 60-degree crossings; axis angle alone would not
  separate the collinear pair. **Both tests are required.**
- **Qualification moved to the pooled group**: a multi-cluster group becomes a
  corridor only when its own tangent-plane aspect ratio reaches
  `aspect_ratio_threshold`, which keeps isotropic blobs from fusing. A lone
  compact cluster is never promoted.
- **Result (Pd_20K, modified Willmore, k_pitch=2.4, k_bridge=3.0)**:
  54 clusters -> 50 units, 6 bridged.
  `MS1 ['MC32','MC43','MC34'] 21CP chi=-1 -> E-SAD`
  `MS2 ['MC33','MC44','MC35'] 21CP chi=-1 -> E-SAD` (the symmetry twin).
  Index sum preserved: clusters 2 == units 2.
- **Invariants kept**: catastrophe classification still runs on raw clusters, so
  detection is identical with aggregation on or off. The claim rule matches
  through `origin_ids`, so a corridor is claimed when any fragment is and
  aggregation cannot smuggle a catastrophe's index into a second glyph.
  `aggregate_strings` defaults to **off**, making the initial state byte-identical
  to the pre-aggregation pipeline.
- **Ordering**: units are sorted by (-size, earliest source cluster) to match
  `compute_micro_clusters`. Sorting by `unit_id` interleaves MC10 before MC2.
- **Test-fixture lesson**: a synthetic string whose members all sit on the
  centroid has a degenerate pooled PCA, so the aspect-ratio gate rejects every
  merge and the collinearity test passes vacuously. Members must be spread along
  the axis. Verified load-bearing by A/B: dropping `and collinear` fails
  `test_non_collinear_fragments_rejected_when_ar_qualifies`.
- **Cosmetic**: a one-constituent macro-string keeps its `MC` id (renaming it
  `MS3` implies a merge that never happened); pooled AR clamped at 999 for
  display, since collinear 3-point clusters report AR ~4e5.
- **Tests**: `tests/test_corridor_aggregation.py` (12 tests) -- bridge/no-bridge
  geometry, disjointness, index preservation, identity partition when off, id
  preservation, adapter contents, empty input. Suite: 74 passed.
- **Open next**: quadrupolar prescreen (2 Max longitudinal + 2 Min transverse) to
  tag corridors explicitly rather than inferring them from geometry alone.

## 2026-10-10 -- Local atom coordinates (off-origin atoms)

- **Symptom** (reported on `ethene3.plt`): boundary rings stood normal to the
  sphere surface instead of lying flat on it.
- **Root cause**: every angular formula in the package derives the outward normal
  as `p / ||p||` and rebuilds surface points as `dir * R`. That is only the
  normal when the sphere is centred on the world origin. `Pd_20K.plt` happens to
  sit at the origin, which hid the assumption for the whole project.
- **Measured** (`gba_topology2/inspect_atom_spheres.py`, read-only diagnostic):

  | file | atom | offset | R (aux) | R (fitted) | R naive | R inflation | δθ true | δθ naive |
  |---|---|---|---|---|---|---|---|---|
  | ethene3 | C1 | 1.2618 Å | 0.412733 | 0.412725 | 1.306666 | 3.17x | 2.2278° | 0.7036° |
  | Pd_20K | Pd1 | 0.000011 Å | 1.038471 | 1.038471 | 1.038471 | 1.00x | 1.6058° | 1.6058° |

  Aux `SphereRadius` agrees with the fitted radius to 0.002%, so the metadata is
  authoritative. The naive radius inflated R 3.17x and shrank δθ to 0.32x, so the
  cluster radius `k · δθ` was 3.2x too small -- CPs that belong together were
  split, and the ring generator built rings on a sphere about the origin, which
  is exactly the "ring normal to the surface" report.
- **Fix, applied once at the boundary**: `extract_atom_sphere_mesh()` now recovers
  the atom's own frame (`compute_atom_frame`: centre = vertex centroid, radius =
  mean centre-relative distance, overridden by the zone's aux `SphereRadius` when
  present), translates the mesh in place, and records `center`, `center_offset`,
  `fitted_radius` and `radius_source` in `atom_info`. Every algorithm keeps
  assuming a centred sphere, so `micro_cluster`, `catastrophe_classifier`,
  `excision_boundary`, `harmonic_nudge` and `polarity_detector` are untouched --
  no `sphere_center` parameter threaded through them.
- **Frame contract** (the part to remember):
  - **Local / centred**: `pts_np`, `CriticalPoint.position`, cluster and
    catastrophe centroids, boundary rings, ports, nudge positions, R, δθ.
  - **Global / molecular**: the UI dicts (`all_cps_list`, `clusters_list`,
    `catastrophes_list`), all VTK geometry, and the picker's `pick_pos`.
    The UI dicts feed the inspector text, the glyphs and the picker, so making
    them global satisfies all three consumers with one convention.
  - `runtime_cps` / `runtime_clusters` stay local, so `on_scene_click` converts
    the pick with `to_local` before measuring against them.
  - `centroid_unit` is a direction, so it is translation invariant -- unchanged.
- **`gba_topology2/src/sphere_frame.py`**: `compute_atom_frame`, `to_local`,
  `to_global`, `to_local_array`, `to_global_array`, `radial_offset`. The point of
  `radial_offset` is that scaling a centre-relative point is a genuine lift along
  the atom's normal, while scaling a world point is not -- the three ring
  z-fight factors (1.008 / 1.012 / 1.014) now run in the local frame. Ports
  previously had no lift at all and z-fought; they now get the 1.008 lift.
- **`analyze_mesh_geometry`** takes an optional `sphere_radius` override so the
  authoritative per-atom radius drives the pitch, matching the viewer.
- **Multi-atom note**: each analysed atom has its own `AtomSphereData` zone, so
  each gets its own centre, radius and δθ (C ~0.413 Å vs H ~0.28 Å). Selecting
  `-a N` resolves that atom's zone metadata; nothing is shared between atoms.
- **Verification**:
  - ethene3 C1, electron density condensed: χ = 2 valid (17 Max / 22 Min /
    37 Sad), 12 clusters, 7 multi-CP, Σχ = 2, ring scale `k·δθ` = 4.46° (was
    1.41°). Cluster angular diameters now 4-20°, i.e. physically sensible.
  - Pd_20K: `verify_step6_toggles.py` "ALL ASSERTIONS PASSED", χ = 2 in every
    toggle combination, first ECP position unchanged -- zero regression, as
    expected for an atom already at the origin.
  - Headless layout build passes for both files. Suite: 91 passed.
- **A/B discipline**: `test_naive_origin_radius_is_inflated_for_offset_atom`,
  `test_radial_offset_about_world_origin_is_the_bug`,
  `test_boundary_ring_from_world_coordinates_is_the_bug` and
  `test_pipeline_is_translation_invariant_after_centring` all assert the *old*
  behaviour is wrong, so reverting the fix fails the suite rather than passing
  silently. The measured buggy ring floats entirely outside the atom (all points
  ~2.2 R from the nucleus, height swinging 0.29 R around the loop) -- it does not
  pierce the sphere, which is why the symptom reads as "standing on edge".
- **UI Addition (num_contours slider)**: added an interactive slider (range 3 to 50,
  step 1) with dynamic count label in the "Condensed Field & Mesh" drawer card beneath
  the "Isocontour Lines" switch, allowing live tuning of the number of contour levels
  rendered on the atom surface.
- **Open next**:
  1. Glyph and tube radii (0.045, 0.065, 0.010 Å) are absolute and tuned for
     Pd's R ≈ 1.04 Å. On a 0.41 Å carbon sphere they are ~2.5x oversized.
     Recommend scaling by `R / R_reference` as its own atomic unit.
  2. The orientation-axes widget stays at the world origin while the atom renders
     offset, so it appears far from the sphere. Consider translating it to the
     nucleus.
  3. `polarity_detector.evaluate_spherical_radial_slope` still accepts a
     `sphere_radius` argument it never uses.

---

### [Entry 018] Step 1 Basin Locator: Basin Center Management & Interactive UI (`basin_center.py`, `sphere_viewer.py`)

- **Objective**: Implement Step 1 of the incremental Basin Locator pipeline to extract, structure, and manage basin centers (canonical raw extrema and reduced effective extrema) with an interactive 3D highlight halo and selection inspector in `sphere_viewer.py`.
- **Changes**:
  - `gba_topology2/src/basin_center.py`:
    - Created `BasinCenter` dataclass representing a basin center with fields: `center_id`, `polarity` (`"maximum"` vs `"minimum"`), `position`, `value`, `is_effective`, `vertex_id`, `origin_id`, `net_index`, and `member_count`.
    - Implemented `extract_basin_centers`: filters out hidden raw CPs that were absorbed into effective entities, ingests visible raw maxima and minima, ingests effective extrema (`E-MAX` and `E-MIN`), and returns a sorted list descending by scalar value.
    - Implemented `find_basin_center_by_id`: fast lookup helper by unique string ID.
  - `tests/test_basin_center.py`:
    - Unit tests covering `BasinCenter` validation, mixed raw + effective CP extraction, saddle/hidden exclusion, descending value ordering, and ID lookup.
  - `gba_topology2/src/sphere_viewer.py`:
    - Integrated `extract_basin_centers` into the topology update pipeline.
    - Added dedicated VTK gold wireframe sphere actor (`basin_center_actor`, color pure gold `(1.0, 0.84, 0.0)`, line width `3.5`) to highlight the active basin center.
    - Added drawer card **Basins: Step 1 Centers** with a candidate basin centers table displaying ID, effective badge, polarity chip (MAX in red, MIN in blue), and scalar value.
    - Added **Basin Center Inspector Card** displaying center origin/vertex details, polarity, exact coordinates in molecular frame, scalar value, and member counts.
    - Wired selection/deselection handlers (`select_basin_center`, `select_basin_center_from_list`, `clear_basin_center_selection`) and linked into `reset_defaults`.
- **Validation**:
  - Unit tests: 93 passed in `pytest`.
  - Format, lint, & typecheck: `ruff check`, `ruff format`, `mypy` passing cleanly with 0 errors across all 30 source files.
  - Pre-commit hook suite (`uv run pre-commit run --all-files`): all 4 hooks passed cleanly.

---

### [Entry 019] Basin Center Frame Fix: Off-Sphere Halos on Off-Origin Atoms (`basin_center.py`)

- **Symptom** (reported on `ethene3.plt`, C1): selecting a basin center from the
  "Basins: Step 1 Centers" table placed the gold halo a long way off the sphere.
- **Measured** (`gba_topology2/inspect_basin_centers.py`, read-only diagnostic
  reporting `||p - center|| / R`, which must be ~1.0 for a point on the sphere):

  | case | min d/R | median d/R | max d/R |
  |---|---|---|---|
  | raw centers, nucleus omitted (the bug) | 2.0622 | **2.6357** | 4.0617 |
  | raw centers, nucleus passed (fixed) | 0.9949 | **0.9972** | 1.0050 |
  | effective `E-MAX` center (world-frame dict) | - | 0.9973 | - |

- **Root cause**: the frame contract was broken at the extraction boundary.
  `CriticalPoint.position` is **centred-frame** (nucleus at the origin), while
  the `effective_cps` dicts are built by the viewer's UI layer and are already
  **world-frame**. `extract_basin_centers` emitted raw positions verbatim, so on
  an atom offset by 1.2618 A the raw centers were drawn relative to the world
  origin -- up to 4.06 R away. `Pd_20K.plt` sits at the origin, so the two
  frames coincide and the bug is invisible there (d/R = 1.0000 before and after).
- **Fix**: `extract_basin_centers(..., center=None)` maps raw CP positions with
  `sphere_frame.to_global` and returns **every** center in the world frame,
  matching the existing convention for `all_cps_list` / `clusters_list` /
  `catastrophes_list`. `center=None` keeps the centred behaviour for
  origin-centred synthetic meshes. The `BasinCenter.position` docstring now
  states the frame explicitly.
- **Note on the report**: the measured defect is on the **raw** `MAX_*` /
  `MIN_*` rows. The `E-MAX_MC*` rows were already world-frame and verified
  on-sphere (d/R = 0.9973), because `effective_cp_to_ui` receives `center`.
- **Tests**: `tests/test_basin_center.py` +2 tests -- raw centers mapped out of
  the centred frame (exact world coordinates + on-sphere distance, and the
  pre-fix pass-through pinned as the origin-centred-only case), and the mixed
  case where a world-frame effective dict and centred-frame raw CPs must land on
  the same sphere. Suite: 95 passed. `ruff` / `mypy` clean.

---

### [Entry 020] Step 2 Basin Locator: Saddle Launch Vectors & UI Visual Deliverable (`saddle_launch.py`, `sphere_viewer.py`)

- **Objective**: Implement Step 2 of the incremental Basin Locator pipeline to extract separatrix launch branches from saddle critical points and render directional ascent/descent launch rays on the sphere in `sphere_viewer.py`.
- **Mathematical Approach**:
  - Rejected the standard surface Hessian approach because higher-order saddles (e.g. $k=3$ monkey saddles where $f \sim x^3 - 3xy^2$) have degenerate second derivatives where Hessian eigenvectors are ill-defined.
  - Implemented the cyclic-link sector scan: orders the 1-ring neighbours cyclically around vertex $v$, locates continuous sign crossings where $f(u) - f(v) = 0$, partitions the cycle into $2k$ sectors ($k$ ascent/ridge sectors and $k$ descent/valley sectors), and computes sector bisectors on the sphere tangent plane.
  - Applies geodesic exponential map stepping: $\vec{p}_{\text{launch}} = R \cdot (\cos\theta\,\hat{n} + \sin\theta\,\hat{v})$ to place launch seed coordinates strictly on the sphere surface.
  - Verifies sector count against discrete Morse multiplicity ($\mu = k - 1$) and surfaces any index mismatch.
- **Changes**:
  - `gba_topology2/src/saddle_launch.py`:
    - Created `LaunchBranch` and `SaddleLaunchSeed` dataclasses.
    - Implemented `build_cyclic_link`, `extract_saddle_launch_seed`, and `compute_all_saddle_launch_seeds`.
  - `tests/test_saddle_launch.py`:
    - 11 unit tests covering cyclic link cycle validation, simple saddle 2-ascent/2-descent branches, $k=3$ monkey saddle 3-ascent/3-descent branches ($120^\circ$ orthogonality on tangent plane), on-sphere geodesic radii, normal orthogonality, scale factor, and non-saddle/error handling.
  - `gba_topology2/src/sphere_viewer.py`:
    - Added VTK pipelines for launch rays: Orange-Red tubes for ascent (towards maxima) and Cyan tubes for descent (towards minima).
    - Added drawer card **Basins: Step 2 Launch Seeds** with toggle switch, launch offset slider (`launch_seed_pitch`), branch count summary chips (Ascent, Descent, Max $k$), and index mismatch warnings.
    - Added `show_launch_seeds` and `launch_seed_pitch` to `DEFAULT_UI_STATE` and reactive state handler.
- **Validation**:
  - Unit tests: 106 passed in `pytest`.
  - Formatting, linting, and type safety: `ruff check`, `ruff format`, and `mypy` passed with 0 errors across 33 source files.
  - Pre-commit hook suite passed cleanly.

---

### [Entry 021] Directional Support Radius for Corridor Bridging (`string_aggregator.py`)

- **Symptom**: on the curvedness field of `Pd_20K` at `k_pitch=1.6`, MC18 (21 CPs,
  chi = -1, AR = 14.4) reduced correctly to a single effective saddle with
  bridging off. With bridging on it absorbed MC131, a **lone minimum sitting at
  90 degrees to the corridor axis and visibly displaced from it**, giving
  MS12 = [MC18, MC131], 22 CPs, chi = 0 -> annihilation. A genuine saddle
  vanished and its ring turned ghost grey.
- **Root cause**: the `bridge_all` gap test subtracted each cluster's scalar
  `semi_major_rad` from the centroid separation. That subtraction is
  **isotropic**, so a long thin corridor is treated as a disk of radius `a`
  rather than the ellipse it actually fitted. Measured on the target pair:

  | quantity | value |
  |---|---|
  | MC18 semi-major / semi-minor | 16.76 deg / 2.77 deg |
  | centroid separation MC18-MC131 | 17.53 deg (10.91 dth) |
  | bearing of MC131 from MC18 axis | **90.0 deg** |
  | old isotropic gap | **-0.46 dth** (limit 3.0) -> linked |
  | new directional gap | **8.26 dth** -> rejected |

  The axis-collinearity test could not help: `_axis_pairwise_angle` returns
  `None` when either cluster is compact (a point has no meaningful axis), so
  `collinear` was `True` vacuously. The pooled aspect-ratio gate also passed,
  since one extra point barely dents the AR of a 21-CP string. The gap test was
  the only gate in play, and it was measuring the wrong shape.
- **Fix**: `cluster_directional_support_radius(cluster, target_unit)` evaluates
  the polar form of the cluster's fitted boundary,
  $r(\phi) = \sqrt{(a\cos\phi)^2 + (b\sin\phi)^2}$, at the bearing of the
  neighbour measured on the tangent plane at the cluster centroid, with
  $\phi$ folded to [0, 90] degrees because the principal axis is an undirected
  line. The gap becomes
  $\text{gap} = \theta_{ij} - r_i(\phi_i) - r_j(\phi_j)$.
  Full length along the axis, narrow width sideways. A compact cluster has
  $a == b$ and returns $a$ unchanged, so isotropic fragments are untouched.
  Falls back to the semi-major on ambiguous geometry (target at the centroid or
  antipodal, null axis) -- the conservative choice, never shrinking reach on
  geometry the code cannot resolve.
- **Why this form**: it reuses the exact polar expression
  `generate_elliptical_boundary_ring` already uses to draw the ring, so the
  reach the bridging test grants is the boundary the user sees on screen. No new
  parameter, no new threshold, symmetric in the two clusters.
- **Result (Pd_20K curvedness, k_pitch=1.6, k_bridge=3.0)**:

  | rule | MC18 unit | units | bridged | sum chi |
  |---|---|---|---|---|
  | old isotropic | MS12 [MC18, MC131] 22CP chi=0 | 96 | 94 | 2 |
  | new directional | MC18 21CP chi=-1 | 98 | 90 | 2 |

  The saddle survives; MC131 stays a standalone +1. Index sum preserved (2 == 2).
- **Lesson on A/B harnesses**: the first A/B run on the real field printed
  nothing and was read as "the bug does not reproduce". The field array name
  contains a non-ASCII prefix (`Ï curvedness (condensed)`), the lookup raised
  `AttributeError` inside `numpy_support.vtk_to_numpy`, and piping through
  `grep` swallowed the traceback. A/B runs must assert on a positive marker,
  not on the absence of output. Passing the field name through a shell `-c`
  string also mangles it; use a script file.
- **Tests**: `tests/test_corridor_aggregation.py` -- 8 new tests: support radius
  along axis, transverse, axis-sign folding, 45-degree polar value,
  unnormalised target, compact isotropy, degenerate fallbacks, plus the two
  end-to-end cases (perpendicular compact fragment rejected, on-axis tip
  fragment still bridged). The perpendicular test asserts the old isotropic gap
  *would* have linked, pinning the regression. Suite: 115 passed.
- **Open next**: the collinearity skip for compact clusters is still a
  deliberate blind spot -- a compact fragment at the *tip* of a corridor joins
  from any direction, which is correct, and one sitting inside the corridor's
  own bounding ellipse joins too. The quadrupolar prescreen (2 Max longitudinal
  + 2 Min transverse) from Entry 017 remains the type-aware route to corridors.

---


### [Entry 022] "Save Parameters" Diagnostic JSON Export (`sphere_viewer.py`)
- **Objective**: One-click capture of the full viewer GUI state so issues
  observed on screen can be reproduced in headless diagnostic scripts without
  manually re-specifying every toggle, slider and field selection.
- **Changes**:
  - `save_params_to_dict(state_like, *, plt_path, atom_info, atom_num)` --
    pure, duck-typed snapshot builder. Emits `schema_version` (=1), `tool`,
    ISO-8601 UTC timestamp, `params` split into `bools` (16 switches),
    `floats` (8 sliders, ints widened), `strings` (`selected_field`), plus
    `context` (abs .plt path, atom, discovered `field_items`),
    `derived_metrics` (geom_info, morse_counts, display_counts,
    reduction_stats, launch_stats -- deep-copied via json round-trip) and
    `selection` (CP/cluster/catastrophe/basin-center ids only).
  - `load_params_from_dict(payload)` -- validates schema_version, requires
    every control key, coerces strictly (bools reject ints, sliders reject
    bools); returns a flat dict safe for `state.update()` plus passthrough
    diagnostic sections. Metrics are outputs, never inputs.
  - `default_params_path(plt_path, atom_num)` -- `<stem>_atom<N>_params.json`
    next to the .plt.
  - Type tables `PARAM_BOOL_KEYS`/`PARAM_FLOAT_KEYS`/`PARAM_STRING_KEYS` are
    derived from `DEFAULT_UI_STATE` at import and checked to partition it
    exactly, so a new control is covered by construction.
  - UI: toolbar "Save Parameters" button -> `@ctrl.add("save_parameters")`
    writes the JSON (`indent=2, sort_keys=True`), prints the abs path, and
    shows a `params_last_saved_path` confirmation chip in the drawer. That
    key is deliberately outside `DEFAULT_UI_STATE` and the `@state.change`
    decorator: it triggers no recompute and survives "Reset to Defaults".
- **Verification**: `gba_topology2/verify_params_export.py` fires the export
  headlessly on ethene4.plt atom 1 at default values (995 nodes, chi=2, 8 Max
  / 6 Min / 12 Sad); the written JSON round-trips through
  `load_params_from_dict` back to all 24 controls + `selected_field`.
  Generated `*_params.json` files are gitignored.
- **Tests**: `tests/test_sphere_viewer_params_export.py` -- 14 tests: control
  coverage parsed from the `@state.change` decorator source (same guard as
  Entry 020's defaults test), widget-type fidelity, json round-trip, context/
  metric/selection capture, metric deep-copy, missing-key and type-mismatch
  rejection on both export and load, int-widening, path naming, table
  partition. Suite: 129 passed.

---

### [Entry 023] `diagnose_params.py`: Headless Stage Runner over Saved Params
- **Objective**: Turn a "Save Parameters" JSON (Entry 022) into a reproducible
  headless run: re-execute every analysis subroutine the GUI controls trigger,
  in the viewer's own order, and answer "does this parameter set reproduce the
  observation?" with an exit code.
- **Changes**: `gba_topology2/diagnose_params.py`.
  - `RunContext` dataclass carries the mesh (pts/triangles/f_vals, centred
    frame), the loaded control dict, and every stage's accumulated results.
  - Nine stage functions in `update_topology()` order: geometry, morse,
    clusters, units, catastrophes, ports, reduction, basins, launch -- each
    with the viewer's exact arguments (tie_break 1e-15, margin 1.5 deg, 48
    ring points, catastrophes on raw clusters not bridged units). `--only
    <stage>` stops after the named stage; consumers guard with
    `require_morse()` so a hand-built context fails loudly.
  - `stage_reduction` replicates the viewer's inline Step 6 block
    (catastrophe collapse, polarity/index decision, annulment/irreducible,
    nudge, display counts) -- the same duplication `verify_step6_toggles.py`
    carries. The clean fix is extracting a shared `run_effective_reduction()`
    subroutine; deferred as its own atomic unit.
  - `diff_metrics` walks the saved `derived_metrics` leaf-wise (float
    tolerance, strict bools, element-wise list leaves) and skips sections the
    run did not reach, so `--only` stops stay comparable. Exit 1 on any drift.
  - `--set key=value` applies typed overrides (bools true/false, floats, the
    field selector) for parameter sweeps; `--out file.json` dumps
    per-entity results (clusters, catastrophes, effective CPs, basin centers,
    launch seeds) for programmatic A/B diffing. `--plt/--atom/--field`
    override the recorded context when the .plt moved.
- **Verification**: on the live-saved `ethene4_atom1_params.json` (viewer
  session, atom C1, Electron Density): full run reproduces **35/35 saved
  metric leaves** (geom, morse 6/8/12 chi=2, 12 clusters, 1 split k=3
  catastrophe 3V+3R, 14 basin centers, 12 saddles -> 24+24 branches), exit 0.
  `--only clusters` compares 14/14; `--set reduce_multi_clusters=true`
  correctly reports 10 drift leaves and exits 1.
- **Tests**: `tests/test_diagnose_params.py` -- 24 tests: leaf comparison
  semantics (bool strictness, tolerance, shape mismatch), recursive diff
  (drift isolation, missing-section skip, list leaves), typed overrides,
  stage order, early stop, morse-first guard, and an end-to-end quadrupole
  octahedron run (2 Max/2 Min/2 Sad, chi=2 preserved through reduction and
  launch). Suite: 154 passed.

---

### [Entry 024] Troubleshooting Protocol in AGENTS.md; Fix Displayed-Count Bug
- **Objective**: Make "reproduce in the viewer, export params, point the agent
  at the JSON" the standard troubleshooting entry point -- and let the
  reproduce check earn its keep by fixing a real runner bug it caught.
- **Changes**:
  - `gba_topology2/AGENTS.md` gained a mandatory **Section 3: Issue
    Troubleshooting Workflow**: the four-step protocol (user reproduces ->
    clicks "Save Parameters" -> describes the issue and points at the JSON ->
    agent investigates with `diagnose_params.py`), the CLI surface with
    worked examples, the stage list, the exit-code contract, and the
    expectation that the runner itself may be extended to gather the
    information a given investigation needs. It also states that a
    reproduce-check mismatch is itself a finding, and that display-only
    controls are inert headlessly.
  - `gba_topology2/diagnose_params.py` -- **bug fix**: `stage_reduction`
    computed the displayed maxima/minima/saddles counts from the visible raw
    CPs only. The viewer's glyph bookkeeping also adds one glyph per
    effective CP (E-MAX -> maxima, E-SAD -> saddles), so with the reduction
    toggles on the runner under-reported both the counts and the index sum.
    The counts are now built exactly as the viewer builds them.
- **How the bug was found**: the user re-saved `ethene4_atom1_params.json`
  from a live viewer session with `reduce_catastrophes` and
  `reduce_multi_clusters` enabled and `cat_barrier_tol = 0.001`. The runner
  reported 2 drifting leaves (`display_counts.maxima` 2->0,
  `display_counts.saddles` 3->1) on a *reproduction* run -- the reproduce
  check flagged the runner, not the data. This is the workflow working as
  intended: the same JSON that reproduces cleanly under the old default
  toggles exposed a divergence the earlier, all-defaults run could not see.
- **Verification**: after the fix, the same saved file reproduces **35/35**
  leaves with exit 0 (reduction on: 4 effective CPs, 21 raw CPs hidden).
  The all-defaults case still reproduces 35/35.
- **Tests**: two new regression guards in `tests/test_diagnose_params.py` --
  displayed counts equal raw-visible + effective-CP glyphs, and the index
  sum stays 2 across the collapse; plus the reduction-off case asserting the
  raw-only counts. Suite: 156 passed.



---

### [Entry 025] Fix: Split Catastrophes Were Collapsed, Hiding Real Extrema
- **Objective**: Resolve the user-reported defect on `ethene4_atom1_params.json`
  (atom C1, Electron Density): a clear maximum had **no maximum glyph**, a
  spurious saddle glyph appeared nearby, and clicking where the maximum should
  be selected MC3 ("2 max, 1 sad, net index +1"), which did not match what was
  drawn.
- **Diagnosis** (via `diagnose_params.py`, per the Section 3 protocol):
  - Full run reproduced the saved observation **35/35, exit 0** — the runner
    and the viewer agreed, so the defect was in the pipeline, not in the
    bookkeeping.
  - `--only catastrophes` showed **1 split** constellation (`CAT_1`,
    `physical_symmetry_breaking`), i.e. `is_fused == False`.
  - `--out` per-entity dump showed `CAT_1`'s constituents were
    **`MC3, MC12, MC11, MC10`** — i.e. **MC3 was absorbed by the catastrophe**.
  - `--set reduce_catastrophes=false` was the decisive bisect: displayed
    counts flipped `maxima 2 -> 3`, `saddles 3 -> 5`, and
    `n_catastrophes_reduced 1 -> 0`.
- **Root cause (two compounding defects)**:
  1. **The spurious/physical distinction was computed and then discarded.**
     `classify_catastrophes` classifies a constellation as `spurious_unfolding`
     (numerical artefact, safe to fuse) or `physical_symmetry_breaking`
     (genuine, distinct poles), and its docstring states a split constellation
     is *"Preserved as distinct physical simple saddles"*. But the viewer's
     Step 6 block collapsed **every** catastrophe into one `E-SAD` at the
     constellation centroid, ignoring `is_fused` entirely. `is_fused` was
     computed, exported, and displayed in the sidebar table — and never acted
     upon.
  2. **CAT_1 is a false-positive constellation.** `classify_catastrophes`
     accepts a constellation on *index arithmetic alone*
     (`chi_core + k*(-1) == 1-k`) with no geometric validation of the k-fold
     pattern. CAT_1's three "flanking saddles" lie **79.7° / 83.5° / 84.7°**
     from the core — across the sphere, not ringing it. The classifier
     computes the azimuthal symmetry deviation σ (43.09° against an ideal
     120° spacing) inside `evaluate_constellation_metrics` and then **discards
     it** (`_ang_std`).
- **Why the maximum vanished**: the core cluster MC3 is a *composite* set
  (2 Max + 1 Sad, χ=+1) that holds the field's **global maximum**
  (v = 0.0648, twice the next highest). Collapsing CAT_1 hid all six of its
  members and replaced them with one synthetic `E-SAD` at a centroid that is
  **not a critical point at all** — the "spurious saddle glyph". The maximum
  glyph was gone because it was a member of a hidden set. Clicking the empty
  spot selected MC3 because MC3's centroid is the nearest cluster, which is
  exactly the mismatch the user reported.
- **Fix**: added `fusable_catastrophes(catastrophes)` to `sphere_viewer.py`,
  returning only constellations with `is_fused == True`. The viewer's Step 6
  collapse now gates on `reduce_cats and cat.is_fused`, and
  `catastrophe_claimed_cluster_ids` claims only fusable constellations'
  constituents (so a split constellation's clusters stay eligible for
  standalone reduction instead of being stranded). Mirrored in
  `diagnose_params.py` (`stage_reduction`) and `verify_step6_toggles.py` so the
  three implementations cannot drift.
- **Result**: `E-SAD_CAT_1` is gone; MC3 now reduces to **`E-MAX_MC3`** (χ=+1),
  so the global maximum is visible again. Displayed counts `3 Max / 4 Min /
  5 Sad`, and the displayed **χ stays 2** — the index invariant holds.
- **Runner instrumentation**: the `catastrophes` stage now reports
  `n_fusable` alongside `n_fused`/`n_split`, so a future investigation can see
  at a glance how many constellations the reduction stage may collapse.
- **Open follow-up (not fixed here)**: CAT_1 is still *detected* as a
  catastrophe — it is simply no longer collapsed. The underlying false
  positive remains in `classify_catastrophes`, which accepts a constellation
  on index arithmetic without validating the k-fold geometry and discards the
  azimuthal symmetry metric it already computes. A symmetry/geometry gate on
  detection (reject σ far above the ideal 360/k spacing) is the natural next
  unit of work. Empirically, genuine k=4 constellations on `ethene4` show
  σ ≈ 2.5–14.8°, while the false positive shows 43.1°; note that `ethene3`
  (mesh pitch 2.23°) produces legitimate constellations with σ up to 98.8°,
  so the gate must be scaled by δθ_mesh, not fixed.
- **Tests**: 4 regression tests in `tests/test_sphere_viewer_reduction.py` —
  only fused catastrophes are fusable; an all-split set yields none; a split
  constellation does not claim its clusters; a fused one still does. Suite:
  160 passed.

---

### [Entry 020] Non-Monotonic "Bridge Radius" Response (ethene4 C1, curvedness)

- **Symptom (reported)**: raising the "Bridge Radius" slider (`k_bridge`)
  from 3.0 to 3.25 *increased* the number of visible effective critical
  points (2 → 4) instead of decreasing it. The expectation is that a wider
  bridge merges more fragments, so fewer effective CPs should survive.
- **Reproduction**: two saved parameter snapshots
  (`ethene4_atom1_params a.json` at k_bridge=3.0 and
  `ethene4_atom1_params b.json` at k_bridge=3.25). Both reproduce exactly
  with `diagnose_params.py` (35/35 match), so the divergence is in the
  pipeline, not in the runner.
- **The bridging rule is two-stage**:
  1. A graph edge is created between two micro-clusters when the uncovered
     gap is within `k_bridge`·δθ_mesh **and** their principal axes are within
     30°.
  2. Each connected component of that graph is gated by its **pooled**
     tangent-plane aspect ratio (≥ 2.0). A group failing the gate is
     rejected **whole** and its members revert to standalone units.
- **Root cause**: the stage-2 gate is a *pooled* property and therefore **not
  decomposable**. One off-axis satellite joining a valid corridor widens the
  pooled minor axis enough to sink the whole group below the threshold — and
  rejecting the group annihilates the corridor as well.

  | k_bridge | component | pooled AR | gate | result |
  |---|---|---|---|---|
  | 3.00 (thr 21.26°) | MC1+MC2+MC3 (17 CP) | **2.477** | pass | one corridor `E-MIN_MS1` |
  | 3.25 (thr 23.03°) | MC1+MC2+MC3+MC13+MC14 (19 CP) | **1.543** | fail | corridor destroyed |

  The two extra links are MC2–MC13 (gap 21.55°) and MC2–MC14 (gap 21.57°),
  both just inside the 23.03° threshold. MC13/MC14 are lone saddles hanging
  off MC2 at a bearing roughly perpendicular to the corridor; they widen the
  pooled minor extent from 13.7° to 45.3° and collapse the aspect ratio.
- **Verdict: this was incorrect behaviour, not a design choice.** Widening the
  bridge radius must not destroy structure a narrower radius had already
  resolved. The invariant that was *not* broken is the displayed Euler
  characteristic, which stayed at 2 for every k_bridge — only the *number* of
  glyphs was wrong.
- **Fix — make the gate decomposable** (`string_aggregator.py`): new
  subroutine `select_elongated_core()`. When a component fails the pooled
  gate, the member whose removal most restores the anisotropy is peeled off
  as a **satellite** and the remainder is retested, so the largest
  still-qualifying sub-corridor survives. A peel is only accepted when the
  remainder stays **connected** in the bridging graph, so a genuine corridor
  is never split into two halves that each look elongated. The peeled
  satellites return as standalone clusters, so their index is preserved and
  they remain eligible for standalone reduction.
- **Latent degeneracy found and fixed on the way**: a group of fewer than
  three points has a rank-deficient tangent-plane covariance, so its minor
  eigenvalue is zero and the aspect ratio is a division-by-zero artefact
  (observed: 7.1e7) rather than a measurement. Such a group trivially
  "qualified" as a corridor. `is_degenerate_group()` now marks it, and a
  degenerate group never qualifies on the pooled ratio.
- **New subroutines** (`string_aggregator.py`):
  - `compute_bridge_links()` — pure, per-pair verdicts with a human-readable
    `reason`; the aggregator now consumes its `is_linked` flags instead of
    re-implementing the rule inline.
  - `compute_group_geometry()` — pooled centroid, principal axis and aspect
    ratio of a set of clusters.
  - `select_elongated_core()` — the decomposable gate.
  - `is_degenerate_group()` — the <3-point shape guard.
- **Runner instrumentation** (`diagnose_params.py`): new `bridge` stage,
  placed between `clusters` and `units`, reporting the per-pair verdicts and
  the per-component gate outcome (pooled AR, whether it passed, the core and
  the satellites). The per-entity dump now carries `bridge_links` and
  `bridge_gate`. This is what makes the mechanism auditable:
  `--only bridge --out results.json` shows exactly which link widened the
  component and which member was peeled.
- **Result**: the response is now monotone. Displayed glyphs over
  k_bridge = 2.0 → 6.0: 12, 10, 10, 10, 10, 10, 9, 9, 8, 8 (non-increasing),
  with the displayed χ = 2 throughout. k_bridge = 3.25 now reproduces the
  k_bridge = 3.0 corridor instead of dissolving it. The saved snapshot for
  k_bridge = 3.0 still reproduces 35/35; the 3.25 snapshot now drifts by
  design, because it recorded the buggy behaviour.
- **Tests**: `tests/test_bridge_decomposition.py` (16 tests) — the link
  evaluator's verdicts and reasons, the group geometry, the core selection
  (peeling, index preservation, connectivity, blob rejection), and four
  regression tests on the real `ethene4.plt` data pinning the dilution, the
  peel, the monotone response and the index sum. Suite: 176 passed.
- **Open follow-up**: the pooled-AR threshold (2.0) is still hardcoded and has
  no viewer control, so the "Bridge Radius" slider is not the only thing that
  governs whether a corridor forms. Exposing it — or deriving it from δθ_mesh
  the way the other tolerances are — would make the corridor decision
  auditable from the UI alone.

### [Entry 026] "Load Parameters": Snapshot Restore, Camera Persistence & Runtime Reload
- **Objective**: Close the loop on the diagnostic snapshot that Entry 022
  introduced. Saving a snapshot was write-only: it could be read by the
  headless runner, but the viewer could not read it back. This entry adds
  the reverse direction -- a "LOAD PARAMETERS" button next to "SAVE
  PARAMETERS", a `--params` command-line flag, and the ability to restore
  the *file*, the *atom*, the *field*, the controls and the **camera**.
- **Why the camera matters**: the snapshot is meant to reproduce an
  observation. Reproducing an observation of a 3D scene means reproducing
  the *view*, not just the numbers: which way the camera looked and how far
  it was zoomed in are part of what the operator saw. Without them, a
  restored snapshot shows the right topology from an arbitrary viewpoint,
  which is not the same observation.
- **New subroutine** (`camera_state.py`) -- owns the serialisable form of
  the camera:
  - `capture_camera_state()` reads a `vtkCamera` into an immutable
    `CameraState` (position, focal point, view up, projection mode,
    parallel scale, view angle).
  - `camera_section_to_dict()` writes it to JSON *together with the nucleus
    it was captured around* (`atom_center`).
  - `camera_state_from_dict()` reads it back with strict validation: every
    field required, types checked, zoom and field-of-view range-checked,
    non-finite values rejected.
  - `apply_camera_state()` writes a snapshot back onto a camera.
  - `rebase_camera_state()` translates a snapshot from one nucleus to
    another, moving the position and the focal point while leaving the
    viewing direction, the roll and the zoom untouched.
- **Why the camera is stored relative to the nucleus**: the camera is
  captured in world coordinates, but the viewer draws in the world frame
  while the algorithms work in the atom's own frame. Recording the
  reference nucleus makes the camera *portable*: restoring a snapshot
  against a different atom (or a different file whose atom sits elsewhere
  in the molecule) moves the camera by the same offset as the nucleus, so
  it still looks **at** the atom instead of at empty space beside it.
  Without the reference point, a camera restored onto another atom would
  silently point at nothing.
- **New module** (`params_schema.py`) -- the snapshot format, extracted
  from `sphere_viewer.py` so the viewer, the headless runner
  (`diagnose_params.py`) and the tests share **one** definition instead of
  three copies. The `camera` section is **optional**: it was added without
  a schema-version bump because files written before it still load (the
  section is simply absent), and files with it are recognised by the
  presence of the key. The two sample snapshots already in the repository
  were written at schema 1 and still load unchanged.
- **New module** (`params_loader.py`) -- turns a loaded snapshot into an
  explicit, inert `RestorePlan`: which `.plt` to open, which atom to
  isolate, which field to select, the control values to push into the
  state, and the camera to apply. The plan resolves *what* to restore; the
  viewer decides *how*. That split keeps the interesting logic (path
  resolution, field fallback, camera re-basing) testable without a browser
  or a render window.
- **Runtime reload** (`sphere_viewer.py`): the mesh-bound state used to be
  a dozen closure variables captured at startup, so a different file could
  not be opened without restarting. Those variables are now grouped in a
  `ViewerSession` dataclass, and the pipeline reads from it. A reload
  replaces the session's contents in place, re-points the flood/contour/
  wireframe mappers at the new mesh, and rebuilds the pipeline. The
  re-entrancy latch (`suppress_param_change`) is reused so the batched
  state write rebuilds the pipeline exactly once, not once per key.
- **The LOAD button**: a toolbar button opens a dialog with a file input
  (`.json`). The file is picked in the browser and uploaded to the server,
  so a snapshot can be restored without the server knowing where it lives
  on disk. The dialog reports failures in an alert and in a drawer chip
  rather than raising into the trame event loop, and the loaded file name
  is shown in a confirmation chip.
- **`--params` CLI flag**: `uv run python gba_topology2/src/sphere_viewer.py
  <file>.plt --params <snapshot>.json` restores a snapshot at startup. The
  snapshot wins over the positional arguments, because it is the more
  specific description of the state to reproduce: it may name its own
  `.plt`, atom and field.
- **Camera capture source**: trame mirrors the browser's camera onto the
  server-side camera on every interaction, so the server's `vtkCamera` is a
  faithful mirror of what the operator is looking at. Capturing it is
  therefore a read of the authoritative camera, not a guess.
- **Tests** (50 new, 226 total, all passing):
  - `tests/test_camera_state.py` (18) -- capture, serialisation, strict
    validation, application, re-basing, and the property that actually
    matters: the restored camera still looks at the atom.
  - `tests/test_params_loader.py` (19) -- the camera in the JSON, the
    optional section, the plan's file/atom/field/control resolution, the
    field fallback, and the end-to-end round trip on real data.
  - `tests/test_sphere_viewer_params_load.py` (13) -- the session, the
    uploaded-file decoding, the CLI flag, and the UI wiring.
- **Open follow-up**: the dialog's file input uploads the snapshot to the
  server, but a snapshot that names a *different* `.plt` still requires
  that `.plt` to exist on the server's filesystem. Uploading the `.plt`
  itself is the natural next step, and would need the mesh extraction to
  accept an in-memory buffer rather than a path.




### [Entry 027] Pair Scale Distribution Diagnostic (`pair_scales.py`, `inspect_pair_scales.py`)

- **Goal**: replace guessed thresholds with measured ones. The pipeline stacks
  three length scales -- micro-clustering ($k_{\text{pitch}}$), bridging
  ($k_{\text{bridge}}$), and whatever macro-clustering range basins need -- and
  all three are currently set by eye. The question is whether the fields
  actually separate these scales, i.e. whether the distribution of pairwise
  separations between consolidated entities has a trough between the meso peak
  (fragments of one feature) and the macro peak (distinct basins). If it does,
  the bottom of the trough is the objective cutoff.
- **New modules**:
  - `gba_topology2/src/pair_scales.py` -- pure math, no viewer/VTK/Trame import.
    `PairwiseClearance` records, `compute_pairwise_clearances`, `histogram_bins`,
    `render_ascii_histogram`, `render_intermediate_registry`,
    `write_svg_histogram`. Consumes the `MicroCluster` interface, so
    `ReductionUnit.as_cluster` is measured exactly like a lone cluster.
  - `gba_topology2/inspect_pair_scales.py` -- CLI. Per field: Morse CPs ->
    micro-clusters -> bridged units -> **measure** -> ASCII histograms +
    intermediate registry (+ optional SVG).
- **Design decisions**:
  - **Measure before catastrophe reduction.** Catastrophe classification is an
    interpretation step; measuring after it would report the shape the
    constellation templates imposed, not the shape of the field. It also keeps
    the diagnostic aligned with `classify_catastrophes`, which itself consumes
    raw clusters.
  - **Two measures, not one.** Centroid separation is dominated by the
    half-lengths of elongated entities, so tip-to-tip fragments look far apart
    and crossing corridors look close. The directional edge clearance
    $\theta_{ij} - r_i(\phi_i) - r_j(\phi_j)$, reusing
    `cluster_directional_support_radius` from Entry 021, puts touching entities
    at zero at any orientation and sharpens the trough.
  - **Normalise by $\Delta\theta_{\text{mesh}}$.** Makes Pd ($R = 1.04$ A) and
    ethene C ($R = 0.41$ A) share one axis, so fields and datasets plot together.
  - **Zero-dependency SVG.** matplotlib is not a project dependency; the chart
    is emitted as standalone XML so it opens in VS Code beside the viewer.
- **Result -- intermediate pairs in the 2.5-6.0 dtheta transition zone**:

  | field | ethene3 C1 | Pd_20K Pd1 |
  |---|---|---|
  | Electron Density | **empty** | **empty** |
  | V (volume) | 2 | 56 |
  | shape index | **empty** | 72 |
  | curvedness | 8 | 4 |
  | Willmore energy | **empty** | 176 |
  | modified Willmore | 13 | 19 |

- **Findings**:
  1. **Density is the cleanest field on both datasets** -- zero intermediate
     pairs, so its entities are strictly near or strictly far. If basin
     identification wants a hard-coded cutoff, density is where it is safest.
  2. **The transition zone is field-specific, not a constant.** Willmore on Pd
     has 176 pairs there; curvedness on the same atom has 4. A single global
     macro-clustering radius cannot be correct for all fields, which argues for
     per-field (or per-field-class) parameters rather than one slider.
  3. **The ethene3 curvedness registry is entirely compensating dipoles** -- all
     8 pairs have $\chi_{\text{sum}} = 0$, and they fall into four
     symmetry-related pairs of two (MC7 and MC8 each facing MC22 at 3.61
     dtheta; MC10 and MC11 each facing MC20 at 4.44). These are the pairs a
     macro rule must decide on, and their index balance says merging them would
     annihilate.
  4. **The directional measure beats the centroid measure as designed.** On
     ethene3 curvedness the centroid distribution is empty across the whole
     2.5-5.5 band while the clearance distribution shows structure at 3.5-6.0 --
     the structure bridging already acts on. The centroid view would have
     reported "clean separation" and hidden the real decisions.
- **Rendering lesson**: scaling bars to the global peak made the charts
  unreadable. On a sphere most pairs are distant, so the overflow bin (164 of
  190 on ethene3 curvedness) dwarfs everything and flattens the meso trough --
  the one feature the chart exists to show. Bars now scale to the peak of the
  **finite** bins, with the overflow bin clamped and labelled off-scale.
- **Tooling lesson**: the volume field is stored as plain `V`, so the natural
  pattern `volume` matches nothing and the short pattern `v` matches
  "cur**v**edness" and "distance". `_field_matches` treats single-letter
  patterns as exact base names (name minus `(condensed)` and non-ASCII prefix)
  and longer patterns as substrings.
- **Tests**: `tests/test_pair_scales.py` -- 26 tests. Collinear pair clearance
  equals the analytic tip gap (0.015 rad); perpendicular pair measures the
  transverse width (0.32 rad) where the isotropic measure reports 0.03;
  crossing corridors report negative clearance; $\binom{N}{2}$ pair count with
  no self-pairs; bin conservation including the negative underflow bin; SVG
  parses as XML and escapes non-ASCII field names. Two tests failed while the
  code was wrong (empty-meso marker, overflow scaling), confirming both are
  load-bearing. Suite: 141 passed.
- **Open next**: (a) run the diagnostic across the remaining ethene variants and
  more atoms to see whether the per-field ordering in the table is stable;
  (b) if it is, derive per-field macro-clustering cutoffs from the measured
  troughs; (c) the density result suggests starting macro-clustering experiments
  on density, where the separation is cleanest.

---

### [Entry 028] Viewer Sidebar Card Background & Contrast Harmonization (`sphere_viewer.py`)

- **Objective**: Standardize drawer card styling and contrast across all steps in `sphere_viewer.py` to ensure slider labels, ticks, counts, badges, and card contents remain legible against dark mode backgrounds.
- **Changes**:
  - `gba_topology2/src/sphere_viewer.py`:
    - Updated scoped `<style>` block in `SinglePageWithDrawerLayout.drawer`:
      - Set `.v-navigation-drawer .v-card` background to `#3e434c` with foreground `#f0f2f5`.
      - Adjusted text medium emphasis color to `#d8dce2` for crisper readability.
      - Styled sub-surfaces (`.surface`, `.v-table`) to `#30343b` and table headers to `#272a30`.
      - Increased contrast on slider tracks (`#202328`, opacity 0.85), ticks (`#ffffff`, opacity 0.8), and slider typography (`#ffffff`, opacity 0.95) including `.v-slider-thumb__label`.
    - Standardized Step card containers (`Field Controls`, `Step 2: Micro-Clusters`) to use `color="surface-variant"` uniformly with the rest of the step cards.
- **Validation**:
  - `uv run pytest tests/test_sphere_viewer_*.py`: 43 passed.
  - `uv run ruff check gba_topology2/src/sphere_viewer.py tests/`: All checks passed.
  - `uv run ruff format --check gba_topology2/src/sphere_viewer.py tests/`: 23 files formatted.
  - `uv run mypy gba_topology2/src/sphere_viewer.py`: Success: no issues found.
  - Headless UI smoke test compiling and initializing viewer layout: passed cleanly.
  - Full test suite (`uv run pytest`): 253 passed.

---


### [Entry 029] High-Contrast Sidebar: Real Vuetify Theme Replaces the Force-Dark CSS (`sphere_viewer.py`)

- **Objective**: Kill the dark-grey-on-light-grey text in the sphere viewer sidebar. Target: WCAG AA (4.5:1) for all body text and chip labels.
- **Root cause -- two dead mechanisms stacked on top of each other**:
  1. **`state.trame__theme = "dark"` was a no-op.** The string `trame__theme` appears **zero times** in `trame_server`, `trame_client`, `trame_vuetify`, the built `vuetify3.js`, or the trame `index-*.js` bundle. The app has always rendered in Vuetify's **default light theme** (`defaultTheme:"light"`).
  2. **`html.Style` inside a Vue template never reaches the document.** Vue strips `<style>` from component templates. Verified two ways: a minimal Vue 3 repro (template containing `<style>` -> `document.querySelectorAll('style').length === 0`, rule not applied, no warning), and the live app (the drawer's rules were absent from `document.styleSheets`). So the Entry 028 block that set `#3e434c` cards and `#f0f2f5` text **has never executed**.
  - The real grey-on-grey was therefore the **stock light theme**. `color="surface-variant"` in the light theme is a `#424242` panel with `#EEEEEE` text -- a dark-theme pair pasted into a light one -- and Vuetify paints emphasis text as `on-background` (black) at the stock **0.6** opacity. Black at 60% over `#424242` composites to `#1A1A1A`: **1.73:1**. Measured on the card:
    - `text-medium-emphasis` (the ~50 field labels): **1.73:1**
    - `text-high-emphasis` / card body text: 1.98:1
    - `.text-teal` `#009688`: 2.74:1, `.text-grey` `#9e9e9e`: 3.75:1, `.text-cyan` `#00bcd4`: 4.38:1
    - trame's own footer links `.text-grey-lighten-1` `#bdbdbd` on the white footer: 1.88:1
- **The mechanism that does work**: `SinglePageWithDrawerLayout(server, vuetify_config=...)` -> `server.state.trame__vuetify3_config` -> `createVuetify(config)`. Vuetify generates `.text-{color}` / `.bg-{color}` rules from `themes.light.colors` into `#vuetify-theme-stylesheet`, injected into `<head>` **after** the static CSS and with `!important` at equal specificity, so **a palette entry beats the static Material rule**. Naming the accents in the palette fixes all 22 drawer tokens at once -- no per-widget edits.
- **Changes** (`gba_topology2/src/sphere_viewer.py`):
  - Added `VUE3_THEME_CONFIG`: black-on-white neutrals, semantic accents, the Material accents the cards name, explicit `-darken-N` entries, and `medium-emphasis-opacity` raised 0.6 -> 0.80.
  - Wired it via `vuetify_config=`; deleted the dead `trame__theme` assignment and the dead `html.Style` block (48 lines).
  - Card borders come from `defaults: {VCard: {border: True}}` -- set centrally, colored by the theme's `border-color` / `border-opacity` variables.
  - Added `grey-lighten-1` to the palette: **trame's own layout footer** hardcodes `text-grey-lighten-1` on the "Powered by trame" links, which is 1.88:1 on a white footer.
- **Findings**:
  1. **`VChip` defaults to `variant="tonal"`, not filled.** The label is the accent and the underlay is `currentColor` at `--v-activated-opacity` (0.12), so an accent must clear AA **against its own 12% tint** as well as against the bare surface. Choosing chip colors for a solid fill with a white label is the wrong model -- that is what made the first palette draft pass on paper and fail in the browser.
  2. **Vuetify derives `on-{color}` by perceptual (CIE) distance, not luminance** (`dt()` in the bundle). A mid-tone like `#C77800` is dark enough for black text by luminance but gets a **white** label, landing at 3.43:1.
  3. **Sliders are self-tinting**: fill and thumb are the accent at full strength, the track is the same accent at 0.38 over the surface. Fill-vs-track is 3.56-4.16:1 (passes the 3:1 non-text floor); the track-vs-surface groove is ~1.9:1 **by Material design** and is not a WCAG target -- the value is carried by the thumb, not the groove.
  4. **`surface-variant` is load-bearing beyond cards**: Vuetify also paints the slider track and ticks from `--v-theme-surface-variant`, so setting it near-white for cards has effects outside the card.
- **Validation** (Chrome 155, real `vuetify3.js` + `vue.global.js` from the installed packages):
  - Isolated harness mounting the real Vuetify with `VUE3_THEME_CONFIG`: 40/40 elements >= 4.5:1.
  - **Live app** (`ethene4.plt`, atom C1) full-page sweep of every visible text node, with the WCAG large-text exemption applied: **175 elements checked, 0 failures**. Dialog sweep: 6 checked, 0 failures.
  - Toolbar tonal buttons measured 7.28-8.13:1; table text 16.1:1; emphasis labels 11.84:1; card titles 15.8:1.
  - `uv run pytest tests/test_sphere_viewer_theme.py`: 63 passed. Mutation-checked the guards: reverting a chip fill to `#C77800`, adding an unregistered `color="lime-darken-2"`, and restoring `state.trame__theme` each fail the suite.
  - `uv run ruff check` / `ruff format --check`: clean. `uv run mypy gba_topology2 tests`: no issues in 48 files. Full suite: **316 passed**.
- **Deliberately out of scope**: the 3D viewport keeps its dark renderer background and light-on-dark actor palette (white contours, yellow highlights, gold basin centers read 8-16:1 on dark and would vanish on white). A white sidebar beside a dark viewport is the normal ParaView arrangement.
- **Open next**: (a) the same `color="surface-variant"` + light-theme trap exists in `BondalyzerParaView/trame_viewer.py` and `gba_topology/stage0_sphere_view.py`, which have no theme config at all; (b) any future `html.Style` in a trame layout is dead code -- if real CSS is ever needed it has to be a served stylesheet registered through `server.enable_module({"styles": [...], "serve": {...}})` (the client reads `trame__styles`), never a widget inside a template.

---

### [Entry 030] Local IDW Interpolation via a Kd-Tree Locator (`excision_boundary.py`, `point_locator.py`)

- **Objective**: Make per-ring field sampling cheap enough to profile an excision boundary for **every displayed saddle**, not just the ~8 catastrophe constellations. This is the enabling unit for the excision-based Launch Seeds rewrite (plan: `Excision-Based Launch Seeds`).
- **Why it is needed**: `interpolate_scalar_at_points` builds a full `(M, N)` `arccos` matrix -- one angular distance per query point per mesh vertex. At `Pd_20K` scale (20174 nodes) that is ~1e6 transcendental evaluations per 48-sample ring. Profiling ~200 rings, each at up to 12 expansion radii, costs seconds inside `update_topology()`, which already rebuilds Steps 1-9 on every toggle.
- **Changes**:
  - `gba_topology2/src/point_locator.py` (new): `KdTreeNearestFinder` wraps `vtkKdTreePointLocator` (in `vtkCommonDataModel`, **not** `vtkFiltersCore`) behind a callable, built once per mesh and reused for every query. `make_kdtree_nearest_finder(poly)` is the factory.
  - `excision_boundary.py`: added the `KNearestFinder` type alias and `interpolate_scalar_at_points_local`, which evaluates the **same** angular IDW kernel on the `k` candidates the index returns instead of on all N vertices.
  - `sphere_viewer.py`: `ViewerSession` carries a `locator` field, built in `build_viewer_session` over `sphere_poly` (the centred frame every algorithm already works in). The `apply_restore_plan` in-place mesh swap now also swaps `locator` -- a stale tree indexes a discarded sphere and returns neighbours of the **old** mesh, which is a silent correctness bug, not a crash.
- **Design decision -- the seam is a callable, not a vtk object**: `excision_boundary` advertises a pure NumPy/stdlib contract and must stay testable without a VTK build. Injecting `find_k_nearest` keeps that contract intact and lets tests pass a brute-force lambda. The adapter therefore lives in its own module.
- **Design decision -- k-nearest, not radius search**: the first draft took a chord search radius with widen-then-global fallback. It is unnecessary: mesh vertices and ring samples both lie on the sphere of radius R, so chord distance `2 R sin(theta / 2)` is a strictly monotone function of the angular distance the kernel uses, which makes the k nearest by chord **provably** the k nearest by angle. Verified empirically over 200 random queries (zero set mismatches) and pinned as `test_chord_k_nearest_equals_angular_k_nearest_on_sphere`. Dropping the radius removed the widening loop, the fallback path, and ~2x of the runtime.
- **Pre-existing bug fixed**: `interpolate_scalar_at_points` raised `ValueError: kth(=50) out of bounds (12)` from `np.argpartition` whenever `k_neighbors > N`. The local variant clamps k to the mesh size, so the two would have diverged on small meshes; the reference kernel now clamps too, preserving the drop-in parity contract.
- **Performance**: the per-query Python loop over 4-element arrays dominated, so the kernel is vectorised over the whole `(M, k)` candidate block (padded with `+inf` distances so short rows drop out of the weighting). Measured on a 20174-node sphere, 200 rings x 48 samples: **6.01s global -> 0.205s local (29.3x)**, max deviation **7.4e-15**. One ring at 12 expansion radii: **21ms**.
- **Validation**:
  - `tests/test_point_locator.py` (new, 14 tests): parity vs the global kernel on ring samples, parity through the real VTK locator, the chord/angle order-equivalence proof, exact-vertex snap, k-clamping above mesh size, zero-norm query, non-positive k rejection, single-query `(3,)` widening, locator-vs-brute-force set equality, closest-first ordering, empty-dataset rejection, and a runtime guard that the rewrite stays faster than the scan it replaces.
  - `uv run pytest`: **330 passed** (316 baseline + 14 new).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 50 files.
- **Environment note**: `uv run mypy .` is broken at baseline -- `BondalyzerParaView/plt_gba_to_vtm.py` is discovered twice under different module names, and those 2 errors abort the run before checking the rest. Confirmed pre-existing by stashing. Use `uv run mypy gba_topology2/ tests/` for this package.
- **Open next**: parabolic port azimuth refinement (unit 2), then the pure `excision_launch.py` subroutine that consumes this locator.

---

### [Entry 031] Sub-Sample Port Azimuth Refinement (`excision_boundary.py`, `micro_cluster.py`)

- **Objective**: Remove the azimuth quantization that makes a boundary port less angularly precise than the mesh it is derived from, so a port is a trustworthy launch point for separatrix integration.
- **The problem, measured**: ring samples sit at a fixed pitch of `2*pi/48 = 7.5 deg`, and `extract_boundary_ports` snaps every port to a sample index. That is a worst-case error of **3.75 deg** -- **4.7x coarser than the 1.606 deg mesh pitch** the port is meant to resolve. The cyclic-link bisector it is replacing is *more* precise than this, so shipping the ring substrate unrefined would have been a regression in launch direction.
- **Changes**:
  - `extract_boundary_ports` / `profile_boundary_loop` take `refine_azimuth: bool = False`. When on, each accepted extremum is located between samples by fitting a parabola to the three smoothed points (`_refine_extremum`) and its position is re-evaluated **on the loop** at the refined azimuth, via the ring's own frame.
  - `micro_cluster.ring_frame` (new, public): the analytic inverse of `generate_boundary_ring`, recovering `(c_unit, t1, t2, angular_radius)` from ring samples. This was the private `_ring_frame` inside `harmonic_nudge`; it is moved to `micro_cluster` because (a) it belongs beside the generator it inverts, and (b) `excision_boundary` cannot import from `harmonic_nudge` -- that is a cycle, since `harmonic_nudge` already imports from `excision_boundary`. Pure relocation, no behaviour change: the 330-test suite passed unchanged.
  - The four duplicated `BoundaryPort(...)` construction blocks in `extract_boundary_ports` are consolidated into one `_port` builder.
- **Design decisions**:
  - **The prominence test stays on the discrete profile.** Refinement moves an accepted port between its bracketing samples; it must never change *which* ports are accepted. Pinning this as `test_refinement_preserves_port_counts_and_types` for k = 2, 3, 4.
  - **The frame is recovered once per call and reused**, not re-derived per port.
  - **`scalar_value` is not refined.** This routine sees only the profile, so a refined azimuth does not imply a re-interpolated scalar; callers that need it re-interpolate (as `reposition_ports_on_ring` already does).
  - **Clamped to one sample step**, with a flat-stencil guard (`den ~ 0`) and non-finite guards, so a degenerate stencil keeps the sample.
- **Empirical validation of the accuracy claim** (this is the part worth keeping):
  - **Isolated, analytic** `cos(3(phi - phase))`, 48 samples, no mesh: mean max-error **3.750 deg -> 0.000-0.019 deg** across phases. The parabola is near-exact on a smooth profile, as expected.
  - **Real 20174-node mesh, quadrupole, phase-swept** through 277 non-grid-aligned orientations (the fair test -- a sweep step equal to the 7.5 deg sample pitch makes every trial grid-aligned and shows snapped as *perfect*, which is an artifact, not a result):

    | ring radius | snapped | refined | gain |
    |---|---|---|---|
    | 1.606 deg (1 dtheta) | 9.314 | 8.618 | 1.1x |
    | 3.20 deg | 5.640 | 5.066 | 1.1x |
    | 8.00 deg | 2.619 | 1.363 | 1.9x |
    | 20.0 deg | 1.941 | 0.493 | 3.9x |

  - **Refinement wins at every radius and every synthetic noise level tested** (0-10% of span). It never loses.
- **The finding that matters for the expansion loop**: at a tight ring the gain is only 1.1x because **IDW interpolation noise dominates the quantization it removes**. Measured against the analytic field on a 20174-node sphere, IDW error is **7.3% of the field span at r = 1.6 deg**, and the high-frequency content of the *interpolated* profile is **3.6x that of the true profile** (1.38e-4 vs 3.79e-5) -- the interpolant is noisier than the signal it samples. At r = 20 deg the IDW error is 0.5% of span.
  - This is independent evidence for the ring-expansion design: a ring at ~1 dtheta is inside the interpolation-noise regime, and expanding it buys real accuracy, not just topological cleanliness. It also bounds what refinement can do -- the residual at tight radii is an **interpolation** limit, not a sampling limit, so raising `ring_samples` will not fix it.
  - Checked and rejected the obvious alternative: **barycentric interpolation on the mesh triangles is worse**, not better (32% of span vs 12.6% at r = 1.6 deg on a structured test mesh, and its profile noise tracks the triangle crossing pattern). The mesh triangles are long and thin near the poles, so linear interpolation across them is a poor fit for a smooth field.
- **Validation**:
  - `tests/test_excision_boundary.py`: 9 new tests (15 total) -- sub-degree phase recovery on an analytic field, refined strictly beats snapped, port counts/types/ordering preserved, refined positions lie on the small circle, reported azimuth consistent with reported position in the ring's own frame, default-off is byte-identical, flat-stencil and clamp guards, and the flag threaded through `profile_boundary_loop` end-to-end on a real mesh.
  - `uv run pytest`: **339 passed** (330 + 9).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 50 files.
- **Open next**: `excision_launch.py` (unit 3), the pure ring-to-seed subroutine, which will call `profile_boundary_loop(..., refine_azimuth=True)`.

---

### [Entry 032] Excision-Based Launch Seeds: the Ring-to-Seed Subroutine (`excision_launch.py`)

- **Objective**: Produce separatrix launch branches from a saddle's **excision boundary**, so seeds can be computed for effective critical points, which have no mesh vertex to scan.
- **Why the cyclic link cannot do this**: `saddle_launch.extract_saddle_launch_seed` needs a `vertex_id` and walks the incident-triangle link graph. An effective CP's position is a mean of member directions, optionally nudged -- `effective_cp_to_ui` hardcodes `"vertex_id": -1`. No link, nothing to scan. Step 9 iterated `morse_res.saddles` and silently omitted every `E-SAD`, contradicting its own comment. The excision loop is defined for **any** point on the sphere, which is the property that makes it the substrate for the reduction stage.
- **Changes**:
  - `gba_topology2/src/excision_launch.py` (new, pure -- no VTK, no viewer): `extract_excision_launch_seed(ring_points, mesh_pts, f_vals, saddle_id, saddle_position, saddle_value, expected_k, ...)` -> the existing `SaddleLaunchSeed` / `LaunchBranch` dataclasses, so `_emit_branch`, the tube filters, the actors and the visibility gate in the viewer all survive unchanged.
  - `SaddleLaunchSeed` gains two defaulted fields: `ring_radius_rad` (the accepted radius, for expansion bookkeeping) and `failure_reason` (why a ring produced nothing). Both zero/`None` for the link path, so existing construction sites are untouched.
  - `profile_boundary_loop` gains `find_k_nearest`, threading the kd-tree locator from Entry 030 into the profiling entry point.
- **Mapping from link scan to ring** (the semantics carry over cleanly):
  - link sector with `f(u) > f(v)` -> **ridge port** (local max of `f|_loop`) -> `kind="ascent"`, unstable manifold.
  - link sector with `f(u) < f(v)` -> **valley port** -> `kind="descent"`, stable manifold.
  - link sign crossing -> profile crossing of `saddle_value`, by chord interpolation (`_crossing_azimuths`), giving the 2k separatrix directions.
  - link sector bisector -> the **port itself**, refined to sub-sample azimuth (Entry 031).
  - `fold_order = len(ascent)` -- unchanged formula, now measured on the loop.
- **Launch direction**: the separatrix runs centre -> port -> outward, so the tangent is the geodesic direction at the port pointing away from the ring centre: `t = normalize((p.c) p - c)`. It is tangent to the sphere by construction (`t.p = 0`) and outward (`t.t1 = +cos(a) > 0`, and moving along it decreases `p.c`, i.e. increases angular distance from the centre).
- **Bug caught by the tests**: the first draft used `c - (p.c) p`, which is the **inward** tangent -- the direction back toward the degenerate core. `test_tangent_points_away_from_the_ring_centre` failed on it immediately. The sign matters: launched inward, every Step 3 path would run into the singularity it is meant to escape.
- **Failure taxonomy** (`FAILURE_REASONS`, the routing keys for the expansion loop):
  | Reason | Meaning |
  |---|---|
  | `empty_ring` / `too_few_samples` | no loop to profile |
  | `no_ports` | structureless profile (e.g. a constant field) |
  | `unbalanced` | valleys != ridges -- still inside the degenerate core, expand |
  | `below_min_k` | balanced 1 + 1 -- a ring around an **extremum**, not a saddle |
- **Design decision -- this subroutine measures, it does not judge**: `index_agrees = (fold_order == expected_k)` is reported, never enforced. Rejecting a ring whose `k` disagrees is the expansion loop's job (unit 4). Keeping the two apart preserves the ring as an **independent** check on the reduction's `chi = 1 - k`; if the loop searched for `expected_k` and this function coerced to it, the audit would be circular.
- **The extremum case, and why it is not fully handled here**: a ring around a maximum has a nearly flat profile, so IDW noise alone produces many balanced "ports" -- measured at `k = 17` on a synthetic maximum. The `k >= 2` gate catches the 1 + 1 case, and a noise-generated 17 is caught by `index_agrees = False`. Both are the expansion loop's `k == expected_k` acceptance test rejecting the ring, which is exactly the division of labour above. Pinned as `test_extremum_ring_is_not_accepted_as_a_saddle`.
- **Validation**:
  - `tests/test_excision_launch.py` (new, 23 tests): k = 2 / 3 / 4 branch counts and crossing counts (4 / 6 / 8), ideal 360/k azimuth spacing, ascent/descent interleaving (6 cyclic sign flips for k = 3), launch positions on the sphere **and** on the loop's small circle, tangents unit-length and perpendicular to the radial, tangents pointing outward, dense counter-clockwise `sector_index`, reported ring radius, index agreement and mismatch (measured `k` never coerced), all five failure reasons, `unbalanced` and `below_min_k` produced by real tilted fields, off-pole centres (frame recovered from the ring, not assumed), local-vs-global sampling parity to 1e-12, and identity-field passthrough.
  - `uv run pytest`: **362 passed** (339 + 23).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 52 files.
- **Open next**: the ring-expansion loop (unit 4) that routes on `FAILURE_REASONS` and the `k == expected_k` audit, then the Step 9 rewire.

---

### [Entry 033] Self-Calibrating Excision Ring Search (`excision_launch_seeds.py`)

- **Objective**: Choose the excision radius for each saddle, so a ring that starts inside the degenerate core expands until its port signature is pure.
- **Why the radius must be searched**: the excision-radius rule puts a lone saddle's ring at `max(0 + 1.5deg, k_pitch*dtheta/2)` = **exactly 1 dtheta** on `Pd_20K`. Entry 031 measured that regime: IDW error is 7.3% of field span there, and the interpolated profile carries 3.6x the high-frequency content of the true one. A ring that tight has no clean alternating signature, and the tell is topological -- valley and ridge counts disagree. So **balance is used as a convergence test**.
- **Changes**: `gba_topology2/src/excision_launch_seeds.py` (new, pure). `expand_excision_seed(SeedRequest, ...) -> ExpansionOutcome` walks radii from `a0` in `expand_pitch * dtheta` steps to `max_radius_pitch * dtheta`, accepting the first ring that is balanced, has `k == expected_k`, and has `k >= 2`. `summarize_outcomes` aggregates into the viewer's `launch_stats`.
- **Separation of concerns**: this module owns the *policy* (which radius to accept); `excision_launch` owns the *measurement* (what the ring says). Kept apart so the ring stays an independent check on the reduction's `chi = 1 - k`.
- **The finding that changed the design -- the overshoot early-exit needed a grace radius**: the plan specified stopping as soon as a balanced ring reported `k > expected_k`, on the assumption that port count grows monotonically with radius. **That assumption fails exactly where the ring is tightest, which is where the search starts.** Measured on a k = 3 monkey saddle:

  | a0 | with immediate exit | with sustained exit |
  |---|---|---|
  | 0.4 deg | expanded, k=3 | expanded, k=3 |
  | 0.8 deg | **k_overshoot, FAILED** | expanded, k=3 |
  | 1.2 deg | **k_overshoot, FAILED** | expanded, k=3 |
  | 2.0-20.0 deg | converged, k=3 | converged, k=3 |

  Interpolation noise inflates the port count at a tight ring, so a single radius reads k = 4 for a k = 3 saddle and the search terminates on a guaranteed-successful path. Requiring the overshoot at **two consecutive radii** fixes it: all starting radii now recover k = 3. A *sustained* overshoot is still a real signal, so the cost saving survives. Mutation-checked: reverting to the immediate exit fails 6 of the 18 tests.
- **Failure taxonomy** (`ExpansionOutcome.status`), and the diagnosis each one implies:
  | Status | Meaning |
  |---|---|
  | `converged_at_a0` | the radius rule was right |
  | `expanded` | the rule was too tight -- tune `min_boundary_margin_deg` |
  | `unbalanced` | never balanced at any radius -- degenerate core |
  | `k_mismatch` | balanced at the wrong k -- **the reduction's index is suspect** |
  | `k_overshoot` | k passed the target and stayed past it -- ceiling too loose |
  | `no_ring` | `expected_k < 2`, nothing to search for |
  The `k_mismatch` / `unbalanced` split is what preserves the audit that accepting on `target_k` would otherwise destroy: `observed_k` is recorded at the **tightest balanced radius**, so "the field disagrees with the reported chi" is distinguishable from "the ring never resolved". Mutation-checked: collapsing the two into one bucket fails 3 tests.
- **Ceiling semantics**: the ceiling bounds **expansion**, not `a0`. A constellation's ring is `span/2 + dtheta` and can legitimately exceed 12 dtheta, so `a0` is always profiled once. Pinning this explicitly, because "the radius never exceeds the ceiling" is false and a future reader will assume it.
- **Validation**:
  - `tests/test_excision_launch_seeds.py` (new, 18 tests): convergence at a0 for k = 2/3/4, expansion from a too-tight ring, all seven starting radii recovering the same k, `k_mismatch` and `unbalanced` diagnoses, ceiling bounds, `expand_pitch` controlling step count, `expected_k < 2` short-circuit, the caller's ring reused unmodified at a0, local-vs-global sampling parity to 1e-12, `num_samples` honoured, and the aggregate stats (converged/expanded/failed counts, failing entity ids with radius, empty-input well-formedness).
  - `uv run pytest`: **380 passed** (362 + 18).
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 54 files.
- **Open next**: port colours (unit 5), the parameter schema (unit 6), then the Step 9 rewire that consumes all of this.

---

### [Entry 034] Excision Ring Controls in the Parameter Schema (`params_schema.py`)

- **Objective**: Expose the excision ring geometry as live, savable controls, so the ring substrate can be tuned interactively and a tuned configuration can be captured in a snapshot.
- **Changes**:
  - `DEFAULT_UI_STATE` gains `ring_samples` (48), `ring_expand_pitch` (1.0), `ring_max_pitch` (12.0).
  - All three added to the viewer's `@state.change` handler so they trigger a rebuild.
  - `PARAMS_SCHEMA_VERSION` **1 -> 2**.
  - The two committed sample snapshots migrated to v2.
- **Why adding keys is breaking, not additive**: `load_params_from_dict` requires **every** control key to be present and raises `KeyError` naming the first missing one. A version 1 file lacks all three new keys, so without the bump it fails as a confusing `Missing parameter 'ring_samples' in section 'floats'` -- it looks like a corrupt file, not an old one. Bumping converts that into the clear `Unsupported params schema_version 1` ValueError. The version gate exists precisely to make stale files reject legibly.
- **The type tables need no edits**: `PARAM_BOOL_KEYS` / `PARAM_FLOAT_KEYS` are derived by `isinstance` from `DEFAULT_UI_STATE`, and the import-time coverage check (`PARAM_*_KEYS` must partition `PARAM_CONTROL_KEYS`) validates the derivation. Adding a key to the defaults is sufficient; the check is the safety net.
- **`ring_samples` is int-valued, exported as float**: it follows the `num_contours: 15` precedent -- `_coerce_float` widens every numeric control, so the round-trip yields `48.0` and consumers must cast with `int()`. Pinned by `test_ring_samples_loads_as_float_and_must_be_cast` so the cast is not later mistaken for a bug and removed.
- **Fixture migration**: the two `ethene4_atom1_params *.json` files are real viewer output used by `test_a_saved_file_without_a_camera_still_resolves`, whose point is that an **absent camera section** still loads -- not that v1 loads. Migrated in place: `schema_version` 1 -> 2, the three new floats added at their defaults, camera left absent so the test keeps covering what it was written to cover. Written with `ensure_ascii=True` and `sort_keys=True` to match the exporter's own output; the first attempt used `ensure_ascii=False` and rewrote every non-ASCII escape as raw UTF-8, a 22-line diff where 5 was correct.
- **`launch_seed_pitch` deliberately retained** in this unit. It is still read by Step 9 and by `diagnose_params.stage_launch`; removing it here would break the viewer for one commit. It is retired in the Step 9 rewire, which is the unit that stops needing it.
- **Validation**:
  - `tests/test_sphere_viewer_params_export.py`: `test_load_rejects_unknown_schema_version` extended to include version **1** in the rejected set -- the real-world stale case, previously untested. Two new tests: the ring controls round-trip with overridden values, and the int-as-float coercion contract.
  - `uv run pytest`: **390 passed** (388 + 2). Params suites specifically: 76 passed, and the fixture test verified to PASS rather than skip.
  - `uv run ruff format` / `ruff check`: clean. `uv run mypy gba_topology2/ tests/`: no issues in 55 files.
- **Open next**: the Step 9 rewire (unit 7), which consumes `ring_samples` / `ring_expand_pitch` / `ring_max_pitch` and retires `launch_seed_pitch`.

---

### [Entry 035] Step 9 Rewired to Excision-Boundary Seeds (`sphere_viewer.py`, `diagnose_params.py`)

- **Objective**: Seed every **displayed** saddle from its excision boundary, so effective saddles -- which the cyclic-link scan structurally cannot serve -- get launch seeds.
- **Changes**:
  - `sphere_viewer.build_launch_requests` (new, module-level, pure): the seed-set rule. Raw saddles not in `hidden_vertex_ids`, plus every `E-SAD` from `effective_cps_ui`. Extracted from the `update_topology` closure so the rule is unit-testable without a viewer, per AGENTS.md Principle 1.
  - Step 9 now builds `SeedRequest`s, runs `expand_excision_seed` over each, and takes `state.launch_stats` straight from `summarize_outcomes`. `compute_all_saddle_launch_seeds` is no longer imported by the viewer.
  - `eff_ring_radius` records each effective CP's initial radius (constellation `angular_radius_rad`, or the unit's cluster radius), keyed by `id_label`.
  - `diagnose_params`: `RunContext` gains `locator` and `eff_ring_radius`; `stage_launch` uses the same rule, so the headless harness and the viewer cannot drift.
- **Effective saddles are centred correctly, including under the nudge**: the ring is generated fresh at `eff_dict["position"]`, which is the **nudged** centre, rather than reusing the constellation's stored ring, which is centred on the pre-nudge centroid. UI positions are world frame, so they are mapped back with `to_local` before the search.
- **The finding that changed the failure taxonomy**: the first run on `Pd_20K` reported 36 `k_overshoot` failures. Diagnosing one (`SAD_5`, multiplicity 1, so expected k = 2) showed the ring reading **k = 3 stably from 3.2 deg all the way to 19.3 deg** -- not interpolation noise, and not a loose ceiling. That is a genuine disagreement with the index the Morse detector reported, which is exactly what `k_mismatch` means.
  - So `k_overshoot` was **removed as a status**. A *sustained* overshoot is an index mismatch by definition; the two classes were the same event wearing different names. The search now terminates when **two consecutive radii agree** on the same inflated `k`, and returns `k_mismatch`.
  - The two-consecutive-radii rule is still required, for the reason found in Entry 033: a *single* inflated reading at a tight ring is noise, and a k = 3 monkey saddle read at 0.8 deg recovers to k = 3 one step later. Verified: the stable path exits at `radii_tried = 2` of ~11 available, so the cost saving survives.
  - Reclassifying also **raised the yield**: 164 -> 185 of 200 saddles seeded, because radii whose readings differ now continue expanding instead of terminating.
- **`Pd_20K` (`V (condensed)`, 20174 nodes, dtheta = 1.606 deg)**:

  | toggle set | reqs | seeded | conv@a0 | expanded | k_mismatch | mean radii | time |
  |---|---|---|---|---|---|---|---|
  | none | 200 raw | 185 | 118 | 67 | 15 | 1.73 | 0.84s |
  | reduce_catastrophes | 194 raw + 2 E-SAD | 181 | 114 | 67 | 15 | 1.74 | 0.85s |
  | cats + clusters | 18 raw + 2 E-SAD | 20 | 15 | 5 | 0 | 1.25 | 0.06s |

  - **The 2 E-SADs converge at a0 with k = 3, 3 ascent + 3 descent, and `vertex_id = -1`** -- the case the rewrite exists for, and one the link scan silently dropped.
  - 8 catastrophes detected, 2 fused; the 6 split ones are physical symmetry breaking and correctly stay as raw saddles.
  - Cost is **0.84s for 200 rings** with the locator, against the 6.0s the global kernel alone would have taken for the same sampling.
  - `n_rings_unbalanced = 0` everywhere: the tight-ring balance failure predicted in the plan did not materialise, because the expansion loop resolves it.
- **The 15 `k_mismatch` findings are the audit working as designed.** Under the plan's `target_k` acceptance the ring constructs the index rather than measuring it; recording the stable measurement and reporting the disagreement is what preserved the check. These 15 are simple saddles whose excision loop consistently shows 3 ridge ports, and they are now visible in `launch_stats.failures` with id, expected/observed k, radii tried and final radius.
- **Validation**:
  - `tests/test_sphere_viewer_reduction.py`: 11 new tests on the seed-set rule -- visible raw saddles requested, hidden ones excluded, E-SADs included with `vertex_id = -1` and the constellation radius, E-MAX/E-MIN not seeded, `expected_k` from multiplicity for raw and `fold_order` for effective, the lone-CP radius rule (both the pitch floor and the 1.5 deg margin branches), `k_pitch` reaching the rule, world-to-centred mapping of effective centres, the fallback when no radius is recorded, and stable raw-then-effective ordering.
  - `tests/test_excision_launch_seeds.py`: 3 new tests pinning the reclassified taxonomy -- a single noisy overshoot must not terminate, a stable one is `k_mismatch`, and `n_rings_k_overshoot` is gone.
  - `tests/test_diagnose_params.py`: fixture gains the ring params and a real locator. The octahedron end-to-end still asserts `n_saddles_seeded == 2` and now yields 2 saddles x k = 2 = 4 ascent + 4 descent, converged at a0 -- a meaningful check on a 6-vertex, 90 deg-pitch mesh.
  - `uv run pytest`: **403 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.
- **Open next**: the sidebar (unit 8) -- `launch_seed_pitch` is now dead in the pipeline and must be replaced by the ring sliders and the failure list.

---

### [Entry 036] Launch Seeds Sidebar: Ring Controls and Failure List (`sphere_viewer.py`)

- **Objective**: Expose the excision ring controls, the expansion outcome taxonomy, and the unresolved-ring list, so the substrate is tunable and a flagged ring is findable in the scene.
- **Changes**:
  - Replaced the Launch Offset slider (`launch_seed_pitch`) with three controls: **Ring Samples** (24-96, step 8), **Expand Step** (0.25-3.0 x dtheta), **Radius Ceiling** (2-24 x dtheta, shown with its degree equivalent).
  - Second stats row: **Ring @ a0** (green) and **Expanded** (amber), plus "Mean N radii profiled (max M)". The split is the tuning signal -- a high expanded count says the excision-radius rule is too tight.
  - Separate lines for `n_rings_k_mismatch` ("ring k != reduction k") and `n_rings_unbalanced` ("never balanced (degenerate core)"), which the old single "Index mismatch (k != mu+1)" line conflated.
  - **Unresolved rings** list: id, status, observed vs expected k, radii tried, final radius.
  - Retired `launch_seed_pitch` from `DEFAULT_UI_STATE` and the `@state.change` handler. No consumers remained after the Step 9 rewire.
- **Retiring a control key needs no fixture migration**: `load_params_from_dict` iterates the *schema's* keys and reads them from the payload, so an extra key in an old file is simply ignored. Only *adding* a key breaks old files, which is why Entry 034's version bump was required and this one is not.
- **Verified live** (headless server on `ethene4.plt`, atom C1, driven through the browser):
  - Layout builds; all three sliders render with correct values and derived labels ("48 pts (7.5 deg pitch)", "1.00x dtheta", "12.0x dtheta (85.0 deg)").
  - **Sliders are reactive**: moving Ring Samples 48 -> 88 recomputed the pipeline live -- seeded 10 -> 9, expanded 0 -> 6, mean radii 1.33 -> 3.33, mismatches 2 -> 3, and the failure list repopulated with new ids.
  - Failure rows render with the id and diagnosis separated ("SAD_444 k_mismatch . k=3 != 2 . 3 radii . @21.3 deg"). The first draft concatenated them ("SAD_444k_mismatch") because inline elements collapse inter-element whitespace; fixed with an `ml-1` margin class, not a leading space, which HTML would also collapse.
  - Viewport check: ridge (orange-red) and valley (cyan) ports render on the purple catastrophe ring, matching the ascent/descent seed stubs from Entry 035.
- **The tuning finding, and it contradicts the obvious intuition**: raising Ring Samples from 48 to 88 made the search **worse** -- fewer saddles seeded (10 -> 9), more expansions (0 -> 6), more mismatches (2 -> 3). Finer azimuthal sampling resolves IDW interpolation noise as additional spurious extrema, inflating the measured k. This is the direct consequence of the Entry 031 measurement: at a tight ring the interpolated profile carries **3.6x the high-frequency content of the true field**, so sampling it more densely samples more noise.
  - The practical rule: **the residual error is an interpolation limit, not a sampling limit.** Raising `ring_samples` cannot buy accuracy; expanding the radius can, because the IDW error falls from 7.3% of field span at 1.6 deg to 0.5% at 20 deg. 48 is a reasonable default and the slider is for exploring the ceiling, not for beating it.
- **Validation**: `uv run pytest`: **403 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.

---

### [Entry 037] Removing the Expansion Early-Exit; Ring Geometry Fixed (3 sliders -> 0)

- **Objective**: Fix the reported defect (E-SAD_MC22 seeds only at expand step >= 1.25, not at 0.75) and reduce the number of adjustable parameters needed to find the correct topology, per the project goal.
- **The bug, and it was mine**: the `k_overshoot` -> `k_mismatch` early-exit added in Entry 033 terminates the search when two **consecutive ladder rungs** agree on an inflated `k`. "Consecutive" counts rungs, and rung spacing **is** the expand step (`step = expand_pitch * dtheta`). IDW noise is spatially correlated, so two closely spaced radii agree almost by construction. A fine, thorough scan was punished with premature termination; a coarse one survived. **The expand step was choosing which saddles seeded**, which is exactly what the goal forbids.
- **Reproduced on the real data before fixing** (Pd_20K, Electron Density (condensed), old code, step 0.75/48 samples):

  | saddle | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 2.0 |
  |---|---|---|---|---|---|---|
  | E-SAD_MC22 | FAIL | FAIL | **FAIL** | FAIL | OK | OK |
  | E-SAD_MC4 | OK | OK | OK | OK | OK | OK |

  Exactly the reported symptom, including MC4 being unaffected. **Coding problem, not logic** -- the user's logic and the predecessor's were both right.
- **The fix**: delete the early-exit. The sweep now profiles every radius on the ladder and returns only on `k == expected_k`, which is precisely the predecessor's `compute_adaptive_excision_regions` (gba_topology/topology_engine.py:1477): sweep `r_sweep`, `break` on success, no early exit. That is why the predecessor never showed non-convergence. `best_k_at_balance` and the final `k_mismatch`/`unbalanced` diagnosis are kept -- that is the chi = 1 - k audit.
- **Verified after fixing**: MC22 and MC4 both converge at **every** step in {0.25, 0.5, 0.75, 1.0, 1.25, 2.0}, all at k = 2.
- **Parameters reduced 3 -> 0.** All three sliders removed; the values are module constants in `excision_launch_seeds`, matching the predecessor, which hardcoded them (3.0/24.0/0.5 deg, 64 samples):

  | Constant | Value | Predecessor |
  |---|---|---|
  | `RING_SAMPLES` | 16 | 64 |
  | `EXPAND_PITCH` | 0.5 dtheta | 0.5 deg |
  | `MAX_RADIUS_PITCH` | 16 dtheta (25.7 deg on Pd_20K) | 24 deg |

  They remain function arguments (defaults = the constants) so tests can vary them; the viewer and `diagnose_params` pass none, so the harness and the viewer cannot drift.
- **`RING_SAMPLES = 16` validated on real data**, and the user's recollection that 48 was overkill is confirmed. Yield is 100% at every count; the difference is *where* it converges:

  | dataset | samples | seeded | conv@a0 | expanded | mean radii | time |
  |---|---|---|---|---|---|---|
  | Pd_20K / Electron Density | 16 | 12/12 | 6 | 6 | 1.50 | 0.03s |
  | Pd_20K / Electron Density | 48 | 12/12 | **0** | **12** | **4.33** | 0.13s |
  | Pd_20K / V | 16 | 20/20 | 20 | 0 | 1.00 | 0.03s |
  | Pd_20K / V | 48 | 20/20 | 15 | 5 | 1.25 | 0.06s |
  | ethene4 / Electron Density | 16 | 5/5 | 5 | 0 | 1.00 | 0.01s |

  48 samples converges at a0 far less often and costs 4x, consistent with Entry 031: finer azimuthal sampling resolves interpolation noise as spurious extrema. 12 also works, so 16 has margin.
- **Two jobs, two constants**: `ring_samples` had been driving BOTH the measurement profile and the DRAWING of the visible purple/black boundary rings. Profiling now uses 16; drawing keeps `num_boundary_points=48` in `compute_micro_clusters`, so displayed rings are not 16-gons.
- **Ceiling semantics, simplified**: with no early exit, a tight ceiling just means fewer rungs -- it can no longer trap a ring. The earlier "the floor is load-bearing" caution from the plan is retired as the user directed.
- **Pd_20K Electron Density, 172 raw saddles, fixed geometry**: 127 converged at a0, 39 expanded, 6 `k_mismatch`, 0 unbalanced. Counters sum to 172 exactly. **0.67s total, 3.9ms per saddle.** The 6 failures each ran the full 31-rung ladder to the 25.7 deg ceiling reading k = 3 against an expected k = 2 -- genuine index disagreements, surfaced with ids in the sidebar.
- **Validation**:
  - `tests/test_excision_launch_seeds.py`: 4 new tests. The regression is pinned by a **scripted stub** that replaces the measurement with `k(radius)`, isolating the search policy from the numerics -- the first attempt used a synthetic noisy field and failed for the right reason (it modelled the opposite shape, k inflated at large radius), which is itself evidence the stub is the honest test. `test_search_is_invariant_to_the_expand_step` reproduces the MC22 ladder (inflated k below 4 deg, true k above) and asserts convergence at all six step sizes; `test_finer_step_is_at_least_as_thorough` asserts success is monotone in step size; `test_no_early_exit_scans_the_whole_ladder` asserts a permanently inflated k is diagnosed only after the full sweep.
  - **Mutation-checked**: restoring the early-exit fails 2 of the new tests, and reproduces MC22's exact failure pattern on real data.
  - `tests/test_sphere_viewer_params_export.py`: the two ring-control tests replaced by `test_ring_geometry_is_not_a_control` (the keys cannot return to `DEFAULT_UI_STATE`) and `test_snapshot_with_retired_ring_keys_still_loads` (old files carrying the keys load fine -- removing a key needs no version bump, only adding one does).
  - `tests/test_diagnose_params.py`: fixture drops the three keys.
  - Live check (Pd_20K, port 8921): card renders with the three sliders gone, stats grid, mismatch line and unresolved-ring list intact.
  - `uv run pytest`: **407 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.
- **Open next**: the 6 `k_mismatch` saddles on electron density are worth inspecting -- they are simple saddles whose loop consistently shows 3 ridge ports, which is a finding about the Morse detector, not the rings.

---

### [Entry 038] Provisional Launch Seeds: Rendering the k_mismatch Geometry

- **Objective**: Show launch points for saddles whose excision ring balanced cleanly at a fold order that **disagrees** with the index the reduction reports, while still flagging them as unresolved.
- **The motivating case (user-reported, reproduced on `Pd_20K.plt` / Curvedness)**: `E-SAD_MC21` reduces as a **-2** saddle (`expected_k = 3`, six ports), but the site is genuinely fourfold -- a nearby saddle and minimum merged into it, leaving an index-0 region adjacent. The ring reads **k = 4 at every radius out to the ceiling**, i.e. eight ports. The user predicted eight launch points; the code produced exactly eight.
- **The bug was a one-line filter, not missing data.** `expand_excision_seed` already returns a fully populated seed for `k_mismatch` -- all 8 branches intact. It was dropped at `seeds = [o.seed for o in outcomes if o.converged]` (`sphere_viewer.py:2320`) and by the identical filter in `summarize_outcomes`.
- **The strongest argument for the change: `n_index_mismatch` was a dead statistic.** It was computed over converged seeds only, and converged *means* `fold_order == expected_k`, so `index_agrees` is `True` for every seed it inspects -- the counter was **structurally always 0**, and a test asserted `== 0` with the comment "no seed emitted, so nothing to audit". The `chi = 1 - k` audit that `best_k_at_balance` exists to preserve was computed and thrown away. Counting over all renderable seeds makes it live.
- **Changes**:
  - `ExpansionOutcome.renderable` -- has branches to draw (converged, or `k_mismatch`). `unverified` -- renderable and not converged. `unbalanced` and `no_ring` seeds have **empty** branch lists, so "has branches" selects exactly the `k_mismatch` cases with no status test.
  - `sphere_viewer._make_seed_actor(renderer, rgb, opacity=1.0)` (new, module-level): builds a seed tube pipeline. Replaces ~35 duplicated lines and is called four times -- verified ascent/descent at full opacity, provisional ascent/descent at `UNVERIFIED_OPACITY = 0.45`.
  - **Colours are now structural, not asserted.** All six manifold-drawing actors (2 ports + 4 seeds) take their hue from the shared `ASCENT_RGB` / `DESCENT_RGB` constants. The provisional pair uses the same hues faded, not a third colour: a port and the stub growing out of it are the same manifold, and a new hue would add a legend entry.
  - `summarize_outcomes`: added `n_saddles_seeded_unverified`, `n_ascent_unverified`, `n_descent_unverified`; `max_fold_order` spans both sets; `n_index_mismatch` now counts over all renderable seeds; failure records carry `renderable`, `n_ascent`, `n_descent`.
  - **Seed/radius consistency fix**: the retained failure seed was the *ceiling* ring's, while `observed_k` was recorded at the *tightest balanced* radius -- the drawn geometry and the reported measurement disagreed. Now the seed, its radius, and `observed_k` all come from the tightest balanced ring (new `observed_radius_rad` field). On MC21 this moved the accepted radius from 25.2 deg (the ceiling) to **8.3 deg** -- a far more local, more trustworthy measurement.
  - **Visibility gate fixed**: it read `n_ascent`/`n_descent`, which excluded provisional seeds, so a populated actor could stay hidden. Now gates on the sum; each actor's own visibility uses its own count.
  - `reset_defaults` hides the two new actors. `apply_restore_plan` needs no change -- it ends in `update_field()` -> `update_topology()`, which recomputes visibility.
  - Sidebar: a "Provisional (faded in scene)" line, and failure rows now append "N seeds shown faded".
  - `failures` is sorted renderable-first before the `[:40]` cap, so a saddle with geometry in the scene is not pushed off the list.
- **Validation on `Pd_20K.plt` / Curvedness, atom 1, reduce_multi_clusters on (30 saddles)**:
  - verified: 29 seeded, 68 ascent + 68 descent; provisional: **1 seeded, 4 ascent + 4 descent = 8**.
  - `n_index_mismatch = 1` (was structurally 0).
  - `E-SAD_MC21`: `status=k_mismatch`, `converged=False`, `renderable=True`, `unverified=True`, `expected_k=3`, `observed_k=4`, seed `fold_order=4`, 4+4 branches, 22 radii, accepted R=8.32 deg.
  - **Live browser check**: sidebar reads "Provisional (faded in scene): 1 saddle(s), 4 ascent + 4 descent", "Index mismatch (ring k != reduction k): 1", and the failure row "E-SAD_MC21 k_mismatch . k=4 != 3 . 22 radii . @8.3 deg . 8 seeds shown faded".
- **Complications considered (the user asked)**:
  1. **Visibility gate coupling** -- real, fixed above.
  2. **`n_saddles_seeded` semantics** -- kept meaning "verified/certified"; provisional is counted separately so the sidebar never mixes them.
  3. **Step 3 must not inherit these silently** -- convergence is the certification that a seed set is consistent with the reduction. The two lists stay separate (`seeds` / `unverified_seeds`), so a future integrator opts in deliberately. A k=4 seed set on a -2 saddle traces separatrices that do not close against the reduction's index.
  4. **Visual ambiguity** -- addressed by opacity, not hue.
- **Validation**:
  - `tests/test_excision_launch_seeds.py`: 9 new tests -- `renderable`/`unverified` truth tables across all five statuses, retained seed matches `observed_k` and its radius, verified/provisional counts stay separate, `n_index_mismatch` is live, failure records carry branch counts, renderable failures sort first.
  - `tests/test_sphere_viewer_seed_actors.py` (new, 5 tests): the factory registers with the renderer, starts hidden, applies the given hue, is opaque by default, fades only when asked, and returns independent pipelines per call (sharing a polydata would mix verified and provisional geometry).
  - `tests/test_sphere_viewer_port_colors.py`: rewritten to assert every manifold actor is wired to a shared constant, plus a guard that none hardcodes a literal RGB. **Mutation-checked**: reverting a port to literal cyan fails 3 tests.
  - `uv run pytest`: **425 passed**. `ruff format` / `ruff check` / `mypy gba_topology2/ tests/`: clean.
- **Pre-existing defects found, NOT fixed (out of scope, unrelated to this work)**: two toolbar/table actions are referenced with `ctrl.<name>` but never defined with `@ctrl.add`, so clicking them raises `FunctionNotImplementedError` -- `select_cp_from_list` (2 refs) and `reset_camera` (2 refs). Both confirmed present at baseline `bee00a5` with 2 references and 0 definitions.
- **Open next**: MC21 is a real reduction defect -- a fourfold site collapsed to -2 by an adjacent saddle+minimum merge. The ring is right and the reduction is wrong, which is what the provisional seeds now make visible.

---

## 2026-10-31 -- Pairwise monkey saddles and the bimodal-quad gate

- **Symptom** (reported on `Pd.plt`, field `V (condensed)`): MC 41/42/43/44 --
  four simple saddles -- were absorbed into a single catastrophe whose purple
  ring enclosed all four. The user expected two monkey saddles, MC 44+42 and
  MC 43+41, each with its own ring.
- **Measured geometry**: the four form two pairs separated by **12.24°**
  (41+43, 42+44) with **63.1-64.5°** between the pairs. All four carry the same
  value (0.0202954, spread 8.9e-7 relative). There is **no core at the centre of
  the four**: the nearest positive cluster, MC25, is 29.3-34.6° from each pair
  midpoint, i.e. 2.4-2.8 pair-widths away.
- **Root cause (two compounding defects)**:
  1. **The coreless monkey saddle was not modelled.** A k=3 monkey saddle
     (chi = -2) unfolds two ways: on a smooth extremum into a core (chi = +1)
     plus three saddles, and on a flat ridge/valley into **two** simple saddles
     whose indices sum to -2 with no core at all. `classify_catastrophes`
     searched only from `local_euler_index > 0` cores, so the second channel was
     invisible. MC25 reached into the 12° search window, grabbed three of the
     four saddles, and reported a bogus constellation.
  2. **k=4 was matched by first fit, not tightest fit.** With four saddles in
     range the quad loop took the first index-balancing combination, so two
     disjoint pairs were merged into one octupole spanning 64.5° -> ring radius
     35.6°.
- **Fix** (`src/catastrophe_classifier.py`):
  - `find_saddle_pairs` (new): bound saddle pairs with `chi = -1 + -1 = -2`.
     Gates: separation <= `pair_max_pitch` (5.0) pitches, relative value barrier
     <= `barrier_tol`, no extremum on the geodesic between them
     (`has_intervening_extremum`), and **coreless** -- nearest positive core
     farther than `coreless_min_ratio` (1.8) x pair separation. Greedy,
     tightest-first, each saddle consumed once.
  - `split_quad_into_pairs` / `is_bimodal_quad` (new): the min-sum perfect
     matching of four saddles into two pairs. A true octupole has four
     comparable gaps; two independent monkey saddles have a small intra-pair
     angle and a large inter-pair gap. A quad is refused as k=4 when
     `min_inter >= bimodal_ratio` (2.5) x `max_intra` and `max_intra` is within
     the pair window.
  - `constellation_span_deg` (new) + combo search rewrite: within each fold
     order the **tightest** index-balancing combination now wins.
  - `evaluate_constellation_metrics` and `build_catastrophe` accept
     `core=None`; `net_euler_index` for a pair is the summed member index (-2),
     which equals `1 - k` at k=3.
- **Why `coreless_min_ratio` is load-bearing**: without it the new channel
  *stole* flanking saddles from legitimate cores. Measured before/after on 10
  field/file combinations, the guard is the difference between a fix and a
  regression:

  | field | pairs found | coverage lost |
  |---|---|---|
  | Pd.plt negative mean curvature | 10 (ratio 0.62-0.96) | 10 clusters |
  | Pd.plt Willmore energy | 3 (ratio 0.86) | 3 clusters |
  | ethene.plt Electron Density | 2 (ratio 0.72) | 6 clusters |
  | ethene3.plt mc sign-change arc | 6 (ratio 0.86-1.40) | 4 clusters |
  | **Pd.plt V (condensed)** | **2 (ratio 2.40, 2.79)** | **0** |

  Every spurious pair sits at ratio <= 1.40; the two real ones at >= 2.40. The
  threshold 1.8 separates them with margin, and it is scale-free -- a ratio of
  angles, so it needs no per-field tuning.
- **Result on `Pd.plt` / `V (condensed)`**: 1 catastrophe (E-4SAD, span 64.5°,
  ring 35.6°, 5 clusters) -> **2 catastrophes** (E-MSAD, span 12.2°, ring
  **9.4°**, 2 clusters each): `['MC42','MC44']` and `['MC41','MC43']`. MC25 is
  released back to standalone reduction.
- **Independent physical confirmation**: profiling the new 9.4° rings through
  `profile_boundary_loop` yields **3 valley + 3 ridge ports, topologically
  balanced** on both -- the exact 6-port signature of a k=3 monkey saddle. The
  geometry was right before any threshold was tuned for it.
- **Blast radius**: across Pd.plt, Pd_20K.plt, ethene/2/3.plt and all 13 fields,
  cluster coverage is unchanged or improved (ethene Electron Density 92 -> 96,
  Gaussian curvature 63 -> 65, Pd Willmore 59 -> 61 as the tightest-fit search
  replaced first-fit). No field loses a cluster.
- **Validation**: 5 new tests in `tests/test_catastrophe_classifier.py` --
  bimodal detection on the real Pd geometry, true-octupole acceptance, the
  four-cluster precondition, the end-to-end two-monkey-saddle outcome with
  per-pair ring radii, and an A/B proving the coreless guard protects core
  constellations. `uv run pytest`: **431 passed**. `ruff format` / `ruff check`
  / `mypy gba_topology2/ tests/`: clean.
- **Open next**: the three new thresholds (`pair_max_pitch`, `bimodal_ratio`,
  `coreless_min_ratio`) are module defaults, not yet exposed as sidebar
  controls. `cat_barrier_tol` and `cat_angular_tol_mult` already are; these
  should join them so the pairwise channel can be swept live on the Pd fields.

### [Entry 039] Geodesic Primitives Promoted to `sphere_frame.py`

- **Objective**: Give the new maximum-connector a single, public home for
  great-circle math, instead of a third private copy.
- **Changes**: `sphere_frame.py` gains `unit`, `tangent_project`,
  `rotate_towards` (the geodesic exponential map, moved from
  `saddle_launch._rotate_towards`) and `angular_distance` (clipped `acos` of the
  normalised dot, so coincident/antipodal points never yield NaN). The
  on-sphere formulas belong in the module that owns the centred frame: they need
  `p/||p||` to be the outward normal, exactly like every angular computation
  already documented there. `saddle_launch` keeps thin aliases
  (`_unit = unit`, etc.) so its call sites and any historical import read the
  same.
- **Validation**: 6 new tests in `tests/test_sphere_frame.py` -- normalisation
  and the degenerate zero, tangency, on-sphere + monotone advance, the
  non-positive-angle identity, the basic 0 / pi/2 / pi cases, and a
  chained-vs-single step test proving the map is a true geodesic (30 steps match
  one step to 1e-9).

### [Entry 040] Maximum Connection Subroutine (Basin Locator Step 3)

- **Objective**: For each ascent launch branch of a visible saddle, extrapolate a
  great-circle ray and claim the maximum it points at.
- **New module `src/max_connection.py`**:
  - `connect_seed_to_maxima(seeds, maxima, sphere_radius, delta_theta_mesh_rad,
    center, max_arc_pitch, direction_mode)` -> one `MaxConnection` per
    (seed, ascent branch).
  - **Matching rule = cross-track minimum, capped arc.** In the frame
    `(o, d, n)` with `n = o x d`, a target scores `along = atan2(t.d, t.o)`
    (signed progress) and `cross = asin(t.n)` (perpendicular deviation). The
    winner is the smallest `|cross|` among `0 < along <= cap`. Ties break on
    `along` then `max_id`, and the candidate list is pre-sorted by id, so the
    result is a pure function of the input SET, not its order.
  - **Why not plain proximity**: "nearest maximum to the launch point" ignores
    direction. On a dense field the nearest maximum is almost always off to the
    side of where the branch points, and two opposite branches of one saddle
    collapse onto the same extremum. `test_cross_track_beats_plain_proximity`
    pins a fixture where the nearer maximum is 90 deg off-ray and must lose.
  - `mark_mutual` flags the tightest claim on each maximum (one per maximum,
    connected only). Reported, never gating -- a non-mutual claim is the crossing
    separatrix signal.
  - **Frame contract**: seeds are centred, `BasinCenter` positions are world.
    `center` maps the latter in, the whole computation runs centred, arcs are
    emitted world-frame. This is the `extract_basin_centers` fix pattern; Pd_20K
    sits at the origin so a frame bug is invisible there --
    `test_world_frame_and_centred_runs_agree` runs identical geometry offset and
    requires identical metrics, and asserts the offset arc is the centred arc
    translated by the nucleus.
- **The two ray definitions are the same great circle (measured).** `chord`
  starts at the saddle centre aiming at the port; `port_tangent` starts at the
  port along `_outward_tangent`. `_outward_tangent` builds the port direction as
  `radial*port - centre`, which is the parallel transport of the centre->port
  geodesic, so both rays lie in the plane spanned by centre and port and share
  its pole. They differ only in the arc's start: `cross_track` (the quantity the
  rule minimises) is identical, `along_track` shifts by the ring radius.
  **Measured on the three saved Pd_20K snapshots** (V 24, curvedness 72, shape
  index 96 branches): identical claim sets, cross-track equal to 1e-11 deg.
  `test_both_modes_share_the_same_great_circle` pins the pole dot = 1 and the
  along-shift = ring radius.
- **Consequence for the UI**: because the mode cannot change an answer, it is an
  API parameter (it will matter to the Step 4 integrator, which starts at the
  port), NOT a viewer control. AGENTS.md Principle 4: a knob that cannot change a
  measurement does not ship.
- **Arc cap `MAX_ARC_PITCH = 40.0`** (~64 deg at Pd_20K pitch 1.6058). Measured
  max saddle->claimed-maximum separation across the three snapshots: 57.1 / 55.2
  / 45.6 deg, so the cap admits every observed connection while excluding the far
  side of the sphere where cross-track stops discriminating. A ceiling, not a
  tuned threshold.
- **Validation**: 24 tests in `tests/test_max_connection.py` -- ray bearing,
  unit/tangency, the two modes' distinct starts, the closed-form
  `along = atan2(sc,c)` / `cross = asin(s*sin az)` decomposition (cross is NOT
  the azimuth offset; it shrinks with sin(colatitude), which is the correct
  great-circle distance -- the first draft of the test asserted the naive
  azimuth reading and failed), cross-track-beats-proximity, opposite branches
  claim opposite maxima, arc cap -> `no_target`, rear hemisphere never claimed,
  ascent-only, minima filtered, mutual uniqueness, arc points on the sphere, arc
  is a great circle (zero torsion), world/centred agreement, order independence
  (8 shuffles), degenerate centre, empty inputs, summary finiteness, verified
  flag.

### [Entry 041] Params Schema v3 + Tolerant Legacy Load; `connect` Stage

- **Objective**: Make the new observation reproducible headlessly, and keep the
  existing v2 snapshots loadable.
- **The trap**: adding a control to `DEFAULT_UI_STATE` is breaking --
  `load_params_from_dict` requires every control key, so the three attached
  `Pd_20K_atom1_params_*.json` (v2) would fail with a confusing "Missing
  parameter 'show_max_saddle_arcs'". The version-2 comment in `params_schema`
  predicted exactly this failure mode.
- **Fix**: `PARAMS_SCHEMA_VERSION = 3` and a new
  `PARAM_LEGACY_SCHEMA_VERSIONS = frozenset({2})`. A legacy snapshot fills absent
  controls from `DEFAULT_UI_STATE` (a control invented after a file was written
  had exactly one value on screen: its default). The **current** version stays
  strict -- a truncated v3 file is a bug to surface, not a gap to paper over.
  `test_current_version_stays_strict_about_missing_controls` pins the asymmetry.
- `PARAM_METRIC_KEYS` gains `connection_stats`. `PARAM_STRING_KEYS` is now
  derived from the str-valued defaults (empty today; every selector is
  data-dependent) so the first str widget self-registers.
- **`diagnose_params.py`**: new `connect` stage after `launch` (consumes
  `ctx.seeds + ctx.unverified_seeds` and the `extract_basin_centers` output,
  mirroring the viewer). `RunContext` gains `unverified_seeds`, `connections`,
  `connection_stats`; `computed_metrics` and `dump_results` carry them.
- **`inspect_max_connections.py`** (new probe, `inspect_basin_centers.py`
  pattern): prints the per-branch table (saddle, branch, maximum, cross, along,
  sep, mutual/verified flags), sortable, with `--mode` / `--max-arc` overrides
  that vary the module constants without touching the saved controls.
- **Validation on the three snapshots**: reproduce check **47/47 match, exit 0**
  on all three (the differ walks the SAVED structure, so a new metric key is
  reported as new-only and cannot cause false drift). V: 24 branches, 24
  connected, 6/6 maxima, cross mean 4.36 deg, max 7.82. Curvedness: 72/72, 12
  maxima, mean cross 0.38. Shape index: 96/96, 26 maxima, mean cross 0.50. Every
  ascent branch claims a maximum; cross-track stays under 8 deg everywhere.

### [Entry 042] Viewer Overlay: Arcs From the Selected Maximum

- **Objective**: Draw great-circle arcs from a *selected* maximum to the saddles
  that claim it.
- **Changes** (`sphere_viewer.py`):
  - Actor built by the shared `_make_seed_actor(renderer, MAX_ARC_RGB,
    opacity=MAX_ARC_OPACITY)` factory. Gold (the basin-center hue) so an arc
    reads as belonging to the selected extremum, semi-transparent because the
    arcs span tens of degrees straight through the seed field.
    `test_max_arc_actor_uses_the_shared_gold_constant` forbids a hardcoded RGB.
  - `update_topology` stage 10 fills `runtime_max_connections`,
    `state.max_connections_list` and `state.connection_stats` for ALL branches
    (cheap, and the sidebar lists it). The arcs are selection-driven, so
    `_emit_max_arcs(basin_id)` is called from `select_basin_center`, not the
    rebuild -- a click is a render-only change. `update_topology` re-invokes
    `select_basin_center` at the end so a rebuild refreshes the overlay against
    the new table.
  - The arc points arrive world-frame; the 1.020 lift is applied about the
    nucleus (`to_local` -> `radial_offset` -> `to_global`), the correct order per
    the origin-centred audit.
  - `show_max_saddle_arcs` toggle in `DEFAULT_UI_STATE` (auto-registers in
    `PARAM_BOOL_KEYS`), added to `@state.change`, hidden in `reset_defaults`.
  - Sidebar: a "Basins: Step 3 Max Connections" card at drawer indent 16 (the
    Step 2 card's indent), with counters, a cross-track/separation readout, and a
    per-arc table for the selected maximum. Selecting a minimum, or nothing,
    shows a hint instead -- minima have no ascent connections.
  - `test_provisional_seeds_share_the_hue_and_differ_only_in_opacity` counted
    every solid `_make_seed_actor` call and broke at 3; narrowed to the
    ASCENT/DESCENT hues, since the gold overlay is not a seed colour.
- **Validation**: headless smoke (the `s.start` stub recipe) reaches
  `server.start`, so the full VTK + Vuetify layout constructs. `pytest`: 466
  passed. `ruff` / `mypy gba_topology2 tests` / `pre-commit --all-files`: clean.
- **Open next**:
  1. The connector is a straight great circle. Step 4 replaces it with the
     traced separatrix; the geodesic is the baseline to diff the integrator
     against.
  2. `direction_mode` is dormant (both modes agree on the excision substrate).
     It becomes load-bearing the moment the integrator starts at the port.
  3. Cross-track is reported, not gated. On the V field the spread is 0-8 deg;
     a cone gate is deferred until the traced path shows what a "real" cross-track
     looks like.
  4. Two ascent branches from one saddle can claim the same maximum (both are
     distinct gateways). Kept; `n_branches` per pair is reported. Dedupe only if
     the traced version proves them redundant.

### [Entry 043] Step 3 Rule Fixed: First-Hit Along the Ray, Not Best-Aim

- **Objective**: The connector claimed maxima a ray passed *through* on the way.
  Add the constraint that a connector cannot cross another critical point.
- **The bug, in one line**: the rule minimised `|cross_track|` with `along` as a
  *tie-break*. That scores **aim** and never checks **reach**. A ray passing
  through a near maximum toward a far one has `cross ~ 0` for BOTH (they are on
  the great circle by construction), so cross-track is *provably* unable to
  separate them and the tie-break was decided by float noise at 1e-11 rad.
  Three contributing defects: the pool was maxima-only (so "passes through a
  critical point" was inexpressible), the 40-pitch cap was the only distance
  gate, and the drawn arc is the saddle->winner geodesic, so the renderer
  faithfully drew the occlusion the matcher ignored -- which is what made it
  visible.
- **My validation missed it**: I checked aggregate counters and schema plumbing
  on the earlier snapshots and never asserted the first-hit property or eyeballed
  one arc against the glyph field. Worse, `test_cross_track_beats_plain_proximity`
  actively *pinned* the wrong behaviour. It tested "direction beats proximity"
  and was blind to "proximity beats direction when direction is degenerate".
- **The fix**: `connect_seed_to_maxima` now minimises **along-track** among
  **all visible critical points** inside the forward sector
  (`0 < along <= cap`, `|cross| <= cone`). If the first hit is a maximum the
  branch CONNECTS; if it is a minimum or saddle the branch is **OCCLUDED** and
  the blocker's id + polarity are recorded. Ties break on `|cross|` then id.
- **Both requested parameters collapsed into one, and the second vanished.**
  The score-then-test design needed a cone *and* an on-ray tolerance `eps`.
  First-hit needs neither tolerance: a maximum exactly on the ray is both the
  nearest hit and its own occluder, so the two rules agree by construction.
  `eps` is gone. The cone remains as one constant.
- **`sector_width_rad` measured UNUSABLE as the cone** (my proposed
  parameter-free option, and I was wrong to propose it). It is the gap between
  bounding separatrix crossings and degenerates whenever a sector contains all
  of them: on Curvedness **40 of 72 branches report 0.0 deg** and several report
  **355-357 deg**; on V four report 352.6 deg. So the cone is the pitch-relative
  `CONE_PITCH = 6.0` (9.63 deg at Pd_20K pitch 1.6058), chosen as the smallest
  pitch multiple admitting every cross-track observed on the four snapshots
  (max 7.82 deg = 4.9 pitches, on V).
- **New pool builder `build_connection_pool`**: basin centers carry extrema only,
  so the visible saddles are added from the two places they live -- raw Morse
  saddles not in `hidden_vertex_ids` (CENTRED frame) and `E-SAD` entries of the
  effective-CP dicts (WORLD frame, no vertex to scan). The frames are reconciled
  with `to_global` inside the builder. This is the recurring effective-CP trap;
  the test asserts the offset is applied.
- **Validation on `Pd_20K_atom1_params_Shape2.json`** (shape index, k_pitch 2.2,
  k_bridge 2.75, 96 ascent branches, pool 98 visible CPs = 26 max + 24 min +
  48 sad). Id mapping confirmed: **MC139=SAD_384, MC140=SAD_385, MC143=SAD_388,
  MC144=SAD_389** (all chi=-1 single-saddle clusters).
  - **24 of 96 branches changed claim.**
  - **`E-MAX_MC109` goes from exactly 2 connectors to 4** -- the count the user
    reported as missing. It gains `SAD_384` (MC139) and `SAD_385` (MC140).
  - The over-claiming maxima collapse from 8 connectors to 4: MC7, MC8, MC111,
    MC112, MC113, MC114. Mean separation 31.3 -> **20.1 deg**, max 57.1 ->
    **33.0 deg**. Connections became local, which is what a separatrix is.
  - Reproduce check **56/59**: the 3 drifts are exactly
    `mean/max_separation_deg` and `mean_cross_track_deg`. Every other metric
    matches, i.e. the change is confined to Step 3. This is the tool working --
    a rule change that did NOT move these numbers would be the surprise.
  - **Occlusion fires on Curvedness: 20 of 72 branches blocked, all by minima**
    (0 by saddles). On V and both shape-index snapshots, 0 occluded.
- **The specific expectation did NOT survive measurement, and the trace says
  why.** The report was that MC143/MC144's uphill directions walk through MC109.
  Measured: **MC109 is 90.0 deg from MC143 and 89.95 deg from MC144** -- it is
  perpendicular, not behind. `--trace SAD_388` shows MC109 at
  `along = -94, cross = +90` on both branches: behind the ray and 90 deg off it,
  unreachable by any forward-sector rule. What MC143/MC144 actually claim is
  `E-MAX_MC110` (along 15.3/16.1) and `MAX_4`/`MAX_5` (along 19.5). MC109 and
  MC110 are 90.08 deg apart, so the two are the symmetric partners of a quartet,
  and the fix moved MC109's connectors from {MC139, MC140} to
  {MC139, MC140, E-SAD_MC73, E-SAD_MC77}. The count the user wanted is now
  correct; the attribution is to MC139/MC140, not MC143/MC144.
- **`--trace` added to `inspect_max_connections.py`**: scores every pool CP
  against each branch of one saddle, in-sector first, with the verdict column
  showing the blocker. This is what turned "the rule is wrong" into a number.
  It had its own instance of the frame bug -- it scored world-frame pool
  positions against centred-frame rays, invisible on origin-centred Pd_20K --
  caught on review and fixed by passing `center` and mapping with `to_local`.
  Verified: the trace's top in-sector entry equals the rule's claim on every
  branch of SAD_384/388/389.
- **UI**: the Step 3 card chip now reads "N/M connect", gains an occlusion line
  (blocked-by-minimum / -saddle / no-target counts), the per-arc table gains an
  **Along** column (the primary key), and the explanatory text describes
  first-hit. `MaxConnection` gains `blocker_id` / `blocker_polarity`;
  `mark_mutual` switched to `dataclasses.replace` so it cannot drop fields when
  the dataclass grows (it would have silently lost the blocker on the first
  version of this change).
- **Validation**: 34 tests in `tests/test_max_connection.py` (was 24) --
  first-hit-beats-farther-on-axis (the MC109/MC8 inversion of the old proximity
  test), minimum-on-ray occludes, saddle-on-ray occludes, source saddle is never
  its own target, cone excludes an off-sector nearer target, 5 pool-construction
  tests (merge, hidden exclusion, E-SAD inclusion, world-frame mapping, dedup),
  mark_mutual preserves blockers, 3-status summary. `pytest`: **477 passed**.
  `ruff` / `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless
  viewer smoke reaches `server.start`.
- **Open next**:
  1. `CONE_PITCH = 6.0` is calibrated on Pd_20K only. The V field's 7.82 deg
     max cross-track is 4.9 pitches, so 6.0 has ~20% headroom there and is
     untested on a coarser mesh.
  2. 0 occlusions on V/shape-index is a real result, not a gap: with 26 maxima
     and 48 saddles in a 9.6 deg sector, the first hit is nearly always a
     maximum. The mechanism is proven on Curvedness.
  3. The straight great circle is still the connector. Step 4's integrator
     replaces it; `direction_mode` becomes load-bearing when it starts at the
     port.

### [Entry 044] Step 3: Launch From the Port, Claim the Disc, Not the Centre

- **Objective**: Monkey saddles on the Curvedness field lost connections (MC18
  had no arc to MC38). The user's model: propagate from the point where the
  launch direction crosses the excision boundary, and treat a maximum as reached
  when the arc meets its boundary.
- **The user's diagnosis was correct on both counts, and I had it wrong twice.**
  My Entry 043 answer ("the geodesic clips a minimum the true ridge goes around,
  only Step 4 fixes it") was wrong. Two real defects, both mine:
  1. **The ray started at the saddle CENTRE, not the port.** `direction_mode`
     defaulted to `"chord"`. The disc inside the excision boundary is the
     degenerate core whose field is meaningless -- the entire reason the boundary
     exists -- and launching from the centre means the connector's first ~ring
     radius traverses it, so any CP the ring swallowed sits on the ray and
     occludes. From the port, every CP inside the boundary has `along < 0` and is
     excluded by construction.
  2. **The rule scored the CP's CENTRE, not its disc.** A minimum at
     `along 7.5, cross 7.4` with a 1.6 deg disc blocked MC18's branch 0 from
     MC38 at `along 45, cross 0.4` with a 2.9 deg disc. The ray passes 7.4 deg
     clear of a 1.6 deg object -- it is not an obstacle. It was a candidate only
     because the 9.63 deg cone is six times wider than its disc, and it won
     because its centre is nearer.
- **The fix**: `disc_entry_angle(along, cross, rho)` returns the arc-length where
  the great circle first enters the disc, `along - acos(cos rho / cos cross)`,
  defined exactly when `|cross| <= rho`. The rule minimises ENTRY, and a CP whose
  disc the ray misses is not a candidate at all. Default mode is now
  `"port_tangent"`. The arc is two-part: leg 1 along the ray to the boundary
  contact, leg 2 the geodesic from contact to centre (the user's drawing rule).
- **Measured on `Pd_20K_atom1_params_curv.json`** (curvedness, k_pitch 2.0,
  k_bridge 1.75, 72 ascent branches, pool 73 visible CPs = 12 max + 31 min +
  30 sad, disc radii 1.61-16.76 deg, median 2.91):
  - **occluded 20 -> 0, connected 52 -> 72.** Every branch now claims a maximum.
  - **MC18 branch 0 -> E-MAX_MC38**, entry 33.79, cross -0.39, disc 2.91. The
    minima that blocked it (MIN_19165/19161 at cross 7.4, disc 1.61) now read
    `miss` -- the great circle passes outside their discs.
  - MC18 is k=4 (ring 8.33 deg), 4 ascent branches, all propagate. The "only 4-5
    of 6" symptom was the connection layer dropping branches, never the seed
    layer: `n_ascent = 68` over 29 converged seeds solves to 10 k=3 + 19 k=2,
    i.e. every port emitted.
  - Reproduce check **53/62**: the 9 drifts are all `connection_stats` leaves
    (direction_mode, the 4 counts, 3 angular stats, n_unverified 0->4 from the
    provisional MC21 seed now being connected). Every non-Step-3 metric matches.
- **The cone is now provably redundant, and I am keeping it.** Cone 9.63 deg and
  144 deg give identical verdicts on this snapshot: every pool target carries a
  real disc radius (0 of 73 are points), so the per-object `|cross| <= rho` test
  subsumes a global gate. Same for the cap (max entry ~42 deg, cap 64 deg). Both
  stay as the `no_target` mechanism for genuinely empty sectors and to bound the
  search, and as the fallback if a target ever has radius 0. Recorded so the next
  tuning pass knows they are not load-bearing here.
- **Frame safety**: `build_connection_pool` takes `ring_radius_by_vertex` (raw
  CPs, keyed by mesh vertex, from `MicroCluster.angular_radius_rad`) and
  `ring_radius_by_id` (effective CPs, the viewer's `eff_ring_radius`). Verified
  on **ethene4 atom 1, an off-origin atom** -- reproduce 35/35, so the centred
  raw / world effective merge is correct. Pd_20K sits at the origin and would
  have hidden this.
- **A latent bug caught in the probe**: `for *_, s in scored` silently bound `s`
  to the newly appended 6th tuple field (disc radius) instead of the in-sector
  flag. Star-unpacking is positionally fragile; the trace now indexes `r[4]`.
  `--trace` also had its own instance of the frame bug (world pool vs centred
  rays), fixed in Entry 043 and re-verified.
- **Validation**: 33 tests in `tests/test_max_connection.py` -- `disc_entry_angle`
  on-axis / tangent / miss / zero-radius-degenerate; port-origin starts at the
  port and puts an in-ring CP at `along < 0`; first-hit on the disc; minimum and
  saddle discs occlude; **the MC18 off-ray-minimum case**; source-saddle skip;
  cone gate; cap; rear hemisphere; two-part arc endpoints (port first, centre
  last, all on the sphere); 5 pool tests now asserting disc radii; world/centred
  agreement; order independence. `pytest`: **476 passed**. `ruff` /
  `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless viewer
  smoke reaches `server.start`.
- **Open next**:
  1. The straight great circle is still the connector. The 20 occlusions it used
     to report were artefacts of the centre-point test, NOT real curvature
     effects -- so the "geodesic vs separatrix diverge" case is now UNMEASURED.
     Step 4's integrator is the baseline test for that.
  2. `CONE_PITCH` / `MAX_ARC_PITCH` are non-binding on curvedness. Untested on a
     field with sparse maxima, where a wide cone could admit a genuine off-ray
     claim.
  3. The arc's leg 2 (contact -> centre) is a drawing convention. Once Step 4
     traces the separatrix, the whole arc should come from the integrator.

### [Entry 045] Click a Glyph, Light the Row: Selection-to-Table Resolvers

- **Objective**: Restore the capability where clicking a CP / effective CP in the
  3D scene highlights the corresponding cluster and its designation in the
  sidebar. Clicking a maximum should say *which* maximum it is, and with the arc
  overlay on, reveal its saddle connections.
- **The gap was narrower than it looked.** `select_cp` / `select_cluster` /
  `select_basin_center` all set `state.selected_*`, and those drive the
  **inspector cards** below the tables. The table **rows** carried a static
  `classes="cursor-pointer"` with no binding to any selection state, so a glyph
  click opened the card and never lit the row. Nothing was lost to restore --
  the row highlight was never wired.
- **Three pure resolvers** at module level, so the id-space joins are testable
  without trame, a render window, or a browser:
  - `basin_center_id_for_cp` -- a raw extremum joins on `MAX_<v>` / `MIN_<v>`
    from `vertex_id`; an effective CP joins on its own `id_label`. A saddle, a
    hidden CP, and an E-SAD have no basin center and resolve to None.
  - `cluster_id_for_cp` -- a raw CP scans `members[].vertex_id`; an effective CP
    matches the row `id` against its `origin_id`. A CAT-prefixed origin lives in
    the catastrophe table, so it resolves to None here.
  - `basin_center_id_for_cluster` -- tries `E-MAX_<unit_id>` / `E-MIN_<unit_id>`
    first (the visible glyph), then falls back to the members' raw `MAX_<v>`.
- **The insight that made it work: wire the CLUSTER path, not just the CP path.**
  `on_scene_click` tests cluster centroids *before* CP glyphs, so clicking a
  maximum glyph usually lands on `select_cluster`, not `select_cp`. Wiring only
  `select_cp` -- the obvious reading of "clicking on a cp" -- would have fixed
  the rare case and missed the common one. Both now resolve a basin center.
- **The arc reveal is free.** `select_basin_center` already calls
  `_emit_max_arcs`, which gates on `state.show_max_saddle_arcs`. So driving the
  basin-center selection from a glyph click gives exactly the requested
  behaviour: toggle on -> the maximum's row lights AND its saddle arcs appear;
  toggle off -> the row lights and the micro-cluster is highlighted, no arcs.
  No new branching was needed, and none was added.
- **One-way edge, guarded.** `select_cp` -> `select_basin_center` and
  `select_cluster` -> `select_basin_center`; `select_basin_center` calls neither,
  so there is no cycle. Each compares the resolved id against the current
  selection first, so re-clicking the same maximum does not rebuild the arcs.
- **No custom CSS is possible.** Vue strips `<style>` from component templates
  (documented at the theme block: that is why the drawer has a Vuetify theme
  instead of CSS overrides). The highlight is therefore a Vuetify utility class,
  `bg-amber-lighten-4`, bound with `:class`. Trame emits a dynamic binding only
  for a **tuple** attribute value -- a plain string is emitted verbatim as a
  static `class=` -- so each use site is `classes=(EXPR,)`. The three JS
  expressions are module constants (`CLUSTER_ROW_CLASSES`,
  `EFFECTIVE_CP_ROW_CLASSES`, `BASIN_CENTER_ROW_CLASSES`) shared by the rows.
- **A separate `highlighted_cluster_id` key, not `selected_cluster`.** A CP click
  should light the cluster *row* without opening the cluster *inspector card*
  (which would cover the CP card the operator just asked for). The id key drives
  the row; `selected_cluster` keeps driving the card.
- **Validation on `Pd_20K_atom1_params_curv.json`** (43 basin centers, 74 unit
  rows, 972 CP dicts):
  - all **12 of 12** E-MAX CPs resolve to their own `id_label`, 0 failures;
  - **972 / 972** CPs resolve to an owning cluster row;
  - **43 / 43** basin centers are reachable from some CP, and 43 of 74 unit rows
    resolve to a basin center. The split is exactly the reduction decision:
    every resolved row has `local_euler_index == +1` (an extremum set), and all
    31 unresolved rows have chi in {-3, -2, -1, 0} -- saddle sets and annihilated
    chi=0 sets, which correctly light no extremum row. The 266 maxima members
    inside unresolved rows are absorbed into a saddle/corridor whose visible
    glyph is the E-SAD, not a basin center;
  - `E-MAX_MC38` resolves in both directions (CP -> basin, cluster -> basin).
- **Validation**: 30 tests in `tests/test_sphere_viewer_selection.py` -- the
  three resolvers over raw/effective/saddle/hidden/absent inputs, falsy and
  non-dict guards (matching the `_selection_id` idiom in `params_schema`), plus
  structural asserts that `select_cp` and `select_cluster` call the resolvers and
  that each table binds its row with a tuple. `pytest`: **506 passed**. `ruff` /
  `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless viewer
  smoke reaches `server.start`, so the three re-bound rows construct.
- **Open next**:
  1. There is no raw-CP table -- only effective CPs are tabulated
     (`all_cps_list.filter(c => c.is_effective)`). Clicking a RAW maximum lights
     its Extrema row and its cluster row, which answers "which maximum is this",
     but there is no CP row to light.
  2. A multi-member cluster containing several extrema resolves to the first
     basin-center match, i.e. the highest-value one, since the list is sorted by
     value descending. Arbitrary but stable; revisit if it misleads.
  3. `select_catastrophe` does not resolve a basin center. A catastrophe is a
     constellation, not an extremum, so there is nothing to light -- confirm that
     is the desired behaviour rather than an omission.

### [Entry 046] Step 3: Cone Captures the Target, Disc Blocks Only Obstacles

- **Objective**: On Pd_20K shape index, maxima surrounded by three saddles claimed
  only 1-2 connections. MAX_7 (3) and MAX_4 (1) are symmetric partners, so the
  pattern demanded identical answers and did not get them.
- **Root cause: the disc gate applied to the TARGET was tuned backwards.** Entry
  044 made the rule "claim the first CP whose excision disc the ray enters", and
  applied that to maxima as well as obstacles. Every raw maximum's excision
  radius is exactly ONE MESH PITCH (1.61 deg at pitch 1.6058), while the
  legitimate rays into a 3-saddle maximum arrive at cross-tracks
  `0.0, 1.35, 1.42` (MAX_7) and `0.0, 2.00, 1.98` (MAX_4). A gate at 1.61 sits
  INSIDE that band, so it accepted all three of MAX_7's rays and rejected two of
  MAX_4's. Identical geometry, different answers, decided by measurement noise.
  Connector counts across the eight raw maxima, MAX_0..MAX_7: **1, 2, 2, 3,
  1, 2, 2, 3** -- two identical symmetry classes of four, each spanning 1 to 3.
  After the fix: **3, 3, 3, 3, 3, 3, 3, 3**.
- **The fix separates the two roles.** Walk the candidates in increasing
  along-track and stop at the first decisive one:
  - a **maximum** ends the walk and the branch CONNECTS. Captured by the CONE,
    which is the directional criterion the sector already encodes. The arc still
    stops at the excision boundary where the ray crosses it, and at the centre
    when the ray passes outside -- the drawing rule from Entry 044 is preserved.
  - a **minimum or saddle** ends the walk as an OCCLUSION only if the great
    circle enters its disc (`|cross| <= rho`). The disc test is retained exactly
    where Entry 044 needed it: MC18's near-miss minimum at cross 7.4 with a
    1.61 deg disc is not an obstacle.
- **A second, independent bug surfaced on the way, and the repo had already
  solved it once.** The 8 occlusions on this snapshot were corridor MS13
  blocking rays at cross 3.4. An excision boundary around an elongated set is a
  fitted ELLIPSE: MS13 has a = 11.35, b = 2.43, aspect 10.6. Using the semi-major
  as a disc inflates the transverse reach by the aspect ratio, so the corridor
  occluded a ray passing 3.58 deg off its axis (3.58 < 11.35, but 3.58 > 2.78).
  `string_aggregator.cluster_directional_support_radius` exists for exactly this
  and its docstring names the failure: an isotropic radius "turns a long thin
  corridor into a disk of radius a and lets it swallow compact fragments sitting
  far off to the side". Extracted the ellipse math to
  `max_connection.directional_support_radius` (pure, no MicroCluster dependency)
  and added `ConnectionTarget.radius_toward()`.
- **The pool now carries the ellipse, not a scalar.** `DiscGeometry`
  (angular/semi-major/semi-minor/principal axis) replaces the two
  `ring_radius_by_*` maps with `disc_by_vertex` / `disc_by_id`, built by
  `disc_geometry_from_cluster`. Catastrophes are isotropic rings, so their
  geometry round-trips with a == b. `basin_centers_to_targets` is the single
  implementation of the conversion; `build_connection_pool` delegates to it.
- **Validation on `Pd_20K_atom1_params_shape.json`** (shape index, k_pitch 2.0,
  k_bridge 2.25, 96 ascent branches, pool 98 visible CPs):
  - **all 8 raw maxima now claim exactly 3 connectors** -- MAX_0..MAX_7 = 3 each,
    from 1, 2, 2, 3, 1, 2, 2, 3. Spread over all 26 maxima: 3-4.
  - verdicts 96 connected, 0 occluded, 0 no target (was 88 / 8 / 0).
  - mean separation 20.07 deg, max 32.99 -- connections stay local.
- **Regressions held**: curvedness stays 72 connected / 0 occluded and MC18
  branch 0 still claims E-MAX_MC38 (entry 33.79, cross -0.39). ethene4 atom 1,
  the off-origin case, reproduces 35/35.
- **Validation**: 40 tests in `tests/test_max_connection.py` (was 33) --
  `directional_support_radius` along-axis vs transverse; a target captured at
  cross 2.0 with a 1.61 deg disc; the symmetry fixture driving rays at cross
  0.0 / 1.4 / 2.0 and asserting all three connect (verified NON-VACUOUS: the old
  disc gate rejects the 2.0 case); an obstacle still needing its disc entered;
  arc stops at the boundary when hit and at the centre when missed; pool
  carrying DiscGeometry. `pytest`: **513 passed**. `ruff` /
  `mypy gba_topology2 tests` / `pre-commit --all-files` clean. Headless viewer
  smoke reaches `server.start`.
- **Process note**: I diagnosed this twice before measuring. Entry 044's
  "occlusion fires as designed" was the design being wrong, and my first fix
  (directional radius) was correct but did not move a single connector count --
  caught only by re-running the counts, not the occlusion total. The metric that
  found it was per-maximum connector counts across a symmetric set, which is the
  test for any rule with a geometric threshold: **symmetry is a free oracle.**
- **Open next**:
  1. `CONE_PITCH = 6.0` is now the sole directional gate for targets, so it is
     load-bearing where the disc used to share the job. Calibrated on Pd_20K
     (max observed cross-track 7.82 deg = 4.9 pitches). Untested on a coarser
     mesh or a sparse-maxima field.
  2. The 1.61 deg = one-pitch excision radius for raw maxima is a property of
     `compute_micro_clusters`, not of this rule. Worth checking it is intended.
  3. Step 4's integrator replaces the straight ray; the cone becomes the
     separatrix's own sector.

### [Entry 047] Step 3 Final Connector: One Great Circle, Centre to Centre

- **Objective**: The connectors bent sharply where they met the maximum's
  excision boundary. The search had found the two endpoints it needed; the drawn
  path should be the single geodesic between them, not the three-segment route
  the search travelled.
- **The change**: `arc_points` is now `_arc_between(centre_unit, hit.position, ...)`,
  the shortest geodesic from the saddle cluster centre to the maximum cluster
  centre. The previous `_arc_to_disc` (leg 1 along the launch ray to the boundary
  contact, leg 2 a geodesic from contact to centre) is deleted; it had exactly
  one caller. Gold is unchanged -- the viewer already colours the arc with
  `MAX_ARC_RGB` and iterates `arc_points` without inspecting its structure, so
  the viewer needed no change at all.
- **The reframing, which is the real content of this entry**: the port, the
  launch bearing, the boundary crossing and the cone are a SOLVER whose job is to
  find two endpoints. They are not part of the connector. `along_track_deg` keeps
  recording the boundary entry as the ranking key and as "how far the ray
  travelled before it reached the basin", but it is no longer a vertex of the
  drawn line. Everything before this was the means to find the endpoints.
- **Narrowing `centre_unit`**: it is `unit(seed.saddle_position)` and mypy saw a
  `None`. It is provably non-None at the call site -- `connection_ray` returns
  `None` exactly when the centre is at the nucleus, and the `ray is None` branch
  already continues -- so the guard is now `if ray is None or centre_unit is
  None`, which is the same degenerate seed, not a new condition. A guard rather
  than an `assert`, so a degenerate seed degrades to `no_target` as it did before.
- **Validation that the RULE is untouched (the load-bearing check).** A/B against
  HEAD by stashing the change and re-running the connect stage on
  `Pd_20K_atom1_params_shape.json`:
  - `connection_stats` drift caused by this change: **NONE**.
  - all **96 of 96** claims identical in target, status, `along_track_deg` and
    `separation_deg`.
  This is a rendering change. If any statistic had moved, the search would have
  been disturbed.
- **Validation that the GEOMETRY did change (the positive marker).** A clean
  refactor can be a silent no-op, so the arcs were measured directly:
  - **every one of the 96 arcs starts at the saddle centre**: 64 bit-exact to
    1e-9 deg, the remaining 32 within **1.2e-6 deg**, which is float64
    round-trip noise through `acos` near 1.0. The offsets are spread across both
    raw (14) and effective (18) saddles, so they are not a systematic frame
    error -- a real offset would be one-sided and far larger.
  - **worst torsion over all 96 arcs: 1.21e-13 A** -- every sample lies in one
    plane through the nucleus, i.e. each is a true great circle.
  - **worst kink over all 96 arcs: 1.375 deg**, which is exactly the uniform
    sampling step (span / `ARC_SAMPLES`), so there is no bend anywhere.
  - The old two-leg path on the same fixture kinks **26.84 deg**. The new
    no-kink test's threshold is 1.875 deg, so it is strongly non-vacuous.
- **Tests** (`tests/test_max_connection.py`, 42, was 40):
  - `test_arc_starts_at_the_port_and_ends_at_the_centre` ->
    `test_arc_is_the_single_geodesic_between_the_two_centres` (first point is the
    pole, not the port).
  - **`test_arc_is_a_great_circle` re-added.** I dropped it in Entry 046 because
    the two-leg path legitimately was not a single great circle; the property is
    back and is now the direct assertion of this feature.
  - **`test_arc_has_no_kink_at_the_maximum_boundary` added** -- the MC20 symptom
    pinned as a regression, by measuring consecutive segment directions.
  - `test_arc_stops_at_the_boundary_...` / `..._closest_approach_...` renamed to
    `test_ranking_key_is_the_boundary_entry_...` / `..._is_the_centre_...`. They
    assert `along_track_deg`, which is unchanged; the old names claimed to
    describe arc geometry, which they no longer do.
- **Sidebar text corrected.** The Step 3 card still said the arc "runs along the
  launch direction to the boundary, then to the centre", and also still described
  the Entry 043 disc-captures-target rule. Both stale; it now states the cone
  captures targets, the disc blocks obstacles, and the connector is the single
  great circle centre to centre.
- **A note on the reported case.** `MC20` does not exist in
  `Pd_20K_atom1_params_shape.json`: its 26 visible maxima are `MAX_0..MAX_7`,
  `E-MAX_MC1..MC8` and `E-MAX_MC105..MC114`. The maxima with exactly four
  connectors are the 18 `E-MAX` sets. The sharp-turn symptom was therefore
  verified on `E-MAX_MC109` and `E-MAX_MC1` (4 connectors each, kink 0.5-0.67 deg
  = the sampling step) and across all 96 arcs, not on MC20 itself. If MC20 is
  from a different field or atom, re-check it in the viewer.
- **Validation**: `pytest` **515 passed**. `ruff` / `mypy gba_topology2 tests` /
  `pre-commit --all-files` clean. Headless viewer smoke reaches `server.start`.
  Curvedness and ethene4 reproduce as in Entry 046 (their drifts are Entry 046's,
  recorded in the snapshots written before it).
- **Open next**:
  1. `MaxConnection.arc_points` is now fully derivable from the two centres. Kept
     because it is the viewer's only geometry contract and keeps the viewer from
     needing the sphere radius and nucleus; revisit if the record grows again.
  2. Occluded branches draw nothing. A faint stub to the blocker would make "why
     is there no arc here" answerable on screen. Deferred to Step 4.
  3. Step 4 replaces the straight connector with the traced separatrix. The
     endpoints found here are its boundary conditions, and the solver's internal
     path (port -> boundary entry) is the natural seed for that integration.

### [Entry 048] Step 4: Gradient Ascent Fallback for Incomplete Branches

- **Objective**: The Step 3 connector is a straight ray, so it finds only a
  maximum FORWARD of the port. On the organic fields (ethene3) the separatrix
  leaves a saddle, curves around the sphere, and arrives at a maximum that sits
  BEHIND the launch direction. The user's diagnosis: designate launch points
  complete (a terminal maximum was found by the ray) or incomplete, and for the
  latter, follow the field gradient uphill to the maximum. Two saddles' uphill
  directions wrap all the way around to connect one maximum from two sides.
- **Is the two-stage approach sound?** Yes, and the reason is that the two stages
  answer different questions. The ray is a straight-line rule with a bounded
  reach (cap 89 deg, cone 13 deg); the gradient is the field's own statement of
  where the ridge goes. They agree wherever the ray succeeds (a dense metallic
  field: every branch resolves geodesically), and the gradient is consulted only
  where the ray reaches nothing. The plan was confirmed with the user before
  building.
- **New module `separatrix_tracer.py`** (Principle 1: a distinct algorithm gets a
  dedicated module). `trace_uphill` integrates the surface gradient by central
  differences of the IDW interpolant: at each point, an orthonormal tangent frame
  `(t1, t2)`, `g1 = f(p+h.t1) - f(p-h.t1)`, `g2` likewise, step along
  `unit(g1.t1 + g2.t2)` by one mesh pitch. Constants: `TRACE_STEP_PITCH = 1.0`,
  `MAX_TRACE_ARC_PITCH = 400.0` (~891 deg, more than a great circle, so a path
  that exhausts it is spiralling and the answer is "unresolved"),
  `STALL_PATIENCE = 6` consecutive non-rising steps. Statuses: arrived / stalled /
  budget / no_start.
- **The arrival-before-stall ordering is load-bearing.** The gradient vanishes at
  a maximum, so reaching the top and going flat is success, not a stall. The
  arrival test runs first in the loop. `centred_targets` maps the world-frame
  pool to centred unit positions before the tracer sees it, because
  `radius_toward` measures from the target's OWN position -- the recurring frame
  trap, which Pd_20K hides at the origin and ethene3 exposes at 1.2618 A.
- **Integration into `connect_seed_to_maxima`** (Phase C). The function gains
  optional `mesh_pts` / `f_vals` / `find_k_nearest`. For a branch the ray leaves
  as `no_target`, `_gradient_connection` climbs from `branch.launch_position`
  (the port, never the saddle centre -- the gradient is zero there). On arrival it
  emits `method="gradient"` with the traced polyline as `arc_points`. Occluded
  branches are NOT rescued: an occlusion is a resolved verdict, and climbing the
  field from the port would contradict the geodesic rule that produced it.
- **`method` and `failure_gate` on the record** (Phase A). `method` is
  geodesic|gradient. `failure_gate` names the test that rejected the maxima:
  behind / cone / cap / occluded / empty. This makes the two-stage design
  auditable -- "the ridge curves" (behind/cone) and "the ridge is merely long"
  (cap) are different statements about the field. `_classify_gate` scores every
  pool maximum against the ray directly (the pre-filtered candidate list cannot be
  used; the maxima it excludes are exactly the ones being diagnosed).
- **A field is not a control** (Principle 4). `mesh_pts`/`f_vals` are data the
  pipeline already has, not tunables. The fallback is inert when they are omitted,
  which is the unit-test contract and the Pd_20K no-op contract.
- **The regression invariant, verified byte-identical.** Pd_20K has zero
  incomplete branches, so the fallback must be a no-op there. A/B against HEAD
  (stash the change, re-run connect on both Pd_20K snapshots):
  - shape index: **96 of 96** connections identical in status, target,
    along/cross/separation. `n_gradient = 0`, `n_geodesic = 96`.
  - curvedness: **52 of 52** identical. `n_gradient = 0`.
  The connection dicts compare equal as tuples; the fallback changed nothing.
- **ethene3, the field the feature is for** (gradient fallback on):
  - Electron Density: 6 -> **10 of 10** connected (4 gradient, all gate=behind).
  - curvedness: 6 -> **10 of 18** connected (4 gradient reaching MAX_4864 at
    110-126 deg separation, traced along-track 115-124 deg -- past the 89 deg cap
    the ray obeys). The 4 remaining no_target are gate=behind with the climb
    stalling; the 4 occluded stay occluded (correct).
  - shape index: **0 of 4, unchanged, and this is the honest result.** The gate is
    cone (the single maximum MAX_4864 sits at cross +/-44.9 deg, outside the
    13.4 deg sector). The fallback runs and correctly stalls: the basin test (a
    prior measurement) showed 96/400 random sphere samples reach MAX_4864 but
    0/4 saddle ports do, at every stall threshold. The ports lie OUTSIDE the
    spike's ascent basin -- the four spikes are absorbed into a saddle corridor
    (MS1) and the field between them is a 10081-vertex plateau. This is a
    reduction/topology finding, not a connector bug: no connector rule can reach a
    maximum the field does not drain to from the saddle.
- **Viewer** (Phase D). A second actor `gradient_arc_poly/actor` in cyan
  (`GRADIENT_ARC_RGB`) draws the traced polylines; the gold actor draws the great
  circles. `_emit_max_arcs` splits by `conn.method` in one pass. The arcs table
  gained a "grad" chip and null guards on along/cross/sep (a gradient connector has
  no cross-track; `.toFixed` on null throws in the browser). A stats line reports
  "N by gradient ascent (cyan), M by great circle (gold)".
- **Probe** `inspect_max_connections.py`: `--no-gradient` A/B flag, a method
  column (G flag), and a "methods" summary line with the gate breakdown.
- **A `direction_mode` bug caught by the reproduce check.** The fallback first
  wrote `direction_mode="gradient"`, which overwrote the ray definition and made
  ethene3 shape index drift 61/62. `direction_mode` names the RAY (port_tangent /
  chord) and is carried through from the caller; `method` is what marks the
  fallback. After the fix: ethene3 shape index reproduces **62/62 PASS**.
- **Tests**: `tests/test_separatrix_tracer.py` (13) -- single peak, nearest of
  two, a curved ridge (two peaks, off-symmetry port; asserts the path bows >5 deg
  off the great circle), flat plateau stalls, tiny budget, degenerate start,
  visited non-maximum discs, world-frame emission about an offset nucleus,
  `centred_targets` map/drop/identity, constants, frozen outcome.
  `tests/test_max_connection.py` (47, +5) -- the behind-ray branch resolves by
  gradient with gate=behind, a stall keeps the gate, occluded is not rescued,
  no-field is inert, the summary counts the split.
- **Validation**: `pytest` **532 passed, 1 skipped**. `ruff` / `mypy gba_topology2
  tests` clean. Headless viewer smoke reaches `server.start` on ethene3.
- **Open next**:
  1. The shape-index basin is a reduction question, not a connector one: should a
     saddle corridor that swallows maxima expose its constituent spikes as
     connection targets? Out of scope for Step 4.
  2. The traced path is drawn at full fidelity; a long spiral could be decimated
     for the scene. Not needed at these budgets.
  3. Occluded branches still draw nothing (Entry 047 open item 2).

### [Entry 049] Step 5 Reverted: the Scope Is Two Stages, Not Three

- **The user's correction.** Entry 049 (connector refinement + topology audit,
  commit `cd99bbd`) was not the step we were on. The agreed scope is exactly two:
  great-circle identification, then gradient following for the INCOMPLETE saddle
  launch seeds -- climb from the excision boundary to the highest point and STOP.
  The coincidence of that endpoint with a topologically identified maximum is a
  concern to be recorded, not a defect to be corrected. Refinement is a separate
  future step, to be done a different way.
- **The revert.** `git revert --no-edit cd99bbd` -> `b48b6cf`. The refinement
  module, its 15 tests, the `refine_connectors` control, the promotion machinery
  and the Entry 049 journal text are all removed from the tree and all recoverable
  from `cd99bbd`. Revert, not reset: the record of what was tried, measured and
  deferred stays in the branch history, which is how this journal works.
- **What the measurements said, preserved for the future step.** Pd_20K shape
  index: 96/96 confirmed, byte-identical, mean endpoint deviation 2.05 deg.
  Pd_20K curvedness: 1 genuine reroute (E-SAD_MC29 br1 claimed E-MAX_MC42 at
  along 13.6 deg; the ridge drains 1.6 deg to MAX_7412, a raw maximum the
  reduction leaked inside the saddle's ring) and 2 corner-sharings. ethene3 shape
  index: 4/4 resolved by promoting the mirror pair {MAX_10052, MAX_5}. The
  tolerance finding stands: 2 pitches = step + disc, and the symmetry oracle
  rejects 1 pitch because it promotes one of a mirror pair.
- **Schema: v4 admitted as LEGACY, not removed.** Snapshots saved while the
  refinement control existed are on disk (the user's
  `ethene3_atom1_params_curv.json` is one). `PARAMS_SCHEMA_VERSION` stays 3;
  `PARAM_LEGACY_SCHEMA_VERSIONS = {2, 4}`. The loader reads the keys it knows and
  ignores `refine_connectors`, so an observation made with refinement on still
  opens and reproduces the pipeline it describes. `test_load_rejects_unknown_schema_version`
  now probes 99, not `PARAMS_SCHEMA_VERSION + 1`, because +1 is 4 and 4 is valid.

### [Entry 050] Stalled Climbs Keep Their Path; Select a Saddle to See It

- **Objective**: the user's report, on `ethene3_atom1_params_curv.json` -- "MC18
  remains incomplete. Every saddle launch point should show a connector that
  follows a gradient path uphill to its highest point and then terminates. It may
  be that I can't see these paths because they are not associated with a maximum."
  Both halves are correct, and they are one bug with two causes.
- **Cause 1: the path was thrown away.** Step 4's `_gradient_connection` returned
  `_no_target(...)` when the climb reached no maximum, discarding `outcome.path`.
  The record said `no_target` with an empty `arc_points`, so there was literally
  nothing to draw. The ascent had run, had travelled 70-120 deg of ridge, and its
  result was dropped on the floor.
- **Cause 2: the overlay is keyed by maximum.** `_emit_max_arcs(basin_id)` selects
  connections with `conn.max_id == basin_id`. A stalled climb has `max_id = None`,
  so it is invisible from EVERY maximum's view -- the operator cannot reach it by
  clicking anything. The connection table has two ends, and only one of them was
  selectable.
- **The fix, part 1.** A stalled climb's record now carries the ridge it travelled:
  `arc_points = outcome.path`, `stall_point = outcome.path[-1]` (new field, the
  highest point the ascent found), `along_track_deg = outcome.arc_length_deg`, and
  `method = "gradient"` -- the geometry was traced, so it draws in cyan. `method`
  names how the geometry was PRODUCED; `status` names what it FOUND. The two are
  independent, and conflating them is what made the record look like nothing
  happened.
- **The fix, part 2: select a saddle.** `_emit_max_arcs(basin_id, saddle_id=None)`
  filters by `conn.saddle_id` when a saddle is named, so the view shows every
  ascent branch LEAVING it, arriving and unresolved alike. `state.selected_saddle_id`
  is the new key, mutually exclusive with `selected_basin_center`: selecting a
  saddle clears the maximum (and vice versa), so the scene and the drawer table
  always agree about which end is being inspected.
- **The join is exact, not geometric.** `build_launch_requests` names a raw
  saddle's seed `SAD_<vertex_id>` and an effective saddle's seed its own
  `id_label` (`E-SAD_MC18`). The CP UI dicts carry `id_label` in the same space,
  so `saddle_seed_id_for_cp` is a set membership test against the seeded ids --
  no tolerance, no nearest-neighbour. `saddle_seed_id_for_cluster` covers the
  picker's other path (cluster centroids are tested before CP glyphs, so a saddle
  glyph click usually lands on `select_cluster`): `E-SAD_<cluster id>` first, then
  any saddle member's `SAD_<vertex>`. Both are module-level pure functions of the
  flattened UI dicts, so they are testable without trame, and both are exported.
- **The arcs table gained a Target column**: the claimed maximum, a `stall` chip
  for a climb that topped out on non-maximum ground, or an em-dash for a branch
  with no geometry at all. The hint text now names both ends of the overlay.
- **MC18, verified.** In the user's snapshot MC18 is the cluster containing vertex
  8487, seeded as `SAD_8487` (raw, un-reduced). Its branch 0 is gate=behind,
  status=no_target, method=gradient, stall_point set -- a cyan path from the port
  to the highest point reachable uphill, terminating in mid-sphere. Selecting
  `SAD_8487` (glyph, cluster row, or CP row) draws it. `SAD_8400` (MC17, the
  mirror partner) behaves identically: 2 stalled branches, symmetric.
- **Tests** (539, was 533):
  - `test_gradient_fallback_records_gate_when_it_stalls` REWRITTEN. It pinned the
    old contract (`arc_points == ()`), which is exactly the behaviour the user is
    reporting as a bug. It now asserts the path survives, the stall point is the
    last vertex and lies on the sphere, `method == "gradient"`, and the gate is
    preserved. The fixture uses a south-pole peak so the climb travels, making the
    path-length assertion non-vacuous.
  - 6 new resolver tests: raw/effective saddle resolve; extrema, hidden, unseeded
    and None resolve to None; cluster resolves via `E-SAD_<id>` and via a raw
    member; a maximum's cluster resolves to None.
- **Validation**:
  - ethene3 curvedness reproduces **70/70 PASS** against the user's v4 snapshot,
    which loads cleanly as legacy.
  - **Pd_20K byte-identical** on both fields (96/96 shape, 52/52 curvedness,
    `stall_point` included in the compared key). Zero stalled branches there, so
    the change is a provable no-op -- the same contract Steps 4 and 5 were gated on.
  - `pytest` 539 passed. ruff / mypy / pre-commit clean. Headless viewer smoke on
    ethene3 curvedness with the v4 snapshot restored.
- **Open next**:
  1. A stalled climb's endpoint is drawn and named, and the disagreement with the
     topology is visible. What to DO about it is the separate refinement step, by
     the user's design, not this one's.
  2. Occluded branches still draw nothing (Entry 047 open item 2).
  3. The saddle view shows ascent branches only. Descent branches are in the seed
     and are not connected to anything -- a minima-keyed view would be the mirror
     of this one.
