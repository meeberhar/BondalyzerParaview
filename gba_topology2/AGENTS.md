# Project Manifesto: `gba_topology2`

## 1. Vision & Ultimate Objective
The goal of the `gba_topology2` project is to develop and test methods that:
1. Extract true topological critical points (CPs) from atomic sphere scalar fields in `.plt` files at exact zero persistence.
2. Group, scan, and resolve sub-grid numerical discretization artifacts into well-defined **Effective Critical Points (ECPs)**.
3. Establish accurate **Excision Boundaries** and their launch **Poles / Ports**.
4. Apply a final **Harmonic Centroid Nudge** to lock each ECP and its excision boundary poles onto their exact continuous, symmetric harmonic centers.
5. Trace and connect gradient paths launched from these perfected poles to establish robust **Basin Boundaries**.
6. **Upstream Integration**: The ultimate destination of this codebase is to be incorporated directly into the routines that generate our `.plt` files, replacing existing inaccurate basin boundaries with mathematically sound boundaries.

The code in `gba_topology2` is an **exploratory research framework** to test, calibrate, and validate these methods across diverse datasets (`Pd_20K.plt`, `Pd.plt`, `ethene4.plt`, etc.).

---

## 2. Core Architectural Principles

### Principle 1: The Guiding Question
Before writing, extending, or embedding any new capability or algorithm, we must always stop and ask:
> **"Do we need a new subroutine for this capability?"**
- If a task performs a distinct mathematical or geometric operation (scanning, fusing, stepping, classifying, nudging, connecting), it **must** be implemented as a separate, self-contained subroutine (function/module).
- It must have a clear input signature, a single responsibility, and a deterministic output.
- It must be testable in isolation without running the full viewer.

### Principle 2: Modular Isolation & Swappability
- A subroutine that works well for one dataset or field may fail or need revision on another.
- Because each capability is a discrete subroutine, we can modify, tune, or completely replace a fusion routine, a polarity classifier, or an excision fitter without touching or risking the rest of the pipeline.

### Principle 3: Zero-Persistence Baseline
- All data ingestion starts at exact **$\tau = 0.0$ persistence**.
- We do not use global persistence filtering to prune points, as non-local pairing distorts local topological charges. Local artifacts must be resolved locally by targeted subroutines.

### Principle 4: Minimal User Burden
- The final pipeline must find basins automatically with minimal or zero user intervention.
- We do not build permanent sliders for users to hunt for basins.
- Diagnostic tools and parameter controls exist in the viewer **only temporarily** while we explore and calibrate empirical relationships against the intrinsic mesh grid pitch ($\delta\theta_{\text{mesh}}$). Once verified, they are codified into subroutines and retired.

### Principle 5: Progressive, Dynamic Viewer with Direct 3D Picking
- The viewer is built in steps, matching the pipeline sequence.
- **Direct 3D Picking & Inspection**: The viewer must always provide instant interactive access to cluster and CP details—by clicking directly on the 3D sphere canvas (via ray-pick) or in the sidebar table.

---

## 3. Issue Troubleshooting Workflow (Mandatory)

**The preferred way to troubleshoot any issue in this project is to reproduce it
in the sphere viewer and export the parameters.** Never ask the user to
hand-specify sliders, toggles or field selections — the viewer already records
them, and hand-typed values are a common source of the very bug being
investigated.

### 3.1 The protocol

1. **User reproduces the issue in the viewer.** They launch
   `uv run python gba_topology2/src/sphere_viewer.py <file>.plt`, set the
   controls that expose the problem, and observe it.
2. **User clicks "Save Parameters"** (toolbar button). This writes
   `<stem>_atom<N>_params.json` next to the `.plt` and echoes the absolute
   path to the terminal and to a confirmation chip in the drawer. The
   snapshot records every control, the run context (`.plt`, atom, field
   list), the view camera, and the derived metrics.
3. **User describes the issue** and **points the agent at the exported
   params file** (the path, typically the JSON in the workspace root).
4. **Agent investigates with `gba_topology2/diagnose_params.py`**, which
   re-runs the analysis subroutines headlessly from that JSON and diffs the
   recomputed metrics against the ones recorded at observation time.

A saved snapshot can also be **restored**, not just inspected: the
"Load Parameters" toolbar button opens a dialog that takes a `.json`
snapshot and restores the file, atom, field, controls and camera it
describes, and `uv run python gba_topology2/src/sphere_viewer.py <file>.plt
--params <snapshot>.json` does the same at startup. Use this to put the
viewer back into a known state without asking the operator to re-click
through a dozen controls.

### 3.2 Using `diagnose_params.py`

```bash
# Full pipeline, the default: reproduce the saved observation
uv run python gba_topology2/diagnose_params.py <params.json>

# Bisect: stop after one stage (inclusive)
uv run python gba_topology2/diagnose_params.py <params.json> --only clusters

# Sweep a control to find the value that triggers the issue
uv run python gba_topology2/diagnose_params.py <params.json> --set k_pitch=3.0

# Dump per-entity results (clusters, catastrophes, effective CPs,
# basin centers, launch seeds) for programmatic A/B diffing
uv run python gba_topology2/diagnose_params.py <params.json> --out results.json

# Context overrides, when the .plt moved or a different atom is wanted
uv run python gba_topology2/diagnose_params.py <params.json> --plt <path> -a 3 -f "ELF (condensed)"
```

- **Stages** run in the viewer's own order: `geometry`, `morse`, `clusters`,
  `bridge`, `units`, `catastrophes`, `ports`, `reduction`, `basins`, `launch`.
  The `bridge` stage reports the corridor-bridging graph and the per-component
  elongation gate — use it whenever the "Bridge Radius" (`k_bridge`) slider
  changes the number of effective critical points, since the gate is a
  *pooled* property and a single satellite can sink an otherwise valid
  corridor (see journal Entry 020).
- **Exit code** is the reproduce check: `0` = the recomputed metrics match the
  saved observation, `1` = at least one leaf drifted (the drifting paths are
  printed as `DRIFT` lines). A `1` is a *result*, not a failure of the tool.
- **`--quiet`** prints only the reproduce-check summary; use it when the
  per-stage detail is not needed.
- **Display-only controls** (`show_*`, `cp_glyph_scale`, `num_contours`) have
  no headless analogue. They are loaded and reported, but inert.

### 3.3 Extending the diagnostic runner

**`diagnose_params.py` is expected to be modified as part of troubleshooting.**
When the saved parameters do not expose what is needed to explain the issue:

- **Add the missing instrumentation to the runner** — a new stage, an extra
  summary field, a per-entity dump — rather than hand-rolling a throwaway
  script that re-implements the pipeline. The runner already calls the
  project's subroutines with the viewer's exact arguments, so it is the
  faithful place to add a probe.
- **Keep the runner in sync with the viewer.** If a stage's logic changes in
  `sphere_viewer.py`, update the corresponding stage here in the same atomic
  unit, or the reproduce check will report drift that is an artefact of the
  runner, not of the data.
- **Prefer adding a stage over editing the viewer's maths.** The viewer is the
  reference implementation; the runner is a mirror of it.
- **Treat a reproduce-check mismatch as a finding.** If the runner disagrees
  with the saved metrics at default parameters, that disagreement is itself
  the bug being reported. Investigate it before changing the runner.
- **Known duplication**: `stage_reduction` mirrors the viewer's inline Step 6
  block, and `verify_step6_toggles.py` mirrors it too. When touching that
  logic, consider extracting a shared `run_effective_reduction()` subroutine
  into `gba_topology2/src/` that the viewer, the runner and the verify script
  all call, so the three cannot drift apart.

### 3.4 Reporting

When a troubleshooting investigation concludes, record it in `journal.md`:
the params file used, the reproduce-check result, the stages that mattered,
and the conclusion. If the investigation changed the runner, say so
explicitly, since the runner is a maintained diagnostic tool rather than a
scratch file.

---

## General rules

- After each atomic unit of work, you must commit changes according to the procedure layed out in the project `AGENTS.md` instructions.
- **Troubleshooting requests**: if the user reports an issue in the viewer but
  has not exported parameters, ask them to reproduce it and click "Save
  Parameters", then point you at the JSON. Do not guess the control values.