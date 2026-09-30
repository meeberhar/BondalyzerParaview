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
