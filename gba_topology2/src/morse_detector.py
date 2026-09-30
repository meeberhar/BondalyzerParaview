"""Morse Detector Subroutine.

Classifies vertices on a closed triangulated 2-sphere into local Discrete Morse
critical points (at exact tau = 0.0):
- Minima: Lower link has 0 connected components (and upper link has 1)
- Maxima: Upper link has 0 connected components (and lower link has 1)
- Regular: Lower link has 1 component and Upper link has 1 component
- Saddles: Lower link has k >= 2 connected components (multiplicity k - 1)

Verifies the global Euler characteristic / Poincaré-Hopf sum:
    N_max + N_min - sum(multiplicities) = 2
"""

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

__all__ = [
    "CriticalPoint",
    "MorseDetectionResult",
    "compute_discrete_morse_cps",
]


@dataclass(frozen=True)
class CriticalPoint:
    """Represents a detected Discrete Morse critical point."""

    vertex_id: int
    cp_type: str  # "minimum", "maximum", or "saddle"
    value: float
    position: tuple[float, float, float]
    multiplicity: int = 1


@dataclass(frozen=True)
class MorseDetectionResult:
    """Container for critical point detection results and Poincaré-Hopf check."""

    minima: list[CriticalPoint]
    maxima: list[CriticalPoint]
    saddles: list[CriticalPoint]
    euler_characteristic: int
    is_valid_euler: bool


def compute_discrete_morse_cps(
    pts: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    tie_break_epsilon: float = 1e-15,
) -> MorseDetectionResult:
    """Classify vertices on a closed triangulated 2-sphere into Discrete Morse CPs.

    Uses Simulation of Simplicity (SoS) tie-breaking by default:
        f_perturbed[v] = f_vals[v] + v * tie_break_epsilon * span(f)
    to guarantee a valid Morse function without flat plateaus or degenerate ties,
    ensuring the Poincaré-Hopf theorem holds: N_max + N_min - N_sad = 2.

    Args:
        pts: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle vertex indices.
        f_vals: (N,) scalar field values at each vertex.
        tie_break_epsilon: Relative epsilon for index-based tie-breaking (default 1e-15).

    Returns:
        MorseDetectionResult containing lists of minima, maxima, saddles,
        and the computed Euler characteristic.

    Raises:
        ValueError: If input dimensions are inconsistent.
    """
    n_pts = int(pts.shape[0])
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must be an (N, 3) array, got shape {pts.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            f"triangles must be an (M, 3) array, got shape {triangles.shape}"
        )
    if f_vals.ndim != 1 or f_vals.shape[0] != n_pts:
        raise ValueError(
            f"f_vals length ({f_vals.shape[0]}) must match pts length ({n_pts})"
        )

    # Compute effective scalar values with tie-breaking perturbation
    if tie_break_epsilon > 0.0:
        f_span = float(np.ptp(f_vals))
        scale = max(f_span, 1.0) * tie_break_epsilon
        perturbation = np.arange(n_pts, dtype=np.float64) * scale
        eval_f = f_vals.astype(np.float64) + perturbation
    else:
        eval_f = f_vals.astype(np.float64)

    # Build vertex to incident triangles mapping
    v_tris: list[list[int]] = [[] for _ in range(n_pts)]
    for t_idx, tri in enumerate(triangles):
        v_tris[int(tri[0])].append(t_idx)
        v_tris[int(tri[1])].append(t_idx)
        v_tris[int(tri[2])].append(t_idx)

    raw_minima: list[CriticalPoint] = []
    raw_maxima: list[CriticalPoint] = []
    raw_saddles: list[CriticalPoint] = []

    for v in range(n_pts):
        fv = float(eval_f[v])
        orig_fv = float(f_vals[v])
        # Build cyclic link graph around v
        link_edges: list[tuple[int, int]] = []
        for t_idx in v_tris[v]:
            tri = triangles[t_idx]
            others = [int(x) for x in tri if int(x) != v]
            if len(others) == 2:
                link_edges.append((others[0], others[1]))

        if not link_edges:
            continue

        g_link = nx.Graph()
        g_link.add_edges_from(link_edges)

        # Lower link L-(v): nodes with eval_f < eval_f(v)
        lower_nodes = [u for u in g_link.nodes() if float(eval_f[u]) < fv]
        g_lower = g_link.subgraph(lower_nodes)
        n_lower = int(nx.number_connected_components(g_lower))

        # Upper link L+(v): nodes with eval_f > eval_f(v)
        upper_nodes = [u for u in g_link.nodes() if float(eval_f[u]) > fv]
        g_upper = g_link.subgraph(upper_nodes)
        n_upper = int(nx.number_connected_components(g_upper))

        pos = (float(pts[v, 0]), float(pts[v, 1]), float(pts[v, 2]))

        if n_lower == 0 and n_upper == 1:
            raw_minima.append(
                CriticalPoint(
                    vertex_id=v,
                    cp_type="minimum",
                    value=orig_fv,
                    position=pos,
                    multiplicity=1,
                )
            )
        elif n_upper == 0 and n_lower == 1:
            raw_maxima.append(
                CriticalPoint(
                    vertex_id=v,
                    cp_type="maximum",
                    value=orig_fv,
                    position=pos,
                    multiplicity=1,
                )
            )
        elif n_lower >= 2:
            multiplicity = n_lower - 1
            raw_saddles.append(
                CriticalPoint(
                    vertex_id=v,
                    cp_type="saddle",
                    value=orig_fv,
                    position=pos,
                    multiplicity=multiplicity,
                )
            )

    total_saddle_index = sum(s.multiplicity for s in raw_saddles)
    euler_chi = len(raw_maxima) + len(raw_minima) - total_saddle_index
    is_valid = euler_chi == 2

    return MorseDetectionResult(
        minima=raw_minima,
        maxima=raw_maxima,
        saddles=raw_saddles,
        euler_characteristic=euler_chi,
        is_valid_euler=is_valid,
    )
