"""Polarity Detector Subroutine for gba_topology2.

Classifies unresolved or composite micro-clusters and macro-strings with net positive
Euler index (chi_local = +1) into either:
- Peak / Mountain (E-MAX): scalar field decreases outward (negative radial derivative)
- Basin / Pit (E-MIN): scalar field increases outward (positive radial derivative)

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Uses mesh pitch delta_theta_mesh_rad for parameter scaling.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "EffectiveExtremum",
    "ReductionDecision",
    "classify_cluster_polarity",
    "effective_reduction_decision",
    "evaluate_spherical_radial_slope",
]

# Effective critical-point kinds produced by the Step 6 reduction stage.
# "E-NONE"  -> the set is topologically neutral and annihilates (glyph removed,
#              boundary ring retained as the record of the collapsed set).
# "IRREDUCIBLE" -> the set carries net index >= +2, which no single effective CP
#              can represent, so it must stay expanded.
ReductionDecision = Literal["E-MAX", "E-MIN", "E-SAD", "E-NONE", "IRREDUCIBLE"]


def effective_reduction_decision(
    local_euler_index: int,
    polarity_type: str = "E-MAX",
) -> ReductionDecision:
    """Map a collapsed set's net Euler index to the effective CP kind it reduces to.

    Topological contract for reducing a critical-point set to ONE effective CP:

    - ``chi == 0``  -> ``"E-NONE"``: the set is topologically neutral (equal
      extremum and saddle index, e.g. a 2 Max / 8 Sad / 6 Min ridge string). It
      carries no net index, so it annihilates: no glyph is emitted.
    - ``chi == +1`` -> ``polarity_type``: a net source of index +1 is an effective
      extremum, resolved to ``"E-MAX"`` / ``"E-MIN"`` by the radial-slope polarity.
    - ``chi <= -1`` -> ``"E-SAD"``: a net sink of index ``chi`` is an effective
      saddle of fold order ``k = 1 - chi`` (chi = -1 -> 2-fold, chi = -2 -> monkey).
    - ``chi >= +2`` -> ``"IRREDUCIBLE"``: a single effective CP carries at most
      +1, so the set cannot be collapsed without destroying index. It stays expanded.

    Args:
        local_euler_index: Net index of the set (+1 per extremum, -multiplicity
            per saddle), i.e. ``MicroCluster.local_euler_index``.
        polarity_type: ``"E-MAX"`` or ``"E-MIN"`` from the radial-slope classifier;
            consulted only when ``local_euler_index == 1``.

    Returns:
        The effective CP kind (or annihilation / irreducible marker) to apply.

    Raises:
        ValueError: If ``polarity_type`` is not ``"E-MAX"`` or ``"E-MIN"`` while
            ``local_euler_index == 1``.
    """
    if local_euler_index == 0:
        return "E-NONE"
    if local_euler_index == 1:
        if polarity_type not in ("E-MAX", "E-MIN"):
            raise ValueError(
                f"polarity_type must be 'E-MAX' or 'E-MIN', got {polarity_type!r}"
            )
        return polarity_type  # type: ignore[return-value]
    if local_euler_index <= -1:
        return "E-SAD"
    return "IRREDUCIBLE"


@dataclass(frozen=True)
class EffectiveExtremum:
    """Represents an effective extremum resolved from a micro-cluster or single CP."""

    entity_id: str
    extremum_type: str  # "E-MAX" or "E-MIN"
    centroid: tuple[float, float, float]
    centroid_unit: tuple[float, float, float]
    effective_value: float
    radial_slope: float  # Mean outward derivative df/dr (rad^-1)
    confidence: float  # Consistency of outward slope direction [0.0, 1.0]
    local_euler_index: int
    constituent_cps: list[CriticalPoint]


def evaluate_spherical_radial_slope(
    centroid_unit: tuple[float, float, float],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    inner_radius_rad: float,
    outer_radius_rad: float,
    sphere_radius: float = 1.0,
) -> tuple[float, float]:
    """Evaluate outward radial slope of scalar field around a centroid on a triangulated sphere.

    Finds mesh vertices within an annular ring [inner_radius_rad, outer_radius_rad]
    and computes the radial directional derivative df / d(theta) via linear regression.

    Args:
        centroid_unit: (3,) unit vector pointing to cluster centroid.
        mesh_pts: (N, 3) vertex coordinates of the sphere mesh.
        f_vals: (N,) scalar field values at mesh vertices.
        inner_radius_rad: Angular inner radius of probe annulus in radians.
        outer_radius_rad: Angular outer radius of probe annulus in radians.
        sphere_radius: Radius R of the sphere.

    Returns:
        Tuple of (mean_radial_slope, confidence_score) where:
        - mean_radial_slope is df/d(theta) in units of scalar_delta / radian.
          Positive => field values increase radially outward (Basin / E-MIN).
          Negative => field values decrease radially outward (Peak / E-MAX).
        - confidence_score is fraction of concordant radial samples in [0.0, 1.0].
    """
    c_u = np.array(centroid_unit, dtype=np.float64)
    norm_c = float(np.linalg.norm(c_u))
    if norm_c > 1e-12:
        c_u /= norm_c
    else:
        c_u = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pts = np.asarray(mesh_pts, dtype=np.float64)
    f = np.asarray(f_vals, dtype=np.float64)

    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts_u = pts / norms

    # Angular distance theta from centroid to all vertices
    dots = np.clip(np.dot(pts_u, c_u), -1.0, 1.0)
    thetas = np.arccos(dots)

    # Filter vertices in the annular probe window
    mask = (thetas >= inner_radius_rad) & (thetas <= outer_radius_rad)
    ring_thetas = thetas[mask]
    ring_f = f[mask]

    if len(ring_thetas) < 3:
        # Fallback to broader window if sparse
        broad_mask = thetas <= max(outer_radius_rad * 1.5, inner_radius_rad + 0.05)
        ring_thetas = thetas[broad_mask]
        ring_f = f[broad_mask]

    if len(ring_thetas) < 2:
        return 0.0, 0.0

    # Fit linear slope: f(theta) ~ slope * theta + intercept
    var_theta = float(np.var(ring_thetas))
    if var_theta < 1e-12:
        slope = 0.0
    else:
        cov = float(np.cov(ring_thetas, ring_f)[0, 1])
        slope = cov / var_theta

    # Evaluate confidence based on sign consistency relative to inner boundary
    # Compare ring points against the minimum/mean at inner edge
    if len(ring_thetas) > 0:
        median_theta = float(np.median(ring_thetas))
        inner_vals = ring_f[ring_thetas <= median_theta]
        outer_vals = ring_f[ring_thetas > median_theta]
        if len(inner_vals) > 0 and len(outer_vals) > 0:
            diff = float(np.mean(outer_vals) - np.mean(inner_vals))
            # Concordance: does finite difference agree with linear regression slope?
            if (diff > 0 and slope > 0) or (diff < 0 and slope < 0):
                confidence = 0.95
            elif abs(slope) < 1e-9:
                confidence = 0.5
            else:
                confidence = 0.7
        else:
            confidence = 0.8
    else:
        confidence = 0.5

    return slope, confidence


def classify_cluster_polarity(
    cluster: MicroCluster,
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    delta_theta_mesh_rad: float,
    sphere_radius: float = 1.0,
    probe_width_pitch: float = 2.0,
) -> EffectiveExtremum:
    """Classify a micro-cluster into an Effective Extremum (E-MAX vs E-MIN).

    For clusters with ambiguous composition (e.g. 2 Max, 1 Sad with chi = +1,
    or flat plateau extrema), probes the outward radial slope to determine
    whether the surrounding terrain descends away (Peak) or ascends away (Basin).

    Args:
        cluster: Input MicroCluster object.
        mesh_pts: (N, 3) vertex coordinates of sphere mesh.
        f_vals: (N,) scalar field values.
        delta_theta_mesh_rad: Mesh pitch delta_theta in radians.
        sphere_radius: Radius of the sphere R.
        probe_width_pitch: Annulus width in delta_theta units (default 2.0).

    Returns:
        EffectiveExtremum dataclass with extremum_type ("E-MAX" or "E-MIN"),
        effective scalar value, radial slope, and confidence score.
    """
    inner_r = max(cluster.angular_radius_rad, delta_theta_mesh_rad)
    outer_r = inner_r + probe_width_pitch * delta_theta_mesh_rad

    slope, confidence = evaluate_spherical_radial_slope(
        centroid_unit=cluster.centroid_unit,
        mesh_pts=mesh_pts,
        f_vals=f_vals,
        inner_radius_rad=inner_r,
        outer_radius_rad=outer_r,
        sphere_radius=sphere_radius,
    )

    # Determine polarity
    # If slope < 0, field drops radially outward -> Peak (E-MAX)
    # If slope > 0, field rises radially outward -> Basin (E-MIN)
    if slope < -1e-12:
        ext_type = "E-MAX"
        # Representative value: maximum scalar among constituent members
        eff_val = max(m.value for m in cluster.members) if cluster.members else 0.0
    elif slope > 1e-12:
        ext_type = "E-MIN"
        # Representative value: minimum scalar among constituent members
        eff_val = min(m.value for m in cluster.members) if cluster.members else 0.0
    else:
        # Fallback to dominant CP member type if slope is flat
        if cluster.n_maxima > cluster.n_minima:
            ext_type = "E-MAX"
            eff_val = max(m.value for m in cluster.members) if cluster.members else 0.0
        else:
            ext_type = "E-MIN"
            eff_val = min(m.value for m in cluster.members) if cluster.members else 0.0
        confidence = 0.5

    return EffectiveExtremum(
        entity_id=cluster.cluster_id,
        extremum_type=ext_type,
        centroid=cluster.centroid,
        centroid_unit=cluster.centroid_unit,
        effective_value=eff_val,
        radial_slope=slope,
        confidence=confidence,
        local_euler_index=cluster.local_euler_index,
        constituent_cps=list(cluster.members),
    )
