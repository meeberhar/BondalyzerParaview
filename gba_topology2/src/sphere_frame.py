"""Atom Sphere Frame Subroutine.

Every angular computation in gba_topology2 derives the outward normal of a
surface point as ``p / ||p||`` and rebuilds surface positions as ``dir * R``.
That is only correct when the sphere is centred on the world origin. An
isolated atom (``Pd_20K.plt``) happens to sit at ``(0, 0, 0)`` so the
assumption is invisible; in a molecule every atom is offset, and the same
formula then measures angles from the world origin instead of from the
nucleus. The consequences are a sphere radius inflated by the offset, an
inflated mesh pitch ``delta_theta = 2 arcsin(l_e / 2R)``, tangent planes that
slice obliquely through the sphere (boundary rings standing on edge), angular
distances compressed on the near side of the atom and stretched on the far
side, skewed catastrophe azimuths, and harmonic-nudge probes sampling the
wrong region of space.

The fix is applied once, at extraction: translate the mesh into the atom's own
frame, run every algorithm there, and add the centre back wherever a
coordinate is drawn or displayed. This module owns that translation so the
sign of the offset is written in exactly one place.

Frame contract used across the pipeline:
- ALGORITHM FRAME (local, centred): mesh points, critical point positions,
  cluster and catastrophe centroids, boundary rings, ports, nudge positions,
  sphere radius and mesh pitch.
- WORLD FRAME (global): everything drawn by VTK and everything shown in the
  inspector cards. ``to_global`` converts; ``to_local`` inverts.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "compute_atom_frame",
    "radial_offset",
    "to_global",
    "to_global_array",
    "to_local",
    "to_local_array",
]


def compute_atom_frame(
    pts: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
) -> tuple[npt.NDArray[np.float64], float]:
    """Recover the nuclear centre and true radius of an atom sphere mesh.

    The centre is the vertex centroid, which for a closed quasi-uniform
    triangulation of a sphere is the nucleus to within the mesh's own
    discretisation error. The radius is the mean distance from that centre,
    i.e. the radius the angular formulas need -- the mean distance from the
    world origin is inflated by the offset whenever the atom is not centred.

    Args:
        pts: (N, 3) vertex coordinates in the world frame.

    Returns:
        Tuple of (center, radius) where center is a (3,) float64 array and
        radius is the mean of ``||pts - center||`` in Angstrom.

    Raises:
        ValueError: If pts is not a non-empty (N, 3) array, or if the
            recovered radius is not positive (a degenerate, coincident mesh).
    """
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"pts must be a non-empty (N, 3) array, got shape {arr.shape}")

    center = arr.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(arr - center, axis=1)))
    if radius <= 0.0:
        raise ValueError(f"Recovered a non-positive sphere radius: {radius}")
    return center, radius


def to_local(
    point: npt.ArrayLike,
    center: npt.ArrayLike,
) -> tuple[float, float, float]:
    """Convert one world-frame point into the centred atom frame.

    Args:
        point: (3,) point in world coordinates.
        center: (3,) nuclear centre in world coordinates.

    Returns:
        The point relative to the centre, as a plain float tuple.
    """
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    out = p - c
    return (float(out[0]), float(out[1]), float(out[2]))


def to_global(
    point: npt.ArrayLike,
    center: npt.ArrayLike,
) -> tuple[float, float, float]:
    """Convert one centred-frame point back into world coordinates.

    Args:
        point: (3,) point relative to the nuclear centre.
        center: (3,) nuclear centre in world coordinates.

    Returns:
        The world-frame point, as a plain float tuple.
    """
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    out = p + c
    return (float(out[0]), float(out[1]), float(out[2]))


def to_local_array(
    pts: npt.NDArray[np.floating[Any]],
    center: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert an (N, 3) array of world-frame points into the centred frame.

    Args:
        pts: (N, 3) points in world coordinates.
        center: (3,) nuclear centre in world coordinates.

    Returns:
        A new (N, 3) float64 array relative to the centre.
    """
    return np.asarray(pts, dtype=np.float64) - np.asarray(center, dtype=np.float64)


def to_global_array(
    pts: npt.NDArray[np.floating[Any]],
    center: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert an (N, 3) array of centred points into world coordinates.

    Args:
        pts: (N, 3) points relative to the nuclear centre.
        center: (3,) nuclear centre in world coordinates.

    Returns:
        A new (N, 3) float64 array in world coordinates.
    """
    return np.asarray(pts, dtype=np.float64) + np.asarray(center, dtype=np.float64)


def radial_offset(
    point: npt.ArrayLike,
    factor: float,
) -> tuple[float, float, float]:
    """Lift a centred-frame surface point radially outward off the sphere.

    Used to float boundary rings a fraction of a percent above the surface so
    the tube geometry does not z-fight with the sphere mesh. Scaling the
    centre-relative position is a genuine radial lift in this frame; the same
    scaling applied to world coordinates would push points through the sphere
    for any atom that is not centred on the origin.

    Args:
        point: (3,) point relative to the nuclear centre.
        factor: Radial multiplier, e.g. 1.008 for a 0.8% lift.

    Returns:
        The lifted point, as a plain float tuple.
    """
    p = np.asarray(point, dtype=np.float64)
    out = p * float(factor)
    return (float(out[0]), float(out[1]), float(out[2]))
