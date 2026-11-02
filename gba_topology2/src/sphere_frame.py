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

This module also owns the ON-SPHERE geometry: the great-circle exponential map
(``rotate_towards``), angular distance, normalisation and tangent projection.
Those formulas need a centred sphere exactly like the angular computations
above do, so they live beside the translation that makes the frame centred and
are written once. ``saddle_launch`` keeps thin aliases for its historical
private helpers.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "angular_distance",
    "compute_atom_frame",
    "radial_offset",
    "rotate_towards",
    "tangent_project",
    "to_global",
    "to_global_array",
    "to_local",
    "to_local_array",
    "unit",
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


def unit(value: npt.ArrayLike) -> npt.NDArray[np.float64] | None:
    """Normalise a vector, returning None for a (near) zero vector.

    A zero vector has no direction, so every caller that wants a unit radial or
    a unit tangent must handle the degenerate case explicitly rather than
    dividing by zero and propagating NaN through an angular formula.

    Args:
        value: (3,) vector in the centred atom frame.

    Returns:
        A (3,) float64 unit vector, or None when the norm is <= 1e-15.
    """
    arr = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-15:
        return None
    return arr / norm


def tangent_project(
    vector: npt.ArrayLike,
    normal: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Remove the component of ``vector`` along the unit ``normal``.

    The result is tangent to the sphere at the point whose radial direction is
    ``normal``. It is not necessarily non-zero: a vector parallel to the normal
    projects away entirely, so pass the result through :func:`unit` and handle
    ``None``.

    Args:
        vector: (3,) displacement in the centred atom frame, e.g. a chord
            between two surface points.
        normal: (3,) unit radial vector of the tangent point.

    Returns:
        The (3,) float64 tangential component of ``vector``.
    """
    v = np.asarray(vector, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    return v - float(np.dot(v, n)) * n


def rotate_towards(
    origin_unit: npt.ArrayLike,
    direction_unit: npt.ArrayLike,
    angle_rad: float,
    radius: float,
) -> tuple[float, float, float]:
    """Move on the sphere from ``origin_unit`` along ``direction_unit``.

    This is the great-circle (geodesic) exponential map: the result lies exactly
    on the sphere of radius ``radius`` rather than being projected back onto it
    after a linear step, so a chain of steps accumulates no radial drift. It is
    the only correct way to extrapolate a launch direction, because a straight
    chord through space leaves the surface immediately.

    Args:
        origin_unit: Unit radial vector of the start point.
        direction_unit: Unit tangent vector at the start giving the travel
            direction (perpendicular to ``origin_unit``).
        angle_rad: Angular step in radians, measured at the centre.
        radius: Sphere radius.

    Returns:
        The new position as a float tuple, at distance ``radius`` from the
        centre. A non-positive ``angle_rad`` returns the start point.
    """
    origin = np.asarray(origin_unit, dtype=np.float64)
    if angle_rad <= 0.0:
        return (
            float(origin[0] * radius),
            float(origin[1] * radius),
            float(origin[2] * radius),
        )
    direction = np.asarray(direction_unit, dtype=np.float64)
    moved = math.cos(angle_rad) * origin + math.sin(angle_rad) * direction
    moved_unit = unit(moved)
    if moved_unit is None:
        moved_unit = origin
    return (
        float(moved_unit[0] * radius),
        float(moved_unit[1] * radius),
        float(moved_unit[2] * radius),
    )


def angular_distance(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
) -> float:
    """Central angle between two surface points, in radians.

    Both arguments are normalised internally, so they need only be radial, not
    unit, and need not share a radius. The dot product is clipped before
    ``acos``, which is what keeps floating-point noise at coincident or
    antipodal points from producing NaN.

    Args:
        a: (3,) point in the centred atom frame.
        b: (3,) point in the centred atom frame.

    Returns:
        The angle in [0, pi] between the two radial directions, in radians.
        Returns 0.0 when either point is degenerate (at the centre).
    """
    ua = unit(a)
    ub = unit(b)
    if ua is None or ub is None:
        return 0.0
    return float(math.acos(float(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))))


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
