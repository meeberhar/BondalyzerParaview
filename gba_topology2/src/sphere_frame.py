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

The geodesic INTERPOLATION primitives live here too -- ``sphere_log``,
``sphere_exp``, ``slerp`` and the SQUAD pair ``squad_control`` /
``squad_point``. A curve that must pass through given points on this surface is
built out of exactly those, and the construction is geometry, not topology: it
belongs with the other on-sphere formulas and not in the module that happens to
need it first.
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
    "slerp",
    "sphere_exp",
    "sphere_log",
    "squad_control",
    "squad_point",
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


# ---------------------------------------------------------------------------
# geodesic interpolation (log / exp maps, slerp, SQUAD)
# ---------------------------------------------------------------------------


def sphere_log(
    at: npt.ArrayLike,
    to: npt.ArrayLike,
) -> npt.NDArray[np.float64] | None:
    """Logarithmic map on the sphere: the tangent vector at ``at`` aimed at ``to``.

    The result is tangent to the surface at ``at`` and its norm is the central
    angle between the two points, so ``sphere_exp(at, sphere_log(at, to))``
    returns ``to``. That round trip is what makes a spline on a curved surface
    possible: a Catmull-Rom tangent is a difference of positions, which lives in
    the ambient space and points off the sphere, while a difference of LOGS is a
    genuine tangent vector and can be averaged and scaled.

    Args:
        at: (3,) surface point (radial; need not be unit) where the tangent
            lives.
        to: (3,) surface point to aim at.

    Returns:
        The (3,) float64 tangent vector at ``at``, or None when the map is
        degenerate: ``to`` coincident with ``at`` (the vector is zero, and the
        caller's convention for "no direction" is None) or antipodal to it (the
        minimizing geodesic is not unique, so no tangent is the right answer).
    """
    ua = unit(at)
    ub = unit(to)
    if ua is None or ub is None:
        return None
    cos_d = float(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))
    if cos_d <= -1.0 + 1e-12:
        return None
    d = math.acos(cos_d)
    if d <= 1e-12:
        return None
    tangent = ub - cos_d * ua
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-15:
        return None
    return tangent * (d / norm)


def sphere_exp(
    at: npt.ArrayLike,
    vector: npt.ArrayLike,
) -> npt.NDArray[np.float64] | None:
    """Exponential map on the sphere: travel along the geodesic ``vector`` points down.

    Exact, not a linear step followed by a renormalisation: the result lies on
    the unit sphere to round-off for any length of ``vector``, which is the same
    property :func:`rotate_towards` provides for a (direction, angle) pair. The
    two forms are interchangeable; this one takes a tangent vector whose NORM is
    the angle, which is what :func:`sphere_log` produces and what a spline
    tangent is.

    Args:
        at: (3,) surface point (radial; need not be unit) to start from.
        vector: (3,) tangent vector at ``at``. Its component along the radial is
            removed, so a slightly off-tangent input degrades gracefully.

    Returns:
        The (3,) float64 unit radial of the arrived point, or None when ``at``
        is degenerate. A zero ``vector`` returns ``at`` itself.
    """
    ua = unit(at)
    if ua is None:
        return None
    v = np.asarray(vector, dtype=np.float64)
    tangent = v - float(np.dot(v, ua)) * ua
    angle = float(np.linalg.norm(tangent))
    if angle <= 1e-15:
        return ua
    direction = tangent / angle
    return math.cos(angle) * ua + math.sin(angle) * direction


def slerp(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    t: float,
) -> npt.NDArray[np.float64] | None:
    """Shortest-arc interpolation between two surface points.

    Falls back to a normalised linear step when the two points are nearly
    coincident, where the ``sin`` denominators of the standard formula vanish;
    over the sub-degree separations of that regime the two agree far inside the
    precision any caller cares about.

    Args:
        a: (3,) start point (radial).
        b: (3,) end point (radial).
        t: Parameter in [0, 1].

    Returns:
        The (3,) float64 unit radial at ``t``, or None when either endpoint is
        degenerate or the two are antipodal (no unique shortest arc).
    """
    ua = unit(a)
    ub = unit(b)
    if ua is None or ub is None:
        return None
    cos_omega = float(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))
    if cos_omega <= -1.0 + 1e-12:
        return None
    omega = math.acos(cos_omega)
    if omega <= 1e-9:
        return unit(ua + float(t) * (ub - ua))
    sin_omega = math.sin(omega)
    s0 = math.sin((1.0 - float(t)) * omega) / sin_omega
    s1 = math.sin(float(t) * omega) / sin_omega
    return s0 * ua + s1 * ub


def squad_control(
    prev: npt.ArrayLike,
    at: npt.ArrayLike,
    nxt: npt.ArrayLike,
) -> npt.NDArray[np.float64] | None:
    """SQUAD intermediate control point at ``at`` for the segment pair around it.

    This is the spherical analogue of the Catmull-Rom tangent
    :math:`\\tfrac12(p_{i+1} - p_{i-1})`: the two geodesic vectors that point
    from ``at`` to its neighbours are averaged and a QUARTER of that average is
    travelled, giving the point :math:`q_i = \\exp_{p_i}(-\\tfrac14(\\log_{p_i}
    p_{i-1} + \\log_{p_i} p_{i+1}))`. The quarter is not a tuning knob -- it is
    what makes the resulting curve pass through ``at`` with the averaged
    tangent, and it is the factor that reduces the construction to the ordinary
    cubic on a flat surface.

    A vertex where either log is degenerate (a repeated point) gets the
    one-sided average, and a vertex where BOTH are degenerate returns None,
    which the caller reads as "no curvature information here".

    Args:
        prev: (3,) previous knot (radial).
        at: (3,) the knot the control point belongs to.
        nxt: (3,) next knot (radial).

    Returns:
        The (3,) float64 unit radial of the control point, or None.
    """
    ua = unit(at)
    if ua is None:
        return None
    log_prev = sphere_log(at, prev)
    log_next = sphere_log(at, nxt)
    if log_prev is None and log_next is None:
        return None
    total = np.zeros(3, dtype=np.float64)
    n = 0
    for lg in (log_prev, log_next):
        if lg is not None:
            total = total + lg
            n += 1
    return sphere_exp(ua, -0.25 * total / float(n))


def squad_point(
    a: npt.ArrayLike,
    qa: npt.ArrayLike,
    qb: npt.ArrayLike,
    b: npt.ArrayLike,
    t: float,
) -> npt.NDArray[np.float64] | None:
    """One point of the SQUAD curve on the segment ``a -> b``.

    :math:`\\mathrm{squad}(t) = \\mathrm{slerp}(\\mathrm{slerp}(a,b,t),
    \\mathrm{slerp}(q_a,q_b,t), 2t(1-t))`. The inner pair is the geodesic
    between the knots and the geodesic between their control points; the outer
    blend is the quadratic that vanishes at both ends, which is what makes the
    curve interpolate ``a`` and ``b`` exactly while taking their tangents.
    :func:`squad_control` supplies ``qa`` and ``qb``.

    Args:
        a: (3,) segment start knot (radial).
        qa: (3,) control point at ``a``.
        qb: (3,) control point at ``b``.
        b: (3,) segment end knot (radial).
        t: Parameter in [0, 1].

    Returns:
        The (3,) float64 unit radial, or None when any needed interpolation is
        degenerate (antipodal endpoints).
    """
    tt = float(t)
    g = slerp(a, b, tt)
    c = slerp(qa, qb, tt)
    if g is None or c is None:
        return None
    return slerp(g, c, 2.0 * tt * (1.0 - tt))
