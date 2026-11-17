"""The polish operators' frame contract: the nucleus is a parameter, not a loop variable.

Added while wiring the "Arc Polish Method" selector into the card, from a mypy finding
that only surfaced on a cold cache: fit_quad_bezier takes the world-frame nucleus as
its ``center`` argument, and the loop that collects the off-route CP discs rebound that
same name to each region's unit centre. A loop variable outlives its loop, so the final
``_world(u, sphere_radius, center)`` described the fitted arc about the LAST region's
unit radial -- translating every fitted connector by a vector of length 1 angstrom, on
spheres whose own radius is often near that (Pd20k: R = 1.038 A). The arc leaves the
sphere, and its endpoints stop welding to the CPs they connect.

Every pre-existing test missed it for the same two reasons, which is the lesson here:
they placed the nucleus at the origin (the identity transform hides a translation), and
the gate tests passed regions=() (an empty loop cannot overwrite anything). This module
puts the nucleus away from the origin AND passes a region, which is the shape of the
real geometry pipeline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "gba_topology2" / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "gba_topology2" / "src"))

from dataclasses import dataclass

from ridge_refiner import (
    GRADIFY_OUTPUT_PITCH,
    QuadBezierFit,
    fit_quad_bezier,
)
from sphere_frame import squad_point

_R = 1.0
_PITCH = math.radians(2.0)
_DISC_PITCH = 3.0
_NUCLEUS = (10.0, 20.0, 30.0)


@dataclass(frozen=True)
class _Region:
    """The GradifyRegion protocol: a CP centre unit and its disc radius (radians)."""

    centre_unit: npt.NDArray[np.float64]
    radius_rad: float


def _unit(theta: float, phi: float) -> npt.NDArray[np.float64]:
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )


def _field() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """An equatorial crest: f = exp(2 y), peaked on the great circle z = 0.

    Analytic and non-degenerate (peak/trough = e^2 - e^-2, no plateau), which is what
    the quadratic's gain gate needs in order to accept a bend, and the mesh is in the
    CENTERED frame -- the frame the nucleus test is about.
    """

    n_lat, n_lon = 181, 360
    lats = np.linspace(0.0, np.pi, n_lat)
    lons = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    pts = (
        np.stack(
            (
                np.sin(lat_grid) * np.cos(lon_grid),
                np.sin(lat_grid) * np.sin(lon_grid),
                np.cos(lat_grid),
            ),
            axis=-1,
        ).reshape(-1, 3)
        * _R
    )
    vals = np.exp(2.0 * pts[:, 1] / _R)
    return pts, vals


def _arc(ends: list[npt.NDArray[np.float64]], n: int = 40) -> list[tuple[float, ...]]:
    """A world-frame great circle through the south pole, in the nucleus's frame.

    Squad with the dip as both controls is the slerp through the endpoints, which is
    the same construction the refiner's own tests use (``sphere_frame.squad_point``,
    the helper the shipped connectors are described with). Points are unit radials
    scaled to R and shifted by the nucleus: the world frame.
    """

    a, b = ends
    centre = np.asarray(_NUCLEUS, dtype=np.float64)
    dip = _unit(math.pi * 0.85, 0.0)
    pts: list[tuple[float, ...]] = []
    for k in range(n):
        t = math.pi * k / (n - 1)
        w = squad_point(a, dip, b, dip, t / (n - 1))
        assert w is not None
        pts.append(((w / np.linalg.norm(w)) * _R + centre).tolist())
    return pts


def _south_pole_ends() -> list[npt.NDArray[np.float64]]:
    """(-1,0,0) to (1,0,0), with both ends pushed off the pole the dip runs by.

    The ends are pushed so a disc centred on a POLE cannot cover them; an arc whose
    endpoint sat on the region's centre would be pinned there and the fit would refuse
    the span, which tests the gate instead of the frame.
    """

    tip = _unit(math.pi * 0.85, 0.0)  # the dip's vertex
    pushed: list[npt.NDArray[np.float64]] = []
    for raw in (np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        direction = raw - tip
        pushed.append(direction / np.linalg.norm(direction))
    return pushed


def _fit(region_centre: npt.NDArray[np.float64]) -> QuadBezierFit:
    pts, vals = _field()
    return fit_quad_bezier(
        _arc(_south_pole_ends()),
        pts,
        vals,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=_NUCLEUS,
        regions=[_Region(region_centre, _R * _DISC_PITCH * _PITCH)],
        output_pitch=GRADIFY_OUTPUT_PITCH,
        span_override=math.exp(2.0) - math.exp(-2.0),
    )


def test_the_fit_actually_ran() -> None:
    """Guard against a vacuous pass: a refused arc is returned verbatim, untransformed.

    The other tests in this module would then be asserting that a path built by the
    caller sits on the caller's sphere, which is true of anything.
    """

    fit = _fit(_unit(0.0, 0.0))
    assert fit.changed, f"the bend was refused ({fit.status}); the emit path never ran"
    assert fit.n_fitted >= 1, fit.status


def test_every_emitted_vertex_is_one_radius_from_the_nucleus() -> None:
    fit = _fit(_unit(0.0, 0.0))
    centre = np.asarray(_NUCLEUS, dtype=np.float64)
    radius = np.linalg.norm(np.asarray(fit.path) - centre, axis=1)
    worst = float(np.max(np.abs(radius - _R)))
    assert worst < 1e-9, (
        f"the arc sits {worst:.4f} A off the sphere about the nucleus: 'center' is "
        "carrying a region's centre unit, not the nucleus"
    )


def test_the_arc_starts_and_ends_where_the_geodesic_does() -> None:
    """Ends are the CP centres; a translation breaks the weld the boundaries need.

    Compared by DIRECTION from the nucleus, the only frame in which a connector's
    endpoints are specified.
    """

    centre = np.asarray(_NUCLEUS, dtype=np.float64)
    arc = _arc(_south_pole_ends())
    fit = _fit(_unit(0.0, 0.0))
    for got_w, want_w in ((fit.path[0], arc[0]), (fit.path[-1], arc[-1])):
        got = (np.asarray(got_w, dtype=np.float64) - centre) / _R
        want = (np.asarray(want_w, dtype=np.float64) - centre) / _R
        assert float(np.linalg.norm(got - want)) < 1e-9, (got, want)


def test_a_region_changes_the_arc_as_little_as_the_gate_allows() -> None:
    """The translation bug's signature, checked for two regions that point apart.

    Under the shadowing, the emitted path rode on the LAST region's own unit radial, so
    for each region the arc sat at radius 1 from the NUCLEUS only if the nucleus happened
    to lie that way. What is asserted per region is that the arc stayed a sphere of
    radius R about the nucleus -- the frame -- and that both endpoints kept welding to the
    CPs they connect.

    NOT asserted: that two regions give the same arc. The region list is an input to the
    fit, not decoration -- it pins ground and gates the control point -- and measured, the
    two runs here emit 23 and 32 vertices, so the fit is genuinely region-sensitive beyond
    the span. Pinning that to an exact value would freeze the layout, not the frame, so it
    is left as an open observation in journal.md rather than a test that can only go red
    when the geometry is working as designed.
    """

    centre = np.asarray(_NUCLEUS, dtype=np.float64)
    arc = _arc(_south_pole_ends())  # _fit builds the same arc internally; deterministic
    for pole in (_unit(0.0, 0.0), _unit(math.pi * 0.5, math.pi)):
        fit = _fit(pole)
        assert fit.changed, fit.status
        radii = np.linalg.norm(np.asarray(fit.path) - centre, axis=1) / _R
        assert float(np.max(np.abs(radii - 1.0))) < 1e-9, (
            "the fitted arc left the sphere by "
            f"{float(np.max(np.abs(radii - 1.0))):.6f} R for an unrelated CP disc"
        )
        for got_w, want_w in ((fit.path[0], arc[0]), (fit.path[-1], arc[-1])):
            got = (np.asarray(got_w, dtype=np.float64) - centre) / _R
            want = (np.asarray(want_w, dtype=np.float64) - centre) / _R
            assert float(np.linalg.norm(got - want)) < 1e-9, (got, want)
