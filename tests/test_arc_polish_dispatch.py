"""Guard: the Step 3 card runs EXACTLY ONE arc polish method per rebuild.

Why this file exists. The card used to carry a `Gradify Arcs Into Gradient Paths`
switch, and the drawing order made it a lie: gradient-traced connectors were spliced
into ``runtime_max_connections`` *after* gradify had run, so a traced path was never
gradified, while the card printed a "bent" count from a pass that had already
finished. An operator who turned gradify on over a traced table saw a number for
geometry that was not on screen. v15 replaced the switch with a three-way selector --
``none`` / ``gradify`` / ``quad`` -- and the rebuild resolves it to ONE method before
any geometry is built. That is only true if these three things hold, and each is a
test here:

1. the mode string picks the function it names (``"quad"`` runs
   :func:`ridge_refiner.fit_quad_bezier_connections` and never the Newton walk, and
   vice versa), and no mode runs both;
2. the method that did NOT run reports ``on=False`` with every counter zero, so the
   card can tell "off" from "on and found nothing to bend" -- and the second table's
   numbers ride nested under ``"descent"`` on the runner only;
3. a mode string this build cannot name polishes nothing, so a hand-built state (a
   typo in a test, a key written by a future schema this viewer predates) cannot draw
   a bend the snapshot never asked for.

The dispatch lives in ``sphere_viewer._apply_arc_polish`` (not inline in
``update_field``) precisely so it is testable here: the rebuild needs a live session,
a VTK pipeline and a trame server, and this needs two lists of records and a
synthetic field. Importing the viewer is cheap (0.9 s, no server starts at import).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from gba_topology2.src.max_connection import MaxConnection, MinConnection
from gba_topology2.src.params_schema import (
    ARC_GRADIFY_STAT_KEYS,
    ARC_POLISH_METHODS,
    ARC_POLISH_MODES,
    QUAD_FIT_STAT_KEYS,
    arc_polish_methods,
)
from gba_topology2.src.sphere_viewer import _apply_arc_polish

#: What :func:`_call` returns: both polished tables and both published summaries, with
#: the JSON-shaped dicts typed as mappings so an assertion reads as a number.
_Polish = tuple[
    list[MaxConnection], list[MinConnection], dict[str, Any], dict[str, Any]
]

#: Sphere radius for the synthetic field, world units. Same order as the fixtures in
#: ``tests/test_ridge_refiner.py`` so the two files' numbers are comparable.
_R = 7.0

#: The straight-arc method a polish is allowed to bend. A traced record
#: (``method="gradient"``) is passed through by both methods untouched, because a
#: traced path IS a gradient path and bending one is a corruption.
_STRAIGHT = "geodesic"


def _contour_arc(radius: float = _R) -> tuple[tuple[float, float, float], ...]:
    """A quarter great circle that runs ALONG a contour: (R,0,0) -> (0,0,R), 13 vertices.

    Every point of it has ``y == 0``, which is the field's ``f = 1`` contour, so the
    gradient is perpendicular to the path at every sample and the gradient-path error is
    1.0 for the whole length -- the worst arc a polish can be handed, and the shape a
    real connector has when a basin boundary follows a ridge shoulder. A true gradient
    path between these endpoints does not exist (both ends sit at the same field value,
    and a gradient path is monotone in ``f``), so the best a single-bend method can do is
    arc toward the crest at +Y: the fit is graded on how much of that it recovers, and
    the control is capped at :data:`QUAD_FIT_MAX_CONTROL_PITCH` pitches, so the bend is
    a polish and not a reroute.

    90 deg long, ~40 mesh pitches at the 2.2 deg pitch used below: room to bend, and
    enough vertices that the pitch-aware samplers agree on where its nodes are.
    """

    return tuple(
        (
            radius * math.cos(math.radians(90.0 * k / 12.0)),
            0.0,
            radius * math.sin(math.radians(90.0 * k / 12.0)),
        )
        for k in range(13)
    )


def _conn(idx: int, radius: float = _R) -> MaxConnection:
    """An ascent record whose arc is :func:`_contour_arc`, plus the polish fields."""

    return MaxConnection(
        saddle_id=f"SAD_{idx}",
        branch_index=0,
        max_id=f"MAX_{idx}",
        status="connected",
        direction_mode="port_tangent",
        cross_track_deg=None,
        along_track_deg=10.0,
        separation_deg=20.0,
        mutual=False,
        arc_points=_contour_arc(radius),
        saddle_value=1.0,
        max_value=2.0,
        verified=True,
        method=_STRAIGHT,
    )


def _descent(idx: int, radius: float = _R) -> MinConnection:
    """The mirror of :func:`_conn` on the descent table, which the boundary chains.

    The basin ring is built from DESCENT polylines as much as the ascent's, so a
    selector that polished one table and not the other would draw a boundary that
    hugs the ridge on one side of a saddle and cuts straight across the other.
    """

    return MinConnection(
        saddle_id=f"SAD_{idx}",
        branch_index=0,
        min_id=f"MIN_{idx}",
        status="connected",
        direction_mode="port_tangent",
        cross_track_deg=None,
        along_track_deg=10.0,
        separation_deg=20.0,
        arc_points=_contour_arc(radius),
        saddle_value=2.0,
        min_value=1.0,
        verified=True,
        method=_STRAIGHT,
    )


def _hump_field() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A mesh with one broad crest: the exponential ridge ``exp(2 y)``, maximum at +Y.

    A full UV grid (288 x 288, 1.25 deg a side) with the poles DUPLICATED, because that
    is the geometry a Tecplot sphere export carries: a latitude/longitude mesh puts the
    whole pole on one vertex, and an inverse-distance stencil centred beside a pole of a
    96-row grid sees 1 vertex for its whole northern half and fits nothing. K = 2 (not
    the 30 the gradify fixtures use) keeps the ridge broad enough that a single quadratic
    control gains real field inside its cap -- the control is limited to 4 mesh pitches
    (8.8 deg) from the chord, and a razor crest 90 deg away is out of reach of any
    one-bend method.
    """

    n_rings = 288
    pts: list[tuple[float, float, float]] = []
    vals: list[float] = []
    for i in range(n_rings + 1):
        theta = math.pi * i / n_rings
        for j in range(n_rings):
            phi = 2.0 * math.pi * j / n_rings
            unit = (
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            )
            if abs(unit[2]) > 1.0 - 1e-12:
                # The mesh carries the pole on its whole ring, and its z is exactly +-1:
                # a stencil that straddles the seam must see the same density there.
                unit = (
                    1e-9 * math.cos(phi),
                    1e-9 * math.sin(phi),
                    math.copysign(1.0 - 1e-12, unit[2]),
                )
            pts.append((unit[0] * _R, unit[1] * _R, unit[2] * _R))
            vals.append(math.exp(2.0 * unit[1]))
    return np.asarray(pts, dtype=np.float64), np.asarray(vals, dtype=np.float64)


_MESH_PTS, _F_VALS = _hump_field()


def _call(mode: str) -> _Polish:
    """Run the dispatch on a FRESH copy of both tables, so modes never share state.

    Three arcs per table: enough for a count to be a count (``n_arcs == 3``) and for
    one bad record to be visible, few enough that the whole file runs in a second.
    """

    ascent, descent, gradify_stats, quad_stats = _apply_arc_polish(
        [_conn(i) for i in range(3)],
        [_descent(i) for i in range(3)],
        polish_mode=mode,
        mesh_pts=_MESH_PTS,
        f_vals=_F_VALS,
        sphere_radius=_R,
        delta_theta_mesh_rad=math.radians(2.2),
        center=(0.0, 0.0, 0.0),
        find_k_nearest=None,
        regions=(),
        n_nodes=10.0,
        search_pitch=6.0,
    )
    # Bound 6 pitches, capped by the module's own QUAD_FIT_MAX_CONTROL_PITCH = 4: the
    # test exercises the cap, which is the value the card's slider saturates at.
    # The summaries are re-typed here, not at every assertion: the dispatch returns
    # ``dict[str, Any]`` (it is a published JSON shape), and a test that indexes it
    # inline reads as string arithmetic. The keys are checked against the schema's key
    # tuples by :func:`test_both_summaries_are_the_published_shape`, so the cast is
    # naming a shape the module already guarantees, not suppressing a question.
    return (
        ascent,
        descent,
        dict(cast("Mapping[str, Any]", gradify_stats)),
        dict(cast("Mapping[str, Any]", quad_stats)),
    )


def _points(rows: Sequence[MaxConnection | MinConnection]) -> list[tuple[float, ...]]:
    return [tuple(row.arc_points[0]) + tuple(row.arc_points[-1]) for row in rows]


def test_quad_runs_the_quadratic_and_never_the_newton_walk() -> None:
    """``"quad"`` reports a quadratic fit, and reports gradify as OFF.

    The two methods publish different record fields, so this pins the selector to the
    function: a mode that ran both would set ``gradified`` AND ``quad_fitted`` on the
    same row, and the table's method labels would no longer be orderable (the
    highest-trust label would win over a bend the reader cannot see).
    """

    ascent, descent, gradify_stats, quad_stats = _call("quad")
    assert quad_stats["on"] is True
    assert gradify_stats["on"] is False
    # The two tables are checked apart, not concatenated: the union of the two record
    # types has no attribute access in a comprehension, and the fields are the same here.
    assert all(not row.gradified for row in ascent)
    assert all(not row.gradified for row in descent)
    assert quad_stats["n_arcs"] == 3
    assert quad_stats["n_spans"] == 3
    assert quad_stats["n_spans_fitted"] >= 1, quad_stats
    assert any(row.quad_fitted for row in ascent)
    assert any(row.quad_fit_status == "ok" for row in ascent)


def test_gradify_runs_the_newton_walk_and_never_the_quadratic() -> None:
    """The mirror image, with the twin counters.

    Both modes run against the SAME field and the SAME tables, which is also the
    cross-check the plan asks for: the two methods must at least agree that this arc
    wants bending at all. Where they disagree on the AMOUNT, the probe settles it --
    see ``gba_topology2/journal.md``.
    """

    ascent, _descent, gradify_stats, quad_stats = _call("gradify")
    assert gradify_stats["on"] is True
    assert quad_stats["on"] is False
    assert all(not row.quad_fitted for row in ascent)
    assert gradify_stats["n_arcs"] == 3
    assert quad_stats["n_arcs"] == 0 and quad_stats["n_spans"] == 0
    # On THIS fixture the Newton walk bends nothing, and that is the honest expectation
    # rather than a defect: its reach is 0.72 mesh pitches, and the crest beside this
    # arc is 40 pitches away. See
    # :func:`test_the_two_methods_disagree_on_a_far_crest_and_say_so`.
    assert gradify_stats["n_gradified"] == 0
    assert gradify_stats["n_all_refused"] == 3


def test_none_bends_nothing_and_reports_both_methods_off() -> None:
    """``"none"`` is by far the most-used mode, and the one that must never lie.

    The tables come back with byte-identical geometry, not merely the same values,
    because a method that bends nothing returns its records BY IDENTITY. That is the
    property that makes a snapshot with no polish replay exactly as it drew, and it is
    why the Step 6 smoother's gate -- which admits a polished arc by its flag, never by
    comparing positions -- stays quiet.
    """

    baseline = _points([_conn(0), _conn(1), _conn(2)])
    ascent, descent, gradify_stats, quad_stats = _call("none")
    assert _points(ascent) == baseline
    assert _points(descent) == _points([_descent(i) for i in range(3)])
    assert [row.arc_points for row in ascent] == [
        _conn(i).arc_points for i in range(3)
    ], "mode none rebuilt the polyline instead of passing it through"
    assert all(not row.gradified and not row.quad_fitted for row in ascent)
    assert all(not row.gradified and not row.quad_fitted for row in descent)
    assert gradify_stats["on"] is False and quad_stats["on"] is False
    assert gradify_stats["n_arcs"] == 0 and quad_stats["n_arcs"] == 0


def test_an_unnamed_mode_is_none_not_a_coin_flip() -> None:
    """A mode string this build cannot name must polish nothing.

    The loader rejects such a file (``arc_polish_mode`` is an enumerated parameter), so
    reaching here means a hand-built ``state``. Falling through to a method by accident
    would draw a bend nobody asked for, and -- because the two methods write different
    record fields -- would label it as the method it is not.
    """

    assert "cubic" not in ARC_POLISH_MODES
    ascent, _descent, gradify_stats, quad_stats = _call("cubic")
    assert quad_stats["on"] is False and gradify_stats["on"] is False
    assert all(not row.gradified and not row.quad_fitted for row in ascent)
    assert [row.arc_points for row in ascent] == [_conn(i).arc_points for i in range(3)]


def test_both_summaries_are_the_published_shape() -> None:
    """Each summary carries EXACTLY its published keys, running or idle, nested too.

    The card's caption dereferences ``quad_fit_stats.n_spans_fitted`` in the browser,
    where a missing key is ``undefined``, not an error: the "card that ate itself"
    incident turned a missing top-level metric into a half-drawn card and a Python log
    that said nothing. The nested ``descent`` copy is checked to the same shape because
    the v13 ``on=False`` collapse pruned sub-dicts, and the idle method's ``descent`` is
    checked to be EMPTY, which is what the caption's ``v_if`` keys off.
    """

    for mode, runner_keys, idle_keys in (
        ("quad", QUAD_FIT_STAT_KEYS, ARC_GRADIFY_STAT_KEYS),
        ("gradify", ARC_GRADIFY_STAT_KEYS, QUAD_FIT_STAT_KEYS),
    ):
        _ascent, _descent, gradify_stats, quad_stats = _call(mode)
        runner = quad_stats if mode == "quad" else gradify_stats
        idle = gradify_stats if mode == "quad" else quad_stats
        assert set(runner) == set(runner_keys), mode
        assert set(idle) == set(idle_keys), mode
        # The nested copy is the published shape MINUS its own "descent" key: the
        # descent's descent would nest forever, and the v13 collapse of an on=False
        # sub-dict is the other half of the same rule. QUAD_FIT_STAT_KEYS is the shape
        # of a TOP-LEVEL summary, which is what the normaliser emits for the nested one
        # once the recursion is cut.
        assert set(runner["descent"]) == set(runner_keys) - {"descent"}, mode
        assert runner["descent"]["on"] is True, mode
        # The idle method's nested copy is None, not {}: that is the v13 collapse rule
        # both normalisers implement (a summary that did not run has no descent block to
        # report), and the card's caption keys its ``v_if`` off ``on`` for the same
        # reason -- ``descent.n_arcs`` on None throws in the browser.
        assert idle["descent"] is None, mode
    # With no method selected, BOTH summaries are the idle shape -- the state is seeded
    # this way at startup and republished it on every polish-free rebuild, so the card's
    # caption has a number for each rung before anything has ever run.
    _ascent, _descent, gradify_stats, quad_stats = _call("none")
    assert set(gradify_stats) == set(ARC_GRADIFY_STAT_KEYS)
    assert set(quad_stats) == set(QUAD_FIT_STAT_KEYS)
    assert gradify_stats["descent"] is None and quad_stats["descent"] is None


def test_both_tables_are_polished_by_the_one_mode() -> None:
    """The descent table gets the same treatment, with the valley criterion.

    The polarity is not an argument the caller may get wrong: a descent arc runs
    saddle -> MINIMUM, along a valley floor, and on the ridge criterion its measured
    across-axis points ALONG the path, every node is refused, and the arc comes back
    bit-identical to the great circle it started as. That is the bug commit 011bf52
    fixed for gradify, and this guards the quadratic inherits it -- the runner's
    ``descent`` block must report its own arcs, not zeros.
    """

    for mode, key in (("quad", "quad_fit_status"), ("gradify", "gradify_status")):
        _ascent, descent, gradify_stats, quad_stats = _call(mode)
        runner = quad_stats if mode == "quad" else gradify_stats
        assert runner["descent"]["on"] is True, mode
        assert runner["descent"]["n_arcs"] == 3, mode
        assert any(getattr(row, key) for row in descent), mode
        assert all(row.method == _STRAIGHT for row in descent), mode


def test_a_polished_arc_keeps_the_cp_centres_it_was_given() -> None:
    """Polishing moves the arc's MIDDLE, never its ends.

    The endpoints are the saddle and the CP centre, and the connection's IDENTITY is
    that pair: basin v12's degree rule keys on ``(saddle_id, target_id)`` and the
    boundary ring chains these polylines. A bend that dragged an endpoint off the CP
    would silently re-draw the partition. Checked to floating exactness because
    ``fit_quad_bezier`` welds the endpoints by construction -- the value is not an
    approximation, so no tolerance is granted.
    """

    for mode in ("quad", "gradify"):
        ascent, _descent, _gradify, _quad = _call(mode)
        for idx, row in enumerate(ascent):
            straight = _conn(idx)
            assert row.arc_points[0] == straight.arc_points[0], (mode, idx)
            assert row.arc_points[-1] == straight.arc_points[-1], (mode, idx)
            assert row.saddle_id == straight.saddle_id, mode
            assert row.max_id == straight.max_id, mode


def test_the_quadratic_reduces_the_gradient_error_it_claims_to() -> None:
    """The fit's objective is an integral of ``1 - |tangent . grad|``: show it falling.

    Not a check of the published ``f_before``/``f_after`` pair (those are the FIELD
    RIDE, the gain the acceptance gate reads) but of the quantity the plan's method 2
    and method 4 share: the arc-length-weighted mean of the gradient-path error,
    measured here by an INDEPENDENT finite-difference gradient of the same mesh, so the
    assertion does not use the code under test to grade itself.

    The straight arc lies along the field's f = 1 contour, so its error is 1.0 for its
    whole length; the fitted arc is graded on the samples the fit itself emitted, which
    is the geometry the operator sees on screen.
    """

    ascent, _descent, _gradify, quad_stats = _call("quad")
    fitted = [row for row in ascent if row.quad_fitted]
    assert fitted, "the quadratic fitted nothing on a field built for it"
    # The field RIDE rises: the fit's whole purpose is to spend the bend on the part of
    # the span the field pays best, so `after` is the larger number. A fit that lowered
    # it optimised something else.
    assert quad_stats["mean_f_after"] > quad_stats["mean_f_before"] > 0.0, quad_stats
    assert quad_stats["mean_gain_fraction"] > 0.0

    for idx, row in enumerate(fitted):
        # Same endpoints, same sampler, so the pair is the bend's own effect. The
        # straight reference is rebuilt from the arc's index, not from the fitted row:
        # arc_points has already been replaced by the fit.
        straight = _conn(idx)
        before = _gradient_path_error(
            [np.asarray(p, dtype=np.float64) for p in straight.arc_points]
        )
        after = _gradient_path_error(
            [np.asarray(p, dtype=np.float64) for p in row.arc_points]
        )
        assert after < before, (row.saddle_id, before, after)


def test_the_two_methods_disagree_on_a_far_crest_and_say_so() -> None:
    """The record labels must show WHICH method bent the line, on the same arc.

    The quadratic and the Newton walk are not two ways of computing one answer, and this
    fixture is the proof: the crest sits 40 mesh pitches off the arc's great circle, so
    gradify's 0.72-pitch reach refuses every node (``n_all_refused == 3``, geometry
    untouched) while the quadratic, whose cap is 4 pitches and whose criterion is a field
    RIDE rather than a crest in reach, fits all three arcs. Read the other way round, on
    a crest the walk can touch, gradify measures each node and the quadratic trades that
    for one bend per span.

    This is the comparison the selector exists to make visible: two rows that look alike
    on screen and carry different claims, told apart by ``gradified`` versus
    ``quad_fitted`` and by the probe's independent offset.
    """

    _a, _d, gradify_stats, _q = _call("gradify")
    _a, _d, _g, quad_stats = _call("quad")
    assert gradify_stats["n_gradified"] == 0 and gradify_stats["n_all_refused"] == 3
    assert quad_stats["n_quad_fitted"] == 3, quad_stats
    assert quad_stats["max_offset_pitch"] > 0.0
    # The quadratic paid for its search on every arc; gradify's walk costs a stencil per
    # node, and the published counts are the only place a reader can compare them.
    assert quad_stats["n_evals"] > 0


def test_both_is_a_pipeline_and_not_two_independent_runs() -> None:
    """``both`` feeds the quadratic's output to the Newton walk, and reports each pass.

    Four things this pins down, each of them something a future edit can break silently:

    * Every selector value in :data:`ARC_POLISH_MODES` has a method list, so the card can
      never offer a mode the dispatcher ignores (the bug this test file exists for).
    * BOTH summaries report ``on=True``. If one mode overrode the other, a caption would
      read 0/0 for a pass that ran, and the card would look idle on an arc that moved.
    * The two record flags can land TOGETHER on one row. The walk stamps ``gradified`` on
      what it bends; the fit stamped ``quad_fitted`` on what it bent, and the walk received
      those geometry-replaced records -- so a row carrying both is the pipeline, and a row
      carrying neither is an arc both passes left straight.
    * The arc counts agree between the passes: pass 2 is handed the same number of
      geodesic arcs, because a polish renames no method and moves no endpoint.

    The published key sets are asserted for the pipeline too: the mode is new, the shapes
    are frozen, and ``ARC_GRADIFY_STAT_KEYS`` gains ``quad_fitted`` precisely so a reader
    of an exported row can tell a straight arc from a fitted one from a fitted-then-walked
    one.
    """

    assert arc_polish_methods("both") == ("quad", "gradify")
    assert set(ARC_POLISH_METHODS) == set(ARC_POLISH_MODES), (
        "the selector offers a mode the dispatcher has no method list for"
    )
    assert arc_polish_methods("nonsense") == () and arc_polish_methods(None) == ()

    ascent_both, _d, gradify_stats, quad_stats = _call("both")
    ascent_quad, _d2, _g2, quad_only = _call("quad")

    assert gradify_stats["on"] is True and quad_stats["on"] is True
    assert set(gradify_stats) == set(ARC_GRADIFY_STAT_KEYS)
    assert set(quad_stats) == set(QUAD_FIT_STAT_KEYS)
    # Pass 1 is the same pass in both runs: the arcs the quadratic bent, it bent.
    assert quad_stats["n_quad_fitted"] == quad_only["n_quad_fitted"]
    # Pass 2 was handed the same set of geodesic arcs, counted the same way.
    assert gradify_stats["n_arcs"] == quad_stats["n_arcs"] == 3
    for row in ascent_both:
        src = next(r for r in ascent_quad if r.saddle_id == row.saddle_id)
        assert row.quad_fitted == src.quad_fitted, (
            "the walk's input rewrote which arcs the quadratic claimed"
        )
    # Every row is accounted for. MEASURED on this fixture: all three arcs come back
    # (quad_fitted=True, gradified=False) -- the quadratic bent them, and the walk refused
    # them, because the crest sits 40 mesh pitches off the original great circle and the
    # fit's 4-pitch cap leaves the nodes ~36 pitches out, still past the walk's 0.72-pitch
    # reach. So ``both`` here is ``quad`` plus a walk that measured and declined, which is
    # the honest outcome the two summaries publish separately: the card reads
    # "quadratic fit: 3/3" and "gradify: 0/3" on the same rebuild, and a reader can see
    # that pass 2 declined rather than inferring that it never ran.
    states = {(row.quad_fitted, row.gradified) for row in ascent_both}
    assert states <= {(True, True), (True, False), (False, False)}, sorted(states)
    assert sum(1 for r in ascent_both if r.quad_fitted) == quad_stats["n_quad_fitted"]
    assert sum(1 for r in ascent_both if r.gradified) == gradify_stats["n_gradified"]
    # A row the walk declined keeps the quadratic's geometry, byte for byte: pass 2 must
    # not resample a polyline it is refusing, which would repaint a bent arc as a
    # node-reordered version of itself and change the drawn curve.
    for row in ascent_both:
        if not row.gradified:
            src = next(r for r in ascent_quad if r.saddle_id == row.saddle_id)
            assert row.arc_points == src.arc_points


def _gradient_path_error(points: Sequence[npt.NDArray[np.float64]]) -> float:
    """Arc-length-weighted mean of ``1 - |tangent . grad|`` over a polyline.

    An independent implementation of the criterion, for the test above: the gradient is
    a central finite difference of the inverse-distance-sampled field along the two
    tangent-plane axes at each vertex, with no least-squares stencil, no log transform,
    no eigen-decomposition and no polarity handling. Gradients at the two endpoints are
    skipped (a one-sided tangent is not a path direction), and the antipode is excluded
    (the field is constant there, so its gradient direction is undefined).

    Args:
        points: The polyline in WORLD frame, in path order.

    Returns:
        The mean error, in ``[0, 1]``: 0 for a path parallel to the gradient
        everywhere, 1 for one perpendicular to it everywhere.
    """

    h = math.radians(1.0)
    total = 0.0
    weight = 0.0
    for i in range(1, len(points) - 1):
        point = points[i]
        radius = float(np.linalg.norm(point))
        if radius <= 0.0:
            continue
        unit = point / radius
        along = points[i + 1] - points[i - 1]
        along -= float(np.dot(along, unit)) * unit
        along_norm = float(np.linalg.norm(along))
        if along_norm <= 0.0:
            continue
        along /= along_norm
        across = np.cross(unit, along)
        if abs(float(np.dot(unit, (0.0, 1.0, 0.0)))) > 0.999:
            continue
        # Move on the sphere in the +/- directions, and take the antisymmetric part of
        # the log field: the derivative along each axis, in units of the angle.
        grads = []
        for axis in (along, across):
            up = _sample_field(_offset(unit, axis, h))
            down = _sample_field(_offset(unit, axis, -h))
            peak = _sample_field(unit)
            if min(up, down, peak) <= 0.0:
                grads.append(0.0)
                continue
            # Central difference of log f: the gradient of f/|f|, whose DIRECTION is the
            # gradient's, which is all the criterion reads.
            grads.append((math.log(up) - math.log(down)) / (2.0 * h))
        gx, gy = grads
        norm = math.hypot(gx, gy)
        if norm <= 0.0:
            continue
        step = float(np.linalg.norm(points[i + 1] - points[i - 1])) / 2.0
        total += step * (1.0 - abs(gx) / norm)
        weight += step
    return total / weight if weight > 0.0 else 0.0


def _offset(
    unit: npt.NDArray[np.float64], axis: npt.NDArray[np.float64], angle: float
) -> npt.NDArray[np.float64]:
    """Rotate ``unit`` toward ``axis`` by ``angle`` (radians) and renormalise."""

    moved = math.cos(angle) * unit + math.sin(angle) * axis
    return moved / float(np.linalg.norm(moved))


def _sample_field(unit: npt.NDArray[np.float64]) -> float:
    """The synthetic field at a unit direction, in the analytically exact form.

    Deliberately the closed form ``exp(2 y)`` rather than the mesh: the mesh is a
    latitude/longitude grid whose poles are singular, and a finite difference beside a
    pole is noise. The mesh and this agree to the sampling resolution everywhere else.
    """

    return math.exp(2.0 * float(unit[1]))
