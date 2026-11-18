"""Guard: every state attribute a card template reads must exist on the state.

Background (the incident this file exists to prevent): the
``Basins: Step 3 Boundaries`` card printed a readout bound to
``trace_connector_stats.n_traced``. The metric was written into
``state.connection_stats`` -- the nested copy the exporter binds to the
records -- but never onto the top-level state the template reads. Turning the
``Gradient Traced Connectors`` switch on therefore re-rendered the card, the
expression evaluated to ``undefined``, the browser threw

    TypeError: undefined is not an object (evaluating 'trace_connector_stats.n_traced')

and the card's whole body disappeared -- title and ``n_connected`` counter
only, every switch gone. A silent visual regression, not a crash, so nothing in
the Python logs said anything and the re-exported JSON simply lacked the metric.

Two lessons are encoded as tests here:

1. A trame template is evaluated in the browser against the client-side state.
   An attribute the server never defined is ``undefined``, not an error, until
   the template dereferences it -- so the failure is a half-drawn card rather
   than a traceback.
2. Every metric the card reports needs a TOP-LEVEL state key, because the
   template cannot read through ``connection_stats`` for the summary line; the
   nested copy exists for the export/reproduce path and is a different reader.

The tests are static (they read ``sphere_viewer.py`` as text) on purpose: the
state is only fully built inside ``run_sphere_viewer``, which needs a live trame
server, so a runtime test would be a launch test -- exactly the kind of check
that would have to be run by hand every time to catch this.
"""

from __future__ import annotations

import pathlib
import re

from gba_topology2.src.params_schema import DEFAULT_UI_STATE

#: Path of the module under test, resolved from the repository root so the test
#: does not depend on the current working directory.
_VIEWER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "gba_topology2"
    / "src"
    / "sphere_viewer.py"
)

#: JavaScript globals an interpolation may legitimately reference. Anything else
#: has to be a state attribute or a ``v_for`` loop variable.
_JS_GLOBALS = frozenset(
    {
        "Array",
        "Boolean",
        "Date",
        "JSON",
        "Math",
        "Number",
        "Object",
        "RegExp",
        "String",
        "console",
        "decodeURIComponent",
        "encodeURIComponent",
        "false",
        "isNaN",
        "null",
        "parseFloat",
        "parseInt",
        "true",
        "undefined",
    }
)

#: Interpolation, e.g. ``"{{ connection_stats.n_connected }}/..."``.
_INTERPOLATION = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)

#: A directive whose value is a JavaScript expression: ``v_if``, ``v_show`` and
#: trame's ``v_model`` tuple form. These throw exactly like an interpolation does
#: when the root is ``undefined`` -- ``v_if="arc_gradify_stats.on"`` with no such
#: attribute is a TypeError during patch, not a falsy guard.
_DIRECTIVE = re.compile(r'v_(?:if|show|model)="([^"]*)"')

#: Leading identifier of an interpolation -- the root the state has to provide.
_ROOT = re.compile(r"\s*([A-Za-z_$][A-Za-z0-9_$]*)")

#: ``v_for="cl in clusters_list"`` and the destructured ``(w, i) in ...`` form,
#: which may also be an f-string (``v_for=f"(w, i) in ..."``). These bind the loop
#: variables an interpolation inside the repeated node reads.
_LOOP = re.compile(
    r'v_for=(?:f)?"\(?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)?'
    r"(?:\s*,\s*\(?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)?)?\s+in\s"
)

#: Every attribute the viewer ever assigns on the trame state.
_ASSIGNED = re.compile(r"\bstate\.([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _source() -> str:
    return _VIEWER.read_text(encoding="utf-8")


def _interpolation_roots(source: str) -> set[str]:
    """Return the root identifier of every ``{{ ... }}`` interpolation.

    Args:
        source: The viewer module's text.

    Returns:
        The set of leading identifiers, e.g. ``{"connection_stats"}`` for
        ``{{ connection_stats.n_connected }}``.
    """
    roots: set[str] = set()
    for expression in _INTERPOLATION.findall(source):
        match = _ROOT.match(expression)
        if match:
            roots.add(match.group(1))
    return roots


def _directive_roots(source: str) -> set[str]:
    """Return the root identifier of every directive expression.

    Args:
        source: The viewer module's text.

    Returns:
        The set of leading identifiers, e.g. ``{"trace_connectors"}`` for
        ``v_if="trace_connectors"``. ``v_model`` is a trame tuple
        (``v_model=("trace_connectors",)``), so the leading ``(`` is skipped.
    """
    roots: set[str] = set()
    for expression in _DIRECTIVE.findall(source):
        match = _ROOT.match(expression.lstrip("("))
        if match:
            roots.add(match.group(1))
    return roots


def _loop_variables(source: str) -> set[str]:
    """Return every variable bound by a ``v_for`` in the module.

    Args:
        source: The viewer module's text.

    Returns:
        Loop variable names, including the destructured index of ``(w, i)``.
    """
    names: set[str] = set()
    for first, second in _LOOP.findall(source):
        names.add(first)
        if second:
            names.add(second)
    return names


def _state_attributes(source: str) -> set[str]:
    """Return every attribute the viewer defines on the trame state.

    Combines the canonical defaults seeded from ``DEFAULT_UI_STATE`` with every
    ``state.<name> = ...`` assignment in the module, so a key that is only
    written during a rebuild still counts as defined.

    Args:
        source: The viewer module's text.

    Returns:
        Attribute names available to a template.
    """
    return set(DEFAULT_UI_STATE) | set(_ASSIGNED.findall(source))


def test_every_template_root_resolves_to_state_or_loop_variable() -> None:
    """No interpolation may name an attribute the state never defines.

    This is the invariant the crashed card broke: a root that is neither a
    ``v_for`` binding, a JS global, nor a state attribute renders as
    ``undefined`` and throws on first dereference.
    """
    source = _source()

    unresolved = (
        _interpolation_roots(source)
        - _loop_variables(source)
        - _state_attributes(source)
        - _JS_GLOBALS
    )

    assert not unresolved, (
        "template interpolation reads state the viewer never defines: "
        f"{sorted(unresolved)}. The card will render as undefined and throw "
        "in the browser."
    )


def test_every_directive_root_resolves_to_state_or_loop_variable() -> None:
    """No ``v_if``/``v_show``/``v_model`` may name an attribute the state omits.

    The interpolation test above is not enough on its own: a directive reading
    ``undefined`` is a TypeError during patch, not a falsy guard, so a card
    gated on a missing key dies the same way a card interpolating one does.
    """
    source = _source()

    unresolved = (
        _directive_roots(source)
        - _loop_variables(source)
        - _state_attributes(source)
        - _JS_GLOBALS
    )

    assert not unresolved, (
        "template directive reads state the viewer never defines: "
        f"{sorted(unresolved)}. The node will throw in the browser."
    )


def test_traced_connector_summary_is_defined_before_anything_is_measured() -> None:
    """``trace_connector_stats`` is a top-level key, not only a nested one.

    The nested copy under ``connection_stats`` is what the export binds to the
    records; the card's readout is a separate reader and needs the top-level
    key. Both exist before the first rebuild so that flipping the switch cannot
    render a card whose body is missing.
    """
    source = _source()

    assert "state.trace_connector_stats" in source, (
        "the traced-connector summary must be seeded on the state; the card "
        "reads trace_connector_stats.<counter> and an undefined root takes the "
        "whole card body with it"
    )
    assert "trace_connector_stats" in _state_attributes(source)
    # The template really does read it, so the two tests above and below cannot
    # drift into guarding a key nobody renders.
    assert "trace_connector_stats" in _interpolation_roots(source)
    # And the nested copy is still written for the exporter.
    assert 'state.connection_stats["trace_connector_stats"]' in source


def test_gradification_summary_is_defined_before_anything_is_measured() -> None:
    """``arc_gradify_stats`` carries the same two-reader contract.

    Kept as its own assertion because the gradify switch shipped first and set
    the pattern; if someone removes the top-level seed here, the same
    half-drawn card comes back for the gradify line.
    """
    source = _source()

    assert "state.arc_gradify_stats" in source
    assert "arc_gradify_stats" in _state_attributes(source)
    assert "arc_gradify_stats" in _interpolation_roots(source)
    assert 'state.connection_stats["arc_gradify_stats"]' in source


def test_a_summary_key_is_seeded_before_the_rebuild_that_writes_it() -> None:
    """Seeding inside the rebuild is not enough: the first paint has no rebuild.

    The card renders as soon as the UI is built, and a readout bound to a key that
    only appears in ``update_topology`` is ``undefined`` for that first paint -- the
    same half-drawn card, just narrower. Both summaries are therefore seeded in the
    initial-state block, which the source text puts above the rebuild that overwrites
    them, so the seed line precedes the first write of ``connection_stats``.
    """
    source = _source()

    seed_trace = source.index(
        "state.trace_connector_stats = make_trace_connector_stats("
    )
    seed_gradify = source.index("state.arc_gradify_stats = make_arc_gradify_stats(")
    # The rebuild's own write, not the seed-time ``summarize_connections([])`` a few
    # lines above the seeds -- that one IS part of the seed block.
    first_rebuild = source.index(
        "state.connection_stats = summarize_connections(runtime_max_connections)"
    )

    assert seed_trace < first_rebuild, (
        "the traced-connector seed must run at seed time, not in the rebuild; "
        "the card paints before anything has been measured"
    )
    assert seed_gradify < first_rebuild, (
        "the gradification seed must run at seed time, same reason"
    )


def test_the_smoothing_post_pass_is_applied_to_both_tables() -> None:
    """``Coarse Structural Curves`` is one switch over both connector tables.

    The post-pass used to run on the ascent table only, so the saddle->min
    connectors -- and the basin boundary, which the overlay projects from the
    DESCENT rows' polylines -- stayed raw sawtooth under a label saying they had been
    reduced to structural curves. The switch's meaning is about the DRAWN line, and
    the drawn line exists on both sides, so both tables go through the call, with one
    mode, and the descent's rows are written after it.
    """
    source = _source()

    # ONE call site, generic over the record: the smoothing arguments exist once, so a
    # table cannot be left out of them -- the 2x2 of explicit calls (fidelity/coarse x
    # ascent/descent) is where the descent table was dropped.
    assert source.count("refine_connections(") == 1, (
        "the smoothing rule must be written once; a second copy of the argument list "
        "is a second chance to smooth only one table"
    )
    assert "def smooth(table: list[_ConnectionT]) -> list[_ConnectionT]:" in source, (
        "the single call site must stay generic over the record, so the ascent keeps "
        "list[MaxConnection] and the descent keeps list[MinConnection]"
    )
    assert "runtime_max_connections = smooth(runtime_max_connections)" in source
    assert "runtime_min_connections = smooth(runtime_min_connections)" in source, (
        "the smoothing must reach the descent table; a traced valley chatters for the "
        "same reason an ascent does"
    )
    # ONE coarse value for both tables: a boundary fitted as a fidelity curve beside
    # a connector fitted as a structural curve is not one picture.
    assert source.count("coarse=coarse_curves") == 1
    assert "coarse=bool(state.coarse_ridge_curves)" not in source
    # The basin overlay reads the rows, not the records, so writing them before the
    # post-pass renders the chatter the operator just switched off. The seed
    # (``state.min_connections_list = []``) is not the write under test.
    descent_pass = source.index(
        "runtime_min_connections = smooth(runtime_min_connections)"
    )
    descent_rows = source.index(
        # Anchored on the WRITE, not on the serializer's argument list: what this
        # asserts is an ordering, and pinning the call text makes the test fail on any
        # signature change (it did, when the rows began carrying arc_polish_mode) while
        # protecting nothing. The trailing newline keeps it off the seed statement
        # (``state.min_connections_list = []``), which is not the write under test.
        "state.min_connections_list = [\n"
    )
    assert descent_pass < descent_rows, (
        "the descent rows must be written after the smoothing post-pass"
    )
    # And the card reads the descent's numbers off the descent's own metric, so the
    # numbers it prints are measured on the table it is describing.
    assert "**summarize_smoothing(runtime_min_connections)" in source
    assert "descent_connection_stats.n_refined" in source


def test_summary_on_flags_follow_their_control_not_a_constant() -> None:
    """``on`` is the control, so an all-zero summary cannot claim it ran.

    The gradification used to write ``on=True`` unconditionally, which did two
    damage: the card's ``v_if=\"arc_gradify_stats.on\"`` line printed
    \"0/0 gold arcs bent\" for an operator who never switched it on, and a saved
    file recorded an ``on`` the replay could not reproduce. Each summary is now
    built with the state's own flag, and the nested copy is the SAME object, so
    the card line and the exported measurement report one number.

    v15 turned the gradify switch into the ``arc_polish_mode`` selector, which
    makes the invariant sharper rather than weaker: a summary may only report
    ``on=True`` for a pass that ran. A constant would let the quadratic claim it ran in
    a session that asked for gradify. ``both`` runs the two methods in a fixed PIPELINE,
    so the flag is MEMBERSHIP in the mode's method list, not equality with the mode:
    ``polish_mode == "gradify"`` is true for one mode and false for the two that also
    run gradify, which suppressed the second summary -- computed, exported, never drawn.
    """
    source = _source()

    assert 'state.arc_gradify_stats["on"] = "gradify" in polish_methods' in source
    assert 'state.quad_fit_stats["on"] = "quad" in polish_methods' in source
    assert "on=bool(state.trace_connectors)" in source
    # The shapes the bug shipped as: a constant True, a nested copy rebuilt from a
    # second call that can drift from the metric the card shows, and equality with the
    # MODE -- true for one selector value and false for every mode that also runs the
    # method, so a pipeline mode loses the summary of the pass it ran first.
    assert "make_arc_gradify_stats(arc_gradify_stats, on=True)" not in source
    assert 'state.arc_gradify_stats["on"] = polish_mode == "gradify"' not in source
    assert 'state.quad_fit_stats["on"] = polish_mode == "quad_fit"' not in source
    assert (
        'state.connection_stats["trace_connector_stats"] = trace_connector_stats'
        in source
    )
    assert 'state.connection_stats["trace_connector_stats"]["descent"]' not in source
