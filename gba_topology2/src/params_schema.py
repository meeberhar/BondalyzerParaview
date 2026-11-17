"""Parameter Snapshot Schema for gba_topology2.

The sphere viewer can export its complete observable state to JSON ("Save
Parameters") and read it back ("Load Parameters"). This module owns the
*format* of that snapshot, so the viewer, the headless diagnostic runner and
the tests all agree on one definition instead of three copies.

The snapshot has five top-level sections:

``schema_version``
    Bumped on any breaking change to the layout below, so a stale file is
    rejected rather than silently misread.
``params``
    Every editable control, split into ``bools`` / ``floats`` / ``strings``
    by the type its widget produces. This is the only section that is fed
    back into the viewer's state.
``context``
    The run context: the absolute .plt path, the atom number and the
    discovered condensed field list. A loader uses it to decide whether a
    different file or atom has to be opened.
``camera``
    The view camera (direction, zoom, projection), stored relative to the
    atom's nucleus so it travels with the atom when a different file or atom
    is loaded. Optional: files written before the camera existed still load.
``derived_metrics`` / ``selection``
    Diagnostics captured at observation time. These are outputs, never inputs:
    a loader must not feed them back into the pipeline.

Everything here is pure and UI-free, so the format can be exercised without a
browser, a render window or a viewer instance.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from gba_topology2.src.camera_state import (
    CameraState,
    camera_section_to_dict,
)

__all__ = [
    "ARC_GRADIFY_STAT_KEYS",
    "ARC_POLISH_DEFAULT",
    "ARC_POLISH_MODES",
    "DEFAULT_UI_STATE",
    "PARAMS_SCHEMA_VERSION",
    "PARAM_BOOL_KEYS",
    "PARAM_CONTEXT_STRING_KEYS",
    "PARAM_CONTROL_KEYS",
    "PARAM_ENUM_KEYS",
    "PARAM_FLOAT_KEYS",
    "PARAM_LEGACY_DEFAULT_OVERRIDES",
    "PARAM_LEGACY_SCHEMA_VERSIONS",
    "PARAM_METRIC_KEYS",
    "PARAM_STRING_KEYS",
    "QUAD_FIT_STAT_KEYS",
    "RIDGE_PROBE_STAT_KEYS",
    "TRACE_CONNECTOR_STAT_KEYS",
    "arc_polish_methods",
    "default_params_path",
    "load_params_from_dict",
    "make_arc_gradify_stats",
    "make_quad_fit_stats",
    "make_ridge_probe_stats",
    "make_trace_connector_stats",
    "save_params_to_dict",
]


# The basin Step 3 polish selector, in the order the card lists them. ``"both"`` runs
# the quadratic and THEN the Newton walk on its output, so it is a method in its own
# right and not just the two single-selects glued together: see
# :func:`arc_polish_methods` for why the
# order is that way round. Named here (and
# not in the viewer) so the widget's options, the migration, and the replay's validation
# all key off one definition: an out-of-range value is rejected at load, never coerced,
# because a selector that silently became "none" would un-bend every arc in a snapshot
# whose author asked for the opposite.
ARC_POLISH_MODES: tuple[str, ...] = ("none", "gradify", "quad_fit", "both")
# ``both`` was ADDED to a v15 enum that is still unreleased (v15 arrived in the same
# unreleased series of commits as the selector itself), so widening it costs no schema
# version: every value any v15 file can carry is still in the list, and the loader
# validates against THIS tuple. The next agent should not "fix" this by bumping to v16 --
# a version is bumped when a file's KEYS change, and no key moved here.
# What each selector value asks for, as the ordered methods to run. The viewer's dispatch,
# the CLI's, and the tests all read this map, so the mode a card offers and the geometry a
# snapshot replays cannot drift apart.
#
# NOTE the deliberate asymmetry: a KEY is the published selector VALUE ("quad_fit", what a
# snapshot stores and the card's v-model carries), a VALUE is the internal METHOD token
# ("quad", compared in the dispatch loops and never written to disk). They are two
# namespaces, and the map is the only translation between them -- which is why the loops
# test `method == "quad"` while the schema validates `arc_polish_mode`. Renaming the
# published value is a schema change; renaming the token is not.
ARC_POLISH_METHODS: dict[str, tuple[str, ...]] = {
    "none": (),
    "gradify": ("gradify",),
    "quad_fit": ("quad",),
    # Quadratic first, then the walk, and the order is load-bearing: the quadratic is a
    # whole-span search with a 4-pitch reach, and the walk is a 0.72-pitch local step. Run
    # them the other way and the walk's work is largely undone by a span-wide bend that
    # ignores where the walk landed, while this order lets the walk MEASURE the nodes the
    # quadratic could not reach -- it inherits the bend and refines it node by node. The
    # two summaries then read as a pipeline: the fit's counts are what pass 1 bent, the
    # walk's are what pass 2 bent ON TOP.
    "both": ("quad", "gradify"),
}


def arc_polish_methods(polish_mode: object) -> tuple[str, ...]:
    """The ordered polish methods one selector value asks for.

    Args:
        polish_mode: A selector value. Anything unrecognised (including the ``None`` trame
            hands back for an unset key) runs nothing, which is the safe direction for a
            geometry control: an arc stays straight rather than bending by a method nobody
            chose.

    Returns:
        The method names in the order to run them, ``()`` for "none".
    """
    return ARC_POLISH_METHODS.get(
        polish_mode if isinstance(polish_mode, str) else "", ()
    )


# "none" is the default because it is what every released build drew: the switch this
# selector replaces, ``gradify_arcs``, shipped OFF, so a fresh session that picks "none"
# is byte-identical to the screen an operator had yesterday. A default of "gradify"
# would silently bend the interior of every connector in every new session -- the same
# class of change the ``terminus_capture`` default was scrutinised for, and the reason
# ``PARAM_LEGACY_DEFAULT_OVERRIDES`` fills v2-v14 with "none" too.
ARC_POLISH_DEFAULT: str = "none"


# Canonical baseline for every interactive control in the viewer. Consumed
# both by the initial state seeding in ``run_sphere_viewer`` and by the
# "Reset to Defaults" action, so the two can never drift apart.
# NOTE: "selected_field" is intentionally absent -- the default field is data
# dependent (the first discovered condensed array) and supplied at runtime.
DEFAULT_UI_STATE: dict[str, Any] = {
    # Field & mesh display
    "show_flood": True,
    "show_contours": True,
    "show_sphere_wire": False,
    "num_contours": 15,
    # Step 1: Morse detection
    "show_cps": True,
    "show_maxima": True,
    "show_minima": True,
    "show_saddles": True,
    "cp_glyph_scale": 1.0,
    # Step 2: Micro-clustering
    "k_pitch": 2.0,
    "show_rings": True,
    # Step 4: Saddle-string / corridor aggregation
    "aggregate_strings": False,
    "k_bridge": 3.0,
    # Step 5: Catastrophes & excision boundary ports
    "show_catastrophes": True,
    "show_ports": True,
    "cat_barrier_tol": 0.005,
    "cat_angular_tol_mult": 4.0,
    # Step 6: Effective CP reduction & harmonic nudge
    "reduce_catastrophes": False,
    "reduce_multi_clusters": False,
    "nudge_cp_positions": False,
    "perfect_port_angles": False,
    "nudge_max_pitch": 1.5,
    # Basin Locator Step 2: saddle separatrix launch seeds. Seeds launch from
    # each saddle's excision boundary, expanded until the port signature is pure.
    # The ring step, ceiling and sample count are deliberately NOT controls: they
    # are module constants in excision_launch_seeds, because the topology found
    # must not depend on operator tuning.
    "show_launch_seeds": False,
    # Basin Locator Step 3: maximum connections. Each ascent branch extrapolates
    # a great-circle ray and claims the maximum it points at; the arcs are drawn
    # only from the SELECTED maximum, so this is an inspection overlay. The arc
    # length cap is likewise a module constant (max_connection.MAX_ARC_DEG, in
    # degrees), not a slider, for the same reason as the ring geometry above.
    "show_max_saddle_arcs": False,
    # Basin Locator Step 6: the coarse structural curve. A gradient-traced
    # connector is an ascent at one mesh pitch per step, so it carries the
    # integrator's step size as chatter and, on a spiralling field, as winding.
    # This replaces each CYAN connector with one smooth spherical curve through its
    # structural nodes -- the launch port, the maximum, the detected corners (which
    # stay sharp), the transited excision centres, and points spaced several mesh
    # pitches along the arc between them. Geodesic connectors are never touched -- a
    # great circle is not an approximation of anything. Off by default: the traced
    # ascent is the measurement and the curve is an interpretation of it, and the
    # operator must be able to see both.
    #
    # The key name is kept from the Entry-058 transverse nudge that this replaced.
    # The semantic change is invisible to the file format: every snapshot on disk
    # has this control OFF, so a v6 snapshot replays identically, and a snapshot
    # saved with it on asks for "the interpretation of the ascent", which is what
    # this is. Renaming it would force a v7 for no change in what a file means.
    "refine_ridge_paths": False,
    # Basin Locator Step 6, the coarse rendering. The structural curve: ONE smooth
    # spherical curve through a handful of nodes -- the launch port, the maximum, the
    # detected corners (which stay sharp), the transited excision centres, and points
    # spaced NODE_PITCH mesh pitches along the arc between them. Where the fidelity
    # smoother preserves the ascent and removes only mesh-scale chatter, this replaces
    # it: the exact path of the ascent is a property of the integrator's step size, and
    # the 3D representation wants the connection. The node spacing is a low-pass
    # filter's cutoff, not a density knob -- at 1 pitch a spiralling connector is
    # reproduced almost exactly (65935 of 66010 deg of turning), at 8 it is filtered to
    # 651 deg. Takes precedence over refine_ridge_paths when both are on, since the
    # two are alternative descriptions of the same drawn line.
    "coarse_ridge_curves": False,
    # Ridge PROBE (diagnostic). Marks, on the SELECTED connector, the sharp
    # continuous bends (black) and the ridge the field puts beside the traced path
    # (white), computed from the Hessian of log f over a wide disc. The path is NOT
    # moved: this measures how far off-crest the ascent runs -- the drift the
    # smoother deliberately leaves alone, since removing a tooth is a change to the
    # path's shape, not a shift across it. Off by default; it costs one disc fit per
    # sample on the visible paths only.
    "probe_ridge_points": False,
    # ARC POLISH METHOD (basin Step 3). The geodesic connector is the ray's answer about
    # an ENDPOINT, drawn as a great circle; the great circle is not a gradient path, and
    # on a field whose crests curve it cuts across the contours. Exactly one method bends
    # its interior, and the endpoints NEVER move, so the connection's IDENTITY is
    # untouched: which saddle, which maximum, which basin boundary it bounds.
    #   "none"    -- draw the ray's line. The straight arc is the measurement of the ray.
    #   "gradify" -- the probe's own ridge criterion at N free nodes, refusals carried
    #                from their accepted neighbours, CP-center and excision-region ground
    #                pinned on the great circle (``ridge_refiner.gradify_arc``). Each node
    #                is measured, and a measurement can REFUSE, which is why 74% of the
    #                nodes it publishes are carried rather than measured.
    #   "quad_fit" -- the polynomial ladder's first rung: one control point per free span,
    #                placed to maximise the arc-length-weighted mean of the function along
    #                the span (``ridge_refiner.fit_quad_bezier``). Nothing is measured, so
    #                nothing refuses on a flat field; the price is that the answer is the
    #                best quadratic, not the crest, and a bend needs a GAIN to be accepted
    #                (``quad_fit_search_pitch`` bounds where it looks).
    # Three-way, not a second switch, because the methods are ALTERNATIVES: run both and
    # the quadratic is fitted to a curve gradify already bent, and the published numbers
    # describe a composition no operator asked for. It also makes "off" a named choice.
    # The mode is a REBUILD trigger: it changes geometry, not presentation.
    "arc_polish_mode": ARC_POLISH_DEFAULT,
    # Free nodes per arc -- a BUDGET, spent on the measurable ground between the
    # pins, in proportion to each span's arc length. 20 is the operator's number.
    # It is a count, not a spacing: on a 12 pitch span the steps land at ~0.6 pitch,
    # which is DENSER than the traced connectors the Step 6 smoother eats, and the
    # emitted line is walked to one pitch so the smoother's pitch-scaled windows see
    # the whole bend. SHARED by both polish methods, on purpose: the same slider is the
    # same sampling of the same arc, which is what makes the two modes' numbers
    # comparable arc for arc.
    "arc_gradify_nodes": 20.0,
    # QUADRATIC FIT SEARCH RADIUS, in mesh pitches: how far the fit LOOKS for a control
    # point, measured from the chord's own midpoint. Independent of the bend cap it is
    # searched against (a module constant, QUAD_FIT_MAX_CONTROL_PITCH) so the two
    # questions stay separate: "how wide a search" vs "how far may an arc bend". A search
    # wider than the cap finds its best at the disc's edge and is then REFUSED by the cap,
    # which is the gate reporting "there is a crest out there, further than this arc may
    # bend for it" instead of silently spending the wider search.
    "quad_fit_search_pitch": 2.0,
    # GRADIENT TRACED CONNECTORS (basin Step 3). The straight great-circle arc is
    # the ray's answer about an ENDPOINT, not a gradient path: on a field whose
    # crests curve it cuts across the contours. This traces the connector with the
    # field instead -- uphill from the saddle's port for an ascent, downhill for a
    # descent -- and DRAWS the traced path. The ray is not discarded: it still runs
    # and still names a candidate maximum or minimum, but as a HYPOTHESIS. A trace
    # that arrives there vindicates the ray; one that arrives elsewhere moves the
    # association and the ray's id stays on the record (``ray_predicted_id``); one
    # that arrives nowhere leaves the ray's verdict standing, with the traced stub
    # drawn beside the dim straight connector. Off by default: it changes the basin
    # ADJACENCY on any field where the ray and the field disagree, and it costs one
    # integration per branch. Independent of ``gradify_arcs`` -- a traced path is
    # already a gradient path, so gradifying it would bend the same geometry twice.
    "trace_connectors": False,
    # TERMINUS CAPTURE (basin Step 3, the companion of ``trace_connectors``). A
    # gradient path stops at the first SAMPLE inside its target's disc, and a
    # disc is a cluster's -- one EFFECTIVE CP standing for a whole family of
    # reduced CPs, on ground the reduction declared untrustworthy. Inside it the
    # path is free to run on to a spurious extremum and stop there: measured on
    # ethene3 shape index (atom 1, "connect", mesh pitch 2.228 deg), the descents
    # into ``E-MIN_MS5`` / ``E-MIN_MS4`` (discs of 15.45-15.50 deg) reach the
    # center EXACTLY and then spend their last step on a spurious well, ending
    # 3.39 deg (1.52 pitches) from the CP they name; six of the twenty connectors
    # are more than half a pitch off, mean 1.11 deg. With this on the path is
    # truncated at its closest approach and the rest of the way is carried
    # straight to the CENTRE, so the connector ends at the CP the table names and
    # the cluster's interior detail is drawn once, not traced into. Seventeen of
    # the twenty then need no leg at all -- the cut is the whole defect -- and the
    # longest carry is 9.73 deg (4.37 pitches), ``E-SAD_MC1 -> E-MIN_MS5``. The
    # cost is explicit and published (``completed_leg_deg``,
    # ``max_completed_leg_deg``):
    # that much of the connector is drawn, not measured. The coarse structural
    # curve (``coarse_ridge_curves``) bends the resulting turn, which is why the
    # two are used together. An ISOLATED CP gets the same rule and is barely
    # touched, since it sits about one pitch from where its own path lands.
    "terminus_capture": True,
    # Minimum basin boundaries: visualization of closed cycles enclosing each
    # minimum CP in the connection topology. Show individual minimum or all at once.
    "show_basin_for_selected_minimum": False,
    "show_all_minimum_basins": False,
    # Maximum basin boundaries (basin v11): the same rule read from the other end,
    # so the same two switches exist for the hill side. Defaults are False on BOTH
    # sides -- a maximum's ring is drawn only when asked, and a snapshot that says
    # nothing says "off", which is why the old file's silence is not a mismatch.
    "show_basin_for_selected_maximum": False,
    "show_all_maximum_basins": False,
}

# Schema version of the JSON file written by the "Save Parameters" button.
# Bump on any breaking change to the layout below so loaders can reject
# stale files instead of silently misreading them.
#
# Version 1 covers the original layout (params / context / derived_metrics /
# selection). The "camera" section was added later as an *optional* section,
# so it did not warrant a version bump: files without it still load, and
# files with it are recognised by the presence of the key.
#
# Version 2 adds the excision ring controls. Adding a control is breaking on its
# own: load_params_from_dict requires every control key to be present, so a
# version 1 file would fail with a confusing "Missing parameter
# 'ring_samples'" KeyError. Bumping makes the rejection a clear version message.
#
# Version 3 adds the maximum-connection overlay control. Because the snapshots we
# diagnose WITH are version 2 files on disk, version 2 is admitted as a legacy
# version (see PARAM_LEGACY_SCHEMA_VERSIONS) and its absent controls are filled
# from DEFAULT_UI_STATE, so an existing observation still reproduces.
#
# Version 4 added the connector-refinement control. That step was reverted -- the
# refinement will be rebuilt differently -- so the control is gone from
# DEFAULT_UI_STATE, but snapshots saved while it existed are still on disk. v4 is
# admitted as legacy: the loader reads the keys it knows and ignores the one it
# no longer has, so an observation made with refinement on still opens and
# reproduces the pipeline it describes.
#
# Version 5 adds the ridge-refinement control (refine_ridge_paths), the rebuilt
# Step 6. Version 3 is the one that matters as legacy: every ethene3 and Pd_20K
# snapshot on disk is v3, and a control invented after them had exactly one value
# on screen -- its default, off -- so filling it is the faithful reading.
#
# Version 6 adds the ridge-probe control (probe_ridge_points). Version 5 is
# admitted as legacy for the same reason v3 was: a snapshot predating the control
# shows it off, which is its default.
#
# Version 7 adds the coarse structural curve (coarse_ridge_curves). It is a NEW key
# rather than a new meaning for refine_ridge_paths, and the reason is evidence, not
# tidiness: the ethene3 snapshots on disk were saved with refine_ridge_paths ON, and
# their recorded connection_stats reproduce the FIDELITY smoother exactly
# (n_refined 4, total_turn_after_deg 5987.902, max_refine_shift_deg 0.898 -- all four
# match to the last digit). Overloading the key would replay those snapshots with
# geometry that was never on screen, which is the one thing a snapshot must not do.
# Version 6 is admitted as legacy: it predates coarse_ridge_curves, so the control
# was off, which is its default, and the snapshot replays the line it recorded.
#
# Version 8 exports the ridge probe's OUTPUT (derived_metrics.ridge_probe_stats), not
# just its switch. No control key was invented, so the strict/legacy split is
# unaffected: metrics are outputs the loader never feeds back into the pipeline, and
# a v7 file simply has no probe block to compare. v7 is admitted as legacy because
# every snapshot on disk predates the export, and the absence of a measurement is not
# a measurement of zero -- diagnose_params reports "no comparable metric" for those
# files rather than a mismatch.
# Version 8 exports the ridge probe's OUTPUT (derived_metrics.ridge_probe_stats), not
# just its switch. No control key was invented, so the strict/legacy split is
# unaffected: metrics are outputs the loader never feeds back into the pipeline, and
# a v7 file simply has no probe block to compare. v7 is admitted as legacy because
# every snapshot on disk predates the export, and the absence of a measurement is not
# a measurement of zero -- diagnose_params reports "no comparable metric" for those
# files rather than a mismatch.
#
# Version 9 exports the structures behind the minimum-basin visualization:
# max_connections_list (the saddle->max arc table, arc geometry included),
# basin_centers_list, and the derived connection_graph / minimum_basins. Basin
# boundaries are cycles in that graph, so a saved observation is not verifiable
# without it -- toggles that render nothing were undiagnosable from the file.
# All four are METRICS, not controls: absent sections in an older file mean the
# feature did not exist, never a computed-empty. v8 is admitted as legacy for the
# same reason v7 was: every snapshot on disk predates the export.
#
# Version 10 exports the descent connector: min_connections_list (the saddle->min
# drain table -- the basin adjacency itself, arc geometry included) and its
# summary descent_connection_stats. basin v10 bounds a minimum's basin by the
# ascent arcs at the saddles that DRAIN into it, so a saved observation is not
# verifiable without the drainage: v9 recorded the arcs and rebuilt the ring from
# spatial proximity (the k=3 nearest saddles), which saturated on few-saddle
# systems and could not be diagnosed from the file. v9 is admitted as legacy: it
# has no descent table, so the loader's replay reports its basins as
# un-derivable-by-v10 rather than reconstructing them from the rule v10 rejected.
#
# Version 11 exports the maximum basin group: the ring/shell rows it draws
# (max_basin_arcs), which centres are on screen (active_max_basins), the shared
# occlusion classification (basin_group_stats, per polarity), and
# selected_maximum_id. v10 shipped the rings for minima only, so a saved
# observation of a hill's boundary was -- literally -- unrecordable: the file had
# the arcs it drew for the valley side and nothing for the hill side, while the
# screen showed both. v10 is admitted as legacy: its two new toggles are absent
# and fill to False, which is what the v10 screen did with them (nothing), and
# the metrics it lacks are absent because the feature did not exist, never
# computed-empty.
#
# Version 12 adds the arc-gradification controls (gradify_arcs, arc_gradify_nodes)
# and exports what they measured (derived_metrics.arc_gradify_stats). v11 is admitted
# as legacy: the switch did not exist, so it fills to False, and a v11 file replays
# exactly the gold great circles it recorded. The metric is absent from those files
# because the measurement was never made -- diagnose_params reports "no comparable
# metric" for them, not "0", which is the v8 rule for a probe that did not run.
#
# Version 13 adds the gradient-traced connectors (trace_connectors) and exports what
# they found (derived_metrics.trace_connector_stats): how many branches were traced,
# how many arrived, and how many DISAGREED with the ray the straight arc came from.
# v12 is admitted as legacy: the switch fills to False, so a v12 file replays the
# straight arcs it recorded. The metric is absent there because no trace was run.
#
# Version 14 adds the terminus capture (terminus_capture): a connector that arrives
# inside a CP's disc is finished on that CP's CENTRE, so a path absorbed by a
# cluster no longer ends on a spurious extremum inside it. It DEFAULTS TO True, so
# a legacy snapshot -- including every v13 file on disk, which was recorded with the
# path-terminates-where-the-integrator-stopped behavior -- is filled with False and
# replays the curves it recorded; see PARAM_LEGACY_DEFAULT_OVERRIDES. The switch is
# free to be on by default for a live session because the traced connector it changes
# is itself behind ``trace_connectors``, which is off by default.
#
# Version 15 replaces the ``gradify_arcs`` switch with the three-way
# ``arc_polish_mode`` selector ("none" / "gradify" / "quad_fit"), because the polish step
# grew a second method: the quadratic Bézier fit, the polynomial ladder's first rung, which
# places one control point per free span to maximise the arc-length-weighted mean of the
# field (``quad_fit_search_pitch`` is its search radius; ``arc_gradify_nodes`` is now the
# NODE BUDGET shared by both methods). The selector also makes "no polish" explicit,
# which a bare switch said by silence. v14 files fill ``arc_polish_mode`` from the legacy
# table ("none"), and :func:`load_params_from_dict` then migrates an ON
# ``bools.gradify_arcs`` to "gradify", so a v14 file with gradification on replays its
# bent arcs and one without replays its straight ones. The fit publishes what it
# integrated (derived_metrics.quad_fit_stats): spans fitted vs refused, the offsets, the
# gains, and the mean field along each path before and after.
PARAMS_SCHEMA_VERSION = 15

# Older schema versions that still load, with absent controls defaulted. A
# snapshot is a record of what was ON SCREEN, so refusing to read a file written
# before a control was invented destroys the evidence. The current version stays
# strict: a truncated CURRENT file is a bug to surface, not a gap to paper over.
PARAM_LEGACY_SCHEMA_VERSIONS: frozenset[int] = frozenset(
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
)

# Controls whose LEGACY fill is NOT the current default. Every absent control
# normally defaults to what the switch does today; a control added after a version
# is recorded must instead default to what it did THEN, or loading the snapshot
# silently changes the picture it is supposed to reproduce. ``terminus_capture``
# is True now and did not exist before v14, and the traced connectors it modifies
# were introduced in v13, so every file written before this change is replayed
# without it. ``arc_polish_mode`` fills to "none" for every version that predates
# it (2-14, where the equivalent of "no polish" was ``gradify_arcs`` off), and the
# migration in :func:`load_params_from_dict` reads that switch when the file has it,
# so a v14 file with gradification ON gets "gradify" and is replayed bent.
PARAM_LEGACY_DEFAULT_OVERRIDES: dict[str, dict[int, Any]] = {
    # v13 and earlier predate the control, so the honest fill is the OFF the screen
    # showed. v14 is the version that SHIPPED it, defaulting ON: a v14 record that
    # somehow omits the key asked for the capture, so it fills True. The map is keyed by
    # the version's own released default, never by today's.
    "terminus_capture": {v: False for v in range(2, 14)} | {14: True},
    "arc_polish_mode": {v: "none" for v in range(2, 15)},
}


def _legacy_default(key: str, version: Any) -> Any:
    """The value an absent control had in a file written at ``version``.

    Today's default is the answer for every control that has always shipped with
    that default; a control invented later is the exception, and
    :data:`PARAM_LEGACY_DEFAULT_OVERRIDES` names the exception. Keying the table by
    version is what lets a new switch ship ON without rewriting the meaning of every
    snapshot already on disk.

    Args:
        key: Control name, e.g. ``"terminus_capture"``.
        version: The file's ``schema_version``, already validated.

    Returns:
        The registered override for that version, else the current default from
        :data:`DEFAULT_UI_STATE`.
    """
    per_version = PARAM_LEGACY_DEFAULT_OVERRIDES.get(key)
    if per_version is not None and isinstance(version, int) and version in per_version:
        return per_version[version]
    return DEFAULT_UI_STATE[key]


# Every editable control, in DEFAULT_UI_STATE's canonical order.
PARAM_CONTROL_KEYS: tuple[str, ...] = tuple(DEFAULT_UI_STATE)

# Type coercion tables: the viewer state is duck-typed, so the export pins
# each control to the JSON type its widget produces (switch -> bool,
# slider -> float, select -> str). Every control key must appear in exactly
# one table; the module-level checks below enforce that at import time.
PARAM_BOOL_KEYS: frozenset[str] = frozenset(
    k for k, v in DEFAULT_UI_STATE.items() if isinstance(v, bool)
)
PARAM_FLOAT_KEYS: frozenset[str] = frozenset(
    k
    for k, v in DEFAULT_UI_STATE.items()
    if isinstance(v, (int, float)) and not isinstance(v, bool)
)
# The first string control with data-INDEPENDENT options: the basin Step 3 polish
# selector. Every other selector is data-dependent (its option list comes from the running
# dataset) and lives in PARAM_CONTEXT_STRING_KEYS, which the export merges into the same
# "strings" section. An enumerated control's values are known from the module alone, so
# the table below validates them; the loader rejects an out-of-range value instead of
# coercing it, and an absent key fills from PARAM_LEGACY_DEFAULT_OVERRIDES.
PARAM_STRING_KEYS: frozenset[str] = frozenset(
    k for k, v in DEFAULT_UI_STATE.items() if isinstance(v, str)
)
PARAM_ENUM_KEYS: dict[str, tuple[str, ...]] = {
    "arc_polish_mode": ARC_POLISH_MODES,
}
if set(PARAM_BOOL_KEYS | PARAM_FLOAT_KEYS | PARAM_STRING_KEYS) != set(
    PARAM_CONTROL_KEYS
):
    raise RuntimeError(
        "PARAM_*_KEYS tables must cover exactly the DEFAULT_UI_STATE controls"
    )

# Data-dependent string control, supplied at runtime (absent from
# DEFAULT_UI_STATE), exported under params["strings"].
PARAM_CONTEXT_STRING_KEYS: tuple[str, ...] = ("selected_field",)

# Derived metric snapshots copied verbatim into the export. These are
# pipeline outputs, not inputs: the loader returns them for diagnostics but
# they must never be fed back into state.update().
PARAM_METRIC_KEYS: tuple[str, ...] = (
    "geom_info",
    "morse_counts",
    "display_counts",
    "reduction_stats",
    "launch_stats",
    "connection_stats",
    # The ridge probe measures the ascent beside the connectors on screen, and
    # until now the ONLY record of that measurement was the pixels in the drawer.
    # Saving it makes a diagnostic observation citable: the switch was already in
    # the snapshot, but a switch says what was asked for, not what was found.
    "ridge_probe_stats",
    # The arc gradification's numbers: how many straight connectors the field let it
    # bend, how far the steps were, how many nodes were measured (``n_moved``) versus
    # inherited (``n_carried``), and the RESIDUAL -- the same criterion asked a second
    # time at the bent positions, i.e. the arc's error as a gradient path. A switch
    # says what was asked for; this says what the field answered, and it is the number
    # a later low-order-polynomial representation (method 3) will be scored against.
    "arc_gradify_stats",
    # The gradient-traced connectors' numbers: how many branches were traced, how
    # many arrived, how many stalled or ran out of budget, and -- the one that
    # matters -- how many arrived somewhere OTHER than the maximum the straight ray
    # predicted. A switch says what was asked for; this says whether the field
    # agreed with the straight line, which is the question the trace exists to ask.
    "trace_connector_stats",
    # The quadratic fit's numbers: how many connectors it reshaped, how many spans it
    # refused as not worth a bend, the control offsets it accepted, and the pair of
    # arc-length-weighted means (straight vs fitted) its objective was built from. The
    # ladder's first rung scored in the same normalisation gradify gates on, so the two
    # methods' summaries are comparable arc for arc -- which is how the next rung (a
    # cubic) will be judged against this one.
    "quad_fit_stats",
    # The saddle->max arc table and the CP table it is scored against. These are
    # the raw material of the minimum-basin boundaries: a saved observation that
    # shows basin cycles must carry the arcs, their geometry, and the minima they
    # enclose, or the file cannot be checked against the screen.
    "max_connections_list",
    "basin_centers_list",
    # Derived topology, serialized by basin_boundary.serialize_* (shared with
    # diagnose_params so the two files diff field-for-field): the graph as
    # saddle_id -> outgoing arcs, and each minimum's boundary cycle.
    "connection_graph",
    "minimum_basins",
    # basin v11: the hill side of the same rule. The viewer draws a maximum's
    # boundary ring, so the replay must produce it -- and it costs nothing, since
    # find_maximum_basins consumes the two tables stage_minimum_basins already
    # builds. Saved with them, a ring on a hill is citable from the file.
    "maximum_basins",
    # basin v10: the descent drain table (the adjacency) and its summary.
    # Absent in a v9-era file -- the feature did not exist, never computed-empty.
    "min_connections_list",
    "descent_connection_stats",
    # basin v11: what the two basin GROUPS drew and what the shared classifier
    # said about them. The arc lists are render truth (the screen shows exactly
    # what the gate warns about), basin_group_stats is the report; both are saved
    # so a ring drawn for a maximum is citable, not just visible.
    "min_basin_arcs",
    "max_basin_arcs",
    "active_min_basins",
    "active_max_basins",
    "basin_group_stats",
)

# The probe's summary, in the order ``make_ridge_probe_stats`` writes it. Exported so
# the drawer, the replay and the tests all key off one definition.
RIDGE_PROBE_STAT_KEYS: tuple[str, ...] = (
    "on",
    "n_connections",
    "n_knots",
    "n_points",
    "n_refused",
    "median_offset_pitch",
    "scope_basin_id",
    "scope_saddle_id",
)

# Per-connection probe fields the aggregator reads off a row. The viewer stamps these
# onto its arc-table rows; diagnose_params builds the same dicts headlessly.
_PROBE_ROW_KEYS = {
    "n_knots": "n_knots",
    "n_probe": "n_points",
    "probe_refused": "n_refused",
}


def make_ridge_probe_stats(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    on: bool,
    basin_id: str | None = None,
    saddle_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-connector ridge-probe rows into the saved summary.

    ONE implementation of the aggregation, deliberately: the viewer measures the
    probe while drawing and the headless replay measures it while reproducing, and
    a saved number is only worth diffing if both sides compute it identically. The
    two callers used to sum by hand, and the sum is exactly the kind of thing that
    drifts -- the viewer's init block had already dropped ``n_refused`` from its
    copy of this dict, so a file saved before any probe ran reported a different
    key set than one saved after.

    Args:
        probe_rows: One mapping per PROBED connector, carrying ``n_knots``,
            ``n_probe``, ``probe_refused`` and ``probe_offset_pitch`` (the field
            names the viewer's arc table uses). Rows are counted, not filtered:
            pass only the connectors the probe actually measured.
        on: Whether the probe ran, i.e. the toggle AND its display gates. False
            zeroes every counter; the scope ids are dropped with it, because a
            scope that measured nothing is not a fact about the field.
        basin_id: The selected maximum whose arcs were on screen, when the probe
            was selection-scoped. Recorded so a reader can tell 4 connectors on
            one basin from 4 connectors in the whole run.
        saddle_id: The selected saddle whose launch branches were probed, when it
            was the saddle that was selected. Mutually exclusive with basin_id,
            as in ``_emit_max_arcs``.

    Returns:
        A JSON-safe dict with exactly ``RIDGE_PROBE_STAT_KEYS``. The median is over
        the PER-CONNECTOR medians, matching ``np.median`` on both parities, so the
        viewer and this function agree bit-for-bit.
    """
    offsets = (
        [
            float(row["probe_offset_pitch"])
            for row in probe_rows
            if "probe_offset_pitch" in row
        ]
        if on
        else []
    )
    stats: dict[str, Any] = {
        "on": bool(on),
        "n_connections": len(probe_rows) if on else 0,
        "n_knots": sum(int(row.get("n_knots", 0)) for row in probe_rows) if on else 0,
        "n_points": sum(int(row.get("n_probe", 0)) for row in probe_rows) if on else 0,
        "n_refused": (
            sum(int(row.get("probe_refused", 0)) for row in probe_rows) if on else 0
        ),
        "median_offset_pitch": (float(statistics.median(offsets)) if offsets else 0.0),
        "scope_basin_id": str(basin_id) if (on and basin_id is not None) else None,
        "scope_saddle_id": str(saddle_id) if (on and saddle_id is not None) else None,
    }
    return {key: stats[key] for key in RIDGE_PROBE_STAT_KEYS}


# The arc gradification's published numbers. ``on`` is the switch; the rest are what
# the field answered. The three counts that must add up are the point: ``n_nodes`` is
# the layout's budget, ``n_moved`` is what the field measured, ``n_carried`` is what
# the rule inherited, and a reader who cannot tell the two apart is being sold
# interpolation as data. ``residual_median_pitch`` is the same criterion asked a SECOND
# time at the bent positions -- the arc's error as a gradient path, and the number a
# later low-order-polynomial representation (method 3) will be scored against.
ARC_GRADIFY_STAT_KEYS: tuple[str, ...] = (
    "on",
    "n_arcs",
    "n_gradified",
    "n_all_refused",
    "n_pinned",
    "n_degenerate",
    "n_nodes",
    "n_moved",
    "n_carried",
    "n_pinned_nodes",
    "median_offset_pitch",
    "max_offset_pitch",
    "residual_median_pitch",
    "residual_max_pitch",
    "n_nodes_target",
    "scope",
    "descent",
)


def make_arc_gradify_stats(
    stats: Mapping[str, Any] | None = None,
    *,
    on: bool = True,
) -> dict[str, Any]:
    """Normalise one arc-gradification summary to the published key set.

    The twin of :func:`make_ridge_probe_stats`, and for the same reason: the key set
    must not depend on whether the rule has run. The viewer initialises its metric
    through this before the first rebuild, and every value the gradifier computes is
    preserved (``stats`` is the dict ``ridge_refiner.gradify_connections`` returns).

    Args:
        stats: The gradifier's aggregate, or None when it has not run. Unknown keys
            are dropped, so the published shape is this function's, not the
            producer's.
        on: Whether the rule ran. False zeroes every counter and drops the scope,
            because a scope that measured nothing is not a fact about the field.

    Returns:
        A JSON-safe dict with exactly :data:`ARC_GRADIFY_STAT_KEYS`.
    """
    src = stats if (on and stats) else {}
    stats_out: dict[str, Any] = {
        "on": bool(on),
        "n_arcs": int(src.get("n_arcs", 0)),
        "n_gradified": int(src.get("n_gradified", 0)),
        "n_all_refused": int(src.get("n_all_refused", 0)),
        "n_pinned": int(src.get("n_pinned", 0)),
        "n_degenerate": int(src.get("n_degenerate", 0)),
        "n_nodes": int(src.get("n_nodes", 0)),
        "n_moved": int(src.get("n_moved", 0)),
        "n_carried": int(src.get("n_carried", 0)),
        "n_pinned_nodes": int(src.get("n_pinned_nodes", 0)),
        "median_offset_pitch": float(src.get("median_offset_pitch", 0.0)),
        "max_offset_pitch": float(src.get("max_offset_pitch", 0.0)),
        "residual_median_pitch": float(src.get("residual_median_pitch", 0.0)),
        "residual_max_pitch": float(src.get("residual_max_pitch", 0.0)),
        "n_nodes_target": int(src.get("n_nodes_target", 0)),
        "scope": src.get("scope") if on else None,
        # The descent table's own numbers, nested (the card reports them on a second
        # line: a basin boundary that did not bend must be tellable from one that bent
        # off-screen). Normalised through this function so the shape is the same.
        "descent": (
            make_arc_gradify_stats(src.get("descent"), on=bool(src.get("descent")))
            if on and src.get("descent")
            else None
        ),
    }
    return {key: stats_out[key] for key in ARC_GRADIFY_STAT_KEYS}


# The QUADRATIC FIT's published shape, the ladder's first rung, and for the same reason as
# ARC_GRADIFY_STAT_KEYS: the key set must not depend on whether the fit ran. The counts
# that must add up are ``n_quad_fitted + n_rejected_gain + n_pinned + n_degenerate ==
# n_arcs`` (an arc reports its WORST span, so no arc is counted twice) and
# ``n_spans_fitted <= n_spans``. ``mean_gain_fraction`` is the number to judge the method
# by -- how much of the field's robust span a bend bought, in the same normalisation
# gradify gates on -- and the f pair is the measurement the objective was built from,
# published so a claim about a ridge is checkable against the exported line.
QUAD_FIT_STAT_KEYS: tuple[str, ...] = (
    "on",
    "n_arcs",
    "n_quad_fitted",
    "n_rejected_gain",
    "n_pinned",
    "n_degenerate",
    "n_spans",
    "n_spans_fitted",
    "max_offset_pitch",
    "mean_gain_fraction",
    "mean_f_before",
    "mean_f_after",
    "n_evals",
    "scope",
    "descent",
)


def make_quad_fit_stats(
    stats: Mapping[str, Any] | None = None,
    *,
    on: bool = True,
) -> dict[str, Any]:
    """Normalise one quadratic-fit summary to the published key set.

    The twin of :func:`make_arc_gradify_stats`. ``stats`` is what
    :func:`ridge_refiner.fit_quad_bezier_connections` returns for ONE polarity (the
    ascent, normally); the descent table's summary rides in ``"descent"`` and is
    normalised through this same function so the two halves have one shape.

    Args:
        stats: The fit's aggregate, or None when it has not run. Unknown keys are
            dropped, so the published shape is this function's, not the producer's.
        on: Whether the fit ran. False zeroes every counter and drops the scope, because
            a scope that measured nothing is not a fact about the field.

    Returns:
        A JSON-safe dict with exactly :data:`QUAD_FIT_STAT_KEYS`.
    """
    src = stats if (on and stats) else {}
    stats_out: dict[str, Any] = {
        "on": bool(on),
        "n_arcs": int(src.get("n_arcs", 0)),
        "n_quad_fitted": int(src.get("n_quad_fitted", 0)),
        "n_rejected_gain": int(src.get("n_rejected_gain", 0)),
        "n_pinned": int(src.get("n_pinned", 0)),
        "n_degenerate": int(src.get("n_degenerate", 0)),
        "n_spans": int(src.get("n_spans", 0)),
        "n_spans_fitted": int(src.get("n_spans_fitted", 0)),
        "max_offset_pitch": float(src.get("max_offset_pitch", 0.0)),
        "mean_gain_fraction": float(src.get("mean_gain_fraction", 0.0)),
        "mean_f_before": float(src.get("mean_f_before", 0.0)),
        "mean_f_after": float(src.get("mean_f_after", 0.0)),
        "n_evals": int(src.get("n_evals", 0)),
        "scope": src.get("scope") if on else None,
        "descent": (
            make_quad_fit_stats(src.get("descent"), on=bool(src.get("descent")))
            if on and src.get("descent")
            else None
        ),
    }
    return {key: stats_out[key] for key in QUAD_FIT_STAT_KEYS}


# The gradient-traced connectors' published shape, and for the same reason as
# ARC_GRADIFY_STAT_KEYS: the key set must not depend on whether the rule ran. The
# counts that must add up are ``n_traced`` against the trace_status histogram, and
# ``n_ray_disagrees`` is the one an operator reads -- it is the field overruling the
# straight ray, which is the whole question a trace exists to ask. ``descent`` is
# nested because the card reports the basin side on its own line.
TRACE_CONNECTOR_STAT_KEYS: tuple[str, ...] = (
    "on",
    "n_branches",
    "n_traced",
    "n_trace_arrived",
    "n_trace_stalled",
    "n_trace_budget",
    "n_trace_no_start",
    "n_trace_unreachable",
    "n_trace_unconfirmed",
    "n_ray_disagrees",
    # The carried last leg: how many connectors ended with a straight run to a CP's
    # center, and the longest such run. The terminus capture (``terminus_capture``)
    # lands every arrival inside a cluster on the cluster's CENTRE, so these two
    # numbers are the size of what that trades away -- drawn, not measured.
    "n_terminus_captured",
    "max_terminus_leg_deg",
    "descent",
)


def make_trace_connector_stats(
    stats: Mapping[str, Any] | None = None,
    *,
    on: bool = True,
) -> dict[str, Any]:
    """Normalise one traced-connector summary to the published key set.

    The twin of :func:`make_arc_gradify_stats`. ``stats`` is what
    ``max_connection.summarize_connections`` returns for ONE polarity (the ascent,
    normally); the descent's summary rides in ``"descent"`` and is normalised through
    this same function so the two halves have one shape.

    Args:
        stats: The summary, or None when tracing has not run. Unknown keys are
            dropped, so the published shape is this function's, not the producer's.
        on: Whether tracing ran. False zeroes every counter and drops the scope,
            because a scope that measured nothing is not a fact about the field.

    Returns:
        A JSON-safe dict with exactly :data:`TRACE_CONNECTOR_STAT_KEYS`.
    """
    src = stats if (on and stats) else {}
    stats_out: dict[str, Any] = {
        "on": bool(on),
        "n_branches": int(src.get("n_branches", 0)),
        "n_traced": int(src.get("n_traced", 0)),
        "n_trace_arrived": int(src.get("n_trace_arrived", 0)),
        "n_trace_stalled": int(src.get("n_trace_stalled", 0)),
        "n_trace_budget": int(src.get("n_trace_budget", 0)),
        "n_trace_no_start": int(src.get("n_trace_no_start", 0)),
        "n_trace_unreachable": int(src.get("n_trace_unreachable", 0)),
        "n_trace_unconfirmed": int(src.get("n_trace_unconfirmed", 0)),
        "n_ray_disagrees": int(src.get("n_ray_disagrees", 0)),
        # Carried legs, from ``max_connection.summarize_connections``: a completion is
        # counted by the producer as ``n_completed`` and measured as
        # ``max_completed_leg_deg``; published under the terminus names because after
        # v14 an ARRIVAL inside a cluster's disc is the ordinary completion.
        "n_terminus_captured": int(src.get("n_completed", 0)),
        "max_terminus_leg_deg": float(src.get("max_completed_leg_deg", 0.0)),
        # The descent table's own numbers, nested (the card reports them on a second
        # line: a basin boundary that was traced off-screen must be tellable from
        # one that was not). Normalised through this function so the shape is same.
        "descent": (
            make_trace_connector_stats(src.get("descent"), on=bool(src.get("descent")))
            if on and src.get("descent")
            else None
        ),
    }
    return {key: stats_out[key] for key in TRACE_CONNECTOR_STAT_KEYS}


# Name of the optional top-level section holding the view camera.
PARAM_CAMERA_SECTION = "camera"


def default_params_path(plt_path: str, atom_num: int) -> str:
    """Return the default output path for a saved parameter snapshot.

    Args:
        plt_path: Path to the .plt file the viewer was launched with.
        atom_num: Target atom number (1-based).

    Returns:
        ``<plt_dir>/<plt_stem>_atom<atom_num>_params.json``.
    """
    abs_plt = os.path.abspath(plt_path)
    directory = os.path.dirname(abs_plt)
    stem = os.path.splitext(os.path.basename(abs_plt))[0]
    return os.path.join(directory, f"{stem}_atom{atom_num}_params.json")


def _coerce_bool(key: str, raw: Any) -> bool:
    """Coerce an exported toggle value, rejecting non-bool types.

    Args:
        key: Control key, for the error message.
        raw: Value read from the viewer state or a JSON payload.

    Returns:
        The value as a bool.

    Raises:
        TypeError: If the value is not a bool.
    """
    if not isinstance(raw, bool):
        raise TypeError(f"Parameter '{key}' must be a bool, got {type(raw).__name__}")
    return raw


def _coerce_float(key: str, raw: Any) -> float:
    """Coerce an exported slider value to float, rejecting bools.

    Args:
        key: Control key, for the error message.
        raw: Value read from the viewer state or a JSON payload.

    Returns:
        The value as a float (ints from JSON, e.g. num_contours, widen to
        float).

    Raises:
        TypeError: If the value is a bool or non-numeric.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"Parameter '{key}' must be a number, got {raw!r}")
    return float(raw)


def _coerce_str(key: str, raw: Any) -> str:
    """Coerce an exported selector value to str, rejecting non-strings.

    Args:
        key: Control key, for the error message.
        raw: Value read from the viewer state or a JSON payload.

    Returns:
        The value as a str.

    Raises:
        TypeError: If the value is not a str.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Parameter '{key}' must be a str, got {type(raw).__name__}")
    allowed = PARAM_ENUM_KEYS.get(key)
    if allowed is not None and raw not in allowed:
        raise ValueError(
            f"Parameter '{key}' must be one of {list(allowed)}, got {raw!r}"
        )
    return raw


def _migrate_arc_polish_mode(
    *, payload: dict[str, Any], result: dict[str, Any], strict: bool
) -> None:
    """Carry a legacy ``gradify_arcs`` switch into the v15 ``arc_polish_mode`` selector.

    v14 and earlier had one switch, off by default; v15 has three modes, default
    "gradify". An absent control fills to the value it had THEN (see
    :data:`PARAM_LEGACY_DEFAULT_OVERRIDES`), which is "none" -- faithful to the switch's
    default of off, so a v14 snapshot's straight arcs replay straight. That leaves the
    other half of the switch's meaning unsaid: a v14 file with gradification ON records an
    intent that fills to the wrong mode, and the replay quietly un-bends every connector
    the operator asked to bend. This reads the old key from the raw payload (it is no
    longer in :data:`PARAM_BOOL_KEYS`, so the coerced section cannot see it) and carries
    the intent across.

    A file that already spells out ``arc_polish_mode`` wins untouched: what the record says
    beats what an older key in the same file implies. A CURRENT (strict) file is never
    migrated -- at v15 there is no old key to read, and inventing one would let a
    hand-edited file overrule its own selector.

    Args:
        payload: The ``params`` section as loaded from disk.
        result: The flat control dict, modified in place.
        strict: True when the file is at :data:`PARAMS_SCHEMA_VERSION`.
    """
    if strict:
        return
    raw_bools = payload.get("bools")
    raw_strings = payload.get("strings")
    if not isinstance(raw_bools, dict) or not isinstance(raw_strings, dict):
        return
    if "arc_polish_mode" in raw_strings:
        return
    if raw_bools.get("gradify_arcs") is True:
        result["arc_polish_mode"] = "gradify"


def _selection_id(selected: Any, key: str = "id_label") -> str | None:
    """Extract a stable id from a selection dict for the export.

    Args:
        selected: The selection state value (a dict or None).
        key: Which key carries the id ("id_label" for CPs, "id" otherwise).

    Returns:
        The id string, or None when nothing is selected.
    """
    if not isinstance(selected, dict):
        return None
    value = selected.get(key)
    return value if isinstance(value, str) else None


def save_params_to_dict(
    state_like: Any,
    *,
    plt_path: str,
    atom_info: dict[str, Any],
    atom_num: int,
    camera: CameraState | None = None,
    camera_center: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Snapshot every editable viewer control plus context and metrics.

    The returned dict is JSON-serializable and fully describes the GUI state
    observed on screen: every toggle (bool), slider (float) and selector
    (str), the run context (absolute .plt path, atom, selected condensed
    field, discovered field list), the view camera, the derived pipeline
    metrics, and the active selection ids. Diagnostic scripts feed the file
    back through ``load_params_from_dict`` to reproduce an observation
    headlessly.

    Args:
        state_like: Object exposing the control/metric keys as attributes
            (the trame ``state`` object satisfies this).
        plt_path: .plt path the viewer was launched with; stored absolute.
        atom_info: Atom metadata dict from ``extract_atom_sphere_mesh``.
        atom_num: Target atom number (1-based).
        camera: Optional view-camera snapshot. When given it is written to
            the optional ``camera`` section so the zoom level and viewing
            direction travel with the parameters.
        camera_center: The world-frame nucleus the camera was captured
            around. Required when ``camera`` is given, so the camera can be
            re-based when it is restored against a different atom.

    Returns:
        Schema-versioned export dict with keys: schema_version, tool,
        timestamp_utc, params (bools/floats/strings), context, camera,
        derived_metrics, selection.

    Raises:
        KeyError: If a control key is missing from the state object.
        TypeError: If a control value has an unexpected type.
        ValueError: If a camera is given without its reference nucleus.
    """
    bools = {
        k: _coerce_bool(k, getattr(state_like, k)) for k in sorted(PARAM_BOOL_KEYS)
    }
    floats = {
        k: _coerce_float(k, getattr(state_like, k)) for k in sorted(PARAM_FLOAT_KEYS)
    }
    strings = {
        k: _coerce_str(k, getattr(state_like, k)) for k in sorted(PARAM_STRING_KEYS)
    }
    for key in PARAM_CONTEXT_STRING_KEYS:
        strings[key] = _coerce_str(key, getattr(state_like, key))

    context: dict[str, Any] = {
        "plt_path": os.path.abspath(plt_path),
        "atom_number": int(atom_num),
        "atom_name": str(atom_info.get("atom_name", "")),
        "atom_symbol": str(atom_info.get("atom_symbol", "")),
        "field_items": [
            {"title": str(f["title"]), "value": str(f["value"])}
            for f in atom_info.get("field_items", [])
        ],
    }

    metrics: dict[str, Any] = {
        key: json.loads(json.dumps(getattr(state_like, key)))
        for key in PARAM_METRIC_KEYS
        if getattr(state_like, key, None) is not None
    }

    selection: dict[str, str | None] = {
        "selected_cp_id": _selection_id(getattr(state_like, "selected_cp", None)),
        "selected_cluster_id": _selection_id(
            getattr(state_like, "selected_cluster", None), "id"
        ),
        "selected_catastrophe_id": _selection_id(
            getattr(state_like, "selected_catastrophe", None), "id"
        ),
        "selected_basin_center_id": _selection_id(
            getattr(state_like, "selected_basin_center", None), "id"
        ),
        # The arc overlay has two ends and the snapshot used to record one of them.
        # Naming a saddle is a first-class selection -- it is how a STALLED ascent
        # gets seen, since it has no maximum to select -- and the ridge probe scopes
        # itself to whatever is on screen, so a saddle-scoped measurement was
        # unreproducible from the file alone.
        "selected_saddle_id": (getattr(state_like, "selected_saddle_id", None) or None),
        # Basin visualization: the minimum selected for boundary display.
        "selected_minimum_id": (
            getattr(state_like, "selected_minimum_id", None) or None
        ),
        # basin v11: the pair. v10 recorded the valley selection and not the hill
        # one, so a snapshot could say which minimum's ring was on screen and never
        # which maximum's. The two are read by the same panel and gated by the same
        # rule; the file now names both.
        "selected_maximum_id": (
            getattr(state_like, "selected_maximum_id", None) or None
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": PARAMS_SCHEMA_VERSION,
        "tool": "gba_topology2.sphere_viewer",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {"bools": bools, "floats": floats, "strings": strings},
        "context": context,
        "derived_metrics": metrics,
        "selection": selection,
    }
    if camera is not None:
        if camera_center is None:
            raise ValueError(
                "A camera snapshot requires the nucleus it was captured around"
            )
        payload[PARAM_CAMERA_SECTION] = camera_section_to_dict(camera, camera_center)
    return payload


def load_params_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a saved parameter snapshot and prepare it for state.update().

    Checks ``schema_version``, requires every control key to be present, and
    coerces values to their widget types (bools strict, ints widened to
    float). The result's top level is safe to pass to ``state.update()``
    directly: it contains exactly the control keys plus ``selected_field``.
    The ``context``, ``derived_metrics``, ``selection`` and ``camera``
    sections are returned alongside for diagnostics; they are outputs/ids, not
    inputs, so callers must not feed them back into the pipeline state.

    Args:
        payload: Dict loaded from a file written by ``save_params_to_dict``
            (e.g. ``json.load(f)``).

    Returns:
        Dict with the flat control params, plus "context",
        "derived_metrics", "selection" and "camera" passthrough sections.

    Raises:
        ValueError: If schema_version is missing, unsupported, or a legacy
            version is missing a control that has no default.
        KeyError: If any required control key is absent at the current version.
        TypeError: If any value has the wrong type.
    """
    version = payload.get("schema_version")
    strict = version == PARAMS_SCHEMA_VERSION
    if not strict and version not in PARAM_LEGACY_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported params schema_version {version!r}; "
            f"expected {PARAMS_SCHEMA_VERSION} "
            f"(legacy: {sorted(PARAM_LEGACY_SCHEMA_VERSIONS)})"
        )
    params = payload.get("params")
    if params is None:
        raise KeyError("Params payload is missing the 'params' section")
    if not isinstance(params, dict):
        raise TypeError("Params payload section 'params' must be a dict")

    def _section(name: str, keys: list[str], coerce: Any) -> dict[str, Any]:
        section = params.get(name)
        if not isinstance(section, dict):
            raise KeyError(f"Params section '{name}' is missing")
        out: dict[str, Any] = {}
        for key in keys:
            if key not in section:
                # A snapshot predates the control (either from an older schema
                # version or an older file from the current version), so the
                # control had a known value: its default. Filling it keeps an
                # old observation reproducible instead of unreadable -- which is
                # why the fill is the default AS OF that version, not today's
                # (see :data:`PARAM_LEGACY_DEFAULT_OVERRIDES`).
                if key in DEFAULT_UI_STATE:
                    out[key] = _legacy_default(key, version)
                    continue
                raise ValueError(
                    f"Params section '{name}' is missing '{key}' and no default exists"
                )
            out[key] = coerce(key, section[key])
        return out

    bools = _section("bools", sorted(PARAM_BOOL_KEYS), _coerce_bool)
    floats = _section("floats", sorted(PARAM_FLOAT_KEYS), _coerce_float)
    strings = _section(
        "strings",
        sorted(PARAM_STRING_KEYS | set(PARAM_CONTEXT_STRING_KEYS)),
        _coerce_str,
    )

    result: dict[str, Any] = {**bools, **floats, **strings}
    _migrate_arc_polish_mode(payload=params, result=result, strict=strict)

    for section in ("context", "derived_metrics", "selection", PARAM_CAMERA_SECTION):
        value = payload.get(section)
        if isinstance(value, dict):
            result[section] = value
    return result
