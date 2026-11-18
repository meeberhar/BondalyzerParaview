"""Tests for the arc polish fields on serialized arc rows.

The quadratic fit publishes its aggregate (``quad_fit_stats``) and, per arc, the
geometry it drew (``quad_fit_controls``). Both live in memory only, which makes a
report like "MC13->MC4 is still straight" undiagnosable from a saved file: the
aggregate says 4 of 6 arcs were fitted and never says *which two* were refused, nor
what their spans were.

These tests pin the contract that closes that gap: every serialized arc row carries
the polish verdict, and the geometry the spheres are drawn from.

NOT covered here: ``sphere_viewer._project_basin_arcs``, which carries these fields
across the basin join. It is defined inside ``run_sphere_viewer`` and closes over
``state``, so it cannot be imported -- lifting it to module scope is the follow-up.
The projection reads the fields off the rows these tests cover, with ``.get()``
defaults, so a missing key degrades to "unpolished" rather than crashing.

Run: uv run pytest tests/test_arc_row_export.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gba_topology2.src.basin_boundary import min_connection_to_state_dict
from gba_topology2.src.max_connection import (
    QUAD_FIT_ROW_KEYS,
    MaxConnection,
    MinConnection,
    quad_fit_state_fields,
)
from gba_topology2.src.params_schema import (
    ARC_POLISH_MODES,
    DEFAULT_UI_STATE,
    PARAM_BOOL_KEYS,
    PARAM_CONTEXT_STRING_KEYS,
    PARAM_FLOAT_KEYS,
    PARAM_STRING_KEYS,
    PARAMS_SCHEMA_VERSION,
    load_params_from_dict,
    save_params_to_dict,
)

# The fields the operator's question is answered with. ``quad_fit_status`` says WHY an
# arc is straight, ``quad_fit_spans``/``quad_fit_fitted`` say how much of it was even
# offered to the fit, and ``quad_fit_controls`` is the black-sphere geometry.
QUAD_ROW_KEYS = (
    "arc_polish_mode",
    "quad_fit_status",
    "quad_fitted",
    "quad_fit_spans",
    "quad_fit_fitted",
    "quad_fit_nodes",
    "quad_fit_max_offset_pitch",
    "quad_fit_gain_fraction",
    "quad_fit_controls",
)


def _make_max(**overrides: Any) -> MaxConnection:
    """A minimal ascent record: 3 points (2 spans), the smallest polishable arc."""
    base: dict[str, Any] = {
        "saddle_id": 22,
        "branch_index": 1,
        "max_id": 11,
        "status": "converged",
        "direction_mode": "auto",
        "cross_track_deg": 0.0,
        "along_track_deg": 0.0,
        "separation_deg": 0.0,
        "mutual": True,
        "verified": True,
        "arc_points": [
            [-0.0125, 0.0, 0.4125],
            [0.0, 0.2917, 0.2917],
            [0.2917, 0.2917, 0.0],
        ],
        "saddle_value": -0.0002,
        "max_value": 0.999,
        "quad_fit_status": "fit",
        "quad_fitted": True,
        "quad_fit_spans": 2,
        "quad_fit_fitted": 2,
        "quad_fit_nodes": 3,
        "quad_fit_max_offset_pitch": 0.0482,
        "quad_fit_gain_fraction": 0.0194,
        "quad_fit_controls": [[0.0, 0.1, 0.2], [0.1, 0.2, 0.0]],
    }
    base.update(overrides)
    return MaxConnection(**base)


def _make_min(**overrides: Any) -> MinConnection:
    """A minimal descent record, refused for length."""
    base: dict[str, Any] = {
        "saddle_id": 22,
        "branch_index": 2,
        "min_id": 33,
        "status": "converged",
        "direction_mode": "auto",
        "cross_track_deg": 0.0,
        "along_track_deg": 0.0,
        "separation_deg": 0.0,
        "verified": True,
        "arc_points": [[0.0, 0.0, -0.4127], [0.0, 0.29, -0.29]],
        "saddle_value": -0.0002,
        "min_value": -0.999,
        "quad_fit_status": "skip_short",
        "quad_fitted": False,
        "quad_fit_spans": 1,
        "quad_fit_fitted": 0,
        "quad_fit_nodes": 0,
        "quad_fit_max_offset_pitch": 0.0,
        "quad_fit_gain_fraction": 0.0,
        "quad_fit_controls": [],
    }
    base.update(overrides)
    return MinConnection(**base)


def _ascent_row(**overrides: Any) -> dict[str, Any]:
    """An arc row in the shape ``update_topology`` publishes it."""
    row: dict[str, Any] = {
        "max_id": 11,
        "saddle_id": 22,
        "arc_points": [[0.0, 0.0, 0.4127], [0.0, 0.29, 0.29]],
        "blocker_id": 11,
    }
    row.update(quad_fit_state_fields(_make_max(), polish_mode="quad_fit"))
    row.update(overrides)
    return row


def _state(**overrides: Any) -> SimpleNamespace:
    """A trame-state stand-in: every control at a valid value, plus the given rows.

    Built from the declared key sets, not from ``DEFAULT_UI_STATE`` alone, because the
    export also reads the context keys (``selected_field`` and friends) that the UI
    default dict does not carry. A stand-in that only covers half the keys would pass
    while the real export path went untested.
    """
    values: dict[str, Any] = dict(DEFAULT_UI_STATE)
    for key in PARAM_BOOL_KEYS:
        values.setdefault(key, False)
    for key in PARAM_FLOAT_KEYS:
        values.setdefault(key, 0.0)
    for key in set(PARAM_STRING_KEYS) | set(PARAM_CONTEXT_STRING_KEYS):
        values.setdefault(key, "")
    values["arc_polish_mode"] = "quad_fit"
    values["max_connections_list"] = [_ascent_row()]
    values["min_connections_list"] = [
        min_connection_to_state_dict(_make_min(), polish_mode="quad_fit")
    ]
    values.update(overrides)
    return SimpleNamespace(**values)


def _payload(**overrides: Any) -> dict[str, Any]:
    return save_params_to_dict(
        _state(**overrides),
        plt_path="/tmp/ethene3.plt",
        atom_info={"atom_name": "atom1", "atom_symbol": "C", "field_items": []},
        atom_num=1,
    )


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(ARC_POLISH_MODES))
def test_quad_fit_state_fields_names_the_mode_that_produced_the_row(mode: str) -> None:
    """``arc_polish_mode`` is the run's, not the arc's, so it is a required argument.

    The basin rows are joined to the arc rows by (saddle, branch); with the mode on the
    row, a basin arc is attributable to the method that bent it rather than to whichever
    switch the operator has set now.
    """
    assert (
        quad_fit_state_fields(_make_max(), polish_mode=mode)["arc_polish_mode"] == mode
    )


def test_quad_fit_state_fields_is_exactly_the_documented_family() -> None:
    """The family is one tuple, so the two tables cannot drift apart key by key."""
    fields = quad_fit_state_fields(_make_max(), polish_mode="quad_fit")
    assert set(fields) == set(QUAD_FIT_ROW_KEYS)
    assert set(fields) == set(QUAD_ROW_KEYS)


def test_quad_fit_state_fields_is_json_safe() -> None:
    """Exported rows go straight to ``json.dumps``: numpy scalars would raise."""
    record = _make_max(
        arc_points=[np.array(p, dtype=np.float64) for p in _make_max().arc_points],
        quad_fit_controls=[np.array(p, dtype=np.float64) for p in [[0.0, 0.1, 0.2]]],
    )
    fields = quad_fit_state_fields(record, polish_mode="quad_fit")
    assert json.loads(json.dumps(fields)) == fields
    assert all(isinstance(c, list) for c in fields["quad_fit_controls"])


def test_quad_fit_state_fields_types_are_stable() -> None:
    """str/bool/int/float/list on every path, including the empty-controls one."""
    for record in (_make_max(), _make_min()):
        fields = quad_fit_state_fields(record, polish_mode="none")
        assert isinstance(fields["quad_fit_status"], str)
        assert isinstance(fields["quad_fitted"], bool)
        assert isinstance(fields["quad_fit_spans"], int)
        assert isinstance(fields["quad_fit_fitted"], int)
        assert isinstance(fields["quad_fit_nodes"], int)
        assert isinstance(fields["quad_fit_max_offset_pitch"], float)
        assert isinstance(fields["quad_fit_gain_fraction"], float)
        assert isinstance(fields["quad_fit_controls"], list)


def test_quad_fit_state_fields_carries_the_spheres_verbatim() -> None:
    """The export is only useful if it is the same array the scene uses.

    ``_emit_max_arcs`` draws one sphere per entry of ``record.quad_fit_controls``; a
    hand audit of a saved file must measure the curve the viewer actually showed.
    """
    record = _make_max()
    fields = quad_fit_state_fields(record, polish_mode="quad_fit")
    assert fields["quad_fit_controls"] == [list(p) for p in record.quad_fit_controls]
    assert len(fields["quad_fit_controls"]) == 2


# ---------------------------------------------------------------------------
# The descent table
# ---------------------------------------------------------------------------


def test_descent_rows_carry_the_quad_block() -> None:
    """The gap this closes: descent rows carried the GRADIFY block and not this one."""
    row = min_connection_to_state_dict(_make_min(), polish_mode="quad_fit")
    for key in QUAD_ROW_KEYS:
        assert key in row, f"descent row is missing {key}"
    assert row["quad_fit_status"] == "skip_short"
    assert row["quad_fit_controls"] == []
    assert row["quad_fit_max_offset_pitch"] == 0.0


def test_descent_rows_default_to_unpolished() -> None:
    """A row serialized with no mode given says so, rather than omitting the key.

    A silent omission is indistinguishable from a pre-polish file at read time.
    """
    assert min_connection_to_state_dict(_make_min())["arc_polish_mode"] == "none"


# ---------------------------------------------------------------------------
# The whole export path
# ---------------------------------------------------------------------------


def test_payload_is_json_and_both_tables_carry_the_block() -> None:
    """The point of the unit: a saved file can answer the question offline."""
    payload = _payload()
    text = json.dumps(payload)

    metrics = json.loads(text)["derived_metrics"]
    for key in ("max_connections_list", "min_connections_list"):
        rows = metrics[key]
        assert rows, f"{key} is empty in the payload"
        for row in rows:
            for field in QUAD_ROW_KEYS:
                assert field in row, f"{key} row is missing {field}"
            assert row["arc_polish_mode"] == "quad_fit"


def test_quad_fields_survive_the_round_trip() -> None:
    """``derived_metrics`` is exported as-is and re-read as-is."""
    loaded = load_params_from_dict(_payload())
    rows = loaded["derived_metrics"]["max_connections_list"]
    assert rows[0]["quad_fit_status"] == "fit"
    assert rows[0]["quad_fitted"] is True
    assert rows[0]["quad_fit_spans"] == 2
    assert rows[0]["quad_fit_fitted"] == 2
    assert rows[0]["quad_fit_max_offset_pitch"] == pytest.approx(0.0482)
    assert rows[0]["quad_fit_gain_fraction"] == pytest.approx(0.0194)
    assert rows[0]["quad_fit_controls"] == [[0.0, 0.1, 0.2], [0.1, 0.2, 0.0]]


def test_additive_keys_need_no_schema_bump() -> None:
    """A file whose rows lack the polish keys must load at today's version.

    ``schema_version`` gates CONTROL keys. The polish fields are per-row outputs read
    with ``.get()`` defaults, so bumping the version would reject the operator's
    existing exports to announce nothing -- and v15 files are already on disk.
    """
    payload = _payload()
    assert payload["schema_version"] == PARAMS_SCHEMA_VERSION

    legacy_row = _ascent_row()
    for key in QUAD_ROW_KEYS:
        legacy_row.pop(key, None)
    payload["derived_metrics"]["max_connections_list"] = [legacy_row]

    loaded = load_params_from_dict(payload)
    row = loaded["derived_metrics"]["max_connections_list"][0]
    for key in QUAD_ROW_KEYS:
        assert key not in row, (
            "the loader must not invent control keys into output rows"
        )
