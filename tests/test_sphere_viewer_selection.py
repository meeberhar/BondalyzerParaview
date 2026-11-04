"""Unit tests for the viewer's selection-to-table-row resolvers.

Clicking a glyph in the 3D scene selects a critical point, a micro-cluster, or a
basin center. The operator's question is "which entity is this?", so the matching
sidebar row must light up. The three resolvers under test map a selection to the
row id in each table, across the id spaces the pipeline uses:

- a RAW CP carries ``vertex_id``; its basin center is ``MAX_<v>`` / ``MIN_<v>``
  and its cluster row is found by scanning ``members[].vertex_id``;
- an EFFECTIVE CP carries ``id_label`` (its basin center id) and ``origin_id``
  (the unit row it was reduced from).

These are pure functions of the flattened UI dicts, so they are testable without
a browser, a render window, or trame state.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest

import gba_topology2.src.sphere_viewer as sv
from gba_topology2.src.sphere_viewer import (
    basin_center_id_for_cluster,
    basin_center_id_for_cp,
    cluster_id_for_cp,
    saddle_seed_id_for_cluster,
    saddle_seed_id_for_cp,
)

SOURCE = pathlib.Path(sv.__file__)


def _raw_cp(vertex_id: int, cp_type: str, hidden: bool = False) -> dict[str, Any]:
    prefix = {"maximum": "MAX", "minimum": "MIN", "saddle": "SAD"}[cp_type]
    return {
        "vertex_id": vertex_id,
        "type": cp_type,
        "id_label": f"{prefix}_{vertex_id}",
        "is_effective": False,
        "is_hidden": hidden,
        "position": [0.0, 0.0, 1.0],
    }


def _eff_cp(id_label: str, origin_id: str, cp_type: str = "maximum") -> dict[str, Any]:
    return {
        "vertex_id": -1,
        "type": f"E-{cp_type[:3].upper()}",
        "id_label": id_label,
        "origin_id": origin_id,
        "is_effective": True,
        "is_hidden": False,
        "position": [0.0, 0.0, 1.0],
    }


def _basin(center_id: str, polarity: str) -> dict[str, Any]:
    return {"id": center_id, "polarity": polarity, "position": [0.0, 0.0, 1.0]}


def _cluster(cid: str, members: list[tuple[int, str]]) -> dict[str, Any]:
    return {
        "id": cid,
        "members": [
            {
                "vertex_id": v,
                "type": t,
                "id_label": f"{'MAX' if t == 'maximum' else 'MIN' if t == 'minimum' else 'SAD'}_{v}",
                "position": [0.0, 0.0, 1.0],
            }
            for v, t in members
        ],
    }


# --- basin_center_id_for_cp ---------------------------------------------------


def test_raw_maximum_resolves_to_its_max_basin_center() -> None:
    """A raw maximum's basin center is MAX_<vertex_id>."""
    bcs = [_basin("MAX_42", "maximum"), _basin("MIN_7", "minimum")]
    assert basin_center_id_for_cp(_raw_cp(42, "maximum"), bcs) == "MAX_42"
    assert basin_center_id_for_cp(_raw_cp(7, "minimum"), bcs) == "MIN_7"


def test_raw_saddle_has_no_basin_center() -> None:
    """Saddles are not extrema; extract_basin_centers emits none, so None."""
    bcs = [_basin("MAX_42", "maximum")]
    assert basin_center_id_for_cp(_raw_cp(99, "saddle"), bcs) is None


def test_hidden_raw_cp_has_no_basin_center() -> None:
    """A CP collapsed into an effective entity is not drawn as a basin center."""
    bcs = [_basin("MAX_42", "maximum")]
    assert basin_center_id_for_cp(_raw_cp(42, "maximum", hidden=True), bcs) is None


def test_effective_cp_resolves_by_id_label() -> None:
    """An E-MAX's basin center id is its own id_label."""
    bcs = [_basin("E-MAX_MC12", "maximum"), _basin("MAX_1", "maximum")]
    cp = _eff_cp("E-MAX_MC12", "MC12", "maximum")
    assert basin_center_id_for_cp(cp, bcs) == "E-MAX_MC12"


def test_effective_cp_with_no_visible_center_is_none() -> None:
    """An E-SAD has no basin center at all."""
    bcs = [_basin("E-MAX_MC12", "maximum")]
    cp = _eff_cp("E-SAD_MC7", "MC7", "saddle")
    assert basin_center_id_for_cp(cp, bcs) is None


def test_missing_basin_center_is_none_not_keyerror() -> None:
    """A CP whose extremum is absent from the table resolves to None."""
    assert basin_center_id_for_cp(_raw_cp(999, "maximum"), []) is None


@pytest.mark.parametrize("cp", [None, {}, "not-a-dict", 42])
def test_falsy_cp_item_is_none(cp: Any) -> None:
    assert basin_center_id_for_cp(cp, [_basin("MAX_1", "maximum")]) is None


# --- cluster_id_for_cp --------------------------------------------------------


def test_raw_cp_finds_its_owning_unit_by_vertex() -> None:
    """A raw CP's row is the unit whose members include its vertex."""
    clusters = [
        _cluster("MC3", [(10, "maximum"), (11, "saddle")]),
        _cluster("MC9", [(20, "minimum")]),
    ]
    assert cluster_id_for_cp(_raw_cp(11, "saddle"), clusters) == "MC3"
    assert cluster_id_for_cp(_raw_cp(20, "minimum"), clusters) == "MC9"


def test_raw_cp_in_no_unit_is_none() -> None:
    clusters = [_cluster("MC3", [(10, "maximum")])]
    assert cluster_id_for_cp(_raw_cp(77, "maximum"), clusters) is None


def test_effective_cp_finds_its_origin_unit() -> None:
    """An effective CP's row is the unit named by origin_id."""
    clusters = [_cluster("MS1", [(10, "maximum")]), _cluster("MC4", [(12, "saddle")])]
    assert cluster_id_for_cp(_eff_cp("E-SAD_MS1", "MS1", "saddle"), clusters) == "MS1"


def test_catastrophe_origin_is_not_a_cluster_row() -> None:
    """A CAT-derived effective CP lives in the catastrophe table, not here."""
    clusters = [_cluster("MC3", [(10, "maximum")])]
    cp = _eff_cp("E-SAD_CAT_2", "CAT_2", "saddle")
    assert cluster_id_for_cp(cp, clusters) is None


@pytest.mark.parametrize("cp", [None, {}, "x"])
def test_falsy_cp_cluster_is_none(cp: Any) -> None:
    assert cluster_id_for_cp(cp, [_cluster("MC3", [(10, "maximum")])]) is None


# --- basin_center_id_for_cluster ----------------------------------------------


def test_cluster_reduced_to_effective_max_resolves_by_id() -> None:
    """A chi=+1 set reduced to E-MAX_<unit_id> is found by prefixing the id."""
    bcs = [_basin("E-MAX_MC12", "maximum"), _basin("MAX_1", "maximum")]
    assert (
        basin_center_id_for_cluster(_cluster("MC12", [(5, "maximum")]), bcs)
        == "E-MAX_MC12"
    )


def test_cluster_reduced_to_effective_min_resolves() -> None:
    bcs = [_basin("E-MIN_MS3", "minimum")]
    assert (
        basin_center_id_for_cluster(_cluster("MS3", [(5, "minimum")]), bcs)
        == "E-MIN_MS3"
    )


def test_unreduced_cluster_resolves_by_member_vertex() -> None:
    """A cluster left alone contributes its raw MAX_<v> basin center."""
    bcs = [_basin("MAX_42", "maximum"), _basin("MIN_7", "minimum")]
    cl = _cluster("MC3", [(42, "maximum"), (43, "saddle")])
    assert basin_center_id_for_cluster(cl, bcs) == "MAX_42"


def test_saddle_cluster_resolves_to_none() -> None:
    """A pure saddle set is not an extremum; clicking it lights no basin row."""
    bcs = [_basin("MAX_42", "maximum")]
    cl = _cluster("MC3", [(10, "saddle"), (11, "saddle")])
    assert basin_center_id_for_cluster(cl, bcs) is None


def test_cluster_prefers_the_effective_over_a_member() -> None:
    """The E-MAX_<id> form is checked first: it is the visible glyph."""
    bcs = [_basin("MAX_42", "maximum"), _basin("E-MAX_MC3", "maximum")]
    cl = _cluster("MC3", [(42, "maximum")])
    assert basin_center_id_for_cluster(cl, bcs) == "E-MAX_MC3"


@pytest.mark.parametrize("cl", [None, {}, "x"])
def test_falsy_cluster_is_none(cl: Any) -> None:
    assert basin_center_id_for_cluster(cl, [_basin("MAX_1", "maximum")]) is None


# --- structural wiring: the resolvers are called and the rows are bound -------


def test_select_cp_drives_basin_center_and_cluster_highlight() -> None:
    """A glyph click must light the extremum row and the cluster row.

    The reactive callbacks are not callable headless, so the wiring is asserted
    structurally: select_cp's body calls basin_center_id_for_cp,
    select_basin_center, and sets highlighted_cluster_id.
    """
    src = SOURCE.read_text(encoding="utf-8")
    body = _function_body(src, "def select_cp(")
    assert "basin_center_id_for_cp(" in body
    assert "select_basin_center(" in body
    assert "state.highlighted_cluster_id = cluster_id_for_cp(" in body


def test_select_cluster_drives_basin_center_highlight() -> None:
    """Clicking a cluster centroid (the picker's first test) identifies its max."""
    src = SOURCE.read_text(encoding="utf-8")
    body = _function_body(src, "def select_cluster(")
    assert "basin_center_id_for_cluster(" in body
    assert "select_basin_center(" in body
    assert "state.highlighted_cluster_id =" in body


def test_three_tables_bind_their_rows_to_selection_state() -> None:
    """Each table's row class is a dynamic :class binding, not a static string.

    Vue strips <style> from templates, so the highlight must be a Vuetify
    utility class bound with :class. Trame emits a dynamic binding only for a
    tuple value, so each use site is `classes=(EXPR,)`.
    """
    src = SOURCE.read_text(encoding="utf-8")
    for name in (
        "CLUSTER_ROW_CLASSES",
        "EFFECTIVE_CP_ROW_CLASSES",
        "BASIN_CENTER_ROW_CLASSES",
    ):
        assert re.search(rf"^{name} = \(", src, re.MULTILINE), f"{name} undefined"
        assert f"classes=({name},)" in src, f"{name} not bound with a tuple"


def test_row_expressions_reference_the_right_state_key() -> None:
    """The cluster row keys on highlighted_cluster_id; the others on selection."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "cl.id === highlighted_cluster_id" in src
    assert "selected_cp.id_label === ecp.id_label" in src
    assert "selected_basin_center.id === bc.id" in src


def test_highlighted_cluster_id_is_initialised_and_reset() -> None:
    """The key must exist before the first render and clear on reset."""
    src = SOURCE.read_text(encoding="utf-8")
    assert src.count("state.highlighted_cluster_id = None") >= 2, (
        "initialised at startup and cleared in reset_defaults"
    )


def _function_body(src: str, header: str) -> str:
    """Extract a nested function's body by indentation, for structural asserts."""
    start = src.index(header)
    lines = src[start:].splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Saddle -> launch seed. Selecting a saddle must show the ascent branches that
# leave it, which is the only view of a gradient climb that ends on ground the
# topology does not call a maximum. The join is the CP's id_label, which IS the
# seed's saddle_id (build_launch_requests names raw saddles SAD_<vertex> and
# effective ones by their id_label, E-SAD_<unit>).
# ---------------------------------------------------------------------------


SEED_IDS = {"SAD_8487", "SAD_1594", "E-SAD_MC18", "E-SAD_MC1"}


def test_saddle_seed_id_resolves_raw_saddle_by_id_label() -> None:
    assert saddle_seed_id_for_cp(_raw_cp(8487, "saddle"), SEED_IDS) == "SAD_8487"


def test_saddle_seed_id_resolves_effective_saddle() -> None:
    cp = {
        "vertex_id": -1,
        "type": "E-SAD",
        "id_label": "E-SAD_MC18",
        "is_effective": True,
        "is_hidden": False,
        "position": [0.0, 0.0, 1.0],
    }
    assert saddle_seed_id_for_cp(cp, SEED_IDS) == "E-SAD_MC18"


def test_saddle_seed_id_is_none_for_extrema_and_unseeded_saddles() -> None:
    """A maximum has no ascent branches; a hidden/absent saddle has no seed."""
    assert saddle_seed_id_for_cp(_raw_cp(12, "maximum"), SEED_IDS) is None
    assert saddle_seed_id_for_cp(_raw_cp(999, "saddle"), SEED_IDS) is None
    assert saddle_seed_id_for_cp(_raw_cp(8487, "saddle", hidden=True), SEED_IDS) is None
    assert saddle_seed_id_for_cp(None, SEED_IDS) is None


def test_saddle_seed_id_for_cluster_matches_effective_saddle_unit() -> None:
    """The picker tests cluster centroids first, so the corridor row must resolve."""
    cluster = {"id": "MC18", "members": [{"vertex_id": 8487, "type": "saddle"}]}
    assert saddle_seed_id_for_cluster(cluster, SEED_IDS) == "E-SAD_MC18"


def test_saddle_seed_id_for_cluster_falls_back_to_a_raw_member() -> None:
    """An un-reduced saddle cluster: E-SAD_<id> is absent, SAD_<vertex> is seeded."""
    cluster = {"id": "MC77", "members": [{"vertex_id": 8487, "type": "saddle"}]}
    assert saddle_seed_id_for_cluster(cluster, SEED_IDS) == "SAD_8487"


def test_saddle_seed_id_for_cluster_skips_non_saddle_members() -> None:
    """A maximum's cluster must not resolve to a saddle seed."""
    cluster = {
        "id": "MC9",
        "members": [
            {"vertex_id": 1, "type": "maximum"},
            {"vertex_id": 2, "type": "minimum"},
        ],
    }
    assert saddle_seed_id_for_cluster(cluster, SEED_IDS) is None
    assert saddle_seed_id_for_cluster(None, SEED_IDS) is None
