"""Every analysis control must be registered as a recompute trigger.

The viewer recomputes the pipeline through one handler, ``on_param_change``,
which is subscribed to an explicit list of state keys with ``@state.change(...)``
and calls ``update_field()`` -> ``update_topology()``. The list is hand-maintained,
so a control added to ``DEFAULT_UI_STATE`` and read inside the pipeline can be
wired to the UI, written to snapshots, and consumed by the analysis, and still be
dead when toggled: no key change, no event, no recompute. The operator sees the
old geometry until an UNRELATED control fires the handler, which then picks up the
pending value -- so the toggle appears to work only in combination with another
switch. That is how ``coarse_ridge_curves`` (schema v7, "Coarse Structural Curves
(Cyan Ridges)") shipped: the pipeline reads it at the refinement call, and the key
was missing from the subscription list.

The registry is the single source of truth for the fix, so the guard is mechanical:
every control the viewer exposes must appear in the subscription list. Parsed from
source with ``ast`` because the handler lives in the ``run_sphere_viewer`` closure
and the list is decorator data -- importing and launching the app to read a
decorator would need a render window and a browser.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Final

from gba_topology2.src.params_schema import DEFAULT_UI_STATE
from gba_topology2.src.sphere_viewer import PARAM_CONTEXT_STRING_KEYS

SOURCE: Final[pathlib.Path] = (
    pathlib.Path(__file__).resolve().parent.parent
    / "gba_topology2"
    / "src"
    / "sphere_viewer.py"
)

#: Runtime string that ``on_param_change`` calls when a watched key changes.
RECOMPUTE_CALL: Final[str] = "update_field"


def _state_change_keys() -> tuple[list[str], str]:
    """Read the ``@state.change(...)`` subscription lists out of the viewer source.

    Returns:
        Tuple ``(keys, handler_name)`` of the first ``state.change`` decorator
        found and the function it decorates.

    Raises:
        AssertionError: If the viewer has no ``@state.change`` subscription at all.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "change"
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "state"
            ):
                keys = [
                    arg.value
                    for arg in dec.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                return keys, node.name
    raise AssertionError("no @state.change(...) subscription found in sphere_viewer.py")


def _calls_recompute(handler_name: str) -> bool:
    """Return True when the subscribed handler calls the recompute entry point.

    Args:
        handler_name: Name of the function ``@state.change`` decorates.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == handler_name:
            body_nodes: list[ast.AST] = list(ast.walk(node))
            return any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == RECOMPUTE_CALL
                for call in body_nodes
            )
    return False


class TestParamTriggerRegistry:
    """The subscription list covers every control the viewer exposes."""

    def test_every_control_is_a_recompute_trigger(self) -> None:
        """A control that changes the analysis must fire the handler.

        This is the regression guard for the ``coarse_ridge_curves`` symptom: the
        switch, its schema default, its snapshot key and its pipeline consumer all
        existed; only the subscription was missing, so toggling it did nothing
        until an unrelated toggle forced a recompute.
        """
        keys, handler = _state_change_keys()
        watched = set(keys)
        unregistered = sorted(set(DEFAULT_UI_STATE) - watched)
        assert not unregistered, (
            f"{handler} is not subscribed to these viewer controls, so toggling "
            f"them cannot recompute the pipeline: {unregistered}"
        )

    def test_no_duplicate_keys_in_subscription(self) -> None:
        """A key listed twice is a merge artifact, not a second subscription."""
        keys, _handler = _state_change_keys()
        assert len(keys) == len(set(keys)), "duplicate key in @state.change(...)"

    def test_subscription_keys_are_known_state_names(self) -> None:
        """Every watched key is a control or a declared context string.

        Guards the other direction: a typo'd key subscribes to nothing and the
        control it was meant to cover stays dead.
        """
        keys, _handler = _state_change_keys()
        known = set(DEFAULT_UI_STATE) | set(PARAM_CONTEXT_STRING_KEYS)
        unknown = sorted(set(keys) - known)
        assert not unknown, f"@state.change(...) watches unknown state keys: {unknown}"

    def test_handler_calls_the_recompute_entry_point(self) -> None:
        """The subscribed handler is the one that rebuilds the scene.

        Registering the key on a handler that never recomputes would satisfy the
        key-list check and still leave the toggle dead, which is the failure mode
        this file exists to catch.
        """
        _keys, handler = _state_change_keys()
        assert _calls_recompute(handler), (
            f"{handler} does not call {RECOMPUTE_CALL}(), so a change to a "
            "watched key cannot recompute the analysis"
        )

    def test_ridge_cards_share_one_subscription(self) -> None:
        """The three Step 3 ridge switches are watched side by side.

        ``refine_ridge_paths``, ``coarse_ridge_curves`` and ``probe_ridge_points``
        are one feature family reading one call in ``update_topology``; the card
        is only predictable if all three behave identically.
        """
        keys, _handler = _state_change_keys()
        for name in ("refine_ridge_paths", "coarse_ridge_curves", "probe_ridge_points"):
            assert name in keys, f"{name} is not a recompute trigger"
