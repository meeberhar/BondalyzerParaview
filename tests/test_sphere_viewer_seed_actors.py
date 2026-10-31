"""Unit tests for the launch-seed actor factory.

The factory exists because the scene now draws two sets of seed stubs: verified
seeds at full opacity, and provisional seeds (a ring that balanced cleanly at a
fold order disagreeing with the reduction's index) faded. Both must share the
ascent/descent hues, so the colour is a parameter and the only visual difference
is opacity.

Headless: builds a real vtkRenderer and inspects the actors, no render window.
"""

import pytest
from vtkmodules.vtkRenderingCore import vtkRenderer

from gba_topology2.src.sphere_viewer import (
    ASCENT_RGB,
    DESCENT_RGB,
    UNVERIFIED_OPACITY,
    _make_seed_actor,
)


def test_actor_is_registered_and_hidden_by_default() -> None:
    """Nothing is drawn until Step 9 fills the polydata and opens the gate."""
    renderer = vtkRenderer()
    assert renderer.GetActors().GetNumberOfItems() == 0

    poly, actor = _make_seed_actor(renderer, ASCENT_RGB)

    assert renderer.GetActors().GetNumberOfItems() == 1
    assert not actor.GetVisibility()
    assert poly.GetNumberOfPoints() == 0


def test_actor_uses_the_given_colour() -> None:
    """Ascent and descent actors take their hues from the shared constants."""
    renderer = vtkRenderer()
    _, ascent = _make_seed_actor(renderer, ASCENT_RGB)
    _, descent = _make_seed_actor(renderer, DESCENT_RGB)

    assert ascent.GetProperty().GetColor() == pytest.approx(ASCENT_RGB, abs=1e-6)
    assert descent.GetProperty().GetColor() == pytest.approx(DESCENT_RGB, abs=1e-6)


def test_verified_actor_is_opaque() -> None:
    """The default is solid: a converged seed is certified, not provisional."""
    _, actor = _make_seed_actor(vtkRenderer(), ASCENT_RGB)
    assert actor.GetProperty().GetOpacity() == pytest.approx(1.0, abs=1e-6)


def test_provisional_actor_is_faded() -> None:
    """Unverified seeds differ only in opacity, not hue."""
    _, actor = _make_seed_actor(vtkRenderer(), ASCENT_RGB, opacity=UNVERIFIED_OPACITY)
    assert actor.GetProperty().GetOpacity() == pytest.approx(
        UNVERIFIED_OPACITY, abs=1e-6
    )
    assert actor.GetProperty().GetColor() == pytest.approx(ASCENT_RGB, abs=1e-6)


def test_each_call_returns_an_independent_pipeline() -> None:
    """Filling one actor's polydata must not touch the others.

    The four seed actors are filled from separate point/cell arrays in Step 9;
    sharing a polydata between them would mix verified and provisional geometry.
    """
    renderer = vtkRenderer()
    poly_a, actor_a = _make_seed_actor(renderer, ASCENT_RGB)
    poly_b, actor_b = _make_seed_actor(renderer, ASCENT_RGB)

    assert poly_a is not poly_b
    assert actor_a is not actor_b
    assert renderer.GetActors().GetNumberOfItems() == 2
