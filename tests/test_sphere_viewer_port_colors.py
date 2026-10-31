"""Source-level guards for the ridge/valley colour convention.

A ridge port and an ascent launch seed are the **same manifold**: the ridge is
the unstable manifold of the saddle, running towards maxima, and the launch seed
is the stub that leaves the saddle along it. Once seeds launch from the ports,
drawing them in opposite colours is not a cosmetic inconsistency -- it tells the
operator that a port and the stub growing out of it are different structures.

The viewer previously had valley = orange / ridge = cyan for ports and
ascent = orange-red / descent = cyan for seeds, and both features default to on.
These guards pin the reconciled convention and stop it drifting back.

Every actor that draws one of the two manifolds now takes its colour from the
shared ``ASCENT_RGB`` / ``DESCENT_RGB`` constants, so a port and the seed stub
growing out of it cannot drift apart by construction -- the invariant is
structural, not asserted. The provisional (unverified) seed pair uses the same
constants at reduced opacity, because a third hue would add a legend entry for
what is the same manifold.

Headless, source-level checks, following the pattern in
``test_sphere_viewer_theme.py``: no server, no render window.
"""

import pathlib
import re

import pytest

from gba_topology2.src import sphere_viewer as sv

SOURCE = pathlib.Path(sv.__file__).resolve()

# The two RGB triples the viewer uses for the ascending and descending manifolds.
ASCENT_RGB = (1.0, 0.35, 0.10)
DESCENT_RGB = (0.10, 0.75, 1.0)

# Every actor in the scene that draws one of the two manifolds, with the constant
# it must take its colour from. Ports set the colour directly; seed actors are
# built by a factory, so their colour is a call argument.
PORT_SITES = {
    "rport_actor": "ASCENT_RGB",
    "vport_actor": "DESCENT_RGB",
}
SEED_SITES = {
    "launch_ascent_poly": "ASCENT_RGB",
    "launch_descent_poly": "DESCENT_RGB",
    "launch_unv_ascent_poly": "ASCENT_RGB",
    "launch_unv_descent_poly": "DESCENT_RGB",
}


def test_module_constants_are_the_expected_hues() -> None:
    """The shared constants carry the reconciled palette."""
    assert sv.ASCENT_RGB == pytest.approx(ASCENT_RGB)
    assert sv.DESCENT_RGB == pytest.approx(DESCENT_RGB)
    # The two manifolds must stay distinguishable from each other.
    assert sv.ASCENT_RGB != sv.DESCENT_RGB


@pytest.mark.parametrize("actor, constant", sorted(PORT_SITES.items()))
def test_port_actors_take_colour_from_the_shared_constant(
    actor: str, constant: str
) -> None:
    """Ridge ports are ascent orange-red; valley ports are descent cyan."""
    source = SOURCE.read_text(encoding="utf-8")
    match = re.search(rf"{actor}\.GetProperty\(\)\.SetColor\(\*([A-Z_]+)\)", source)
    assert match, f"{actor} must set its colour from a shared constant"
    assert match.group(1) == constant


@pytest.mark.parametrize("poly_var, constant", sorted(SEED_SITES.items()))
def test_seed_actors_take_colour_from_the_shared_constant(
    poly_var: str, constant: str
) -> None:
    """Every seed actor, verified and provisional, is built from a constant."""
    source = SOURCE.read_text(encoding="utf-8")
    match = re.search(
        rf"{poly_var},\s*\w+ = _make_seed_actor\(\s*renderer,\s*(\w+)", source
    )
    assert match, f"{poly_var} must be built by _make_seed_actor from a colour constant"
    assert match.group(1) == constant


def test_no_manifold_actor_hardcodes_a_literal_rgb() -> None:
    """A literal triple on any of these actors would let the two features drift.

    The whole point of routing every site through the constants is that the
    port/seed correspondence cannot be broken by editing one of them.
    """
    source = SOURCE.read_text(encoding="utf-8")
    for actor in list(PORT_SITES) + [
        "launch_ascent_actor",
        "launch_descent_actor",
        "launch_unv_ascent_actor",
        "launch_unv_descent_actor",
    ]:
        assert not re.search(
            rf"{actor}\.GetProperty\(\)\.SetColor\(\s*[0-9.]+,",
            source,
        ), f"{actor} hardcodes an RGB literal; use ASCENT_RGB/DESCENT_RGB"


def test_provisional_seeds_share_the_hue_and_differ_only_in_opacity() -> None:
    """Unverified seeds are the same manifold: same colour, faded, not a new hue."""
    assert 0.0 < sv.UNVERIFIED_OPACITY < 1.0
    source = SOURCE.read_text(encoding="utf-8")
    faded = re.findall(
        r"_make_seed_actor\(\s*renderer,\s*\w+,\s*opacity=([\w.]+)", source
    )
    assert faded == ["UNVERIFIED_OPACITY", "UNVERIFIED_OPACITY"], (
        "both provisional seed actors must fade by the shared constant"
    )
    # Verified actors pass no opacity, i.e. stay solid.
    solid = re.findall(r"_make_seed_actor\(\s*renderer,\s*\w+\s*\)", source)
    assert len(solid) == 2


def test_port_switch_label_matches_the_actors() -> None:
    """The legend is the operator's only key to the colours, so it must agree."""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'label="Show Boundary Ports (Orange=Ridge, Cyan=Valley)"' in source
    assert "Orange=Valley" not in source, "stale legend: valley is cyan, not orange"


def test_seed_switch_label_matches_the_actors() -> None:
    """The seed legend names ascent as orange and descent as cyan."""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'label="Show Launch Seeds (Orange=Ascent, Cyan=Descent)"' in source


def test_port_colours_are_not_reintroduced_as_literals() -> None:
    """The old port palette must not return under a different actor name."""
    source = SOURCE.read_text(encoding="utf-8")
    # (1.0, 0.55, 0.0) was the valley-port orange; (0.0, 0.85, 0.95) the ridge cyan.
    assert "1.0, 0.55, 0.0" not in source, (
        "old valley-port orange collides with the ascent hue"
    )
    assert "0.0, 0.85, 0.95" not in source, "old ridge-port cyan collides with descent"
