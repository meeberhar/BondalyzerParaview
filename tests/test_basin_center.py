"""Unit tests for Basin Center Subroutine (gba_topology2/src/basin_center.py)."""

import pytest

from gba_topology2.src.basin_center import (
    BasinCenter,
    extract_basin_centers,
    find_basin_center_by_id,
)
from gba_topology2.src.morse_detector import CriticalPoint


def test_basin_center_dataclass_validation() -> None:
    """Verify dataclass creation and validation rules."""
    bc = BasinCenter(
        center_id="MAX_10",
        polarity="maximum",
        position=(1.0, 0.0, 0.0),
        value=3.14,
        is_effective=False,
        vertex_id=10,
    )
    assert bc.center_id == "MAX_10"
    assert bc.polarity == "maximum"
    assert bc.position == (1.0, 0.0, 0.0)
    assert bc.value == 3.14
    assert not bc.is_effective
    assert bc.vertex_id == 10
    assert bc.net_index == 1

    with pytest.raises(ValueError, match="Invalid basin center polarity"):
        BasinCenter(
            center_id="SAD_5",
            polarity="saddle",  # type: ignore[arg-type]
            position=(0.0, 1.0, 0.0),
            value=1.0,
            is_effective=False,
        )

    with pytest.raises(ValueError, match="Position must be 3D tuple"):
        BasinCenter(
            center_id="MAX_1",
            polarity="maximum",
            position=(0.0, 1.0),  # type: ignore[arg-type]
            value=1.0,
            is_effective=False,
        )


def test_extract_basin_centers_raw_and_effective() -> None:
    """Verify extraction from mixed raw critical points and effective CPs."""
    raw_cps = [
        CriticalPoint(
            vertex_id=1, cp_type="maximum", position=(0.0, 0.0, 1.0), value=10.0
        ),
        CriticalPoint(
            vertex_id=2, cp_type="minimum", position=(0.0, 0.0, -1.0), value=1.0
        ),
        CriticalPoint(
            vertex_id=3, cp_type="saddle", position=(1.0, 0.0, 0.0), value=5.0
        ),
        CriticalPoint(
            vertex_id=4, cp_type="maximum", position=(0.0, 1.0, 0.0), value=8.0
        ),
    ]

    effective_cps = [
        {
            "id_label": "E-MAX_MC12",
            "type": "E-MAX",
            "position": [0.5, 0.5, 0.707],
            "value": 12.0,
            "origin_id": "MC12",
            "net_index": 1,
            "member_count": 3,
        },
        {
            "id_label": "E-SAD_CAT1",
            "type": "E-SAD",
            "position": [0.707, 0.707, 0.0],
            "value": 4.0,
            "origin_id": "CAT1",
            "net_index": -1,
            "member_count": 2,
        },
    ]

    # Vertex 4 was collapsed into E-MAX_MC12, so it is hidden
    hidden_ids = {4}

    centers = extract_basin_centers(
        raw_cps=raw_cps,
        effective_cps=effective_cps,
        hidden_vertex_ids=hidden_ids,
    )

    # Saddles (raw and E-SAD) and hidden vertex 4 must not appear as basin centers
    ids = [c.center_id for c in centers]
    assert "E-MAX_MC12" in ids
    assert "MAX_1" in ids
    assert "MIN_2" in ids
    assert "MAX_4" not in ids
    assert "SAD_3" not in ids
    assert "E-SAD_CAT1" not in ids

    # Sorted descending by value
    values = [c.value for c in centers]
    assert values == [12.0, 10.0, 1.0]

    # Check lookup
    found = find_basin_center_by_id(centers, "E-MAX_MC12")
    assert found is not None
    assert found.is_effective is True
    assert found.polarity == "maximum"
    assert found.origin_id == "MC12"
    assert found.member_count == 3

    assert find_basin_center_by_id(centers, "NONEXISTENT") is None
    assert find_basin_center_by_id(centers, None) is None


def test_extract_basin_centers_maps_raw_cp_out_of_centred_frame() -> None:
    """Raw CPs are centred-frame; basin centers must come back in world frame.

    Regression test for the reported symptom on ``ethene3.plt``: selecting an
    ``E-MAX_MC*`` / raw extremum placed the gold halo far from the sphere,
    because raw CP positions (nucleus at the origin) were emitted verbatim while
    VTK draws in the molecular frame.
    """
    center = (1.2618, -0.0001, -0.0002)
    r = 0.412733

    # A raw maximum sitting on the +X pole of the centred sphere.
    raw_cps = [
        CriticalPoint(
            vertex_id=5217,
            cp_type="maximum",
            position=(-r, 0.0, 0.0),
            value=10.0,
        ),
        CriticalPoint(
            vertex_id=100,
            cp_type="minimum",
            position=(r, 0.0, 0.0),
            value=1.0,
        ),
    ]

    centers = extract_basin_centers(raw_cps, center=center)
    assert len(centers) == 2

    by_id = {c.center_id: c for c in centers}
    max_pos = by_id["MAX_5217"].position
    min_pos = by_id["MIN_100"].position

    # Mapped onto the world sphere: distance from the nucleus equals R.
    import math

    for pos in (max_pos, min_pos):
        d = math.sqrt(sum((pos[i] - center[i]) ** 2 for i in range(3)))
        assert abs(d - r) < 1e-9, f"{pos} is {d} from the nucleus, expected {r}"

    # Explicit world coordinates for the +X/-X poles.
    assert max_pos[0] == pytest.approx(center[0] - r, abs=1e-12)
    assert min_pos[0] == pytest.approx(center[0] + r, abs=1e-12)

    # Without a nucleus the centred positions pass through unchanged, which is
    # the pre-fix behaviour and is only correct for origin-centred meshes.
    unshifted = extract_basin_centers(raw_cps, center=None)
    unshifted_by_id = {c.center_id: c for c in unshifted}
    assert unshifted_by_id["MAX_5217"].position == pytest.approx((-r, 0.0, 0.0))


def test_extract_basin_centers_mixes_world_effective_and_centred_raw() -> None:
    """Effective dicts are already world-frame; raw CPs are centred-frame.

    Both must land on the same sphere, so the halo, the inspector and the picker
    agree for raw and effective centers alike.
    """
    center = (1.2618, -0.0001, -0.0002)
    r = 0.412733

    raw_cps = [
        CriticalPoint(
            vertex_id=1, cp_type="maximum", position=(-r, 0.0, 0.0), value=10.0
        ),
        CriticalPoint(
            vertex_id=2, cp_type="maximum", position=(0.0, r, 0.0), value=8.0
        ),
    ]

    # The viewer builds effective positions with to_global, i.e. world frame.
    effective_cps = [
        {
            "id_label": "E-MAX_MC12",
            "type": "E-MAX",
            "position": [center[0], center[1] + r, center[2]],
            "value": 12.0,
            "origin_id": "MC12",
            "net_index": 1,
            "member_count": 3,
        }
    ]

    centers = extract_basin_centers(
        raw_cps,
        effective_cps=effective_cps,
        hidden_vertex_ids={2},
        center=center,
    )

    ids = {c.center_id for c in centers}
    assert ids == {"E-MAX_MC12", "MAX_1"}

    import math

    for bc in centers:
        d = math.sqrt(sum((bc.position[i] - center[i]) ** 2 for i in range(3)))
        assert abs(d - r) < 1e-9, f"{bc.center_id} at {bc.position} is off-sphere"
