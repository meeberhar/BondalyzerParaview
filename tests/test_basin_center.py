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
