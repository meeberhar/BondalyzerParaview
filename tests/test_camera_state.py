"""Unit tests for the camera capture/restore subroutine.

Covers ``gba_topology2.src.camera_state``:
1. ``capture_camera_state`` reads a camera's geometry.
2. ``camera_section_to_dict`` writes the JSON-safe section, including the
   reference nucleus that makes the camera portable.
3. ``camera_state_from_dict`` validates every field strictly.
4. ``apply_camera_state`` writes a snapshot back onto a camera.
5. ``rebase_camera_state`` translates a snapshot onto another nucleus.

A ``vtkCamera`` can be created and read without an OpenGL context, so these
tests run headless.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from vtkmodules.vtkRenderingCore import vtkCamera

from gba_topology2.src.camera_state import (
    CAMERA_STATE_KEYS,
    CameraState,
    apply_camera_state,
    camera_reference_center,
    camera_section_to_dict,
    camera_state_from_dict,
    capture_camera_state,
    rebase_camera_state,
)


def _make_camera(
    position: tuple[float, float, float] = (0.0, 0.0, 5.0),
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    view_up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> vtkCamera:
    """Build a camera with a known configuration.

    Args:
        position: Camera position.
        focal_point: Point the camera looks at.
        view_up: The view's up direction.

    Returns:
        The configured camera.
    """
    camera = vtkCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(*focal_point)
    camera.SetViewUp(*view_up)
    camera.SetParallelProjection(0)
    camera.SetParallelScale(1.5)
    camera.SetViewAngle(30.0)
    return camera


def test_capture_reads_every_field() -> None:
    """The snapshot mirrors the camera's position, target, roll and zoom."""
    camera = _make_camera(
        position=(3.0, 4.0, 5.0),
        focal_point=(1.0, 2.0, 3.0),
        view_up=(0.0, 0.0, 1.0),
    )

    state = capture_camera_state(camera)

    assert state.position == pytest.approx((3.0, 4.0, 5.0))
    assert state.focal_point == pytest.approx((1.0, 2.0, 3.0))
    assert state.view_up == pytest.approx((0.0, 0.0, 1.0))
    assert state.parallel_projection == 0
    assert state.parallel_scale == pytest.approx(1.5)
    assert state.view_angle == pytest.approx(30.0)


def test_capture_preserves_orthographic_zoom() -> None:
    """In orthographic mode the zoom lives in parallel_scale, and it is kept."""
    camera = _make_camera()
    camera.SetParallelProjection(1)
    camera.SetParallelScale(4.25)

    state = capture_camera_state(camera)

    assert state.parallel_projection == 1
    assert state.parallel_scale == pytest.approx(4.25)


def test_section_round_trips_through_json() -> None:
    """A captured camera survives a JSON round trip unchanged."""
    import json

    camera = _make_camera(position=(1.5, -2.5, 3.5), focal_point=(0.5, 0.5, 0.5))
    state = capture_camera_state(camera)

    section = camera_section_to_dict(state, (0.5, 0.5, 0.5))
    reloaded = json.loads(json.dumps(section))

    restored = camera_state_from_dict(reloaded)
    assert restored == state


def test_section_carries_the_reference_nucleus() -> None:
    """The section records the nucleus the camera was captured around."""
    camera = _make_camera()
    state = capture_camera_state(camera)

    section = camera_section_to_dict(state, (1.0, 2.0, 3.0))

    assert section["atom_center"] == pytest.approx((1.0, 2.0, 3.0))
    assert camera_reference_center(section) == pytest.approx((1.0, 2.0, 3.0))


def test_section_has_exactly_the_expected_keys() -> None:
    """The serialised field list is fixed, so the JSON stays diffable."""
    camera = _make_camera()
    section = camera_section_to_dict(capture_camera_state(camera), (0.0, 0.0, 0.0))

    assert set(section) == set(CAMERA_STATE_KEYS) | {"atom_center"}


def test_reference_center_is_optional() -> None:
    """A section without a nucleus reports None rather than failing."""
    assert camera_reference_center({"position": [0.0, 0.0, 1.0]}) is None


def test_apply_writes_the_camera_back() -> None:
    """Applying a snapshot restores every field on the camera."""
    source = _make_camera(
        position=(2.0, 3.0, 4.0),
        focal_point=(0.1, 0.2, 0.3),
        view_up=(0.0, 0.0, 1.0),
    )
    source.SetParallelProjection(1)
    source.SetParallelScale(7.5)
    state = capture_camera_state(source)

    target = vtkCamera()
    apply_camera_state(target, state)

    assert target.GetPosition() == pytest.approx((2.0, 3.0, 4.0))
    assert target.GetFocalPoint() == pytest.approx((0.1, 0.2, 0.3))
    assert target.GetViewUp() == pytest.approx((0.0, 0.0, 1.0))
    assert target.GetParallelProjection() == 1
    assert target.GetParallelScale() == pytest.approx(7.5)


def test_apply_preserves_the_viewing_direction() -> None:
    """The angle between the view direction and the target is unchanged.

    This is the property that actually matters: the camera must still be
    looking at the atom after a restore, not merely sit in the same place.
    """
    source = _make_camera(position=(3.0, 4.0, 5.0), focal_point=(1.0, 1.0, 1.0))
    state = capture_camera_state(source)

    target = vtkCamera()
    apply_camera_state(target, state)

    def _direction(cam: vtkCamera) -> tuple[float, float, float]:
        px, py, pz = cam.GetPosition()
        fx, fy, fz = cam.GetFocalPoint()
        length = math.sqrt((px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2)
        return ((px - fx) / length, (py - fy) / length, (pz - fz) / length)

    assert _direction(target) == pytest.approx(_direction(source))


def test_rebase_translates_position_and_target() -> None:
    """Moving the nucleus moves the camera by the same offset."""
    camera = _make_camera(position=(1.0, 2.0, 3.0), focal_point=(0.0, 0.0, 0.0))
    state = capture_camera_state(camera)

    rebased = rebase_camera_state(state, (0.0, 0.0, 0.0), (10.0, 20.0, 30.0))

    assert rebased.position == pytest.approx((11.0, 22.0, 33.0))
    assert rebased.focal_point == pytest.approx((10.0, 20.0, 30.0))


def test_rebase_preserves_orientation_and_zoom() -> None:
    """Translating the nucleus does not change how the camera is oriented."""
    camera = _make_camera(view_up=(0.0, 0.0, 1.0))
    camera.SetParallelScale(3.5)
    state = capture_camera_state(camera)

    rebased = rebase_camera_state(state, (1.0, 1.0, 1.0), (5.0, 5.0, 5.0))

    assert rebased.view_up == pytest.approx(state.view_up)
    assert rebased.parallel_scale == pytest.approx(state.parallel_scale)
    assert rebased.view_angle == pytest.approx(state.view_angle)
    assert rebased.parallel_projection == state.parallel_projection


def test_rebase_by_zero_is_the_identity() -> None:
    """Rebasing onto the same nucleus changes nothing."""
    camera = _make_camera()
    state = capture_camera_state(camera)

    rebased = rebase_camera_state(state, (2.0, 3.0, 4.0), (2.0, 3.0, 4.0))

    assert rebased == state


def test_from_dict_rejects_a_missing_field() -> None:
    """Every field is required; a partial payload is an error."""
    camera = _make_camera()
    section = camera_section_to_dict(capture_camera_state(camera), (0.0, 0.0, 0.0))
    del section["view_up"]

    with pytest.raises(ValueError, match="view_up"):
        camera_state_from_dict(section)


def test_from_dict_rejects_a_non_dict() -> None:
    """A non-dict payload is a type error, not a crash."""
    with pytest.raises(TypeError):
        camera_state_from_dict([1, 2, 3])


def test_from_dict_rejects_wrong_types() -> None:
    """Type errors are reported per field."""
    camera = _make_camera()
    base = camera_section_to_dict(capture_camera_state(camera), (0.0, 0.0, 0.0))

    bad_position = dict(base, position="nope")
    with pytest.raises(TypeError, match="position"):
        camera_state_from_dict(bad_position)

    bad_projection = dict(base, parallel_projection=1.0)
    with pytest.raises(TypeError, match="parallel_projection"):
        camera_state_from_dict(bad_projection)

    bad_scale = dict(base, parallel_scale=True)
    with pytest.raises(TypeError, match="parallel_scale"):
        camera_state_from_dict(bad_scale)


def test_from_dict_rejects_out_of_range_values() -> None:
    """Zoom and field-of-view must be physically meaningful."""
    camera = _make_camera()
    base = camera_section_to_dict(capture_camera_state(camera), (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="parallel_scale"):
        camera_state_from_dict(dict(base, parallel_scale=0.0))

    with pytest.raises(ValueError, match="view_angle"):
        camera_state_from_dict(dict(base, view_angle=0.0))

    with pytest.raises(ValueError, match="view_angle"):
        camera_state_from_dict(dict(base, view_angle=180.0))

    with pytest.raises(ValueError, match="parallel_projection"):
        camera_state_from_dict(dict(base, parallel_projection=7))


def test_from_dict_rejects_non_finite_values() -> None:
    """NaN and infinity would silently poison the camera."""
    camera = _make_camera()
    base = camera_section_to_dict(capture_camera_state(camera), (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="position"):
        camera_state_from_dict(dict(base, position=[0.0, float("nan"), 0.0]))

    with pytest.raises(ValueError, match="parallel_scale"):
        camera_state_from_dict(dict(base, parallel_scale=float("inf")))


def test_from_dict_rejects_a_wrong_length_vector() -> None:
    """A 2- or 4-component vector is a malformed camera."""
    camera = _make_camera()
    base = camera_section_to_dict(capture_camera_state(camera), (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="3 components"):
        camera_state_from_dict(dict(base, position=[0.0, 1.0]))


def test_camera_state_is_immutable() -> None:
    """The snapshot is a value object, so it cannot be mutated by accident."""
    state = CameraState(
        position=(0.0, 0.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        parallel_projection=0,
        parallel_scale=1.0,
        view_angle=30.0,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.parallel_scale = 5.0  # type: ignore[misc]
