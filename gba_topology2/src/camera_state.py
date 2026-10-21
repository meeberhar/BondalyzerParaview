"""Camera State Capture and Restore Subroutine.

The sphere viewer renders on the server and streams images to the browser, so
the authoritative camera is a ``vtkCamera`` owned by the render window. Trame
mirrors the browser's interactive camera onto that server-side camera on every
interaction, which makes the server camera a faithful mirror of what the
operator is looking at.

This module owns the *serialisable* form of that camera:

- :func:`capture_camera_state` reads it off a camera,
- :func:`camera_state_to_dict` writes it into the parameter JSON,
- :func:`camera_state_from_dict` reads it back with strict validation,
- :func:`apply_camera_state` puts it back onto a camera.

Keeping the format here means the viewer never hand-rolls the field list, and
the round trip can be tested without a browser, a render window or an OpenGL
context -- a ``vtkCamera`` can be created and read headlessly.

The captured fields are exactly the ones that determine what is on screen:

``position``
    Where the camera sits in world coordinates. Together with ``focal_point``
    this fixes the viewing direction.
``focal_point``
    The point the camera looks at (the rotation centre).
``view_up``
    Which way is "up" on screen, so the roll is preserved too.
``parallel_projection`` / ``parallel_scale`` / ``view_angle``
    The projection mode and the zoom. ``parallel_scale`` is the half-height of
    the view volume in orthographic mode; ``view_angle`` is the vertical field
    of view in perspective mode. Zooming changes exactly one of the two,
    depending on the projection mode, so both are stored.
"""

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CAMERA_STATE_KEYS",
    "CameraState",
    "apply_camera_state",
    "camera_reference_center",
    "camera_section_to_dict",
    "camera_state_from_dict",
    "capture_camera_state",
    "rebase_camera_state",
]


# The serialised field names, in a fixed order so the JSON is stable and
# diffable. Also used to validate a loaded payload.
CAMERA_STATE_KEYS: tuple[str, ...] = (
    "position",
    "focal_point",
    "view_up",
    "parallel_projection",
    "parallel_scale",
    "view_angle",
)


@dataclass(frozen=True)
class CameraState:
    """An immutable snapshot of a render camera.

    Attributes:
        position: (3,) camera position in world coordinates.
        focal_point: (3,) point the camera looks at.
        view_up: (3,) "up" direction of the view.
        parallel_projection: 0 for perspective, 1 for orthographic.
        parallel_scale: Half-height of the orthographic view volume.
        view_angle: Vertical field of view in degrees (perspective mode).
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_projection: int
    parallel_scale: float
    view_angle: float


def _coerce_vec3(key: str, raw: Any) -> tuple[float, float, float]:
    """Coerce a payload entry into a 3-tuple of finite floats.

    Args:
        key: Field name, used in the error message.
        raw: The value read from the payload.

    Returns:
        The value as a (3,) tuple of floats.

    Raises:
        TypeError: If the value is not a sequence of three numbers.
        ValueError: If any component is not finite.
    """
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise TypeError(f"Camera field '{key}' must be a sequence of 3 numbers")
    values = list(raw)
    if len(values) != 3:
        raise ValueError(
            f"Camera field '{key}' must have 3 components, got {len(values)}"
        )
    out: list[float] = []
    for component in values:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError(f"Camera field '{key}' must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"Camera field '{key}' must be finite, got {number}")
        out.append(number)
    return (out[0], out[1], out[2])


def _coerce_float(key: str, raw: Any) -> float:
    """Coerce a payload entry into a finite float.

    Args:
        key: Field name, used in the error message.
        raw: The value read from the payload.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is not a number.
        ValueError: If the value is not finite.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(
            f"Camera field '{key}' must be a number, got {type(raw).__name__}"
        )
    number = float(raw)
    if not math.isfinite(number):
        raise ValueError(f"Camera field '{key}' must be finite, got {number}")
    return number


def capture_camera_state(camera: Any) -> CameraState:
    """Read a camera's current state into an immutable snapshot.

    Args:
        camera: A ``vtkCamera`` (duck-typed: it only needs ``GetPosition``,
            ``GetFocalPoint``, ``GetViewUp``, ``GetParallelProjection``,
            ``GetParallelScale`` and ``GetViewAngle``).

    Returns:
        The captured :class:`CameraState`.
    """
    return CameraState(
        position=_vec3(camera.GetPosition()),
        focal_point=_vec3(camera.GetFocalPoint()),
        view_up=_vec3(camera.GetViewUp()),
        parallel_projection=int(camera.GetParallelProjection()),
        parallel_scale=float(camera.GetParallelScale()),
        view_angle=float(camera.GetViewAngle()),
    )


def _vec3(values: Any) -> tuple[float, float, float]:
    """Convert a 3-component sequence into a plain 3-tuple of floats.

    Args:
        values: The sequence to convert.

    Returns:
        The value as a (3,) tuple of floats.
    """
    x, y, z = values
    return (float(x), float(y), float(z))


def camera_section_to_dict(
    state: CameraState,
    atom_center: tuple[float, float, float],
) -> dict[str, Any]:
    """Convert a camera snapshot into a JSON-serialisable section.

    The nucleus the camera was captured around is stored alongside it. That
    reference point is what makes the camera portable: loading the same
    parameters against a different atom (or a different file whose atom sits
    elsewhere in the molecule) moves the camera by the same offset as the
    nucleus, so it still looks *at* the atom instead of at empty space.

    Args:
        state: The snapshot to convert.
        atom_center: The world-frame nucleus the camera was captured around.

    Returns:
        A dict with the keys of :data:`CAMERA_STATE_KEYS` plus
        ``"atom_center"``.
    """
    return {
        "position": [float(c) for c in state.position],
        "focal_point": [float(c) for c in state.focal_point],
        "view_up": [float(c) for c in state.view_up],
        "parallel_projection": int(state.parallel_projection),
        "parallel_scale": float(state.parallel_scale),
        "view_angle": float(state.view_angle),
        "atom_center": [float(c) for c in atom_center],
    }


def camera_state_from_dict(payload: Any) -> CameraState:
    """Rebuild a camera snapshot from a JSON payload, validating every field.

    Extra keys (such as the ``atom_center`` reference point) are ignored, so
    the camera fields can be read from a section that carries more than the
    camera.

    Args:
        payload: The ``camera`` section of a parameter file.

    Returns:
        The validated :class:`CameraState`.

    Raises:
        TypeError: If the payload is not a dict, or a field has the wrong type.
        ValueError: If a field is missing, malformed or out of range.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"Camera payload must be a dict, got {type(payload).__name__}")

    missing = [key for key in CAMERA_STATE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Camera payload is missing: {', '.join(missing)}")

    parallel_projection = payload["parallel_projection"]
    if isinstance(parallel_projection, bool) or not isinstance(
        parallel_projection, int
    ):
        raise TypeError("Camera field 'parallel_projection' must be an int")
    if parallel_projection not in (0, 1):
        raise ValueError(
            f"Camera field 'parallel_projection' must be 0 or 1, got {parallel_projection}"
        )

    parallel_scale = _coerce_float("parallel_scale", payload["parallel_scale"])
    if parallel_scale <= 0.0:
        raise ValueError(
            f"Camera field 'parallel_scale' must be positive, got {parallel_scale}"
        )

    view_angle = _coerce_float("view_angle", payload["view_angle"])
    if not 0.0 < view_angle < 180.0:
        raise ValueError(
            f"Camera field 'view_angle' must lie in (0, 180), got {view_angle}"
        )

    return CameraState(
        position=_coerce_vec3("position", payload["position"]),
        focal_point=_coerce_vec3("focal_point", payload["focal_point"]),
        view_up=_coerce_vec3("view_up", payload["view_up"]),
        parallel_projection=parallel_projection,
        parallel_scale=parallel_scale,
        view_angle=view_angle,
    )


def apply_camera_state(camera: Any, state: CameraState) -> None:
    """Write a camera snapshot back onto a camera.

    Args:
        camera: The ``vtkCamera`` to modify.
        state: The snapshot to apply.
    """
    camera.SetPosition(*state.position)
    camera.SetFocalPoint(*state.focal_point)
    camera.SetViewUp(*state.view_up)
    camera.SetParallelProjection(state.parallel_projection)
    camera.SetParallelScale(state.parallel_scale)
    camera.SetViewAngle(state.view_angle)


def camera_reference_center(payload: Any) -> tuple[float, float, float] | None:
    """Read the reference nucleus stored alongside a camera section.

    Args:
        payload: The ``camera`` section of a parameter file.

    Returns:
        The stored nucleus as a 3-tuple, or None when the section does not
        record one (files written before the camera existed).

    Raises:
        TypeError: If the payload is not a dict.
        ValueError: If the stored centre is malformed.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"Camera payload must be a dict, got {type(payload).__name__}")
    if "atom_center" not in payload:
        return None
    return _coerce_vec3("atom_center", payload["atom_center"])


def rebase_camera_state(
    state: CameraState,
    old_center: tuple[float, float, float],
    new_center: tuple[float, float, float],
) -> CameraState:
    """Translate a camera snapshot from one atom's frame to another's.

    Loading the same parameters against a *different* atom (or a different
    file whose atom sits elsewhere in the molecule) must move the camera by
    the same offset as the nucleus. Without this, restoring the camera would
    leave it pointing at empty space beside the atom.

    The offset is applied to the position and the focal point; the view
    direction, roll and zoom are untouched, because translating the nucleus
    does not change the viewing geometry relative to it.

    Args:
        state: The captured camera snapshot.
        old_center: The nucleus the camera was captured around.
        new_center: The nucleus the camera should be applied to.

    Returns:
        A new snapshot expressed relative to ``new_center``.
    """
    offset = tuple(n - o for n, o in zip(new_center, old_center, strict=True))
    return CameraState(
        position=_vec3(
            tuple(c + d for c, d in zip(state.position, offset, strict=True))
        ),
        focal_point=_vec3(
            tuple(c + d for c, d in zip(state.focal_point, offset, strict=True))
        ),
        view_up=state.view_up,
        parallel_projection=state.parallel_projection,
        parallel_scale=state.parallel_scale,
        view_angle=state.view_angle,
    )
