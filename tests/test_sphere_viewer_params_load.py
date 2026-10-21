"""Unit tests for the viewer's parameter load / reload machinery.

Covers the pieces of ``sphere_viewer`` that make "Load Parameters" work:
1. ``build_viewer_session`` -- assembles the mesh-bound state for an atom.
2. The session swap -- a different file/atom produces a new session whose
   geometry and nucleus differ.
3. ``_uploaded_file_content`` -- extracts bytes from a browser upload.
4. The CLI -- the ``--params`` flag is parsed and forwarded.
5. The UI wiring -- the LOAD button and the dialog are present.

These are source-level and headless checks: they do not start a server.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

import gba_topology2.src.sphere_viewer as sv

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLT = str(ROOT / "ethene4.plt")


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


def test_build_viewer_session_assembles_the_state() -> None:
    """The session carries the mesh, the nucleus and the geometry."""
    session = sv.build_viewer_session(PLT, 1)

    assert session.plt_path == os.path.abspath(PLT)
    assert session.atom_number == 1
    assert session.pts_np.shape[0] == session.geom_info["num_vertices"]
    assert session.triangles_np.shape[0] == session.geom_info["num_triangles"]
    assert session.sphere_radius > 0.0
    assert session.delta_theta_rad > 0.0
    assert len(session.center) == 3
    assert session.field_items


def test_build_viewer_session_rejects_a_missing_file() -> None:
    """A missing .plt is reported, not silently ignored."""
    with pytest.raises(FileNotFoundError):
        sv.build_viewer_session(str(ROOT / "does_not_exist.plt"), 1)


def test_build_viewer_session_rejects_an_unknown_atom() -> None:
    """An atom without a sphere mesh is an error, not an empty session."""
    with pytest.raises(RuntimeError):
        sv.build_viewer_session(PLT, 999)


def test_two_sessions_are_independent() -> None:
    """Sessions do not share mutable state, so a swap is safe."""
    first = sv.build_viewer_session(PLT, 1)
    second = sv.build_viewer_session(PLT, 1)

    assert first.sphere_poly is not second.sphere_poly
    assert first.pts_np is not second.pts_np


# ---------------------------------------------------------------------------
# Uploaded file content
# ---------------------------------------------------------------------------


def test_uploaded_content_from_a_single_blob() -> None:
    """A file uploaded as one bytes blob is returned as-is."""
    content = sv._uploaded_file_content({"name": "a.json", "content": b"{}"})

    assert content == b"{}"


def test_uploaded_content_from_chunks() -> None:
    """A file uploaded as a list of chunks is joined."""
    content = sv._uploaded_file_content(
        {"name": "a.json", "content": [b'{"a"', b": 1}"]}
    )

    assert content == b'{"a": 1}'


def test_uploaded_content_of_nothing_is_none() -> None:
    """No selection yields no content."""
    assert sv._uploaded_file_content(None) is None
    assert sv._uploaded_file_content({}) is None
    assert sv._uploaded_file_content({"content": "not bytes"}) is None


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_cli_exposes_the_params_flag() -> None:
    """--params is a documented command-line option."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "gba_topology2" / "src" / "sphere_viewer.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=180,
        check=False,
    )

    assert result.returncode == 0
    assert "--params" in result.stdout


def test_cli_forwards_the_params_path() -> None:
    """The parsed --params value reaches run_sphere_viewer."""
    src = (ROOT / "gba_topology2" / "src" / "sphere_viewer.py").read_text(
        encoding="utf-8"
    )

    assert "params_path=args.params" in src


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------


def test_the_load_button_is_next_to_save() -> None:
    """The toolbar carries both the save and the load action."""
    src = (ROOT / "gba_topology2" / "src" / "sphere_viewer.py").read_text(
        encoding="utf-8"
    )

    save_at = src.index('"Save Parameters"')
    load_at = src.index('"Load Parameters"')

    assert save_at < load_at
    assert "click=ctrl.open_load_dialog" in src


def test_the_load_dialog_is_wired() -> None:
    """The dialog offers a file input and a confirm action."""
    src = (ROOT / "gba_topology2" / "src" / "sphere_viewer.py").read_text(
        encoding="utf-8"
    )

    assert "v3.VDialog(" in src
    assert "v3.VFileInput(" in src
    assert "click=ctrl.confirm_load_parameters" in src
    assert "click=ctrl.close_load_dialog" in src


def test_the_load_error_is_surfaced_in_the_drawer() -> None:
    """A failed load is reported to the operator, not swallowed."""
    src = (ROOT / "gba_topology2" / "src" / "sphere_viewer.py").read_text(
        encoding="utf-8"
    )

    assert "params_load_error" in src
    assert "state.params_last_loaded_path" in src


def test_the_load_error_reaches_the_state() -> None:
    """The error is written to the trame state, not to a local.

    The drawer chip is bound to the state variable, so an error kept in a
    local would never be shown to the operator.
    """
    src = (ROOT / "gba_topology2" / "src" / "sphere_viewer.py").read_text(
        encoding="utf-8"
    )

    assert "state.params_load_error = str(exc)" in src
    assert "nonlocal params_load_error" not in src


# ---------------------------------------------------------------------------
# The exported snapshot
# ---------------------------------------------------------------------------


def test_the_viewer_exports_a_real_camera() -> None:
    """The save path captures the live camera and its reference nucleus."""
    session = sv.build_viewer_session(PLT, 1)
    state = {
        **{k: v for k, v in sv.DEFAULT_UI_STATE.items()},
        "selected_field": session.field_items[0]["value"],
        "geom_info": {},
        "morse_counts": {},
        "display_counts": {},
        "reduction_stats": {},
        "launch_stats": {},
        "selected_cp": None,
        "selected_cluster": None,
        "selected_catastrophe": None,
        "selected_basin_center": None,
    }

    class _State:
        def __init__(self, values: dict[str, object]) -> None:
            for key, value in values.items():
                setattr(self, key, value)

    from vtkmodules.vtkRenderingCore import vtkCamera

    camera = vtkCamera()
    camera.SetPosition(1.0, 2.0, 3.0)
    captured = sv.capture_camera_state(camera)

    payload = sv.save_params_to_dict(
        _State(state),
        plt_path=PLT,
        atom_info={**session.atom_info, "field_items": session.field_items},
        atom_num=1,
        camera=captured,
        camera_center=session.center,
    )

    assert payload["camera"]["position"] == pytest.approx([1.0, 2.0, 3.0])
    assert payload["camera"]["atom_center"] == pytest.approx(list(session.center))
    assert json.dumps(payload)
