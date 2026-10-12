"""Basin Center Subroutine for gba_topology2.

Identifies, structures, and manages Basin Centers (local maxima and minima,
both raw and reduced effective extrema) that serve as the anchor seeds for
gradient-based basin detection.
"""

from dataclasses import dataclass
from typing import Any

from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.sphere_frame import to_global

__all__ = [
    "BasinCenter",
    "extract_basin_centers",
    "find_basin_center_by_id",
]


@dataclass(frozen=True)
class BasinCenter:
    """Represents a topological extremum serving as the center of a basin.

    Attributes:
        center_id: Unique identifier (e.g. ``"MAX_142"``, ``"E-MAX_MC12"``, ``"MIN_5"``).
        polarity: Extremum kind, either ``"maximum"`` or ``"minimum"``.
        position: 3D coordinates (X, Y, Z) in the **world / molecular frame**,
            i.e. the frame VTK draws in and the picker measures in. Raw extrema
            are mapped out of the centred atom frame by
            :func:`extract_basin_centers`.
        value: Representative scalar field value.
        is_effective: True if collapsed from an effective multi-CP set, False if raw.
        vertex_id: Original mesh vertex index (>= 0 for raw, -1 for effective).
        origin_id: Source cluster or catastrophe ID if effective, else None.
        net_index: Topological Euler index (+1 for simple extrema).
        member_count: Number of constituent critical points (1 for raw, >= 1 for effective).
    """

    center_id: str
    polarity: str
    position: tuple[float, float, float]
    value: float
    is_effective: bool
    vertex_id: int = -1
    origin_id: str | None = None
    net_index: int = 1
    member_count: int = 1

    def __post_init__(self) -> None:
        if self.polarity not in ("maximum", "minimum"):
            raise ValueError(
                f"Invalid basin center polarity '{self.polarity}'. Must be 'maximum' or 'minimum'."
            )
        if len(self.position) != 3:
            raise ValueError(f"Position must be 3D tuple, got {self.position}")


def extract_basin_centers(
    raw_cps: list[CriticalPoint],
    effective_cps: list[dict[str, Any]] | None = None,
    hidden_vertex_ids: set[int] | None = None,
    center: tuple[float, float, float] | None = None,
) -> list[BasinCenter]:
    """Extract candidate basin centers from raw critical points and effective CPs.

    Frame contract: ``raw_cps`` carry positions in the centred atom frame, where
    the nucleus is the origin, while the ``effective_cps`` dicts are built by the
    viewer's UI layer and already carry world (molecular) coordinates. Every
    emitted :class:`BasinCenter` is returned in the **world** frame so the
    inspector, the VTK glyphs and the 3D picker all agree. Pass ``center``
    (the world-frame nucleus) to map the raw extrema; omit it only for
    origin-centred synthetic meshes, where the two frames coincide.

    Args:
        raw_cps: Raw CriticalPoint instances from discrete Morse detection,
            in the centred atom frame.
        effective_cps: List of effective critical point dictionaries produced by
            the reduction stage (e.g. ``E-MAX`` and ``E-MIN``), in world frame.
        hidden_vertex_ids: Set of vertex indices that were absorbed into effective
            entities and should be excluded from raw basin centers.
        center: World-frame nuclear centre used to map raw CP positions out of
            the centred frame. When ``None``, raw positions are used verbatim.

    Returns:
        List of structured BasinCenter instances, sorted descending by scalar value.
    """
    hidden = hidden_vertex_ids or set()
    effective_list = effective_cps or []
    basin_centers: list[BasinCenter] = []

    # 1. Ingest visible raw extrema (centred frame -> world frame)
    for cp in raw_cps:
        if cp.vertex_id in hidden:
            continue
        if center:
            world_pos: tuple[float, float, float] = to_global(cp.position, center)
        else:
            world_pos = (
                float(cp.position[0]),
                float(cp.position[1]),
                float(cp.position[2]),
            )
        if cp.cp_type == "maximum":
            basin_centers.append(
                BasinCenter(
                    center_id=f"MAX_{cp.vertex_id}",
                    polarity="maximum",
                    position=world_pos,
                    value=float(cp.value),
                    is_effective=False,
                    vertex_id=cp.vertex_id,
                    origin_id=None,
                    net_index=1,
                    member_count=1,
                )
            )
        elif cp.cp_type == "minimum":
            basin_centers.append(
                BasinCenter(
                    center_id=f"MIN_{cp.vertex_id}",
                    polarity="minimum",
                    position=world_pos,
                    value=float(cp.value),
                    is_effective=False,
                    vertex_id=cp.vertex_id,
                    origin_id=None,
                    net_index=1,
                    member_count=1,
                )
            )

    # 2. Ingest effective extrema (E-MAX, E-MIN)
    for eff in effective_list:
        eff_type = eff.get("type", "")
        if eff_type not in ("E-MAX", "E-MIN"):
            continue

        pol = "maximum" if eff_type == "E-MAX" else "minimum"
        pos = eff.get("position", (0.0, 0.0, 0.0))
        basin_centers.append(
            BasinCenter(
                center_id=str(
                    eff.get("id_label", f"{eff_type}_{eff.get('origin_id', 'unknown')}")
                ),
                polarity=pol,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                value=float(eff.get("value", 0.0)),
                is_effective=True,
                vertex_id=-1,
                origin_id=str(eff.get("origin_id", ""))
                if eff.get("origin_id")
                else None,
                net_index=int(eff.get("net_index", 1)),
                member_count=int(eff.get("member_count", 1)),
            )
        )

    # Return sorted by value descending
    basin_centers.sort(key=lambda b: -b.value)
    return basin_centers


def find_basin_center_by_id(
    basin_centers: list[BasinCenter],
    center_id: str | None,
) -> BasinCenter | None:
    """Find a BasinCenter by its unique ID.

    Args:
        basin_centers: List of BasinCenter objects.
        center_id: Unique string ID to locate.

    Returns:
        The matching BasinCenter or None if not found or center_id is None.
    """
    if center_id is None:
        return None
    for bc in basin_centers:
        if bc.center_id == center_id:
            return bc
    return None
