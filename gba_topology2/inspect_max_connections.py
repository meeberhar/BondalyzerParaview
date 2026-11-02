"""inspect_max_connections.py - read-only probe for Basin Locator Step 3.

Prints, for every ascent launch branch of every visible saddle, the maximum the
branch claims and the three angles that made the claim:

    cross    perpendicular deviation from the branch's great-circle ray. This is
             the quantity the rule minimises, so it is the number to read first:
             a large cross means the branch aims at nothing in particular.
    along    progress along the ray to the target.
    sep      raw saddle-to-maximum angle. Comparing sep against cross shows
             whether the rule picked a directional match or merely a near one.

It exists because the connection table is a claim, not a measurement: the only
way to tell a good claim from a plausible one is to see the numbers next to the
glyphs. The viewer draws the arcs for the selected maximum; this prints all of
them at once.

Run with:
    uv run python gba_topology2/inspect_max_connections.py Pd_20K_atom1_params_Volume.json
    uv run python gba_topology2/inspect_max_connections.py <params.json> --mode port_tangent
    uv run python gba_topology2/inspect_max_connections.py <params.json> --sort cross
    uv run python gba_topology2/inspect_max_connections.py <params.json> --max-arc 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.diagnose_params import build_context, run_stages
from gba_topology2.src.max_connection import (
    DIRECTION_MODES,
    MaxConnection,
    connect_seed_to_maxima,
    summarize_connections,
)
from gba_topology2.src.sphere_viewer import load_params_from_dict

__all__ = ["format_table", "main"]

_SORT_KEYS: dict[str, str] = {
    "cross": "cross_track_deg",
    "along": "along_track_deg",
    "sep": "separation_deg",
    "saddle": "saddle_id",
}


def format_table(
    connections: list[MaxConnection],
    sort_key: str = "cross",
) -> str:
    """Render the connection list as an aligned text table.

    Args:
        connections: Output of :func:`connect_seed_to_maxima`.
        sort_key: One of ``cross``, ``along``, ``sep``, ``saddle``. Unclaimed
            branches sort last under every key.

    Returns:
        The table, header included, without a trailing newline.
    """
    field = _SORT_KEYS[sort_key]

    def _rank(c: MaxConnection) -> tuple[int, float, str]:
        """Sort by the chosen column numerically, by saddle id for the str column.

        Unclaimed branches (None metrics) sort last under every key.
        """
        value = getattr(c, field)
        if isinstance(value, str):
            return (0, 0.0, value)
        if value is None:
            return (2, 0.0, c.saddle_id)
        return (1, float(value), c.saddle_id)

    rows = sorted(connections, key=_rank)
    lines = [
        (
            f"{'saddle':<18}{'br':>3}  {'maximum':<18}"
            f"{'cross':>8}{'along':>8}{'sep':>8}  flags"
        )
    ]
    for c in rows:
        flags = "".join(
            [
                "M" if c.mutual else "-",
                "V" if c.verified else "u",
            ]
        )
        lines.append(
            f"{c.saddle_id:<18}{c.branch_index:>3}  "
            f"{(c.max_id or '-'):<18}"
            f"{(f'{c.cross_track_deg:.2f}' if c.cross_track_deg is not None else '-'):>8}"
            f"{(f'{c.along_track_deg:.2f}' if c.along_track_deg is not None else '-'):>8}"
            f"{(f'{c.separation_deg:.2f}' if c.separation_deg is not None else '-'):>8}"
            f"  {flags}"
        )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint.

    Returns:
        0 always; this probe reports and never judges.
    """
    parser = argparse.ArgumentParser(
        prog="inspect_max_connections.py",
        description="Print the ascent-branch -> maximum connection table from a "
        'saved "Save Parameters" JSON.',
    )
    parser.add_argument("params_json", help="Path to *_params.json")
    parser.add_argument("--plt", default=None, help="Override the recorded .plt path")
    parser.add_argument("-a", "--atom", type=int, default=None)
    parser.add_argument("-f", "--field", default=None)
    parser.add_argument(
        "--mode",
        choices=list(DIRECTION_MODES),
        default="chord",
        help="Ray definition: chord = centre->port bearing, port_tangent = "
        "outward tangent at the port. Both lie on the same great circle.",
    )
    parser.add_argument(
        "--max-arc",
        type=float,
        default=None,
        help="Arc-length cap in mesh-pitch units (default: the module constant)",
    )
    parser.add_argument(
        "--sort",
        choices=sorted(_SORT_KEYS),
        default="cross",
        help="Column to sort by, ascending (default: cross, so the tightest "
        "claims come first and the unclaimed last)",
    )
    args = parser.parse_args()

    with open(args.params_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    params = load_params_from_dict(payload)
    ctx = build_context(
        params, plt_path=args.plt, atom_num=args.atom, field_name=args.field
    )
    # Everything up to and including launch; the connect stage is re-run here so
    # the CLI overrides can vary it without touching the saved controls.
    run_stages(ctx, stop_after="launch")

    kwargs: dict[str, object] = {"direction_mode": args.mode}
    if args.max_arc is not None:
        kwargs["max_arc_pitch"] = args.max_arc
    connections = connect_seed_to_maxima(
        seeds=[*ctx.seeds, *ctx.unverified_seeds],
        maxima=ctx.basin_centers,
        sphere_radius=ctx.sphere_radius,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        center=ctx.center,
        **kwargs,  # type: ignore[arg-type]
    )
    stats = summarize_connections(connections)

    n_maxima = sum(1 for b in ctx.basin_centers if b.polarity == "maximum")
    print(f"file        : {ctx.plt_path}")
    print(f"atom        : {ctx.atom_num}   field: {ctx.field_name}")
    print(f"R           : {ctx.sphere_radius:.6f} A")
    print(f"pitch       : {math.degrees(ctx.delta_theta_rad):.4f} deg")
    print(f"mode        : {args.mode}")
    print(
        f"seeds       : {len(ctx.seeds)} verified + {len(ctx.unverified_seeds)} provisional"
    )
    print(f"maxima pool : {n_maxima} visible maxima")
    print(
        f"cap         : {stats['n_branches']} branches, "
        f"{stats['n_connected']} connected, {stats['n_no_target']} unclaimed"
    )
    print(
        f"cross       : mean {stats['mean_cross_track_deg']:.2f} deg, "
        f"max {stats['max_cross_track_deg']:.2f} deg"
    )
    print(
        f"separation  : mean {stats['mean_separation_deg']:.2f} deg, "
        f"max {stats['max_separation_deg']:.2f} deg"
    )
    print(f"mutual      : {stats['n_mutual']} of {stats['n_maxima_connected']} maxima")
    print()
    print(format_table(connections, sort_key=args.sort))
    return 0


if __name__ == "__main__":
    sys.exit(main())
