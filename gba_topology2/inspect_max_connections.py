"""inspect_max_connections.py - read-only probe for Basin Locator Step 3.

Prints, for every ascent launch branch of every visible saddle, the first
critical point the branch runs into along its great-circle ray, and the angles
that made the claim:

    along    progress along the ray to the hit. This is the quantity the rule
             minimises, so it is the number to read first: the smallest along
             inside the sector is the winner.
    cross    perpendicular deviation of the hit from the ray. The sector gate
             (|cross| <= cone) is what keeps the rule directional; a large cross
             means the branch aims at nothing in particular.
    sep      raw saddle-to-hit angle. Comparing sep against along shows whether
             the hit is genuinely on the ray.

The verdict column is the point: a branch is CONNECTED when its first hit is a
maximum, OCCLUDED when the first hit is a minimum or a saddle (the target column
then names the blocker in brackets), and NO_TARGET when nothing is in the sector.

It exists because the connection table is a claim, not a measurement: the only
way to tell a good claim from a plausible one is to see the numbers next to the
glyphs. The viewer draws the arcs for the selected maximum; this prints all of
them at once, and --trace explains a single saddle.

Run with:
    uv run python gba_topology2/inspect_max_connections.py Pd_20K_atom1_params_Volume.json
    uv run python gba_topology2/inspect_max_connections.py <params.json> --mode port_tangent
    uv run python gba_topology2/inspect_max_connections.py <params.json> --sort along
    uv run python gba_topology2/inspect_max_connections.py <params.json> --cone 4
    uv run python gba_topology2/inspect_max_connections.py <params.json> --trace SAD_388
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.diagnose_params import build_context, run_stages
from gba_topology2.src.max_connection import (
    CONE_PITCH,
    DIRECTION_MODES,
    MAX_ARC_DEG,
    ConnectionTarget,
    MaxConnection,
    build_connection_pool,
    connect_seed_to_maxima,
    connection_ray,
    disc_entry_angle,
    disc_geometry_from_cluster,
    summarize_connections,
    target_frame_angles,
)
from gba_topology2.src.saddle_launch import SaddleLaunchSeed
from gba_topology2.src.separatrix_tracer import excision_regions
from gba_topology2.src.sphere_frame import to_local, unit
from gba_topology2.src.sphere_viewer import load_params_from_dict

__all__ = ["format_table", "format_trace", "main"]

_SORT_KEYS: dict[str, str] = {
    "cross": "cross_track_deg",
    "along": "along_track_deg",
    "sep": "separation_deg",
    "saddle": "saddle_id",
}


def format_table(
    connections: list[MaxConnection],
    sort_key: str = "along",
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
            f"{'saddle':<18}{'br':>3}  {'target':<18}"
            f"{'cross':>8}{'along':>8}{'sep':>8}  flags  status"
        )
    ]
    for c in rows:
        flags = "".join(
            [
                "M" if c.mutual else "-",
                "V" if c.verified else "u",
                "G" if c.method == "gradient" else "-",
                "T" if c.transited_ids else "-",
            ]
        )
        # The target column shows the claimed maximum, or the blocker for an
        # occluded branch, so "why is there no arc here?" is readable inline.
        target = c.max_id or (f"[{c.blocker_id}]" if c.blocker_id else "-")
        lines.append(
            f"{c.saddle_id:<18}{c.branch_index:>3}  "
            f"{target:<18}"
            f"{(f'{c.cross_track_deg:.2f}' if c.cross_track_deg is not None else '-'):>8}"
            f"{(f'{c.along_track_deg:.2f}' if c.along_track_deg is not None else '-'):>8}"
            f"{(f'{c.separation_deg:.2f}' if c.separation_deg is not None else '-'):>8}"
            f"  {flags}  {c.status}"
        )
    return "\n".join(lines)


def format_trace(
    seeds: list[SaddleLaunchSeed],
    pool: list[ConnectionTarget],
    saddle_id: str,
    mode: str,
    sphere_radius: float,
    cone_rad: float,
    cap_rad: float,
    center: tuple[float, float, float] | None = None,
) -> str:
    """Audit one saddle: score every pool critical point against each branch.

    The connection table reports the winner. When the winner is not the one you
    expected, the answer is in the runners-up: a nearer maximum that was excluded
    by the sector gate, a blocker sitting just off the ray, a target past the
    cap. This prints the full scoring so the rule can be checked against the
    geometry rather than inferred from the verdict.

    The scoring must happen in the CENTRED frame, exactly as the rule does: the
    rays are centred, the pool is world. Pd_20K sits at the origin so a missing
    ``to_local`` is invisible there and wrong on every molecule -- the recurring
    frame trap, which is why this takes ``center`` explicitly.

    Args:
        seeds: All launch seeds, verified and provisional.
        pool: The visible-critical-point pool the claims were made against.
        saddle_id: The seed to audit.
        mode: The ray definition in force.
        sphere_radius: Atom sphere radius.
        cone_rad: Forward-sector half-angle, radians.
        cap_rad: Arc-length cap, radians.
        center: World-frame nucleus, used to map the pool into the centred frame.

    Returns:
        The trace text, or a message naming the valid ids when saddle_id is
        not among the seeds.
    """
    seed = next((s for s in seeds if s.saddle_id == saddle_id), None)
    if seed is None:
        ids = ", ".join(sorted(s.saddle_id for s in seeds))
        return f"No seed with id {saddle_id!r}. Valid ids: {ids}"

    lines = [
        (
            f"TRACE {seed.saddle_id}  k={seed.fold_order}  "
            f"{len(seed.ascent_branches)} ascent branch(es)  mode={mode}"
        )
    ]
    for branch in seed.ascent_branches:
        ray = connection_ray(seed, branch, mode, sphere_radius)  # type: ignore[arg-type]
        if ray is None:
            lines.append(f"  branch {branch.sector_index}: degenerate ray")
            continue
        scored: list[tuple[float, float, str, str, bool, float]] = []
        for t in pool:
            if t.cp_id == seed.saddle_id:
                continue  # a branch never claims its own saddle
            pos = np.asarray(t.position, dtype=np.float64)
            local = to_local(pos, center) if center is not None else pos
            u = unit(local)
            if u is None:
                continue
            along, cross = target_frame_angles(u, ray)
            # Score the DISC ENTRY, exactly as the rule does: a CP whose disc the
            # great circle misses is not a candidate at all, and the key is where
            # the ray first touches the disc, not where the centre sits.
            # Score the DISC ENTRY, exactly as the rule does: the boundary is
            # evaluated TOWARD the ray (an elongated corridor is narrow across),
            # a great circle that misses it is not a candidate at all, and the
            # key is where the ray first touches the boundary, not where the
            # centre sits.
            rho = t.radius_toward(ray.direction_unit)
            entry = disc_entry_angle(along, cross, rho)
            in_sector = (
                entry is not None and 0.0 < entry <= cap_rad and abs(cross) <= cone_rad
            )
            scored.append(
                (
                    # inf = the great circle misses the disc; sorts last and is
                    # never in-sector, so it reads as a miss in the table.
                    math.degrees(entry) if entry is not None else math.inf,
                    math.degrees(cross),
                    t.cp_id,
                    t.polarity,
                    in_sector,
                    math.degrees(rho),
                )
            )
        # Discs the great circle misses entirely sort last, by cross-track.
        scored.sort(key=lambda r: (not r[4], r[0], r[1]))
        lines.append(
            f"  branch {branch.sector_index} "
            f"(in-sector first, by disc-entry arc-length):"
        )
        lines.append(
            f"    {'id':<18}{'kind':<9}{'entry':>8}{'cross':>8}{'rho':>7}   in-sector"
        )
        for entry, cross, cp_id, polarity, in_sector, disc in scored[:14]:
            entry_txt = "     miss" if math.isinf(entry) else f"{entry:8.2f}"
            lines.append(
                f"    {cp_id:<18}{polarity:<9}{entry_txt}{cross:>8.2f}{disc:>7.2f}"
                f"   {'YES' if in_sector else ''}"
            )
        n_in = sum(1 for r in scored if r[4])
        lines.append(
            f"    ({n_in} of {len(scored)} pool entries inside "
            f"+/-{math.degrees(cone_rad):.2f} deg, <= {math.degrees(cap_rad):.2f} deg)"
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
        default="port_tangent",
        help="Ray definition: port_tangent (default) = outward tangent at the "
        "port on the excision boundary; chord = centre->port bearing from the "
        "saddle centre. Both lie on the same great circle; the port is the "
        "origin because the field inside the boundary is the degenerate core.",
    )
    parser.add_argument(
        "--max-arc",
        type=float,
        default=None,
        help="Arc-length cap in DEGREES (default: the module constant MAX_ARC_DEG, "
        "180 = the whole sphere). Reach is an angular property, so it is not "
        "expressed in mesh pitches: the same pitch count means 64 deg on Pd_20K "
        "and 89 deg on ethene3.",
    )
    parser.add_argument(
        "--cone",
        type=float,
        default=None,
        help="Forward-sector half-angle in mesh-pitch units (default: the module "
        "constant CONE_PITCH)",
    )
    parser.add_argument(
        "--sort",
        choices=sorted(_SORT_KEYS),
        default="along",
        help="Column to sort by, ascending (default: along, the quantity the rule "
        "minimises, so the shortest connections come first and unclaimed last)",
    )
    parser.add_argument(
        "--trace",
        default=None,
        metavar="SADDLE_ID",
        help="Audit one saddle: for each ascent branch, list every pool critical "
        "point by along/cross track and mark which are inside the forward sector. "
        "This is how a claim is explained, and how a missing one is diagnosed.",
    )
    parser.add_argument(
        "--no-gradient",
        action="store_true",
        help="Disable the Step 4 gradient fallback: run the pure geodesic ray only, "
        "so branches the straight connector cannot resolve stay unresolved. Use this "
        "to A/B the two-stage rule against Step 3 on the same field.",
    )
    parser.add_argument(
        "--no-transit",
        action="store_true",
        help="Disable the excision-region transit: an ascent that terminates inside "
        "a cluster's noisy disc stays there. Use this to A/B the transit rule "
        "against the pure Step 4 ascent on the same field.",
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
    pool = build_connection_pool(
        basin_centers=ctx.basin_centers,
        raw_saddles=ctx.require_morse().saddles,
        effective_cps=ctx.effective_cps,
        hidden_vertex_ids=ctx.hidden_vertex_ids,
        center=ctx.center,
        disc_by_vertex={
            m.vertex_id: disc_geometry_from_cluster(cl)
            for cl in ctx.clusters
            for m in cl.members
        },
        disc_by_id=ctx.eff_disc_geometry,
    )

    kwargs: dict[str, object] = {"direction_mode": args.mode}
    if args.max_arc is not None:
        kwargs["max_arc_deg"] = args.max_arc
    if args.cone is not None:
        kwargs["cone_pitch"] = args.cone
    connections = connect_seed_to_maxima(
        seeds=[*ctx.seeds, *ctx.unverified_seeds],
        targets=pool,
        sphere_radius=ctx.sphere_radius,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        center=ctx.center,
        mesh_pts=None if args.no_gradient else ctx.pts,
        f_vals=None if args.no_gradient else ctx.f_vals,
        find_k_nearest=None if args.no_gradient else ctx.locator,
        excision_regions=(
            ()
            if args.no_transit
            else excision_regions([u.as_cluster for u in ctx.units])
        ),
        **kwargs,  # type: ignore[arg-type]
    )
    stats = summarize_connections(connections)

    n_maxima = sum(1 for b in ctx.basin_centers if b.polarity == "maximum")
    n_pool_sad = sum(1 for t in pool if t.polarity == "saddle")
    n_pool_min = sum(1 for t in pool if t.polarity == "minimum")
    print(f"file        : {ctx.plt_path}")
    print(f"atom        : {ctx.atom_num}   field: {ctx.field_name}")
    print(f"R           : {ctx.sphere_radius:.6f} A")
    print(f"pitch       : {math.degrees(ctx.delta_theta_rad):.4f} deg")
    print(f"mode        : {args.mode}")
    print(
        f"seeds       : {len(ctx.seeds)} verified + {len(ctx.unverified_seeds)} provisional"
    )
    print(
        f"pool        : {len(pool)} visible CPs "
        f"({n_maxima} max, {n_pool_min} min, {n_pool_sad} sad)"
    )
    cone_pitch = args.cone if args.cone is not None else CONE_PITCH
    arc_deg = args.max_arc if args.max_arc is not None else MAX_ARC_DEG
    print(
        f"sector      : +/- {cone_pitch} pitches "
        f"= {math.degrees(cone_pitch * ctx.delta_theta_rad):.2f} deg, "
        f"cap {arc_deg:.2f} deg"
    )
    print(
        f"verdicts    : {stats['n_connected']} connected, "
        f"{stats['n_occluded']} occluded, {stats['n_no_target']} no target "
        f"of {stats['n_branches']} branches"
    )
    print(
        f"methods     : {stats['n_geodesic']} great circle, "
        f"{stats['n_gradient']} gradient ascent "
        f"(gates: {stats['n_gate_behind']} behind, {stats['n_gate_cone']} cone, "
        f"{stats['n_gate_cap']} cap, {stats['n_gate_occluded']} occluded, "
        f"{stats['n_gate_empty']} empty)"
    )
    print(
        f"transit     : {stats['n_transited']} branches crossed an excision region "
        f"({', '.join(stats['transited_regions']) or 'none'})"
    )
    print(
        f"occluded by : {stats['n_blocked_by_minimum']} minima, "
        f"{stats['n_blocked_by_saddle']} saddles"
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

    if args.trace:
        print()
        print(
            format_trace(
                seeds=[*ctx.seeds, *ctx.unverified_seeds],
                pool=pool,
                saddle_id=args.trace,
                mode=args.mode,
                sphere_radius=ctx.sphere_radius,
                cone_rad=cone_pitch * ctx.delta_theta_rad,
                cap_rad=math.radians(arc_deg),
                center=ctx.center,
            )
        )
        return 0
    print()
    print(format_table(connections, sort_key=args.sort))
    return 0


if __name__ == "__main__":
    sys.exit(main())
