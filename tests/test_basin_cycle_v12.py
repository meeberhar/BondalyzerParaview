"""Regression tests for the closed separatrix cycle rule (basin v12).

These cover :func:`gba_topology2.src.basin_boundary._boundary_cycle` as reached
through ``find_minimum_basins`` / ``find_maximum_basins``.

The case that motivated them, from a real Pd20k snapshot: minimum MIN_19182
(MC73) is bounded by two saddles, MC15 (2 ascending branches) and MC20 (4 --
a monkey saddle). MC15's two branches and MC20's first two close on maxima
MC36 and MC40, so the basin is a ring of 4 arcs. MC20's branches 2 and 3 run
to MC42 and MC34, which bound the basins of *other* minima, not this one.
Drawing them paints the hill next door with MC73's color.

The pre-v12 filter selected arcs whose target had degree exactly 2, and bailed
out entirely if any target had degree 1. MC42 and MC34 have degree 1 *because*
they are spurious, so the escape hatch fired in precisely the case the filter
was written for and nothing was ever dropped. Leaf-pruning has no such hole:
the degree-1 target is the thing being removed, not the reason to give up.
"""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from gba_topology2.src.basin_boundary import (
    MaximumBasin,
    MinimumBasin,
    _boundary_cycle,
    find_maximum_basins,
    find_minimum_basins,
    serialize_maximum_basins,
    serialize_minimum_basins,
)
from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection, MinConnection
from gba_topology2.src.ridge_refiner import (
    fit_quad_bezier_connections,
    gradify_connections,
)
from gba_topology2.src.sphere_frame import to_global

# (saddle, max_id) targets are optional: a stalled arc carries None, and the
# graph builders do not filter it out, so the rule has to decide what it means.
_ArcSpec = Sequence[tuple[int, str | None]]

# An endpoint weld is asserted to this: the bend re-normalizes onto the sphere,
# which shifts a coordinate by ~2 ULP; the SMALLEST real node move is ~100x more.
_WELD_TOL_A = 1e-9


def _conn(
    saddle_id: str,
    max_id: str | None,
    branch_index: int,
) -> MaxConnection:
    """An ascending saddle -> maximum branch."""
    return MaxConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        max_id=max_id,
        status="connected",
        direction_mode="ray",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        mutual=True,
        arc_points=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        saddle_value=0.5,
        max_value=1.0 if max_id else None,
        verified=True,
    )


def _desc(
    saddle_id: str,
    min_id: str | None,
    branch_index: int,
) -> MinConnection:
    """A descending saddle -> minimum branch."""
    return MinConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        min_id=min_id,
        status="connected",
        direction_mode="port_tangent",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        arc_points=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        saddle_value=0.5,
        min_value=0.1 if min_id else None,
        verified=True,
    )


def _center(center_id: str, polarity: str) -> BasinCenter:
    """A critical point, at a position the rule never looks at."""
    z = -1.0 if polarity == "minimum" else 1.0
    return BasinCenter(
        center_id=center_id,
        polarity=polarity,
        position=(0.0, 0.0, z),
        value=z,
        is_effective=True,
        vertex_id=-1,
        origin_id=None,
        net_index=1,
        member_count=1,
    )


def _min_side(
    arcs: Mapping[str, _ArcSpec],
    drains: Sequence[tuple[str, int, str]],
    minima: Sequence[str],
) -> dict[str, MinimumBasin]:
    """Basins of ``minima`` from ``arcs`` (saddle -> (branch, max_id)).

    ``drains`` is the descent table: which saddle drains into which minimum,
    i.e. what makes a saddle a boundary saddle of that basin.
    """
    connections = [
        _conn(saddle, max_id, bi)
        for saddle, branches in arcs.items()
        for bi, max_id in branches
    ]
    centers = [_center(m, "minimum") for m in minima]
    descents = [_desc(saddle, min_id, bi) for saddle, bi, min_id in drains]
    return find_minimum_basins(connections, centers, descents)


def _shape(basin: MinimumBasin) -> list[tuple[str, int, str | None]]:
    """A basin's ring as (saddle, branch, target) triples."""
    return [(a.saddle_id, a.branch_index, a.max_id) for a in basin.boundary_arcs]


class TestMonkeySaddleCycle:
    """The reported defect, at the exact topology that caused it."""

    # MC15 + MC20 bound MIN_19182; MC20's far branches reach MC42 / MC34.
    RING: ClassVar[Mapping[str, _ArcSpec]] = {
        "E-SAD_MC15": [(0, "E-MAX_MC36"), (1, "E-MAX_MC40")],
        "E-SAD_MC20": [
            (0, "E-MAX_MC36"),
            (1, "E-MAX_MC40"),
            (2, "E-MAX_MC42"),
            (3, "E-MAX_MC34"),
        ],
    }
    DRAINS: ClassVar[Sequence[tuple[str, int, str]]] = [
        ("E-SAD_MC15", 0, "MIN_19182"),
        ("E-SAD_MC20", 0, "MIN_19182"),
    ]

    def test_mc73_ring_is_the_four_closed_arcs(self) -> None:
        basins = _min_side(self.RING, self.DRAINS, ["MIN_19182"])
        assert _shape(basins["MIN_19182"]) == [
            ("E-SAD_MC15", 0, "E-MAX_MC36"),
            ("E-SAD_MC15", 1, "E-MAX_MC40"),
            ("E-SAD_MC20", 0, "E-MAX_MC36"),
            ("E-SAD_MC20", 1, "E-MAX_MC40"),
        ]

    def test_spurious_branches_leave_the_serializer_too(self) -> None:
        """The screen draws the serialized ids, so the drop must reach them."""
        basins = _min_side(self.RING, self.DRAINS, ["MIN_19182"])
        out = serialize_minimum_basins(basins)["MIN_19182"]
        assert out["boundary_maxima"] == ["E-MAX_MC36", "E-MAX_MC40"]
        assert out["boundary_saddles"] == ["E-SAD_MC15", "E-SAD_MC20"]

    def test_dropping_leaves_the_saddle_that_bounds_the_basin(self) -> None:
        """MC20 stays a boundary saddle: 2 of its 4 branches, not zero of them."""
        basins = _min_side(self.RING, self.DRAINS, ["MIN_19182"])
        saddles = serialize_minimum_basins(basins)["MIN_19182"]["boundary_saddles"]
        assert "E-SAD_MC20" in saddles


class TestMaxSideIsTheDual:
    """The same defect on the maximum side, where it was equally dead."""

    @staticmethod
    def _max_side(
        adjacency: Mapping[str, Mapping[str, Sequence[int]]],
        geometry: Mapping[str, _ArcSpec],
        maxima: Sequence[str],
    ) -> dict[str, MaximumBasin]:
        """Basins of ``maxima``.

        Mirrors :func:`_min_side` with the tables exchanged, which is what the
        duality is: ``adjacency`` (saddle -> branches climbing into a maximum)
        decides who bounds, ``geometry`` (saddle -> (branch, min_id)) is the
        arc that gets drawn.
        """
        connections = [
            _conn(saddle, maximum, bi)
            for maximum, saddles in adjacency.items()
            for saddle, branches in saddles.items()
            for bi in branches
        ]
        centers = [_center(m, "maximum") for m in maxima]
        descents = [
            _desc(saddle, min_id, bi)
            for saddle, branches in geometry.items()
            for bi, min_id in branches
        ]
        return find_maximum_basins(connections, centers, descents)

    def test_monkey_saddle_branches_pruned_on_a_hill(self) -> None:
        basins = self._max_side(
            adjacency={"MAX_4864": {"S1": [0, 1], "S2": [0, 1]}},
            geometry={
                "S1": [(0, "M1"), (1, "M2")],
                "S2": [(0, "M1"), (1, "M2"), (2, "M3"), (3, "M4")],
            },
            maxima=["MAX_4864"],
        )
        ring = basins["MAX_4864"].boundary_arcs
        assert [(a.saddle_id, a.branch_index, a.min_id) for a in ring] == [
            ("S1", 0, "M1"),
            ("S1", 1, "M2"),
            ("S2", 0, "M1"),
            ("S2", 1, "M2"),
        ]
        assert serialize_maximum_basins(basins)["MAX_4864"]["boundary_minima"] == [
            "M1",
            "M2",
        ]


class TestTwoSaddlesSharingOneTarget:
    """The shape that looks like a tail and is not."

    My first reading of the rule expected a multi-arc tail to be pruned whole.
    It is not, and the rule is right: a target met by two bounding saddles is
    on the ring by definition, which is the same 2-saddles-into-1-minimum shape
    as ethene3's hill MAX_4864. Cascading is not a thing this rule can do --
    see test_single_pass_is_the_fixed_point.
    """

    def test_two_saddles_sharing_a_maximum_are_part_of_the_ring(self) -> None:
        # Cycle on M1/M2 via S1,S2. S2's third branch and S3's first meet on
        # M3, so M3 has degree 2 and stays; only M4, S3's other branch, is a
        # leaf.
        arcs: dict[str, _ArcSpec] = {
            "S1": [(0, "M1"), (1, "M2")],
            "S2": [(0, "M1"), (1, "M2"), (2, "M3")],
            "S3": [(0, "M3"), (1, "M4")],
        }
        drains = [(s, 0, "MIN_X") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        assert _shape(basin) == [
            ("S1", 0, "M1"),
            ("S1", 1, "M2"),
            ("S2", 0, "M1"),
            ("S2", 1, "M2"),
            ("S2", 2, "M3"),
            ("S3", 0, "M3"),
        ]


class TestShapesThatMustSurvive:
    """Everything the rule is not allowed to eat."""

    def test_degree_three_target_inside_a_cycle_is_kept(self) -> None:
        # Three saddles sharing both maxima: degree 3, no leaves, all kept.
        arcs = {f"S{n}": [(0, "M1"), (1, "M2")] for n in (1, 2, 3)}
        drains = [(s, 0, "MIN_X") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        assert len(basin.boundary_arcs) == 6

    def test_parallel_branches_of_one_saddle_are_kept(self) -> None:
        """Two branches of one saddle into the same max: ethene3's MS3 -> 4864.

        Degree 4, and the two arcs are not a tail -- they are the double pass
        over the same shoulder, so both are drawn.
        """
        arcs: dict[str, _ArcSpec] = {
            "E-SAD_MS3": [(0, "MAX_4864"), (1, "MAX_4864")],
            "E-SAD_MS4": [(0, "MAX_4864"), (1, "MAX_4864")],
        }
        drains = [("E-SAD_MS3", 0, "E-MIN_MC3"), ("E-SAD_MS4", 0, "E-MIN_MC3")]
        basin = _min_side(arcs, drains, ["E-MIN_MC3"])["E-MIN_MC3"]
        assert len(basin.boundary_arcs) == 4

    def test_single_saddle_basin_keeps_all_its_arcs(self) -> None:
        """No cycle exists: the two arcs are each other's only endpoints.

        Pd20k's MIN_19177/78/79 look like this. There is no closed boundary to
        extract, so every candidate is drawn, as it was before v12.
        """
        basin = _min_side(
            {"S1": [(0, "M1"), (1, "M2")]},
            [("S1", 0, "MIN_X")],
            ["MIN_X"],
        )["MIN_X"]
        assert len(basin.boundary_arcs) == 2

    def test_a_basin_with_no_leaves_is_untouched(self) -> None:
        """The clean 6-arc ring (3 saddles x 2 branches, 3 maxima) is the norm."""
        arcs = {
            "E-SAD_MC23": [(0, "E-MAX_MC39"), (1, "E-MAX_MC41")],
            "E-SAD_MC25": [(0, "E-MAX_MC39"), (1, "E-MAX_MC37")],
            "E-SAD_MC27": [(0, "E-MAX_MC41"), (1, "E-MAX_MC37")],
        }
        drains = [(s, 0, "MIN_0") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_0"])["MIN_0"]
        assert len(basin.boundary_arcs) == 6
        out = serialize_minimum_basins({"MIN_0": basin})["MIN_0"]
        assert out["boundary_maxima"] == [
            "E-MAX_MC37",
            "E-MAX_MC39",
            "E-MAX_MC41",
        ]


class TestDegenerateAndStable:
    """Order, determinism, stalled arcs."""

    def test_ring_order_follows_sorted_saddles(self) -> None:
        arcs: dict[str, _ArcSpec] = {
            "S-C": [(0, "M1"), (1, "M2"), (2, "M3")],
            "S-A": [(0, "M1"), (1, "M2")],
            "S-B": [(0, "M1"), (1, "M2"), (2, "M4")],
        }
        drains = [(s, 0, "MIN_X") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        # M3 and M4 are leaves (one arc each); the two survivors per saddle are
        # kept in sorted-saddle order, both branches of each.
        assert [(a.saddle_id, a.branch_index) for a in basin.boundary_arcs] == [
            ("S-A", 0),
            ("S-A", 1),
            ("S-B", 0),
            ("S-B", 1),
            ("S-C", 0),
            ("S-C", 1),
        ]

    def test_single_pass_is_the_fixed_point(self) -> None:
        """One pass equals iterate-to-stability, so the loop is not needed.

        Deleting a leaf drops that target's only arc, removing the target
        outright; every other target's degree is untouched. No target can
        become a leaf in a second pass. Pinned here because the property is
        what licenses the straight-line implementation.
        """
        seeds: list[dict[str, _ArcSpec]] = [
            {"S1": [(0, "M1"), (1, "M2"), (2, "M3"), (3, "M4")]},
            {
                "S1": [(0, "M1"), (1, "M2")],
                "S2": [(0, "M1"), (1, "M2"), (2, "M3")],
                "S3": [(0, "M3"), (1, "M4")],
            },
            {"S1": [(0, None), (1, "M1")]},
            {
                "S1": [(0, "M1"), (1, "M1")],
                "S2": [(0, "M1"), (1, "M2")],
            },
        ]
        for spec in seeds:
            arcs = [
                _conn(s, max_id, bi)
                for s, branches in spec.items()
                for bi, max_id in branches
            ]
            once = _boundary_cycle(arcs, lambda a: a.max_id)
            twice = _boundary_cycle(once, lambda a: a.max_id)
            assert [id(a) for a in once] == [id(a) for a in twice]

    def test_determinism_across_input_permutations(self) -> None:
        arcs: dict[str, _ArcSpec] = {
            "S1": [(0, "M1"), (1, "M2")],
            "S2": [(0, "M1"), (1, "M2"), (2, "M3"), (3, "M4")],
        }
        drains = [("S1", 0, "MIN_X"), ("S2", 0, "MIN_X")]
        first = _shape(_min_side(arcs, drains, ["MIN_X"])["MIN_X"])
        flipped_arcs: dict[str, _ArcSpec] = {
            k: list(reversed(v)) for k, v in reversed(list(arcs.items()))
        }
        flipped = _shape(
            _min_side(flipped_arcs, list(reversed(drains)), ["MIN_X"])["MIN_X"]
        )
        assert sorted(first) == sorted(flipped)

    def test_stalled_arcs_are_leaves_and_drop_out(self) -> None:
        """max_id None reaches the rule; it is a leaf, not a target."""
        arcs: dict[str, _ArcSpec] = {
            "S1": [(0, "M1"), (1, "M2")],
            "S2": [(0, "M1"), (1, "M2"), (2, None)],
        }
        drains = [("S1", 0, "MIN_X"), ("S2", 0, "MIN_X")]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        assert _shape(basin) == [
            ("S1", 0, "M1"),
            ("S1", 1, "M2"),
            ("S2", 0, "M1"),
            ("S2", 1, "M2"),
        ]

    def test_all_stalled_falls_back_to_every_arc(self) -> None:
        """No cycle at all: keep the candidates rather than draw nothing."""
        arcs: dict[str, _ArcSpec] = {"S1": [(0, None), (1, None)]}
        basin = _min_side(arcs, [("S1", 0, "MIN_X")], ["MIN_X"])["MIN_X"]
        assert len(basin.boundary_arcs) == 2

    def test_helper_is_idempotent_on_its_own_output(self) -> None:
        arcs = [
            _conn("S1", "M1", 0),
            _conn("S1", "M2", 1),
            _conn("S2", "M1", 0),
            _conn("S2", "M2", 1),
            _conn("S2", "M3", 2),
        ]
        once = _boundary_cycle(arcs, lambda a: a.max_id)
        twice = _boundary_cycle(once, lambda a: a.max_id)
        assert [id(a) for a in once] == [id(a) for a in twice]

    def test_empty_input_is_empty(self) -> None:
        empty: list[MaxConnection] = []
        assert _boundary_cycle(empty, lambda a: a.max_id) == []

    def test_helper_binds_to_descent_arcs(self) -> None:
        """The TypeVar covers MinConnection as well."""
        arcs: list[MinConnection] = [
            _desc("S1", "M1", 0),
            _desc("S1", "M2", 1),
            _desc("S2", "M1", 0),
            _desc("S2", "M2", 1),
            _desc("S2", "M3", 2),
        ]
        kept = _boundary_cycle(arcs, lambda a: a.min_id)
        assert [(a.saddle_id, a.min_id) for a in kept] == [
            ("S1", "M1"),
            ("S1", "M2"),
            ("S2", "M1"),
            ("S2", "M2"),
        ]


# ---------------------------------------------------------------------------
# Gradified geometry x the closed cycle (basin v12 rule, plan behavior 6)
# ---------------------------------------------------------------------------

# The bend's fixture, self-contained on purpose: `tests/` has no `__init__.py`, so
# the ridge fixtures in test_ridge_refiner.py cannot be imported, and a copy that
# drifts is the price of a test that runs.
_RING_MESH_POINTS = 1200
_RING_SPHERE_R = 1.0
_RING_PITCH = math.radians(3.0)
_RING_X = np.array([1.0, 0.0, 0.0])
_RING_Y = np.array([0.0, 1.0, 0.0])
_RING_Z = np.array([0.0, 0.0, 1.0])


def _ring_mesh() -> npt.NDArray[np.float64]:
    """A quasi-uniform sphere, the fixture the ridge tests already use."""
    i = np.arange(0, _RING_MESH_POINTS, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / _RING_MESH_POINTS)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack(
        [
            _RING_SPHERE_R * np.sin(phi) * np.cos(theta),
            _RING_SPHERE_R * np.sin(phi) * np.sin(theta),
            _RING_SPHERE_R * np.cos(phi),
        ]
    )


def _crest_ring_field(pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A field that peaks on the EQUATOR, so the crest is a known great circle.

    It depends only on the distance to that circle, so the gradient vanishes ALONG
    the crest: the only move this field permits is the transverse one the bend
    makes. A test built on it cannot pass by sliding along the ridge.
    """
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    cross = np.arcsin(np.clip(u @ _RING_Z, -1.0, 1.0))
    return np.asarray(np.exp(-((cross / (4.0 * _RING_PITCH)) ** 2)), dtype=np.float64)


def _desc_with_geometry(
    saddle_id: str,
    min_id: str,
    branch_index: int,
    lat_a_pitch: float,
    lat_b_pitch: float,
    span_deg: float = 24.0,
    samples: int = 25,
    center: tuple[float, float, float] | None = None,
) -> MinConnection:
    """A descent arc on a real geodesic: ``samples`` vertices, true positions.

    The rest of this file builds topology at ``arc_points=((0,0,1),(1,0,0))``,
    geometry the cycle rule never looks at. Here the geometry IS the point, so the
    polyline is the genuine chord a ray produces between two latitudes (their
    slerp), with the endpoints on OPPOSITE sides of the crest so the bend has
    something to bend onto. ``center`` lifts the fixture into the world frame --
    the project's frame bugs have all been invisible on a centered atom.
    """
    half = math.radians(span_deg) / 2.0

    def end(lat_pitch: float, sign: float) -> npt.NDArray[np.float64]:
        off = lat_pitch * _RING_PITCH
        u = (
            math.cos(off) * (math.cos(half) * _RING_X + sign * math.sin(half) * _RING_Y)
            + math.sin(off) * _RING_Z
        )
        return u / np.linalg.norm(u)

    a, b = end(lat_a_pitch, 1.0), end(lat_b_pitch, -1.0)
    omega = math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
    path: list[tuple[float, float, float]] = []
    for k in range(samples):
        t = k / (samples - 1)
        w_a = math.sin((1.0 - t) * omega) / math.sin(omega)
        w_b = math.sin(t * omega) / math.sin(omega)
        u = w_a * a + w_b * b
        u = u / np.linalg.norm(u)
        path.append(
            (
                float(u[0] * _RING_SPHERE_R),
                float(u[1] * _RING_SPHERE_R),
                float(u[2] * _RING_SPHERE_R),
            )
            if center is None
            else to_global(
                (
                    float(u[0] * _RING_SPHERE_R),
                    float(u[1] * _RING_SPHERE_R),
                    float(u[2] * _RING_SPHERE_R),
                ),
                center,
            )
        )
    return MinConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        min_id=min_id,
        status="connected",
        direction_mode="ray",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        arc_points=tuple(path),
        saddle_value=0.5,
        min_value=0.1,
        verified=True,
    )


class TestGradifiedGeometryKeepsTheCycle:
    """The bend moves geometry; the v12 rule reads keys. Both facts, one test.

    Everything this work claims rests on the bend being GEOMETRY ONLY: a basin
    boundary may be drawn onto the field's crest without changing WHICH arcs bound
    it. The cycle rule selects on ``(saddle_id, max_id)`` / ``(saddle_id, min_id)``
    and never touches a polyline, so that is provable -- and a regression that made
    it read geometry would show up here as a ring that changes when a switch that
    only bends lines is flipped. The polylines must also ACTUALLY move: a test of
    the cycle alone would pass on a bend that did nothing.
    """

    RING: ClassVar[tuple[tuple[str, tuple[tuple[int, str], ...]], ...]] = (
        ("E-SAD_MC15", ((0, "E-MAX_MC36"), (1, "E-MAX_MC40"))),
        ("E-SAD_MC20", ((0, "E-MAX_MC36"), (1, "E-MAX_MC40"))),
    )
    # The saddle's two drains, as (saddle, branch, minimum). TUPLES: a class-level
    # dict/list is shared mutable state across every test instance.
    DRAINS: ClassVar[tuple[tuple[str, int, str], ...]] = (
        ("E-SAD_MC15", 0, "MIN_19182"),
        ("E-SAD_MC20", 0, "MIN_19182"),
    )

    def _descent(self) -> list[MinConnection]:
        """The two drains, as geodesics that straddle the fixture's crest ring."""
        return [
            _desc_with_geometry(s, m, b, -0.6, 0.5 if i == 0 else -0.6)
            for i, (s, b, m) in enumerate(self.DRAINS)
        ]

    def _gradify(
        self,
        descent: list[MinConnection],
        center: tuple[float, float, float] | None = None,
    ) -> list[MinConnection]:
        """Bend the descent table onto the fixture's crest, asserting it bent."""
        pts = _ring_mesh()
        bent, stats = gradify_connections(
            descent,
            mesh_pts=pts,
            f_vals=_crest_ring_field(pts),
            sphere_radius=_RING_SPHERE_R,
            delta_theta_mesh_rad=_RING_PITCH,
            center=center,
        )
        assert stats["n_arcs"] == len(descent), stats
        assert stats["n_gradified"] == len(descent), (
            "the bend refused a crest that is here by construction -- the cycle "
            "test below would pass vacuously on an untouched table"
        )
        for original, moved in zip(descent, bent):
            assert moved.gradified is True
            # The endpoints are welded: a basin ring joins these polylines tip to
            # tip, and the join's tolerance dedup is what keeps them closed. NOT
            # exact equality -- the weld re-normalizes onto the sphere, which moves
            # a coordinate by a couple of ULP (2e-16 here, vs. the >=0.01 A of a
            # real node displacement). Tolerance says "the endpoint is the CP".
            for a, b in (
                (moved.arc_points[0], original.arc_points[0]),
                (moved.arc_points[-1], original.arc_points[-1]),
            ):
                assert max(abs(x - y) for x, y in zip(a, b)) < _WELD_TOL_A, (a, b)
            assert moved.arc_points != original.arc_points, "the arc did not bend"
        return bent

    def test_bending_the_descent_arcs_leaves_the_ring_and_degrees_alone(
        self,
    ) -> None:
        """The minimum side: the ring chains the ASCENT arcs, drawn off the bend."""
        descent = self._descent()
        arcs = [_conn(s, m, b) for s, branches in self.RING for b, m in branches]
        centers = [_center("MIN_19182", "minimum")]

        before = find_minimum_basins(arcs, centers, descent)["MIN_19182"]
        cycle_before = [
            (a.saddle_id, a.branch_index, a.max_id)
            for a in _boundary_cycle(before.boundary_arcs, lambda a: a.max_id)
        ]

        bent = self._gradify(descent)

        after = find_minimum_basins(arcs, centers, bent)["MIN_19182"]
        cycle_after = [
            (a.saddle_id, a.branch_index, a.max_id)
            for a in _boundary_cycle(after.boundary_arcs, lambda a: a.max_id)
        ]

        # The ring is the same ring, in the same order, and it serialises the same.
        assert cycle_after == cycle_before
        assert _shape(after) == _shape(before)
        assert (
            serialize_minimum_basins({"M": after})["M"]
            == (serialize_minimum_basins({"M": before})["M"])
        )

    def test_the_dual_side_keeps_its_valley_ring_too(self) -> None:
        """basin v11's maximum side, where the ring IS the bent descent table.

        The duality means the same rule runs with the tables exchanged, and this is
        the side on which the arcs that GET drawn are the ones the bend moved -- the
        minimum-side test above leaves them as scenery. Off-origin nucleus, so the
        frame is exercised on the side that renders.
        """
        center = (0.37, -1.13, 0.29)
        # A ring, not a pair. With two minima shared by two saddles, each minimum
        # is reached by ONE bounding saddle, i.e. it is a leaf: v12 prunes to empty
        # and the keep-all fallback returns the unpruned set, so a second
        # _boundary_cycle pass is NOT the identity and the comparison measures the
        # fixture instead of the bend. Four saddles draining a four-minimum cycle
        # give every target degree 2, which is the shape a real basin boundary has.
        # lat_i and lat_o have opposite signs: the arc crosses the crest, so the
        # bend has something to bend onto.
        ring_edges: Sequence[tuple[str, int, str, float, float]] = (
            ("E-SAD_MC15", 0, "MIN_A", -2.0, 2.0),
            ("E-SAD_MC15", 1, "MIN_D", 2.0, -2.0),
            ("E-SAD_MC16", 0, "MIN_A", 2.0, -2.0),
            ("E-SAD_MC16", 1, "MIN_B", -2.0, 2.0),
            ("E-SAD_MC17", 0, "MIN_B", 2.0, -2.0),
            ("E-SAD_MC17", 1, "MIN_C", -2.0, 2.0),
            ("E-SAD_MC20", 0, "MIN_C", 2.0, -2.0),
            ("E-SAD_MC20", 1, "MIN_D", -2.0, 2.0),
        )
        descent = [
            _desc_with_geometry(s, m, b, lat_i, lat_o, center=center)
            for s, b, m, lat_i, lat_o in ring_edges
        ]
        connections = [
            _conn(saddle, maximum, bi)
            for saddle, maxima in (
                ("E-SAD_MC15", ("E-MAX_MC36", "E-MAX_MC40")),
                ("E-SAD_MC16", ("E-MAX_MC36", "E-MAX_MC40")),
                ("E-SAD_MC17", ("E-MAX_MC36", "E-MAX_MC40")),
                ("E-SAD_MC20", ("E-MAX_MC36", "E-MAX_MC40")),
            )
            for bi, maximum in enumerate(maxima)
        ]
        centers = [_center("E-MAX_MC36", "maximum"), _center("E-MAX_MC40", "maximum")]

        def keys(basins: dict[str, MaximumBasin]) -> list[tuple[str, int, str | None]]:
            """E-MAX_MC36's boundary arcs, by key, ORDER-INDEPENDENT on purpose.

            `find_maximum_basins` has already run the v12 prune, so this is the
            ring. The bend is claimed to change no topology, and the cycle rule's
            walk direction is not part of a basin's identity: sets of keys say the
            claim, a sequence equality would also test the walker, which is not
            what this test is about.
            """
            return sorted(
                (a.saddle_id, a.branch_index, a.min_id)
                for a in basins["E-MAX_MC36"].boundary_arcs
            )

        before = find_maximum_basins(connections, centers, descent)
        # The fixture is a closed ring: pruning must be a no-op, i.e. this really is
        # the shape the rule was written for and not an escape hatch.
        assert keys(before) == sorted((s, b, m) for s, b, m, _, _ in ring_edges), keys(
            before
        )
        bent_keys = (("E-SAD_MC15", 0), ("E-SAD_MC20", 1))
        bent = self._gradify(
            [d for d in descent if (d.saddle_id, d.branch_index) in bent_keys],
            center=center,
        )
        # Spliced BY KEY. `d not in bent` would not do it: MinConnection is a plain
        # dataclass, its __eq__ is the generated one, and the bent copy keeps every
        # key field -- so the original of a bent arc compares EQUAL to nothing only
        # when the bend moved a vertex, and the table then carries that arc twice.
        after = find_maximum_basins(
            connections,
            centers,
            [d for d in descent if (d.saddle_id, d.branch_index) not in bent_keys]
            + bent,
        )
        assert keys(after) == keys(before)


class TestQuadFittedGeometryKeepsTheCycle:
    """The plan's edge case, for the second polish method: a fitted arc must leave
    basin v12's rings identical.

    `TestGradifiedGeometryKeepsTheCycle` proves this for the Newton walk. The
    quadratic fit moves the SAME vertices by a different rule -- one bend per span,
    welded at the pins, and it is the method that will be ON by default for the
    ladder -- so the claim needs its own proof, not a borrowed one. Self-contained
    on purpose: sharing the bend through a callable would have meant editing 19
    passing tests to add a 20th, and a topology claim that quietly stopped running
    is the exact failure this session has already been bitten by.

    The fit must ACTUALLY bend the fixture (asserted in `_fit`): a cycle test on an
    untouched table proves nothing about the bend.
    """

    RING: ClassVar[tuple[tuple[str, tuple[tuple[int, str], ...]], ...]] = (
        ("E-SAD_MC15", ((0, "E-MAX_MC36"), (1, "E-MAX_MC40"))),
        ("E-SAD_MC20", ((0, "E-MAX_MC36"), (1, "E-MAX_MC40"))),
    )
    DRAINS: ClassVar[tuple[tuple[str, int, str], ...]] = (
        ("E-SAD_MC15", 0, "MIN_19182"),
        ("E-SAD_MC20", 0, "MIN_19182"),
    )

    def _descent(self) -> list[MinConnection]:
        return [
            _desc_with_geometry(s, m, b, -0.6, 0.5 if i == 0 else -0.6)
            for i, (s, b, m) in enumerate(self.DRAINS)
        ]

    def _fit(
        self,
        descent: list[MinConnection],
        center: tuple[float, float, float] | None = None,
    ) -> list[MinConnection]:
        """Fit the descent table onto the fixture's crest, asserting it fitted."""
        pts = _ring_mesh()
        bent, stats = fit_quad_bezier_connections(
            descent,
            mesh_pts=pts,
            f_vals=_crest_ring_field(pts),
            sphere_radius=_RING_SPHERE_R,
            delta_theta_mesh_rad=_RING_PITCH,
            center=center,
        )
        assert stats["n_arcs"] == len(descent), stats
        # AT LEAST ONE, not all: the fit's gain gate (0.05% of the field's robust span)
        # is stricter than gradify's crest gate on this fixture, which is a real
        # property of a one-bend-per-span method -- an arc whose two ends already sit
        # on the same side of the crest has nothing for a single bend to buy, so the
        # gain floor refuses it and gradify's per-node walk does not. Requiring a full
        # bend here would encode the walk's criterion into the fit's test.
        assert stats["n_quad_fitted"] >= 1, (
            "the fit refused every crest that is here by construction -- the cycle "
            "test below would pass vacuously on an untouched table. stats: "
            f"{ {k: v for k, v in stats.items() if k.startswith('n_')} }"
        )
        for original, moved in zip(descent, bent):
            # Welded ends, same tolerance as the gradify class: the weld renormalizes
            # onto the sphere, which moves a coordinate by a couple of ULP. Asserted on
            # EVERY row, refusal included -- an untouched record trivially welds, so
            # the bend assertion below is what proves rows actually moved.
            for a, b in (
                (moved.arc_points[0], original.arc_points[0]),
                (moved.arc_points[-1], original.arc_points[-1]),
            ):
                assert max(abs(x - y) for x, y in zip(a, b)) < _WELD_TOL_A, (a, b)
            if moved.quad_fitted:
                assert moved.arc_points != original.arc_points, "the arc did not bend"
            else:
                assert moved.arc_points == original.arc_points
        return bent

    def test_fitting_the_descent_arcs_leaves_the_ring_and_degrees_alone(
        self,
    ) -> None:
        """The minimum side: the ring chains the ASCENT arcs, drawn off the fit."""
        descent = self._descent()
        arcs = [_conn(s, m, b) for s, branches in self.RING for b, m in branches]
        centers = [_center("MIN_19182", "minimum")]

        before = find_minimum_basins(arcs, centers, descent)["MIN_19182"]
        cycle_before = [
            (a.saddle_id, a.branch_index, a.max_id)
            for a in _boundary_cycle(before.boundary_arcs, lambda a: a.max_id)
        ]

        bent = self._fit(descent)

        after = find_minimum_basins(arcs, centers, bent)["MIN_19182"]
        cycle_after = [
            (a.saddle_id, a.branch_index, a.max_id)
            for a in _boundary_cycle(after.boundary_arcs, lambda a: a.max_id)
        ]

        assert cycle_after == cycle_before
        assert _shape(after) == _shape(before)
        assert (
            serialize_minimum_basins({"M": after})["M"]
            == (serialize_minimum_basins({"M": before})["M"])
        )

    def test_the_dual_side_keeps_its_valley_ring_too(self) -> None:
        """basin v11's maximum side, where the ring IS the fitted descent table.

        Four saddles draining a four-minimum cycle, off-origin nucleus: the same
        reasoning as the gradify class (a pruned-to-empty fixture would measure the
        fallback, not the bend), and this is the side on which the arcs that render
        are the arcs the fit moved.
        """
        center = (0.37, -1.13, 0.29)
        ring_edges: Sequence[tuple[str, int, str, float, float]] = (
            ("E-SAD_MC15", 0, "MIN_A", -2.0, 2.0),
            ("E-SAD_MC15", 1, "MIN_D", 2.0, -2.0),
            ("E-SAD_MC16", 0, "MIN_A", 2.0, -2.0),
            ("E-SAD_MC16", 1, "MIN_B", -2.0, 2.0),
            ("E-SAD_MC17", 0, "MIN_B", 2.0, -2.0),
            ("E-SAD_MC17", 1, "MIN_C", -2.0, 2.0),
            ("E-SAD_MC20", 0, "MIN_C", 2.0, -2.0),
            ("E-SAD_MC20", 1, "MIN_D", -2.0, 2.0),
        )
        descent = [
            _desc_with_geometry(s, m, b, lat_i, lat_o, center=center)
            for s, b, m, lat_i, lat_o in ring_edges
        ]
        connections = [
            _conn(saddle, maximum, bi)
            for saddle, maxima in (
                ("E-SAD_MC15", ("E-MAX_MC36", "E-MAX_MC40")),
                ("E-SAD_MC16", ("E-MAX_MC36", "E-MAX_MC40")),
                ("E-SAD_MC17", ("E-MAX_MC36", "E-MAX_MC40")),
                ("E-SAD_MC20", ("E-MAX_MC36", "E-MAX_MC40")),
            )
            for bi, maximum in enumerate(maxima)
        ]
        centers = [_center("E-MAX_MC36", "maximum"), _center("E-MAX_MC40", "maximum")]

        def keys(basins: dict[str, MaximumBasin]) -> list[tuple[str, int, str | None]]:
            return sorted(
                (a.saddle_id, a.branch_index, a.min_id)
                for a in basins["E-MAX_MC36"].boundary_arcs
            )

        before = find_maximum_basins(connections, centers, descent)
        assert keys(before) == sorted((s, b, m) for s, b, m, _, _ in ring_edges), keys(
            before
        )
        bent_keys = (("E-SAD_MC15", 0), ("E-SAD_MC20", 1))
        bent = self._fit(
            [d for d in descent if (d.saddle_id, d.branch_index) in bent_keys],
            center=center,
        )
        # Spliced BY KEY (frozen dataclass __eq__ + a moved vertex = a doubled row).
        after = find_maximum_basins(
            connections,
            centers,
            [d for d in descent if (d.saddle_id, d.branch_index) not in bent_keys]
            + bent,
        )
        assert keys(after) == keys(before)
