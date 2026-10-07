"""Unit tests for string_aggregator module."""

import math

import numpy as np
import pytest

from gba_topology2.src.micro_cluster import compute_micro_clusters
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.string_aggregator import aggregate_cluster_strings


def test_aggregate_single_string_cluster() -> None:
    """Test aggregation of an intrinsically elongated micro-cluster into a MacroString."""
    angles_x = [-0.04, -0.02, 0.0, 0.02, 0.04]
    cps = [
        CriticalPoint(
            vertex_id=i,
            cp_type="maximum" if i % 2 == 0 else "saddle",
            value=float(i),
            position=(math.sin(ax), 0.0, math.cos(ax)),
        )
        for i, ax in enumerate(angles_x)
    ]

    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.025,
        k_pitch=2.0,
        sphere_radius=1.0,
        aspect_ratio_threshold=2.0,
    )
    assert len(clusters) == 1
    assert clusters[0].morphology == "string"

    macro_strings, isolated = aggregate_cluster_strings(
        clusters=clusters,
        delta_theta_mesh_rad=0.025,
        sphere_radius=1.0,
        k_bridge=3.0,
    )

    assert len(macro_strings) == 1
    assert len(isolated) == 0

    ms = macro_strings[0]
    assert ms.string_id == "MS1"
    assert len(ms.all_members) == 5
    assert ms.n_maxima == 3
    assert ms.n_saddles == 2
    assert ms.total_saddle_index == 2
    assert ms.local_euler_index == 1  # 3 - 2 = +1
    assert ms.composition_summary == "3 Max, 2 Sad"
    assert ms.aspect_ratio > 2.0
    assert len(ms.boundary_ring) == 48

    for pt in ms.boundary_ring:
        norm_val = np.linalg.norm(np.array(pt.position))
        assert pytest.approx(norm_val, abs=1e-6) == 1.0


def test_compact_clusters_not_aggregated() -> None:
    """Test that isotropic compact clusters remain isolated and are not converted to strings."""
    # 2 isolated points separated by 45 degrees
    ang1 = math.radians(0.0)
    ang2 = math.radians(45.0)
    cps = [
        CriticalPoint(
            vertex_id=1,
            cp_type="maximum",
            value=1.0,
            position=(math.sin(ang1), 0.0, math.cos(ang1)),
        ),
        CriticalPoint(
            vertex_id=2,
            cp_type="minimum",
            value=-1.0,
            position=(math.sin(ang2), 0.0, math.cos(ang2)),
        ),
    ]

    clusters = compute_micro_clusters(
        cps=cps,
        delta_theta_mesh_rad=0.025,
        k_pitch=2.0,
        sphere_radius=1.0,
    )
    assert len(clusters) == 2
    assert all(c.morphology == "compact" for c in clusters)

    macro_strings, isolated = aggregate_cluster_strings(
        clusters=clusters,
        delta_theta_mesh_rad=0.025,
        sphere_radius=1.0,
        k_bridge=3.0,
    )

    assert len(macro_strings) == 0
    assert len(isolated) == 2


def test_input_validation() -> None:
    """Test input validation for aggregate_cluster_strings."""
    with pytest.raises(ValueError):
        aggregate_cluster_strings([], delta_theta_mesh_rad=-0.1)

    with pytest.raises(ValueError):
        aggregate_cluster_strings([], delta_theta_mesh_rad=0.1, k_bridge=0.0)

    strings, iso = aggregate_cluster_strings([], delta_theta_mesh_rad=0.1)
    assert strings == []
    assert iso == []
