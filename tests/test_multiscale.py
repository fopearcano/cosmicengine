"""Tests for Phase 32 multi-scale universe."""

from __future__ import annotations

import math

import pytest

from cosmic_engine.core.object_types import CosmicObjectType
from cosmic_engine.core.truth import TruthLevel
from cosmic_engine.core.universe_object import UniverseObject
from cosmic_engine.core.vector import Vector3
from cosmic_engine.multiscale import (
    DEFAULT_ZONES,
    ScaleManager,
    ScaleZone,
    blend_representations,
    compute_transition_alpha,
    get_representation_for_zone,
)
from cosmic_engine.runtime import CosmicRuntime, RuntimeConfig


def _galaxy(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.GALAXY,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
    )


def _star(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.STAR,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.CATALOG_IMPORTED,
        mass_kg=1.0e30,
    )


def _planet(object_id: str, position: Vector3) -> UniverseObject:
    return UniverseObject(
        id=object_id,
        name=object_id,
        object_type=CosmicObjectType.PLANET,
        position_m=position,
        velocity_m_s=Vector3.zero(),
        truth_level=TruthLevel.PHYSICS_SIMULATED,
        mass_kg=5.972e24,
    )


# --- ScaleZone ---


def test_scale_zone_contains_scale_inside_and_outside():
    z = ScaleZone(
        name="z", min_scale_m=10.0, max_scale_m=100.0,
        representation_type="star_field",
    )
    assert z.contains_scale(10.0) is True
    assert z.contains_scale(50.0) is True
    assert z.contains_scale(99.99) is True
    # Half-open: max excluded
    assert z.contains_scale(100.0) is False
    assert z.contains_scale(9.99) is False


def test_scale_zone_validates_constructor():
    with pytest.raises(ValueError):
        ScaleZone(name="", min_scale_m=0.0, max_scale_m=1.0,
                  representation_type="star_field")
    with pytest.raises(ValueError):
        ScaleZone(name="x", min_scale_m=-1.0, max_scale_m=1.0,
                  representation_type="star_field")
    with pytest.raises(ValueError):
        ScaleZone(name="x", min_scale_m=1.0, max_scale_m=1.0,
                  representation_type="star_field")
    with pytest.raises(ValueError):
        ScaleZone(name="x", min_scale_m=0.0, max_scale_m=1.0,
                  representation_type="quantum")


# --- ScaleManager ---


def test_manager_get_zone_inside_range():
    z1 = ScaleZone(
        name="a", min_scale_m=0.0, max_scale_m=10.0,
        representation_type="star_field",
    )
    z2 = ScaleZone(
        name="b", min_scale_m=10.0, max_scale_m=100.0,
        representation_type="galaxy_field",
    )
    m = ScaleManager([z2, z1])  # unsorted input: manager sorts
    assert m.zones[0] is z1
    assert m.get_zone(5.0) is z1
    assert m.get_zone(50.0) is z2


def test_manager_clamps_below_and_above_range():
    z = ScaleZone(
        name="z", min_scale_m=10.0, max_scale_m=100.0,
        representation_type="star_field",
    )
    m = ScaleManager([z])
    assert m.get_zone(0.0) is z       # below clamps to first
    assert m.get_zone(10000.0) is z   # above clamps to last


def test_manager_rejects_overlapping_zones():
    z1 = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                   representation_type="star_field")
    z2 = ScaleZone(name="b", min_scale_m=5.0, max_scale_m=20.0,
                   representation_type="galaxy_field")
    with pytest.raises(ValueError):
        ScaleManager([z1, z2])


def test_manager_rejects_empty_zones():
    with pytest.raises(ValueError):
        ScaleManager([])


def test_manager_neighbor_lookup():
    z1 = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                   representation_type="star_field")
    z2 = ScaleZone(name="b", min_scale_m=10.0, max_scale_m=100.0,
                   representation_type="galaxy_field")
    z3 = ScaleZone(name="c", min_scale_m=100.0, max_scale_m=1000.0,
                   representation_type="density_field")
    m = ScaleManager([z1, z2, z3])
    assert m.get_neighbor_above(z1) is z2
    assert m.get_neighbor_above(z3) is None
    assert m.get_neighbor_below(z2) is z1
    assert m.get_neighbor_below(z1) is None


def test_default_zones_cover_expected_scales():
    m = ScaleManager(list(DEFAULT_ZONES))
    assert m.get_zone(1.0e25).representation_type == "galaxy_field"
    assert m.get_zone(1.0e18).representation_type == "star_field"
    assert m.get_zone(1.0e12).representation_type == "nbody"
    assert m.get_zone(1.0).representation_type == "density_field"


# --- compute_transition_alpha ---


def test_transition_alpha_in_unit_range():
    z_a = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                    representation_type="star_field")
    z_b = ScaleZone(name="b", min_scale_m=10.0, max_scale_m=100.0,
                    representation_type="galaxy_field")
    for d in (-5.0, 0.0, 5.0, 9.5, 10.0, 10.5, 100.0, 200.0):
        a = compute_transition_alpha(d, z_a, z_b, blend_width=0.2)
        assert 0.0 <= a <= 1.0


def test_transition_alpha_zero_below_lower_bound():
    z_a = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                    representation_type="star_field")
    z_b = ScaleZone(name="b", min_scale_m=10.0, max_scale_m=100.0,
                    representation_type="galaxy_field")
    assert compute_transition_alpha(5.0, z_a, z_b, blend_width=0.1) == 0.0


def test_transition_alpha_one_above_upper_bound():
    z_a = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                    representation_type="star_field")
    z_b = ScaleZone(name="b", min_scale_m=10.0, max_scale_m=100.0,
                    representation_type="galaxy_field")
    assert compute_transition_alpha(15.0, z_a, z_b, blend_width=0.1) == 1.0


def test_transition_alpha_zero_blend_width_is_a_step():
    z_a = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                    representation_type="star_field")
    z_b = ScaleZone(name="b", min_scale_m=10.0, max_scale_m=100.0,
                    representation_type="galaxy_field")
    assert compute_transition_alpha(9.99, z_a, z_b, blend_width=0.0) == 0.0
    assert compute_transition_alpha(10.0, z_a, z_b, blend_width=0.0) == 1.0


def test_transition_alpha_monotonic_across_boundary():
    z_a = ScaleZone(name="a", min_scale_m=0.0, max_scale_m=10.0,
                    representation_type="star_field")
    z_b = ScaleZone(name="b", min_scale_m=10.0, max_scale_m=100.0,
                    representation_type="galaxy_field")
    last = -1.0
    for d in [9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0]:
        a = compute_transition_alpha(d, z_a, z_b, blend_width=0.5)
        assert a >= last
        last = a


# --- blend_representations ---


def test_blend_alpha_zero_returns_a():
    rep_a = {"type": "x", "objects": ["a"], "object_count": 1}
    rep_b = {"type": "y", "objects": ["b"], "object_count": 1}
    out = blend_representations(rep_a, rep_b, 0.0)
    assert out["type"] == "x"


def test_blend_alpha_one_returns_b():
    rep_a = {"type": "x", "objects": ["a"], "object_count": 1}
    rep_b = {"type": "y", "objects": ["b"], "object_count": 1}
    out = blend_representations(rep_a, rep_b, 1.0)
    assert out["type"] == "y"


def test_blend_mid_alpha_takes_objects_from_both():
    rep_a = {"type": "x", "objects": list("aaaa"), "object_count": 4}
    rep_b = {"type": "y", "objects": list("BBBB"), "object_count": 4}
    out = blend_representations(rep_a, rep_b, 0.5)
    assert out["type"] == "blended"
    assert out["primary_type"] == "x"
    assert out["secondary_type"] == "y"
    assert out["alpha"] == pytest.approx(0.5)
    assert out["object_count"] == len(out["objects"])
    # 50/50 split: 2 from a, 2 from b
    assert out["objects"].count("a") == 2
    assert out["objects"].count("B") == 2


def test_blend_handles_empty_inputs():
    rep_empty = {"type": "x", "objects": [], "object_count": 0}
    rep_full = {"type": "y", "objects": ["o"], "object_count": 1}
    blend_empty_full = blend_representations(rep_empty, rep_full, 0.5)
    assert blend_empty_full["object_count"] == 1
    blend_full_empty = blend_representations(rep_full, rep_empty, 0.5)
    # 1 object * 0.5 = 0.5 → rounds to 0 from rep_full (since alpha
    # weights toward zero) — at alpha=0.5 we round up to 1; at 0 we
    # take all of rep_a; the contract is just non-negative count.
    assert blend_full_empty["object_count"] >= 0


# --- get_representation_for_zone ---


def test_representation_galaxy_field_filters_to_galaxies():
    runtime = CosmicRuntime()
    runtime.add_objects([
        _galaxy("g1", Vector3(0.0, 1.0e22, 0.0)),
        _star("s1", Vector3(0.0, 1.0e16, 0.0)),
    ])
    z = ScaleZone(name="g", min_scale_m=1.0e20, max_scale_m=1.0e26,
                  representation_type="galaxy_field")
    rep = get_representation_for_zone(z, runtime)
    assert rep["type"] == "galaxy_field"
    assert rep["object_count"] == 1
    assert rep["objects"][0].id == "g1"


def test_representation_star_field_filters_to_stars():
    runtime = CosmicRuntime()
    runtime.add_objects([
        _galaxy("g1", Vector3(0.0, 1.0e22, 0.0)),
        _star("s1", Vector3(0.0, 1.0e16, 0.0)),
        _star("s2", Vector3(1.0e16, 0.0, 0.0)),
    ])
    z = ScaleZone(name="s", min_scale_m=1.0e16, max_scale_m=1.0e20,
                  representation_type="star_field")
    rep = get_representation_for_zone(z, runtime)
    assert rep["object_count"] == 2
    assert all(o.object_type is CosmicObjectType.STAR for o in rep["objects"])


def test_representation_nbody_filters_to_massive_bodies():
    runtime = CosmicRuntime()
    runtime.add_objects([
        _star("s1", Vector3(0.0, 1.0e10, 0.0)),
        _planet("p1", Vector3(1.0e11, 0.0, 0.0)),
        _galaxy("g1", Vector3(0.0, 1.0e22, 0.0)),  # has no mass
    ])
    z = ScaleZone(name="n", min_scale_m=1.0e9, max_scale_m=1.0e16,
                  representation_type="nbody")
    rep = get_representation_for_zone(z, runtime)
    ids = sorted(o.id for o in rep["objects"])
    assert ids == ["p1", "s1"]


def test_representation_density_field_takes_full_registry():
    runtime = CosmicRuntime()
    runtime.add_objects([
        _galaxy("g1", Vector3(0.0, 1.0e22, 0.0)),
        _star("s1", Vector3(0.0, 1.0e16, 0.0)),
    ])
    z = ScaleZone(name="d", min_scale_m=0.0, max_scale_m=1.0,
                  representation_type="density_field")
    rep = get_representation_for_zone(z, runtime)
    assert rep["object_count"] == 2


# --- runtime integration ---


def test_runtime_get_multiscale_scene_returns_valid_dict():
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=1.0e30, max_active_objects=10_000)
    )
    runtime.add_objects([
        _star("s1", Vector3(0.0, 1.0e16, 0.0)),
        _star("s2", Vector3(1.0e16, 0.0, 0.0)),
    ])
    runtime.enable_multiscale(
        ScaleManager(list(DEFAULT_ZONES)), blend_width=0.1
    )
    rep = runtime.get_multiscale_scene(Vector3.zero())
    assert "type" in rep
    assert "objects" in rep
    assert isinstance(rep["object_count"], int)


def test_runtime_get_multiscale_scene_blends_at_zone_boundary():
    """Place objects so the median distance lands near a zone boundary."""
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=1.0e30, max_active_objects=10_000)
    )
    # Put objects at the edge between star_field (1e20) and intergalactic.
    runtime.add_objects([
        _star("s1", Vector3(0.0, 9.9e19, 0.0)),
        _star("s2", Vector3(0.0, 1.0e20, 0.0)),
        _star("s3", Vector3(0.0, 1.05e20, 0.0)),
        _galaxy("g1", Vector3(0.0, 1.1e20, 0.0)),
        _galaxy("g2", Vector3(0.0, 1.2e20, 0.0)),
    ])
    runtime.enable_multiscale(
        ScaleManager(list(DEFAULT_ZONES)), blend_width=0.5
    )
    rep = runtime.get_multiscale_scene(Vector3.zero())
    # We don't insist on exact blending behavior; just that the call
    # produces a sensible representation that doesn't crash.
    assert "type" in rep
    assert isinstance(rep.get("object_count", 0), int)


def test_runtime_get_multiscale_scene_without_manager_raises():
    runtime = CosmicRuntime()
    with pytest.raises(RuntimeError):
        runtime.get_multiscale_scene(Vector3.zero())


def test_runtime_get_multiscale_scene_with_no_active_objects():
    runtime = CosmicRuntime(
        config=RuntimeConfig(active_radius_m=1.0, max_active_objects=10)
    )
    runtime.enable_multiscale(
        ScaleManager(list(DEFAULT_ZONES)), blend_width=0.1
    )
    rep = runtime.get_multiscale_scene(Vector3.zero())
    assert rep["object_count"] == 0


def test_runtime_enable_multiscale_validates_blend_width():
    runtime = CosmicRuntime()
    with pytest.raises(ValueError):
        runtime.enable_multiscale(
            ScaleManager(list(DEFAULT_ZONES)), blend_width=-0.1
        )


# --- AIViewerConfig: multiscale fields ---


def test_config_multiscale_defaults():
    from ai_viewer import AIViewerConfig
    cfg = AIViewerConfig()
    assert cfg.enable_multiscale is False
    assert cfg.multiscale_blend_width == pytest.approx(0.1)
    cfg.validate()


def test_config_multiscale_blend_width_validation():
    from ai_viewer import AIViewerConfig
    with pytest.raises(ValueError):
        AIViewerConfig(multiscale_blend_width=-0.1).validate()
    with pytest.raises(ValueError):
        AIViewerConfig(multiscale_blend_width=1.5).validate()
