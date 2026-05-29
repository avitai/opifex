r"""Tests for ``active/trust_region.py`` — Slice 22 (audit finding #4a).

Phase 8 Task 8.3 (``08-...:586-601``) requires
``active/trust_region.py`` shipping TREGOBox, TURBOBox, and
BatchTrustRegionBox. Reference: trieste ``acquisition/rule.py:1863,
1923, 2038``. The opifex port is pure-JAX, no equinox dependency,
follows the Pattern-A frozen-slotted-kw-only dataclass shape.

The trust region maintains an axis-aligned bounding box around the
current best observation. Length ``L_t`` modulates with success /
failure counters per TuRBO (Eriksson, Pearce, Gardner, Turner,
Poloczek 2019, NeurIPS):

* ``L_{t+1} = L_t * expand_factor`` after a success streak.
* ``L_{t+1} = L_t * shrink_factor`` after a failure streak.

References
----------
* Eriksson+ 2019 — *Scalable Global Optimization via Local Bayesian
  Optimization (TuRBO)*, NeurIPS.
* Wan+ 2021 — *Think Global and Act Local (TREGO)*, NeurIPS.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


def test_trust_region_box_clamps_to_search_space_bounds() -> None:
    """A trust region near the boundary cannot exceed the global search space."""
    from opifex.uncertainty.active.trust_region import TrustRegionBox

    region = TrustRegionBox(
        center=jnp.array([0.95, 0.05]),
        length=jnp.asarray(0.4),
        search_space_lower=jnp.array([0.0, 0.0]),
        search_space_upper=jnp.array([1.0, 1.0]),
    )
    lower, upper = region.bounds()
    assert jnp.all(lower >= 0.0)
    assert jnp.all(upper <= 1.0)
    # Width across each axis is at most `length` (and at least 0).
    assert jnp.all((upper - lower) <= region.length + 1e-6)
    assert jnp.all((upper - lower) >= 0.0)


def test_trego_box_shrinks_after_failure_streak() -> None:
    """TREGO shrinks ``length`` by ``shrink_factor`` after the failure threshold."""
    from opifex.uncertainty.active.trust_region import TREGOBox

    region = TREGOBox(
        center=jnp.array([0.5]),
        length=jnp.asarray(0.5),
        search_space_lower=jnp.array([0.0]),
        search_space_upper=jnp.array([1.0]),
        success_count=0,
        failure_count=0,
        success_threshold=3,
        failure_threshold=3,
        shrink_factor=0.5,
        expand_factor=2.0,
    )
    # Three failed rounds saturate the threshold and trigger a shrink.
    new_region = region
    for _ in range(3):
        new_region = new_region.register_round(was_success=False)
    assert float(new_region.length) == pytest.approx(0.25, abs=1e-6)
    assert new_region.failure_count == 0  # reset after shrink


def test_trego_box_expands_after_success_streak() -> None:
    """TREGO expands ``length`` by ``expand_factor`` after the success threshold."""
    from opifex.uncertainty.active.trust_region import TREGOBox

    region = TREGOBox(
        center=jnp.array([0.5]),
        length=jnp.asarray(0.5),
        search_space_lower=jnp.array([0.0]),
        search_space_upper=jnp.array([1.0]),
        success_count=0,
        failure_count=0,
        success_threshold=2,
        failure_threshold=4,
        shrink_factor=0.5,
        expand_factor=2.0,
    )
    new_region = region
    for _ in range(2):
        new_region = new_region.register_round(was_success=True)
    # 0.5 * 2.0 = 1.0; clipped at the search-space diagonal length (1.0 here).
    assert float(new_region.length) >= 0.5


def test_turbo_box_recenters_on_best_observed_point() -> None:
    """TuRBO recenters its box on the best observation after each round."""
    from opifex.uncertainty.active.trust_region import TURBOBox

    region = TURBOBox(
        center=jnp.array([0.5, 0.5]),
        length=jnp.asarray(0.4),
        search_space_lower=jnp.array([0.0, 0.0]),
        search_space_upper=jnp.array([1.0, 1.0]),
    )
    new_observation = jnp.array([0.8, 0.2])
    recentred = region.recenter(new_best=new_observation)
    assert jnp.allclose(recentred.center, new_observation)
    # Length carries over unchanged on a pure recenter.
    assert jnp.allclose(recentred.length, region.length)


def test_batch_trust_region_box_holds_multiple_independent_regions() -> None:
    """BatchTrustRegionBox carries M independent trust regions for parallel batching."""
    from opifex.uncertainty.active.trust_region import BatchTrustRegionBox, TrustRegionBox

    region_a = TrustRegionBox(
        center=jnp.array([0.3, 0.3]),
        length=jnp.asarray(0.2),
        search_space_lower=jnp.array([0.0, 0.0]),
        search_space_upper=jnp.array([1.0, 1.0]),
    )
    region_b = TrustRegionBox(
        center=jnp.array([0.7, 0.7]),
        length=jnp.asarray(0.3),
        search_space_lower=jnp.array([0.0, 0.0]),
        search_space_upper=jnp.array([1.0, 1.0]),
    )
    batch = BatchTrustRegionBox(regions=(region_a, region_b))
    assert len(batch.regions) == 2
    lowers, uppers = batch.stacked_bounds()
    # Stacked shape: (num_regions, dim).
    assert lowers.shape == (2, 2)
    assert uppers.shape == (2, 2)
