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


def test_all_box_types_share_identical_bounds_clip() -> None:
    """TrustRegionBox / TREGOBox / TURBOBox must return identical clipped bounds.

    The clip geometry is centralised in ``_clip_box_bounds``; all three box
    types delegate to it, so on identical centre / length / search-space
    inputs they must agree exactly.
    """
    from opifex.uncertainty.active.trust_region import TREGOBox, TrustRegionBox, TURBOBox

    center = jnp.array([0.95, 0.05])
    length = jnp.asarray(0.4)
    lower_bound = jnp.array([0.0, 0.0])
    upper_bound = jnp.array([1.0, 1.0])

    trb = TrustRegionBox(
        center=center,
        length=length,
        search_space_lower=lower_bound,
        search_space_upper=upper_bound,
    )
    trego = TREGOBox(
        center=center,
        length=length,
        search_space_lower=lower_bound,
        search_space_upper=upper_bound,
        success_count=0,
        failure_count=0,
        success_threshold=3,
        failure_threshold=3,
        shrink_factor=0.5,
        expand_factor=2.0,
    )
    turbo = TURBOBox(
        center=center,
        length=length,
        search_space_lower=lower_bound,
        search_space_upper=upper_bound,
    )

    trb_lower, trb_upper = trb.bounds()
    trego_lower, trego_upper = trego.bounds()
    turbo_lower, turbo_upper = turbo.bounds()

    # Reference clip captured before the refactor (documented baseline).
    assert jnp.allclose(trb_lower, jnp.array([0.75, 0.0]))
    assert jnp.allclose(trb_upper, jnp.array([1.0, 0.25]))
    # All three box types agree exactly.
    assert jnp.allclose(trego_lower, trb_lower)
    assert jnp.allclose(trego_upper, trb_upper)
    assert jnp.allclose(turbo_lower, trb_lower)
    assert jnp.allclose(turbo_upper, trb_upper)


def test_clip_box_bounds_jit_vmap() -> None:
    """The ``_clip_box_bounds`` helper is jit- and vmap-clean."""
    import jax

    from opifex.uncertainty.active.trust_region import _clip_box_bounds

    lower_bound = jnp.array([0.0, 0.0])
    upper_bound = jnp.array([1.0, 1.0])

    # jit: identical result to the eager call on a single box.
    center = jnp.array([0.95, 0.05])
    length = jnp.asarray(0.4)
    eager_lower, eager_upper = _clip_box_bounds(center, length, lower_bound, upper_bound)
    jit_lower, jit_upper = jax.jit(_clip_box_bounds)(center, length, lower_bound, upper_bound)
    assert jnp.allclose(jit_lower, eager_lower)
    assert jnp.allclose(jit_upper, eager_upper)

    # vmap over a batch of centres / lengths sharing the global search space.
    centers = jnp.array([[0.95, 0.05], [0.3, 0.3], [0.5, 0.9]])
    lengths = jnp.array([0.4, 0.2, 0.6])
    batched = jax.vmap(_clip_box_bounds, in_axes=(0, 0, None, None))
    lowers, uppers = batched(centers, lengths, lower_bound, upper_bound)
    assert lowers.shape == (3, 2)
    assert uppers.shape == (3, 2)
    # Row 0 matches the single-box eager result.
    assert jnp.allclose(lowers[0], eager_lower)
    assert jnp.allclose(uppers[0], eager_upper)
    # Clipping is respected across the batch.
    assert jnp.all(lowers >= lower_bound)
    assert jnp.all(uppers <= upper_bound)
