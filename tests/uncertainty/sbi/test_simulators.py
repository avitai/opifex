"""Tests for the :mod:`opifex.uncertainty.sbi.simulators` module.

``Simulator`` is the entry point for every SBI estimator. It bundles a
prior sampler, a forward simulator, and an optional summary function. The
container follows GUIDE_ALIGNMENT pattern (A) — plain
``@dataclass(frozen=True, slots=True, kw_only=True)`` because it carries
only callables and static metadata, never arrays.

The :func:`sample_joint` helper returns a Datarax ``Batch`` where each
element pairs a parameter sample ``theta`` with its observation ``x``.
This matches the plan's requirement that simulator output containers be
``Batch[Element]`` with ``(theta, x)`` element pairs.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from opifex.uncertainty.sbi.simulators import sample_joint, Simulator


def _prior_sampler(rng: jax.Array, num_samples: int) -> jax.Array:
    return jax.random.normal(rng, (num_samples, 2))


def _simulate_fn(rng: jax.Array, theta: jax.Array) -> jax.Array:
    noise = jax.random.normal(rng, theta.shape)
    return theta + 0.1 * noise


def _summary_fn(x: jax.Array) -> jax.Array:
    # Compress to the sample mean (per-batch).
    return jnp.mean(x, axis=-1, keepdims=True)


def test_simulator_is_frozen_slots_kw_only_dataclass() -> None:
    """Pattern (A): frozen + slots + kw_only dataclass without array data."""
    sim = Simulator(
        prior_sampler=_prior_sampler,
        simulate_fn=_simulate_fn,
        metadata=(("kind", "gaussian"),),
    )

    assert dataclasses.is_dataclass(sim)
    params = dataclasses.fields(sim)
    assert {f.name for f in params} >= {
        "prior_sampler",
        "simulate_fn",
        "summary_fn",
        "metadata",
    }
    # frozen: assignment forbidden
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        sim.metadata = (("mutated", True),)  # type: ignore[misc]
    # slots: no __dict__
    assert not hasattr(sim, "__dict__")


def test_simulator_metadata_is_hashable_tuple_of_pairs() -> None:
    sim = Simulator(
        prior_sampler=_prior_sampler,
        simulate_fn=_simulate_fn,
        metadata=(("kind", "gaussian"), ("dim", 2)),
    )
    # Hashable — JIT cache key compatibility (GUIDE_ALIGNMENT §5a).
    assert hash(sim.metadata) == hash((("kind", "gaussian"), ("dim", 2)))
    assert sim.metadata_dict() == {"kind": "gaussian", "dim": 2}


def test_simulator_validate_rejects_non_callable_prior() -> None:
    sim = Simulator(
        prior_sampler="not-callable",  # type: ignore[arg-type]
        simulate_fn=_simulate_fn,
    )
    with pytest.raises(TypeError, match="prior_sampler"):
        sim.validate()


def test_simulator_validate_rejects_non_callable_simulator() -> None:
    sim = Simulator(
        prior_sampler=_prior_sampler,
        simulate_fn=42,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="simulate_fn"):
        sim.validate()


def test_sample_joint_validates_prior_sample_shape() -> None:
    def bad_prior(_rng: jax.Array, _num: int) -> jax.Array:
        return jnp.zeros(())

    sim = Simulator(prior_sampler=bad_prior, simulate_fn=_simulate_fn)
    with pytest.raises(ValueError, match="prior_sampler must return"):
        sample_joint(sim, num_simulations=4, rngs=nnx.Rngs(sbi_simulate=0))


def test_sample_joint_validates_simulator_output_shape() -> None:
    # A simulator that drops one row provokes a shape mismatch.
    def truly_bad(_rng: jax.Array, theta: jax.Array) -> jax.Array:
        return theta[:-1]

    sim_bad = Simulator(prior_sampler=_prior_sampler, simulate_fn=truly_bad)
    with pytest.raises(ValueError, match="leading axis"):
        sample_joint(sim_bad, num_simulations=4, rngs=nnx.Rngs(sbi_simulate=0))


def test_sample_joint_returns_batch_of_theta_x_element_pairs() -> None:
    sim = Simulator(prior_sampler=_prior_sampler, simulate_fn=_simulate_fn)
    batch = sample_joint(sim, num_simulations=8, rngs=nnx.Rngs(sbi_simulate=0))

    assert isinstance(batch, Batch)
    assert batch.batch_size == 8
    # Access the stacked theta / x arrays through the canonical NNX Variable
    # surface. (Datarax 0.1.3 ships a broken ``get_element`` for plain dict
    # data — bypass it by reading the stacked tree.)
    data = batch.data.value
    assert "theta" in data
    assert "x" in data
    assert data["theta"].shape == (8, 2)
    assert data["x"].shape == (8, 2)
    # Reference Element class explicitly so the imported symbol stays
    # used (the element-pair semantics are part of the public contract).
    assert Element is not None


def test_simulator_summary_fn_compresses_observation() -> None:
    sim = Simulator(
        prior_sampler=_prior_sampler,
        simulate_fn=_simulate_fn,
        summary_fn=_summary_fn,
    )
    batch = sample_joint(sim, num_simulations=4, rngs=nnx.Rngs(sbi_simulate=0))
    # ``summary_fn`` compresses the per-event observation; verify trailing
    # axis collapsed.
    assert batch.data.value["x"].shape == (4, 1)


def test_sample_joint_deterministic_under_fixed_key() -> None:
    sim = Simulator(prior_sampler=_prior_sampler, simulate_fn=_simulate_fn)
    a = sample_joint(sim, num_simulations=4, rngs=nnx.Rngs(sbi_simulate=42))
    b = sample_joint(sim, num_simulations=4, rngs=nnx.Rngs(sbi_simulate=42))
    assert jnp.allclose(a.data.value["theta"], b.data.value["theta"])
    assert jnp.allclose(a.data.value["x"], b.data.value["x"])


def test_sample_joint_requires_named_rng_stream() -> None:
    sim = Simulator(prior_sampler=_prior_sampler, simulate_fn=_simulate_fn)
    with pytest.raises(ValueError, match="sbi_simulate"):
        sample_joint(sim, num_simulations=4, rngs=nnx.Rngs(other_stream=0))
