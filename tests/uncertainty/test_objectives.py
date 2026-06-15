"""ObjectiveConfig + UQLossComponents contract tests.

Container patterns:

* ``ObjectiveConfig`` — plain ``@dataclass(frozen=True, slots=True,
  kw_only=True)``. Hashable static argument; validation runs in
  ``__post_init__``.
* ``UQLossComponents`` — ``@flax.struct.dataclass(slots=True,
  kw_only=True)``. Carries scalar loss arrays through ``jit``/``grad``.
  ``validate()`` is public, never called from ``__post_init__``/``tree_unflatten``.

Canonical KL helper tests live in ``tests/uncertainty/kernels/test_bayesian.py``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl
from opifex.uncertainty.objectives import (
    ObjectiveConfig,
    scale_kl,
    UQLossComponents,
)


def _make_config(**overrides: float | str | None) -> ObjectiveConfig:
    base: dict[str, float | int | str | None] = {
        "kl_weight": 1.0,
        "dataset_size": 100,
        "physics_weight": 1.0,
        "data_weight": 1.0,
        "boundary_weight": 1.0,
        "initial_condition_weight": 1.0,
        "regularization_weight": 1.0,
        "calibration_weight": 1.0,
        "conformal_weight": 1.0,
        "pac_bayes_weight": 1.0,
        "reduction": "mean",
    }
    base.update(overrides)
    return ObjectiveConfig(**base)  # type: ignore[arg-type]


def test_objective_config_is_pattern_a_frozen_slotted_kw_only_dataclass() -> None:
    assert dataclasses.is_dataclass(ObjectiveConfig)
    field_names = {f.name for f in dataclasses.fields(ObjectiveConfig)}
    assert field_names >= {
        "kl_weight",
        "dataset_size",
        "physics_weight",
        "data_weight",
        "boundary_weight",
        "initial_condition_weight",
        "regularization_weight",
        "calibration_weight",
        "conformal_weight",
        "pac_bayes_weight",
        "reduction",
    }
    assert hasattr(ObjectiveConfig, "__slots__")
    assert not hasattr(_make_config(), "__dict__")


def test_objective_config_is_hashable_static_argument() -> None:
    config = _make_config()
    same = _make_config()
    other = _make_config(kl_weight=2.0)
    assert hash(config) == hash(same)
    assert hash(config) != hash(other)


def test_objective_config_rejects_negative_weights_via_post_init() -> None:
    with pytest.raises(ValueError, match=r"weight"):
        _make_config(kl_weight=-0.1)
    with pytest.raises(ValueError, match=r"weight"):
        _make_config(physics_weight=-1.0)
    with pytest.raises(ValueError, match=r"weight"):
        _make_config(pac_bayes_weight=-0.5)


def test_objective_config_rejects_non_positive_dataset_size() -> None:
    with pytest.raises(ValueError, match=r"dataset_size"):
        _make_config(dataset_size=0)
    with pytest.raises(ValueError, match=r"dataset_size"):
        _make_config(dataset_size=-10)


def test_objective_config_accepts_none_dataset_size() -> None:
    config = _make_config(dataset_size=None)
    assert config.dataset_size is None


def test_objective_config_rejects_unknown_reduction() -> None:
    with pytest.raises(ValueError, match=r"reduction"):
        _make_config(reduction="bogus")


def test_uq_loss_components_is_pattern_b_struct_dataclass() -> None:
    components = UQLossComponents(total=jnp.array(0.0))
    assert hasattr(components, "__slots__")
    assert not hasattr(components, "__dict__")
    flat, treedef = jax.tree_util.tree_flatten(components)
    rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
    assert isinstance(rebuilt, UQLossComponents)
    assert hasattr(components, "replace")


def test_uq_loss_components_required_fields_are_present() -> None:
    field_names = {f.name for f in dataclasses.fields(UQLossComponents)}
    expected = {
        "data",
        "negative_log_likelihood",
        "physics_residual",
        "boundary",
        "initial_condition",
        "regularization",
        "kl",
        "negative_elbo",
        "calibration",
        "conformal",
        "pac_bayes",
        "total",
        "metadata",
    }
    assert expected <= field_names


def test_uq_loss_components_metadata_is_tuple_of_pairs() -> None:
    components = UQLossComponents(total=jnp.array(0.0), metadata=(("method", "elbo"), ("step", 7)))
    assert components.metadata == (("method", "elbo"), ("step", 7))
    assert components.metadata_dict() == {"method": "elbo", "step": 7}
    assert hash(components.metadata) is not None


def test_uq_loss_components_metadata_default_is_empty_tuple() -> None:
    components = UQLossComponents(total=jnp.array(0.0))
    assert components.metadata == ()
    assert components.metadata_dict() == {}


def test_uq_loss_components_from_components_weights_present_fields_only() -> None:
    config = _make_config(
        data_weight=2.0,
        physics_weight=3.0,
        kl_weight=4.0,
        regularization_weight=5.0,
    )
    components = UQLossComponents.from_components(
        config=config,
        data=jnp.array(1.0),
        physics_residual=jnp.array(0.5),
        kl=jnp.array(2.0),
        regularization=jnp.array(0.25),
    )
    expected_total = 2.0 * 1.0 + 3.0 * 0.5 + (4.0 * 2.0 / 100) + 5.0 * 0.25
    assert float(components.total) == pytest.approx(expected_total)


def test_uq_loss_components_from_components_missing_components_are_zero() -> None:
    config = _make_config()
    components = UQLossComponents.from_components(
        config=config,
        data=jnp.array(0.5),
    )
    assert float(components.total) == pytest.approx(0.5)
    assert jnp.isfinite(components.total)


def test_uq_loss_components_elbo_negates_negative_elbo() -> None:
    components = UQLossComponents(total=jnp.array(0.0), negative_elbo=jnp.array(3.5))
    elbo = components.elbo
    assert elbo is not None
    assert float(elbo) == pytest.approx(-3.5)


def test_uq_loss_components_elbo_is_none_when_no_negative_elbo() -> None:
    components = UQLossComponents(total=jnp.array(0.0))
    assert components.elbo is None


def test_scale_kl_divides_by_dataset_size_when_positive() -> None:
    config = _make_config(dataset_size=200)
    scaled = scale_kl(jnp.array(40.0), config)
    assert float(scaled) == pytest.approx(40.0 / 200)


def test_scale_kl_returns_kl_unchanged_when_dataset_size_is_none() -> None:
    config_with_none = _make_config(dataset_size=None)
    assert float(scale_kl(jnp.array(7.0), config_with_none)) == pytest.approx(7.0)


def test_value_and_grad_works_with_uq_loss_components_as_aux() -> None:
    config = _make_config()

    def loss_fn(params: jax.Array) -> tuple[jax.Array, UQLossComponents]:
        mean = params
        logvar = jnp.zeros_like(params)
        kl = diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=1.0)
        components = UQLossComponents.from_components(config=config, kl=kl)
        return components.total, components

    params = jnp.array([1.0, -0.5, 0.25])
    (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert isinstance(aux, UQLossComponents)
    assert jnp.isfinite(total)
    assert grads.shape == params.shape
    assert jnp.all(jnp.isfinite(grads))
