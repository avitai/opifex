"""Mixed precision composes flax's dynamic loss scaling; opifex keeps the policy and the stats.

``MixedPrecisionComponent`` casts inputs to the compute dtype and differentiates
through ``flax.training.dynamic_scale.DynamicScale``: gradients come back
unscaled, a non-finite step reports ``is_finite=False`` and backs the scale off,
and a run of finite steps grows it. No hand-rolled overflow bookkeeping remains.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from flax.training.dynamic_scale import DynamicScale

from opifex.core.training.components import MixedPrecisionComponent


class _Linear(nnx.Module):
    def __init__(self) -> None:
        self.layer = nnx.Linear(4, 1, rngs=nnx.Rngs(0))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layer(x)


def _loss(model: _Linear, x: jax.Array) -> jax.Array:
    return jnp.mean(model(x) ** 2)


def test_component_owns_a_dynamic_scale_built_from_its_config() -> None:
    component = MixedPrecisionComponent(
        {"loss_scale": 1024.0, "growth_interval": 5, "growth_factor": 4.0, "backoff_factor": 0.25}
    )

    assert isinstance(component.dynamic_scale, DynamicScale)
    assert float(component.dynamic_scale.scale) == 1024.0
    assert component.dynamic_scale.growth_interval == 5
    assert component.dynamic_scale.growth_factor == 4.0
    assert component.dynamic_scale.backoff_factor == 0.25
    assert component.loss_scale == 1024.0


def test_policy_casts_parameter_dtype_arrays_to_the_compute_dtype() -> None:
    component = MixedPrecisionComponent({"compute_dtype": jnp.bfloat16, "param_dtype": jnp.float32})
    policy = component.create_precision_policy()

    assert policy(jnp.ones(3, dtype=jnp.float32)).dtype == jnp.bfloat16
    assert policy(jnp.ones(3, dtype=jnp.int32)).dtype == jnp.int32


def test_value_and_grad_returns_unscaled_finite_gradients() -> None:
    component = MixedPrecisionComponent({"loss_scale": 256.0})
    model = _Linear()
    x = jnp.ones((8, 4))
    expected_loss, expected_grads = nnx.value_and_grad(_loss)(model, x)

    is_finite, loss, grads = component.value_and_grad(_loss)(model, x)

    assert bool(is_finite)
    assert loss == pytest.approx(float(expected_loss), rel=1e-5)
    got = jax.tree.leaves(grads)
    want = jax.tree.leaves(expected_grads)
    assert all(jnp.allclose(g, w, rtol=1e-4, atol=1e-6) for g, w in zip(got, want, strict=True))


def test_scale_backs_off_on_a_non_finite_step_and_grows_after_finite_ones() -> None:
    component = MixedPrecisionComponent(
        {"loss_scale": 1024.0, "growth_interval": 2, "growth_factor": 2.0, "backoff_factor": 0.5}
    )
    model = _Linear()

    def exploding(model: _Linear, x: jax.Array) -> jax.Array:
        return _loss(model, x) * jnp.inf

    is_finite, _, _ = component.value_and_grad(exploding)(model, jnp.ones((2, 4)))
    assert not bool(is_finite)
    assert component.loss_scale == pytest.approx(512.0)
    assert component.overflow_count == 1

    # flax grows on the finite step that follows ``growth_interval`` finite steps.
    for _ in range(2):
        is_finite, _, _ = component.value_and_grad(_loss)(model, jnp.ones((2, 4)))
        assert bool(is_finite)
    assert component.loss_scale == pytest.approx(512.0)
    component.value_and_grad(_loss)(model, jnp.ones((2, 4)))
    assert component.loss_scale == pytest.approx(1024.0)
    assert component.step_count == 4


def test_static_scaling_never_moves_the_scale() -> None:
    component = MixedPrecisionComponent({"loss_scale": 64.0, "dynamic_loss_scaling": False})
    model = _Linear()

    component.value_and_grad(lambda m, x: _loss(m, x) * jnp.nan)(model, jnp.ones((2, 4)))
    for _ in range(3):
        component.value_and_grad(_loss)(model, jnp.ones((2, 4)))

    assert component.loss_scale == 64.0


def test_setup_resets_the_scale_and_stats() -> None:
    component = MixedPrecisionComponent({"loss_scale": 32.0, "growth_interval": 1})
    model = _Linear()
    for _ in range(2):  # one finite step, then the growth step
        component.value_and_grad(_loss)(model, jnp.ones((2, 4)))
    assert component.loss_scale == 64.0

    component.setup(model, training_state=None)

    assert component.loss_scale == 32.0
    assert component.step_count == 0
    assert component.overflow_count == 0


def test_stats_report_scale_dtypes_and_counts() -> None:
    component = MixedPrecisionComponent({"loss_scale": 8.0})

    assert component.get_mixed_precision_stats() == {
        "loss_scale": 8.0,
        "overflow_count": 0,
        "step_count": 0,
        "compute_dtype": str(component.compute_dtype),
        "param_dtype": str(component.param_dtype),
    }


def test_rejects_a_string_dtype() -> None:
    with pytest.raises(TypeError, match="dtype"):
        MixedPrecisionComponent({"compute_dtype": "invalid_dtype"})


def test_the_hand_rolled_loss_scaling_is_gone() -> None:
    for module in (
        "opifex.core.training.strategies.mixed_precision",
        "opifex.core.training.strategies.mixed_precision_ops",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    components = importlib.import_module("opifex.core.training.components")
    assert not hasattr(components, "MixedPrecisionState")
    assert not hasattr(MixedPrecisionComponent, "scale_gradients")
    assert not hasattr(MixedPrecisionComponent, "update_loss_scale")
