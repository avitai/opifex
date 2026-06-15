"""Tests for the PAC-Bayes training objective.

These tests pin the contract from Task 8.1 TDD step #1: the objective
collapses to the Phase-1 ELBO scaling ``R_hat + KL / n`` exactly when
``delta == 1`` (the limit where the confidence-correction term vanishes).
For ``delta in (0, 1)`` the objective must match the McAllester bound from
Dziugaite & Roy (2017) / Alquier (2024).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.pac_bayes.bounds import mcallester_bound
from opifex.uncertainty.pac_bayes.objectives import pac_bayes_kl_objective


# ---- delta validation -------------------------------------------------------


@pytest.mark.parametrize("bad_delta", [0.0, -0.5, 1.5])
def test_pac_bayes_objective_rejects_delta_outside_open_closed_interval(
    bad_delta: float,
) -> None:
    with pytest.raises(ValueError, match=r"delta"):
        pac_bayes_kl_objective(jnp.asarray(0.1), jnp.asarray(1.0), 100, delta=bad_delta)


# ---- TDD requirement #1: ELBO limit at delta=1 -----------------------------


def test_pac_bayes_objective_reduces_to_elbo_scaling_when_delta_is_one() -> None:
    """``pac_bayes_kl_objective(R, KL, n, delta=1) == R + KL/n``."""
    risk = jnp.asarray(0.1)
    kl = jnp.asarray(5.0)
    n = 100
    expected = float(risk) + float(kl) / n
    result = pac_bayes_kl_objective(risk, kl, n, delta=1.0)
    assert float(result) == pytest.approx(expected, rel=1e-6)


def test_pac_bayes_objective_matches_mcallester_for_general_delta() -> None:
    """For ``delta in (0, 1)`` the objective must equal :func:`mcallester_bound`."""
    risk = jnp.asarray(0.07)
    kl = jnp.asarray(3.2)
    n = 512
    delta = 0.05
    expected = mcallester_bound(risk, kl, n, delta)
    result = pac_bayes_kl_objective(risk, kl, n, delta=delta)
    assert float(result) == pytest.approx(float(expected), rel=1e-6)


# ---- transform compatibility -----------------------------------------------


def test_pac_bayes_objective_is_jit_compatible() -> None:
    jitted = jax.jit(lambda r, k: pac_bayes_kl_objective(r, k, 256, delta=0.05))
    value = jitted(jnp.asarray(0.1), jnp.asarray(2.0))
    assert bool(jnp.isfinite(value))


def test_pac_bayes_objective_is_grad_compatible() -> None:
    def f(risk: jax.Array) -> jax.Array:
        return pac_bayes_kl_objective(risk, jnp.asarray(2.0), 256, delta=0.05)

    g = jax.grad(f)(jnp.asarray(0.1))
    assert bool(jnp.isfinite(g))


# ---- TDD requirement #5: nnx.value_and_grad runs on a tiny model ------------


class _TinyBayesianMLP(nnx.Module):
    """Two-layer MLP whose first linear layer carries a diagonal-Gaussian posterior.

    Used purely as a synthetic NNX target for the PAC-Bayes loss test; the
    posterior KL is computed from the layer's ``weight_mean`` / ``weight_logvar``
    via :func:`diagonal_gaussian_kl`.
    """

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        # Parameter init keys are drawn from ``rngs`` so no hidden seeds leak.
        key = rngs.params()
        k_mean, k_logvar = jax.random.split(key)
        self.weight_mean = nnx.Param(0.1 * jax.random.normal(k_mean, (4, 4)))
        self.weight_logvar = nnx.Param(
            -3.0 * jnp.ones((4, 4)) + 0.01 * jax.random.normal(k_logvar, (4, 4))
        )
        self.head = nnx.Linear(4, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jnp.tanh(x @ self.weight_mean.value)
        return self.head(h)

    def kl_divergence(self) -> jax.Array:
        return diagonal_gaussian_kl(self.weight_mean.value, self.weight_logvar.value)


def _objective_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=64,
        physics_weight=0.0,
        data_weight=1.0,
        boundary_weight=0.0,
        initial_condition_weight=0.0,
        regularization_weight=0.0,
        calibration_weight=0.0,
        conformal_weight=0.0,
        pac_bayes_weight=1.0,
    )


def test_pac_bayes_loss_composes_with_nnx_value_and_grad_on_tiny_model() -> None:
    """The PAC-Bayes term reaches ``UQLossComponents.pac_bayes`` end-to-end.

    Mirrors the canonical Flax NNX pattern: the loss closure receives
    ``model`` and ``rngs`` as traced arguments; we use
    ``nnx.value_and_grad(..., has_aux=True)`` to return the loss components
    alongside the scalar total.
    """
    model = _TinyBayesianMLP(rngs=nnx.Rngs(params=0))
    config = _objective_config()
    x = jax.random.normal(jax.random.PRNGKey(7), (16, 4))
    y = jax.random.normal(jax.random.PRNGKey(11), (16, 1))
    n = 64
    delta = 0.05

    def pac_bayes_loss(m: _TinyBayesianMLP) -> tuple[jax.Array, UQLossComponents]:
        preds = m(x)
        empirical_risk = jnp.mean((preds - y) ** 2)
        kl = m.kl_divergence()
        pac_bayes_term = pac_bayes_kl_objective(empirical_risk, kl, n, delta=delta)
        components = UQLossComponents.from_components(
            config=config,
            data=empirical_risk,
            kl=kl,
            pac_bayes=pac_bayes_term,
            metadata=(("source", "pac_bayes_test"),),
        )
        return components.total, components

    (total, aux), grads = nnx.value_and_grad(pac_bayes_loss, has_aux=True)(model)
    assert bool(jnp.isfinite(total))
    assert isinstance(aux, UQLossComponents)
    assert aux.pac_bayes is not None
    # The aux ``pac_bayes`` field must equal the McAllester term we plugged in.
    expected = mcallester_bound(jnp.mean((model(x) - y) ** 2), model.kl_divergence(), n, delta)
    assert float(aux.pac_bayes) == pytest.approx(float(expected), rel=1e-5)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "expected at least one gradient leaf from the tiny model"
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
