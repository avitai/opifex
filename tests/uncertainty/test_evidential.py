r"""Tests for the Normal-Inverse-Gamma (NIG) evidential-regression primitive.

Covers, against hand-computed reference values:

* :func:`positive_evidential_params` — softplus parameterisation
  (``nu, beta > 0``; ``alpha > 1`` via ``softplus + 1``), matching the
  chemprop ``EvidentialFFN.forward`` reference
  (``../chemprop/chemprop/nn/predictors.py:197-200``).
* :func:`evidential_nll` — the Amini et al. 2020 NIG loss
  ``NLL_NIG + lambda * |y - gamma| * (2 nu + alpha)`` transcribed from the
  chemprop ``EvidentialLoss`` reference
  (``../chemprop/chemprop/nn/metrics.py:222-257``).
* :func:`nig_to_predictive_distribution` — the eIP variance decomposition
  ``mean = gamma``, ``aleatoric = beta/(alpha-1)``,
  ``epistemic = beta/(nu*(alpha-1))`` (arXiv:2407.13994, Nat. Commun. 2025).

The final test is a REQUIRED ``jit``/``grad``/``vmap`` smoke test: every
public surface must be transform-clean.
"""

from __future__ import annotations

from math import lgamma, log, pi

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.evidential import (
    evidential_nll,
    nig_to_predictive_distribution,
    NIGParams,
    positive_evidential_params,
)
from opifex.uncertainty.types import PredictiveDistribution


def _softplus(x: float) -> float:
    """Reference scalar softplus matching ``jax.nn.softplus``."""
    return float(jnp.logaddexp(0.0, jnp.asarray(x)))


def _require(array: jax.Array | None) -> jax.Array:
    """Assert an optional ``PredictiveDistribution`` field is populated."""
    assert array is not None
    return array


def test_positive_params_apply_softplus_with_alpha_offset() -> None:
    """nu/beta = softplus(raw); alpha = softplus(raw) + 1 per chemprop.

    A tiny documented positivity floor (1e-6) is added to nu/alpha/beta to keep
    the closed-form NIG moments finite, so a 1e-5 absolute tolerance is used.
    """
    raw = jnp.array([0.0, 0.0, 0.0, 0.0])  # gamma, nu, alpha, beta
    params = positive_evidential_params(raw)

    assert float(params.gamma) == pytest.approx(0.0)
    assert float(params.nu) == pytest.approx(_softplus(0.0), abs=1e-5)
    assert float(params.alpha) == pytest.approx(_softplus(0.0) + 1.0, abs=1e-5)
    assert float(params.beta) == pytest.approx(_softplus(0.0), abs=1e-5)


def test_positive_params_enforce_alpha_strictly_above_one() -> None:
    """alpha = softplus(raw) + 1 is strictly > 1 even for very negative raw."""
    raw = jnp.array([5.0, -50.0, -50.0, -50.0])
    params = positive_evidential_params(raw)

    assert float(params.alpha) > 1.0
    assert float(params.nu) > 0.0
    assert float(params.beta) > 0.0


def test_nig_to_predictive_distribution_matches_hand_computed() -> None:
    """mean/aleatoric/epistemic equal the closed-form NIG moments."""
    gamma, nu, alpha, beta = 2.5, 0.5, 3.0, 4.0
    params = NIGParams(
        gamma=jnp.asarray(gamma),
        nu=jnp.asarray(nu),
        alpha=jnp.asarray(alpha),
        beta=jnp.asarray(beta),
    )

    predictive = nig_to_predictive_distribution(params)

    expected_aleatoric = beta / (alpha - 1.0)  # 4 / 2 = 2.0
    expected_epistemic = beta / (nu * (alpha - 1.0))  # 4 / (0.5*2) = 4.0
    assert isinstance(predictive, PredictiveDistribution)
    assert float(predictive.mean) == pytest.approx(gamma)
    assert float(_require(predictive.aleatoric)) == pytest.approx(expected_aleatoric)
    assert float(_require(predictive.epistemic)) == pytest.approx(expected_epistemic)
    assert float(_require(predictive.total_uncertainty)) == pytest.approx(
        expected_aleatoric + expected_epistemic
    )
    assert float(_require(predictive.variance)) == pytest.approx(
        expected_aleatoric + expected_epistemic
    )


def test_predictive_distribution_passes_variance_additivity_validate() -> None:
    """The produced container satisfies total == epistemic + aleatoric."""
    params = NIGParams(
        gamma=jnp.asarray(1.0),
        nu=jnp.asarray(2.0),
        alpha=jnp.asarray(4.0),
        beta=jnp.asarray(6.0),
    )
    predictive = nig_to_predictive_distribution(params)
    predictive.validate()  # raises if additivity is violated


def test_aleatoric_strictly_below_total() -> None:
    """Epistemic variance is positive, so aleatoric < total."""
    params = NIGParams(
        gamma=jnp.asarray(0.0),
        nu=jnp.asarray(1.0),
        alpha=jnp.asarray(2.0),
        beta=jnp.asarray(3.0),
    )
    predictive = nig_to_predictive_distribution(params)
    assert float(_require(predictive.aleatoric)) < float(_require(predictive.total_uncertainty))


def test_evidential_nll_matches_reference_formula() -> None:
    """Loss equals the chemprop ``EvidentialLoss`` closed form (lambda=0)."""
    gamma, nu, alpha, beta = 1.0, 0.5, 2.0, 1.5
    target = 1.5
    params = NIGParams(
        gamma=jnp.asarray(gamma),
        nu=jnp.asarray(nu),
        alpha=jnp.asarray(alpha),
        beta=jnp.asarray(beta),
    )

    loss = evidential_nll(params, jnp.asarray(target), coefficient=0.0)

    residual = target - gamma
    two_b_lambda = 2.0 * beta * (1.0 + nu)
    expected_nll = (
        0.5 * log(pi / nu)
        - alpha * log(two_b_lambda)
        + (alpha + 0.5) * log(nu * residual**2 + two_b_lambda)
        + lgamma(alpha)
        - lgamma(alpha + 0.5)
    )
    assert float(loss) == pytest.approx(expected_nll, rel=1e-5)


def test_evidential_nll_regularizer_adds_error_evidence_term() -> None:
    """With lambda>0 the loss adds ``lambda * |y-gamma| * (2 nu + alpha)``."""
    gamma, nu, alpha, beta = 0.0, 1.0, 2.0, 1.0
    target = 2.0
    coefficient = 0.3
    params = NIGParams(
        gamma=jnp.asarray(gamma),
        nu=jnp.asarray(nu),
        alpha=jnp.asarray(alpha),
        beta=jnp.asarray(beta),
    )

    loss_no_reg = evidential_nll(params, jnp.asarray(target), coefficient=0.0)
    loss_reg = evidential_nll(params, jnp.asarray(target), coefficient=coefficient)

    residual_abs = abs(target - gamma)
    expected_reg = coefficient * residual_abs * (2.0 * nu + alpha)
    assert float(loss_reg - loss_no_reg) == pytest.approx(expected_reg, rel=1e-5)


def test_evidential_nll_is_finite_for_batched_inputs() -> None:
    """Batched NIG params + targets yield a finite per-element loss vector."""
    raw = jnp.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-1.0, 0.5, 1.0, -0.5],
            [2.0, -0.3, 0.7, 0.1],
        ]
    )
    params = positive_evidential_params(raw)
    targets = jnp.array([0.0, 1.0, -1.0])

    loss = jax.vmap(lambda p, y: evidential_nll(p, y, coefficient=0.1))(params, targets)

    assert loss.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(loss)))


def test_evidential_surfaces_are_jit_grad_vmap_clean() -> None:
    """REQUIRED transform smoke test over the loss w.r.t. raw logits."""

    def loss_from_raw(raw: jax.Array, target: jax.Array) -> jax.Array:
        params = positive_evidential_params(raw)
        return evidential_nll(params, target, coefficient=0.1)

    raw_batch = jnp.array(
        [
            [0.5, 0.1, 0.2, 0.3],
            [-0.5, 0.4, -0.2, 0.1],
        ]
    )
    targets = jnp.array([0.3, -0.7])

    batched = jax.jit(jax.vmap(loss_from_raw))(raw_batch, targets)
    assert batched.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(batched)))

    grad_fn = jax.jit(jax.grad(lambda raw: loss_from_raw(raw, jnp.asarray(0.3))))
    grads = grad_fn(raw_batch[0])
    assert grads.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(grads)))

    # nig_to_predictive_distribution must also be jit/vmap clean.
    def mean_under_jit(raw: jax.Array) -> jax.Array:
        return nig_to_predictive_distribution(positive_evidential_params(raw)).mean

    means = jax.jit(jax.vmap(mean_under_jit))(raw_batch)
    assert means.shape == (2,)
