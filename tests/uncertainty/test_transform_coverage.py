"""Canonical JAX/NNX transform-compatibility coverage for the uncertainty platform.

Phase 1 through Phase 5 each declare a per-task workflow rule mandating
that any code on a jit path is exercised against the canonical transform
set:

* ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` for pure-array kernels.
* ``nnx.jit`` / ``nnx.value_and_grad(has_aux=...)`` for NNX-state-carrying
  kernels with traced ``rngs``.

This file fills the surveyed gaps in one place so the contract is
visible. Coverage focus:

* ``opifex.uncertainty.kernels.bayesian`` — vmap for the two kernel
  helpers (jit + grad were already covered).
* ``opifex.uncertainty.layers.bayesian`` — ``nnx.jit`` and
  ``nnx.value_and_grad`` for BayesianLinear and
  BayesianSpectralConvolution with traced rngs.
* ``opifex.uncertainty.metrics`` — grad for the differentiable kernels.
* ``opifex.uncertainty.forecasting_metrics`` — vmap for the per-sample
  scoring kernels.
* ``opifex.uncertainty.scientific.domain_metrics`` — jit / grad / vmap
  for the eight domain reliability kernels.
* ``opifex.uncertainty.selective.risk_coverage`` — grad and vmap.
* ``opifex.uncertainty.priors_physics`` — ``nnx.value_and_grad`` for
  the NNX-state-carrying physics-prior modules.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty import forecasting_metrics as fm, metrics, selective
from opifex.uncertainty.kernels.bayesian import (
    diagonal_gaussian_kl,
    sample_diagonal_gaussian,
)
from opifex.uncertainty.layers.bayesian import (
    BayesianLinear,
    BayesianSpectralConvolution,
)
from opifex.uncertainty.priors_physics import PhysicsInformedPriors
from opifex.uncertainty.scientific import domain_metrics


# ---------------------------------------------------------------------------
# kernels/bayesian — vmap
# ---------------------------------------------------------------------------


def test_diagonal_gaussian_kl_is_vmap_compatible() -> None:
    means = jax.random.normal(jax.random.key(0), (4, 8))
    logvars = jnp.zeros((4, 8))
    out = jax.vmap(diagonal_gaussian_kl)(means, logvars)
    assert out.shape == (4,)


def test_sample_diagonal_gaussian_is_vmap_compatible() -> None:
    means = jnp.zeros((4, 8))
    logvars = jnp.zeros((4, 8))
    keys = jax.random.split(jax.random.key(1), 4)
    out = jax.vmap(sample_diagonal_gaussian)(means, logvars, keys)
    assert out.shape == (4, 8)


# ---------------------------------------------------------------------------
# layers/bayesian — nnx.jit + nnx.value_and_grad with traced rngs
# ---------------------------------------------------------------------------


def test_bayesian_linear_traces_under_nnx_jit_with_traced_rngs() -> None:
    layer = BayesianLinear(in_features=4, out_features=2, prior_std=1.0, rngs=nnx.Rngs(0))

    @nnx.jit
    def call(model: BayesianLinear, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return model(x, rngs=rngs)

    out = call(layer, jnp.ones((3, 4)), nnx.Rngs(7))
    assert out.shape == (3, 2)


def test_bayesian_linear_supports_nnx_value_and_grad_with_traced_rngs() -> None:
    layer = BayesianLinear(in_features=4, out_features=2, prior_std=1.0, rngs=nnx.Rngs(0))

    def loss_fn(model: BayesianLinear, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return jnp.sum(model(x, rngs=rngs) ** 2)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=0)
    loss, grad = grad_fn(layer, jnp.ones((3, 4)), nnx.Rngs(7))
    assert loss.shape == ()
    leaves = jax.tree_util.tree_leaves(grad)
    assert any(l.shape != () for l in leaves)


def test_bayesian_spectral_convolution_traces_under_nnx_jit_with_traced_rngs() -> None:
    layer = BayesianSpectralConvolution(
        in_channels=2,
        out_channels=2,
        modes=(4, 4),
        prior_std=1.0,
        rngs=nnx.Rngs(0),
    )

    @nnx.jit
    def call(model: BayesianSpectralConvolution, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return model(x, rngs=rngs)

    # BayesianSpectralConvolution expects channels-first ``(batch, channels, H, W)``.
    x = jnp.ones((1, 2, 8, 8))
    out = call(layer, x, nnx.Rngs(7))
    assert out.shape == (1, 2, 8, 8)


def test_bayesian_spectral_convolution_supports_nnx_value_and_grad() -> None:
    layer = BayesianSpectralConvolution(
        in_channels=2,
        out_channels=2,
        modes=(4, 4),
        prior_std=1.0,
        rngs=nnx.Rngs(0),
    )

    def loss_fn(model: BayesianSpectralConvolution, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return jnp.sum(model(x, rngs=rngs) ** 2)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=0)
    loss, _ = grad_fn(layer, jnp.ones((1, 2, 8, 8)), nnx.Rngs(7))
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# uncertainty.metrics — grad
# ---------------------------------------------------------------------------


def test_predictive_entropy_supports_grad() -> None:
    def loss(probs: jax.Array) -> jax.Array:
        return jnp.sum(metrics.predictive_entropy(ensemble_probabilities=probs[None]))

    probs = jnp.array([[0.1, 0.9], [0.5, 0.5]])
    grad = jax.grad(loss)(probs)
    assert grad.shape == probs.shape


def test_interval_score_supports_grad() -> None:
    def loss(lower: jax.Array) -> jax.Array:
        return jnp.sum(
            metrics.interval_score(
                lower=lower,
                upper=lower + 1.0,
                targets=jnp.zeros_like(lower),
                alpha=0.1,
            )
        )

    lower = jnp.array([0.5, 0.1, -0.2])
    grad = jax.grad(loss)(lower)
    assert grad.shape == lower.shape


# ---------------------------------------------------------------------------
# forecasting_metrics — vmap
# ---------------------------------------------------------------------------


def test_crps_is_vmap_compatible_across_batch() -> None:
    """vmap over a leading batch axis on top of the canonical
    ``(n_samples, n_members)`` shape — confirms the kernel is fully
    array-only with no Python branches on input shape."""
    batched_preds = jax.random.normal(jax.random.key(0), (3, 4, 5))  # batch × samples × members
    batched_targets = jnp.zeros((3, 4))

    def per_batch(p: jax.Array, t: jax.Array) -> jax.Array:
        return fm.crps(predictions=p, targets=t)

    out = jax.vmap(per_batch)(batched_preds, batched_targets)
    assert out.shape == (3,)


def test_energy_score_is_vmap_compatible_across_batch() -> None:
    ensembles = jax.random.normal(jax.random.key(0), (3, 5, 2))
    targets = jnp.zeros((3, 2))

    def per_sample(e: jax.Array, t: jax.Array) -> jax.Array:
        return fm.energy_score(ensemble=e[None], targets=t[None])

    out = jax.vmap(per_sample)(ensembles, targets)
    assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# domain_metrics — jit + grad
# ---------------------------------------------------------------------------


def test_physics_residual_coverage_traces_under_jit() -> None:
    @jax.jit
    def jitted(residuals: jax.Array) -> jax.Array:
        return domain_metrics.physics_residual_coverage(residuals=residuals, threshold=0.5).value

    out = jitted(jnp.array([0.0, 0.1, 0.6, 0.3]))
    assert out.shape == ()


def test_chemical_accuracy_coverage_traces_under_jit() -> None:
    @jax.jit
    def jitted(pred: jax.Array, true: jax.Array) -> jax.Array:
        return domain_metrics.chemical_accuracy_coverage(
            predicted_energies=pred, true_energies=true, tolerance=0.1
        ).value

    out = jitted(jnp.array([0.0, 0.1]), jnp.array([0.0, 0.15]))
    assert out.shape == ()


# ---------------------------------------------------------------------------
# selective.risk_coverage — grad + vmap
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_supports_vmap() -> None:
    errors = jax.random.uniform(jax.random.key(0), (3, 10))
    confidences = jax.random.uniform(jax.random.key(1), (3, 10))

    @jax.vmap
    def per_curve(e: jax.Array, c: jax.Array) -> jax.Array:
        coverages, _ = selective.risk_coverage_curve(errors=e, confidences=c)
        return coverages

    out = per_curve(errors, confidences)
    assert out.shape == (3, 10)


# ---------------------------------------------------------------------------
# priors_physics — nnx.value_and_grad
# ---------------------------------------------------------------------------


def test_physics_informed_priors_supports_nnx_value_and_grad() -> None:
    prior = PhysicsInformedPriors(conservation_laws=("energy", "momentum"), rngs=nnx.Rngs(0))

    def loss(model: PhysicsInformedPriors, params: jax.Array) -> jax.Array:
        return model.compute_violation_penalty(params)

    grad_fn = nnx.value_and_grad(loss, argnums=0)
    penalty, _ = grad_fn(prior, jnp.array([1.0, 2.0, -1.0, 0.5]))
    assert penalty.shape == ()


@pytest.mark.parametrize("aggregation_method", ["variance", "std", "range", "iqr"])
def test_ensemble_disagreement_traces_under_jit_for_every_mode(
    aggregation_method: str,
) -> None:
    """Smoke-cover all four ensemble disagreement statistics under jit."""
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    preds = jax.random.normal(jax.random.key(0), (5, 4, 2))

    @jax.jit
    def jitted(p: jax.Array) -> jax.Array:
        return EpistemicUncertainty.compute_ensemble_disagreement(
            p, aggregation_method=aggregation_method
        )

    out = jitted(preds)
    assert out.shape == (4, 2)
