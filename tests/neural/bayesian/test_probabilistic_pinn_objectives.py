"""Failing tests pinning the shared-objective surface ``ProbabilisticPINN`` must expose.

These tests codify the public contract for ``kl_divergence`` /
``predict_distribution`` / ``loss_components`` / ``negative_elbo`` so the
implementation cannot drift back to ad-hoc dict returns or hidden seeds.

Reference implementation: Blundell et al. 2015 *"Weight Uncertainty in
Neural Networks"* for the ELBO formula; Bishop 2006 PRML §10.1 for the
KL-aware variational lower bound. The per-example scaling
``KL / dataset_size`` follows the BBB paper Section 3.4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.bayesian.probabilistic_pinns import ProbabilisticPINN
from opifex.uncertainty.objectives import (
    ObjectiveConfig,
    UQLossComponents,
)
from opifex.uncertainty.types import PredictiveDistribution, PredictiveMode


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_pinn(rngs: nnx.Rngs | None = None) -> ProbabilisticPINN:
    if rngs is None:
        rngs = nnx.Rngs(params=0, sample=1, default=2, noise=3)
    return ProbabilisticPINN(
        input_dim=2,
        hidden_dims=(8, 8),
        use_bayesian=True,
        rngs=rngs,
    )


def _poisson_residual(x: jax.Array, u_pred: jax.Array) -> jax.Array:
    """Toy linear residual: r = u - x[:, 0:1] (deliberately trivial)."""
    return u_pred - x[:, :1]


def _toy_batch(n: int = 8, dim: int = 2) -> dict:
    rng = jax.random.PRNGKey(0)
    x_key, y_key = jax.random.split(rng)
    return {
        "x": jax.random.normal(x_key, (n, dim)),
        "y": jax.random.normal(y_key, (n, 1)),
        "pde_residual_fn": _poisson_residual,
        "boundary_conditions": {"weight": 1.0, "value": 0.0},
    }


def _objective(
    *,
    kl_weight: float = 1e-3,
    dataset_size: int | None = 100,
    physics_weight: float = 1.0,
    data_weight: float = 1.0,
    boundary_weight: float = 0.5,
) -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=kl_weight,
        dataset_size=dataset_size,
        physics_weight=physics_weight,
        data_weight=data_weight,
        boundary_weight=boundary_weight,
        initial_condition_weight=0.0,
        regularization_weight=0.0,
        calibration_weight=0.0,
        conformal_weight=0.0,
        pac_bayes_weight=0.0,
    )


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------


def test_kl_divergence_returns_jax_array_and_sums_layer_kls() -> None:
    """``kl_divergence()`` aggregates the per-layer KL into a scalar ``jax.Array``."""
    pinn = _make_pinn()
    kl = pinn.kl_divergence()
    assert isinstance(kl, jax.Array)
    assert kl.shape == ()
    assert jnp.isfinite(kl)
    assert kl >= 0.0


# ---------------------------------------------------------------------------
# predict_distribution
# ---------------------------------------------------------------------------


def test_predict_distribution_returns_predictive_distribution_with_metadata() -> None:
    """The predictive call returns a :class:`PredictiveDistribution` with method metadata."""
    pinn = _make_pinn()
    x = jnp.zeros((4, 2))
    out = pinn.predict_distribution(
        x,
        rngs=nnx.Rngs(sample=0),
        num_samples=8,
        mode=PredictiveMode.PREDICTIVE,
    )
    assert isinstance(out, PredictiveDistribution)
    assert out.mean.shape[0] == 4
    metadata = out.metadata_dict()
    assert "method" in metadata
    assert "num_samples" in metadata


def test_predict_distribution_raises_value_error_for_unknown_mode() -> None:
    """An unknown predictive mode is rejected loudly, not silently coerced."""
    pinn = _make_pinn()
    x = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match=r"(?i)mode"):
        pinn.predict_distribution(x, rngs=nnx.Rngs(sample=0), num_samples=4, mode="bogus")  # type: ignore[arg-type]


def test_predict_distribution_advances_rng_yields_different_samples() -> None:
    """Calling with two distinct RNG seeds gives different posterior samples."""
    pinn = _make_pinn()
    x = jnp.zeros((4, 2))
    out_a = pinn.predict_distribution(x, rngs=nnx.Rngs(sample=0), num_samples=4)
    out_b = pinn.predict_distribution(x, rngs=nnx.Rngs(sample=1), num_samples=4)
    assert not jnp.allclose(out_a.mean, out_b.mean)


def test_predict_distribution_deterministic_path_returns_zero_epistemic() -> None:
    """When ``use_bayesian=False``, the returned distribution declares no epistemic uncertainty."""
    pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(4,), use_bayesian=False, rngs=nnx.Rngs(0))
    x = jnp.zeros((3, 2))
    out = pinn.predict_distribution(x, rngs=nnx.Rngs(sample=0), num_samples=1)
    assert isinstance(out, PredictiveDistribution)
    if out.epistemic is not None:
        assert jnp.allclose(out.epistemic, 0.0)
    assert out.metadata_dict().get("method") == "deterministic"


# ---------------------------------------------------------------------------
# loss_components / negative_elbo
# ---------------------------------------------------------------------------


def test_loss_components_returns_uq_loss_components() -> None:
    """``loss_components(...)`` returns the Phase 1 container, not a plain dict."""
    pinn = _make_pinn()
    out = pinn.loss_components(_toy_batch(), rngs=nnx.Rngs(sample=0), objective=_objective())
    assert isinstance(out, UQLossComponents)
    out.validate()


def test_negative_elbo_returns_components_with_negative_elbo_set() -> None:
    """``negative_elbo(...)`` guarantees ``negative_elbo`` is populated."""
    pinn = _make_pinn()
    out = pinn.negative_elbo(_toy_batch(), rngs=nnx.Rngs(sample=0), objective=_objective())
    assert isinstance(out, UQLossComponents)
    assert out.negative_elbo is not None
    out.validate()


def test_negative_elbo_total_matches_uq_loss_components_from_components_formula() -> None:
    """``total`` equals the weight-driven sum produced by ``UQLossComponents.from_components``."""
    pinn = _make_pinn()
    config = _objective(kl_weight=1e-2, physics_weight=2.0, data_weight=3.0, boundary_weight=0.5)
    batch = _toy_batch()
    out = pinn.negative_elbo(batch, rngs=nnx.Rngs(sample=0), objective=config)
    expected = UQLossComponents.from_components(
        config=config,
        data=out.data,
        physics_residual=out.physics_residual,
        boundary=out.boundary,
        kl=out.kl,
    )
    assert jnp.allclose(out.total, expected.total, rtol=1e-6, atol=1e-6)


def test_negative_elbo_omitting_optional_field_drops_only_that_term() -> None:
    """A boundary-less batch removes only the boundary contribution from ``total``."""
    pinn = _make_pinn()
    config = _objective(boundary_weight=2.0)
    with_bc = _toy_batch()
    without_bc = {**with_bc, "boundary_conditions": None}

    full = pinn.negative_elbo(with_bc, rngs=nnx.Rngs(sample=0), objective=config)
    partial = pinn.negative_elbo(without_bc, rngs=nnx.Rngs(sample=0), objective=config)

    assert full.boundary is not None
    assert partial.boundary is None
    # Total of the partial run must equal the full-run total minus the dropped boundary term.
    delta = full.total - partial.total
    assert jnp.allclose(delta, config.boundary_weight * full.boundary, rtol=1e-5, atol=1e-6)


def test_negative_elbo_missing_required_field_raises_value_error() -> None:
    """Omitting ``x`` (required) raises a ``ValueError`` naming the field."""
    pinn = _make_pinn()
    bad_batch = {k: v for k, v in _toy_batch().items() if k != "x"}
    with pytest.raises(ValueError, match=r"(?i)x"):
        pinn.negative_elbo(bad_batch, rngs=nnx.Rngs(sample=0), objective=_objective())


def test_negative_elbo_dataset_size_scales_kl() -> None:
    """Doubling ``dataset_size`` halves the KL contribution to the total."""
    pinn = _make_pinn()
    base = _objective(
        kl_weight=1.0, dataset_size=100, physics_weight=0.0, data_weight=0.0, boundary_weight=0.0
    )
    bigger = _objective(
        kl_weight=1.0, dataset_size=200, physics_weight=0.0, data_weight=0.0, boundary_weight=0.0
    )
    batch = _toy_batch()
    small = pinn.negative_elbo(batch, rngs=nnx.Rngs(sample=0), objective=base)
    large = pinn.negative_elbo(batch, rngs=nnx.Rngs(sample=0), objective=bigger)
    assert small.kl is not None and large.kl is not None
    # KL term itself is identical (same weights); only the dataset-scaled contribution changes.
    assert jnp.allclose(small.kl, large.kl)
    # Total contribution of KL is halved.
    assert jnp.allclose(large.total, small.total / 2.0, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# nnx.value_and_grad smoke
# ---------------------------------------------------------------------------


def test_loss_components_works_inside_nnx_value_and_grad() -> None:
    """``loss_components`` is consumable by ``nnx.value_and_grad(..., has_aux=True)``.

    Follows the canonical Flax NNX pattern from
    ``memory-bank/guides/flax-nnx-guide.md`` — ``rngs`` is passed as an
    argument so ``nnx.value_and_grad`` tracks it as part of the traced
    graph (closing over an outer ``rngs`` would raise
    ``TraceContextError`` when ``RngStream`` is advanced across trace
    levels).
    """
    pinn = _make_pinn()
    objective = _objective()
    batch = _toy_batch()

    def loss_fn(model: ProbabilisticPINN, rngs: nnx.Rngs) -> tuple[jax.Array, UQLossComponents]:
        components = model.loss_components(batch, rngs=rngs, objective=objective)
        return components.total, components

    (loss, aux), _ = nnx.value_and_grad(loss_fn, has_aux=True)(pinn, nnx.Rngs(sample=0))
    assert isinstance(aux, UQLossComponents)
    assert isinstance(loss, jax.Array)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# NNX boundary preservation
# ---------------------------------------------------------------------------


def test_kl_divergence_is_nnx_jit_compatible() -> None:
    """``kl_divergence`` must compile under ``nnx.jit`` for use in jit'd training loops."""
    pinn = _make_pinn()
    jitted = nnx.jit(lambda model: model.kl_divergence())
    out = jitted(pinn)
    assert isinstance(out, jax.Array)
    assert out.shape == ()


def test_loss_components_is_nnx_jit_compatible() -> None:
    """``loss_components`` traces under ``nnx.jit`` with rngs passed as a traced arg."""
    pinn = _make_pinn()
    objective = _objective()
    batch = _toy_batch()

    @nnx.jit
    def step(model: ProbabilisticPINN, rngs: nnx.Rngs) -> jax.Array:
        return model.loss_components(batch, rngs=rngs, objective=objective).total

    out = step(pinn, nnx.Rngs(sample=0))
    assert isinstance(out, jax.Array)
    assert out.shape == ()


def test_probabilistic_pinn_remains_nnx_module_surface() -> None:
    """``ProbabilisticPINN`` stays an ``nnx.Module``; raw JAX transforms cross
    via explicit ``nnx.split``/``nnx.merge`` if they cross at all.
    """
    pinn = _make_pinn()
    assert isinstance(pinn, nnx.Module)
    graphdef, state = nnx.split(pinn)
    rebuilt = nnx.merge(graphdef, state)
    assert isinstance(rebuilt, ProbabilisticPINN)
    # Round-trip yields a usable model.
    x = jnp.zeros((2, 2))
    out = rebuilt(x)
    assert out.shape[0] == 2
