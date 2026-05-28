"""Tests for the Probabilistic Fourier Neural Operator (PNO).

PNO equips an FNO backbone with twin pointwise heads — a *mean* head
and a *log-variance* head — over the operator output, producing a
calibrated heteroscedastic-Gaussian
:class:`~opifex.uncertainty.types.PredictiveDistribution` at every
spatial location. The training objective is the
heteroscedastic-Gaussian negative log-likelihood that opifex already
ships at :func:`opifex.uncertainty.likelihoods.heteroscedastic_gaussian_log_likelihood`
(Kendall & Gal 2017, arXiv:1703.04977 §3.1).

Canonical reference:
* Magnani et al. 2024 line of work (companion to the linearised neural
  operator / LUNO of arXiv:2406.04317) treats neural operators as
  probabilistic surrogates; the heteroscedastic-Gaussian head pattern
  used here is the standard aleatoric-uncertainty recipe from
  Kendall & Gal 2017 §3.1, ported to operator-valued outputs.

References
----------
* Magnani, E. et al. 2024 — function-uncertainty neural operators
  (LUNO companion thread; same Phase 10 citation as
  ``curvature/luno.py``), arXiv:2406.04317.
* Kendall, A., Gal, Y. 2017 — *What Uncertainties Do We Need in
  Bayesian Deep Learning for Computer Vision?*, arXiv:1703.04977.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.probabilistic import (
    probabilistic_fno_negative_log_likelihood,
    ProbabilisticFourierNeuralOperator,
)
from opifex.uncertainty.types import PredictiveDistribution


def _make_pno(
    *,
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 8,
    modes: int = 4,
    num_layers: int = 2,
    rngs: nnx.Rngs | None = None,
) -> ProbabilisticFourierNeuralOperator:
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ProbabilisticFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        rngs=rngs,
    )


def test_call_returns_mean_and_log_variance_with_matching_shapes() -> None:
    """``__call__`` returns ``(mean, log_variance)``, both ``(batch, out, *spatial)``."""
    pno = _make_pno(in_channels=2, out_channels=3)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 2, 16, 16))

    mean, log_variance = pno(x)
    assert mean.shape == (4, 3, 16, 16)
    assert log_variance.shape == (4, 3, 16, 16)


def test_predict_distribution_returns_calibrated_predictive() -> None:
    """``predict_distribution`` packs the head outputs into a ``PredictiveDistribution``."""
    pno = _make_pno()
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 1, 8, 8))

    predictive = pno.predict_distribution(x)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.aleatoric is not None
    assert predictive.mean.shape == (2, 1, 8, 8)
    assert predictive.variance.shape == predictive.mean.shape
    assert jnp.all(predictive.variance > 0.0)
    # Aleatoric == variance (only aleatoric is captured; epistemic requires a
    # higher-level wrapper such as ``LaplaceAdapterSpec`` or an ensemble).
    assert jnp.allclose(predictive.aleatoric, predictive.variance)
    assert predictive.epistemic is None or jnp.allclose(
        predictive.epistemic, jnp.zeros_like(predictive.mean)
    )


def test_variance_equals_exp_log_variance_head() -> None:
    """Predictive variance is exactly ``exp(log_variance_head(x))``."""
    pno = _make_pno()
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 1, 8, 8))

    _mean, log_variance = pno(x)
    predictive = pno.predict_distribution(x)
    assert predictive.variance is not None
    assert jnp.allclose(predictive.variance, jnp.exp(log_variance), atol=1e-5)


def test_negative_log_likelihood_matches_heteroscedastic_gaussian_formula() -> None:
    r"""The training objective is ``-Σ log N(y_i; μ_i, σ_i²)`` per element.

    For a single observation,
    ``-log N(y; μ, σ²) = 0.5 (log(2π) + log σ² + (y - μ)² / σ²)``.
    """
    pno = _make_pno()
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 1, 8, 8))
    y = jax.random.normal(jax.random.PRNGKey(5), (2, 1, 8, 8))

    mean, log_variance = pno(x)
    nll = probabilistic_fno_negative_log_likelihood(pno, x, y)

    variance = jnp.exp(log_variance)
    elementwise_nll = 0.5 * (jnp.log(2.0 * jnp.pi) + log_variance + (y - mean) ** 2 / variance)
    expected = jnp.mean(elementwise_nll)
    assert jnp.allclose(nll, expected, atol=1e-5)


def test_negative_log_likelihood_is_grad_compatible_over_parameters() -> None:
    """``nnx.value_and_grad`` over PNO parameters returns a non-zero gradient."""
    pno = _make_pno()
    x = jax.random.normal(jax.random.PRNGKey(6), (2, 1, 8, 8))
    y = jax.random.normal(jax.random.PRNGKey(7), (2, 1, 8, 8))

    @nnx.value_and_grad
    def loss_fn(model: ProbabilisticFourierNeuralOperator) -> jax.Array:
        return probabilistic_fno_negative_log_likelihood(model, x, y)

    value, grads = loss_fn(pno)
    assert jnp.isfinite(value)
    # At least one leaf of the gradient pytree should be non-zero.
    flat_grads = jax.tree_util.tree_leaves(grads)
    assert len(flat_grads) > 0
    nonzero = any(bool(jnp.any(jnp.abs(g) > 0.0)) for g in flat_grads)
    assert nonzero


def test_predictive_metadata_advertises_pno_source() -> None:
    """Metadata records both the aleatoric strategy and the PNO source package."""
    pno = _make_pno()
    x = jax.random.normal(jax.random.PRNGKey(8), (1, 1, 8, 8))

    predictive = pno.predict_distribution(x)
    keys = {k for k, _ in predictive.metadata}
    assert "method" in keys
    assert "source_package" in keys


def test_call_is_jit_compatible_under_nnx_jit() -> None:
    """``nnx.jit`` compiles the forward path."""
    pno = _make_pno()
    x = jax.random.normal(jax.random.PRNGKey(9), (2, 1, 8, 8))

    @nnx.jit
    def forward(model: ProbabilisticFourierNeuralOperator, inp: jax.Array) -> jax.Array:
        return model.predict_distribution(inp).mean

    mean = forward(pno, x)
    assert mean.shape == (2, 1, 8, 8)
    assert jnp.all(jnp.isfinite(mean))


def test_rejects_nonpositive_log_variance_floor() -> None:
    """``log_variance_floor`` and ``log_variance_ceiling`` must bound the head."""
    with pytest.raises(ValueError, match="log_variance"):
        ProbabilisticFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            modes=2,
            num_layers=1,
            log_variance_floor=1.0,
            log_variance_ceiling=-1.0,  # ceiling below floor
            rngs=nnx.Rngs(0),
        )
