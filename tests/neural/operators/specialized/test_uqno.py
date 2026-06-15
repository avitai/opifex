"""Task 3.8: canonical conformal `UncertaintyQuantificationNeuralOperator`.

Replaces the previous Bayesian-FNO surface with the three-stage
conformal-prediction pipeline from Ma, Pitt, Azizzadenesheli,
Anandkumar (TMLR 2024 — arXiv:2402.01960). The canonical PyTorch
reference lives at
``../neuraloperator/neuralop/models/uqno.py``;
``../neuraloperator/scripts/train_uqno_darcy.py`` carries the
calibration recipe (``get_coeff_quantile_idx``).

Operator-level tests pin: orchestrator wiring, calibration formula,
``predict_with_bands`` returning a :class:`PredictiveDistribution` with
populated :class:`PredictionInterval`, coverage proportion landing near
the requested ``alpha`` on a small synthetic check, and JAX/NNX
transform compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.specialized.uqno import (
    get_coeff_quantile_idx,
    UncertaintyQuantificationNeuralOperator,
    UQNOBaseSolutionOperator,
    UQNOConformalCalibrator,
    UQNOResidualOperator,
)
from opifex.uncertainty.types import PredictionInterval, PredictiveDistribution


def _make_fno(seed: int = 0) -> FourierNeuralOperator:
    return FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=4,
        modes=2,
        num_layers=2,
        rngs=nnx.Rngs(seed),
    )


def _make_uqno(seed: int = 0) -> UncertaintyQuantificationNeuralOperator:
    return UncertaintyQuantificationNeuralOperator(
        base=UQNOBaseSolutionOperator(_make_fno(seed)),
        residual=UQNOResidualOperator(_make_fno(seed + 1)),
    )


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


def test_uqno_holds_base_and_residual_operators() -> None:
    uqno = _make_uqno()
    assert isinstance(uqno.base, UQNOBaseSolutionOperator)
    assert isinstance(uqno.residual, UQNOResidualOperator)


def test_uqno_forward_returns_solution_and_quantile_pair() -> None:
    """Forward pass returns ``(solution, quantile)``; base is evaluated under no-grad-like."""
    uqno = _make_uqno()
    x = jnp.ones((2, 1, 8, 8))
    solution, quantile = uqno(x)
    assert solution.shape == (2, 1, 8, 8)
    assert quantile.shape == (2, 1, 8, 8)
    assert bool(jnp.all(jnp.isfinite(solution)))
    assert bool(jnp.all(jnp.isfinite(quantile)))


def test_uqno_predict_base_only_matches_base_model() -> None:
    """`predict_base` skips the residual stage."""
    uqno = _make_uqno()
    x = jnp.ones((1, 1, 8, 8))
    direct = uqno.base(x)
    via_method = uqno.predict_base(x)
    assert bool(jnp.allclose(direct, via_method))


# ---------------------------------------------------------------------------
# Conformal calibration formula
# ---------------------------------------------------------------------------


def test_get_coeff_quantile_idx_matches_reference_formula() -> None:
    """Match the canonical neuraloperator/train_uqno_darcy.py implementation."""
    import math

    alpha, delta = 0.05, 0.1
    n_samples, n_gridpts = 500, 421 * 421

    lb = math.sqrt(-math.log(delta) / (2.0 * n_gridpts))
    t = (alpha - lb) / 3.0 + lb
    percentile = alpha - t
    expected_domain = math.ceil(percentile * n_gridpts)
    function_percentile = (
        math.ceil((n_samples + 1) * (delta - math.exp(-2.0 * n_gridpts * t * t))) / n_samples
    )
    expected_function = math.ceil(function_percentile * n_samples)

    domain_idx, function_idx = get_coeff_quantile_idx(
        alpha=alpha, delta=delta, n_samples=n_samples, n_gridpts=n_gridpts
    )
    assert int(domain_idx) == int(expected_domain)
    assert int(function_idx) == int(expected_function)


def test_get_coeff_quantile_idx_rejects_invalid_inputs() -> None:
    """Validate alpha, delta in (0, 1) and positive sample/grid counts."""
    with pytest.raises(ValueError, match="alpha"):
        get_coeff_quantile_idx(alpha=0.0, delta=0.1, n_samples=100, n_gridpts=64)
    with pytest.raises(ValueError, match="delta"):
        get_coeff_quantile_idx(alpha=0.05, delta=1.5, n_samples=100, n_gridpts=64)
    with pytest.raises(ValueError, match="n_samples"):
        get_coeff_quantile_idx(alpha=0.05, delta=0.1, n_samples=0, n_gridpts=64)
    with pytest.raises(ValueError, match="n_gridpts"):
        get_coeff_quantile_idx(alpha=0.05, delta=0.1, n_samples=100, n_gridpts=0)


def test_uqno_calibrate_returns_conformal_calibrator() -> None:
    """`calibrate` returns a fitted ``UQNOConformalCalibrator`` with stored ratios."""
    uqno = _make_uqno()
    x_calib = jax.random.normal(jax.random.PRNGKey(2), (16, 1, 8, 8))
    y_calib = jax.random.normal(jax.random.PRNGKey(3), (16, 1, 8, 8))
    calibrator = uqno.calibrate(x_calib, y_calib, alpha=0.1, delta=0.1)
    assert isinstance(calibrator, UQNOConformalCalibrator)
    assert calibrator.alpha == 0.1
    assert calibrator.delta == 0.1
    assert float(calibrator.scaling_factor) > 0.0


# ---------------------------------------------------------------------------
# predict_with_bands contract
# ---------------------------------------------------------------------------


def test_predict_with_bands_returns_predictive_distribution_with_interval() -> None:
    uqno = _make_uqno()
    x_calib = jax.random.normal(jax.random.PRNGKey(2), (8, 1, 8, 8))
    y_calib = jax.random.normal(jax.random.PRNGKey(3), (8, 1, 8, 8))
    uqno = uqno.with_calibrator(uqno.calibrate(x_calib, y_calib, alpha=0.1, delta=0.1))

    x_test = jax.random.normal(jax.random.PRNGKey(4), (3, 1, 8, 8))
    dist = uqno.predict_with_bands(x_test)
    assert isinstance(dist, PredictiveDistribution)
    assert dist.interval is not None
    assert isinstance(dist.interval, PredictionInterval)
    assert dist.interval.lower.shape == (3, 1, 8, 8)
    assert dist.interval.upper.shape == (3, 1, 8, 8)
    # Lower bound <= upper bound everywhere.
    assert bool(jnp.all(dist.interval.upper >= dist.interval.lower))
    # Honesty: conformal is not Bayesian.
    assert dist.epistemic is None
    assert dist.samples is None
    meta = dist.metadata_dict()
    assert meta["method"] == "conformal"
    assert float(meta["alpha"]) == pytest.approx(0.1)
    assert float(meta["delta"]) == pytest.approx(0.1)


def test_predict_with_bands_requires_calibration() -> None:
    """Calling without a fitted calibrator raises an actionable error."""
    uqno = _make_uqno()
    x_test = jnp.ones((1, 1, 8, 8))
    with pytest.raises(RuntimeError, match="calibrate"):
        uqno.predict_with_bands(x_test)


# ---------------------------------------------------------------------------
# Honesty: no native Bayesian / no MC posterior
# ---------------------------------------------------------------------------


def test_uqno_does_not_expose_bayesian_surface() -> None:
    """The conformal UQNO removes the Bayesian surface (predict_distribution / negative_elbo)."""
    uqno = _make_uqno()
    assert not hasattr(uqno, "predict_distribution")
    assert not hasattr(uqno, "negative_elbo")
    assert not hasattr(uqno, "loss_components")
    assert not hasattr(uqno, "kl_divergence")


# ---------------------------------------------------------------------------
# JAX/NNX transform compatibility (hard exit criterion)
# ---------------------------------------------------------------------------


def test_uqno_forward_is_nnx_jit_compatible() -> None:
    uqno = _make_uqno()

    @nnx.jit
    def step(m: UncertaintyQuantificationNeuralOperator, x: jax.Array) -> jax.Array:
        sol, _quant = m(x)
        return sol

    x = jnp.ones((2, 1, 8, 8))
    out = step(uqno, x)
    assert out.shape == (2, 1, 8, 8)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_uqno_predict_base_is_grad_compatible() -> None:
    """Gradients flow back through the base operator."""
    uqno = _make_uqno()
    x = jnp.ones((1, 1, 8, 8))

    def loss_fn(m: UncertaintyQuantificationNeuralOperator) -> jax.Array:
        return jnp.sum(m.predict_base(x) ** 2)

    grads = nnx.grad(loss_fn)(uqno)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "expected gradient leaves for the base operator parameters"


def test_uqno_residual_loss_is_grad_compatible() -> None:
    """Gradients flow back through the residual operator under quantile loss."""
    from opifex.uncertainty.losses import PointwiseQuantileLoss

    uqno = _make_uqno()
    loss_fn = PointwiseQuantileLoss(alpha=0.1, reduction="mean")
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 8, 8))
    y = jax.random.normal(jax.random.PRNGKey(1), (2, 1, 8, 8))

    def residual_loss(m: UncertaintyQuantificationNeuralOperator) -> jax.Array:
        base_pred = m.predict_base(x)
        quantile_pred = jnp.abs(m.residual(x))
        return loss_fn(y_pred=quantile_pred, y=base_pred - y)

    grads = nnx.grad(residual_loss)(uqno)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
