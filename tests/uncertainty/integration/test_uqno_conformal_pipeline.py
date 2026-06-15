"""Pin the conformal UQNO three-stage pipeline against the shared platform surface.

After the conformal rewrite, ``UncertaintyQuantificationNeuralOperator``
exposes:

* a ``(solution, quantile_width)`` forward pass mediated by
  ``UQNOBaseSolutionOperator`` + ``UQNOResidualOperator`` (both thin
  wrappers around the shared
  :class:`opifex.neural.operators.fno.base.FourierNeuralOperator`);
* :meth:`calibrate` returning a :class:`UQNOConformalCalibrator` derived
  from the per-grid ratios ``|y - base(x)| / (residual(x) + eps)`` via
  :func:`get_coeff_quantile_idx`; and
* :meth:`predict_with_bands` returning a
  :class:`opifex.uncertainty.types.PredictiveDistribution` with a
  populated :class:`PredictionInterval`.

This integration test pins those contracts plus the
``FNOConformalAdapterSpec`` capability declaration that matches the
operator and a coverage smoke check on a synthetic linear toy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
    UQNOBaseSolutionOperator,
    UQNOConformalCalibrator,
    UQNOResidualOperator,
)
from opifex.uncertainty.adapters import FNOConformalAdapterSpec
from opifex.uncertainty.losses import PointwiseQuantileLoss
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictionInterval, PredictiveDistribution


def _make_uqno(seed: int = 0) -> UncertaintyQuantificationNeuralOperator:
    return UncertaintyQuantificationNeuralOperator(
        base=UQNOBaseSolutionOperator(
            FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=8,
                modes=2,
                num_layers=2,
                rngs=nnx.Rngs(seed),
            )
        ),
        residual=UQNOResidualOperator(
            FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=8,
                modes=2,
                num_layers=2,
                rngs=nnx.Rngs(seed + 1),
            )
        ),
    )


def test_uqno_module_exports_match_conformal_pipeline() -> None:
    """`opifex.neural.operators.specialized.uqno` exports the three-stage surface."""
    from opifex.neural.operators.specialized import uqno as uqno_module

    for name in (
        "UncertaintyQuantificationNeuralOperator",
        "UQNOBaseSolutionOperator",
        "UQNOResidualOperator",
        "UQNOConformalCalibrator",
        "get_coeff_quantile_idx",
    ):
        assert hasattr(uqno_module, name), f"uqno module missing {name!r}"


def test_fno_conformal_adapter_spec_matches_uqno_strategy() -> None:
    """The conformal UQNO advertises a capability the adapter spec can describe."""
    spec = FNOConformalAdapterSpec()
    cap = spec.recommended_capability()
    assert cap.native_bayesian is False
    assert cap.default_strategy is DefaultStrategy.CONFORMAL
    assert cap.supports_conformal is True
    assert cap.supports_function_space is True


def test_conformal_pipeline_end_to_end_on_synthetic_data() -> None:
    """Calibrated bands cover the target across a small synthetic batch.

    Training the residual operator on synthetic noise that matches the
    base-residual structure should produce a calibrator whose
    ``scaling_factor`` is positive and finite, and bands that actually
    bracket a held-out sample.
    """
    uqno = _make_uqno()
    quantile_loss = PointwiseQuantileLoss(alpha=0.1, reduction="mean")

    # Synthetic dataset: input x, target y = base(x) + noise scaled by a
    # per-pixel pattern; the residual operator should learn the noise
    # magnitude pattern.
    key = jax.random.PRNGKey(0)
    k_x, k_y, k_noise = jax.random.split(key, 3)
    x = jax.random.normal(k_x, (16, 1, 8, 8))
    noise_scale = 0.05 + 0.05 * jax.random.uniform(k_y, (1, 1, 8, 8))
    base_pred_at_init = uqno.predict_base(x)
    y = base_pred_at_init + noise_scale * jax.random.normal(k_noise, x.shape)

    # Train the residual operator only (gradients through base are stopped
    # by jax.lax.stop_gradient inside UQNO.__call__).
    residual_opt = nnx.Optimizer(uqno.residual, optax.adam(1e-2), wrt=nnx.Param)

    @nnx.jit
    def residual_step(
        residual: UQNOResidualOperator,
        opt: nnx.Optimizer,
        x_in: jax.Array,
        y_diff_abs: jax.Array,
    ) -> jax.Array:
        def loss_fn(r: UQNOResidualOperator) -> jax.Array:
            quantile_widths = jnp.abs(r(x_in))
            return quantile_loss(y_pred=quantile_widths, y=y_diff_abs)

        loss, grads = nnx.value_and_grad(loss_fn)(residual)
        opt.update(residual, grads)
        return loss

    target_diffs = uqno.predict_base(x) - y
    for _ in range(40):
        residual_step(uqno.residual, residual_opt, x, target_diffs)

    # Calibrate on a held-out batch.
    k_calib_x, k_calib_noise = jax.random.split(jax.random.PRNGKey(11))
    x_calib = jax.random.normal(k_calib_x, (12, 1, 8, 8))
    y_calib = uqno.predict_base(x_calib) + noise_scale * jax.random.normal(
        k_calib_noise, x_calib.shape
    )
    calibrator = uqno.calibrate(x_calib, y_calib, alpha=0.1, delta=0.1)
    assert isinstance(calibrator, UQNOConformalCalibrator)
    assert bool(jnp.isfinite(calibrator.scaling_factor))
    assert float(calibrator.scaling_factor) > 0.0

    uqno = uqno.with_calibrator(calibrator)

    # Predict + verify the band contract on held-out data.
    k_test_x, k_test_noise = jax.random.split(jax.random.PRNGKey(99))
    x_test = jax.random.normal(k_test_x, (5, 1, 8, 8))
    y_test = uqno.predict_base(x_test) + noise_scale * jax.random.normal(k_test_noise, x_test.shape)
    dist = uqno.predict_with_bands(x_test)
    assert isinstance(dist, PredictiveDistribution)
    assert dist.interval is not None
    assert isinstance(dist.interval, PredictionInterval)
    # Bands must order correctly + lie around the mean.
    assert bool(jnp.all(dist.interval.lower <= dist.mean))
    assert bool(jnp.all(dist.mean <= dist.interval.upper))
    # And actually cover *some* meaningful fraction of held-out targets.
    in_band = (y_test >= dist.interval.lower) & (y_test <= dist.interval.upper)
    coverage = float(jnp.mean(in_band))
    # On a small synthetic batch we don't insist on hitting 1 - alpha
    # exactly, but coverage must be at least non-trivial.
    assert coverage > 0.0, "calibrated bands cover zero points on the test batch"


def test_predict_with_bands_is_nnx_jit_compatible() -> None:
    """`predict_with_bands` traces under ``nnx.jit`` once a calibrator is attached."""
    uqno = _make_uqno()
    x_calib = jax.random.normal(jax.random.PRNGKey(0), (8, 1, 8, 8))
    y_calib = uqno.predict_base(x_calib) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(1), x_calib.shape
    )
    uqno = uqno.with_calibrator(uqno.calibrate(x_calib, y_calib, alpha=0.1, delta=0.1))

    @nnx.jit
    def step(model: UncertaintyQuantificationNeuralOperator, x: jax.Array) -> jax.Array:
        return model.predict_with_bands(x).mean

    out = step(uqno, jnp.ones((2, 1, 8, 8)))
    assert out.shape == (2, 1, 8, 8)
    assert bool(jnp.all(jnp.isfinite(out)))
