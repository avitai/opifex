r"""Non-linear multi-fidelity GP (NARGP, Perdikaris+ 2017).

Implements the Perdikaris autoregressive non-linear multi-fidelity
chain

    f_0(x) ~ GP(0, k_0(x, x')),
    f_i(x) = g_i(x, f_{i-1}(x))                 for i >= 1,

where each ``g_i`` is an independent GP whose input is the
*augmented* tuple ``(x, f_{i-1}(x))`` — feeding the previous-fidelity
output as an extra dimension lets the model capture **non-linear**
cross-fidelity relationships that the K&O AR(1) chain cannot.

Training proceeds **greedily and recursively**:

1. Fit ``f_0`` to ``(x_low, y_low)``.
2. Predict ``f_0`` at the level-1 training inputs to obtain a
   point-evaluated "low-fidelity feature" column.
3. Fit ``g_1`` to ``([x_high, f_0(x_high)], y_high)``.
4. Repeat for additional fidelity levels.

**Prediction** at level ``i`` propagates posterior uncertainty by
Monte-Carlo sampling from the level-``(i-1)`` predictive at the test
inputs, feeding each sample as the ``(d+1)``-th input column of the
level-``i`` GP, and averaging the level-``i`` mean / variance across
samples.

References
----------
* Perdikaris, Raissi, Damianou, Lawrence, Karniadakis 2017 — *Nonlinear
  information fusion algorithms for data-efficient multi-fidelity
  modelling*, Proc. R. Soc. A.
* ``emukit.multi_fidelity.models.non_linear_multi_fidelity_model``
  (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp import ExactGPState, fit_exact_gp, predict_exact_gp
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_NONLINEAR_MF_SOURCE_PACKAGE = "opifex.uncertainty.multi_fidelity"
_PSEUDO_NOISE_FLOOR: float = 1e-6


@dataclass(frozen=True, slots=True, kw_only=True)
class NonLinearMultiFidelityGPState:
    """Fitted state for a NARGP non-linear multi-fidelity GP.

    Stores one :class:`ExactGPState` per fidelity level. Levels >= 1
    are fitted on the augmented design ``[x, f_{i-1}(x)]``.

    Attributes:
        level_states: One :class:`ExactGPState` per fidelity level.
        lengthscales: Per-level length-scale used at fit time.
        output_scales: Per-level output-scale used at fit time.
        noise_std: Observation noise scale used across levels.
    """

    level_states: tuple[ExactGPState, ...]
    lengthscales: tuple[float, ...]
    output_scales: tuple[float, ...]
    noise_std: float


def fit_nonlinear_multi_fidelity_gp(
    *,
    x_train_per_level: Sequence[jax.Array],
    y_train_per_level: Sequence[jax.Array],
    lengthscales: Sequence[float],
    output_scales: Sequence[float],
    noise_std: float,
) -> NonLinearMultiFidelityGPState:
    r"""Greedy recursive fit of a NARGP non-linear multi-fidelity GP.

    Args:
        x_train_per_level: Per-fidelity training inputs (each ``(n_i, d)``).
        y_train_per_level: Per-fidelity training targets (each ``(n_i,)``).
        lengthscales: ``(L,)`` per-level length-scales for the
            independent ExactGPs.
        output_scales: ``(L,)`` per-level output-scales.
        noise_std: Observation noise scale (shared across levels —
            also acts as the joint jitter for Cholesky stability).

    Returns:
        :class:`NonLinearMultiFidelityGPState` with one fitted
        :class:`ExactGPState` per fidelity level.
    """
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    num_levels = len(x_train_per_level)
    if len(y_train_per_level) != num_levels:
        raise ValueError("x_train_per_level / y_train_per_level length mismatch.")
    if len(lengthscales) != num_levels or len(output_scales) != num_levels:
        raise ValueError("lengthscales / output_scales must have one entry per level.")
    level_states: list[ExactGPState] = []
    for level_index in range(num_levels):
        x_level = jnp.asarray(x_train_per_level[level_index])
        y_level = jnp.asarray(y_train_per_level[level_index])
        if level_index == 0:
            augmented_x = x_level
        else:
            prev_pred = predict_exact_gp(
                state=level_states[level_index - 1],
                x_test=x_level,
            )
            augmented_x = jnp.concatenate([x_level, prev_pred.mean.reshape(-1, 1)], axis=1)
        state = fit_exact_gp(
            x_train=augmented_x,
            y_train=y_level,
            lengthscale=float(lengthscales[level_index]),
            output_scale=float(output_scales[level_index]),
            noise_std=float(noise_std),
        )
        level_states.append(state)
    return NonLinearMultiFidelityGPState(
        level_states=tuple(level_states),
        lengthscales=tuple(float(value) for value in lengthscales),
        output_scales=tuple(float(value) for value in output_scales),
        noise_std=float(noise_std),
    )


def predict_nonlinear_multi_fidelity_gp(
    *,
    state: NonLinearMultiFidelityGPState,
    x_test: jax.Array,
    target_level: int,
    num_samples: int = 64,
    rng_key: jax.Array,
) -> PredictiveDistribution:
    r"""Predict at ``x_test`` for the given fidelity level via MC propagation.

    Uncertainty propagates through the chain by Monte-Carlo: at each
    level above 0, ``num_samples`` posterior samples of the previous
    level are fed as the ``(d+1)``-th input column, and the resulting
    per-sample predictive means / variances are aggregated via the
    standard mixture moments.

    Args:
        state: Fitted :class:`NonLinearMultiFidelityGPState`.
        x_test: ``(m, d)`` test inputs (without level column).
        target_level: Integer fidelity level at which to predict.
        num_samples: Monte-Carlo samples used to propagate
            previous-level uncertainty.
        rng_key: JAX PRNG key for sampling.

    Returns:
        :class:`PredictiveDistribution` with mean / variance.
    """
    if target_level == 0:
        level_pred = predict_exact_gp(state=state.level_states[0], x_test=x_test)
        return _wrap_predict(level_pred, target_level=target_level)

    previous_predictive = predict_nonlinear_multi_fidelity_gp(
        state=state,
        x_test=x_test,
        target_level=target_level - 1,
        num_samples=num_samples,
        rng_key=rng_key,
    )
    if previous_predictive.variance is None:
        raise RuntimeError("Previous-level predictive missing variance.")
    previous_mean = previous_predictive.mean
    previous_std = jnp.sqrt(previous_predictive.variance)
    sample_keys = jax.random.split(rng_key, num_samples)

    def per_sample_prediction(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Predict the target-level GP for one Monte Carlo draw of the lower fidelity."""
        noise = jax.random.normal(key, previous_mean.shape)
        previous_sample = previous_mean + previous_std * noise
        augmented_x = jnp.concatenate([x_test, previous_sample.reshape(-1, 1)], axis=1)
        level_pred = predict_exact_gp(state=state.level_states[target_level], x_test=augmented_x)
        if level_pred.variance is None:
            raise RuntimeError("Level predictive missing variance.")
        return level_pred.mean, level_pred.variance

    means, variances = jax.vmap(per_sample_prediction)(sample_keys)
    pooled_mean = jnp.mean(means, axis=0)
    # Law of total variance:
    # Var(y) = E_q[Var(y | sample)] + Var_q(E[y | sample]).
    pooled_variance = jnp.mean(variances, axis=0) + jnp.var(means, axis=0)
    pooled_variance = jnp.clip(pooled_variance, a_min=_PSEUDO_NOISE_FLOOR)
    return gaussian_process_predictive(
        pooled_mean,
        pooled_variance,
        epistemic=pooled_variance,
        total_uncertainty=pooled_variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_NONLINEAR_MF_SOURCE_PACKAGE,
            extra=(
                ("estimator", "nonlinear_multi_fidelity_gp"),
                (
                    "paper",
                    "Perdikaris, Raissi, Damianou, Lawrence, Karniadakis 2017 "
                    "(NARGP autoregressive non-linear multi-fidelity GP)",
                ),
                ("target_level", str(target_level)),
            ),
        ),
    )


def _wrap_predict(
    predictive: PredictiveDistribution, *, target_level: int
) -> PredictiveDistribution:
    """Refresh metadata to advertise the non-linear-MF route at level 0."""
    return PredictiveDistribution(
        mean=predictive.mean,
        variance=predictive.variance,
        epistemic=predictive.epistemic,
        aleatoric=predictive.aleatoric,
        total_uncertainty=predictive.total_uncertainty,
        samples=predictive.samples,
        covariance=predictive.covariance,
        quantiles=predictive.quantiles,
        interval=predictive.interval,
        prediction_set=predictive.prediction_set,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_NONLINEAR_MF_SOURCE_PACKAGE,
            extra=(
                ("estimator", "nonlinear_multi_fidelity_gp"),
                (
                    "paper",
                    "Perdikaris, Raissi, Damianou, Lawrence, Karniadakis 2017 "
                    "(NARGP autoregressive non-linear multi-fidelity GP)",
                ),
                ("target_level", str(target_level)),
            ),
        ),
    )


__all__ = [
    "NonLinearMultiFidelityGPState",
    "fit_nonlinear_multi_fidelity_gp",
    "predict_nonlinear_multi_fidelity_gp",
]
