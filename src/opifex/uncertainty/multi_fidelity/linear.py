r"""Linear multi-fidelity GP (Kennedy & O'Hagan 2000 AR(1)).

Models multiple fidelity levels through an auto-regressive linear chain

    f_0(x) = delta_0(x),
    f_i(x) = rho_i * f_{i-1}(x) + delta_i(x)    for i >= 1,

where each ``delta_i`` is an independent zero-mean GP with its own
length-scale and output-scale. The induced joint covariance between
fidelity levels ``i`` and ``j`` at inputs ``x``, ``x'`` is

    K_{ij}(x, x') = sum_{k=0}^{min(i, j)} [prod_{l=k+1}^{i} rho_l]
                                          [prod_{l=k+1}^{j} rho_l]
                                          k_k(x, x')

(Kennedy & O'Hagan §2.5; see also
``emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel`` — the
**PRIMARY** reference implementation, which this module mirrors as a
JAX-native pytree-free port).

Training data per fidelity level is supplied as the parallel tuples
``x_train_per_level`` and ``y_train_per_level``; the fit assembles the
augmented design matrix (input plus integer level column) and runs a
single Cholesky factorisation on the joint Gram matrix.

References
----------
* Kennedy, O'Hagan 2000 — *Predicting the output from a complex
  computer code when fast approximations are available*, Biometrika
  87(1).
* Le Gratiet, Garnier 2014 — *Recursive co-kriging model for design
  of computer experiments with multiple levels of fidelity*,
  Int. J. Uncertainty Quantification.
* ``emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel``
  (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp import rbf_kernel
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_LINEAR_MF_SOURCE_PACKAGE = "opifex.uncertainty.multi_fidelity"
_PSEUDO_NOISE_FLOOR: float = 1e-6


KernelFn = Callable[..., jax.Array]


def _product_scaling(
    scaling_factors: Sequence[float] | jax.Array,
    *,
    start: int,
    stop: int,
) -> jax.Array:
    r"""``prod_{l=start+1}^{stop} scaling_factors[l-1]`` (1.0 if range is empty).

    Indices follow the Kennedy-O'Hagan convention where
    ``scaling_factors[i-1]`` couples fidelity ``i-1`` into ``i``.
    """
    if stop <= start:
        return jnp.asarray(1.0)
    factors = jnp.asarray(scaling_factors)
    return jnp.prod(factors[start:stop])


def linear_multi_fidelity_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    lengthscales: Sequence[float],
    output_scales: Sequence[float],
    scaling_factors: Sequence[float],
    base_kernel_fn: KernelFn = rbf_kernel,
) -> jax.Array:
    r"""Kennedy & O'Hagan AR(1) joint kernel block matrix.

    Args:
        x1: ``(n1, d+1)`` augmented inputs whose final column is the
            integer fidelity level in ``{0, 1, ..., L-1}``.
        x2: ``(n2, d+1)`` augmented inputs.
        lengthscales: One length-scale per fidelity level.
        output_scales: One output-scale per fidelity level.
        scaling_factors: ``(L-1,)`` AR(1) coupling factors ``rho_1``,
            ..., ``rho_{L-1}``. Empty tuple for a single fidelity.
        base_kernel_fn: Base kernel callable; defaults to
            :func:`opifex.uncertainty.gp.rbf_kernel`.

    Returns:
        ``(n1, n2)`` joint Gram matrix.
    """
    num_levels = len(lengthscales)
    if len(output_scales) != num_levels:
        raise ValueError(
            f"output_scales / lengthscales length mismatch: {len(output_scales)} vs {num_levels}."
        )
    if len(scaling_factors) != num_levels - 1:
        raise ValueError(
            "scaling_factors length must equal num_levels-1: got "
            f"{len(scaling_factors)} for {num_levels} levels."
        )
    levels_1 = x1[:, -1].astype(jnp.int32)
    levels_2 = x2[:, -1].astype(jnp.int32)
    inputs_1 = x1[:, :-1]
    inputs_2 = x2[:, :-1]
    zeros_block: jax.Array = jnp.zeros((x1.shape[0], x2.shape[0]))
    base_gram_per_level: list[jax.Array] = [
        jnp.asarray(
            base_kernel_fn(
                inputs_1,
                inputs_2,
                lengthscale=float(lengthscales[k]),
                output_scale=float(output_scales[k]),
            )
        )
        for k in range(num_levels)
    ]
    gram: jax.Array = zeros_block
    for i in range(num_levels):
        mask_i = (levels_1 == i)[:, None]
        for j in range(num_levels):
            mask_j = (levels_2 == j)[None, :]
            mask_ij = mask_i & mask_j
            min_ij = min(i, j)
            block_value: jax.Array = zeros_block
            for k in range(min_ij + 1):
                coupling = _product_scaling(scaling_factors, start=k, stop=i) * _product_scaling(
                    scaling_factors, start=k, stop=j
                )
                block_value = block_value + coupling * base_gram_per_level[k]
            gram = jnp.where(mask_ij, block_value, gram)
    return gram


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearMultiFidelityGPState:
    """Fitted state for a linear (K&O AR(1)) multi-fidelity GP.

    Attributes:
        x_augmented: ``(n_total, d+1)`` augmented training inputs
            including the fidelity-level column.
        y_train: ``(n_total,)`` training targets (concatenated across
            levels in increasing fidelity order).
        cholesky: Lower-triangular Cholesky factor of the joint
            ``K + noise_std**2 I`` Gram matrix.
        alpha: Pre-solved ``(K + sigma**2 I)^{-1} y``.
        lengthscales: Per-level length-scales used at fit time.
        output_scales: Per-level output-scales used at fit time.
        scaling_factors: AR(1) coupling factors used at fit time.
        noise_std: Observation noise scale.
        base_kernel_fn: Kernel used internally (defaults to RBF).
    """

    x_augmented: jax.Array
    y_train: jax.Array
    cholesky: jax.Array
    alpha: jax.Array
    lengthscales: tuple[float, ...]
    output_scales: tuple[float, ...]
    scaling_factors: tuple[float, ...]
    noise_std: float
    base_kernel_fn: KernelFn = field(default=rbf_kernel)


def _augment_with_level(x: jax.Array, level: int) -> jax.Array:
    """Append a constant fidelity-level column to ``x``."""
    level_column = jnp.full((x.shape[0], 1), float(level))
    return jnp.concatenate([x, level_column], axis=1)


def fit_linear_multi_fidelity_gp(
    *,
    x_train_per_level: Sequence[jax.Array],
    y_train_per_level: Sequence[jax.Array],
    lengthscales: Sequence[float],
    output_scales: Sequence[float],
    scaling_factors: Sequence[float],
    noise_std: float,
    base_kernel_fn: KernelFn = rbf_kernel,
) -> LinearMultiFidelityGPState:
    r"""Fit a linear (K&O AR(1)) multi-fidelity GP.

    Args:
        x_train_per_level: Per-fidelity training inputs (each ``(n_i, d)``).
        y_train_per_level: Per-fidelity training targets (each ``(n_i,)``).
        lengthscales: ``(L,)`` per-level length-scales.
        output_scales: ``(L,)`` per-level output-scales.
        scaling_factors: ``(L-1,)`` AR(1) coupling factors.
        noise_std: Observation noise scale (also acts as the joint
            jitter for Cholesky stability).
        base_kernel_fn: Optional base kernel; defaults to
            :func:`opifex.uncertainty.gp.rbf_kernel`.

    Returns:
        :class:`LinearMultiFidelityGPState` with Cholesky + ``alpha``.
    """
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    if len(x_train_per_level) != len(y_train_per_level):
        raise ValueError(
            f"x_train_per_level and y_train_per_level must have equal length; got "
            f"{len(x_train_per_level)} vs {len(y_train_per_level)}."
        )
    augmented_per_level = [
        _augment_with_level(x_train_per_level[level], level)
        for level in range(len(x_train_per_level))
    ]
    x_augmented = jnp.concatenate(augmented_per_level, axis=0)
    y_train = jnp.concatenate(list(y_train_per_level), axis=0)
    gram = linear_multi_fidelity_kernel(
        x_augmented,
        x_augmented,
        lengthscales=lengthscales,
        output_scales=output_scales,
        scaling_factors=scaling_factors,
        base_kernel_fn=base_kernel_fn,
    )
    gram_jittered = gram + (noise_std**2) * jnp.eye(gram.shape[0])
    cholesky = jnp.linalg.cholesky(gram_jittered)
    alpha = jax.scipy.linalg.cho_solve((cholesky, True), y_train)
    return LinearMultiFidelityGPState(
        x_augmented=x_augmented,
        y_train=y_train,
        cholesky=cholesky,
        alpha=alpha,
        lengthscales=tuple(float(value) for value in lengthscales),
        output_scales=tuple(float(value) for value in output_scales),
        scaling_factors=tuple(float(value) for value in scaling_factors),
        noise_std=float(noise_std),
        base_kernel_fn=base_kernel_fn,
    )


def predict_linear_multi_fidelity_gp(
    *,
    state: LinearMultiFidelityGPState,
    x_test: jax.Array,
    target_level: int,
) -> PredictiveDistribution:
    r"""Predict the GP posterior at ``x_test`` for the given fidelity level.

    Args:
        state: Fitted linear-MF state.
        x_test: ``(m, d)`` test inputs (without level column).
        target_level: Integer fidelity level at which to predict.

    Returns:
        :class:`PredictiveDistribution` with mean and variance.
    """
    augmented_test = _augment_with_level(x_test, target_level)
    k_cross = linear_multi_fidelity_kernel(
        augmented_test,
        state.x_augmented,
        lengthscales=state.lengthscales,
        output_scales=state.output_scales,
        scaling_factors=state.scaling_factors,
        base_kernel_fn=state.base_kernel_fn,
    )
    k_test = linear_multi_fidelity_kernel(
        augmented_test,
        augmented_test,
        lengthscales=state.lengthscales,
        output_scales=state.output_scales,
        scaling_factors=state.scaling_factors,
        base_kernel_fn=state.base_kernel_fn,
    )
    mean = k_cross @ state.alpha
    v_solve = jax.scipy.linalg.solve_triangular(state.cholesky, k_cross.T, lower=True)
    variance = jnp.diag(k_test) - jnp.sum(v_solve**2, axis=0)
    variance = jnp.clip(variance, a_min=_PSEUDO_NOISE_FLOOR)
    return gaussian_process_predictive(
        mean,
        variance,
        epistemic=variance,
        total_uncertainty=variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_LINEAR_MF_SOURCE_PACKAGE,
            extra=(
                ("estimator", "linear_multi_fidelity_gp"),
                (
                    "paper",
                    "Kennedy & O'Hagan 2000 AR(1) multi-fidelity GP "
                    "(emukit linear_multi_fidelity_kernel mirror)",
                ),
                ("target_level", str(target_level)),
            ),
        ),
    )


__all__ = [
    "LinearMultiFidelityGPState",
    "fit_linear_multi_fidelity_gp",
    "linear_multi_fidelity_kernel",
    "predict_linear_multi_fidelity_gp",
]
