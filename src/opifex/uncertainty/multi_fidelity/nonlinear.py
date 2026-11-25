r"""Non-linear multi-fidelity GP (NARGP, Perdikaris et al. 2017).

Implements the autoregressive non-linear multi-fidelity Gaussian process chain of
Perdikaris, Raissi, Damianou, Lawrence & Karniadakis (2017, Eq. 2.12):

.. math::

    f_0(x) &\sim \mathcal{GP}\!\left(0,\, k_0(x, x'; \ell_0, \sigma_0)\right), \\
    f_i(x) &= g_i\!\left(x,\, f_{*,i-1}(x)\right), \qquad i = 1, \dots, L - 1,

where each higher-fidelity GP ``g_i`` is fitted on the raw augmented input
``z = [x, f_{*,i-1}(x)]`` with the two-term structured composite kernel

.. math::

    k_i([x, f_*], [x', f_*'])
    = k_{\rho_i}(x, x'; \ell_{\rho,i}, \sigma_{\rho,i})
      \,k_{f_i}(f_*, f_*'; \ell_{f,i}, 1)
      + k_{\delta_i}(x, x'; \ell_{\delta,i}, \sigma_{\delta,i}).

Fixing unit output scale on ``k_{f_i}`` keeps the product variance
``\sigma_{\rho,i}^2 \cdot 1`` identifiable without redundant scale factors in the
multiplicative interaction term (Perdikaris et al. 2017, Eq. 2.12). Posterior
predictions at level ``target_level`` propagate ``num_samples`` particle trajectories
across levels ``1 .. target_level`` in ``O(num_samples * L)`` batched evaluations and
combine final-level moments via the law of total variance.

References:
----------
* Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., & Karniadakis, G. E.
  (2017). *Nonlinear information fusion algorithms for data-efficient
  multi-fidelity modelling*. Proc. R. Soc. A 473(2198), 20160751.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from opifex.core.capabilities import DefaultStrategy
from opifex.core.solver.implicit import infer_inexact_dtype
from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp import (
    ExactGPState,
    fit_exact_gp,
    kernel_diagonal,
    predict_exact_gp,
    rbf_kernel,
)
from opifex.uncertainty.gp._validation import nonpositive_message, PositiveScalar
from opifex.uncertainty.multi_fidelity.linear import _target_level_error
from opifex.uncertainty.types import PredictiveDistribution


_NONLINEAR_MF_SOURCE_PACKAGE = "opifex.uncertainty.multi_fidelity"


def nargp_composite_kernel(
    z1: jax.Array,
    z2: jax.Array,
    *,
    interaction_lengthscale: PositiveScalar,
    interaction_output_scale: PositiveScalar,
    f_lengthscale: PositiveScalar,
    bias_lengthscale: PositiveScalar,
    bias_output_scale: PositiveScalar,
    base_kernel_fn: Callable[..., jax.Array] = rbf_kernel,
    f_kernel_fn: Callable[..., jax.Array] | None = None,
) -> jax.Array:
    r"""Evaluate the Perdikaris et al. (2017, Eq. 2.12) two-term NARGP kernel.

    Evaluates ``base_kernel_fn`` strictly on ``z[:, :-1]`` and ``f_kernel_fn`` (with
    unit output scale for identifiability) on the raw previous-fidelity column
    ``z[:, -1:]``:

    .. math::

        k_i([x, f_*], [x', f_*'])
        = k_{\rho}(x, x'; \ell_\rho, \sigma_\rho)\,k_f(f_*, f_*'; \ell_f, 1)
          + k_{\delta}(x, x'; \ell_\delta, \sigma_\delta).

    Args:
        z1: Augmented inputs of shape ``(n, d + 1)``.
        z2: Augmented inputs of shape ``(m, d + 1)``.
        interaction_lengthscale: Length-scale ``\ell_\rho`` of ``k_\rho`` on ``x``.
        interaction_output_scale: Output-scale ``\sigma_\rho`` of ``k_\rho`` on ``x``.
        f_lengthscale: Length-scale ``\ell_f`` of ``k_f`` on ``f_*``.
        bias_lengthscale: Length-scale ``\ell_\delta`` of additive discrepancy ``k_\delta``.
        bias_output_scale: Output-scale ``\sigma_\delta`` of additive discrepancy ``k_\delta``.
        base_kernel_fn: Kernel callable on ``d``-dimensional ``x`` inputs.
        f_kernel_fn: Optional kernel callable on 1-D ``f_*`` inputs (defaults to
            ``base_kernel_fn``).

    Returns:
        ``(n, m)`` composite covariance matrix.
    """
    resolved_f_kernel = f_kernel_fn if f_kernel_fn is not None else base_kernel_fn
    x1_base, f1 = z1[:, :-1], z1[:, -1:]
    x2_base, f2 = z2[:, :-1], z2[:, -1:]
    k_rho = base_kernel_fn(
        x1_base,
        x2_base,
        lengthscale=interaction_lengthscale,
        output_scale=interaction_output_scale,
    )
    dtype = k_rho.dtype
    k_f = resolved_f_kernel(
        f1.astype(dtype),
        f2.astype(dtype),
        lengthscale=f_lengthscale,
        output_scale=jnp.asarray(1.0, dtype=dtype),
    )
    k_bias = base_kernel_fn(
        x1_base,
        x2_base,
        lengthscale=bias_lengthscale,
        output_scale=bias_output_scale,
    )
    return k_rho * k_f + k_bias


@dataclass(frozen=True, slots=True)
class _BoundCompositeKernel:
    """Static value-equal kernel adapter unpacking ``ExactGPState`` composite scales."""

    base_kernel_fn: Callable[..., jax.Array]
    f_kernel_fn: Callable[..., jax.Array]

    def __call__(
        self,
        z1: jax.Array,
        z2: jax.Array,
        *,
        lengthscale: PositiveScalar,
        output_scale: PositiveScalar,
    ) -> jax.Array:
        """Unpack ``(3,)`` length-scales and ``(2,)`` output-scales into scalar kwargs."""
        ls_vec = jnp.asarray(lengthscale)
        os_vec = jnp.asarray(output_scale)
        return nargp_composite_kernel(
            z1,
            z2,
            interaction_lengthscale=ls_vec[0],
            interaction_output_scale=os_vec[0],
            f_lengthscale=ls_vec[1],
            bias_lengthscale=ls_vec[2],
            bias_output_scale=os_vec[1],
            base_kernel_fn=self.base_kernel_fn,
            f_kernel_fn=self.f_kernel_fn,
        )


@struct.dataclass(slots=True, kw_only=True)
class NonLinearMultiFidelityGPState:
    """Fitted state PyTree for a Perdikaris et al. (2017) NARGP emulator.

    Stores all numeric hyperparameters as traceable ``jax.Array`` PyTree leaves
    and keeps only Python kernel callables in static ``pytree_node=False``
    metadata so returning or passing the state across ``jax.jit``, ``jax.vmap``,
    and ``jax.grad`` boundaries never leaks tracers.

    Attributes:
        level_states: One :class:`ExactGPState` per fidelity level.
        lengthscales: Per-level base / interaction length-scales ``(L,)``.
        output_scales: Per-level base / interaction output-scales ``(L,)``.
        noise_stds: Per-level observation noise scales ``(L,)``.
        f_lengthscales: Per-transition previous-fidelity length-scales ``(L - 1,)``.
        bias_lengthscales: Per-transition additive discrepancy length-scales ``(L - 1,)``.
        bias_output_scales: Per-transition additive discrepancy output-scales ``(L - 1,)``.
        base_kernel_fn: Kernel callable on ``d``-dimensional ``x`` inputs.
        f_kernel_fn: Kernel callable on 1-D previous-fidelity ``f_*`` inputs.
        has_explicit_kernels: Whether custom per-level ``kernel_fns`` were supplied.
    """

    level_states: tuple[ExactGPState, ...]
    lengthscales: tuple[jax.Array, ...]
    output_scales: tuple[jax.Array, ...]
    noise_stds: tuple[jax.Array, ...]
    f_lengthscales: tuple[jax.Array, ...] = ()
    bias_lengthscales: tuple[jax.Array, ...] = ()
    bias_output_scales: tuple[jax.Array, ...] = ()
    base_kernel_fn: Callable[..., jax.Array] = struct.field(pytree_node=False, default=rbf_kernel)
    f_kernel_fn: Callable[..., jax.Array] = struct.field(pytree_node=False, default=rbf_kernel)
    has_explicit_kernels: bool = struct.field(pytree_node=False, default=False)


def _validate_positive_sequence(
    name: str,
    values: Sequence[PositiveScalar],
    expected_len: int,
) -> None:
    """Validate sequence length and concrete positivity of every entry."""
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}; got {len(values)}.")
    for idx, val in enumerate(values):
        if message := nonpositive_message(f"{name}[{idx}]", val):
            raise ValueError(message)


def _resolve_noise_stds(
    noise_std: PositiveScalar | Sequence[PositiveScalar],
    num_levels: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, ...]:
    """Validate scalar or per-level observation noise and cast to ``dtype``."""
    if isinstance(noise_std, Sequence):
        _validate_positive_sequence("noise_std", noise_std, num_levels)
        return tuple(jnp.asarray(val, dtype=dtype) for val in noise_std)
    if message := nonpositive_message("noise_std", noise_std):
        raise ValueError(message)
    scalar_noise = jnp.asarray(noise_std, dtype=dtype)
    return (scalar_noise,) * num_levels


def _resolve_transition_hyperparams(
    *,
    num_levels: int,
    lengthscales: Sequence[PositiveScalar],
    output_scales: Sequence[PositiveScalar],
    f_lengthscales: Sequence[PositiveScalar] | None,
    bias_lengthscales: Sequence[PositiveScalar] | None,
    bias_output_scales: Sequence[PositiveScalar] | None,
    dtype: jnp.dtype,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Validate and resolve ``(L - 1)`` transition hyperparameter tuples."""
    n_trans = num_levels - 1
    raw_f_ls = (1.0,) * n_trans if f_lengthscales is None else f_lengthscales
    raw_b_ls = tuple(lengthscales[1:]) if bias_lengthscales is None else bias_lengthscales
    raw_b_os = tuple(output_scales[1:]) if bias_output_scales is None else bias_output_scales

    _validate_positive_sequence("f_lengthscales", raw_f_ls, n_trans)
    _validate_positive_sequence("bias_lengthscales", raw_b_ls, n_trans)
    _validate_positive_sequence("bias_output_scales", raw_b_os, n_trans)

    return (
        tuple(jnp.asarray(v, dtype=dtype) for v in raw_f_ls),
        tuple(jnp.asarray(v, dtype=dtype) for v in raw_b_ls),
        tuple(jnp.asarray(v, dtype=dtype) for v in raw_b_os),
    )


def _predict_composite_samples(
    *,
    lvl_state: ExactGPState,
    x_query: jax.Array,
    f_samples: jax.Array,
    interaction_lengthscale: jax.Array,
    interaction_output_scale: jax.Array,
    f_lengthscale: jax.Array,
    bias_lengthscale: jax.Array,
    bias_output_scale: jax.Array,
    base_kernel_fn: Callable[..., jax.Array],
    f_kernel_fn: Callable[..., jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Evaluate level ``>= 1`` GP posterior moments across ``(S, m)`` trajectories in one solve."""
    num_samples, num_points = f_samples.shape
    dtype = x_query.dtype
    x_train_base = lvl_state.x_train[:, :-1]
    f_train = lvl_state.x_train[:, -1:]
    n_train = x_train_base.shape[0]
    one = jnp.asarray(1.0, dtype=dtype)

    # Evaluate d-dimensional base kernels once on (n_train, m) and (m,), not S times.
    k_rho_cross = base_kernel_fn(
        x_train_base,
        x_query,
        lengthscale=interaction_lengthscale,
        output_scale=interaction_output_scale,
    )
    k_bias_cross = base_kernel_fn(
        x_train_base,
        x_query,
        lengthscale=bias_lengthscale,
        output_scale=bias_output_scale,
    )
    k_rho_diag = kernel_diagonal(
        base_kernel_fn,
        x_query,
        lengthscale=interaction_lengthscale,
        output_scale=interaction_output_scale,
    )
    k_bias_diag = kernel_diagonal(
        base_kernel_fn,
        x_query,
        lengthscale=bias_lengthscale,
        output_scale=bias_output_scale,
    )

    f_flat = f_samples.reshape(num_samples * num_points, 1)
    k_f_cross = f_kernel_fn(
        f_train,
        f_flat,
        lengthscale=f_lengthscale,
        output_scale=one,
    ).reshape(n_train, num_samples, num_points)
    k_f_diag = kernel_diagonal(
        f_kernel_fn,
        f_flat,
        lengthscale=f_lengthscale,
        output_scale=one,
    ).reshape(num_samples, num_points)

    k_cross = (k_rho_cross[:, None, :] * k_f_cross + k_bias_cross[:, None, :]).reshape(
        n_train, num_samples * num_points
    )
    k_diag = (k_rho_diag[None, :] * k_f_diag + k_bias_diag[None, :]).reshape(
        num_samples * num_points
    )

    flat_mean = k_cross.T @ lvl_state.alpha
    v_mat = jax.scipy.linalg.solve_triangular(lvl_state.cholesky, k_cross, lower=True)
    flat_var = jnp.maximum(k_diag - jnp.sum(v_mat * v_mat, axis=0), jnp.asarray(0.0, dtype=dtype))
    return flat_mean.reshape(num_samples, num_points), flat_var.reshape(num_samples, num_points)


def _predict_level_samples(
    state: NonLinearMultiFidelityGPState,
    level_index: int,
    x_query: jax.Array,
    f_samples: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Predict ``(S, m)`` posterior means and variances at ``level_index >= 1``."""
    num_samples, num_points = f_samples.shape
    if state.has_explicit_kernels:
        x_tiled = jnp.broadcast_to(
            x_query[None, :, :], (num_samples, num_points, x_query.shape[1])
        ).reshape(num_samples * num_points, x_query.shape[1])
        aug_batch = jnp.concatenate(
            [x_tiled, f_samples.reshape(num_samples * num_points, 1)], axis=1
        )
        pred = predict_exact_gp(state=state.level_states[level_index], x_test=aug_batch)
        if pred.variance is None:
            raise RuntimeError("Level predictive missing variance.")
        return pred.mean.reshape(num_samples, num_points), pred.variance.reshape(
            num_samples, num_points
        )

    return _predict_composite_samples(
        lvl_state=state.level_states[level_index],
        x_query=x_query,
        f_samples=f_samples,
        interaction_lengthscale=state.lengthscales[level_index],
        interaction_output_scale=state.output_scales[level_index],
        f_lengthscale=state.f_lengthscales[level_index - 1],
        bias_lengthscale=state.bias_lengthscales[level_index - 1],
        bias_output_scale=state.bias_output_scales[level_index - 1],
        base_kernel_fn=state.base_kernel_fn,
        f_kernel_fn=state.f_kernel_fn,
    )


def _fit_composite_level(
    *,
    augmented_x: jax.Array,
    y_level: jax.Array,
    interaction_lengthscale: jax.Array,
    interaction_output_scale: jax.Array,
    f_lengthscale: jax.Array,
    bias_lengthscale: jax.Array,
    bias_output_scale: jax.Array,
    noise_std: jax.Array,
    base_kernel_fn: Callable[..., jax.Array],
    f_kernel_fn: Callable[..., jax.Array],
) -> ExactGPState:
    """Fit a higher-fidelity GP with the Perdikaris (2017) two-term composite kernel."""
    k_train = nargp_composite_kernel(
        augmented_x,
        augmented_x,
        interaction_lengthscale=interaction_lengthscale,
        interaction_output_scale=interaction_output_scale,
        f_lengthscale=f_lengthscale,
        bias_lengthscale=bias_lengthscale,
        bias_output_scale=bias_output_scale,
        base_kernel_fn=base_kernel_fn,
        f_kernel_fn=f_kernel_fn,
    )
    n_obs = k_train.shape[0]
    gram = k_train + (noise_std**2) * jnp.eye(n_obs, dtype=k_train.dtype)
    cholesky = jnp.linalg.cholesky(gram)
    alpha = jax.scipy.linalg.cho_solve((cholesky, True), y_level)
    return ExactGPState(
        x_train=augmented_x,
        y_train=y_level,
        cholesky=cholesky,
        alpha=alpha,
        lengthscale=jnp.stack([interaction_lengthscale, f_lengthscale, bias_lengthscale]),
        output_scale=jnp.stack([interaction_output_scale, bias_output_scale]),
        noise_std=noise_std,
        kernel_fn=_BoundCompositeKernel(base_kernel_fn, f_kernel_fn),
    )


def _validate_explicit_kernel_exclusivity(
    *,
    kernel_fns: Sequence[Callable[..., jax.Array]] | None,
    num_levels: int,
    f_lengthscales: Sequence[PositiveScalar] | None,
    bias_lengthscales: Sequence[PositiveScalar] | None,
    bias_output_scales: Sequence[PositiveScalar] | None,
    base_kernel_fn: Callable[..., jax.Array],
    f_kernel_fn: Callable[..., jax.Array] | None,
) -> None:
    """Validate ``kernel_fns`` length and reject conflicting composite-only arguments."""
    if kernel_fns is None:
        return
    if len(kernel_fns) != num_levels:
        raise ValueError("kernel_fns must have one entry per fidelity level.")
    has_composite_arg = (
        f_lengthscales is not None
        or bias_lengthscales is not None
        or bias_output_scales is not None
        or f_kernel_fn is not None
        or base_kernel_fn is not rbf_kernel
    )
    if has_composite_arg:
        raise ValueError(
            "kernel_fns cannot be combined with composite-kernel arguments "
            "(f_lengthscales, bias_lengthscales, bias_output_scales, base_kernel_fn, "
            "or f_kernel_fn)."
        )


def fit_nonlinear_multi_fidelity_gp(
    *,
    x_train_per_level: Sequence[jax.Array],
    y_train_per_level: Sequence[jax.Array],
    lengthscales: Sequence[PositiveScalar],
    output_scales: Sequence[PositiveScalar],
    noise_std: PositiveScalar | Sequence[PositiveScalar],
    f_lengthscales: Sequence[PositiveScalar] | None = None,
    bias_lengthscales: Sequence[PositiveScalar] | None = None,
    bias_output_scales: Sequence[PositiveScalar] | None = None,
    base_kernel_fn: Callable[..., jax.Array] = rbf_kernel,
    f_kernel_fn: Callable[..., jax.Array] | None = None,
    kernel_fns: Sequence[Callable[..., jax.Array]] | None = None,
) -> NonLinearMultiFidelityGPState:
    r"""Greedy recursive fit of a Perdikaris et al. (2017) NARGP multi-fidelity GP.

    Args:
        x_train_per_level: Per-fidelity training inputs (each ``(n_i, d)``).
        y_train_per_level: Per-fidelity training targets (each ``(n_i,)``).
        lengthscales: ``(L,)`` base length-scale at level ``0`` and interaction
            kernel ``k_{\rho_i}`` length-scales on ``x`` at levels ``>= 1``.
        output_scales: ``(L,)`` base output-scale at level ``0`` and interaction
            kernel ``k_{\rho_i}`` output-scales on ``x`` at levels ``>= 1``.
        noise_std: Scalar observation noise shared across levels or ``(L,)``
            per-level observation noise scales.
        f_lengthscales: Optional ``(L - 1,)`` length-scales for ``k_{f_i}`` on raw
            unnormalized ``f_{*, i-1}``; defaults to ``(1.0,) * (L - 1)``.
        bias_lengthscales: Optional ``(L - 1,)`` length-scales for discrepancy
            kernel ``k_{\delta_i}`` on ``x``; defaults to ``lengthscales[1:]``.
        bias_output_scales: Optional ``(L - 1,)`` output-scales for discrepancy
            kernel ``k_{\delta_i}`` on ``x``; defaults to ``output_scales[1:]``.
        base_kernel_fn: Base kernel on ``d``-dimensional ``x`` inputs.
        f_kernel_fn: Optional 1-D kernel on ``f_{*, i-1}`` inputs (defaults to
            ``base_kernel_fn``).
        kernel_fns: Optional explicit sequence of ``L`` per-level kernel callables
            operating directly on ``x`` (level 0) and ``[x, f_{*, i-1}]`` (levels >= 1).

    Returns:
        :class:`NonLinearMultiFidelityGPState` with one fitted :class:`ExactGPState`
        per fidelity level and PyTree-leaf hyperparameters.

    Raises:
        ValueError: When ``x_train_per_level`` is empty, sequence lengths mismatch,
            or concrete scale/noise hyperparameters are non-positive.
    """
    num_levels = len(x_train_per_level)
    if num_levels < 1:
        raise ValueError("x_train_per_level must contain at least 1 fidelity level.")
    if len(y_train_per_level) != num_levels:
        raise ValueError("x_train_per_level / y_train_per_level length mismatch.")
    _validate_positive_sequence("lengthscales", lengthscales, num_levels)
    _validate_positive_sequence("output_scales", output_scales, num_levels)
    _validate_explicit_kernel_exclusivity(
        kernel_fns=kernel_fns,
        num_levels=num_levels,
        f_lengthscales=f_lengthscales,
        bias_lengthscales=bias_lengthscales,
        bias_output_scales=bias_output_scales,
        base_kernel_fn=base_kernel_fn,
        f_kernel_fn=f_kernel_fn,
    )

    dtype = infer_inexact_dtype(
        x_train_per_level,
        y_train_per_level,
        lengthscales,
        output_scales,
        noise_std,
        f_lengthscales,
        bias_lengthscales,
        bias_output_scales,
    )
    ls_arrs = tuple(jnp.asarray(v, dtype=dtype) for v in lengthscales)
    os_arrs = tuple(jnp.asarray(v, dtype=dtype) for v in output_scales)
    noise_arrs = _resolve_noise_stds(noise_std, num_levels, dtype)
    f_ls_arrs, b_ls_arrs, b_os_arrs = _resolve_transition_hyperparams(
        num_levels=num_levels,
        lengthscales=lengthscales,
        output_scales=output_scales,
        f_lengthscales=f_lengthscales,
        bias_lengthscales=bias_lengthscales,
        bias_output_scales=bias_output_scales,
        dtype=dtype,
    )
    resolved_f_kernel = f_kernel_fn if f_kernel_fn is not None else base_kernel_fn
    has_explicit = kernel_fns is not None

    level_states: list[ExactGPState] = []
    for lvl in range(num_levels):
        x_lvl = jnp.asarray(x_train_per_level[lvl], dtype=dtype)
        y_lvl = jnp.asarray(y_train_per_level[lvl], dtype=dtype)
        if lvl == 0:
            lvl0_kernel = kernel_fns[0] if kernel_fns is not None else base_kernel_fn
            st_lvl = fit_exact_gp(
                x_train=x_lvl,
                y_train=y_lvl,
                lengthscale=ls_arrs[0],
                output_scale=os_arrs[0],
                noise_std=noise_arrs[0],
                kernel_fn=lvl0_kernel,
            )
            level_states.append(st_lvl)
            continue

        partial_state = NonLinearMultiFidelityGPState(
            level_states=tuple(level_states),
            lengthscales=ls_arrs[:lvl],
            output_scales=os_arrs[:lvl],
            noise_stds=noise_arrs[:lvl],
            f_lengthscales=f_ls_arrs[: lvl - 1],
            bias_lengthscales=b_ls_arrs[: lvl - 1],
            bias_output_scales=b_os_arrs[: lvl - 1],
            base_kernel_fn=base_kernel_fn,
            f_kernel_fn=resolved_f_kernel,
            has_explicit_kernels=has_explicit,
        )
        mean_curr = predict_exact_gp(state=level_states[0], x_test=x_lvl).mean
        for prev_lvl in range(1, lvl):
            mean_curr = _predict_level_samples(partial_state, prev_lvl, x_lvl, mean_curr[None, :])[
                0
            ][0]
        augmented_x = jnp.concatenate([x_lvl, mean_curr[:, None]], axis=1)

        if kernel_fns is not None:
            st_lvl = fit_exact_gp(
                x_train=augmented_x,
                y_train=y_lvl,
                lengthscale=ls_arrs[lvl],
                output_scale=os_arrs[lvl],
                noise_std=noise_arrs[lvl],
                kernel_fn=kernel_fns[lvl],
            )
        else:
            st_lvl = _fit_composite_level(
                augmented_x=augmented_x,
                y_level=y_lvl,
                interaction_lengthscale=ls_arrs[lvl],
                interaction_output_scale=os_arrs[lvl],
                f_lengthscale=f_ls_arrs[lvl - 1],
                bias_lengthscale=b_ls_arrs[lvl - 1],
                bias_output_scale=b_os_arrs[lvl - 1],
                noise_std=noise_arrs[lvl],
                base_kernel_fn=base_kernel_fn,
                f_kernel_fn=resolved_f_kernel,
            )
        level_states.append(st_lvl)

    return NonLinearMultiFidelityGPState(
        level_states=tuple(level_states),
        lengthscales=ls_arrs,
        output_scales=os_arrs,
        noise_stds=noise_arrs,
        f_lengthscales=f_ls_arrs,
        bias_lengthscales=b_ls_arrs,
        bias_output_scales=b_os_arrs,
        base_kernel_fn=base_kernel_fn,
        f_kernel_fn=resolved_f_kernel,
        has_explicit_kernels=has_explicit,
    )


def predict_nonlinear_multi_fidelity_gp(
    *,
    state: NonLinearMultiFidelityGPState,
    x_test: jax.Array,
    target_level: int,
    num_samples: int = 64,
    rng_key: jax.Array,
) -> PredictiveDistribution:
    r"""Predict at ``x_test`` for ``target_level`` via vectorized trajectory MC.

    Propagates ``num_samples`` joint posterior trajectories across levels
    ``1 .. target_level`` in ``O(num_samples * L)`` batched Cholesky solves and
    aggregates final-level moments via the law of total variance.

    Args:
        state: Fitted :class:`NonLinearMultiFidelityGPState`.
        x_test: ``(m, d)`` test inputs.
        target_level: Integer fidelity level in ``[0, num_levels)``.
        num_samples: Number of Monte-Carlo trajectories propagated through the chain.
        rng_key: JAX PRNG key for trajectory sampling.

    Returns:
        :class:`PredictiveDistribution` carrying ``mean`` and ``variance``.

    Raises:
        ValueError: If ``target_level`` lies outside ``[0, len(state.level_states))``,
            ``num_samples < 1``, or ``x_test`` is not 2-D.
        RuntimeError: If level-0 predictive distribution lacks variance.
    """
    if (err := _target_level_error(target_level, len(state.level_states))) is not None:
        raise ValueError(err)
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1; got {num_samples}.")
    if x_test.ndim != 2:
        raise ValueError(f"x_test must be a 2-D array of shape (m, d); got shape {x_test.shape}.")

    dtype = infer_inexact_dtype(state.level_states[0].x_train, x_test)
    x_q = jnp.asarray(x_test, dtype=dtype)
    pred_0 = predict_exact_gp(state=state.level_states[0], x_test=x_q)
    if target_level == 0:
        return _wrap_predict(pred_0, target_level=0)
    if pred_0.variance is None:
        raise RuntimeError("Level-0 predictive missing variance.")

    num_points = x_q.shape[0]
    sample_means = jnp.broadcast_to(pred_0.mean[None, :], (num_samples, num_points))
    sample_vars = jnp.broadcast_to(pred_0.variance[None, :], (num_samples, num_points))
    level_keys = jax.random.split(rng_key, target_level)
    zero = jnp.asarray(0.0, dtype=dtype)

    for lvl in range(1, target_level + 1):
        noise = jax.random.normal(level_keys[lvl - 1], (num_samples, num_points), dtype=dtype)
        f_samples = sample_means + jnp.sqrt(jnp.maximum(sample_vars, zero)) * noise
        sample_means, sample_vars = _predict_level_samples(state, lvl, x_q, f_samples)

    pooled_mean = jnp.mean(sample_means, axis=0)
    pooled_variance = jnp.maximum(
        jnp.mean(sample_vars, axis=0) + jnp.var(sample_means, axis=0),
        zero,
    )
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
    "nargp_composite_kernel",
    "predict_nonlinear_multi_fidelity_gp",
]
