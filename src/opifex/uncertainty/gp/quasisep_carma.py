r"""Scalable CARMA Gaussian process via state-space Kalman filtering.

The continuous-time autoregressive moving-average covariance shipped as
:func:`opifex.uncertainty.gp.carma_kernel` evaluates a dense ``O(n²)``
Gram matrix. Every CARMA(p, q) process is a linear-Gaussian SDE, so the
standard forward-filter / backward-smoother recursion delivers exact GP
fit + predict in ``O(n)`` time and ``O(1)`` memory per step.

State-space realization
-----------------------

A CARMA process is realized as the celerite quasiseparable state space
of Foreman-Mackey, Agol, Ambikasaran, Angus 2017 (arXiv:1703.09710),
parameterized by the autoregressive roots ``ρ_j`` and the ACVF
coefficients ``acf_j`` of Kelly+ 2014 (arXiv:1402.5978, Eq. 4). Roots
come in real and complex-conjugate pairs; a *real-valued* block drift
keeps every quantity real and jittable:

* each **real** root ``ρ_j = -c`` contributes a scalar block with drift
  ``-c``, observation ``√|acf_j|`` and unit stationary variance;
* each **complex-conjugate pair** ``ρ = -c ± i d`` contributes a
  :math:`2\times2` rotation block

  .. math::

      F_{\text{pair}} = \begin{pmatrix} -c & d \\ -d & -c \end{pmatrix},

  whose closed-form transition is the damped rotation
  ``A(\Delta t) = e^{-c\,\Delta t}\,R(d\,\Delta t)``. The observation
  weights ``(h_1, h_2)`` and the stationary block covariance are the
  celerite factorization of the pair's ACVF term.

With drift ``F``, stationary covariance ``P_∞``, transition
``A(\Delta t)`` and observation vector ``h``, the CARMA kernel is

.. math::

    k(x_1, x_2) = h^{\top} P_{\infty}\, A(|x_1 - x_2|)\, h,

so the state-space GP reproduces :func:`carma_kernel` exactly. The
``lengthscale`` rescales the lag (``Δt → Δt / ℓ``) and ``output_scale²``
scales the stationary covariance, matching the direct-kernel
conventions.

The module reuses opifex's tested CARMA ACVF helpers
(:func:`opifex.uncertainty.gp.kernels._carma_roots`,
:func:`opifex.uncertainty.gp.kernels._carma_acvf`) and Kalman primitives
(:mod:`opifex.uncertainty.statespace.kalman`):

* :func:`kalman_log_likelihood` — exact log marginal in ``O(n)``.
* :func:`kalman_filter` + :func:`kalman_smoother` — used at predict
  time on the augmented ``x_train ∪ x_test`` sequence, with test
  indices marked as "missing" via a large observation covariance so the
  Kalman gain at those positions is numerically zero.

Restrictions
------------

* **Celerite-representable processes** — each ACVF term must admit a
  positive-semidefinite per-block celerite factorization (the standard
  CARMA realization; NaN propagates otherwise, exactly as in the
  reference). CARMA(1, 0)/(2, 0)/(2, 1) and most higher orders are
  representable.
* **One-dimensional time-like inputs** — ``x_train`` must be shape
  ``(n,)`` or ``(n, 1)`` with strictly increasing entries.

Reference implementations consulted (READ-ONLY)
-----------------------------------------------

* ``../tinygp/src/tinygp/kernels/quasisep.py:CARMA`` (``design_matrix``,
  ``stationary_covariance``, ``observation_model``,
  ``transition_matrix``).
* ``../bayesnewton/bayesnewton/kernels.py`` — analogous state-space
  layer for celerite/CARMA covariances.

References
----------
* Kelly, B. C., et al. 2014 — *Flexible and scalable methods for
  quantifying stochastic variability in the era of massive time-domain
  astronomical data sets*, ApJ, arXiv:1402.5978 (PRIMARY; CARMA
  definition + ACVF).
* Foreman-Mackey, D., Agol, E., Ambikasaran, S., Angus, R. 2017 —
  *Fast and scalable Gaussian process modeling*, AJ, arXiv:1703.09710
  (celerite state-space realization).
* Sarkka, S. 2013 — *Bayesian Filtering and Smoothing*, CUP
  (state-space-GP equivalence).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.kernels import _carma_acvf, _carma_roots
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace.kalman import (
    kalman_filter,
    kalman_log_likelihood,
    kalman_smoother,
)
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_QUASISEP_CARMA_SOURCE_PACKAGE = "opifex.uncertainty.gp"
_LARGE_OBSERVATION_COVARIANCE = 1.0e8
"""Sentinel ``R`` at masked test points — large enough that the Kalman
gain is numerically zero but small enough that the filter / smoother
stay well-conditioned.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class QuasisepCarmaGPState:
    """Fitted state for the scalable CARMA Gaussian process.

    Attributes:
        x_train: ``(n,)`` or ``(n, 1)`` strictly-increasing training
            times.
        y_train: ``(n,)`` training observations.
        log_marginal_likelihood: Exact log marginal evaluated in
            ``O(n)`` via :func:`kalman_log_likelihood`.
        ar_coefficients: ``(p,)`` autoregressive coefficients
            ``α_0, …, α_{p-1}`` (implicit ``α_p = 1``).
        ma_coefficients: ``(q + 1,)`` moving-average coefficients
            ``β_0, …, β_q`` with ``q + 1 ≤ p``.
        lengthscale: ``ℓ`` — rescales the lag (``τ → τ / ℓ``).
        output_scale: ``σ_f`` — kernel output-scale (``k → σ_f² k``).
        noise_std: ``σ`` — observation noise scale.
    """

    x_train: jax.Array
    y_train: jax.Array
    log_marginal_likelihood: jax.Array
    ar_coefficients: jax.Array
    ma_coefficients: jax.Array
    lengthscale: float
    output_scale: float | jax.Array
    noise_std: float


@dataclass(frozen=True, slots=True, kw_only=True)
class _CarmaRealization:
    """Real, block-structured state-space realization of a CARMA process.

    Attributes:
        ar_roots: ``(p,)`` complex autoregressive roots.
        observation_vector: ``(p,)`` real observation weights ``h``.
        stationary_cov: ``(p, p)`` real stationary covariance ``P_∞``.
        real_mask: ``(p,)`` boolean — true for real roots.
        complex_mask: ``(p,)`` boolean — true for complex roots.
        complex_select: ``(p,)`` selector marking the first entry of
            each complex-conjugate pair (used for the upper off-diagonal
            rotation entries).
    """

    ar_roots: jax.Array
    observation_vector: jax.Array
    stationary_cov: jax.Array
    real_mask: jax.Array
    complex_mask: jax.Array
    complex_select: jax.Array


def _build_carma_realization(
    *, ar_coefficients: jax.Array, ma_coefficients: jax.Array
) -> _CarmaRealization:
    """Build the celerite real state-space realization from AR/MA coefficients.

    Ports ``../tinygp/src/tinygp/kernels/quasisep.py:CARMA``
    (``__init__`` observation model + ``stationary_covariance``),
    reusing opifex's :func:`_carma_roots` / :func:`_carma_acvf`.
    """
    ar_roots = _carma_roots(jnp.append(ar_coefficients, 1.0))
    acf = _carma_acvf(ar_roots=ar_roots, alpha=ar_coefficients, beta=ma_coefficients)
    order = acf.shape[0]

    root_eps = 10.0 * jnp.finfo(ar_roots.imag.dtype).eps
    real_mask = jnp.abs(ar_roots.imag) < root_eps
    complex_mask = ~real_mask
    complex_index = jnp.cumsum(complex_mask) * complex_mask
    complex_select = complex_mask * complex_index % 2

    observation_vector = _carma_observation_vector(acf=acf, ar_roots=ar_roots, real_mask=real_mask)
    stationary_cov = _carma_stationary_covariance(
        acf=acf,
        ar_roots=ar_roots,
        real_mask=real_mask,
        complex_mask=complex_mask,
        complex_select=complex_select,
        order=order,
    )
    return _CarmaRealization(
        ar_roots=ar_roots,
        observation_vector=observation_vector,
        stationary_cov=stationary_cov,
        real_mask=real_mask,
        complex_mask=complex_mask,
        complex_select=complex_select,
    )


def _carma_observation_vector(
    *, acf: jax.Array, ar_roots: jax.Array, real_mask: jax.Array
) -> jax.Array:
    """Celerite observation weights ``h`` for the CARMA realization.

    Real roots use ``√|Re acf|``; each complex-conjugate pair uses the
    celerite ``(h_1, h_2)`` factorization of its ACVF term.
    """
    obs_real = jnp.sqrt(jnp.abs(acf.real))
    amp_real, amp_imag = 2.0 * acf.real, 2.0 * acf.imag
    decay, freq = -ar_roots.real, -ar_roots.imag
    decay_sq, freq_sq = jnp.square(decay), jnp.square(freq)
    radius_sq = decay_sq + freq_sq

    denom = jnp.where(real_mask, 1.0, 2.0 * decay * radius_sq)
    h2_squared = freq_sq * (amp_real * decay - amp_imag * freq) / denom
    h2 = jnp.sqrt(h2_squared)
    denom = jnp.where(real_mask, 1.0, freq)
    h1 = (decay * h2 - jnp.sqrt(amp_real * freq_sq - radius_sq * h2_squared)) / denom
    obs_complex = jnp.array([h1, h2])
    return jnp.where(real_mask, obs_real, jnp.ravel(obs_complex)[::2]).real


def _carma_stationary_covariance(
    *,
    acf: jax.Array,
    ar_roots: jax.Array,
    real_mask: jax.Array,
    complex_mask: jax.Array,
    complex_select: jax.Array,
    order: int,
) -> jax.Array:
    r"""Stationary covariance ``P_∞`` for the CARMA realization.

    Ports ``tinygp.kernels.quasisep.CARMA.stationary_covariance``: a
    signed-identity diagonal for real roots plus the celerite
    :math:`2\times2` block per complex-conjugate pair.
    """
    sign_diag = jnp.diag(jnp.where(acf.real > 0, jnp.ones(order), -jnp.ones(order)))
    freq = ar_roots.imag
    denom = jnp.where(real_mask, 1.0, freq)
    decay_over_freq = ar_roots.real / denom
    pair_diag = jnp.diag(
        2.0 * jnp.square(decay_over_freq * jnp.roll(complex_select, 1) * complex_mask)
    )
    upper = jnp.diag((-decay_over_freq * complex_select)[:-1], k=1)
    return (sign_diag + pair_diag + upper + upper.T).real


def _carma_transitions(*, realization: _CarmaRealization, scaled_steps: jax.Array) -> jax.Array:
    r"""Per-step transition matrices ``A(Δt) = exp(F Δt)`` for each lag.

    Ports ``tinygp.kernels.quasisep.CARMA.transition_matrix``: a decay
    on each real root and a damped rotation per complex-conjugate pair.
    The reference returns the transition in the row-vector convention;
    the opifex Kalman primitives use the column-vector convention
    ``state_next = A @ state_prev``, so the rotation block off-diagonals
    are transposed here (matching the joint covariance
    ``Cov(s_i, s_j) = A(|t_i - t_j|)\,P_\infty``).
    """
    decay_rate = -realization.ar_roots.real
    freq = -realization.ar_roots.imag
    real_mask = realization.real_mask
    complex_mask = realization.complex_mask
    complex_select = realization.complex_select

    def _single(step: jax.Array) -> jax.Array:
        decay = jnp.exp(-decay_rate * step)
        angle = freq * step
        diag_real = jnp.diag(decay * real_mask)
        diag_complex = jnp.diag(decay * jnp.cos(angle) * complex_mask)
        upper = jnp.diag((decay * jnp.sin(angle) * complex_select)[:-1], k=1)
        return (diag_real + diag_complex + upper.T - upper).real

    return jax.vmap(_single)(scaled_steps)


def _build_carma_state_space(
    *,
    times: jax.Array,
    ar_coefficients: jax.Array,
    ma_coefficients: jax.Array,
    lengthscale: float,
    output_scale: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(initial_mean, initial_cov, transitions, process_noises, H)``."""
    realization = _build_carma_realization(
        ar_coefficients=ar_coefficients, ma_coefficients=ma_coefficients
    )
    state_dim = realization.observation_vector.shape[0]
    initial_mean = jnp.zeros(state_dim)
    initial_cov = (output_scale**2) * realization.stationary_cov
    scaled_steps = jnp.concatenate(
        [jnp.zeros((1,), dtype=times.dtype), jnp.diff(times) / lengthscale]
    )
    transitions = _carma_transitions(realization=realization, scaled_steps=scaled_steps)
    process_noises = initial_cov[None] - jnp.einsum(
        "kij,jl,kml->kim", transitions, initial_cov, transitions
    )
    observation_matrix = realization.observation_vector.reshape(1, -1)
    return initial_mean, initial_cov, transitions, process_noises, observation_matrix


def _flatten_times(x: jax.Array) -> jax.Array:
    """Accept ``(n,)`` or ``(n, 1)`` time inputs; return ``(n,)``."""
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    raise ValueError(
        f"Quasiseparable CARMA GP expects 1-D times of shape (n,) or (n, 1); got shape {x.shape}."
    )


def _validate_fit_inputs(
    *,
    ar_coefficients: jax.Array,
    ma_coefficients: jax.Array,
    lengthscale: float,
    output_scale: float | jax.Array,
    noise_std: float,
    times: jax.Array,
) -> None:
    """Validate CARMA orders, hyperparameters and time ordering."""
    if ma_coefficients.shape[0] > ar_coefficients.shape[0]:
        raise ValueError(
            "CARMA(p, q) requires q + 1 ≤ p (Kelly+ 2014 Eq. 1); got "
            f"len(ma_coefficients) = {ma_coefficients.shape[0]} > "
            f"len(ar_coefficients) = {ar_coefficients.shape[0]}."
        )
    if lengthscale <= 0.0:
        raise ValueError(f"lengthscale must be strictly positive; got {lengthscale!r}.")
    if not isinstance(output_scale, jax.core.Tracer) and output_scale <= 0.0:  # type: ignore[attr-defined]
        raise ValueError(f"output_scale must be strictly positive; got {output_scale!r}.")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    if not isinstance(times, jax.core.Tracer) and not bool(  # type: ignore[attr-defined]
        jnp.all(jnp.diff(times) > 0.0)
    ):
        raise ValueError(
            "x_train must be sorted in strictly increasing order for the "
            "quasiseparable CARMA state-space GP."
        )


def fit_quasisep_carma_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    ar_coefficients: jax.Array,
    ma_coefficients: jax.Array,
    lengthscale: float,
    output_scale: float | jax.Array,
    noise_std: float,
) -> QuasisepCarmaGPState:
    r"""Fit the scalable CARMA state-space GP in ``O(n)``.

    Maps the CARMA(p, q) coefficients into the celerite real
    linear-Gaussian SDE and evaluates the exact log marginal likelihood
    via opifex's Kalman primitives.

    Args:
        x_train: ``(n,)`` or ``(n, 1)`` strictly-increasing training
            times.
        y_train: ``(n,)`` training observations.
        ar_coefficients: ``(p,)`` autoregressive coefficients
            ``α_0, …, α_{p-1}`` (implicit ``α_p = 1``). For stationarity
            all AR roots must have negative real parts (NaN propagates
            otherwise).
        ma_coefficients: ``(q + 1,)`` moving-average coefficients
            ``β_0, …, β_q`` with ``q + 1 ≤ p``.
        lengthscale: ``ℓ > 0`` — rescales the lag (``τ → τ / ℓ``).
        output_scale: ``σ_f > 0`` — kernel output-scale.
        noise_std: ``σ > 0`` — observation noise scale.

    Returns:
        :class:`QuasisepCarmaGPState` carrying the data, coefficients,
        hyperparameters and exact log marginal likelihood.

    Raises:
        ValueError: If ``q + 1 > p``, if hyperparameters are
            non-positive, or if ``x_train`` is not strictly increasing
            (checked at trace-time only).
    """
    times = _flatten_times(x_train)
    _validate_fit_inputs(
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        times=times,
    )
    initial_mean, initial_cov, transitions, process_noises, observation_matrix = (
        _build_carma_state_space(
            times=times,
            ar_coefficients=ar_coefficients,
            ma_coefficients=ma_coefficients,
            lengthscale=lengthscale,
            output_scale=output_scale,
        )
    )
    observation_covs = jnp.broadcast_to(jnp.asarray([[noise_std**2]]), (times.shape[0], 1, 1))
    observations = y_train.reshape(-1, 1)
    log_marginal = kalman_log_likelihood(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    return QuasisepCarmaGPState(
        x_train=x_train,
        y_train=y_train,
        log_marginal_likelihood=log_marginal,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )


def predict_quasisep_carma_gp(
    *,
    state: QuasisepCarmaGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Posterior moments at ``x_test`` via augmented filter + smoother.

    Augments ``x_train ∪ x_test`` into a single sorted sequence, runs
    the forward Kalman filter and backward RTS smoother on it, and
    extracts the smoothed observed-channel moments at the test
    positions. Test entries are marked "missing" via a large observation
    covariance so the Kalman gain at those points is numerically zero —
    equivalent to the joint GP posterior conditioned on the training
    data only.

    Args:
        state: Fitted :class:`QuasisepCarmaGPState`.
        x_test: ``(m,)`` or ``(m, 1)`` test times (any order).

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` and ``variance``
        carry ``E[f(x*)]`` and ``Var[f(x*)]`` for each test point.
    """
    train_times = _flatten_times(state.x_train)
    test_times = _flatten_times(x_test)
    num_train = train_times.shape[0]
    num_test = test_times.shape[0]
    all_times = jnp.concatenate([train_times, test_times])
    is_observed = jnp.concatenate(
        [jnp.ones((num_train,), dtype=jnp.bool_), jnp.zeros((num_test,), dtype=jnp.bool_)]
    )
    order = jnp.argsort(all_times)
    sorted_times = all_times[order]
    sorted_observed = is_observed[order]
    inverse_order = jnp.argsort(order)
    sorted_observation_scalar = jnp.zeros((num_train + num_test,), dtype=state.y_train.dtype)
    sorted_observation_scalar = sorted_observation_scalar.at[inverse_order[:num_train]].set(
        state.y_train
    )
    sorted_observations = sorted_observation_scalar.reshape(-1, 1)
    noise_variance = jnp.asarray(state.noise_std**2, dtype=state.y_train.dtype)
    large_variance = jnp.asarray(_LARGE_OBSERVATION_COVARIANCE, dtype=state.y_train.dtype)
    sorted_obs_cov_scalar = jnp.where(sorted_observed, noise_variance, large_variance)
    sorted_observation_covs = sorted_obs_cov_scalar.reshape(-1, 1, 1)
    initial_mean, initial_cov, transitions, process_noises, observation_matrix = (
        _build_carma_state_space(
            times=sorted_times,
            ar_coefficients=state.ar_coefficients,
            ma_coefficients=state.ma_coefficients,
            lengthscale=state.lengthscale,
            output_scale=state.output_scale,
        )
    )
    filter_means, filter_covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=sorted_observations,
        observation_matrix=observation_matrix,
        observation_covs=sorted_observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    smoothed_means, smoothed_covs = kalman_smoother(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=transitions,
        process_noises=process_noises,
    )
    observation_vector = observation_matrix[0]
    test_sorted_positions = inverse_order[num_train:]
    test_means = smoothed_means[test_sorted_positions] @ observation_vector
    test_variances = jnp.einsum(
        "i,kij,j->k",
        observation_vector,
        smoothed_covs[test_sorted_positions],
        observation_vector,
    )
    return gaussian_process_predictive(
        test_means,
        test_variances,
        epistemic=test_variances,
        total_uncertainty=test_variances,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_QUASISEP_CARMA_SOURCE_PACKAGE,
            extra=(
                ("estimator", "quasisep_carma_gp"),
                ("paper", "Kelly+ 2014 (arXiv:1402.5978)"),
                ("kernel", "carma"),
            ),
        ),
    )


__all__ = [
    "QuasisepCarmaGPState",
    "fit_quasisep_carma_gp",
    "predict_quasisep_carma_gp",
]
