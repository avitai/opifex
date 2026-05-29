r"""Scalable SHO Gaussian process via state-space Kalman filtering — Task 11.1 D1.

The damped-oscillator (SHO) covariance shipped in slice 10 as
:func:`opifex.uncertainty.gp.damped_oscillator_kernel` evaluates a
direct ``O(n²)`` Gram matrix. Foreman-Mackey, Agol, Ambikasaran,
Angus 2017 (AJ, arXiv:1703.09710) observed that this kernel — like
every celerite kernel — corresponds to a 2-dimensional linear-Gaussian
SDE, so the standard forward-filter / backward-smoother recursion
delivers exact GP fit + predict in ``O(n)`` time and ``O(1)`` memory
per step.

The 2-state SHO SDE has drift

.. math::

    F = \begin{pmatrix} 0 & 1 \\ -\omega^{2} & -\omega/Q \end{pmatrix},
    \qquad
    P_{\infty} = \sigma_{f}^{2}\,\mathrm{diag}(1, \omega^{2}),
    \qquad
    H = (1, 0),

where ``ω = 1 / lengthscale``, ``Q = quality_factor`` and ``σ_f =
output_scale``. The closed-form discrete transition matrix at lag
``Δt`` is (underdamped regime ``Q > 1/2``; ``f = √(4Q² − 1)``):

.. math::

    A(\Delta t) = \mathrm{e}^{-\omega\,\Delta t/(2Q)}\,
        \begin{pmatrix}
            \cos\theta + \sin\theta/f & -2Q\,\omega\,\sin\theta/f \\
            2Q\,\sin\theta/(\omega\,f) & \cos\theta - \sin\theta/f
        \end{pmatrix},
    \quad \theta = \frac{f\,\omega\,\Delta t}{2 Q}.

Discrete process noise: ``Q_k(Δt) = P_∞ − A(Δt)\,P_∞\,A(Δt)^{T}``
(the stationary-covariance identity).

The module reuses opifex's tested Kalman primitives in
:mod:`opifex.uncertainty.statespace.kalman`:

* :func:`kalman_log_likelihood` — exact log marginal in ``O(n)``.
* :func:`kalman_filter` + :func:`kalman_smoother` — used at predict
  time on the augmented ``x_train ∪ x_test`` sequence, with test
  indices marked as "missing observations" via a large observation
  covariance (``LARGE_R = 1e8``) so the Kalman gain at those
  positions is effectively zero.

Restrictions for this slice:

* **Underdamped regime only** — ``quality_factor > 1/2``. The
  critically-damped (``Q = 1/2``) and overdamped (``Q < 1/2``)
  branches are present in the tinygp reference but are deferred to a
  follow-up slice; they require their own ``A(Δt)`` parametrisations.
* **One-dimensional time-like inputs** — ``x_train`` must be shape
  ``(n,)`` or ``(n, 1)`` with strictly increasing entries.

Reference implementations consulted (READ-ONLY)
-----------------------------------------------

* ``../tinygp/src/tinygp/kernels/quasisep.py:SHO``
  (``design_matrix``, ``stationary_covariance``,
  ``transition_matrix``).
* ``../bayesnewton/bayesnewton/kernels.py`` — analogous celerite
  state-space layer.

References
----------
* Foreman-Mackey, D., Agol, E., Ambikasaran, S., Angus, R. 2017 —
  *Fast and scalable Gaussian process modeling with applications to
  astronomical time series*, AJ, arXiv:1703.09710 (PRIMARY).
* Sarkka, S. 2013 — *Bayesian Filtering and Smoothing*, CUP
  (state-space-GP equivalence).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace.kalman import (
    kalman_filter,
    kalman_log_likelihood,
    kalman_smoother,
)
from opifex.uncertainty.types import PredictiveDistribution


_QUASISEP_SHO_SOURCE_PACKAGE = "opifex.uncertainty.gp"
_LARGE_OBSERVATION_COVARIANCE = 1.0e8
"""Sentinel ``R`` at masked test points — large enough that the
Kalman gain is numerically zero (``≈ noise_std² / 1e8``) but small
enough that the filter / smoother stay well-conditioned in float32.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class QuasisepGPState:
    """Fitted state for the scalable SHO Gaussian process.

    Attributes:
        x_train: ``(n,)`` or ``(n, 1)`` strictly-increasing training
            times.
        y_train: ``(n,)`` training observations.
        log_marginal_likelihood: Exact log marginal evaluated in
            ``O(n)`` via :func:`kalman_log_likelihood`.
        lengthscale: ``ℓ = 1/ω`` — kernel length-scale.
        output_scale: ``σ_f`` — kernel output-scale.
        noise_std: ``σ`` — observation noise scale.
        quality_factor: ``Q > 1/2`` — SHO quality factor.
    """

    x_train: jax.Array
    y_train: jax.Array
    log_marginal_likelihood: jax.Array
    lengthscale: float
    output_scale: float
    noise_std: float
    quality_factor: float


def _underdamped_sho_transition(
    dt: jax.Array, *, omega: jax.Array, quality_factor: jax.Array
) -> jax.Array:
    r"""Closed-form ``A(dt) = exp(F dt)`` for the underdamped 2-state SHO SDE.

    The tinygp ``SHO.transition_matrix`` reference returns ``A^T``
    (row-vector state convention); the opifex Kalman primitives use
    the column-vector convention ``state_next = A @ state_prev``, so
    the off-diagonal entries are transposed relative to tinygp.
    """
    f_root = jnp.sqrt(4.0 * quality_factor**2 - 1.0)
    theta = 0.5 * f_root * omega * dt / quality_factor
    decay = jnp.exp(-omega * dt / (2.0 * quality_factor))
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    a00 = cos_theta + sin_theta / f_root
    a01 = 2.0 * quality_factor * sin_theta / (omega * f_root)
    a10 = -2.0 * quality_factor * omega * sin_theta / f_root
    a11 = cos_theta - sin_theta / f_root
    return decay * jnp.array([[a00, a01], [a10, a11]])


def _build_sho_state_space(
    *,
    times: jax.Array,
    lengthscale: float,
    output_scale: float,
    quality_factor: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(initial_mean, initial_cov, transitions, process_noises)``."""
    omega = jnp.asarray(1.0 / lengthscale)
    quality = jnp.asarray(quality_factor)
    initial_mean = jnp.zeros(2)
    initial_cov = (output_scale**2) * jnp.diag(jnp.stack([jnp.asarray(1.0), omega**2]))
    dt = jnp.concatenate([jnp.zeros((1,), dtype=times.dtype), jnp.diff(times)])
    transitions = jax.vmap(
        lambda step: _underdamped_sho_transition(step, omega=omega, quality_factor=quality)
    )(dt)
    process_noises = initial_cov[None] - jnp.einsum(
        "kij,jl,kml->kim", transitions, initial_cov, transitions
    )
    return initial_mean, initial_cov, transitions, process_noises


def _flatten_times(x: jax.Array) -> jax.Array:
    """Accept ``(n,)`` or ``(n, 1)`` time inputs; return ``(n,)``."""
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    raise ValueError(
        f"Quasiseparable SHO GP expects 1-D times of shape (n,) or (n, 1); got shape {x.shape}."
    )


def fit_quasisep_sho_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
    quality_factor: float,
) -> QuasisepGPState:
    r"""Fit the scalable SHO state-space GP in ``O(n)``.

    Maps the SHO hyperparameters into a 2-state linear-Gaussian SDE
    and evaluates the exact log marginal likelihood via opifex's
    Kalman primitives.

    Args:
        x_train: ``(n,)`` or ``(n, 1)`` strictly-increasing training
            times.
        y_train: ``(n,)`` training observations.
        lengthscale: ``ℓ > 0`` — kernel length-scale.
        output_scale: ``σ_f > 0`` — kernel output-scale.
        noise_std: ``σ > 0`` — observation noise scale.
        quality_factor: ``Q > 1/2`` — SHO quality factor
            (underdamped regime only in this slice).

    Returns:
        :class:`QuasisepGPState` carrying the data, hyperparameters
        and exact log marginal likelihood.

    Raises:
        ValueError: If hyperparameters are non-positive or
            ``quality_factor ≤ 1/2``, or if ``x_train`` is not
            strictly increasing (checked at trace-time only).
    """
    if lengthscale <= 0.0:
        raise ValueError(f"lengthscale must be strictly positive; got {lengthscale!r}.")
    if output_scale <= 0.0:
        raise ValueError(f"output_scale must be strictly positive; got {output_scale!r}.")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    if quality_factor <= 0.5:
        raise ValueError(
            "fit_quasisep_sho_gp implements the underdamped regime only "
            f"(quality_factor > 1/2); got {quality_factor!r}."
        )
    times = _flatten_times(x_train)
    if not isinstance(times, jax.core.Tracer) and not bool(  # type: ignore[attr-defined]
        jnp.all(jnp.diff(times) > 0.0)
    ):
        raise ValueError(
            "x_train must be sorted in strictly increasing order for the "
            "quasiseparable SHO state-space GP."
        )
    initial_mean, initial_cov, transitions, process_noises = _build_sho_state_space(
        times=times,
        lengthscale=lengthscale,
        output_scale=output_scale,
        quality_factor=quality_factor,
    )
    observation_matrix = jnp.asarray([[1.0, 0.0]])
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
    return QuasisepGPState(
        x_train=x_train,
        y_train=y_train,
        log_marginal_likelihood=log_marginal,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        quality_factor=quality_factor,
    )


def predict_quasisep_sho_gp(
    *,
    state: QuasisepGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Posterior moments at ``x_test`` via augmented filter + smoother.

    Augments ``x_train ∪ x_test`` into a single sorted sequence, runs
    the forward Kalman filter and backward RTS smoother on it, and
    extracts the smoothed first-state moments at the test positions.
    Test entries are marked as "missing" via a large observation
    covariance so the Kalman gain at those points is numerically
    zero — equivalent to running the joint GP posterior conditioned
    on the training data only.

    Args:
        state: Fitted :class:`QuasisepGPState`.
        x_test: ``(m,)`` or ``(m, 1)`` test times (any order).

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` and
        ``variance`` carry the latent ``E[f(x*)]`` and ``Var[f(x*)]``
        for each test point. Metadata records the source paper.
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
    initial_mean, initial_cov, transitions, process_noises = _build_sho_state_space(
        times=sorted_times,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
        quality_factor=state.quality_factor,
    )
    observation_matrix = jnp.asarray([[1.0, 0.0]])
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
    test_sorted_positions = inverse_order[num_train:]
    test_means = smoothed_means[test_sorted_positions, 0]
    test_variances = smoothed_covs[test_sorted_positions, 0, 0]
    return PredictiveDistribution(
        mean=test_means,
        variance=test_variances,
        epistemic=test_variances,
        total_uncertainty=test_variances,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_QUASISEP_SHO_SOURCE_PACKAGE,
            extra=(
                ("estimator", "quasisep_sho_gp"),
                ("paper", "Foreman-Mackey+ 2017 (arXiv:1703.09710)"),
                ("kernel", "sho"),
                ("regime", "underdamped"),
            ),
        ),
    )


__all__ = [
    "QuasisepGPState",
    "fit_quasisep_sho_gp",
    "predict_quasisep_sho_gp",
]
