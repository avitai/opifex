r"""Random Fourier Features (Rahimi & Recht 2007) for scalable approx GP.

For a shift-invariant kernel ``k(x, x') = k(x - x')`` Bochner's theorem
guarantees a spectral density ``ρ(ω)`` such that

.. math::

    k(x - x') = \int e^{i\,\omega^{T} (x - x')}\,\rho(\omega)\,d\omega.

Sampling ``\omega_{1}, \dots, \omega_{D/2} \sim \rho`` and stacking
the cosine / sine pairs into

.. math::

    \phi(x) = \sigma_{f}\sqrt{\tfrac{2}{D}}\,\bigl[
        \cos(\omega_{1}^{T} x), \sin(\omega_{1}^{T} x), \dots,
        \cos(\omega_{D/2}^{T} x), \sin(\omega_{D/2}^{T} x)
    \bigr]^{T}

yields the unbiased Monte-Carlo estimator
``φ(x)^T φ(x') ≈ k(x, x')`` (Rahimi & Recht 2007 Algorithm 1). For the
RBF kernel ``ρ`` is Gaussian with covariance ``ℓ^{-2} I``; the
implementation here draws ``ω_q \sim N(0, ℓ^{-2} I_d)`` via caller-owned
``nnx.Rngs`` so the same key tree governs every randomised primitive in
the opifex training stack.

Once the feature map is constructed, an approximate GP regression
collapses to **ridge regression on the lifted features**
``Φ(X) ∈ R^{n × D}``: the predictive mean and variance are

.. math::

    \mu(x^{*}) &= \phi(x^{*})^{T}\,(\Phi^{T}\Phi + \sigma^{2} I_{D})^{-1}\,
                 \Phi^{T}\,y, \\
    \mathrm{Var}(x^{*}) &= \sigma^{2}\,
                 \phi(x^{*})^{T}\,(\Phi^{T}\Phi + \sigma^{2} I_{D})^{-1}\,
                 \phi(x^{*}).

Compared with exact GP the cost reduces from ``O(n³)`` to
``O(n D² + D³)``; for ``D ≪ n`` this is the canonical large-N
work-around (Rahimi & Recht 2007, GPJax ``rff.py``).

References
----------
* Rahimi, A., Recht, B. 2007 — *Random Features for Large-Scale Kernel
  Machines*, NeurIPS, arXiv:0708.0234 (PRIMARY).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx  # noqa: TC002 — kept eager for consistency

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_RFF_STREAMS = ("params", "sample", "default")
_RFF_SOURCE_PACKAGE = "opifex.uncertainty.gp"


def rbf_random_fourier_features(
    *,
    x: jax.Array,
    lengthscale: float,
    output_scale: float,
    num_features: int,
    rngs: nnx.Rngs | jax.Array,
) -> jax.Array:
    r"""Construct the RFF feature map for the RBF kernel.

    Args:
        x: ``(n, d)`` inputs.
        lengthscale: RBF length-scale ``ℓ``.
        output_scale: RBF output-scale ``σ_f``.
        num_features: ``D`` — total feature count. Must be a strictly
            positive even integer (cos and sin contribute a pair per
            ``ω``).
        rngs: Caller-owned ``nnx.Rngs`` or raw key.

    Returns:
        Feature matrix ``(n, num_features)`` such that
        ``φ(x)^T φ(x') ≈ k_RBF(x, x')`` in expectation.

    Raises:
        ValueError: If ``num_features`` is non-positive or odd.
    """
    if num_features <= 0:
        raise ValueError(f"num_features must be strictly positive; got {num_features!r}.")
    if num_features % 2 != 0:
        raise ValueError(f"num_features must be even (cos/sin pairing); got {num_features!r}.")
    if lengthscale <= 0.0:
        raise ValueError(f"lengthscale must be strictly positive; got {lengthscale!r}.")
    if output_scale <= 0.0:
        raise ValueError(f"output_scale must be strictly positive; got {output_scale!r}.")
    key = extract_rng_key(rngs, streams=_RFF_STREAMS, context="rbf_random_fourier_features")
    num_pairs = num_features // 2
    omega = jax.random.normal(key, (num_pairs, x.shape[-1])) / lengthscale
    projection = x @ omega.T  # (n, num_pairs)
    scale = output_scale * jnp.sqrt(2.0 / num_features)
    return scale * jnp.concatenate(
        [jnp.cos(projection), jnp.sin(projection)],
        axis=-1,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RFFGPState:
    """Fitted state for an RFF-approximate conjugate GP.

    Attributes:
        features_train: Feature matrix ``Φ(X) ∈ R^{n × D}``.
        omega: Spectral samples ``(D/2, d)`` used to build the feature
            map; needed at predict time so the same map is applied to
            test inputs.
        feature_scale: Scalar prefactor ``σ_f sqrt(2 / D)`` applied to
            the cos/sin features.
        gram: ``(D, D)`` ridge-Gram ``Φ^T Φ + σ² I``.
        gram_cholesky: Lower-triangular Cholesky factor of ``gram``.
        alpha: Pre-solved ``(Φ^T Φ + σ² I)^{-1} Φ^T y`` of shape
            ``(D,)``.
        lengthscale: RBF length-scale used at fit time.
        output_scale: RBF output-scale used at fit time.
        noise_std: Observation noise scale ``σ``.
    """

    features_train: jax.Array
    omega: jax.Array
    feature_scale: jax.Array
    gram: jax.Array
    gram_cholesky: jax.Array
    alpha: jax.Array
    lengthscale: float
    output_scale: float
    noise_std: float


def fit_rff_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
    num_features: int,
    rngs: nnx.Rngs | jax.Array,
) -> RFFGPState:
    r"""Fit the RFF-approximate conjugate GP (Rahimi & Recht 2007).

    Solves the ridge-regression normal equations
    ``(Φ^T Φ + σ² I) α = Φ^T y`` via Cholesky.

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` targets.
        lengthscale: RBF length-scale.
        output_scale: RBF output-scale.
        noise_std: Observation noise scale ``σ`` (also acts as the
            ridge jitter).
        num_features: Feature count ``D`` (even, strictly positive).
        rngs: Caller-owned RNG for the spectral sample.

    Returns:
        :class:`RFFGPState` carrying the feature map data + Cholesky
        factor + pre-solved ``α``.

    Raises:
        ValueError: If ``noise_std`` is non-positive.
    """
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    key = extract_rng_key(rngs, streams=_RFF_STREAMS, context="fit_rff_gp")
    num_pairs = num_features // 2
    if num_features <= 0 or num_features % 2 != 0:
        raise ValueError(f"num_features must be a positive even integer; got {num_features!r}.")
    omega = jax.random.normal(key, (num_pairs, x_train.shape[-1])) / lengthscale
    projection = x_train @ omega.T
    feature_scale = output_scale * jnp.sqrt(2.0 / num_features)
    features_train = feature_scale * jnp.concatenate(
        [jnp.cos(projection), jnp.sin(projection)], axis=-1
    )
    gram = features_train.T @ features_train + (noise_std**2) * jnp.eye(num_features)
    gram_cholesky = jnp.linalg.cholesky(gram)
    alpha = jax.scipy.linalg.cho_solve((gram_cholesky, True), features_train.T @ y_train)
    return RFFGPState(
        features_train=features_train,
        omega=omega,
        feature_scale=jnp.asarray(feature_scale),
        gram=gram,
        gram_cholesky=gram_cholesky,
        alpha=alpha,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )


def predict_rff_gp(
    *,
    state: RFFGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Predict at ``x_test`` using a fitted :class:`RFFGPState`.

    The lifted predictive moments

    .. math::

        \mu(x^{*})  &= \phi(x^{*})^{T}\,\alpha, \\
        \mathrm{Var}(x^{*}) &= \sigma^{2}\,
            \phi(x^{*})^{T}\,(\Phi^{T}\Phi + \sigma^{2} I)^{-1}\,\phi(x^{*}),

    are evaluated via the Cholesky factor of the ridge-Gram.

    Args:
        state: Fitted :class:`RFFGPState`.
        x_test: ``(m, d)`` test inputs.

    Returns:
        :class:`PredictiveDistribution` with ``mean`` ``(m,)``,
        ``variance`` ``(m,)``, ``epistemic == variance``, and metadata
        advertising ``estimator=rff_gp`` plus the Rahimi & Recht
        citation.
    """
    projection = x_test @ state.omega.T
    features_test = state.feature_scale * jnp.concatenate(
        [jnp.cos(projection), jnp.sin(projection)], axis=-1
    )
    mean = features_test @ state.alpha
    v = jax.scipy.linalg.solve_triangular(state.gram_cholesky, features_test.T, lower=True)
    variance = (state.noise_std**2) * jnp.sum(v * v, axis=0)
    return PredictiveDistribution(
        mean=mean,
        variance=variance,
        epistemic=variance,
        total_uncertainty=variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_RFF_SOURCE_PACKAGE,
            extra=(
                ("estimator", "rff_gp"),
                ("paper", "Rahimi & Recht 2007 arXiv:0708.0234"),
            ),
        ),
    )


__all__ = [
    "RFFGPState",
    "fit_rff_gp",
    "predict_rff_gp",
    "rbf_random_fourier_features",
]
