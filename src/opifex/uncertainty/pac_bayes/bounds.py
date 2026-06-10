"""Pure JAX PAC-Bayes generalization-bound formulas.

This module contains the differentiable, ``jax.jit`` / ``jax.grad`` /
``jax.vmap``-compatible bound kernels. No ``flax.nnx`` imports — pure JAX only.
Every function takes scalar ``jax.Array`` inputs and returns a scalar
``jax.Array``; integer ``dataset_size`` arguments are accepted as Python ``int``
or ``jax.Array``.

Canonical references (read-only):

* McAllester (1999), formulated as in **Alquier (2024)** survey
  (``arXiv:2110.11216``) and **Dziugaite & Roy (2017)** training-by-PAC-Bayes
  (``arXiv:1703.11008``):

  ``B(R, KL, n, delta) = R + sqrt((KL + log(2 * sqrt(n) / delta)) / (2 * n))``.

* Catoni (2007) variational PAC-Bayes bound with temperature ``beta > 0``
  (Alquier 2024, eq. 2.10):

  ``B_beta(R, KL, n, delta) = (beta * R + (KL + log(1 / delta)) / n) /
  (1 - exp(-beta))``.

* Quadratic / Maurer kl-inversion bound (Alquier 2024, §3.1 and Pérez-Ortiz
  et al. JMLR v22 §3.2): the tightest empirical-risk bound is the unique
  ``B in [R, 1]`` satisfying

  ``kl_bernoulli(R, B) = (KL + log(2 * sqrt(n) / delta)) / n``,

  where ``kl_bernoulli(p, q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))`` is
  the Bernoulli KL. We solve by ``_INVERT_ITERS`` steps of bisection over
  ``[R, 1 - eps]``; bisection is JIT- and grad-friendly (a fixed sequence of
  ``jnp.where`` updates), unlike Newton (which would require derivative
  evaluation at the moving boundary).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---- numerical constants ----------------------------------------------------

# Floor for ``delta`` and for the bisection upper bound. The Bernoulli KL
# diverges at the open boundary ``q in {0, 1}``; clamping to ``1 - _EPS`` keeps
# the quadratic-bound iteration finite without changing the realised root.
_EPS: float = 1e-12

# Number of bisection iterations for the quadratic (kl-inversion) bound. 50
# halvings shrink the bracket by ``2^-50 < 1e-15`` — well below float32 noise
# but still tracing as a fixed Python loop (jit-compatible).
_INVERT_ITERS: int = 50


# ---- validation -------------------------------------------------------------


def _validate_delta(delta: float) -> None:
    """Validate the confidence parameter ``delta in (0, 1)`` at Python level."""
    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta!r}.")


def _as_array(x: jax.Array | float) -> jax.Array:
    """Promote ``x`` to a JAX scalar array (float dtype)."""
    return jnp.asarray(x, dtype=jnp.float32)


# ---- bound kernels ----------------------------------------------------------


def mcallester_bound(
    empirical_risk: jax.Array,
    kl: jax.Array,
    dataset_size: jax.Array | int,
    delta: float,
) -> jax.Array:
    """Evaluate the McAllester (1999) PAC-Bayes bound.

    With confidence ``1 - delta``:

        ``B = R + sqrt((KL + log(2 * sqrt(n) / delta)) / (2 * n))``.

    Args:
        empirical_risk: Scalar empirical risk ``R`` in ``[0, 1]``.
        kl: Scalar KL divergence between posterior and prior, in nats.
        dataset_size: Sample size ``n`` (positive scalar).
        delta: Confidence parameter in ``(0, 1)``.

    Returns:
        Scalar PAC-Bayes upper bound on the population risk.

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)``.

    """
    _validate_delta(delta)
    risk = _as_array(empirical_risk)
    kl_arr = _as_array(kl)
    n = _as_array(dataset_size)
    confidence_term = kl_arr + jnp.log(2.0 * jnp.sqrt(n) / delta)
    return risk + jnp.sqrt(confidence_term / (2.0 * n))


def catoni_bound(
    empirical_risk: jax.Array,
    kl: jax.Array,
    dataset_size: jax.Array | int,
    delta: float,
    *,
    beta: float = 1.0,
) -> jax.Array:
    """Catoni (2007) PAC-Bayes bound with temperature ``beta > 0``.

    With confidence ``1 - delta``:

        ``B = (beta * R + (KL + log(1 / delta)) / n) / (1 - exp(-beta))``.

    The Catoni bound is *not* a probability — it is a real-valued upper bound
    on the population risk; the temperature ``beta`` trades empirical-risk
    sensitivity against the KL term.

    Args:
        empirical_risk: Scalar empirical risk ``R``.
        kl: Scalar KL divergence between posterior and prior, in nats.
        dataset_size: Sample size ``n`` (positive scalar).
        delta: Confidence parameter in ``(0, 1)``.
        beta: Strictly positive temperature.

    Returns:
        Scalar Catoni upper bound.

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)`` or ``beta <= 0``.

    """
    _validate_delta(delta)
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0; got {beta!r}.")
    risk = _as_array(empirical_risk)
    kl_arr = _as_array(kl)
    n = _as_array(dataset_size)
    numerator = beta * risk + (kl_arr + jnp.log(1.0 / delta)) / n
    denominator = 1.0 - jnp.exp(-beta)
    return numerator / denominator


def kl_bernoulli(p: jax.Array, q: jax.Array) -> jax.Array:
    """Bernoulli KL ``kl(p || q) = p log(p/q) + (1-p) log((1-p)/(1-q))``.

    Inputs are clamped to ``(_EPS, 1 - _EPS)`` so the log terms stay finite at
    the boundary ``p in {0, 1}``. The function is symmetric in neither
    argument and is convex in ``q`` for fixed ``p`` — the property used by the
    bisection in :func:`quadratic_bound`.
    """
    p_safe = jnp.clip(p, _EPS, 1.0 - _EPS)
    q_safe = jnp.clip(q, _EPS, 1.0 - _EPS)
    return p_safe * jnp.log(p_safe / q_safe) + (1.0 - p_safe) * jnp.log(
        (1.0 - p_safe) / (1.0 - q_safe)
    )


def quadratic_bound(
    empirical_risk: jax.Array,
    kl: jax.Array,
    dataset_size: jax.Array | int,
    delta: float,
) -> jax.Array:
    """Quadratic / Maurer kl-inversion PAC-Bayes bound.

    Returns the unique ``B in [R, 1)`` solving

        ``kl_bernoulli(R, B) = (KL + log(2 * sqrt(n) / delta)) / n``,

    obtained by ``_INVERT_ITERS`` bisection steps. Because ``kl_bernoulli``
    is strictly convex in its second argument with minimum at ``B = R``, the
    root is unique on ``[R, 1)`` whenever the right-hand side is non-negative.

    Args:
        empirical_risk: Scalar empirical risk in ``[0, 1]``.
        kl: Scalar KL divergence, in nats.
        dataset_size: Sample size ``n``.
        delta: Confidence parameter in ``(0, 1)``.

    Returns:
        Scalar quadratic-bound upper bound on the population risk.

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)``.

    """
    _validate_delta(delta)
    risk = _as_array(empirical_risk)
    kl_arr = _as_array(kl)
    n = _as_array(dataset_size)
    target = (kl_arr + jnp.log(2.0 * jnp.sqrt(n) / delta)) / n
    # Force the target to be non-negative so the root lies in [R, 1 - _EPS].
    target = jnp.maximum(target, 0.0)

    lower = jnp.clip(risk, _EPS, 1.0 - _EPS)
    upper = jnp.asarray(1.0 - _EPS, dtype=lower.dtype)

    def body(
        state: tuple[jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Perform one bisection step of the KL-inversion root-find."""
        lo, hi = state
        mid = 0.5 * (lo + hi)
        value = kl_bernoulli(risk, mid)
        is_below = value < target
        new_lo = jnp.where(is_below, mid, lo)
        new_hi = jnp.where(is_below, hi, mid)
        return (new_lo, new_hi), None

    (lower, upper), _ = jax.lax.scan(body, (lower, upper), xs=None, length=_INVERT_ITERS)
    return 0.5 * (lower + upper)


__all__ = [
    "catoni_bound",
    "kl_bernoulli",
    "mcallester_bound",
    "quadratic_bound",
]
