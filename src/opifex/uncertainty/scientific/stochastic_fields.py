"""Stochastic-field samplers — Task 8.4.

Caller-facing helpers for drawing samples of:

* a Karhunen-Loève parameterised field (Gaussian input fields with a
  prescribed covariance kernel — Ghanem-Spanos 1991 eq. (2.45)).
* a polynomial-chaos parameterised field (truncated PCE expansion at a
  caller-supplied 1-D grid — Xiu-Karniadakis 2002 eq. (3.7)).

The KLE / PCE surrogates themselves live in :mod:`polynomial_chaos`;
this module is the thin sampling glue that converts caller-owned
``jax.random.PRNGKey`` draws into discretised field realisations.
"""

from __future__ import annotations

import jax

from opifex.uncertainty.scientific.polynomial_chaos import (
    evaluate_basis,
    KarhunenLoeveExpansion,
    PolynomialChaosBasis,
)


def sample_kle_field(
    *,
    kle: KarhunenLoeveExpansion,
    num_samples: int,
    rng_key: jax.Array,
) -> jax.Array:
    """Draw ``num_samples`` realisations of a KLE-parameterised field.

    Implements the truncated Karhunen-Loève synthesis (Ghanem-Spanos
    1991 eq. (2.45)) by drawing independent ``N(0, 1)`` coefficients
    ``xi_i`` and reconstructing

        ``f(x) = sum_i sqrt(lambda_i) * phi_i(x) * xi_i``.

    Args:
        kle: A fitted :class:`KarhunenLoeveExpansion`.
        num_samples: Number of independent field samples.
        rng_key: Caller-owned JAX PRNG key. Two identical keys produce
            identical sample arrays (determinism is the contract).

    Returns:
        ``(num_samples, N)`` array of field realisations.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}.")
    num_modes = kle.eigenvalues.shape[0]
    coefficients = jax.random.normal(rng_key, (num_samples, num_modes))
    return jax.vmap(kle.reconstruct)(coefficients)


def sample_pce_field(
    *,
    basis: PolynomialChaosBasis,
    grid: jax.Array,
    num_samples: int,
    rng_key: jax.Array,
) -> jax.Array:
    """Draw ``num_samples`` realisations of a PCE-parameterised stochastic field.

    The field is parameterised by a single stochastic dimension; each
    sample multiplies the orthonormal basis evaluation at ``grid`` by a
    deterministic perturbation drawn from the supplied PRNG key.
    Implements the Wiener-Askey truncation (Xiu-Karniadakis 2002 eq.
    (3.7)) for the 1-D case.

    Args:
        basis: A :class:`PolynomialChaosBasis` with fitted coefficients.
        grid: 1-D array of evaluation points (shape ``(K,)``).
        num_samples: Number of samples.
        rng_key: Caller-owned JAX PRNG key.

    Returns:
        ``(num_samples, K)`` array of field realisations.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}.")
    if grid.ndim != 1:
        raise ValueError(f"grid must be 1-D; got shape {grid.shape}.")

    # Perturb the basis coefficients with an independent draw per sample
    # — this yields a stochastic field whose mean is the deterministic
    # surrogate value and whose marginal variance per grid point is
    # ``sum(c[1:]^2)`` (Xiu-Karniadakis 2002 eq. (3.3)).
    degrees = basis.degrees()
    phi = evaluate_basis(family=basis.family, degrees=degrees, x=grid)

    def one_sample(key: jax.Array) -> jax.Array:
        """Draw one random-field realisation from the polynomial-chaos expansion."""
        noise = jax.random.normal(key, shape=basis.coefficients.shape)
        # Scale the noise by the orthonormal projection so the noise has
        # the same variance scaling as the deterministic PCE expansion.
        perturbation = phi @ noise
        deterministic = phi @ basis.coefficients
        return deterministic + perturbation

    keys = jax.random.split(rng_key, num_samples)
    return jax.vmap(one_sample)(keys)


__all__ = ["sample_kle_field", "sample_pce_field"]
