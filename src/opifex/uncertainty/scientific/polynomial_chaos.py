"""Polynomial-chaos expansion (PCE) primitives — Task 6.6.

This is the **canonical home** for PCE in opifex. The intentionally
small scope for this slice covers:

* Orthogonal one-dimensional basis evaluation for Legendre (uniform
  inputs) and probabilists' Hermite (Gaussian inputs).
* Two-dimensional tensor-product basis evaluation.
* Mean / variance extraction from explicit PCE coefficients on an
  orthonormal basis.

Phase 8 Task 8.4 extends *this same file* with Karhunen-Loève
expansion, stochastic-Galerkin / stochastic-collocation helpers,
adaptive sparse-grid quadrature, anisotropic multi-element PCE, and
coefficient regression — no intermediate ``surrogate/pce.py`` is
introduced (Task 6.6 plan explicitly forbids it).

References:
* Xiu, D. & Karniadakis, G. E. (2002), "The Wiener-Askey Polynomial
  Chaos for Stochastic Differential Equations", SIAM J. Sci. Comput.
  24(2), 619-644 — the orthonormal basis recipe and the closed-form
  mean / variance extraction used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special


_SUPPORTED_FAMILIES = frozenset({"legendre", "hermite"})


@dataclass(frozen=True, slots=True, kw_only=True)
class PCESummary:
    """Summary statistics extracted from PCE coefficients.

    Assumes the supplied coefficients reference an orthonormal basis so
    the variance is the squared L2-norm of the non-constant
    coefficients (Xiu-Karniadakis 2002 equation (3.3)).

    Attributes:
        mean: PCE mean = ``coefficients[0]``.
        variance: PCE variance = ``sum(coefficients[1:]**2)``.
        coefficients: The coefficient vector, copied through for
            traceability.
        family: ``"legendre"`` or ``"hermite"`` — the input
            distribution family the basis is orthonormal against.
    """

    mean: jax.Array
    variance: jax.Array
    coefficients: jax.Array
    family: str


def _legendre_basis(degree: int, x: jax.Array) -> jax.Array:
    """Orthonormal Legendre polynomial of given degree at ``x in [-1, 1]``."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0; got {degree}.")

    def cond(carry: tuple[int, jax.Array, jax.Array]) -> jax.Array:
        n, _, _ = carry
        return n < degree

    def body(carry: tuple[int, jax.Array, jax.Array]) -> tuple[int, jax.Array, jax.Array]:
        n, p_prev, p_curr = carry
        n_next = n + 1
        p_next = ((2 * n_next - 1) * x * p_curr - (n_next - 1) * p_prev) / n_next
        return n_next, p_curr, p_next

    if degree == 0:
        raw = jnp.ones_like(x)
    elif degree == 1:
        raw = x
    else:
        _, _, raw = jax.lax.while_loop(cond, body, (1, jnp.ones_like(x), x))
    norm = jnp.sqrt((2.0 * degree + 1.0) / 2.0)
    return norm * raw


def _hermite_basis(degree: int, x: jax.Array) -> jax.Array:
    """Orthonormal probabilists' Hermite polynomial at ``x ~ N(0, 1)``."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0; got {degree}.")

    def cond(carry: tuple[int, jax.Array, jax.Array]) -> jax.Array:
        n, _, _ = carry
        return n < degree

    def body(carry: tuple[int, jax.Array, jax.Array]) -> tuple[int, jax.Array, jax.Array]:
        n, h_prev, h_curr = carry
        n_next = n + 1
        h_next = x * h_curr - n * h_prev
        return n_next, h_curr, h_next

    if degree == 0:
        raw = jnp.ones_like(x)
    elif degree == 1:
        raw = x
    else:
        _, _, raw = jax.lax.while_loop(cond, body, (1, jnp.ones_like(x), x))
    # Orthonormalise w.r.t. the standard normal weight: ||He_n||^2 = n!.
    log_factorial = jsp_special.gammaln(jnp.asarray(degree + 1, dtype=jnp.float32))
    norm = jnp.exp(-0.5 * log_factorial)
    return norm * raw


def evaluate_basis(
    *,
    family: str,
    degrees: jax.Array,
    x: jax.Array,
) -> jax.Array:
    """Compute the requested orthonormal basis on ``x``.

    Args:
        family: ``"legendre"`` (uniform on ``[-1, 1]``) or
            ``"hermite"`` (probabilists' Hermite for ``x ~ N(0, 1)``).
        degrees: 1-D integer array of degrees ``(P,)`` to evaluate.
        x: Input array of shape ``(N, d)`` for a ``d``-dimensional
            tensor-product basis, or ``(N,)`` for the 1-D basis.

    Returns:
        ``(N, P)`` array of basis values for a 1-D ``x`` and
        ``(N, P, d)`` for a 2-D ``x``.

    Raises:
        ValueError: On unsupported family or empty degrees.
    """
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported PCE family: {family!r}. Choose from {sorted(_SUPPORTED_FAMILIES)}."
        )
    if degrees.shape[0] == 0:
        raise ValueError("degrees must contain at least one non-empty entry.")

    basis_one = _legendre_basis if family == "legendre" else _hermite_basis

    if x.ndim == 1:
        return jnp.stack([basis_one(int(d), x) for d in degrees], axis=1)
    if x.ndim == 2:
        return jnp.stack(
            [
                jnp.stack([basis_one(int(d), x[:, j]) for d in degrees], axis=1)
                for j in range(x.shape[1])
            ],
            axis=2,
        )
    raise ValueError(f"x must be 1-D or 2-D; got shape {x.shape}.")


def pce_summary(*, coefficients: jax.Array, family: str) -> PCESummary:
    """Mean / variance extraction from an orthonormal-PCE coefficient vector.

    Assumes the leading coefficient is the constant-mode coefficient
    and the remaining entries reference an orthonormal basis. Then
    ``mean = c[0]`` and ``variance = sum(c[1:]**2)``.

    Raises:
        ValueError: On unsupported family or empty coefficients.
    """
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported PCE family: {family!r}. Choose from {sorted(_SUPPORTED_FAMILIES)}."
        )
    if coefficients.shape[0] == 0:
        raise ValueError("coefficients must contain at least one entry.")

    mean = coefficients[0]
    variance = jnp.sum(coefficients[1:] ** 2)
    return PCESummary(mean=mean, variance=variance, coefficients=coefficients, family=family)


def fit_pce_coefficients(*, x: jax.Array, y: jax.Array, family: str) -> jax.Array:
    """Reserved for Phase 8 Task 8.4 — raises ``NotImplementedError`` for now.

    The plan defers PCE-coefficient regression with cross-validation
    to Task 8.4; this stub guards the public surface so callers get an
    actionable error instead of a silent placeholder.
    """
    del x, y, family
    raise NotImplementedError("PCE coefficient regression added in Phase 8 Task 8.4.")


__all__ = [
    "PCESummary",
    "evaluate_basis",
    "fit_pce_coefficients",
    "pce_summary",
]
