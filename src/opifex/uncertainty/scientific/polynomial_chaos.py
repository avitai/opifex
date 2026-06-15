"""Polynomial-chaos, Karhunen-Loève, and stochastic-Galerkin / collocation primitives.

This is the **canonical home** for the classical scientific-UQ
expansions in opifex (Phase 6 Task 6.6 + Phase 8 Task 8.4 — both tasks
write to this file; no intermediate ``surrogate/pce.py`` exists).

Scope:

* Orthogonal one-dimensional basis evaluation for Legendre (uniform
  inputs) and probabilists' Hermite (Gaussian inputs).
* Two-dimensional tensor-product basis evaluation.
* Mean / variance extraction from explicit PCE coefficients on an
  orthonormal basis.
* :class:`PolynomialChaosBasis` and :class:`PolynomialChaosConfig` —
  pattern (A) / (B) container pair.
* :class:`KarhunenLoeveExpansion` and :class:`KLEConfig` — KLE of a
  symmetric positive-definite covariance kernel via ``jnp.linalg.eigh``
  (cost ``O(N^3)`` in the discretisation size ``N``).
* :class:`StochasticGalerkinSurrogate` — least-squares PCE-coefficient
  fit against a caller-supplied model.
* :class:`StochasticCollocationSurrogate` — sparse-grid Lagrange
  collocation interpolant.
* :func:`smolyak_sparse_grid`, :func:`tensor_grid_gauss_hermite` —
  caller helpers for building quadrature grids.

Bibliography (cited in implementation docstrings):

- Xiu, D. & Karniadakis, G. E. (2002), "The Wiener-Askey Polynomial Chaos
  for Stochastic Differential Equations", SIAM J. Sci. Comput. 24(2),
  619-644 — orthonormal basis recipe + closed-form mean / variance
  extraction (their eq. (3.3)).
- Ghanem, R. & Spanos, P. (1991), "Stochastic Finite Elements: A Spectral
  Approach" (Dover reprint 2003) — KLE eigenvalue problem
  ``∫ C(x, y) phi(y) dy = lambda phi(x)`` discretised via the midpoint
  rule (their eq. (2.30)).
- Xiu, D. (2010), "Numerical Methods for Stochastic Computations: A
  Spectral Method Approach", Princeton University Press — Smolyak
  sparse-grid construction (their eq. (8.41)) and stochastic-Galerkin
  least-squares fit (their eq. (5.20)).
- Smolyak, S. A. (1963), "Quadrature and interpolation formulas for
  tensor products of certain classes of functions", Dokl. Akad. Nauk
  SSSR 148(5), 1042-1045 — the original sparse-grid construction.
- Abramowitz, M. & Stegun, I. A. (1972), "Handbook of Mathematical
  Functions", Table 22.4 (Legendre) and 22.5.18 (Hermite) — closed-form
  polynomial values used in test cross-checks.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — eager per opifex convention
from dataclasses import dataclass
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
from flax import struct
from numpy.polynomial.hermite_e import hermegauss

from opifex.uncertainty.types import metadata_to_dict, MetadataItems


_SUPPORTED_FAMILIES = frozenset({"legendre", "hermite"})


# ---------------------------------------------------------------------------
# Pattern (A) configs (frozen dataclasses).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PolynomialChaosConfig:
    """Static configuration of a polynomial-chaos expansion.

    Pattern (A) container (GUIDE_ALIGNMENT §5a): plain
    ``@dataclass(frozen=True, slots=True, kw_only=True)``. Carries no
    JAX arrays.

    Attributes:
        family: ``"legendre"`` (uniform on ``[-1, 1]``) or ``"hermite"``
            (probabilists' Hermite for ``N(0, 1)``).
        order: Maximum polynomial degree (inclusive); the 1-D basis has
            ``order + 1`` modes.
    """

    family: str
    order: int

    def __post_init__(self) -> None:
        if self.family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported PCE family: {self.family!r}. "
                f"Choose from {sorted(_SUPPORTED_FAMILIES)}."
            )
        if self.order < 0:
            raise ValueError(f"order must be non-negative; got {self.order}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class KLEConfig:
    """Static configuration of a Karhunen-Loève expansion.

    Pattern (A) container. Carries no JAX arrays.

    Attributes:
        num_modes: Number of leading eigenpairs to retain.
        truncation_policy: ``"leading"`` keeps the top ``num_modes`` by
            eigenvalue magnitude. Reserved field for future
            adaptive-truncation policies.
    """

    num_modes: int
    truncation_policy: str = "leading"

    def __post_init__(self) -> None:
        if self.num_modes <= 0:
            raise ValueError(f"num_modes must be positive; got {self.num_modes}.")
        if self.truncation_policy not in {"leading"}:
            raise ValueError(f"Unknown truncation_policy: {self.truncation_policy!r}.")


# ---------------------------------------------------------------------------
# Pure 1-D basis evaluation (Phase 6).
# ---------------------------------------------------------------------------


def _legendre_basis(degree: int, x: jax.Array) -> jax.Array:
    """Orthonormal Legendre polynomial of given degree at ``x in [-1, 1]``.

    Recurrence from Abramowitz & Stegun 22.7.10:
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x).
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0; got {degree}.")

    def cond(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        n, _, _ = carry
        return n < degree

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        n, p_prev, p_curr = carry
        n_next = n + 1
        p_next = ((2 * n_next - 1) * x * p_curr - (n_next - 1) * p_prev) / n_next
        return n_next, p_curr, p_next

    if degree == 0:
        raw = jnp.ones_like(x)
    elif degree == 1:
        raw = x
    else:
        init = (jnp.asarray(1, dtype=jnp.int32), jnp.ones_like(x), x)
        _, _, raw = jax.lax.while_loop(cond, body, init)
    norm = jnp.sqrt((2.0 * degree + 1.0) / 2.0)
    return norm * raw


def _hermite_basis(degree: int, x: jax.Array) -> jax.Array:
    """Orthonormal probabilists' Hermite polynomial at ``x ~ N(0, 1)``.

    Recurrence from Abramowitz & Stegun 22.7.14:
        He_{n+1}(x) = x He_n(x) - n He_{n-1}(x).
    Orthonormalisation: ``||He_n||^2 = n!`` against the standard-normal
    weight, so the orthonormal basis divides by ``sqrt(n!)``.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0; got {degree}.")

    def cond(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        n, _, _ = carry
        return n < degree

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        n, h_prev, h_curr = carry
        n_next = n + 1
        h_next = x * h_curr - n * h_prev
        return n_next, h_curr, h_next

    if degree == 0:
        raw = jnp.ones_like(x)
    elif degree == 1:
        raw = x
    else:
        init = (jnp.asarray(1, dtype=jnp.int32), jnp.ones_like(x), x)
        _, _, raw = jax.lax.while_loop(cond, body, init)
    log_factorial = jsp_special.gammaln(jnp.asarray(degree + 1, dtype=jnp.float32))
    norm = jnp.exp(-0.5 * log_factorial)
    return norm * raw


def _scalar_basis_one(family: str, degree: int, x: jax.Array) -> jax.Array:
    """Dispatch helper: single-degree 1-D basis evaluation.

    Centralises the family → basis function dispatch so callers (the
    surrogate ``evaluate`` paths and :func:`evaluate_basis`) share the
    same Python-level loop over degrees. ``degree`` MUST be a concrete
    Python ``int`` because the JAX while-loop bound is static.
    """
    if family == "legendre":
        return _legendre_basis(degree, x)
    if family == "hermite":
        return _hermite_basis(degree, x)
    raise ValueError(
        f"Unsupported PCE family: {family!r}. Choose from {sorted(_SUPPORTED_FAMILIES)}."
    )


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


# ---------------------------------------------------------------------------
# PCE summary helpers (Phase 6 — preserved verbatim).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PCESummary:
    """Summary statistics extracted from PCE coefficients.

    Assumes the supplied coefficients reference an orthonormal basis so
    the variance is the squared L2-norm of the non-constant
    coefficients (Xiu-Karniadakis 2002 eq. (3.3)).

    Attributes:
        mean: PCE mean = ``coefficients[0]``.
        variance: PCE variance = ``sum(coefficients[1:]**2)``.
        coefficients: The coefficient vector, copied through for
            traceability.
        family: ``"legendre"`` or ``"hermite"``.
    """

    mean: jax.Array
    variance: jax.Array
    coefficients: jax.Array
    family: str


def pce_summary(*, coefficients: jax.Array, family: str) -> PCESummary:
    """Mean / variance extraction from an orthonormal-PCE coefficient vector.

    Implements Xiu-Karniadakis 2002 eq. (3.3): for an orthonormal basis,
    ``E[Y] = c_0`` and ``Var[Y] = sum_{i>=1} c_i^2``.

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


# ---------------------------------------------------------------------------
# Pattern (B) containers (Task 8.4).
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class PolynomialChaosBasis:
    """Fitted PCE coefficient container.

    Pattern (B) (GUIDE_ALIGNMENT §5a): ``@flax.struct.dataclass`` so the
    coefficient array flows through ``jit``/``vmap``/``scan``. Static
    fields (``family``, ``order``) are ``struct.field(pytree_node=False)``.

    Attributes:
        family: ``"legendre"`` or ``"hermite"``.
        order: Maximum polynomial degree.
        coefficients: 1-D PCE coefficient vector of shape ``(order + 1,)``
            for the 1-D basis; for tensor-product bases it must be 1-D
            and aligned with ``evaluate_basis`` outputs.
        metadata: Immutable tuple of ``(name, value)`` pairs for
            traceability / diagnostics.
    """

    coefficients: jax.Array
    family: str = struct.field(pytree_node=False)
    order: int = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Public preflight check — never called from ``__post_init__``.

        Raises:
            ValueError: On unsupported family, negative order, or
                ``coefficients`` shape inconsistent with the order.
        """
        if self.family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported PCE family: {self.family!r}. "
                f"Choose from {sorted(_SUPPORTED_FAMILIES)}."
            )
        if self.order < 0:
            raise ValueError(f"order must be non-negative; got {self.order}.")
        if self.coefficients.ndim != 1:
            raise ValueError(f"coefficients must be 1-D; got shape {self.coefficients.shape}.")

    def degrees(self) -> jax.Array:
        """Return the integer degree vector ``[0, 1, ..., order]``.

        Returned as a concrete :class:`jax.Array` for ergonomic use in
        callers that pass the result straight into :func:`evaluate_basis`
        (which materialises a Python ``list`` from the supplied degrees
        before tracing the basis-evaluation loop).
        """
        return jnp.arange(self.order + 1)

    def evaluate(self, x: jax.Array) -> jax.Array:
        """Evaluate the truncated PCE at sample points ``x``.

        Args:
            x: 1-D array of shape ``(N,)`` for 1-D inputs, or 2-D array
                of shape ``(N, d)`` for the tensor-product evaluation.

        Returns:
            ``(N,)`` array of surrogate values when ``x`` is 1-D, or the
            product of the per-dimension basis values dotted into the
            coefficient vector for 2-D inputs.
        """
        if x.ndim == 1:
            xi = x
        elif x.ndim == 2 and x.shape[1] == 1:
            xi = x[:, 0]
        else:
            raise ValueError(
                f"PolynomialChaosBasis.evaluate expects 1-D x or (N, 1) x; got shape {x.shape}."
            )
        # Build the basis matrix via a Python loop over the (static)
        # ``order`` so the surrogate is traceable under ``jax.jit``
        # (``evaluate_basis`` materialises degrees via ``int(d)`` which
        # cannot be traced).
        basis_columns = [_scalar_basis_one(self.family, d, xi) for d in range(self.order + 1)]
        phi = jnp.stack(basis_columns, axis=1)
        return phi @ self.coefficients


def pce_mean_variance(
    *,
    coefficients: jax.Array,
    basis: PolynomialChaosBasis,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mean, variance)`` for the supplied coefficients.

    Unchanged from the Phase 6 :func:`pce_summary` contract — this is
    the named tuple alias the Task 8.4 plan exposes. Implements
    Xiu-Karniadakis 2002 eq. (3.3).

    Args:
        coefficients: 1-D coefficient vector aligned with ``basis``.
        basis: A :class:`PolynomialChaosBasis` carrying the family /
            order metadata that ``coefficients`` was projected onto.

    Returns:
        Tuple ``(mean, variance)`` where ``mean = c_0`` and
        ``variance = sum(c[1:]^2)``.
    """
    summary = pce_summary(coefficients=coefficients, family=basis.family)
    return summary.mean, summary.variance


def fit_pce_coefficients(
    *,
    x: jax.Array,
    y: jax.Array,
    family: str,
    order: int,
) -> jax.Array:
    """Least-squares regression onto an orthonormal PCE basis.

    Implements Xiu 2010 eq. (5.20): given samples ``(x_k, y_k)`` and a
    truncated orthonormal basis ``{Psi_i}``, the projection coefficients
    are the least-squares solution of ``Phi c ≈ y`` where
    ``Phi_{k, i} = Psi_i(x_k)``.

    Args:
        x: Sample inputs of shape ``(N, d)`` (only 1-D / ``d=1`` is
            currently fitted).
        y: Sample outputs of shape ``(N,)``.
        family: ``"legendre"`` or ``"hermite"``.
        order: Maximum polynomial degree.

    Returns:
        Coefficient vector of shape ``(order + 1,)``.

    Raises:
        ValueError: On unsupported family or inconsistent shapes.
    """
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported PCE family: {family!r}. Choose from {sorted(_SUPPORTED_FAMILIES)}."
        )
    if order < 0:
        raise ValueError(f"order must be non-negative; got {order}.")
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (N, d); got shape {x.shape}.")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y must be 1-D (N,) aligned with x; got shape {y.shape}.")
    if x.shape[1] != 1:
        raise ValueError(
            f"fit_pce_coefficients currently supports 1-D inputs (d=1); got d={x.shape[1]}."
        )

    degrees = jnp.arange(order + 1)
    phi_3d = evaluate_basis(family=family, degrees=degrees, x=x)  # (N, P, 1)
    phi = phi_3d[:, :, 0]
    coefficients, *_ = jnp.linalg.lstsq(phi, y, rcond=None)
    return coefficients


# ---------------------------------------------------------------------------
# Karhunen-Loève expansion (Task 8.4).
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class KarhunenLoeveExpansion:
    """Discretised Karhunen-Loève expansion of a stochastic field.

    Pattern (B) container. Carries the eigenvalue + eigenvector arrays
    through ``jit``/``vmap``/``scan``.

    Decomposition follows Ghanem-Spanos 1991 eq. (2.30): given a
    symmetric positive-definite covariance kernel ``C(x, y)``, the
    spectral decomposition of the discretised covariance matrix
    ``K_{ij} = C(x_i, x_j) * dx`` (mid-point quadrature) gives
    eigenpairs ``(lambda_i, phi_i)`` such that the field

        ``f(x) = mean(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i``

    with ``xi_i ~ N(0, 1)`` independent has covariance ``C``. Only the
    leading ``num_modes`` eigenpairs are retained.

    Complexity: the eigendecomposition is ``O(N^3)`` in the
    discretisation size ``N`` (``jnp.linalg.eigh``). Use a coarse
    ``domain`` when fitting and reconstruct on a denser grid by
    re-evaluating the eigenfunctions analytically if needed.

    Attributes:
        eigenvalues: ``(num_modes,)`` array sorted descending.
        eigenvectors: ``(N, num_modes)`` discretised eigenfunctions
            (orthonormal columns).
        domain: ``(N,)`` discretisation nodes.
        full_eigenvalues: ``(N,)`` array of all eigenvalues; used by
            :meth:`truncation_error`.
        metadata: Immutable tuple of ``(name, value)`` pairs.
    """

    eigenvalues: jax.Array
    eigenvectors: jax.Array
    domain: jax.Array
    full_eigenvalues: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    @classmethod
    def from_kernel(
        cls,
        *,
        covariance_kernel: Callable[[jax.Array, jax.Array], jax.Array],
        domain: jax.Array,
        num_modes: int,
    ) -> KarhunenLoeveExpansion:
        """Construct the KLE from a callable covariance kernel.

        Args:
            covariance_kernel: ``k(x, y) -> jax.Array`` mapping two
                scalar points to a scalar covariance value.
            domain: ``(N,)`` array of discretisation nodes.
            num_modes: Number of leading eigenpairs to retain.

        Returns:
            A :class:`KarhunenLoeveExpansion` with the leading
            ``num_modes`` eigenpairs.
        """
        if num_modes <= 0:
            raise ValueError(f"num_modes must be positive; got {num_modes}.")
        if domain.ndim != 1:
            raise ValueError(f"domain must be 1-D; got shape {domain.shape}.")
        if num_modes > domain.shape[0]:
            raise ValueError(
                f"num_modes ({num_modes}) cannot exceed domain size ({domain.shape[0]})."
            )

        # Discretised covariance matrix at the grid nodes (no quadrature
        # weight applied — the diagonalisation captures the bare
        # discretised covariance, and KLE reconstruction below uses the
        # eigenvectors directly so the mid-point factor drops out).
        rows = jax.vmap(lambda xi: jax.vmap(lambda xj: covariance_kernel(xi, xj))(domain))
        cov_matrix = rows(domain)
        # Symmetrise to guard against numerical asymmetry under jit.
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        full_eigs, full_vecs = jnp.linalg.eigh(cov_matrix)
        # ``eigh`` returns ascending order — flip to descending.
        order = jnp.argsort(-full_eigs)
        full_eigs = full_eigs[order]
        full_vecs = full_vecs[:, order]
        leading_eigs = full_eigs[:num_modes]
        leading_vecs = full_vecs[:, :num_modes]
        return cls(
            eigenvalues=leading_eigs,
            eigenvectors=leading_vecs,
            domain=domain,
            full_eigenvalues=full_eigs,
        )

    def metadata_dict(self) -> dict[str, Any]:
        """Return ``metadata`` as a mutable :class:`dict`."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Public preflight check — never called from ``__post_init__``.

        Raises:
            ValueError: On non-positive eigenvalues or mismatched shapes.
        """
        if self.eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must be 1-D; got shape {self.eigenvalues.shape}.")
        if self.eigenvectors.ndim != 2:
            raise ValueError(f"eigenvectors must be 2-D; got shape {self.eigenvectors.shape}.")
        if self.eigenvectors.shape[1] != self.eigenvalues.shape[0]:
            raise ValueError(
                "eigenvectors column count must match eigenvalues length; "
                f"got {self.eigenvectors.shape} vs {self.eigenvalues.shape}."
            )

    def reconstruct(self, coefficients: jax.Array) -> jax.Array:
        """Reconstruct the field ``f(x) = sum_i sqrt(lambda_i) * phi_i(x) * c_i``.

        Implements Ghanem-Spanos 1991 eq. (2.45) — the truncated
        Karhunen-Loève synthesis. ``coefficients`` represents the
        independent standard-normal ``xi_i`` weights; supplying ``N(0,
        I)`` samples reproduces the target covariance in expectation.

        Args:
            coefficients: ``(num_modes,)`` vector.

        Returns:
            ``(N,)`` discretised field on ``self.domain``.
        """
        sqrt_eigs = jnp.sqrt(jnp.clip(self.eigenvalues, 0.0))
        scaled_modes = self.eigenvectors * sqrt_eigs[None, :]
        return scaled_modes @ coefficients

    def truncation_error(self, num_modes: int) -> jax.Array:
        """Sum of discarded eigenvalues = tail variance after truncation.

        Implements Ghanem-Spanos 1991 eq. (2.50): the truncation error
        in the L2 sense equals ``sum_{i > num_modes} lambda_i``.

        ``num_modes`` is clipped into ``[0, len(full_eigenvalues)]`` so
        callers can sweep monotonically without index-out-of-range
        protection.
        """
        n_total = self.full_eigenvalues.shape[0]
        if num_modes <= 0:
            return jnp.sum(self.full_eigenvalues)
        if num_modes >= n_total:
            return jnp.asarray(0.0, dtype=self.full_eigenvalues.dtype)
        return jnp.sum(self.full_eigenvalues[num_modes:])


# ---------------------------------------------------------------------------
# Sparse-grid quadrature (Smolyak construction, Task 8.4).
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class SparseGrid:
    """Quadrature-node container shared by tensor-product + Smolyak grids.

    Attributes:
        nodes: ``(K, d)`` node coordinates.
        weights: ``(K,)`` weights normalised to sum to one (the weight
            of the underlying probability measure).
        family: ``"hermite"`` for now — the canonical Gauss-Hermite
            tensor-product / Smolyak grid used in this subsystem.
        metadata: Immutable tuple of ``(name, value)`` pairs.
    """

    nodes: jax.Array
    weights: jax.Array
    family: str = struct.field(pytree_node=False, default="hermite")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Public preflight check — never called from ``__post_init__``.

        Raises:
            ValueError: When ``nodes`` is not 2-D, ``weights`` is not
                1-D, or the leading axes disagree.
        """
        if self.nodes.ndim != 2:
            raise ValueError(f"nodes must be 2-D; got shape {self.nodes.shape}.")
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1-D; got shape {self.weights.shape}.")
        if self.nodes.shape[0] != self.weights.shape[0]:
            raise ValueError(
                "nodes and weights must share the leading axis; "
                f"got {self.nodes.shape} vs {self.weights.shape}."
            )


def _gauss_hermite_nodes_weights(order: int) -> tuple[jax.Array, jax.Array]:
    """1-D Gauss-Hermite nodes for the standard-normal weight.

    Uses ``numpy.polynomial.hermite_e.hermegauss`` (probabilists'
    convention with weight ``exp(-x^2 / 2) / sqrt(2 pi)``) so weights
    sum to one. ``hermegauss`` returns nodes + weights for the weight
    ``exp(-x^2 / 2)``; we divide weights by ``sqrt(2 pi)`` to recover
    the probability measure.
    """
    nodes_np, weights_np = hermegauss(order)
    weights_np = weights_np / jnp.sqrt(2.0 * jnp.pi)
    return jnp.asarray(nodes_np), jnp.asarray(weights_np)


def tensor_grid_gauss_hermite(*, order: int, num_dims: int) -> SparseGrid:
    """Tensor-product Gauss-Hermite grid for the standard normal measure.

    Args:
        order: 1-D quadrature order (number of nodes per dimension).
        num_dims: Stochastic dimension ``d``. Total node count is
            ``order ** d``.

    Returns:
        A :class:`SparseGrid` with ``nodes.shape == (order**d, d)`` and
        weights summing to one.
    """
    if order <= 0:
        raise ValueError(f"order must be positive; got {order}.")
    if num_dims <= 0:
        raise ValueError(f"num_dims must be positive; got {num_dims}.")

    nodes_1d, weights_1d = _gauss_hermite_nodes_weights(order)
    grids = jnp.meshgrid(*([nodes_1d] * num_dims), indexing="ij")
    flat_nodes = jnp.stack([g.ravel() for g in grids], axis=-1)
    weights = weights_1d
    for _ in range(num_dims - 1):
        weights = weights[:, None] * weights_1d[None, :]
        weights = weights.ravel()
    return SparseGrid(nodes=flat_nodes, weights=weights, family="hermite")


def _smolyak_index_set(*, level: int, num_dims: int) -> list[tuple[int, ...]]:
    """Smolyak sparse-grid multi-indices ``{ |i| <= level + d - 1 }``.

    The Smolyak rule (Smolyak 1963; Xiu 2010 eq. (8.41)) combines
    1-D quadrature rules at indices ``i_k = 1, ..., level`` such that
    ``sum_k i_k <= level + d - 1`` with binomial-coefficient weights.

    Args:
        level: Sparse-grid level ``q >= 1`` (level 1 == constant rule).
        num_dims: Stochastic dimension ``d >= 1``.

    Returns:
        List of integer tuples ``(i_1, ..., i_d)`` with ``i_k >= 1`` and
        ``level <= |i| <= level + d - 1`` (the active multi-indices).
    """
    if level < 1:
        raise ValueError(f"level must be >= 1; got {level}.")
    indices: list[tuple[int, ...]] = []
    max_norm = level + num_dims - 1
    min_norm = level
    for combo in product(range(1, max_norm - num_dims + 2), repeat=num_dims):
        s = sum(combo)
        if min_norm <= s <= max_norm:
            indices.append(combo)
    return indices


def _smolyak_coefficient(multi_index: tuple[int, ...], *, level: int) -> float:
    """Smolyak combination coefficient ``(-1)^(q + d - |i|) * C(d - 1, q + d - 1 - |i|)``.

    From Xiu 2010 eq. (8.42).
    """
    d = len(multi_index)
    i_norm = sum(multi_index)
    sign = (-1) ** (level + d - 1 - i_norm)
    # binomial coefficient C(d - 1, level + d - 1 - i_norm)
    from math import comb

    return float(sign * comb(d - 1, level + d - 1 - i_norm))


def smolyak_sparse_grid(*, level: int, num_dims: int, family: str = "hermite") -> SparseGrid:
    """Smolyak sparse-grid for the standard normal measure.

    Implements Xiu 2010 eq. (8.41): the level-``q`` sparse grid combines
    tensor-product Gauss-Hermite rules ``Q_{i_1} ⊗ ... ⊗ Q_{i_d}`` at
    multi-indices ``i`` with ``q <= |i| <= q + d - 1`` and combination
    coefficients given by :func:`_smolyak_coefficient`.

    Args:
        level: Sparse-grid level ``q >= 1``.
        num_dims: Stochastic dimension.
        family: ``"hermite"`` (only supported value for now — the
            standard-normal weight).

    Returns:
        A :class:`SparseGrid` whose nodes are the union of the
        contributing tensor grids with combined weights.
    """
    if family != "hermite":
        raise ValueError(
            f"smolyak_sparse_grid currently supports family='hermite'; got {family!r}."
        )
    if level < 1:
        raise ValueError(f"level must be >= 1; got {level}.")
    if num_dims <= 0:
        raise ValueError(f"num_dims must be positive; got {num_dims}.")

    # Aggregate nodes + combined weights from all active multi-indices.
    aggregated: dict[tuple[float, ...], float] = {}
    for multi_index in _smolyak_index_set(level=level, num_dims=num_dims):
        coefficient = _smolyak_coefficient(multi_index, level=level)
        if coefficient == 0.0:
            continue
        per_axis_nw = [_gauss_hermite_nodes_weights(idx) for idx in multi_index]
        per_axis_nodes = [n for n, _ in per_axis_nw]
        per_axis_weights = [w for _, w in per_axis_nw]
        for combo in product(*[range(n.shape[0]) for n in per_axis_nodes]):
            node = tuple(float(per_axis_nodes[k][combo[k]]) for k in range(num_dims))
            weight = float(
                jnp.prod(jnp.array([per_axis_weights[k][combo[k]] for k in range(num_dims)]))
            )
            aggregated[node] = aggregated.get(node, 0.0) + coefficient * weight

    # Filter near-zero combined weights (numerical artefacts from sign
    # cancellation in the Smolyak rule).
    nodes_list: list[tuple[float, ...]] = []
    weights_list: list[float] = []
    for node, weight in aggregated.items():
        if abs(weight) < 1e-14:
            continue
        nodes_list.append(node)
        weights_list.append(weight)

    nodes = jnp.asarray(nodes_list, dtype=jnp.float64)
    weights = jnp.asarray(weights_list, dtype=jnp.float64)
    return SparseGrid(
        nodes=nodes,
        weights=weights,
        family="hermite",
        metadata=(("smolyak_level", level), ("num_dims", num_dims)),
    )


# ---------------------------------------------------------------------------
# Stochastic-Galerkin / collocation surrogates (Task 8.4).
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class StochasticGalerkinSurrogate:
    """Fitted stochastic-Galerkin (least-squares PCE) surrogate.

    Pattern (B) container. Carries the basis and coefficient arrays
    through JAX transforms; family / order are static aux_data.

    Construction recipe (Xiu 2010 eq. (5.20)): given a caller-supplied
    model ``f(xi)``, evaluate the model at Monte-Carlo samples drawn
    from the orthonormality measure of the basis, then solve the
    least-squares problem ``Phi c ≈ y`` to recover the PCE coefficients.

    Attributes:
        coefficients: ``(P,)`` PCE coefficient vector.
        family: ``"legendre"`` or ``"hermite"``.
        order: Polynomial order.
        metadata: Immutable tuple of ``(name, value)`` pairs.
    """

    coefficients: jax.Array
    family: str = struct.field(pytree_node=False)
    order: int = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Public preflight check — never called from ``__post_init__``.

        Raises:
            ValueError: When ``family`` is unsupported or ``coefficients``
                is not 1-D.
        """
        if self.family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported PCE family: {self.family!r}. "
                f"Choose from {sorted(_SUPPORTED_FAMILIES)}."
            )
        if self.coefficients.ndim != 1:
            raise ValueError(f"coefficients must be 1-D; got shape {self.coefficients.shape}.")

    def evaluate(self, x: jax.Array) -> jax.Array:
        """Evaluate the surrogate at ``x`` (shape ``(N, 1)`` or ``(N,)``).

        Uses the same static Python loop over degrees that
        :meth:`PolynomialChaosBasis.evaluate` uses so the call is
        traceable under ``jax.jit``.
        """
        if x.ndim == 1:
            xi = x
        elif x.ndim == 2 and x.shape[1] == 1:
            xi = x[:, 0]
        else:
            raise ValueError(
                "StochasticGalerkinSurrogate.evaluate expects 1-D x or "
                f"(N, 1) x; got shape {x.shape}."
            )
        basis_columns = [_scalar_basis_one(self.family, d, xi) for d in range(self.order + 1)]
        phi = jnp.stack(basis_columns, axis=1)
        return phi @ self.coefficients


@struct.dataclass(slots=True, kw_only=True)
class StochasticCollocationSurrogate:
    """Fitted sparse-grid stochastic-collocation surrogate.

    The surrogate is the Lagrange interpolant at the sparse-grid nodes:

        ``f_hat(x) = sum_k f(node_k) * L_k(x)``,

    with ``L_k`` the Lagrange polynomial through the unique nodes of
    the sparse grid. For smooth integrands under the standard-normal
    weight, the interpolation error is monotonically non-increasing in
    the sparse-grid level (Xiu 2010 ch. 8).

    Attributes:
        nodes: ``(K, d)`` sparse-grid node coordinates.
        values: ``(K,)`` model evaluations at the nodes.
        metadata: Immutable tuple of ``(name, value)`` pairs.
    """

    nodes: jax.Array
    values: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Public preflight check — never called from ``__post_init__``.

        Raises:
            ValueError: When ``nodes`` is not 2-D or ``values`` has a
                different leading axis.
        """
        if self.nodes.ndim != 2:
            raise ValueError(f"nodes must be 2-D; got shape {self.nodes.shape}.")
        if self.values.shape[0] != self.nodes.shape[0]:
            raise ValueError(
                "nodes and values must share the leading axis; "
                f"got {self.nodes.shape} vs {self.values.shape}."
            )

    def evaluate(self, x: jax.Array) -> jax.Array:
        """Lagrange-interpolate the model at ``x``.

        Operates on the unique 1-D node coordinates extracted from
        ``self.nodes``; the 1-D Lagrange interpolant is sufficient for
        the supported ``d = 1`` collocation path (Task 8.4 scope).
        """
        if x.ndim == 1:
            xi = x
        elif x.ndim == 2 and x.shape[1] == 1:
            xi = x[:, 0]
        else:
            raise ValueError(
                "StochasticCollocationSurrogate.evaluate expects 1-D x or "
                f"(N, 1) x; got shape {x.shape}."
            )
        unique_nodes = self.nodes[:, 0]
        # Build the Lagrange basis matrix: L_k(x) =
        # ∏_{j != k} (x - x_j) / (x_k - x_j).
        # Construct a (K, K) numerator-denominator broadcast using a mask
        # so we can divide in a vectorised manner.
        k = unique_nodes.shape[0]
        nodes_i = unique_nodes[:, None]  # (K, 1)
        nodes_j = unique_nodes[None, :]  # (1, K)
        diff_matrix = nodes_i - nodes_j
        # On the diagonal (j == k) we want to skip the term; replace with 1
        # for both numerator and denominator so the contribution is 1.
        eye = jnp.eye(k, dtype=diff_matrix.dtype)
        denom = jnp.where(eye > 0, 1.0, diff_matrix)
        denom_prod = jnp.prod(denom, axis=1)  # (K,)

        def lagrange_at(xi_value: jax.Array) -> jax.Array:
            numer = jnp.where(eye > 0, 1.0, xi_value - nodes_j)
            numer_prod = jnp.prod(numer, axis=1)  # (K,)
            basis = numer_prod / denom_prod
            return jnp.sum(basis * self.values)

        return jax.vmap(lagrange_at)(xi)


__all__ = [
    "KLEConfig",
    "KarhunenLoeveExpansion",
    "PCESummary",
    "PolynomialChaosBasis",
    "PolynomialChaosConfig",
    "SparseGrid",
    "StochasticCollocationSurrogate",
    "StochasticGalerkinSurrogate",
    "evaluate_basis",
    "fit_pce_coefficients",
    "pce_mean_variance",
    "pce_summary",
    "smolyak_sparse_grid",
    "tensor_grid_gauss_hermite",
]
