"""Stochastic-Galerkin / stochastic-collocation fitting glue — Task 8.4.

The surrogate containers themselves live in :mod:`polynomial_chaos`
(both :class:`~polynomial_chaos.StochasticGalerkinSurrogate` and
:class:`~polynomial_chaos.StochasticCollocationSurrogate`).  This module
holds the caller-facing ``fit`` / ``evaluate`` helpers that:

* draw samples / accept a sparse-grid quadrature rule.
* evaluate the caller-supplied ``model_callable: Callable[[jax.Array],
  jax.Array]`` (no coupling to any specific Opifex solver class).
* return a fitted surrogate container (deterministic given a fixed key).

Bibliography (cited in implementation docstrings):

- Xiu, D. (2010), "Numerical Methods for Stochastic Computations", ch. 5
  — least-squares stochastic-Galerkin projection.
- Xiu, D. (2010), "Numerical Methods for Stochastic Computations", ch. 8
  — sparse-grid stochastic collocation.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.rng import key_from

from opifex.uncertainty.scientific.polynomial_chaos import (
    fit_pce_coefficients,
    PolynomialChaosBasis,
    SparseGrid,
    StochasticCollocationSurrogate,
    StochasticGalerkinSurrogate,
)


_SG_RNG_STREAM = "sg_samples"


def fit_galerkin_surrogate(
    *,
    model: Callable[[jax.Array], jax.Array],
    basis: PolynomialChaosBasis,
    num_samples: int,
    rngs: nnx.Rngs | jax.Array,
) -> StochasticGalerkinSurrogate:
    """Fit a stochastic-Galerkin (least-squares PCE) surrogate.

    Implements the regression-projection variant of stochastic-Galerkin
    (Xiu 2010 eq. (5.20)): the surrogate's coefficients are the
    least-squares solution of ``Phi c ≈ y`` over Monte-Carlo samples
    drawn from the basis's orthonormality measure.

    Args:
        model: A pure JAX callable ``model(xi) -> y`` mapping
            ``(N, 1)`` stochastic inputs to ``(N,)`` outputs.
        basis: A :class:`PolynomialChaosBasis` declaring the family +
            order. The ``coefficients`` field of the input ``basis`` is
            ignored — only ``family`` and ``order`` are read.
        num_samples: Monte-Carlo sample count.
        rngs: The samples' key, or an ``nnx.Rngs`` whose first stream of
            ``"sg_samples"``, ``"sample"``, ``"default"`` draws it.

    Returns:
        A fitted :class:`StochasticGalerkinSurrogate` whose
        ``coefficients`` reproduce the regression solution.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}.")

    key = key_from(
        rngs, streams=(_SG_RNG_STREAM, "sample", "default"), context="stochastic-Galerkin samples"
    )

    samples = _sample_from_measure(family=basis.family, key=key, shape=(num_samples,))
    targets = model(samples[:, None])
    coefficients = fit_pce_coefficients(
        x=samples[:, None],
        y=targets,
        family=basis.family,
        order=basis.order,
    )
    return StochasticGalerkinSurrogate(
        coefficients=coefficients,
        family=basis.family,
        order=basis.order,
        metadata=(("fit_method", "least_squares"), ("num_samples", num_samples)),
    )


def evaluate_galerkin_surrogate(
    *,
    surrogate: StochasticGalerkinSurrogate,
    x: jax.Array,
) -> jax.Array:
    """Public ``evaluate`` glue — delegates to ``surrogate.evaluate(x)``.

    Provided so callers can rely on a free-function entry point that
    mirrors :func:`fit_galerkin_surrogate`.
    """
    return surrogate.evaluate(x)


def fit_collocation_surrogate(
    *,
    model: Callable[[jax.Array], jax.Array],
    sparse_grid: SparseGrid,
) -> StochasticCollocationSurrogate:
    """Fit a sparse-grid stochastic-collocation surrogate.

    Evaluates ``model`` at the unique grid nodes; the returned surrogate
    re-evaluates via Lagrange interpolation in :meth:`evaluate`.

    Implements the sparse-grid collocation construction of Xiu 2010
    ch. 8 — for smooth integrands under the standard-normal weight the
    interpolation error is monotonically non-increasing in the
    Smolyak-grid level.

    Args:
        model: A pure JAX callable.
        sparse_grid: A :class:`SparseGrid` (Smolyak or tensor-product).

    Returns:
        A fitted :class:`StochasticCollocationSurrogate`.
    """
    if sparse_grid.nodes.ndim != 2:
        raise ValueError(f"sparse_grid.nodes must be 2-D; got shape {sparse_grid.nodes.shape}.")

    # Restrict to unique 1-D nodes — sparse grids may include duplicate
    # node coordinates across multi-indices.
    unique_nodes = jnp.unique(sparse_grid.nodes[:, 0])
    nodes_2d = unique_nodes[:, None]
    values = model(nodes_2d)
    return StochasticCollocationSurrogate(
        nodes=nodes_2d,
        values=values,
        metadata=(
            ("grid_family", sparse_grid.family),
            ("num_unique_nodes", int(unique_nodes.shape[0])),
        ),
    )


def evaluate_collocation_surrogate(
    *,
    surrogate: StochasticCollocationSurrogate,
    x: jax.Array,
) -> jax.Array:
    """Public ``evaluate`` glue — delegates to ``surrogate.evaluate(x)``."""
    return surrogate.evaluate(x)


def _sample_from_measure(*, family: str, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Draw samples from the orthonormality measure of ``family``.

    * ``"legendre"`` → uniform on ``[-1, 1]``.
    * ``"hermite"`` → standard normal.
    """
    if family == "legendre":
        return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
    if family == "hermite":
        return jax.random.normal(key, shape)
    raise ValueError(f"Unsupported PCE family: {family!r}.")


__all__ = [
    "evaluate_collocation_surrogate",
    "evaluate_galerkin_surrogate",
    "fit_collocation_surrogate",
    "fit_galerkin_surrogate",
]
