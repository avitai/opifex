r"""Diagonal Laplace posterior approximation.

For a MAP estimate ``θ*`` and a Gaussian prior with precision
``prior_precision`` (scalar applied isotropically), the diagonal Laplace
posterior is

.. math::

    \\theta\\,|\\,\\mathcal{D} \\sim \\mathcal{N}\\bigl(
        \\theta^{*},\\,
        \\operatorname{diag}\\bigl(\\tau\\,\\mathbf{1} + F(\\theta^{*})\\bigr)^{-1}
    \\bigr),

where ``F(θ*)`` is the empirical-Fisher diagonal at the MAP point and
``τ`` is the scalar prior precision.

Canonical reference:
* Daxberger Laplace package and bayesian-torch use this same diagonal
  precision formula. opifex computes the empirical-Fisher term via
  :func:`empirical_fisher_diagonal` and adds ``τ * 1`` for the prior.

References
----------
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806.
* MacKay, D. J. C. 1992 — *A practical Bayesian framework for
  backpropagation networks*, Neural Computation 4(3).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass

import jax  # noqa: TC002 — kept eager for consistency with the rest of opifex.uncertainty

from opifex.uncertainty.curvature.fisher import empirical_fisher_diagonal


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagonalLaplacePosterior:
    """Mean + diagonal precision of a Laplace posterior.

    Attributes:
        mean: MAP estimate ``θ*``.
        precision_diagonal: Per-parameter precision ``τ + F_ii``.
    """

    mean: jax.Array
    precision_diagonal: jax.Array


def diagonal_laplace_posterior(
    *,
    per_sample_loss: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    map_estimate: jax.Array,
    inputs: jax.Array,
    targets: jax.Array,
    prior_precision: float,
) -> DiagonalLaplacePosterior:
    """Build a diagonal-Fisher Laplace posterior at a MAP point.

    Args:
        per_sample_loss: Maps ``(parameters, input, target) -> scalar``.
        map_estimate: MAP estimate ``θ*``.
        inputs: Batched inputs.
        targets: Batched targets.
        prior_precision: Scalar prior precision ``τ`` (must be ``> 0``).

    Returns:
        :class:`DiagonalLaplacePosterior` with ``precision_diagonal`` of
        the same shape as ``map_estimate``.

    Raises:
        ValueError: If ``prior_precision`` is not strictly positive.
    """
    if prior_precision <= 0.0:
        raise ValueError(f"prior_precision must be positive; got {prior_precision!r}")
    fisher_diagonal = empirical_fisher_diagonal(per_sample_loss, map_estimate, inputs, targets)
    precision_diagonal = prior_precision + fisher_diagonal
    return DiagonalLaplacePosterior(mean=map_estimate, precision_diagonal=precision_diagonal)
