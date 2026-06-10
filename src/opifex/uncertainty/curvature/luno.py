r"""Linearised Neural Operator (LUNO) predictive posterior.

For a trained model ``f(x; θ)`` with MAP estimate ``θ*`` and diagonal
Laplace posterior ``θ | data ~ N(θ*, Σ)`` (with ``Σ = diag(1 /
precision)``), the first-order Taylor expansion of the network output
around ``θ*`` yields a closed-form function-valued Gaussian Process
predictive:

.. math::

    \mu(x) &= f(x; \theta^{*}), \\
    \Sigma_{\text{pred}}(x, x') &=
        J_{\theta} f(x; \theta^{*})\,\Sigma\,J_{\theta} f(x'; \theta^{*})^{T},

where ``J_θ`` is the Jacobian of the network output with respect to the
flattened parameter vector. This module computes the *marginal pointwise
variance* at every input ``x`` — ``Var(x) = J_θ f(x) · Σ · J_θ f(x)^T``
— which is the quantity consumed by downstream
:class:`PredictiveDistribution` reporting.

For the linear toy model ``f(θ, x) = x · θ`` the Jacobian wrt ``θ`` is
``x`` exactly, so the predictive marginal variance collapses to
``Σ_i x_i^2 / precision_i``. The tests in
``tests/uncertainty/curvature/test_luno.py`` verify this closed-form
identity end-to-end.

Implementation note: opifex implements LUNO *natively in JAX*. The
recommended reference substrate ``tinygp`` (Magnani+ 2024 also points to
it for its ``Transform(Kernel)`` pattern) is an optional adapter-only
dependency in opifex — :class:`TinygpAdapterSpec` exposes it without
making it a runtime import. The local implementation uses
:func:`jax.jacrev` over the parameter axis and contracts with the
diagonal posterior covariance ``1 / precision_diagonal`` to compute the
marginal variance in pure JAX.

References
----------
* Magnani, E. et al. 2024 — *Linearised neural operators for function
  uncertainty quantification*, arXiv:2406.04317 (PRIMARY).
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806 (parameter-space Laplace posterior).
* MacKay, D. J. C. 1992 — *A practical Bayesian framework for
  backpropagation networks*, Neural Computation 4(3) (original
  linearised-Laplace formulation).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.curvature.laplace import (
    DiagonalLaplacePosterior,  # noqa: TC001 — kept eager for consistency
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_LUNO_SOURCE_PACKAGE = "opifex.uncertainty.curvature"


def linearized_neural_operator_posterior(
    *,
    model_fn: Callable[[jax.Array, jax.Array], jax.Array],
    laplace_posterior: DiagonalLaplacePosterior,
    x: jax.Array,
) -> PredictiveDistribution:
    r"""Linearised-Laplace function-valued GP predictive (Magnani+ 2024).

    Args:
        model_fn: ``(parameters, x) -> y`` — the trained network. The
            ``parameters`` axis must be the flat parameter vector that
            ``laplace_posterior.mean`` is defined over.
        laplace_posterior: Diagonal Laplace posterior at ``θ*``; supplies
            the MAP point and the per-parameter posterior precision.
        x: Input batch of shape ``(batch, ...)``.

    Returns:
        A :class:`PredictiveDistribution` whose ``mean`` is
        ``model_fn(θ*, x)`` and whose ``variance`` is the pointwise
        marginal variance ``diag(J Σ J^T)`` with ``Σ = diag(1 /
        precision)``. ``epistemic`` and ``total_uncertainty`` are set
        equal to ``variance`` because the linearised-Laplace predictive
        carries only parameter uncertainty (no aleatoric noise term).
    """
    posterior_variance = 1.0 / laplace_posterior.precision_diagonal

    def _predict(theta: jax.Array) -> jax.Array:
        """Evaluate the neural operator at parameters ``theta`` for the linearisation point."""
        return model_fn(theta, x)

    mean = _predict(laplace_posterior.mean)
    jacobian = jax.jacrev(_predict)(laplace_posterior.mean)
    # ``jacobian`` has shape ``mean.shape + posterior.mean.shape``; the
    # marginal variance is the sum over the parameter axis (the trailing
    # axes that match ``laplace_posterior.mean``) of ``J^2 * Σ_diag``.
    param_ndim = laplace_posterior.mean.ndim
    param_axes = tuple(range(jacobian.ndim - param_ndim, jacobian.ndim))
    variance = jnp.sum((jacobian**2) * posterior_variance, axis=param_axes)

    metadata = compose_method_metadata(
        method=DefaultStrategy.LAPLACE.value,
        source_package=_LUNO_SOURCE_PACKAGE,
        extra=(
            ("estimator", "linearized_neural_operator"),
            ("paper", "Magnani+ 2024 arXiv:2406.04317"),
        ),
    )
    return gaussian_process_predictive(
        mean,
        variance,
        epistemic=variance,
        total_uncertainty=variance,
        metadata=metadata,
    )


__all__ = ["linearized_neural_operator_posterior"]
