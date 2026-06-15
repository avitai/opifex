r"""GP-PINN — function-valued GP predictive over a Physics-Informed NN.

A *Gaussian-Process Physics-Informed Neural Network* (GP-PINN) treats a
trained PINN as a function-valued Gaussian Process via the **linearised
Laplace equivalence** of Immer, Korzepa & Bauer 2021 (AISTATS,
arXiv:2008.08400 §3 — "Improving predictions of Bayesian neural nets
via local linearisation"). Given a PINN forward ``f(x; θ)`` and a
diagonal Laplace posterior ``θ ~ N(θ*, Σ)`` with
``Σ = diag(1 / precision)``, the first-order Taylor expansion about
``θ*`` yields the closed-form

.. math::

    \mu(x)        &= f(x; \theta^{*}), \\
    \mathrm{Var}(x) &= J_\theta f(x; \theta^{*})\,\Sigma\,J_\theta f(x; \theta^{*})^{T}
                     = \sum_i \bigl(\partial f / \partial \theta_i\bigr)^2 / \mathrm{precision}_i,

which is exactly the formula implemented by Task 10.1's
:func:`linearized_neural_operator_posterior`. GP-PINN therefore
**delegates the math** to LUNO and augments the resulting
:class:`~opifex.uncertainty.types.PredictiveDistribution` metadata
with the chosen Task 6.3 GP adapter spec (``GPJaxAdapterSpec``,
``TinygpAdapterSpec``, ``BayesnewtonAdapterSpec``, …) so the surface
advertises the GP-equivalence relationship explicitly. When Phase 11
ships concrete GP fit / predict implementations, ``gp_pinn_predictive_posterior``
will gain an alternative branch that fits a real GP on top of the PINN
features rather than relying solely on the linearised-Laplace
equivalence — the current adapter-mediated form is forward-compatible.

References
----------
* Immer, A., Korzepa, M., Bauer, M. 2021 — *Improving predictions of
  Bayesian neural nets via local linearisation*, AISTATS,
  arXiv:2008.08400 (PRIMARY — NN-as-GP linearised-Laplace equivalence).
* Daxberger, E. et al. 2021 — *Laplace Redux*, arXiv:2106.14806
  (parameter-space diagonal Laplace posterior).
* Karniadakis, G. et al. 2022 — *Physics-informed machine learning*,
  Nat. Rev. Phys. (GP-PINN motivation).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax  # noqa: TC002 — kept eager for consistency

from opifex.uncertainty.adapters.gp import _GPAdapterSpecBase
from opifex.uncertainty.curvature.laplace import (
    DiagonalLaplacePosterior,  # noqa: TC001 — kept eager for consistency
)
from opifex.uncertainty.curvature.luno import linearized_neural_operator_posterior
from opifex.uncertainty.types import PredictiveDistribution


_GP_PINN_SOURCE_PACKAGE = "opifex.neural.pinns.gp_pinn"


def gp_pinn_predictive_posterior(
    *,
    pinn_forward: Callable[[jax.Array, jax.Array], jax.Array],
    laplace_posterior: DiagonalLaplacePosterior,
    coordinates: jax.Array,
    gp_adapter_spec: _GPAdapterSpecBase,
) -> PredictiveDistribution:
    r"""GP-PINN function-valued predictive (linearised-Laplace equivalence).

    Args:
        pinn_forward: ``(parameters, coordinates) -> field_value`` PINN
            forward function. The ``parameters`` axis must be the flat
            parameter vector that ``laplace_posterior.mean`` is defined
            over; ``coordinates`` is a batched spatial /
            spatio-temporal coordinate tensor.
        laplace_posterior: Diagonal Laplace posterior at ``θ*``;
            supplies the MAP point and the per-parameter posterior
            precision.
        coordinates: Coordinate batch passed straight through to
            ``pinn_forward`` and to the LUNO Jacobian computation.
        gp_adapter_spec: A :class:`_GPAdapterSpecBase` subclass (e.g.
            :class:`TinygpAdapterSpec`, :class:`GPJaxAdapterSpec`,
            :class:`BayesnewtonAdapterSpec`) that advertises the GP
            substrate this predictive is *equivalent to*. The spec is
            not invoked at runtime; its ``source_package`` and
            ``family_tags`` are merged into the predictive metadata so
            downstream consumers can resolve the linearised-Laplace ↔
            GP correspondence.

    Returns:
        A :class:`PredictiveDistribution` whose ``mean`` is
        ``pinn_forward(θ*, coordinates)`` and whose marginal
        ``variance`` is ``diag(J Σ J^T)``. Metadata carries the
        ``gp_adapter_source_package`` + family tags of the chosen GP
        spec and the ``estimator=gp_pinn_linearized_laplace`` tag.

    Raises:
        TypeError: If ``gp_adapter_spec`` is not a subclass of
            :class:`_GPAdapterSpecBase`.
    """
    if not isinstance(gp_adapter_spec, _GPAdapterSpecBase):
        raise TypeError(
            "gp_pinn_predictive_posterior requires a GP adapter spec "
            "(subclass of opifex.uncertainty.adapters.gp._GPAdapterSpecBase) "
            f"— e.g. TinygpAdapterSpec / GPJaxAdapterSpec; got {type(gp_adapter_spec)!r}."
        )

    base = linearized_neural_operator_posterior(
        model_fn=pinn_forward,
        laplace_posterior=laplace_posterior,
        x=coordinates,
    )

    augmented_metadata = (
        *base.metadata,
        ("estimator", "gp_pinn_linearized_laplace"),
        ("source_package", _GP_PINN_SOURCE_PACKAGE),
        ("gp_adapter_source_package", gp_adapter_spec.source_package),
        ("gp_adapter_family_tags", gp_adapter_spec.family_tags),
        ("paper", "Immer+ 2021 arXiv:2008.08400"),
    )
    return PredictiveDistribution(
        mean=base.mean,
        variance=base.variance,
        epistemic=base.epistemic,
        total_uncertainty=base.total_uncertainty,
        metadata=augmented_metadata,
    )


__all__ = ["gp_pinn_predictive_posterior"]
