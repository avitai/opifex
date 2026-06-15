"""Curvature primitives for second-order uncertainty quantification.

Pure-JAX building blocks for second-order methods — Hessian-vector
products, generalized Gauss-Newton (GGN) products, empirical-Fisher
diagonal estimators, and the Laplace posterior approximation. The
public adapter contract for the diagonal-Laplace posterior
(:class:`LaplaceAdapterSpec` + :class:`LaplaceState`) is co-located in
:mod:`.laplace` so the curvature kernels and the spec that consumes
them live in a single canonical home.

References
----------
* MacKay, D. J. C. 1992 — *A practical Bayesian framework for
  backpropagation networks*, Neural Computation 4(3).
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806.
* Pearlmutter, B. A. 1994 — *Fast Exact Multiplication by the Hessian*,
  Neural Computation 6(1).
"""

from __future__ import annotations

from opifex.uncertainty.curvature._uq_capabilities import CURVATURE_CAPABILITIES
from opifex.uncertainty.curvature.fisher import empirical_fisher_diagonal
from opifex.uncertainty.curvature.ggn import ggn_vector_product
from opifex.uncertainty.curvature.hessian import hessian_vector_product
from opifex.uncertainty.curvature.kfac import (
    kfac_factors,
    kfac_laplace_posterior,
    KroneckerLaplacePosterior,
    TappedModel,
)
from opifex.uncertainty.curvature.laplace import (
    diagonal_laplace_posterior,
    DiagonalLaplacePosterior,
    LaplaceAdapterSpec,
    LaplaceState,
)
from opifex.uncertainty.curvature.luno import linearized_neural_operator_posterior
from opifex.uncertainty.curvature.structured import (
    BlockDiagonal,
    DiagonalOperator,
    IdentityOperator,
    KroneckerProduct,
    LowRankUpdate,
    StructuredOperator,
)
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.2. Singleton :class:`UQRegistry`
# guarded against duplicate registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in CURVATURE_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "CURVATURE_CAPABILITIES",
    "BlockDiagonal",
    "DiagonalLaplacePosterior",
    "DiagonalOperator",
    "IdentityOperator",
    "KroneckerLaplacePosterior",
    "KroneckerProduct",
    "LaplaceAdapterSpec",
    "LaplaceState",
    "LowRankUpdate",
    "StructuredOperator",
    "TappedModel",
    "diagonal_laplace_posterior",
    "empirical_fisher_diagonal",
    "ggn_vector_product",
    "hessian_vector_product",
    "kfac_factors",
    "kfac_laplace_posterior",
    "linearized_neural_operator_posterior",
]
