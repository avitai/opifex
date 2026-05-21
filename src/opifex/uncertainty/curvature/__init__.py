"""Curvature primitives for second-order uncertainty quantification.

Pure-JAX building blocks for second-order methods — Hessian-vector
products, generalized Gauss-Newton (GGN) products, empirical-Fisher
diagonal estimators, and the Laplace posterior approximation. The
``LaplaceAdapter`` in ``opifex.uncertainty.adapters.model`` wires these
together with the public adapter contract.

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

from opifex.uncertainty.curvature.fisher import empirical_fisher_diagonal
from opifex.uncertainty.curvature.ggn import ggn_vector_product
from opifex.uncertainty.curvature.hessian import hessian_vector_product
from opifex.uncertainty.curvature.laplace import (
    diagonal_laplace_posterior,
    DiagonalLaplacePosterior,
)


__all__ = [
    "DiagonalLaplacePosterior",
    "diagonal_laplace_posterior",
    "empirical_fisher_diagonal",
    "ggn_vector_product",
    "hessian_vector_product",
]
