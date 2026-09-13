"""Matrix-free linear algebra primitives for uncertainty quantification.

JAX-native implementations of Krylov decompositions, stochastic
trace and diagonal estimators, low-rank approximations, randomized SVD,
matrix functions, log-determinant integrands, differentiable least-squares,
and higher-moment trace UQ.

Pure JAX; no NNX imports anywhere in this subpackage. Each algorithm cites
its defining paper in the module docstring.

References:
----------
* Krämer, Moreno-Muñoz, Roy, Hauberg arXiv:2405.17277 — *Gradients of
  functions of large matrices* (differentiable Lanczos/Arnoldi).
* Epperly, Tropp, Webber arXiv:2301.07825 — *XTrace* (XTrace, XNysTrace);
  Meyer, Musco, Musco, Woodruff arXiv:2010.09649 — *Hutch++*.
* Potapczynski, Finzi, Pleiss, Wilson arXiv:2309.03060 — *CoLA* (structured
  operators; used by :mod:`opifex.uncertainty.curvature.structured`).
"""

from __future__ import annotations

from opifex.uncertainty.linalg._uq_capabilities import LINALG_CAPABILITIES
from opifex.uncertainty.linalg.eig import eig_partial, eigh_partial, svd_partial
from opifex.uncertainty.linalg.funm import (
    dense_funm_sym_eigh,
    funm_arnoldi,
    funm_chebyshev,
    funm_lanczos_sym,
)
from opifex.uncertainty.linalg.krylov import (
    arnoldi_hessenberg,
    golub_kahan_bidiag,
    lanczos_tridiag,
)
from opifex.uncertainty.linalg.logdet import slq_logdet
from opifex.uncertainty.linalg.lowrank import cholesky_greedy, rp_cholesky
from opifex.uncertainty.linalg.lstsq import lsmr
from opifex.uncertainty.linalg.moments import trace_moments
from opifex.uncertainty.linalg.rsvd import randomized_svd
from opifex.uncertainty.linalg.trace import (
    hutch_plus_plus_trace,
    hutchinson_trace,
    xnys_trace,
    xtrace,
)
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.2. Singleton :class:`UQRegistry`
# guarded against duplicate registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in LINALG_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "LINALG_CAPABILITIES",
    "arnoldi_hessenberg",
    "cholesky_greedy",
    "dense_funm_sym_eigh",
    "eig_partial",
    "eigh_partial",
    "funm_arnoldi",
    "funm_chebyshev",
    "funm_lanczos_sym",
    "golub_kahan_bidiag",
    "hutch_plus_plus_trace",
    "hutchinson_trace",
    "lanczos_tridiag",
    "lsmr",
    "randomized_svd",
    "rp_cholesky",
    "slq_logdet",
    "svd_partial",
    "trace_moments",
    "xnys_trace",
    "xtrace",
]
