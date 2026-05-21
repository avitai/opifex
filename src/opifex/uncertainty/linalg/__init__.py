"""Matrix-free linear algebra primitives for uncertainty quantification.

Vendored JAX-native implementations of Krylov decompositions, stochastic
trace and diagonal estimators, low-rank approximations, randomized SVD,
matrix functions, log-determinant integrands, differentiable least-squares,
and higher-moment trace UQ.

Pure JAX; no NNX imports anywhere in this subpackage. Each algorithm cites
its canonical reference (paper + sibling-repo path) in the module docstring.

The sibling repositories ``/mnt/ssd2/Works/{matfree,traceax,cola}`` are
reference implementations only — opifex never carries them as runtime
dependencies. Algorithms are implemented natively in JAX and cite the
sibling source line-by-line.

References
----------
* matfree — Krämer arXiv:2405.17277 (differentiable Lanczos/Arnoldi).
* traceax — Nahid et al. (XTrace, XNysTrace, Hutch++).
* cola — Potapczynski et al. arXiv:2309.03060 (structured operators;
  vendored under :mod:`opifex.uncertainty.curvature.structured`).
"""

from __future__ import annotations

from opifex.uncertainty.linalg.trace import hutch_plus_plus_trace, hutchinson_trace


__all__ = ["hutch_plus_plus_trace", "hutchinson_trace"]
