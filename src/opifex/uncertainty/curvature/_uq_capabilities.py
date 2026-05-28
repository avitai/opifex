"""UQ capability declaration for the curvature subpackage (Task 7.2).

Static, module-level constant — no import-time mutable side effects beyond
the constant itself (Rule 13).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_CURVATURE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.LAPLACE,
    source_package="traceax+matfree+kfac-jax",
    notes=(
        "Curvature primitives — Hessian-vector products (Pearlmutter "
        "Neural Computation 6(1)), generalized Gauss-Newton (GGN) "
        "vector products, empirical-Fisher diagonal estimators, and "
        "the diagonal Laplace posterior approximation (MacKay 1992; "
        "Daxberger et al. arXiv:2106.14806). Pure JAX kernels."
    ),
)


CURVATURE_CAPABILITIES: dict[str, UQCapability] = {
    "subpackage:curvature": _CURVATURE_CAPABILITY,
}


__all__ = ["CURVATURE_CAPABILITIES"]
