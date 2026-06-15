"""UQ capability declaration for the matrix-free linalg subpackage (Task 7.2).

Static, module-level constant — no import-time mutable side effects beyond
the constant itself (Rule 13). Imported by
``opifex.uncertainty.linalg.__init__`` which registers it into the
singleton :class:`UQRegistry`.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_LINALG_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.RANDOMIZED_LINALG,
    source_package="matfree+traceax",
    notes=(
        "Matrix-free linalg primitives — Krylov decompositions, "
        "stochastic trace / diagonal estimators (Hutchinson, Hutch++, "
        "XTrace, XNysTrace), low-rank approximations, randomized SVD, "
        "matrix-function evaluation, log-determinant integrands, "
        "differentiable LSMR. Pure JAX kernels citing matfree "
        "(Krämer arXiv:2405.17277) and traceax (Nahid et al.)."
    ),
)


LINALG_CAPABILITIES: dict[str, UQCapability] = {
    "subpackage:linalg": _LINALG_CAPABILITY,
}


__all__ = ["LINALG_CAPABILITIES"]
