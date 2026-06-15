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


_LUNO_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.LAPLACE,
    source_package="opifex.uncertainty.curvature",
    notes=(
        "Linearised neural-operator predictive posterior (Magnani et al. "
        "2024, arXiv:2406.04317). Given a diagonal Laplace posterior at "
        "the MAP point, the network is locally linearised and treated "
        "as a function-valued Gaussian process; the marginal predictive "
        "variance is ``diag(J Σ J^T)`` computed via ``jax.jacrev``. "
        "Pure JAX kernel; passes jit / vmap smokes."
    ),
)


_KFAC_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.LAPLACE,
    source_package="opifex.uncertainty.curvature",
    notes=(
        "Kronecker-factored approximate curvature (KFAC) Laplace posterior "
        "(Martens & Grosse 2015, arXiv:1503.05671; Ritter, Botev & Barber "
        "2018, ICLR). Each layer's GGN/Fisher block is factored as A ⊗ G "
        "with A = E[a a^T] (input-activation covariance) and G = "
        "E[J_s^T H_y J_s] (pre-activation GGN factor), assembled into a "
        "damped block-diagonal-of-Kronecker posterior precision over the "
        "structured operators of CoLA (Potapczynski et al. 2023, "
        "arXiv:2309.03060). Factors are computed natively via jax.vjp "
        "through a tapped forward; pure JAX kernel; passes jit / grad / "
        "vmap smokes."
    ),
)


CURVATURE_CAPABILITIES: dict[str, UQCapability] = {
    "subpackage:curvature": _CURVATURE_CAPABILITY,
    "estimator:linearized_neural_operator_posterior": _LUNO_CAPABILITY,
    "estimator:kfac_laplace_posterior": _KFAC_CAPABILITY,
}


__all__ = ["CURVATURE_CAPABILITIES"]
