"""UQ capability declarations for the inference-backends subpackage (Task 7.2).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.uncertainty.inference_backends.__init__`` which then registers each
declaration into the singleton :class:`UQRegistry`.

Three peer posterior-sampler backends shipped under Task 6.3.9a/b/c:

* ``PathfinderBackend`` — variational + L-BFGS path (BlackJAX primitives).
* ``SVGDBackend`` — Stein Variational Gradient Descent (BlackJAX primitives).
* ``ADVIBackend`` — Automatic Differentiation Variational Inference
  (BlackJAX ``meanfield_vi`` primitives).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_PATHFINDER_BACKEND_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.VARIATIONAL,
    source_package="blackjax",
    notes=(
        "PathfinderBackend — Pathfinder variational + L-BFGS posterior "
        "sampler. Backend primitives vendored from BlackJAX "
        "(Zhang et al. arXiv:2108.03782); peer to the BlackJAX MCMC "
        "backend at the inference-backend protocol layer."
    ),
)


_SVGD_BACKEND_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.VARIATIONAL,
    source_package="blackjax",
    notes=(
        "SVGDBackend — Stein Variational Gradient Descent posterior "
        "sampler. Backend primitives vendored from BlackJAX "
        "(Liu & Wang NIPS 2016); peer to the Pathfinder / ADVI "
        "backends at the inference-backend protocol layer."
    ),
)


_ADVI_BACKEND_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.VARIATIONAL,
    source_package="blackjax",
    notes=(
        "ADVIBackend — Automatic Differentiation Variational Inference "
        "(mean-field Gaussian). Backend primitives vendored from "
        "BlackJAX ``meanfield_vi`` (Kucukelbir et al. JMLR 18(14)); "
        "peer to the Pathfinder / SVGD backends at the "
        "inference-backend protocol layer."
    ),
)


INFERENCE_BACKEND_CAPABILITIES: dict[str, UQCapability] = {
    "backend:pathfinder": _PATHFINDER_BACKEND_CAPABILITY,
    "backend:svgd": _SVGD_BACKEND_CAPABILITY,
    "backend:advi": _ADVI_BACKEND_CAPABILITY,
}


__all__ = ["INFERENCE_BACKEND_CAPABILITIES"]
