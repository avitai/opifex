"""Pathfinder variational inference peer backend.

Pathfinder runs a quasi-Newton optimisation along an L-BFGS trajectory,
fits a Gaussian variational approximation at each step, and selects the
best-ELBO mean-field posterior. It scales well to large parameter
counts and produces useful proposal distributions for downstream
sampling.

Canonical reference:
* Zhang, L., Carpenter, B., Gelman, A., Vehtari, A. 2022 —
  *Pathfinder: Parallel quasi-Newton variational inference*, JMLR
  23(306). arXiv:2108.03782.

This slice ships the Pattern-A metadata + protocol surface via
:class:`_DeferredInferenceBackend`; the concrete L-BFGS-Gaussian
fitting algorithm lands in a follow-up.
"""

from __future__ import annotations

import dataclasses

from opifex.uncertainty.inference_backends._deferred import _DeferredInferenceBackend


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PathfinderBackend(_DeferredInferenceBackend):
    """Pathfinder variational inference backend."""

    name: str = "pathfinder"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ("pathfinder",)
    notes: str = (
        "Pathfinder (Zhang+ 2022, arXiv:2108.03782) — quasi-Newton "
        "variational inference along an L-BFGS trajectory."
    )


__all__ = ["PathfinderBackend"]
