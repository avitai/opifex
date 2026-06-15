"""Automatic Differentiation Variational Inference (ADVI) peer backend.

ADVI fits a mean-field or full-rank Gaussian variational posterior on
an unconstrained reparametrisation of the model parameters. The
ELBO is optimised via stochastic gradient ascent over Monte-Carlo
estimates of the expectation.

Canonical reference:
* Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., Blei, D. M. 2017
  — *Automatic Differentiation Variational Inference*, JMLR 18(14).

This slice ships the Pattern-A metadata + protocol surface via
:class:`_DeferredInferenceBackend`; the concrete ELBO-maximisation
algorithm lands in a follow-up.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from opifex.uncertainty.inference_backends._deferred import _DeferredInferenceBackend


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ADVIBackend(_DeferredInferenceBackend):
    """ADVI variational inference backend.

    Adds the ``family`` axis (``"meanfield"`` or ``"fullrank"``) on top of
    the shared deferred-backend metadata surface.
    """

    name: str = "advi"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ("advi",)
    family: Literal["meanfield", "fullrank"] = "meanfield"
    notes: str = (
        "ADVI (Kucukelbir+ 2017) — mean-field or full-rank Gaussian "
        "variational posterior optimised over the ELBO via "
        "automatic-differentiation reparametrisation."
    )


__all__ = ["ADVIBackend"]
