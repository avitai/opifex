"""State-owning UQ layers (NNX modules).

Mirror of :mod:`opifex.uncertainty.kernels` on the state-owning side:

* :mod:`opifex.uncertainty.kernels.bayesian` — pure JAX Bayesian math
  (no ``flax.nnx`` imports; see Phase 9 Task 9.3 boundary scan).
* :mod:`opifex.uncertainty.layers.bayesian` — trainable Bayesian NNX
  modules that *consume* the pure helpers.

Path names describe the UQ domain (Bayesian / spectral / etc.); they do not
name the underlying framework — that is an implementation detail.
"""

from __future__ import annotations

from opifex.uncertainty.layers.bayesian import (
    BayesianLinear,
    BayesianSpectralConvolution,
)


__all__ = ["BayesianLinear", "BayesianSpectralConvolution"]
