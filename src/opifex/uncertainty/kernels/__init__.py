"""Pure JAX kernel helpers for UQ.

Modules under :mod:`opifex.uncertainty.kernels` are pure JAX functions only — no
``flax.nnx`` imports. NNX modules consume these kernels through call-time
composition; a container-pattern audit enforces the boundary.
"""

from __future__ import annotations

from opifex.uncertainty.kernels.bayesian import (
    diagonal_gaussian_kl,
    sample_diagonal_gaussian,
)


__all__ = ["diagonal_gaussian_kl", "sample_diagonal_gaussian"]
