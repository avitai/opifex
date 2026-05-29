r"""Continuous Thompson sampling for Bayesian optimisation — Slice 22.

Ports the continuous Thompson-sampling acquisition from
``../trieste/acquisition/function/continuous_thompson_sampling.py``.

The acquisition consumes one posterior-function realisation per
candidate (the ``samples`` field of
:class:`opifex.uncertainty.types.PredictiveDistribution` with leading
sample axis of size 1) and returns the argmin over the candidate set
— the canonical Thompson-sampling decision under a **minimisation**
objective. Flip the sign of ``mean`` and ``samples`` for a
maximisation convention.

References
----------
* Russo, Van Roy, Kazerouni, Osband, Wen 2018 — *A Tutorial on
  Thompson Sampling*, FnT in ML 11(1).
* Hernandez-Lobato+ 2017 — *Parallel and Distributed Thompson
  Sampling for Large-scale Accelerated Exploration*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — runtime use


def continuous_thompson_sampling(
    *,
    predictive: PredictiveDistribution,
) -> jax.Array:
    r"""Return the argmin (over candidates) of a single posterior-function draw.

    Args:
        predictive: Predictive distribution whose ``samples`` field is
            non-``None`` and has shape ``(num_samples, num_candidates)``
            with ``num_samples >= 1``. The first row is the realisation
            used for the Thompson decision.

    Returns:
        Scalar integer index of the chosen candidate.

    Raises:
        ValueError: If ``predictive.samples`` is ``None``.
    """
    if predictive.samples is None:
        raise ValueError(
            "continuous_thompson_sampling requires PredictiveDistribution.samples; "
            "got None. Populate samples with a single MC draw per candidate."
        )
    one_draw = predictive.samples[0]
    return jnp.argmin(one_draw)


__all__ = ["continuous_thompson_sampling"]
