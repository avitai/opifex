"""Simulator container and joint-sampling helper for the SBI subsystem.

The :class:`Simulator` is a pattern (A) value object — plain
``@dataclass(frozen=True, slots=True, kw_only=True)``. It holds only
callables (prior sampler, simulator, optional summary) and a hashable
metadata tuple. It carries no JAX arrays, so it is safe to pass as a
*static* argument to ``jit`` -compiled training and sampling loops.

The :func:`sample_joint` helper realises a Datarax ``Batch`` whose
elements pair a parameter sample ``theta`` with its observation ``x``:

.. code-block:: text

    Batch( Element(data={"theta": (d_theta,), "x": (d_x,)}), ... )

This matches the plan's contract that simulator output containers expose
``Batch[Element]`` with ``(theta, x)`` element pairs (datarax.typing.Element
and datarax.typing.Batch).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from datarax.core.element_batch import Batch, Element
from flax import nnx  # noqa: TC002 — eager per opifex convention

from opifex.uncertainty.types import metadata_to_dict, MetadataItems


# Type aliases for clarity.
PriorSampler = Callable[[jax.Array, int], jax.Array]
SimulateFn = Callable[[jax.Array, jax.Array], jax.Array]
SummaryFn = Callable[[jax.Array], jax.Array]

_SIMULATE_STREAMS: tuple[str, ...] = ("sbi_simulate", "sample", "default")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Simulator:
    """Static SBI simulator description (pattern (A)).

    Args:
        prior_sampler: ``(rng_key, num_samples) -> theta`` returning an
            array of shape ``(num_samples, *theta_event_shape)``.
        simulate_fn: ``(rng_key, theta) -> x`` returning observations of
            shape ``(num_samples, *x_event_shape)`` whose leading axis
            matches the leading axis of ``theta``.
        summary_fn: Optional ``x -> s(x)`` mapping applied per-element
            after simulation. Used for summary-statistic compression
            (e.g., neural compression to a low-dim s(x)).
        metadata: Immutable ``tuple[tuple[str, Any], ...]`` of static
            metadata. Hashable so callers can pass the simulator through
            ``jit``'s static-argnums without losing the cache.

    The dataclass is intentionally minimal — array state belongs in the
    estimator state container (pattern (B)), not here.

    """

    prior_sampler: PriorSampler
    simulate_fn: SimulateFn
    summary_fn: SummaryFn | None = None
    metadata: MetadataItems = ()

    def metadata_dict(self) -> dict[str, Any]:
        """Return a mutable ``dict`` view of the immutable metadata."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the simulator's contract.

        Public method — never called from ``__post_init__`` or the pytree
        path (GUIDE_ALIGNMENT item 7).

        Raises:
            TypeError: When ``prior_sampler`` / ``simulate_fn`` / ``summary_fn``
                are not callable, or ``metadata`` is not a tuple-of-pairs.

        """
        if not callable(self.prior_sampler):
            raise TypeError(
                f"prior_sampler must be callable; got {type(self.prior_sampler).__name__}."
            )
        if not callable(self.simulate_fn):
            raise TypeError(f"simulate_fn must be callable; got {type(self.simulate_fn).__name__}.")
        if self.summary_fn is not None and not callable(self.summary_fn):
            raise TypeError(
                f"summary_fn must be callable or None; got {type(self.summary_fn).__name__}."
            )
        if not isinstance(self.metadata, tuple):
            raise TypeError(
                "metadata must be a tuple of (str, Any) pairs for static aux_data; "
                f"got {type(self.metadata).__name__}."
            )


def sample_joint(
    simulator: Simulator,
    *,
    num_simulations: int,
    rngs: nnx.Rngs,
) -> Batch:
    """Sample ``num_simulations`` joint draws from the prior and simulator.

    Returns a Datarax ``Batch`` where every element pairs ``theta`` with
    its observation ``x`` (and optionally a compressed summary).

    The RNG resolution uses Artifex's :func:`extract_rng_key` with the
    canonical SBI stream name ``"sbi_simulate"``.

    Args:
        simulator: Static simulator description.
        num_simulations: Number of ``(theta, x)`` pairs to draw.
        rngs: Caller-owned ``nnx.Rngs`` carrying the ``sbi_simulate``
            named stream.

    Raises:
        ValueError: When ``num_simulations`` is non-positive, when the
            prior sampler returns a malformed leading axis, or when the
            simulator output's leading axis disagrees with the parameters.

    """
    if num_simulations <= 0:
        raise ValueError(f"num_simulations must be positive; got {num_simulations}.")

    sim_key = extract_rng_key(
        rngs,
        streams=_SIMULATE_STREAMS,
        context="Simulator.sample_joint",
    )
    prior_key, sim_step_key = jax.random.split(sim_key)
    theta = simulator.prior_sampler(prior_key, num_simulations)
    if not hasattr(theta, "shape") or theta.ndim < 1 or theta.shape[0] != num_simulations:
        raise ValueError(
            "prior_sampler must return an array of shape "
            f"(num_simulations, *theta_event_shape); got shape "
            f"{getattr(theta, 'shape', '<no shape>')!r} for num_simulations={num_simulations}."
        )

    x = simulator.simulate_fn(sim_step_key, theta)
    if not hasattr(x, "shape") or x.ndim < 1 or x.shape[0] != num_simulations:
        raise ValueError(
            "simulate_fn output leading axis must match prior sample count "
            f"({num_simulations}); got shape {getattr(x, 'shape', '<no shape>')!r}."
        )

    if simulator.summary_fn is not None:
        x = simulator.summary_fn(x)
        if x.shape[0] != num_simulations:
            raise ValueError(
                "summary_fn must preserve the leading batch axis; "
                f"got shape {x.shape!r} for num_simulations={num_simulations}."
            )

    elements = [
        Element(data={"theta": jnp.asarray(theta[i]), "x": jnp.asarray(x[i])})
        for i in range(num_simulations)
    ]
    return Batch(elements, validate=False)


__all__ = [
    "PriorSampler",
    "SimulateFn",
    "Simulator",
    "SummaryFn",
    "sample_joint",
]
