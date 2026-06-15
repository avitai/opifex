r"""Trust-region Bayesian optimisation rules — Slice 22 (audit finding #4a).

Ports the trust-region BO family from
``../trieste/acquisition/rule.py:1863,1923,2038``:

* :class:`TrustRegionBox` — axis-aligned bounding box clipped to the
  global search space; pure geometry container.
* :class:`TREGOBox` — Wan+ 2021 *Think Global and Act Local* trust
  region with success / failure streak counters driving an
  exponential length update.
* :class:`TURBOBox` — Eriksson+ 2019 TuRBO trust region; recenters on
  the best-observed point each round and applies the same
  expand-on-success / shrink-on-failure update.
* :class:`BatchTrustRegionBox` — ``M`` independent trust regions for
  parallel BO (one box per parallel worker, per Eriksson+ 2019 §3).

Pattern-A frozen-slotted-kw-only dataclasses throughout; no equinox
dependency; pure JAX arrays for the centre / length / bounds.

References
----------
* Eriksson, Pearce, Gardner, Turner, Poloczek 2019 — *Scalable Global
  Optimization via Local Bayesian Optimization (TuRBO)*, NeurIPS.
* Wan, Diouane, Lambin, Mockus 2021 — *Think Global and Act Local
  (TREGO)*, NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True, kw_only=True)
class TrustRegionBox:
    """Axis-aligned trust-region box clipped to the global search space.

    Attributes:
        center: Centre of the trust region, shape ``(d,)``.
        length: Trust-region edge length (scalar). Each axis is
            constrained to ``[center - length/2, center + length/2]``
            clipped to ``[search_space_lower, search_space_upper]``.
        search_space_lower: Global lower bounds, shape ``(d,)``.
        search_space_upper: Global upper bounds, shape ``(d,)``.
    """

    center: jax.Array
    length: jax.Array
    search_space_lower: jax.Array
    search_space_upper: jax.Array

    def bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(lower, upper)`` of the trust-region box clipped to the search space."""
        half = 0.5 * self.length
        lower = jnp.maximum(self.center - half, self.search_space_lower)
        upper = jnp.minimum(self.center + half, self.search_space_upper)
        return lower, upper


@dataclass(frozen=True, slots=True, kw_only=True)
class TREGOBox:
    """TREGO trust region (Wan+ 2021) with success / failure streak counters.

    Attributes:
        center: Trust-region centre.
        length: Trust-region edge length (scalar).
        search_space_lower / search_space_upper: Global bounds.
        success_count: Consecutive successful rounds since the last
            length update.
        failure_count: Consecutive failed rounds since the last length
            update.
        success_threshold: Counter value that triggers an expand.
        failure_threshold: Counter value that triggers a shrink.
        shrink_factor: Multiplier applied to ``length`` on shrink.
        expand_factor: Multiplier applied to ``length`` on expand.
    """

    center: jax.Array
    length: jax.Array
    search_space_lower: jax.Array
    search_space_upper: jax.Array
    success_count: int
    failure_count: int
    success_threshold: int
    failure_threshold: int
    shrink_factor: float
    expand_factor: float

    def bounds(self) -> tuple[jax.Array, jax.Array]:
        """Same as :meth:`TrustRegionBox.bounds`."""
        half = 0.5 * self.length
        lower = jnp.maximum(self.center - half, self.search_space_lower)
        upper = jnp.minimum(self.center + half, self.search_space_upper)
        return lower, upper

    def register_round(self, *, was_success: bool) -> TREGOBox:
        """Functional update — increment counters and trigger length update if saturated."""
        if was_success:
            new_success = self.success_count + 1
            new_failure = 0
            if new_success >= self.success_threshold:
                return replace(
                    self,
                    length=self.length * self.expand_factor,
                    success_count=0,
                    failure_count=0,
                )
            return replace(self, success_count=new_success, failure_count=new_failure)
        new_failure = self.failure_count + 1
        new_success = 0
        if new_failure >= self.failure_threshold:
            return replace(
                self,
                length=self.length * self.shrink_factor,
                success_count=0,
                failure_count=0,
            )
        return replace(self, success_count=new_success, failure_count=new_failure)


@dataclass(frozen=True, slots=True, kw_only=True)
class TURBOBox:
    """TuRBO trust region (Eriksson+ 2019). Recenters on the best observation each round."""

    center: jax.Array
    length: jax.Array
    search_space_lower: jax.Array
    search_space_upper: jax.Array

    def bounds(self) -> tuple[jax.Array, jax.Array]:
        """Same as :meth:`TrustRegionBox.bounds`."""
        half = 0.5 * self.length
        lower = jnp.maximum(self.center - half, self.search_space_lower)
        upper = jnp.minimum(self.center + half, self.search_space_upper)
        return lower, upper

    def recenter(self, *, new_best: jax.Array) -> TURBOBox:
        """Return a new TuRBO box centred on ``new_best``; length unchanged."""
        return replace(self, center=new_best)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchTrustRegionBox:
    """``M`` parallel trust regions for batched / asynchronous BO (Eriksson+ 2019 §3)."""

    regions: tuple[TrustRegionBox, ...]

    def stacked_bounds(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(lowers, uppers)`` of shape ``(M, d)``."""
        if len(self.regions) == 0:
            raise ValueError("BatchTrustRegionBox requires at least one region.")
        lower_list = []
        upper_list = []
        for region in self.regions:
            lo, hi = region.bounds()
            lower_list.append(lo)
            upper_list.append(hi)
        return jnp.stack(lower_list, axis=0), jnp.stack(upper_list, axis=0)


__all__ = [
    "BatchTrustRegionBox",
    "TREGOBox",
    "TURBOBox",
    "TrustRegionBox",
]
