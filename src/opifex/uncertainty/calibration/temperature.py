"""Temperature scaling — Guo et al. 2017 (arXiv:1706.04599).

A single learnable scalar ``T > 0`` is fit on held-out logits by minimising
multiclass NLL; at predict time the calibrated probabilities are
``softmax(logits / T)``.

``T`` is parameterised through ``log_temperature`` so the optimiser sees an
unconstrained variable and ``T = exp(log_temperature)`` stays strictly
positive. Fitting uses ``optax.lbfgs`` for a few dozen iterations — the
problem is convex in ``log T``.

The fitted state ``TemperatureScalingState`` is a Pattern-B
``flax.struct.dataclass`` (see ``opifex.uncertainty.types`` for the
project's value-object convention) so it traces cleanly under JAX
transforms while keeping ``method`` / ``metadata`` as non-pytree static
aux_data.
"""

from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import struct

from opifex.uncertainty.types import require_fitted_state


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


_MAX_LBFGS_STEPS: int = 64


@struct.dataclass(slots=True, kw_only=True)
class TemperatureScalingState:
    """Fitted state for :class:`TemperatureScaling`.

    The ``temperature`` is a 0-d ``jax.Array`` so it can flow through JAX
    transforms; ``method`` and ``metadata`` are pytree-static aux_data.
    """

    temperature: jax.Array
    method: str = struct.field(pytree_node=False, default="temperature_scaling")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def nll_loss_at_temperature(
    *,
    logits: jax.Array,
    targets: jax.Array,
    log_temperature: jax.Array,
) -> jax.Array:
    """Multiclass NLL of ``softmax(logits / exp(log_temperature))`` against ``targets``.

    Public so callers and tests can pin the fit objective without poking
    into private helpers.
    """
    temperature = jnp.exp(log_temperature)
    scaled = logits / temperature
    log_probs = jax.nn.log_softmax(scaled, axis=-1)
    chosen = jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
    return -jnp.mean(chosen)


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class TemperatureScaling:
    """Temperature-scaling calibrator with explicit fitted state.

    Usage::

        calibrator = TemperatureScaling()
        state = calibrator.fit(logits=val_logits, targets=val_labels)
        probs = calibrator.with_state(state).predict(test_logits)
    """

    max_steps: int = _MAX_LBFGS_STEPS
    _state: TemperatureScalingState | None = dc.field(default=None)

    def with_state(self, state: TemperatureScalingState) -> TemperatureScaling:
        """Return a fresh calibrator carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        logits: jax.Array,
        targets: jax.Array,
    ) -> TemperatureScalingState:
        """Fit a single scalar temperature by minimising multiclass NLL.

        Args:
            logits: ``(batch, num_classes)`` validation logits.
            targets: ``(batch,)`` integer class labels.

        Returns:
            Fitted :class:`TemperatureScalingState`.
        """

        def loss(log_temp: jax.Array) -> jax.Array:
            return nll_loss_at_temperature(logits=logits, targets=targets, log_temperature=log_temp)

        solver = optax.lbfgs()
        log_temp = jnp.asarray(0.0)
        opt_state = solver.init(log_temp)
        value_and_grad = optax.value_and_grad_from_state(loss)
        for _ in range(self.max_steps):
            value, grad = value_and_grad(log_temp, state=opt_state)
            updates, opt_state = solver.update(
                grad, opt_state, log_temp, value=value, grad=grad, value_fn=loss
            )
            log_temp = jnp.asarray(optax.apply_updates(log_temp, updates))
        temperature = jnp.exp(log_temp)
        metadata: MetadataItems = (
            ("method", "temperature_scaling"),
            ("calibration_size", int(logits.shape[0])),
            ("num_classes", int(logits.shape[-1])),
            ("max_steps", int(self.max_steps)),
        )
        return TemperatureScalingState(temperature=temperature, metadata=metadata)

    def predict(self, logits: jax.Array) -> jax.Array:
        """Return ``softmax(logits / state.temperature)`` along the last axis.

        Raises:
            RuntimeError: If called before ``fit`` (or ``with_state``).
        """
        state = require_fitted_state(self._state, surface="TemperatureScaling.predict")
        return _softmax_with_temperature(logits=logits, temperature=state.temperature)


def _softmax_with_temperature(*, logits: jax.Array, temperature: jax.Array) -> jax.Array:
    """Pure helper so jit can trace the predict body without method dispatch."""
    return jax.nn.softmax(logits / temperature, axis=-1)
