"""Uncertainty Quantification Neural Operator (UQNO).

A Bayesian Fourier Neural Operator that exposes the shared opifex UQ
surface:

* :meth:`UncertaintyQuantificationNeuralOperator.predict_distribution`
  returns a :class:`PredictiveDistribution` populated from Monte-Carlo
  posterior samples; metadata advertises spatial axes so function-space
  callers can identify the operator output.
* :meth:`UncertaintyQuantificationNeuralOperator.loss_components` returns
  a :class:`UQLossComponents` built from
  :class:`ObjectiveConfig`-driven weights, with the aggregated
  Bayesian-layer KL in the ``kl`` slot.
* :meth:`UncertaintyQuantificationNeuralOperator.negative_elbo` mirrors
  the Phase 3 ``ProbabilisticPINN`` convention, populating the
  ``negative_elbo`` slot with the weighted total.

All stochastic methods take caller-owned ``nnx.Rngs`` at the boundary;
the operator owns no hidden RNG state. Bayesian Fourier and linear
parameters use the canonical shared layers from
:mod:`opifex.uncertainty.layers.bayesian`.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx

from opifex.uncertainty.layers.bayesian import (
    BayesianLinear,
    BayesianSpectralConvolution,
)
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.types import PredictiveDistribution, PredictiveMode


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


__all__ = [
    "BayesianLinear",
    "BayesianSpectralConvolution",
    "UQNOLayer",
    "UncertaintyQuantificationNeuralOperator",
    "create_bayesian_inverse_uqno",
    "create_robust_design_uqno",
    "create_safety_critical_uqno",
]

logger = logging.getLogger(__name__)

_UQNO_RNG_STREAMS = ("sample", "posterior", "default")


def _coerce_predictive_mode(mode: PredictiveMode | str) -> PredictiveMode:
    if isinstance(mode, PredictiveMode):
        return mode
    return PredictiveMode(mode)


class UQNOLayer(nnx.Module):
    """Single UQNO block: Bayesian spectral conv + 1×1 local conv + skip.

    The block stays in NCHW (channels-first) layout so the FFT path can
    operate on the spatial axes directly. Local conv is a non-Bayesian
    pointwise nnx.Conv mirroring the canonical Li FNO block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        use_skip_connection: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip_connection = use_skip_connection

        self.spectral_conv = BayesianSpectralConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=tuple(modes),
            rngs=rngs,
        )
        self.local_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        if use_skip_connection and in_channels != out_channels:
            self.channel_proj: nnx.Conv | None = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                padding="SAME",
                rngs=rngs,
            )
        else:
            self.channel_proj = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        """Apply the UQNO block.

        ``x`` is NCHW. ``deterministic`` mirrors the shared
        ``BayesianSpectralConvolution`` convention: when ``True``, the
        spectral-conv weights collapse to their posterior mean and
        ``rngs`` is ignored. Otherwise ``rngs`` MUST be supplied.
        """
        x_spec = self.spectral_conv(x, deterministic=deterministic, rngs=rngs)

        x_channels_last = x.transpose(0, 2, 3, 1)
        conv_out = self.local_conv(x_channels_last)
        x_local = conv_out.transpose(0, 3, 1, 2)

        if self.use_skip_connection:
            if self.channel_proj is not None:
                proj_out = self.channel_proj(x.transpose(0, 2, 3, 1))
                x_skip = proj_out.transpose(0, 3, 1, 2)
            else:
                x_skip = x
        else:
            x_skip = jnp.zeros_like(x_spec)

        return nnx.gelu(x_spec + x_local + x_skip)

    def kl_divergence(self) -> jax.Array:
        """KL contribution of the Bayesian spectral conv inside this block."""
        return self.spectral_conv.kl_divergence()


class UncertaintyQuantificationNeuralOperator(nnx.Module):
    """Bayesian FNO with the shared opifex UQ surface.

    Forward layout is channels-last on the boundary and channels-first
    through the spectral stack:

    * Input ``x``: ``(batch, height, width, in_channels)`` — channels-last.
    * Internal: lifts via Bayesian linear, transposes to NCHW, runs the
      spectral stack, transposes back, projects via Bayesian linear.
    * Output mean: ``(batch, height, width, out_channels)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        modes: Sequence[int] = (16, 16),
        num_layers: int = 4,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.lifting = BayesianLinear(in_channels, hidden_channels, rngs=rngs)
        self.uqno_layers = nnx.List(
            [
                UQNOLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    modes=modes,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )
        self.projection = BayesianLinear(hidden_channels, out_channels, rngs=rngs)

    def _forward(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        """Single forward pass; channels-last in/out, channels-first internal."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor (batch, H, W, C); got {x.ndim}D")

        x = self.lifting(x, deterministic=deterministic, rngs=rngs)
        x = x.transpose(0, 3, 1, 2)
        for layer in self.uqno_layers:
            x = layer(x, deterministic=deterministic, rngs=rngs)
        x = x.transpose(0, 2, 3, 1)
        return self.projection(x, deterministic=deterministic, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass.

        ``deterministic`` mirrors the shared Bayesian-layer convention: when
        ``True`` every Bayesian layer collapses to its posterior mean and
        ``rngs`` is ignored. Otherwise ``rngs`` MUST be supplied so the
        reparameterization-trick samples have caller-owned keys.
        """
        is_deterministic = deterministic if deterministic is not None else False
        return self._forward(x, rngs=rngs, deterministic=is_deterministic)

    def kl_divergence(self) -> jax.Array:
        """Aggregate KL across every shared Bayesian layer in the operator."""
        total = self.lifting.kl_divergence() + self.projection.kl_divergence()
        for layer in self.uqno_layers:
            total = total + layer.kl_divergence()
        return total

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        num_samples: int = 10,
        mode: PredictiveMode | str = PredictiveMode.PREDICTIVE,
    ) -> PredictiveDistribution:
        """Return a :class:`PredictiveDistribution` from MC posterior samples.

        Metadata advertises the spatial axes of the input so function-space
        consumers can dispatch on operator-style outputs. The MC loop runs
        under ``jax.lax.scan`` so a single jit-trace covers every sample
        and compile cost is independent of ``num_samples``.
        """
        coerced_mode = _coerce_predictive_mode(mode)
        key = extract_rng_key(
            rngs,
            streams=_UQNO_RNG_STREAMS,
            context="UQNO.predict_distribution",
        )
        sample_keys = jax.random.split(key, num_samples)

        def _draw(_carry: None, sample_key: jax.Array) -> tuple[None, jax.Array]:
            pred = self._forward(x, rngs=nnx.Rngs(sample=sample_key), deterministic=False)
            return None, pred

        _, samples = jax.lax.scan(_draw, None, sample_keys)
        mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        quantiles = {
            0.025: jnp.quantile(samples, 0.025, axis=0),
            0.5: jnp.quantile(samples, 0.5, axis=0),
            0.975: jnp.quantile(samples, 0.975, axis=0),
        }
        return PredictiveDistribution(
            mean=mean,
            samples=samples,
            variance=variance,
            epistemic=variance,
            quantiles=quantiles,
            metadata=(
                ("method", coerced_mode.value),
                ("num_samples", int(num_samples)),
                ("spatial_axes", (1, 2)),
                ("source", "uqno"),
            ),
        )

    def loss_components(
        self,
        batch: Mapping[str, Any],
        *,
        rngs: nnx.Rngs,
        objective: ObjectiveConfig,
    ) -> UQLossComponents:
        """Compute UQ loss components on a supervised batch.

        Required batch fields: ``x``, ``y``. ``x`` is channels-last
        ``(batch, H, W, in_channels)``; ``y`` matches the projection
        output shape ``(batch, H, W, out_channels)``.
        """
        missing = [field for field in ("x", "y") if field not in batch]
        if missing:
            raise ValueError(f"batch missing required field(s): {missing!r}")
        x = batch["x"]
        y = batch["y"]

        y_pred = self._forward(x, rngs=rngs, deterministic=False)
        data = jnp.mean((y_pred - y) ** 2)
        kl = self.kl_divergence()

        return UQLossComponents.from_components(
            config=objective,
            data=data,
            kl=kl,
            metadata=(("source", "uqno"),),
        )

    def negative_elbo(
        self,
        batch: Mapping[str, Any],
        *,
        rngs: nnx.Rngs,
        objective: ObjectiveConfig,
    ) -> UQLossComponents:
        """Populate the ``negative_elbo`` slot with the weighted total.

        ``total`` is unchanged; ``negative_elbo`` is set to ``total`` so
        downstream code can read ``components.negative_elbo`` without
        recomputing the weighted sum (matches the Task 3.2 PINN convention).
        """
        import dataclasses

        base = self.loss_components(batch, rngs=rngs, objective=objective)
        return dataclasses.replace(base, negative_elbo=base.total)


def create_safety_critical_uqno(
    in_channels: int, out_channels: int, *, rngs: nnx.Rngs
) -> UncertaintyQuantificationNeuralOperator:
    """UQNO sized for safety-critical applications (wide + deep stack)."""
    return UncertaintyQuantificationNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        modes=(32, 32),
        num_layers=6,
        rngs=rngs,
    )


def create_robust_design_uqno(
    in_channels: int, out_channels: int, *, rngs: nnx.Rngs
) -> UncertaintyQuantificationNeuralOperator:
    """UQNO sized for robust engineering design (mid-capacity stack)."""
    return UncertaintyQuantificationNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=96,
        modes=(24, 24),
        num_layers=5,
        rngs=rngs,
    )


def create_bayesian_inverse_uqno(
    in_channels: int, out_channels: int, *, rngs: nnx.Rngs
) -> UncertaintyQuantificationNeuralOperator:
    """UQNO sized for Bayesian inverse problems (more samples downstream)."""
    return UncertaintyQuantificationNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        modes=(16, 16),
        num_layers=4,
        rngs=rngs,
    )
