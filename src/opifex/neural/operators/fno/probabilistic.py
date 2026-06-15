r"""Probabilistic Fourier Neural Operator (PNO) — heteroscedastic-Gaussian heads.

Equips a :class:`FourierNeuralOperator` backbone with twin pointwise
output heads — a *mean* head and a *log-variance* head — over the
operator output, producing a calibrated per-location
heteroscedastic-Gaussian
:class:`~opifex.uncertainty.types.PredictiveDistribution`:

.. math::

    y(x) \\mid \\theta &\\sim \\mathcal{N}\\bigl(\\mu_{\\theta}(x),\\,
        \\sigma_{\\theta}^{2}(x)\\bigr), \\\\
    -\\log p(y \\mid x) &= \\tfrac{1}{2}\\!\\left[\\log(2\\pi)
        + \\log \\sigma_{\\theta}^{2}(x)
        + \\frac{(y - \\mu_{\\theta}(x))^{2}}{\\sigma_{\\theta}^{2}(x)}\\right].

The negative log-likelihood is the canonical heteroscedastic-Gaussian
aleatoric loss of **Kendall & Gal 2017** (NeurIPS, arXiv:1703.04977,
§3.1) ported to operator-valued outputs. The mean head and log-variance
head are simple pointwise ``nnx.Linear`` projections from the shared
FNO hidden representation; the log-variance head is optionally clipped
to ``[log_variance_floor, log_variance_ceiling]`` to keep training
numerically stable on noisy targets (the floor prevents
``σ² → 0`` and the ceiling caps gradient magnitude).

This module supplies the *aleatoric* axis of the Phase 10 PNO surface;
epistemic uncertainty is provided orthogonally by wrapping a fitted
PNO with the existing :class:`LaplaceAdapterSpec` from
:mod:`opifex.uncertainty.curvature` or with a deep-ensemble adapter
(:class:`FNODeepEnsembleAdapterSpec`).

References
----------
* Kendall, A., Gal, Y. 2017 — *What Uncertainties Do We Need in
  Bayesian Deep Learning for Computer Vision?*, NeurIPS,
  arXiv:1703.04977 (PRIMARY — canonical heteroscedastic-Gaussian
  aleatoric formulation).
* Magnani, E. et al. 2024 — function-uncertainty neural operator
  thread (companion to the LUNO predictive of arXiv:2406.04317;
  treats trained neural operators as probabilistic surrogates).
* Li, Z. et al. 2021 — *Fourier Neural Operator for Parametric
  Partial Differential Equations*, arXiv:2010.08895 (the FNO backbone
  this module wraps).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.likelihoods import heteroscedastic_gaussian_log_likelihood
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_PNO_SOURCE_PACKAGE = "opifex.neural.operators.fno.probabilistic"
_PNO_METHOD = "heteroscedastic_gaussian"


class ProbabilisticFourierNeuralOperator(nnx.Module):
    """FNO backbone with mean + log-variance heads (heteroscedastic Gaussian).

    The backbone is a standard :class:`FourierNeuralOperator` configured
    to emit a shared hidden representation; two pointwise
    :class:`nnx.Linear` heads then produce ``mean`` and ``log_variance``
    of the per-location predictive distribution. The pair
    ``(mean, exp(log_variance))`` is the heteroscedastic-Gaussian
    parameterisation of Kendall & Gal 2017 §3.1.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels (the heads emit
            this many ``mean`` and ``log_variance`` values per spatial
            location).
        hidden_channels: Channel width of the FNO backbone.
        modes: Number of Fourier modes retained per spatial axis.
        num_layers: FNO layer depth.
        activation: Activation between Fourier layers.
        spatial_dims: Number of spatial dimensions (1, 2, or 3).
        log_variance_floor: Lower bound on the log-variance head
            output. Defaults to ``-10.0`` (≈ ``σ² ≥ 4.5e-5``); the
            head is clipped during ``__call__``. Pass ``-jnp.inf`` to
            disable.
        log_variance_ceiling: Upper bound on the log-variance head
            output. Defaults to ``10.0`` (≈ ``σ² ≤ 2.2e+4``).
        rngs: Caller-owned ``nnx.Rngs`` for parameter initialisation.

    Raises:
        ValueError: If ``log_variance_floor >= log_variance_ceiling``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: int,
        num_layers: int,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        spatial_dims: int = 2,
        log_variance_floor: float = -10.0,
        log_variance_ceiling: float = 10.0,
        rngs: nnx.Rngs,
    ) -> None:
        if log_variance_floor >= log_variance_ceiling:
            raise ValueError(
                "log_variance_floor must be strictly below log_variance_ceiling; "
                f"got floor={log_variance_floor!r}, ceiling={log_variance_ceiling!r}."
            )
        super().__init__()
        self.out_channels = out_channels
        self.log_variance_floor = log_variance_floor
        self.log_variance_ceiling = log_variance_ceiling

        # Shared FNO backbone emitting a hidden representation; the
        # heads are responsible for mapping to mean / log-variance.
        self.backbone = FourierNeuralOperator(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            activation=activation,
            spatial_dims=spatial_dims,
            rngs=rngs,
        )
        self.mean_head = nnx.Linear(
            in_features=hidden_channels,
            out_features=out_channels,
            rngs=rngs,
        )
        self.log_variance_head = nnx.Linear(
            in_features=hidden_channels,
            out_features=out_channels,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(mean, log_variance)`` over the input tensor ``x``.

        The heads run pointwise over the spatial axes by transposing
        the channel axis to last position, applying the
        :class:`nnx.Linear` head, and transposing back. The
        log-variance is clipped to
        ``[log_variance_floor, log_variance_ceiling]``.
        """
        hidden = self.backbone(x)
        mean = _apply_pointwise(hidden, self.mean_head)
        log_variance = _apply_pointwise(hidden, self.log_variance_head)
        log_variance = jnp.clip(log_variance, self.log_variance_floor, self.log_variance_ceiling)
        return mean, log_variance

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        """Return the heteroscedastic-Gaussian predictive distribution."""
        mean, log_variance = self(x)
        variance = jnp.exp(log_variance)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            aleatoric=variance,
            total_uncertainty=variance,
            metadata=compose_method_metadata(
                method=_PNO_METHOD,
                source_package=_PNO_SOURCE_PACKAGE,
                extra=(
                    ("estimator", "probabilistic_fno"),
                    ("default_strategy", DefaultStrategy.VARIATIONAL.value),
                    ("paper", "Kendall & Gal 2017 arXiv:1703.04977"),
                ),
            ),
        )


def probabilistic_fno_negative_log_likelihood(
    model: ProbabilisticFourierNeuralOperator,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    r"""Mean heteroscedastic-Gaussian NLL over the spatial-channel tensor.

    Args:
        model: A :class:`ProbabilisticFourierNeuralOperator`.
        x: Input batch.
        y: Ground-truth targets with the same shape as the predictive mean.

    Returns:
        Scalar mean negative log-likelihood
        ``- mean_i log N(y_i; μ_i, σ_i^2)``.
    """
    mean, log_variance = model(x)
    scale = jnp.exp(0.5 * log_variance)
    log_likelihood = heteroscedastic_gaussian_log_likelihood(y, mean=mean, scale=scale)
    return -jnp.mean(log_likelihood)


def _apply_pointwise(x: jax.Array, head: nnx.Linear) -> jax.Array:
    """Apply ``head`` pointwise across the spatial axes of ``(batch, C, *S)``.

    ``nnx.Linear`` expects the input feature axis to be the trailing
    axis, so the channel axis is rotated to the end, the head runs,
    and the result is rotated back to ``(batch, out_channels, *S)``.
    """
    moved = jnp.moveaxis(x, 1, -1)
    transformed = head(moved)
    return jnp.moveaxis(transformed, -1, 1)


__all__ = [
    "ProbabilisticFourierNeuralOperator",
    "probabilistic_fno_negative_log_likelihood",
]
