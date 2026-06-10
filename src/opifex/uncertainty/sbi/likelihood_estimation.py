r"""Neural Likelihood Estimation (NLE) for the SBI subsystem.

NLE fits a conditional density estimator
:math:`q_\phi(x \mid \theta)` on joint simulation samples
``(theta, x)``. Posterior inference at an observation ``x_obs`` then
proceeds by MCMC on the unnormalised log-posterior

.. math::

    \log p(\theta \mid x_\text{obs}) \propto
        \log q_\phi(x_\text{obs} \mid \theta) + \log p(\theta).

The MCMC step routes through :class:`opifex.uncertainty.inference_backends.BlackJAXBackend`
— there is no direct ``blackjax`` import in :mod:`opifex.uncertainty.sbi`.

The fitted-state container, backend resolution, conditional-flow
construction, training loop, and MCMC predictive block are shared with
NPE / NRE via :mod:`opifex.uncertainty.sbi._base`; NLE keeps only its
likelihood-flow loss and log-posterior closures.

References
----------
* Papamakarios, Sterratt, Murray (2019) — Sequential Neural Likelihood,
  ``arXiv:1805.07226``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — eager per opifex convention
from typing import ClassVar

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.rng import extract_rng_key
from artifex.generative_models.models.flow.conditional import (  # noqa: TC002 — pyproject dep, kept eager
    ConditionalRealNVP,
)
from flax import nnx, struct

from opifex.uncertainty.sbi._base import (
    _build_conditional_flow,
    _DEFAULT_BACKEND_NAME,
    _mcmc_posterior_predictive,
    _resolve_sbi_backend,
    _SAMPLE_STREAMS,
    _SBIFittedState,
    _simulate_training_data,
    _train_loop,
    _TRAIN_STREAMS,
)
from opifex.uncertainty.sbi.simulators import Simulator  # noqa: TC001 — eager per opifex convention
from opifex.uncertainty.types import (
    PredictiveDistribution,
    require_fitted_state,
)


@struct.dataclass(slots=True, kw_only=True)
class NLEState(_SBIFittedState):
    """Fitted-state container for :class:`NeuralLikelihoodEstimator` (pattern (B))."""

    _method_label: ClassVar[str] = "NLE"


@dataclasses.dataclass(slots=True)
class NeuralLikelihoodEstimator:
    """Neural Likelihood Estimator with a conditional-flow likelihood and BlackJAX MCMC.

    Args:
        theta_dim: Parameter-space dimension.
        x_dim: Observation-space dimension.
        backend: Density-estimator backend name (default ``"RealNVP"``).
        num_steps: Number of training steps (full-batch Adam).
        learning_rate: Adam learning rate.
        hidden_dim: Width of the coupling MLP hidden layers.
        num_coupling_layers: Coupling-layer depth.
        mcmc_method: BlackJAX sampler family (``"nuts"`` / ``"hmc"`` / ``"mala"``).
        mcmc_samples: Number of posterior MCMC samples to draw at predict time.
        mcmc_burnin: Number of warmup MCMC samples to discard.
        mcmc_step_size: BlackJAX step size; tuned to a small value for
            low-dim toys where NUTS adaptation can still mix well.

    """

    theta_dim: int
    x_dim: int
    backend: str = _DEFAULT_BACKEND_NAME
    num_steps: int = 100
    learning_rate: float = 1e-3
    hidden_dim: int = 32
    num_coupling_layers: int = 4
    mcmc_method: str = "nuts"
    mcmc_samples: int = 200
    mcmc_burnin: int = 50
    mcmc_step_size: float = 0.1

    state: NLEState | None = dataclasses.field(default=None, init=False)
    _flow: ConditionalRealNVP | None = dataclasses.field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate the configured MCMC backend at construction time."""
        _resolve_sbi_backend(self.backend)

    def fit(
        self,
        simulator: Simulator,
        num_simulations: int,
        *,
        rngs: nnx.Rngs,
    ) -> NeuralLikelihoodEstimator:
        """Fit ``q(x | theta)`` on ``num_simulations`` joint draws.

        Raises:
            ImportError: When the requested optional backend is not installed.

        """
        theta, x = _simulate_training_data(
            simulator=simulator,
            num_simulations=num_simulations,
            backend=self.backend,
            rngs=rngs,
        )
        train_key = extract_rng_key(
            rngs, streams=_TRAIN_STREAMS, context="NeuralLikelihoodEstimator.fit"
        )
        # NLE models ``q(x | theta)`` — the flow's input is ``x`` and the
        # conditioning variable is ``theta`` (the inverse of NPE's roles).
        flow = _build_conditional_flow(
            name="nle",
            input_dim=self.x_dim,
            condition_dim=self.theta_dim,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=self.num_coupling_layers,
            rngs=nnx.Rngs(params=train_key),
        )
        optimizer = nnx.Optimizer(flow, optax.adam(self.learning_rate), wrt=nnx.Param)

        def loss_fn(model: ConditionalRealNVP) -> jax.Array:
            """Negative mean log-likelihood of ``x`` under the flow conditioned on ``theta``."""
            return -jnp.mean(model.log_prob(x, condition=theta))

        losses = _train_loop(
            model=flow, optimizer=optimizer, loss_fn=loss_fn, num_steps=self.num_steps
        )
        self._flow = flow
        self.state = NLEState(
            train_losses=losses,
            num_simulations=jnp.asarray(num_simulations),
            metadata=(("method", "nle"), ("backend", self.backend)),
        )
        return self

    def predict_distribution(
        self,
        observation: jax.Array,
        *,
        rngs: nnx.Rngs,
        num_samples: int,
        log_prior: Callable[[jax.Array], jax.Array],
    ) -> PredictiveDistribution:
        """Run posterior MCMC at ``observation`` and return a :class:`PredictiveDistribution`."""
        require_fitted_state(self.state, surface="NeuralLikelihoodEstimator.predict_distribution")
        flow = require_fitted_state(
            self._flow, surface="NeuralLikelihoodEstimator.predict_distribution"
        )

        def log_posterior(theta: jax.Array) -> jax.Array:
            """Return the unnormalised log-posterior ``log lik(x_obs|theta) + log prior``."""
            theta_batch = theta[None, :]
            x_batch = observation[None, :]
            log_lik = jnp.squeeze(flow.log_prob(x_batch, condition=theta_batch), axis=0)
            return log_lik + log_prior(theta)

        sample_key = extract_rng_key(
            rngs, streams=_SAMPLE_STREAMS, context="NeuralLikelihoodEstimator.predict_distribution"
        )
        return _mcmc_posterior_predictive(
            log_posterior=log_posterior,
            theta_dim=self.theta_dim,
            num_samples=num_samples,
            mcmc_samples=self.mcmc_samples,
            mcmc_burnin=self.mcmc_burnin,
            mcmc_method=self.mcmc_method,
            mcmc_step_size=self.mcmc_step_size,
            sample_key=sample_key,
            metadata=(
                ("method", "nle"),
                ("backend", self.backend),
                ("mcmc_method", self.mcmc_method),
                ("num_samples", num_samples),
            ),
        )


__all__ = ["NLEState", "NeuralLikelihoodEstimator"]
