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

References
----------
* Papamakarios, Sterratt, Murray (2019) — Sequential Neural Likelihood,
  ``arXiv:1805.07226``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — eager per opifex convention
from typing import Any

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.configuration.flow_config import (
    ConditionalFlowConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.core.rng import extract_rng_key
from artifex.generative_models.models.flow.conditional import ConditionalRealNVP
from flax import nnx, struct

from opifex.uncertainty.inference_backends.blackjax import BlackJAXBackend
from opifex.uncertainty.sbi.posterior_estimation import (
    _resolve_sbi_backend,
    _trigger_optional_backend_import,
)
from opifex.uncertainty.sbi.simulators import sample_joint, Simulator
from opifex.uncertainty.types import (
    metadata_to_dict,
    MetadataItems,
    PredictiveDistribution,
    require_fitted_state,
)


_DEFAULT_BACKEND_NAME: str = "RealNVP"
_TRAIN_STREAMS: tuple[str, ...] = ("sbi_train", "params", "default")
_SAMPLE_STREAMS: tuple[str, ...] = ("sbi_sample", "sample", "default")


@struct.dataclass(slots=True, kw_only=True)
class NLEState:
    """Fitted-state container for :class:`NeuralLikelihoodEstimator` (pattern (B))."""

    train_losses: jax.Array
    num_simulations: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the fitted state.

        Raises:
            ValueError: When ``train_losses`` is empty / all-NaN.

        """
        if self.train_losses.ndim != 1 or self.train_losses.shape[0] == 0:
            raise ValueError(
                f"train_losses must be 1-d non-empty; got shape={self.train_losses.shape}."
            )
        if bool(jnp.all(jnp.isnan(self.train_losses))):
            raise ValueError("train_losses is entirely NaN — NLE training diverged.")


def _build_conditional_flow(
    *, theta_dim: int, x_dim: int, hidden_dim: int, num_coupling_layers: int, rngs: nnx.Rngs
) -> ConditionalRealNVP:
    coupling = CouplingNetworkConfig(
        name="nle_coupling", hidden_dims=(hidden_dim, hidden_dim), activation="relu"
    )
    cfg = ConditionalFlowConfig(
        name="nle_flow",
        coupling_network=coupling,
        input_dim=x_dim,
        condition_dim=theta_dim,
        num_coupling_layers=num_coupling_layers,
        mask_type="checkerboard",
    )
    return ConditionalRealNVP(cfg, rngs=rngs)


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
        if self.backend != _DEFAULT_BACKEND_NAME:
            _trigger_optional_backend_import(self.backend)

        batch = sample_joint(simulator, num_simulations=num_simulations, rngs=rngs)
        data = batch.data.value
        theta = data["theta"]
        x = data["x"]
        train_key = extract_rng_key(
            rngs, streams=_TRAIN_STREAMS, context="NeuralLikelihoodEstimator.fit"
        )
        flow = _build_conditional_flow(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=self.num_coupling_layers,
            rngs=nnx.Rngs(params=train_key),
        )
        optimizer = nnx.Optimizer(flow, optax.adam(self.learning_rate), wrt=nnx.Param)
        losses = _train_likelihood_flow(
            flow=flow, optimizer=optimizer, theta=theta, x=x, num_steps=self.num_steps
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
            theta_batch = theta[None, :]
            x_batch = observation[None, :]
            log_lik = jnp.squeeze(flow.log_prob(x_batch, condition=theta_batch), axis=0)
            return log_lik + log_prior(theta)

        # Initial state — start at the prior mean (zeros for a centred prior).
        init_state = jnp.zeros((self.theta_dim,))
        sampler = BlackJAXBackend(
            target_log_prob=log_posterior,
            init_state=init_state,
            n_samples=max(num_samples, self.mcmc_samples),
            n_burnin=self.mcmc_burnin,
            method=self.mcmc_method,
            step_size=self.mcmc_step_size,
        )
        # The MCMC backend creates its own RNG stream; forward the user's
        # ``rngs`` so it can pull ``sample``/``default``.
        sample_key = extract_rng_key(
            rngs, streams=_SAMPLE_STREAMS, context="NeuralLikelihoodEstimator.predict_distribution"
        )
        result = sampler.fit(log_posterior, rngs=nnx.Rngs(sample=sample_key))
        samples = jnp.asarray(result.sampler_state)
        # Take the last ``num_samples`` posterior draws as the predictive
        # samples.
        samples = samples[-num_samples:]
        mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            samples=samples,
            metadata=(
                ("method", "nle"),
                ("backend", self.backend),
                ("mcmc_method", self.mcmc_method),
                ("num_samples", num_samples),
            ),
        )


def _train_likelihood_flow(
    *,
    flow: ConditionalRealNVP,
    optimizer: nnx.Optimizer,
    theta: jax.Array,
    x: jax.Array,
    num_steps: int,
) -> jax.Array:
    """Train ``flow`` to maximize ``log q(x | theta)`` for ``num_steps``."""

    @nnx.jit
    def step(model: ConditionalRealNVP, opt: nnx.Optimizer) -> jax.Array:
        def loss_fn(m: ConditionalRealNVP) -> jax.Array:
            return -jnp.mean(m.log_prob(x, condition=theta))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    losses = [step(flow, optimizer) for _ in range(num_steps)]
    return jnp.stack(losses)


__all__ = ["NLEState", "NeuralLikelihoodEstimator"]
