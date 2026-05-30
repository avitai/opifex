r"""Neural Ratio Estimation (NRE) for the SBI subsystem.

NRE trains a binary classifier
:math:`f_\phi(\theta, x) \in \mathbb{R}` to distinguish joint pairs
``(theta, x) ~ p(theta) p(x | theta)`` from marginal pairs
``(theta', x) ~ p(theta) p(x)``. The classifier logits approximate
:math:`\log r(\theta, x) = \log p(x \mid \theta) - \log p(x)`,
which differs from the true log-likelihood only by a constant in
:math:`x`. Posterior MCMC on
``log r(theta, x_obs) + log prior(theta)`` therefore yields posterior
samples without any further normalisation.

MCMC routes through :class:`opifex.uncertainty.inference_backends.BlackJAXBackend`.

The fitted-state container, backend resolution, training loop, and MCMC
predictive block are shared with NPE / NLE via
:mod:`opifex.uncertainty.sbi._base`; NRE keeps only its classifier module,
binary-cross-entropy loss, and log-ratio posterior closures.

References
----------
* Hermans, Begy, Louppe (2020) — Likelihood-free MCMC with Amortized
  Approximate Ratio Estimators, ``arXiv:1903.04057``.
* Durkan, Murray, Papamakarios (2020) — On Contrastive Learning for
  Likelihood-free Inference, ``arXiv:2002.03712``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — eager per opifex convention
from typing import ClassVar

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct

from opifex.uncertainty.sbi._base import (
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
class NREState(_SBIFittedState):
    """Fitted-state container for :class:`NeuralRatioEstimator` (pattern (B))."""

    _method_label: ClassVar[str] = "NRE"


class _RatioClassifier(nnx.Module):
    """Simple MLP classifier ``(theta, x) -> logit`` (log-density ratio)."""

    def __init__(self, theta_dim: int, x_dim: int, hidden_dim: int, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        in_dim = theta_dim + x_dim
        self.linear1 = nnx.Linear(in_features=in_dim, out_features=hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(in_features=hidden_dim, out_features=1, rngs=rngs)

    def __call__(self, theta: jax.Array, x: jax.Array) -> jax.Array:
        h = jnp.concatenate([theta, x], axis=-1)
        h = nnx.relu(self.linear1(h))
        h = nnx.relu(self.linear2(h))
        return jnp.squeeze(self.linear3(h), axis=-1)


@dataclasses.dataclass(slots=True)
class NeuralRatioEstimator:
    """Neural Ratio Estimator with an MLP classifier and BlackJAX MCMC.

    Args:
        theta_dim: Parameter-space dimension.
        x_dim: Observation-space dimension.
        backend: Density-estimator backend name (``"RealNVP"`` by default
            mirrors NPE/NLE for consistent backend-selection semantics).
        num_steps: Number of training steps.
        learning_rate: Adam learning rate.
        hidden_dim: Width of the classifier MLP.
        mcmc_method: BlackJAX sampler family.
        mcmc_samples: Number of posterior samples to draw.
        mcmc_burnin: MCMC warmup samples to discard.
        mcmc_step_size: BlackJAX step size.

    """

    theta_dim: int
    x_dim: int
    backend: str = _DEFAULT_BACKEND_NAME
    num_steps: int = 100
    learning_rate: float = 1e-3
    hidden_dim: int = 32
    mcmc_method: str = "nuts"
    mcmc_samples: int = 200
    mcmc_burnin: int = 50
    mcmc_step_size: float = 0.1

    state: NREState | None = dataclasses.field(default=None, init=False)
    _classifier: _RatioClassifier | None = dataclasses.field(default=None, init=False)

    def __post_init__(self) -> None:
        _resolve_sbi_backend(self.backend)

    def fit(
        self,
        simulator: Simulator,
        num_simulations: int,
        *,
        rngs: nnx.Rngs,
    ) -> NeuralRatioEstimator:
        """Fit the ratio classifier on positives ``(theta, x)`` vs marginals.

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
            rngs, streams=_TRAIN_STREAMS, context="NeuralRatioEstimator.fit"
        )
        perm_key, init_key = jax.random.split(train_key)
        # Marginal pairs: shuffle theta against x to break the joint coupling.
        perm = jax.random.permutation(perm_key, num_simulations)
        theta_neg = theta[perm]

        classifier = _RatioClassifier(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            rngs=nnx.Rngs(params=init_key),
        )
        optimizer = nnx.Optimizer(classifier, optax.adam(self.learning_rate), wrt=nnx.Param)

        def loss_fn(model: _RatioClassifier) -> jax.Array:
            logit_pos = model(theta, x)
            logit_neg = model(theta_neg, x)
            # BCE-with-logits for {pos: 1, neg: 0}.
            loss_pos = jnp.mean(jax.nn.softplus(-logit_pos))
            loss_neg = jnp.mean(jax.nn.softplus(logit_neg))
            return loss_pos + loss_neg

        losses = _train_loop(
            model=classifier, optimizer=optimizer, loss_fn=loss_fn, num_steps=self.num_steps
        )
        self._classifier = classifier
        self.state = NREState(
            train_losses=losses,
            num_simulations=jnp.asarray(num_simulations),
            metadata=(("method", "nre"), ("backend", self.backend)),
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
        """Run MCMC over ``log r(theta, x_obs) + log prior(theta)``."""
        require_fitted_state(self.state, surface="NeuralRatioEstimator.predict_distribution")
        classifier = require_fitted_state(
            self._classifier, surface="NeuralRatioEstimator.predict_distribution"
        )

        def log_posterior(theta: jax.Array) -> jax.Array:
            theta_batch = theta[None, :]
            x_batch = observation[None, :]
            log_ratio = jnp.squeeze(classifier(theta_batch, x_batch), axis=0)
            return log_ratio + log_prior(theta)

        sample_key = extract_rng_key(
            rngs, streams=_SAMPLE_STREAMS, context="NeuralRatioEstimator.predict_distribution"
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
                ("method", "nre"),
                ("backend", self.backend),
                ("mcmc_method", self.mcmc_method),
                ("num_samples", num_samples),
            ),
        )


__all__ = ["NREState", "NeuralRatioEstimator"]
