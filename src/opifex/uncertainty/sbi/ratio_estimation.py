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
from typing import Any

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.rng import extract_rng_key
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
class NREState:
    """Fitted-state container for :class:`NeuralRatioEstimator` (pattern (B))."""

    train_losses: jax.Array
    num_simulations: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the fitted state."""
        if self.train_losses.ndim != 1 or self.train_losses.shape[0] == 0:
            raise ValueError(
                f"train_losses must be 1-d non-empty; got shape={self.train_losses.shape}."
            )
        if bool(jnp.all(jnp.isnan(self.train_losses))):
            raise ValueError("train_losses is entirely NaN — NRE training diverged.")


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
        if self.backend != _DEFAULT_BACKEND_NAME:
            _trigger_optional_backend_import(self.backend)

        batch = sample_joint(simulator, num_simulations=num_simulations, rngs=rngs)
        data = batch.data.value
        theta = data["theta"]
        x = data["x"]
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
        losses = _train_ratio_classifier(
            classifier=classifier,
            optimizer=optimizer,
            theta_pos=theta,
            x_pos=x,
            theta_neg=theta_neg,
            x_neg=x,
            num_steps=self.num_steps,
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

        init_state = jnp.zeros((self.theta_dim,))
        sampler = BlackJAXBackend(
            target_log_prob=log_posterior,
            init_state=init_state,
            n_samples=max(num_samples, self.mcmc_samples),
            n_burnin=self.mcmc_burnin,
            method=self.mcmc_method,
            step_size=self.mcmc_step_size,
        )
        sample_key = extract_rng_key(
            rngs, streams=_SAMPLE_STREAMS, context="NeuralRatioEstimator.predict_distribution"
        )
        result = sampler.fit(log_posterior, rngs=nnx.Rngs(sample=sample_key))
        samples = jnp.asarray(result.sampler_state)[-num_samples:]
        return PredictiveDistribution(
            mean=jnp.mean(samples, axis=0),
            variance=jnp.var(samples, axis=0),
            samples=samples,
            metadata=(
                ("method", "nre"),
                ("backend", self.backend),
                ("mcmc_method", self.mcmc_method),
                ("num_samples", num_samples),
            ),
        )


def _train_ratio_classifier(
    *,
    classifier: _RatioClassifier,
    optimizer: nnx.Optimizer,
    theta_pos: jax.Array,
    x_pos: jax.Array,
    theta_neg: jax.Array,
    x_neg: jax.Array,
    num_steps: int,
) -> jax.Array:
    """Train classifier with the BCE log-density-ratio loss for ``num_steps``."""

    @nnx.jit
    def step(model: _RatioClassifier, opt: nnx.Optimizer) -> jax.Array:
        def loss_fn(m: _RatioClassifier) -> jax.Array:
            logit_pos = m(theta_pos, x_pos)
            logit_neg = m(theta_neg, x_neg)
            # BCE-with-logits for {pos: 1, neg: 0}.
            loss_pos = jnp.mean(jax.nn.softplus(-logit_pos))
            loss_neg = jnp.mean(jax.nn.softplus(logit_neg))
            return loss_pos + loss_neg

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    losses = [step(classifier, optimizer) for _ in range(num_steps)]
    return jnp.stack(losses)


__all__ = ["NREState", "NeuralRatioEstimator"]
