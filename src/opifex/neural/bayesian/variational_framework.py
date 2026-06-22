"""Variational inference framework for Bayesian neural networks.

This module provides a full framework for variational Bayesian neural networks
with physics-informed priors and amortized uncertainty estimation.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.types import PredictiveDistribution


# Lazy import for distrax to avoid tf_keras dependency issues
_distrax_cache: dict[str, object] = {}


def _get_distrax():  # type: ignore[return]
    """Lazily import distrax to avoid import errors when tf_keras is not installed."""
    if "module" not in _distrax_cache:
        try:
            import distrax  # type: ignore[import]

            _distrax_cache["module"] = distrax
        except ImportError as e:
            msg = (
                "distrax is required for variational inference but could not "
                "be imported. This is often due to missing tf_keras. "
                "Install with: pip install tf-keras"
            )
            raise ImportError(msg) from e
    return _distrax_cache["module"]  # type: ignore[return-value]


from typing import TYPE_CHECKING

from jaxtyping import Array, Float  # noqa: TC002


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


@dataclasses.dataclass
class PriorConfig:
    """Configuration for physics-informed priors.

    Attributes:
        conservation_laws: List of conservation laws to enforce
            (e.g., ['energy', 'momentum']).
        boundary_conditions: List of boundary conditions to incorporate.
        physics_constraints: List of physics constraints to respect.
        prior_scale: Scale parameter for the prior distribution.
    """

    conservation_laws: Sequence[str] = ()
    boundary_conditions: Sequence[str] = ()
    physics_constraints: Sequence[str] = ()
    prior_scale: float = 1.0


@dataclasses.dataclass
class VariationalConfig:
    """Configuration for variational inference.

    Attributes:
        input_dim: Dimensionality of input features.
        hidden_dims: Tuple of hidden layer dimensions for the encoder.
        num_samples: Number of samples to draw during inference.
        kl_weight: Weight for the KL divergence term in ELBO.
        temperature: Temperature parameter for variational distribution.
    """

    input_dim: int
    hidden_dims: Sequence[int] = (64, 32)
    num_samples: int = 10
    kl_weight: float = 1.0
    temperature: float = 1.0


class MeanFieldGaussian(nnx.Module):
    """Mean-field Gaussian variational posterior over a weight vector.

    The factorized posterior ``q(w) = N(mu, diag(sigma^2))`` over a weight
    vector ``w in R^num_params`` is the variational object injected into a base
    network by :class:`AmortizedVariationalFramework`.

    On its own it is also a complete **Bayesian linear model** (Bishop, *PRML*
    3.3): for an input ``x in R^num_params`` the prediction ``f(x) = w . x`` has
    the closed-form predictive ``f(x) ~ N(mu . x, sum_i x_i^2 sigma_i^2)`` --
    Gaussian because the map is linear and ``q(w)`` is Gaussian. A homoscedastic
    Gaussian observation noise ``y ~ N(f(x), sigma_y^2)`` (the learnable
    ``log_observation_std``) completes the likelihood, so the layer exposes the
    platform UQ protocol surfaces (:meth:`predict_distribution`,
    :meth:`loss_components`, :meth:`negative_elbo`, :meth:`kl_divergence`)
    directly, with the expected NLL available in closed form (no sampling).
    """

    def __init__(
        self,
        num_params: int,
        *,
        rngs: nnx.Rngs,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        observation_noise: float = 0.1,
    ) -> None:
        """Initialize the mean-field Gaussian posterior.

        Args:
            num_params: Number of weights ``w`` the posterior factorizes over.
            rngs: Random number generator state.
            prior_mean: Mean of the factorized Gaussian prior ``p(w)`` used by
                :meth:`kl_divergence`.
            prior_std: Standard deviation of the prior ``p(w)``; must be
                positive.
            observation_noise: Initial homoscedastic observation-noise standard
                deviation ``sigma_y`` of the Gaussian likelihood; learnable via
                ``log_observation_std``. Must be positive.
        """
        super().__init__()
        if prior_std <= 0.0:
            raise ValueError(f"prior_std must be positive; got {prior_std!r}.")
        if observation_noise <= 0.0:
            raise ValueError(f"observation_noise must be positive; got {observation_noise!r}.")
        self.num_params = num_params
        # Static prior configuration (Python floats -> not pytree leaves).
        self._prior_mean = float(prior_mean)
        self._prior_std = float(prior_std)

        # Variational parameters of q(w) = N(mu, diag(exp(2 log_std))).
        self.mean = nnx.Param(nnx.initializers.zeros_init()(rngs.params(), (num_params,)))
        self.log_std = nnx.Param(nnx.initializers.constant(-2.0)(rngs.params(), (num_params,)))
        # Learnable homoscedastic observation noise of the Gaussian likelihood.
        self.log_observation_std = nnx.Param(jnp.asarray(math.log(observation_noise)))

    def sample(self, num_samples: int, *, rngs: nnx.Rngs) -> Float[Array, "samples params"]:
        """Sample from variational posterior.

        Args:
            num_samples: Number of samples to draw.
            rngs: Random number generator state.

        Returns:
            Array of shape (num_samples, num_params) containing parameter samples.
        """
        eps = jax.random.normal(rngs.sample(), (num_samples, self.num_params))
        return self.mean.value + jnp.exp(self.log_std.value) * eps

    def log_prob(self, samples: Float[Array, "samples params"]) -> Float[Array, "samples"]:  # noqa: F821, UP037 — jaxtyping shape string, not a forward ref
        """Compute log probability of samples.

        Args:
            samples: Parameter samples of shape (num_samples, num_params).

        Returns:
            Log probabilities for each sample of shape (num_samples,).
        """
        distrax = jax.tree.map(lambda x: x, _get_distrax())  # type: ignore # noqa: PGH003
        return jnp.asarray(
            distrax.MultivariateNormalDiag(self.mean.value, jnp.exp(self.log_std.value)).log_prob(
                samples
            )
        )

    def kl_divergence(
        self, prior_mean: float | None = None, prior_std: float | None = None
    ) -> Float[Array, ""]:
        """Compute ``KL(q(w) || p(w))`` from the factorized Gaussian prior.

        Args:
            prior_mean: Prior mean; defaults to the value supplied at
                construction (``prior_mean=0.0`` unless overridden).
            prior_std: Prior standard deviation; defaults to the value supplied
                at construction (``prior_std=1.0`` unless overridden).

        Returns:
            KL divergence scalar value.
        """
        prior_mean = self._prior_mean if prior_mean is None else prior_mean
        prior_std = self._prior_std if prior_std is None else prior_std
        distrax = jax.tree.map(lambda x: x, _get_distrax())  # type: ignore # noqa: PGH003
        posterior_dist = distrax.MultivariateNormalDiag(
            self.mean.value, jnp.exp(self.log_std.value)
        )
        prior_dist = distrax.MultivariateNormalDiag(
            jnp.full_like(self.mean.value, prior_mean),
            jnp.full_like(self.mean.value, prior_std),
        )
        return jnp.array(posterior_dist.kl_divergence(prior_dist))

    def _predictive_moments(
        self, x: Float[Array, "batch params"]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(predictive mean, epistemic variance, observation variance)``.

        Closed-form moments of the Bayesian-linear prediction ``f(x) = w . x``
        under ``q(w) = N(mu, diag(sigma^2))``: ``E[f] = mu . x`` and
        ``Var[f] = sum_i x_i^2 sigma_i^2``. ``sigma_y^2`` is the homoscedastic
        observation (aleatoric) variance.
        """
        x = jnp.asarray(x)
        if x.shape[-1] != self.num_params:
            raise ValueError(
                "input feature dimension must equal num_params="
                f"{self.num_params}; got x.shape={x.shape}."
            )
        weight_variance = jnp.exp(2.0 * self.log_std.value)
        predictive_mean = x @ self.mean.value
        epistemic_variance = (x**2) @ weight_variance
        observation_variance = jnp.exp(2.0 * self.log_observation_std.value)
        return predictive_mean, epistemic_variance, observation_variance

    def predict_distribution(
        self,
        x: Float[Array, "batch params"],
        *,
        rngs: nnx.Rngs | None = None,  # noqa: ARG002 — deterministic closed form; rngs unused
    ) -> PredictiveDistribution:
        """Return the closed-form Bayesian-linear predictive for inputs ``x``.

        The predictive ``f(x) ~ N(mu . x, x^2 . sigma^2)`` plus the homoscedastic
        observation noise ``sigma_y^2`` gives ``epistemic = x^2 . sigma^2``,
        ``aleatoric = sigma_y^2``, and ``total = epistemic + aleatoric`` -- all
        in closed form, so no Monte-Carlo ``rngs`` are needed.

        Args:
            x: Inputs of shape ``(batch, num_params)`` (or ``(num_params,)``).
            rngs: Unused -- the predictive is exact; accepted for protocol
                conformance with stochastic models.

        Returns:
            A :class:`PredictiveDistribution` with mean, variance, and the
            epistemic / aleatoric / total decomposition.
        """
        predictive_mean, epistemic_variance, observation_variance = self._predictive_moments(x)
        aleatoric_variance = jnp.broadcast_to(observation_variance, epistemic_variance.shape)
        total_variance = epistemic_variance + aleatoric_variance
        return PredictiveDistribution(
            mean=predictive_mean,
            variance=total_variance,
            epistemic=epistemic_variance,
            aleatoric=aleatoric_variance,
            total_uncertainty=total_variance,
            metadata=(
                ("method", "bayesian_linear_closed_form"),
                ("model", "mean_field_gaussian"),
            ),
        )

    def loss_components(
        self,
        batch: Mapping[str, Any],
        *,
        config: ObjectiveConfig,
        rngs: nnx.Rngs | None = None,  # noqa: ARG002 — closed-form ELBO; rngs unused
    ) -> UQLossComponents:
        """Return the per-batch negative-ELBO decomposition.

        The expected negative log-likelihood under ``q(w)`` is available in
        closed form for the Gaussian likelihood::

            E_q[NLL] = 0.5 log(2 pi sigma_y^2)
                       + (mean((y - mu.x)^2) + mean(Var_q[f])) / (2 sigma_y^2)

        and is combined with ``KL(q || p)`` by :meth:`UQLossComponents.from_components`
        using the weights / dataset scaling in ``config``.

        Args:
            batch: Mapping with required fields ``x`` (``(batch, num_params)``)
                and ``y`` (``(batch,)``).
            config: Loss weights and dataset metadata.
            rngs: Unused -- the expected NLL is exact; accepted for protocol
                conformance.

        Returns:
            The optimizer-facing :class:`UQLossComponents` decomposition.
        """
        missing = [field for field in ("x", "y") if field not in batch]
        if missing:
            raise ValueError(f"batch missing required field(s): {missing!r}")
        x = jnp.asarray(batch["x"])
        y = jnp.asarray(batch["y"])

        predictive_mean, predictive_variance, observation_variance = self._predictive_moments(x)
        squared_error = (y - predictive_mean) ** 2
        expected_nll = 0.5 * jnp.log(2.0 * jnp.pi * observation_variance) + (
            jnp.mean(squared_error) + jnp.mean(predictive_variance)
        ) / (2.0 * observation_variance)
        kl = self.kl_divergence()

        return UQLossComponents.from_components(
            config=config,
            negative_log_likelihood=expected_nll,
            kl=kl,
            metadata=(("source", "mean_field_gaussian"),),
        )

    def negative_elbo(
        self,
        batch: Mapping[str, Any],
        *,
        config: ObjectiveConfig,
        rngs: nnx.Rngs | None = None,
    ) -> Float[Array, ""]:
        """Return the scalar negative-ELBO objective for one batch.

        Args:
            batch: Mapping with required fields ``x`` and ``y``.
            config: Loss weights and dataset metadata.
            rngs: Forwarded to :meth:`loss_components` (unused there).

        Returns:
            The scalar ``total`` of :meth:`loss_components` -- the value passed
            to ``jax.value_and_grad`` / ``optimizer.update``.
        """
        return self.loss_components(batch, config=config, rngs=rngs).total


class UncertaintyEncoder(nnx.Module):
    """Neural network for amortized uncertainty estimation.

    This encoder network predicts the parameters of the variational posterior
    directly from input data, enabling amortized variational inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the uncertainty encoder.

        Args:
            input_dim: Dimensionality of input features.
            hidden_dims: Sequence of hidden layer dimensions.
            output_dim: Dimensionality of output (typically 2 * num_params
                for mean and log_std).
            rngs: Random number generator state.
        """
        super().__init__()

        # Build encoder layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nnx.Linear(current_dim, hidden_dim, rngs=rngs),
                    nnx.relu,
                ]
            )
            current_dim = hidden_dim

        # Output layer (mean and log_std)
        layers.append(nnx.Linear(current_dim, output_dim, rngs=rngs))

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: Float[Array, "batch input_dim"]) -> Float[Array, "batch output_dim"]:
        """Forward pass through uncertainty encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim) containing
                posterior parameters.
        """
        return self.layers(x)


class AmortizedVariationalFramework(nnx.Module):
    """Variational framework with amortized uncertainty estimation.

    This framework combines a base neural network model with variational Bayesian
    inference capabilities, enabling uncertainty quantification through amortized
    variational inference.
    """

    def __init__(
        self,
        base_model: nnx.Module,
        prior_config: PriorConfig,
        variational_config: VariationalConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the amortized variational framework.

        Args:
            base_model: Base neural network model to augment with uncertainty.
            prior_config: Configuration for physics-informed priors.
            variational_config: Configuration for variational inference.
            rngs: Random number generator state.
        """
        super().__init__()

        self.base_model = base_model
        self.config = variational_config

        # Split the base model into a static graph definition plus its
        # ``nnx.Param`` leaves. ``ravel_pytree`` gives a flat <-> tree
        # bijection so a sampled parameter *vector* can be injected back
        # into a functional copy of the model via ``nnx.merge`` (the
        # canonical NNX state pattern). The unravel closure and graphdef
        # are static under ``jax.jit`` — only the flat vector is traced.
        base_graphdef, base_params = nnx.split(base_model, nnx.Param)
        flat_params, unravel_params = jax.flatten_util.ravel_pytree(base_params)
        self._base_graphdef = base_graphdef
        self._unravel_params = unravel_params
        self.num_params = int(flat_params.shape[0])

        # Variational posterior approximation
        self.variational_posterior = MeanFieldGaussian(num_params=self.num_params, rngs=rngs)

        # Amortized inference network
        self.amortization_network = UncertaintyEncoder(
            input_dim=variational_config.input_dim,
            hidden_dims=variational_config.hidden_dims,
            output_dim=2 * self.num_params,  # mean and log_std
            rngs=rngs,
        )

        # Store prior config for physics constraints
        self.prior_config = prior_config

    def _split_posterior_params(
        self, posterior_params: Float[Array, "batch params"]
    ) -> tuple[Float[Array, "batch half_params"], Float[Array, "batch half_params"]]:
        """Split posterior parameters into mean and log_std.

        Args:
            posterior_params: Combined posterior parameters.

        Returns:
            Tuple of (mean_params, log_std_params).
        """
        half_params = self.num_params
        mean_params = posterior_params[..., :half_params]
        log_std_params = posterior_params[..., half_params:]
        return mean_params, log_std_params

    def predict_with_uncertainty(
        self,
        x: Float[Array, "batch input_dim"],
        num_samples: int | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> tuple[Float[Array, "batch output_dim"], Float[Array, "batch output_dim"]]:
        """Forward pass with uncertainty quantification.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            num_samples: Number of Monte Carlo samples for uncertainty estimation.
            rngs: Random number generator state.

        Returns:
            Tuple of (mean_prediction, uncertainty) both of shape
                (batch_size, output_dim).
        """

        if num_samples is None:
            num_samples = self.config.num_samples

        # Amortized inference: predict posterior parameters
        posterior_params = self.amortization_network(x)
        mean_params, log_std_params = self._split_posterior_params(posterior_params)

        # Build a posterior seeded with the amortized (batch-mean)
        # predictions, draw parameter vectors from it, and inject each
        # draw into the base model for a true variational forward pass.
        temp_posterior = MeanFieldGaussian(num_params=self.num_params, rngs=rngs)
        batch_mean_params = jnp.mean(mean_params, axis=0)
        batch_log_std_params = jnp.mean(log_std_params, axis=0)
        temp_posterior.mean[...] = batch_mean_params
        temp_posterior.log_std[...] = batch_log_std_params

        param_samples = temp_posterior.sample(num_samples, rngs=rngs)

        predictions = jax.vmap(lambda pv: self._forward_with_params(x, pv))(param_samples)
        mean_prediction = jnp.mean(predictions, axis=0)
        uncertainty = jnp.var(predictions, axis=0)

        return mean_prediction, uncertainty

    def _forward_with_params(
        self,
        x: Float[Array, "batch input_dim"],
        params_vector: Float[Array, ...],
    ) -> Float[Array, "batch output_dim"]:
        """Run the base model with an injected sampled parameter vector.

        The flat ``params_vector`` (a single draw from the variational
        posterior) is unravelled into the base model's ``nnx.Param`` tree
        and merged with the static graph definition captured at
        construction. The reconstructed functional copy is then evaluated
        on ``x``, so the perturbation lives in the network **weights** —
        this is the true variational forward pass, not an input-space
        surrogate. The routine is a pure function of ``x`` and
        ``params_vector`` and is therefore ``jit`` / ``grad`` / ``vmap``
        compatible.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            params_vector: Flat parameter vector of shape ``(num_params,)``
                drawn from the variational posterior.

        Returns:
            Model output of shape (batch_size, output_dim) under the
            injected weights.
        """
        params_state = self._unravel_params(params_vector)
        model = nnx.merge(self._base_graphdef, params_state)
        return model(x)  # type: ignore[misc, operator]

    def compute_elbo(
        self,
        x: Float[Array, "batch input_dim"],
        y: Float[Array, "batch output_dim"],
        num_samples: int | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> Float[Array, ""]:
        """Compute Evidence Lower BOund (ELBO).

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            y: Target tensor of shape (batch_size, output_dim).
            num_samples: Number of Monte Carlo samples for ELBO estimation.
            rngs: Random number generator state.

        Returns:
            ELBO scalar value (higher is better).
        """
        if num_samples is None:
            num_samples = self.config.num_samples

        # Amortized inference: predict posterior parameters
        posterior_params = self.amortization_network(x)
        mean_params, log_std_params = self._split_posterior_params(posterior_params)

        # Update variational posterior with amortized predictions
        batch_mean_params = jnp.mean(mean_params, axis=0)
        batch_log_std_params = jnp.mean(log_std_params, axis=0)

        # Create temporary posterior for ELBO computation
        temp_posterior = MeanFieldGaussian(num_params=self.num_params, rngs=rngs)
        temp_posterior.mean[...] = batch_mean_params
        temp_posterior.log_std[...] = batch_log_std_params

        # Sample from variational posterior
        param_samples = temp_posterior.sample(num_samples, rngs=rngs)

        # Monte-Carlo expected log likelihood under the injected weights.
        # Gaussian likelihood with sigma^2 = 0.01 (negative scaled MSE).
        sigma_squared = 0.01

        def _sample_log_likelihood(params_vector: Float[Array, ...]) -> Float[Array, ""]:
            y_pred = self._forward_with_params(x, params_vector)
            return -jnp.sum((y - y_pred) ** 2) / (2.0 * sigma_squared)

        per_sample_ll = jax.vmap(_sample_log_likelihood)(param_samples)
        log_likelihood = jnp.mean(per_sample_ll)

        # Compute KL divergence from prior
        kl_divergence = temp_posterior.kl_divergence()

        # Evidence Lower BOund
        return log_likelihood - self.config.kl_weight * kl_divergence

    def sample_predictive_distribution(
        self,
        x: Float[Array, "batch input_dim"],
        num_samples: int | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> Float[Array, "samples batch output_dim"]:
        """Sample from predictive distribution.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            num_samples: Number of predictive samples to generate.
            rngs: Random number generator state.

        Returns:
            Predictive samples of shape (num_samples, batch_size, output_dim).
        """
        if num_samples is None:
            num_samples = self.config.num_samples

        # Get posterior parameters from amortization network
        posterior_params = self.amortization_network(x)
        mean_params, log_std_params = self._split_posterior_params(posterior_params)

        # Create temporary posterior
        batch_mean_params = jnp.mean(mean_params, axis=0)
        batch_log_std_params = jnp.mean(log_std_params, axis=0)

        temp_posterior = MeanFieldGaussian(num_params=self.num_params, rngs=rngs)
        temp_posterior.mean[...] = batch_mean_params
        temp_posterior.log_std[...] = batch_log_std_params

        # Sample parameter vectors and inject each into the base model.
        param_samples = temp_posterior.sample(num_samples, rngs=rngs)

        return jax.vmap(lambda pv: self._forward_with_params(x, pv))(param_samples)

    def __call__(
        self,
        x: Float[Array, "batch input_dim"],
        num_samples: int = 10,
        *,
        rngs: nnx.Rngs,
    ) -> tuple[Float[Array, "batch output_dim"], Float[Array, "batch output_dim"]]:
        """Forward pass with uncertainty quantification.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            num_samples: Number of Monte Carlo samples for uncertainty estimation.
            rngs: Caller-owned ``nnx.Rngs`` bundle driving the Monte Carlo
                draws — required, no hidden fixed-seed fallback.

        Returns:
            Tuple of (mean_prediction, uncertainty) both of shape
                (batch_size, output_dim).
        """
        return self.predict_with_uncertainty(x, num_samples, rngs=rngs)
