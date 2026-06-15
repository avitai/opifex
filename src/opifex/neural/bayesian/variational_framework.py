"""Variational inference framework for Bayesian neural networks.

This module provides a full framework for variational Bayesian neural networks
with physics-informed priors and amortized uncertainty estimation.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx


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
    from collections.abc import Sequence


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
    """Mean-field Gaussian variational posterior.

    This class implements a factorized Gaussian posterior distribution for
    variational inference in neural networks.
    """

    def __init__(self, num_params: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the mean-field Gaussian posterior.

        Args:
            num_params: Number of parameters in the posterior.
            rngs: Random number generator state.
        """
        super().__init__()
        self.num_params = num_params

        # Variational parameters
        self.mean = nnx.Param(nnx.initializers.zeros_init()(rngs.params(), (num_params,)))
        self.log_std = nnx.Param(nnx.initializers.constant(-2.0)(rngs.params(), (num_params,)))

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

    def log_prob(self, samples: Float[Array, "samples params"]) -> Float[Array, samples]:  # type: ignore[reportUndefinedVariable]  # noqa: F821
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

    def kl_divergence(self, prior_mean: float = 0.0, prior_std: float = 1.0) -> Float[Array, ""]:
        """Compute KL divergence from prior.

        Args:
            prior_mean: Mean of the prior distribution.
            prior_std: Standard deviation of the prior distribution.

        Returns:
            KL divergence scalar value.
        """
        distrax = jax.tree.map(lambda x: x, _get_distrax())  # type: ignore # noqa: PGH003
        posterior_dist = distrax.MultivariateNormalDiag(
            self.mean.value, jnp.exp(self.log_std.value)
        )
        prior_dist = distrax.MultivariateNormalDiag(
            jnp.full_like(self.mean.value, prior_mean),
            jnp.full_like(self.mean.value, prior_std),
        )
        return jnp.array(posterior_dist.kl_divergence(prior_dist))


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
