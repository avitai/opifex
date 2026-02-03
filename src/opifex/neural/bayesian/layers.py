"""Bayesian neural network layers for uncertainty quantification."""

import jax
import jax.numpy as jnp
from flax import nnx


class BayesianLayer(nnx.Module):
    """
    Bayesian neural network layer for uncertainty quantification.

    Implements variational Bayesian layer with weight distributions
    for epistemic uncertainty estimation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Bayesian layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_std: Standard deviation of weight prior
            rngs: Random number generator state
        """
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight mean and log variance
        self.weight_mean = nnx.Param(
            nnx.initializers.xavier_normal()(rngs.params(), (out_features, in_features))
        )
        self.weight_logvar = nnx.Param(jnp.full((out_features, in_features), -10.0))

        # Bias mean and log variance
        self.bias_mean = nnx.Param(jnp.zeros(out_features))
        self.bias_logvar = nnx.Param(jnp.full(out_features, -10.0))

    def __call__(
        self, x: jax.Array, training: bool = True, sample: bool = True
    ) -> jax.Array:
        """Forward pass with optional weight sampling."""
        if sample and training:
            # Sample weights
            weight_eps = jax.random.normal(
                jax.random.PRNGKey(0), self.weight_mean.value.shape
            )
            weight_std = jnp.exp(0.5 * self.weight_logvar.value)
            weight = self.weight_mean.value + weight_std * weight_eps

            # Sample bias
            bias_eps = jax.random.normal(
                jax.random.PRNGKey(1), self.bias_mean.value.shape
            )
            bias_std = jnp.exp(0.5 * self.bias_logvar.value)
            bias = self.bias_mean.value + bias_std * bias_eps
        else:
            weight = self.weight_mean.value
            bias = self.bias_mean.value

        return jnp.dot(x, weight.T) + bias
