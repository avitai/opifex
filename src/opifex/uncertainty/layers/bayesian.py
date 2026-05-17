"""Phase 2 Task 2.1 — shared :class:`BayesianLinear` NNX module.

Single canonical implementation of the variational diagonal-Gaussian dense
layer. Replaces the two parallel implementations in
``src/opifex/neural/bayesian/layers.py`` and
``src/opifex/neural/operators/specialized/uqno.py``; Phase 2 Task 2.3 and
Phase 3 Task 3.4 migrate those call sites to import from here.

RNG safety (GUIDE_ALIGNMENT items 4, 4a, 5, 7, 9):

* Constructor ``rngs`` initializes parameters only.
* Stochastic sampling routes every call through
  ``artifex.generative_models.core.rng.extract_rng_key`` — caller-owned
  ``nnx.Rngs`` (advancing the ``"posterior"`` stream) or an explicit
  ``jax.Array`` key. No hidden ``jax.random.PRNGKey(...)`` seeds in the
  production path.

KL math: delegated to
:func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl` which itself
delegates to Artifex ``gaussian_kl_divergence`` for the N(0,1) case
(GUIDE_ALIGNMENT item 8). One formula, owned in one place.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx

from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl


# Named-stream resolution order, matching the Avitai canonical convention.
_POSTERIOR_STREAMS: tuple[str, ...] = ("posterior", "sample", "default")


class BayesianLinear(nnx.Module):
    """Variational diagonal-Gaussian dense layer.

    Weight and bias each carry a ``(mean, log-variance)`` posterior; sampling
    uses the reparameterization trick. Pre-defined ``__call__`` flags:

    * ``training=False`` — return the posterior-mean prediction (no sampling).
    * ``sample=False`` — same; equivalent to deterministic mode.
    * ``training=True, sample=True`` — sample weights / bias from the
      diagonal-Gaussian posterior; ``rngs`` MUST be provided by the caller.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize variational parameters; ``rngs`` initializes parameters only."""
        if prior_std <= 0.0:
            raise ValueError(f"prior_std must be > 0; got {prior_std!r}.")

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mean = nnx.Param(
            nnx.initializers.xavier_normal()(rngs.params(), (out_features, in_features))
        )
        self.weight_logvar = nnx.Param(jnp.full((out_features, in_features), -10.0))
        self.bias_mean = nnx.Param(jnp.zeros(out_features))
        self.bias_logvar = nnx.Param(jnp.full(out_features, -10.0))

    def __call__(
        self,
        x: jax.Array,
        *,
        training: bool = True,
        sample: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass; samples weights when ``training and sample``."""
        if training and sample:
            if rngs is None:
                raise ValueError(
                    "BayesianLinear sampling requires caller-owned `rngs` "
                    "(nnx.Rngs with a posterior/sample/default stream, or a "
                    "jax.Array key)."
                )
            key = extract_rng_key(
                rngs,
                streams=_POSTERIOR_STREAMS,
                context="BayesianLinear sampling",
            )
            weight_key, bias_key = jax.random.split(key)
            weight = _reparameterize(self.weight_mean[...], self.weight_logvar[...], weight_key)
            bias = _reparameterize(self.bias_mean[...], self.bias_logvar[...], bias_key)
        else:
            weight = self.weight_mean[...]
            bias = self.bias_mean[...]

        return jnp.dot(x, weight.T) + bias

    def kl_divergence(self) -> jax.Array:
        """Total KL divergence (weights + bias) under the layer's diagonal Gaussian prior."""
        weight_kl = diagonal_gaussian_kl(
            self.weight_mean[...],
            self.weight_logvar[...],
            prior_mean=0.0,
            prior_std=self.prior_std,
        )
        bias_kl = diagonal_gaussian_kl(
            self.bias_mean[...],
            self.bias_logvar[...],
            prior_mean=0.0,
            prior_std=self.prior_std,
        )
        return weight_kl + bias_kl


def _reparameterize(mean: jax.Array, logvar: jax.Array, key: jax.Array) -> jax.Array:
    """Reparameterization-trick sample from ``N(mean, exp(logvar))``."""
    std = jnp.exp(0.5 * logvar)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    return mean + std * noise


__all__ = ["BayesianLinear"]
