"""UDE → SINDy distillation pipeline.

Extracts the learned behavior of a UDE's neural residual component and
discovers its symbolic form via SINDy sparse regression. This enables
interpretable equation discovery from grey-box models.

Pipeline:
    1. Evaluate trained neural residual on sample data
    2. Treat (x, neural_residual(x)) as (state, derivative) pairs
    3. Run SINDy to find sparse symbolic representation

Reference:
    Rackauckas et al. (2020) "Universal Differential Equations for
    Scientific Machine Learning"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opifex.discovery.sindy.sindy import SINDy


if TYPE_CHECKING:
    from collections.abc import Callable

    import jax
    import jax.numpy as jnp

    from opifex.discovery.sindy.config import SINDyConfig


def distill_ude_residual(
    neural_residual: Callable[[jax.Array], jax.Array],
    x_eval: jnp.ndarray,
    *,
    config: SINDyConfig | None = None,
) -> SINDy:
    """Distill a UDE neural residual into symbolic equations via SINDy.

    Evaluates the neural residual on sample data and uses SINDy to find
    the sparsest symbolic representation of the learned function.

    Args:
        neural_residual: Trained neural network module that maps
            state -> residual dynamics. Should accept (batch, state_dim).
        x_eval: Evaluation data, shape (n_samples, state_dim).
        config: SINDy configuration. Uses defaults if None.

    Returns:
        Fitted SINDy model whose coefficients represent the symbolic
        form of the neural residual.
    """
    # Evaluate the neural residual on sample data
    y_residual = neural_residual(x_eval)

    # Treat (x_eval, y_residual) as (state, derivative) pairs
    # and run SINDy to discover the symbolic form
    model = SINDy(config)
    model.fit(x_eval, y_residual)

    return model


__all__ = ["distill_ude_residual"]
