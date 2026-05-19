"""Pin ``RobustPINNOptimizer`` integration against the shared UQ surface.

The legacy ``RobustPINNOptimizer.compute_loss_components`` returned a plain
``dict[str, jax.Array]`` and stored loss-component weights as instance
attributes; the legacy ``uncertainty_guided_sampling`` and the internal
``_compute_robustness_penalty`` both fell back to a hidden ``nnx.Rngs(0)``
default. The migration (Phase 3 Task 3.3) replaces all three with:

* return type :class:`opifex.uncertainty.objectives.UQLossComponents`,
* :class:`opifex.uncertainty.objectives.ObjectiveConfig`-driven weights, and
* caller-owned ``rngs`` at every stochastic method boundary.

This file pins those three contracts so the migration cannot regress.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random

from opifex.neural.bayesian.probabilistic_pinns import (
    ProbabilisticPINN,
    RobustPINNOptimizer,
)
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents


def _make_objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=32,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=0.1,
        calibration_weight=1.0,
        conformal_weight=1.0,
        pac_bayes_weight=1.0,
    )


def _make_batch(seed: int = 0) -> dict:
    key = random.PRNGKey(seed)
    k_x, k_y = random.split(key)
    return {
        "x": random.normal(k_x, (8, 2)),
        "y_true": random.normal(k_y, (8, 16)),
        "pde_residual_fn": lambda _x, predictions: jnp.sum(predictions**2, axis=-1),
    }


def test_robust_pinn_optimizer_constructs_uq_loss_components_from_shared_surface() -> None:
    """The migrated caller imports the shared surface and emits ``UQLossComponents``."""
    pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(16, 8), rngs=nnx.Rngs(0))
    optimizer = RobustPINNOptimizer(model=pinn)

    components = optimizer.compute_loss_components(
        _make_batch(), rngs=nnx.Rngs(7), objective=_make_objective()
    )

    assert isinstance(components, UQLossComponents)
    assert jnp.isfinite(components.total)
    assert components.data is not None
    assert components.physics_residual is not None
    assert components.regularization is not None
    assert components.kl is not None
    assert components.metadata_dict()["source"] == "robust_pinn_optimizer"


def test_robust_pinn_optimizer_total_is_differentiable_under_nnx_value_and_grad() -> None:
    """``compute_loss_components`` composes with ``nnx.value_and_grad(has_aux=True)``.

    Follows the canonical Flax NNX pattern: ``rngs`` is passed as a
    traced argument (closing over an outer ``nnx.Rngs`` would advance an
    ``RngStream`` across trace levels and raise ``TraceContextError``).
    """
    pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(16, 8), rngs=nnx.Rngs(0))
    optimizer = RobustPINNOptimizer(model=pinn)
    batch = _make_batch()
    objective = _make_objective()

    def loss_fn(opt: RobustPINNOptimizer, rngs: nnx.Rngs) -> tuple[jax.Array, UQLossComponents]:
        comps = opt.compute_loss_components(batch, rngs=rngs, objective=objective)
        return comps.total, comps

    (total, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer, nnx.Rngs(noise=7))

    assert jnp.isfinite(total)
    assert isinstance(aux, UQLossComponents)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "expected at least one gradient leaf from the Bayesian PINN parameters"
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_uncertainty_guided_sampling_rejects_call_without_rngs() -> None:
    """The migrated active-learning helper never falls back to a hidden fixed key."""
    import pytest

    pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(16, 8), rngs=nnx.Rngs(0))
    optimizer = RobustPINNOptimizer(model=pinn)
    x_candidates = random.normal(random.PRNGKey(11), (16, 2))

    with pytest.raises(TypeError):
        optimizer.uncertainty_guided_sampling(  # type: ignore[call-arg]
            x_candidates, num_samples=4
        )
