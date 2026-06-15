"""Tests pinning physics-correct boundary-condition handling in B-PINNs.

These tests codify the boundary-condition (BC) likelihood contract for the
Bayesian PINN surface so the implementation cannot regress to the previous
stub behaviour (BC applied to *all* collocation points, a hard-coded ``0.1``
multiplier overriding the configured ``boundary_weight``, and Neumann/Robin
BC types silently returning zero).

Reference implementation
------------------------
Yang, Meng & Karniadakis (2021), *"B-PINNs: Bayesian Physics-Informed
Neural Networks for forward and inverse PDE problems with noisy data"*,
J. Comput. Phys. 425:109913 (arXiv:2003.06097). A B-PINN places a Gaussian
likelihood on BOTH the PDE residual AND the boundary/initial-condition data,
evaluated at dedicated boundary points ``x_b`` (distinct from the interior
PDE collocation points). The Dirichlet/Neumann/Robin residual form follows
deepxde's ``DirichletBC.error`` / ``NeumannBC.error`` (``../deepxde``):
``r_b = u_theta(x_b) - g(x_b)`` for Dirichlet. The posterior over weights
then yields a predictive whose mean matches the BC and whose interval covers
the BC value (low predictive std at the boundary).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from opifex.neural.bayesian.probabilistic_pinns import (
    compute_boundary_residual,
    ProbabilisticPINN,
    RobustPINNOptimizer,
)
from opifex.uncertainty.objectives import ObjectiveConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_pinn(rngs: nnx.Rngs | None = None, input_dim: int = 1) -> ProbabilisticPINN:
    if rngs is None:
        rngs = nnx.Rngs(params=0, sample=1, default=2, noise=3)
    return ProbabilisticPINN(
        input_dim=input_dim,
        hidden_dims=(16, 16),
        use_bayesian=True,
        rngs=rngs,
    )


def _objective(*, boundary_weight: float = 1.0) -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=0.0,
        dataset_size=100,
        physics_weight=0.0,
        data_weight=0.0,
        boundary_weight=boundary_weight,
        initial_condition_weight=0.0,
        regularization_weight=0.0,
        calibration_weight=0.0,
        conformal_weight=0.0,
        pac_bayes_weight=0.0,
    )


# ---------------------------------------------------------------------------
# compute_boundary_residual — the shared, reference-cited BC helper
# ---------------------------------------------------------------------------


def test_dirichlet_residual_is_zero_when_prediction_matches_constant() -> None:
    """Dirichlet residual ``u - g`` vanishes when ``u`` equals the BC value."""
    y_pred = jnp.full((4, 1), 2.5)
    bc = {"type": "dirichlet", "value": 2.5}
    residual = compute_boundary_residual(x=jnp.zeros((4, 1)), y_pred=y_pred, boundary=bc)
    assert jnp.allclose(residual, 0.0)


def test_dirichlet_residual_nonzero_and_decreases_toward_bc() -> None:
    """The BC residual MSE is nonzero off-target and decreases on approach."""
    x = jnp.zeros((4, 1))
    bc = {"type": "dirichlet", "value": 1.0}
    far = jnp.mean(compute_boundary_residual(x, jnp.full((4, 1), 5.0), bc) ** 2)
    near = jnp.mean(compute_boundary_residual(x, jnp.full((4, 1), 1.1), bc) ** 2)
    assert float(far) > 0.0
    assert float(near) < float(far)


def test_dirichlet_residual_accepts_callable_target() -> None:
    """A callable target ``g(x)`` is honoured (deepxde pattern)."""
    x = jnp.array([[0.0], [1.0], [2.0]])
    y_pred = x  # u(x) = x exactly satisfies g(x) = x
    bc = {"type": "dirichlet", "value": lambda coords: coords}
    residual = compute_boundary_residual(x, y_pred, bc)
    assert jnp.allclose(residual, 0.0)


def test_neumann_residual_matches_normal_derivative_minus_target() -> None:
    """Neumann BC penalises ``du/dx - g``; a flat field with g=0 has zero residual."""
    x = jnp.array([[0.0], [0.25], [0.5]])

    def flat_field(coords: jax.Array) -> jax.Array:
        return jnp.full((coords.shape[0], 1), 3.0)  # constant -> derivative 0

    bc = {"type": "neumann", "value": 0.0, "model": flat_field}
    residual = compute_boundary_residual(x, flat_field(x), bc)
    assert jnp.allclose(residual, 0.0, atol=1e-5)


def test_neumann_residual_nonzero_for_linear_field() -> None:
    """A linear field u=2x has du/dx=2; Neumann residual against g=0 is 2."""
    x = jnp.array([[0.0], [0.5], [1.0]])

    def linear_field(coords: jax.Array) -> jax.Array:
        return 2.0 * coords

    bc = {"type": "neumann", "value": 0.0, "model": linear_field}
    residual = compute_boundary_residual(x, linear_field(x), bc)
    assert jnp.allclose(residual, 2.0, atol=1e-4)


def test_unknown_bc_type_raises_value_error() -> None:
    """An unsupported BC type fails fast instead of silently returning zero."""
    with pytest.raises(ValueError, match="unsupported boundary-condition type"):
        compute_boundary_residual(jnp.zeros((2, 1)), jnp.zeros((2, 1)), {"type": "periodic"})


def test_neumann_without_model_raises_value_error() -> None:
    """Neumann/Robin need a differentiable ``model`` callable; missing it fails fast."""
    with pytest.raises(ValueError, match="requires a 'model' callable"):
        compute_boundary_residual(
            jnp.zeros((2, 1)), jnp.zeros((2, 1)), {"type": "neumann", "value": 0.0}
        )


# ---------------------------------------------------------------------------
# Dedicated boundary points (B-PINN: BC evaluated at x_b, not interior x)
# ---------------------------------------------------------------------------


def test_loss_components_uses_dedicated_boundary_points() -> None:
    """When ``boundary_x`` is supplied the BC is scored there, not at interior x.

    The BC must depend on ``boundary_x``: moving the boundary points changes the
    boundary term, and using a deterministic ``physics_loss`` path the boundary
    term equals the BC MSE evaluated at ``boundary_x`` (NOT at the interior x).
    """
    pinn = _make_pinn()
    interior = jnp.linspace(0.2, 0.8, 6).reshape(-1, 1)
    boundary_x = jnp.array([[0.0], [1.0]])
    bc = {"type": "dirichlet", "value": 0.5, "boundary_x": boundary_x}

    # physics_loss uses the deterministic forward, so the boundary term is
    # reproducible: it must equal the BC MSE at boundary_x, and differ from the
    # BC MSE at the interior points.
    def trivial_residual(_x: jax.Array, _u: jax.Array) -> jax.Array:
        return jnp.zeros_like(_u)

    boundary_term = pinn.physics_loss(interior, trivial_residual, bc)
    y_b = pinn(boundary_x, deterministic=True)
    expected_at_boundary = jnp.mean((y_b - 0.5) ** 2)
    y_interior = pinn(interior, deterministic=True)
    at_interior = jnp.mean((y_interior - 0.5) ** 2)

    assert jnp.allclose(boundary_term, expected_at_boundary, rtol=1e-5, atol=1e-6)
    assert not jnp.allclose(boundary_term, at_interior, rtol=1e-3)


def test_loss_components_boundary_has_no_hidden_scale() -> None:
    """The BC component is the raw MSE — no hard-coded ``0.1`` multiplier.

    The configured ``boundary_weight`` is the single source of truth; doubling
    it must exactly double the boundary contribution to ``total``.
    """
    pinn = _make_pinn()
    boundary_x = jnp.array([[0.0], [1.0]])
    batch = {
        "x": jnp.linspace(0.2, 0.8, 4).reshape(-1, 1),
        "y": jnp.zeros((4, 1)),
        "boundary_conditions": {"type": "dirichlet", "value": 0.5, "boundary_x": boundary_x},
    }
    one = pinn.loss_components(
        batch, rngs=nnx.Rngs(sample=0), objective=_objective(boundary_weight=1.0)
    )
    two = pinn.loss_components(
        batch, rngs=nnx.Rngs(sample=0), objective=_objective(boundary_weight=2.0)
    )
    assert jnp.allclose(two.total, 2.0 * one.total, rtol=1e-5, atol=1e-6)


def test_robust_optimizer_supports_neumann_bc_without_dropping_it() -> None:
    """RobustPINNOptimizer scores a Neumann BC instead of returning zero."""
    pinn = _make_pinn()
    optimizer = RobustPINNOptimizer(model=pinn)
    boundary_x = jnp.array([[0.0], [1.0]])
    batch = {
        "x": jnp.linspace(0.2, 0.8, 4).reshape(-1, 1),
        "y_true": jnp.zeros((4, 1)),
        "boundary_conditions": {
            "type": "neumann",
            "value": 0.0,
            "boundary_x": boundary_x,
        },
    }
    out = optimizer.compute_loss_components(
        batch, rngs=nnx.Rngs(sample=0, noise=1), objective=_objective()
    )
    assert out.boundary is not None
    assert float(out.boundary) > 0.0  # untrained net has nonzero normal derivative


# ---------------------------------------------------------------------------
# Predictive coverage of the boundary value (calibration contract)
# ---------------------------------------------------------------------------


def test_boundary_coverage_method_reports_mean_and_interval_coverage() -> None:
    """``boundary_coverage`` returns predictive mean error and interval coverage."""
    pinn = _make_pinn()
    boundary_x = jnp.array([[0.0], [1.0]])
    report = pinn.boundary_coverage(
        boundary_x,
        boundary_value=0.0,
        rngs=nnx.Rngs(sample=0),
        num_samples=16,
    )
    assert set(report) >= {"mean_abs_error", "coverage", "predictive_std"}
    assert 0.0 <= float(report["coverage"]) <= 1.0


def test_trained_predictive_covers_dirichlet_boundary_value() -> None:
    """After fitting a constant Dirichlet BC the predictive interval covers it.

    Scope: a small bounded variational fit (1-D, two boundary points, a few
    hundred steps) — enough to drive the predictive MEAN onto the BC value and
    the 95% predictive interval to COVER it, demonstrating the B-PINN BC
    likelihood is wired into the posterior rather than dropped.
    """
    rngs = nnx.Rngs(params=0, sample=1, default=2, noise=3)
    pinn = _make_pinn(rngs)
    boundary_x = jnp.array([[0.0], [1.0]])
    bc_value = 0.7
    objective = ObjectiveConfig(
        kl_weight=1e-4,
        dataset_size=64,
        physics_weight=0.0,
        data_weight=0.0,
        boundary_weight=1.0,
        initial_condition_weight=0.0,
        regularization_weight=0.0,
        calibration_weight=0.0,
        conformal_weight=0.0,
        pac_bayes_weight=0.0,
    )
    batch = {
        "x": jnp.linspace(0.1, 0.9, 8).reshape(-1, 1),
        "y": jnp.zeros((8, 1)),
        "boundary_conditions": {
            "type": "dirichlet",
            "value": bc_value,
            "boundary_x": boundary_x,
        },
    }
    optimizer = nnx.Optimizer(pinn, optax.adam(5e-3), wrt=nnx.Param)

    @nnx.jit
    def step(model: ProbabilisticPINN, opt: nnx.Optimizer, key: jax.Array) -> jax.Array:
        # Build the ``nnx.Rngs`` INSIDE the trace from a traced key so the
        # RngStream is created at this trace level (avoids the cross-trace
        # ``TraceContextError`` when mutating a captured RngStream).
        def loss_fn(m: ProbabilisticPINN) -> jax.Array:
            rngs = nnx.Rngs(sample=key)
            return m.loss_components(batch, rngs=rngs, objective=objective).total

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    base_key = jax.random.PRNGKey(10)
    for seed in range(400):
        step(pinn, optimizer, jax.random.fold_in(base_key, seed))

    report = pinn.boundary_coverage(
        boundary_x, boundary_value=bc_value, rngs=nnx.Rngs(sample=99), num_samples=64
    )
    # Mean matches the BC value and the 95% interval covers it everywhere.
    assert float(report["mean_abs_error"]) < 0.1
    assert float(report["coverage"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# jit / grad / vmap smoke for the BC-aware transformable objective
# ---------------------------------------------------------------------------


def _bc_loss(model: ProbabilisticPINN, x_b: jax.Array, rngs: nnx.Rngs) -> jax.Array:
    batch = {
        "x": x_b,
        "y": jnp.zeros((x_b.shape[0], 1)),
        "boundary_conditions": {"type": "dirichlet", "value": 0.3, "boundary_x": x_b},
    }
    return model.loss_components(batch, rngs=rngs, objective=_objective()).total


def test_bc_objective_is_jit_compatible() -> None:
    """The BC-aware objective traces under ``nnx.jit``."""
    pinn = _make_pinn()
    x_b = jnp.array([[0.0], [1.0]])

    @nnx.jit
    def run(model: ProbabilisticPINN, rngs: nnx.Rngs) -> jax.Array:
        return _bc_loss(model, x_b, rngs)

    out = run(pinn, nnx.Rngs(sample=0))
    assert jnp.isfinite(out)


def test_bc_objective_is_grad_compatible() -> None:
    """``nnx.grad`` yields finite gradients through the BC term."""
    pinn = _make_pinn()
    x_b = jnp.array([[0.0], [1.0]])
    grads = nnx.grad(lambda m: _bc_loss(m, x_b, nnx.Rngs(sample=0)))(pinn)
    leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_bc_residual_is_vmap_compatible() -> None:
    """``compute_boundary_residual`` vmaps over a batch of boundary targets."""
    x = jnp.zeros((3, 1))
    y_pred = jnp.full((3, 1), 1.0)

    def residual_for(value: jax.Array) -> jax.Array:
        return jnp.mean(
            compute_boundary_residual(x, y_pred, {"type": "dirichlet", "value": value}) ** 2
        )

    values = jnp.array([0.0, 1.0, 2.0])
    out = jax.vmap(residual_for)(values)
    assert out.shape == (3,)
    # value == 1.0 gives zero residual (matches y_pred); the others are positive.
    assert jnp.allclose(out[1], 0.0, atol=1e-6)
    assert float(out[0]) > 0.0 and float(out[2]) > 0.0
