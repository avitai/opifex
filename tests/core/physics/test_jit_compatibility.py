"""Test JIT compilation compatibility for physics loss components.

This module verifies that all physics loss components are JIT-compatible
and don't break due to Python control flow or other JIT restrictions.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.physics import (
    AutoDiffEngine,
    PDEResidualRegistry,
    PhysicsInformedLoss,
    PhysicsLossConfig,
    ResidualComputer,
)


class TestPDEResidualJIT:
    """Test that PDE residuals are JIT-compatible."""

    def test_poisson_jit_compilation(self):
        """Test that Poisson residual can be JIT-compiled."""

        def u_solution(x):
            """Test solution: u = x² + y²"""
            return jnp.sum(x**2, axis=-1)

        poisson_fn = PDEResidualRegistry.get("poisson")

        # Wrap in JIT
        @jax.jit
        def compute_jitted(x, source):
            return poisson_fn(u_solution, x, AutoDiffEngine, source_term=source)

        # Test compilation and execution
        x = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        source = jnp.array([4.0, 4.0])

        residual = compute_jitted(x, source)

        assert residual.shape == (2,)
        assert jnp.isfinite(residual).all()
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_heat_jit_compilation(self):
        """Test that heat equation residual can be JIT-compiled."""

        def u_harmonic(x):
            """Harmonic function: u = x² - y²"""
            return x[..., 0] ** 2 - x[..., 1] ** 2

        heat_fn = PDEResidualRegistry.get("heat")

        @jax.jit
        def compute_jitted(x, alpha):
            return heat_fn(u_harmonic, x, AutoDiffEngine, alpha=alpha)

        x = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        residual = compute_jitted(x, 1.0)

        assert residual.shape == (2,)
        assert jnp.isfinite(residual).all()

    def test_wave_jit_compilation(self):
        """Test that wave equation residual can be JIT-compiled."""

        def u_wave(x):
            """Wave solution"""
            return jnp.sin(jnp.pi * x[..., 0])

        wave_fn = PDEResidualRegistry.get("wave")

        @jax.jit
        def compute_jitted(x, c):
            return wave_fn(u_wave, x, AutoDiffEngine, wave_speed=c)

        x = jnp.linspace(0, 1, 50).reshape(-1, 1)
        residual = compute_jitted(x, 1.0)

        assert residual.shape == (50,)
        assert jnp.isfinite(residual).all()

    def test_schrodinger_jit_compilation(self):
        """Test that Schrödinger equation residual can be JIT-compiled."""

        def psi_ground(x):
            """Ground state wavefunction"""
            r_sq = jnp.sum(x**2, axis=-1)
            return jnp.exp(-0.5 * r_sq)

        schrodinger_fn = PDEResidualRegistry.get("schrodinger")

        @jax.jit
        def compute_jitted(x):
            return schrodinger_fn(
                psi_ground, x, AutoDiffEngine, potential_type="harmonic"
            )

        x = jnp.linspace(-2, 2, 64).reshape(-1, 1)
        residual = compute_jitted(x)

        assert residual.shape == (64,)
        assert jnp.isfinite(residual).all()

    def test_burgers_jit_compilation(self):
        """Test that Burgers equation residual can be JIT-compiled."""

        def u_burgers(x):
            """Test solution for Burgers"""
            return x[..., 0]

        burgers_fn = PDEResidualRegistry.get("burgers")

        @jax.jit
        def compute_jitted(x, nu):
            return burgers_fn(u_burgers, x, AutoDiffEngine, nu=nu)

        x = jnp.linspace(0, 1, 32).reshape(-1, 1)
        residual = compute_jitted(x, 0.01)

        assert residual.shape == (32,)
        assert jnp.isfinite(residual).all()


class TestResidualComputerJIT:
    """Test that ResidualComputer is JIT-compatible."""

    def test_residual_computer_jit(self):
        """Test that ResidualComputer.compute_residual can be JIT-compiled."""

        computer = ResidualComputer(equation_type="poisson", domain_type="2d")

        def u_solution(x):
            return jnp.sum(x**2, axis=-1)

        @jax.jit
        def compute_jitted(x, source):
            return computer.compute_residual(u_solution, x, source=source)

        x = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        source = jnp.array([4.0, 4.0])

        residual = compute_jitted(x, source)

        assert residual.shape == (2,)
        assert jnp.isfinite(residual).all()


class TestAutoDiffEngineJIT:
    """Test that AutoDiffEngine functions are JIT-compatible."""

    def test_compute_gradient_jit(self):
        """Test JIT compilation of compute_gradient."""

        def f(x):
            return jnp.sum(x**2, axis=-1)

        @jax.jit
        def grad_jitted(x):
            return AutoDiffEngine.compute_gradient(f, x)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        grad = grad_jitted(x)

        assert grad.shape == (2, 2)
        assert jnp.allclose(grad, x * 2, atol=1e-5)

    def test_compute_laplacian_jit(self):
        """Test JIT compilation of compute_laplacian."""

        def f(x):
            return jnp.sum(x**2, axis=-1)

        @jax.jit
        def laplacian_jitted(x):
            return AutoDiffEngine.compute_laplacian(f, x)

        x = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        lap = laplacian_jitted(x)

        assert lap.shape == (2,)
        assert jnp.allclose(lap, 4.0, atol=1e-5)

    def test_compute_divergence_jit(self):
        """Test JIT compilation of compute_divergence."""

        def F(x):  # noqa: N802
            """Radial field"""
            return x

        @jax.jit
        def divergence_jitted(x):
            return AutoDiffEngine.compute_divergence(F, x)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        div = divergence_jitted(x)

        assert div.shape == (2,)
        assert jnp.allclose(div, 2.0, atol=1e-5)


class TestNNXModelJIT:
    """Test JIT compatibility with NNX models."""

    def test_pde_residual_with_nnx_model_jit(self):
        """Test that PDE residuals work with JIT-compiled NNX models."""

        class SimplePINN(nnx.Module):
            def __init__(self, rngs):
                self.dense1 = nnx.Linear(2, 10, rngs=rngs)
                self.dense2 = nnx.Linear(10, 1, rngs=rngs)

            def __call__(self, x):
                """Return (batch, 1) shape as expected by autodiff engine."""
                h = nnx.relu(self.dense1(x))
                return self.dense2(h)  # Don't squeeze - keep (batch, 1) shape

        model = SimplePINN(rngs=nnx.Rngs(42))
        poisson_fn = PDEResidualRegistry.get("poisson")

        @jax.jit
        def compute_jitted(x, source):
            return poisson_fn(model, x, AutoDiffEngine, source_term=source)

        x = jnp.array([[0.5, 0.5], [0.25, 0.75]])
        source = jnp.zeros(2)

        residual = compute_jitted(x, source)

        assert residual.shape == (2,)
        assert jnp.isfinite(residual).all()


class TestGradientThroughPDE:
    """Test that we can backprop through PDE residuals (critical for PINN training)."""

    def test_gradient_through_poisson_residual(self):
        """Test backpropagation through Poisson residual computation."""

        class SimplePINN(nnx.Module):
            def __init__(self, rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                """Return (batch, 1) shape for autodiff compatibility."""
                return self.dense(x)

        model = SimplePINN(rngs=nnx.Rngs(42))
        poisson_fn = PDEResidualRegistry.get("poisson")

        def loss_fn(model, x, source):
            """PINN loss: mean squared PDE residual"""
            residual = poisson_fn(model, x, AutoDiffEngine, source_term=source)
            return jnp.mean(residual**2)

        # Compute gradient with respect to model parameters
        x = jnp.array([[0.5, 0.5], [0.25, 0.75]])
        source = jnp.zeros(2)

        loss, grads = nnx.value_and_grad(loss_fn)(model, x, source)

        # Verify gradients exist and are finite
        assert jnp.isfinite(loss)
        assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))

    def test_jit_gradient_through_pde(self):
        """Test JIT-compiled gradient computation through PDE."""

        class SimplePINN(nnx.Module):
            def __init__(self, rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                """Return (batch, 1) shape for autodiff compatibility."""
                return self.dense(x)

        model = SimplePINN(rngs=nnx.Rngs(42))
        poisson_fn = PDEResidualRegistry.get("poisson")

        @nnx.jit
        def train_step(model, x, source):
            """JIT-compiled training step"""

            def loss_fn(model):
                residual = poisson_fn(model, x, AutoDiffEngine, source_term=source)
                return jnp.mean(residual**2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            return loss, grads

        x = jnp.array([[0.5, 0.5], [0.25, 0.75]])
        source = jnp.zeros(2)

        loss, grads = train_step(model, x, source)

        # Verify JIT compilation worked
        assert jnp.isfinite(loss)
        assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))


class TestVmapCompatibility:
    """Test that PDE residuals work with vmap batching."""

    def test_vmap_over_multiple_inputs(self):
        """Test vmap over multiple input configurations."""

        def u_solution(x):
            return jnp.sum(x**2, axis=-1)

        poisson_fn = PDEResidualRegistry.get("poisson")

        # Create multiple input arrays to vmap over
        x_batch = jnp.array(
            [
                [[1.0, 1.0], [0.5, 0.5]],  # Config 1
                [[0.25, 0.75], [0.1, 0.9]],  # Config 2
            ]
        )  # shape: (2, 2, 2) = (n_configs, batch, dim)

        source_batch = jnp.array(
            [
                [4.0, 4.0],  # Config 1
                [4.0, 4.0],  # Config 2
            ]
        )  # shape: (2, 2) = (n_configs, batch)

        def compute_for_config(x, source):
            return poisson_fn(u_solution, x, AutoDiffEngine, source_term=source)

        # vmap over configurations
        vmapped_compute = jax.vmap(compute_for_config)
        residuals = vmapped_compute(x_batch, source_batch)

        assert residuals.shape == (2, 2)  # (n_configs, batch)
        assert jnp.isfinite(residuals).all()


class TestStaticVsDynamicControl:
    """Test that we correctly handle static vs dynamic control flow."""

    def test_static_model_check_in_physics_loss(self):
        """Verify that model=None check doesn't break JIT."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
        )

        pi_loss = PhysicsInformedLoss(
            config=config, equation_type="poisson", domain_type="2d"
        )

        # Test with model=None (static branch)
        @jax.jit
        def compute_without_model(predictions, targets, inputs):
            loss, _ = pi_loss.compute_loss(
                predictions=predictions,
                targets=targets,
                inputs=inputs,
                model=None,  # Static None
            )
            return loss

        x = jnp.array([[0.5, 0.5], [0.25, 0.75]])
        y_true = jnp.array([0.5, 0.625])
        y_pred = y_true + 0.1

        loss = compute_without_model(y_pred, y_true, x)
        assert jnp.isfinite(loss)

    def test_static_model_provided_in_physics_loss(self):
        """Verify that providing a model also works with JIT."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
        )

        pi_loss = PhysicsInformedLoss(
            config=config, equation_type="poisson", domain_type="2d"
        )

        def model(x):
            return jnp.sum(x**2, axis=-1)

        @jax.jit
        def compute_with_model(predictions, targets, inputs):
            loss, _ = pi_loss.compute_loss(
                predictions=predictions,
                targets=targets,
                inputs=inputs,
                model=model,  # Static function
                source=jnp.array([4.0, 4.0]),
            )
            return loss

        x = jnp.array([[0.5, 0.5], [0.25, 0.75]])
        y_true = jnp.array([0.5, 0.625])
        y_pred = y_true + 0.1

        loss = compute_with_model(y_pred, y_true, x)
        assert jnp.isfinite(loss)
