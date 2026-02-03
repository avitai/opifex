"""Tests for second-order optimization wrappers.

TDD: These tests define the expected behavior for the optax/optimistix wrappers.
"""

import jax
import jax.numpy as jnp
import optax
import optimistix as optx

from opifex.optimization.second_order.config import (
    GaussNewtonConfig,
    LBFGSConfig,
    LinesearchType,
)
from opifex.optimization.second_order.wrappers import (
    create_bfgs_solver,
    create_gauss_newton_solver,
    create_lbfgs_optimizer,
    create_levenberg_marquardt_solver,
)


class TestCreateLBFGSOptimizer:
    """Test L-BFGS optimizer creation."""

    def test_creates_valid_optimizer(self):
        """Should create a valid optax optimizer."""
        optimizer = create_lbfgs_optimizer()
        assert optimizer is not None
        # Should be a GradientTransformation
        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")

    def test_default_config_creates_optimizer(self):
        """Default config should work."""
        optimizer = create_lbfgs_optimizer(None)
        assert optimizer is not None

    def test_custom_config_applied(self):
        """Custom config should be applied."""
        config = LBFGSConfig(memory_size=5, linesearch=LinesearchType.BACKTRACKING)
        optimizer = create_lbfgs_optimizer(config)
        assert optimizer is not None

    def test_optimizer_can_be_initialized(self):
        """Optimizer should initialize with parameters."""
        optimizer = create_lbfgs_optimizer()
        params = {"w": jnp.ones(10), "b": jnp.zeros(5)}
        state = optimizer.init(params)
        assert state is not None

    def test_optimizer_can_update(self):
        """Optimizer should compute updates with value_and_grad."""
        optimizer = create_lbfgs_optimizer(LBFGSConfig(memory_size=5))
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # L-BFGS requires value and grad from state
        value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=state)
        updates, _new_state = optimizer.update(  # type: ignore # noqa: PGH003
            grad,
            state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
        )
        new_params = optax.apply_updates(params, updates)

        # Should make progress toward minimum
        assert loss_fn(new_params) <= loss_fn(params)

    def test_convergence_on_quadratic(self):
        """L-BFGS should converge on a quadratic function."""
        optimizer = create_lbfgs_optimizer(
            LBFGSConfig(memory_size=10, max_iterations=50)
        )
        params = jnp.array([5.0, -3.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return 0.5 * jnp.sum(p**2)

        # Run several steps
        for _ in range(20):
            value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=state)
            updates, state = optimizer.update(  # type: ignore # noqa: PGH003
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=loss_fn,
            )
            params = optax.apply_updates(params, updates)

        # Should be very close to zero
        assert jnp.allclose(params, 0.0, atol=1e-3)  # type: ignore # noqa: PGH003


class TestCreateGaussNewtonSolver:
    """Test Gauss-Newton solver creation."""

    def test_creates_valid_solver(self):
        """Should create a valid optimistix solver."""
        solver = create_gauss_newton_solver()
        assert solver is not None
        assert isinstance(solver, optx.AbstractLeastSquaresSolver)

    def test_default_config_creates_solver(self):
        """Default config should work."""
        solver = create_gauss_newton_solver(None)
        assert solver is not None

    def test_custom_config_applied(self):
        """Custom config should be applied."""
        config = GaussNewtonConfig(rtol=1e-8, atol=1e-8)
        solver = create_gauss_newton_solver(config)
        assert solver is not None


class TestCreateLevenbergMarquardtSolver:
    """Test Levenberg-Marquardt solver creation."""

    def test_creates_valid_solver(self):
        """Should create a valid optimistix solver."""
        solver = create_levenberg_marquardt_solver()
        assert solver is not None
        assert isinstance(solver, optx.AbstractLeastSquaresSolver)

    def test_can_solve_least_squares(self):
        """Should solve a simple least-squares problem."""
        solver = create_levenberg_marquardt_solver()

        # Simple linear regression problem: find x such that Ax = b
        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = jnp.array([3.0, 7.0, 11.0])

        def residual_fn(x, args):
            return A @ x - b

        x0 = jnp.zeros(2)
        sol = optx.least_squares(residual_fn, solver, x0, args=None)

        # Should find approximately x = [1, 1]
        assert jnp.allclose(sol.value, jnp.array([1.0, 1.0]), atol=1e-4)


class TestCreateBFGSSolver:
    """Test BFGS solver creation."""

    def test_creates_valid_solver(self):
        """Should create a valid optimistix minimizer."""
        solver = create_bfgs_solver()
        assert solver is not None
        assert isinstance(solver, optx.AbstractMinimiser)

    def test_can_minimize_quadratic(self):
        """Should minimize a simple quadratic function."""
        solver = create_bfgs_solver()

        def fn(x, args):
            return 0.5 * jnp.sum(x**2)

        x0 = jnp.array([5.0, -3.0, 2.0])
        sol = optx.minimise(fn, solver, x0, args=None)

        # Should find minimum at zero
        assert jnp.allclose(sol.value, jnp.zeros(3), atol=1e-4)


class TestJITCompatibility:
    """Test JIT compatibility of all wrappers."""

    def test_lbfgs_jit_compatible(self):
        """L-BFGS operations should be JIT-compatible."""
        optimizer = create_lbfgs_optimizer()
        params = jnp.array([1.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        @jax.jit
        def step(params, state):
            value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=state)
            updates, new_state = optimizer.update(  # type: ignore # noqa: PGH003
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=loss_fn,
            )
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state

        new_params, _new_state = step(params, state)
        assert jnp.isfinite(new_params).all()

    def test_levenberg_marquardt_jit_compatible(self):
        """LM solver should be JIT-compatible."""
        solver = create_levenberg_marquardt_solver()

        def residual_fn(x, args):
            return x**2 - 1.0

        @jax.jit
        def solve(x0):
            sol = optx.least_squares(residual_fn, solver, x0, args=None)
            # Return just the value for JIT compatibility
            return sol.value

        x0 = jnp.array([0.5])
        result = solve(x0)
        assert jnp.isfinite(result).all()
