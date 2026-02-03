"""Test suite for classical problem types (PDEs, ODEs, Optimization).

This module tests classical problem definitions including:
- Partial Differential Equations (PDEs)
- Ordinary Differential Equations (ODEs)
- Optimization problems
- Problem integration and validation

Tests extracted from test_problems.py during refactoring.
"""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import jax.numpy as jnp

from opifex.core.problems import (
    create_molecular_system,
    create_neural_dft_problem,
    create_ode_problem,
    create_optimization_problem,
    create_pde_problem,
    PDEProblem,
)
from opifex.geometry.base import Geometry
from opifex.geometry.csg import Rectangle


if TYPE_CHECKING:
    from collections.abc import Callable


class TestPDEProblem:
    """Test PDE problem definitions."""

    def test_heat_equation_problem(self):
        """Test heat equation problem definition."""

        def heat_equation(x, u, u_derivs):
            return u_derivs["dt"] - 0.1 * u_derivs["d2x"]

        # Define geometry instead of domain
        geometry = Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0)
        boundary_conditions = {"x0": 0.0, "x1": 0.0}

        problem = create_pde_problem(
            geometry=geometry,
            equation=heat_equation,
            boundary_conditions=boundary_conditions,
        )

        assert problem.validate()
        assert problem.get_geometry() == geometry
        assert isinstance(problem.get_geometry(), Geometry)

    def test_wave_equation_problem(self):
        """Test wave equation problem definition."""

        def wave_equation(x, u, u_derivs):
            c_squared = 1.0
            return u_derivs["d2t"] - c_squared * u_derivs["d2x"]

        geometry = Rectangle(center=jnp.array([0.0, 0.5]), width=2.0, height=1.0)
        boundary_conditions = {"x_neg1": 0.0, "x_1": 0.0}
        initial_conditions = {"u0": "sin(pi*x)", "ut0": "0"}

        problem = create_pde_problem(
            geometry=geometry,
            equation=wave_equation,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
        )

        assert problem.validate()

    def test_invalid_pde_problem(self):
        """Test invalid PDE problem validation."""
        # Test with invalid geometry type (not a Geometry instance)
        problem1 = create_pde_problem(
            geometry=None,  # type: ignore[arg-type]  # Invalid geometry for testing
            equation=lambda x, u, u_d: u,  # Valid callable but empty domain
            boundary_conditions={},
        )
        assert not problem1.validate()

        # Test validation of problem with invalid equation via direct instantiation
        # This bypasses the factory function to test validation logic
        class TestPDEProblem(PDEProblem):
            def residual(self, x, u, u_derivatives):
                return u

        problem2 = TestPDEProblem.__new__(TestPDEProblem)  # Create without __init__
        problem2.geometry = Rectangle(
            center=jnp.array([0.0, 0.0]), width=1.0, height=1.0
        )
        problem2.equation = cast("Callable", None)  # Invalid equation (type-safe)
        problem2.boundary_conditions = {}
        problem2.initial_conditions = {}
        problem2.parameters = {}

        assert not problem2.validate()


class TestODEProblem:
    """Test ODE problem definitions."""

    def test_harmonic_oscillator(self):
        """Test harmonic oscillator ODE problem."""

        def harmonic_oscillator(t, y):
            omega = 1.0
            return jnp.array([y[1], -(omega**2) * y[0]])

        time_span = (0.0, 10.0)
        initial_conditions = {"y0": jnp.array([1.0, 0.0])}

        problem = create_ode_problem(
            time_span=time_span,
            equation=harmonic_oscillator,
            initial_conditions=initial_conditions,
        )

        assert problem.validate()
        assert problem.get_time_domain()["t"] == time_span

    def test_lorenz_system(self):
        """Test Lorenz system ODE problem."""

        def lorenz_system(t, y):
            sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
            return jnp.array(
                [
                    sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    y[0] * y[1] - beta * y[2],
                ]
            )

        problem = create_ode_problem(
            time_span=(0.0, 30.0),
            equation=lorenz_system,
            initial_conditions={"y0": jnp.array([1.0, 1.0, 1.0])},
        )

        assert problem.validate()

    def test_invalid_ode_problem(self):
        """Test invalid ODE problem validation."""
        problem = create_ode_problem(
            time_span=(1.0, 0.0),  # Invalid time span
            equation=lambda t, y: y,
        )

        assert not problem.validate()


class TestOptimizationProblem:
    """Test optimization problem definitions."""

    def test_quadratic_optimization(self):
        """Test quadratic optimization problem."""

        def quadratic(x):
            return jnp.sum(x**2)

        problem = create_optimization_problem(
            dimension=3, objective=quadratic, bounds=[(-1.0, 1.0)] * 3
        )

        assert problem.validate()
        assert problem.dimension == 3
        assert problem.bounds is not None
        assert problem.bounds is not None
        assert len(problem.bounds) == 3

    def test_rosenbrock_optimization(self):
        """Test Rosenbrock function optimization."""

        def rosenbrock(x):
            return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        problem = create_optimization_problem(dimension=2, objective=rosenbrock)

        assert problem.validate()
        assert problem.bounds is None

    def test_invalid_optimization_problem(self):
        """Test invalid optimization problem validation."""
        problem = create_optimization_problem(
            dimension=0,  # Invalid dimension
            objective=lambda x: x[0],
        )

        assert not problem.validate()

    def test_optimization_gradient_and_hessian(self):
        """Test gradient and hessian computation for optimization problems."""

        def quadratic(x):
            return jnp.sum(x**2)

        problem = create_optimization_problem(
            dimension=3, objective=quadratic, bounds=[(-2.0, 2.0)] * 3
        )

        # Test gradient computation
        x = jnp.array([1.0, 2.0, 3.0])
        gradient = problem.gradient(x)
        expected_gradient = jnp.array([2.0, 4.0, 6.0])  # d/dx(x^2) = 2x
        assert gradient.shape == (3,)
        assert jnp.allclose(gradient, expected_gradient, atol=1e-6)

        # Test hessian computation
        hessian = problem.hessian(x)
        expected_hessian = jnp.eye(3) * 2.0  # d²/dx²(x^2) = 2
        assert hessian.shape == (3, 3)
        assert jnp.allclose(hessian, expected_hessian, atol=1e-6)

    def test_optimization_rosenbrock_derivatives(self):
        """Test gradient and hessian for Rosenbrock function."""

        def rosenbrock(x):
            return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        problem = create_optimization_problem(dimension=2, objective=rosenbrock)

        # Test at a known point
        x = jnp.array([0.0, 0.0])
        gradient = problem.gradient(x)
        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))

        hessian = problem.hessian(x)
        assert hessian.shape == (2, 2)
        assert jnp.all(jnp.isfinite(hessian))
        # Hessian should be symmetric
        assert jnp.allclose(hessian, hessian.T, atol=1e-8)


class TestProblemIntegration:
    """Test integration between different problem types."""

    def test_problem_protocol_compliance(self):
        """Test that all problem types implement the Problem protocol."""
        water = create_molecular_system(
            [
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.757, 0.586, 0.0)),
                ("H", (-0.757, 0.586, 0.0)),
            ]
        )

        # Create instances of each problem type
        pde_problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=lambda x, u, u_d: u_d["dx"],
            boundary_conditions={"x0": 0.0},
        )

        ode_problem = create_ode_problem(time_span=(0.0, 1.0), equation=lambda t, y: y)

        opt_problem = create_optimization_problem(
            dimension=2, objective=lambda x: jnp.sum(x**2)
        )

        neural_dft_problem = create_neural_dft_problem(water)

        # Test Protocol compliance
        problems = [pde_problem, ode_problem, opt_problem, neural_dft_problem]

        for problem in problems:
            # get_domain is no longer consistent, but validate and parameters are
            if isinstance(problem, PDEProblem):
                assert isinstance(problem.get_geometry(), Geometry)
            assert hasattr(problem, "get_parameters")
            assert hasattr(problem, "validate")
            assert callable(problem.get_parameters)
            assert callable(problem.validate)

    def test_quantum_chemistry_workflow(self):
        """Test complete quantum chemistry workflow setup."""
        # Create a complex molecule (benzene)
        benzene = create_molecular_system(
            [
                ("C", (1.40, 0.00, 0.00)),
                ("C", (0.70, 1.21, 0.00)),
                ("C", (-0.70, 1.21, 0.00)),
                ("C", (-1.40, 0.00, 0.00)),
                ("C", (-0.70, -1.21, 0.00)),
                ("C", (0.70, -1.21, 0.00)),
                ("H", (2.49, 0.00, 0.00)),
                ("H", (1.24, 2.15, 0.00)),
                ("H", (-1.24, 2.15, 0.00)),
                ("H", (-2.49, 0.00, 0.00)),
                ("H", (-1.24, -2.15, 0.00)),
                ("H", (1.24, -2.15, 0.00)),
            ]
        )

        # Create Neural DFT problem
        problem = create_neural_dft_problem(
            molecular_system=benzene,
            functional_type="neural_xc",
            scf_method="neural_scf",
            convergence_threshold=1e-8,
        )

        assert problem.validate()
        assert problem.molecular_system.n_atoms == 12
        assert problem.molecular_system.n_electrons == 42  # 6*6 + 6*1 = 42

        # Verify Neural DFT setup
        functional_config = problem.setup_neural_functional()
        scf_config = problem.setup_scf_cycle()

        assert functional_config["functional_type"] == "neural_xc"
        assert scf_config["method"] == "neural_scf"
        assert scf_config["acceleration"] == "neural"


class TestProblemsEnhancement:
    """Test class to enhance code coverage for classical problems."""

    def test_pde_problem_validation_edge_cases(self):
        """Test PDE problem validation with invalid configurations."""

        # Test with empty domain
        def dummy_equation(x, u, u_derivs):
            return u

        # Test validation with invalid geometry
        problem = create_pde_problem(
            geometry=None,  # type: ignore[arg-type]  # Invalid for testing
            equation=dummy_equation,
            boundary_conditions={},
        )
        assert not problem.validate()

        # Test validation with non-callable equation
        problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=lambda x, u, u_d: u,  # Valid callable but will be overridden
            boundary_conditions={},
        )
        # Manually set equation to None to test validation
        problem.equation = None  # type: ignore[assignment]
        assert not problem.validate()

    def test_ode_problem_validation_edge_cases(self):
        """Test ODE problem validation with invalid configurations."""

        def dummy_equation(t, y):
            return y

        # Test with invalid time span (end <= start)
        problem = create_ode_problem(
            time_span=(1.0, 0.0),  # Invalid time span
            equation=dummy_equation,
        )
        assert not problem.validate()

        # Test with equal start and end times
        problem = create_ode_problem(
            time_span=(1.0, 1.0),  # Equal start and end
            equation=dummy_equation,
        )
        assert not problem.validate()

        # Test with non-callable equation
        problem = create_ode_problem(
            time_span=(0.0, 1.0),
            equation=lambda x, y: y,  # Valid callable but will be overridden
        )
        # Manually set equation to None to test validation
        problem.equation = None  # type: ignore[assignment]
        assert not problem.validate()

    def test_optimization_problem_validation_edge_cases(self):
        """Test optimization problem validation with invalid configurations."""

        def dummy_objective(x):
            return float(jnp.sum(x**2))

        # Test with negative dimension
        problem = create_optimization_problem(
            dimension=-1,  # Invalid dimension
            objective=dummy_objective,
        )
        assert not problem.validate()

        # Test with zero dimension
        problem = create_optimization_problem(
            dimension=0,  # Invalid dimension
            objective=dummy_objective,
        )
        assert not problem.validate()

        # Test with mismatched bounds
        problem = create_optimization_problem(
            dimension=2,
            objective=dummy_objective,
            bounds=[(0, 1)],  # Only one bound for 2D problem
        )
        assert not problem.validate()

    def test_optimization_problem_gradient_and_hessian_edge_cases(self):
        """Test optimization problem gradient and hessian computation edge cases."""

        # Test with simple quadratic function
        def simple_quadratic(x):
            return jnp.sum(x**2)  # Remove float() call to avoid JAX tracer issues

        problem = create_optimization_problem(
            dimension=2,
            objective=simple_quadratic,
        )

        # Test gradient computation
        x = jnp.array([1.0, 2.0])
        grad = problem.gradient(x)
        assert grad.shape == (2,)
        assert jnp.allclose(grad, 2.0 * x)

        # Test hessian computation
        hessian = problem.hessian(x)
        assert hessian.shape == (2, 2)
        assert jnp.allclose(hessian, 2.0 * jnp.eye(2))

    def test_problem_protocol_methods(self):
        """Test problem protocol methods for various problem types."""

        # Test PDE problem protocol compliance
        def heat_equation(x, u, u_derivs):
            return u_derivs["dt"] - 0.1 * u_derivs["d2x"]

        pde_problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=heat_equation,
            boundary_conditions={"x0": 0.0, "x1": 0.0},
        )

        # Test get_geometry, get_parameters, validate
        geometry = pde_problem.get_geometry()
        assert isinstance(geometry, Geometry)

        params = pde_problem.get_parameters()
        assert isinstance(params, dict)

        assert pde_problem.validate()

        # Test ODE problem protocol compliance
        def harmonic_oscillator(t, y):
            return jnp.array([y[1], -y[0]])

        ode_problem = create_ode_problem(
            time_span=(0.0, 10.0),
            equation=harmonic_oscillator,
            initial_conditions={"y0": [1.0, 0.0]},
        )

        domain = ode_problem.get_time_domain()
        assert "t" in domain

        params = ode_problem.get_parameters()
        assert isinstance(params, dict)

        assert ode_problem.validate()

    def test_problem_parameter_handling(self):
        """Test parameter handling for different problem types."""

        # Test PDE with parameters
        def parameterized_pde(x, u, u_derivs):
            return u_derivs["dt"] - 0.1 * u_derivs["d2x"]

        pde_problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=parameterized_pde,
            boundary_conditions={"x0": 0.0, "x1": 0.0},
            parameters={"diffusion_coeff": 0.1, "source_term": 0.0},
        )

        params = pde_problem.get_parameters()
        assert params["diffusion_coeff"] == 0.1
        assert params["source_term"] == 0.0

        # Test ODE with parameters
        def parameterized_ode(t, y):
            return jnp.array([y[1], -y[0]])

        ode_problem = create_ode_problem(
            time_span=(0.0, 10.0),
            equation=parameterized_ode,
            parameters={"frequency": 1.0, "damping": 0.1},
        )

        params = ode_problem.get_parameters()
        assert params["frequency"] == 1.0
        assert params["damping"] == 0.1
