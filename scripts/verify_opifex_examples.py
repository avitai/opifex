#!/usr/bin/env python3
"""
Comprehensive verification script for Opifex documentation examples.

This script tests all the key code examples from README files and documentation
to ensure they work correctly with the current implementation.
"""

import os
import sys
import traceback

import jax
import jax.numpy as jnp
from flax import nnx


os.environ["JAX_ENABLE_X64"] = "true"


class OpifexExampleTester:
    """Test runner for Opifex documentation examples."""

    def __init__(self, verbose: bool = False):
        self.passed = 0
        self.failed = 0
        self.verbose = verbose
        self.results: list[tuple[str, bool, str]] = []

    def test(self, name: str, test_func) -> None:
        """Run a single test."""
        print(f"Testing {name}...", end=" ")
        try:
            test_func()
            print("✅ PASSED")
            self.passed += 1
            self.results.append((name, True, ""))
        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                error_msg = traceback.format_exc()
            print(f"❌ FAILED: {error_msg}")
            self.failed += 1
            self.results.append((name, False, error_msg))

    def print_summary(self) -> bool:
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print("Opifex Documentation Examples Test Summary")
        print(f"{'=' * 60}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        if total > 0:
            print(f"Success rate: {(self.passed / total) * 100:.1f}%")

        if self.failed > 0:
            print("\nFailed Tests:")
            for name, passed, error in self.results:
                if not passed:
                    print(f"  - {name}: {error.split(chr(10))[0]}")  # First line only

        return self.failed == 0


def test_problem_creation():
    """Test problem creation examples from core documentation."""
    from opifex.core.problems import (
        create_neural_dft_problem,
        create_ode_problem,
        create_pde_problem,
    )
    from opifex.core.quantum.molecular_system import create_molecular_system

    domain = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

    def heat_equation(x, u, u_derivatives):
        """Simple heat equation."""
        return u_derivatives["u_t"] - 0.1 * (
            u_derivatives["u_xx"] + u_derivatives["u_yy"]
        )

    # Test PDE problem creation
    pde_problem = create_pde_problem(
        domain=domain,
        equation=heat_equation,
        boundary_conditions={"type": "dirichlet", "value": 0.0},
    )
    assert pde_problem is not None, "PDE problem should be created"

    # Test ODE problem creation
    def ode_equation(t, y):
        """Simple ODE."""
        return jnp.array([-y[0], y[1], -y[2]])

    ode_problem = create_ode_problem(
        time_span=(0.0, 1.0),
        equation=ode_equation,
        initial_conditions={"y0": jnp.array([1.0, 0.0, 0.0])},
    )
    assert ode_problem is not None, "ODE problem should be created"

    # Test molecular system creation
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.74, 0.0, 0.0))]
    molecular_system = create_molecular_system(atoms=atoms)
    assert molecular_system is not None, "Molecular system should be created"
    assert molecular_system.n_atoms == 2, "Should have 2 atoms"

    # Test Neural DFT problem creation
    neural_dft_problem = create_neural_dft_problem(
        molecular_system=molecular_system,
        functional_type="neural_xc",
        scf_method="neural_scf",
    )
    assert neural_dft_problem is not None, "Neural DFT problem should be created"


def test_geometry_operations():
    """Test geometry system examples."""
    from opifex.geometry.algebra.groups import SO3Group
    from opifex.geometry.csg import Circle, CSGUnion, Rectangle

    # Test 2D CSG operations
    circle = Circle(center=jnp.array([0.0, 0.0]), radius=1.0)
    rectangle = Rectangle(center=jnp.array([0.0, 0.0]), width=1.0, height=1.0)

    # Test containment
    point_inside = jnp.array([0.0, 0.0])
    assert circle.contains(point_inside), "Point should be inside circle"

    # Test union operation
    union_shape = CSGUnion(circle, rectangle)
    assert union_shape.contains(point_inside), "Point should be inside union"

    # Test SO(3) group operations
    so3 = SO3Group()
    identity = so3.identity()
    assert identity.shape == (3, 3), "Identity should be 3x3 matrix"

    # Test rotation
    axis_angle = jnp.array([0.1, 0.2, 0.3])
    rotation = so3.exp_map(axis_angle)
    assert rotation.shape == (3, 3), "Rotation should be 3x3 matrix"


def test_boundary_conditions():
    """Test boundary condition examples."""
    from opifex.core.conditions import DirichletBC, NeumannBC, WavefunctionBC

    # Test classical boundary conditions
    dirichlet_bc = DirichletBC(boundary="left", value=0.0)
    assert dirichlet_bc is not None, "Dirichlet BC should be created"

    neumann_bc = NeumannBC(boundary="right", value=1.0)
    assert neumann_bc is not None, "Neumann BC should be created"

    # Test quantum boundary conditions
    wavefunction_bc = WavefunctionBC(
        condition_type="vanishing", boundary="all", value=complex(0.0)
    )
    assert wavefunction_bc is not None, "Wavefunction BC should be created"


def test_neural_networks():
    """Test neural network examples."""
    from opifex.neural.activations import get_activation
    from opifex.neural.base import QuantumMLP, StandardMLP

    # Test StandardMLP creation
    key = jax.random.key(42)
    rngs = nnx.Rngs(params=key)

    mlp = StandardMLP(layer_sizes=[10, 64, 32, 1], activation="relu", rngs=rngs)

    # Test forward pass
    batch_size = 8
    input_dim = 10
    x = jax.random.normal(key, (batch_size, input_dim))
    output = mlp(x)

    assert output.shape == (batch_size, 1), f"Expected shape (8, 1), got {output.shape}"

    # Test QuantumMLP
    quantum_mlp = QuantumMLP(layer_sizes=[10, 32, 16, 1], activation="tanh", rngs=rngs)

    quantum_output = quantum_mlp(x)
    assert quantum_output.shape == (batch_size, 1), "Quantum MLP output shape incorrect"

    # Test activation functions
    relu_fn = get_activation("relu")
    test_input = jnp.array([-1.0, 0.0, 1.0])
    relu_output = relu_fn(test_input)
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(relu_output, expected), "ReLU activation incorrect"


def test_training_infrastructure():
    """Test training infrastructure examples."""
    from opifex.training.basic_trainer import BasicTrainer, TrainerConfig

    # Test trainer configuration
    config = TrainerConfig(learning_rate=0.001, num_epochs=10, batch_size=32)
    assert config is not None, "Trainer config should be created"

    # Test basic trainer creation
    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key)

    from opifex.neural.base import StandardMLP

    model = StandardMLP(layer_sizes=[32, 16, 1], activation="relu", rngs=rngs)

    trainer = BasicTrainer(model=model, config=config, rngs=rngs)
    assert trainer is not None, "Basic trainer should be created"


def test_activation_functions():
    """Test activation function library."""
    from opifex.neural.activations import (
        list_available_activations,
        quantum_tanh,
        sin_activation,
    )

    # Test activation registry
    available = list_available_activations()
    assert len(available) > 10, "Should have many activation functions"
    assert "relu" in available, "ReLU should be available"
    assert "sin_activation" in available, "Sin activation should be available"

    # Test scientific activations
    x = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    sin_output = sin_activation(x)
    expected = jnp.sin(x)
    assert jnp.allclose(sin_output, expected), "Sin activation should match jnp.sin"

    # Test quantum activation
    quantum_output = quantum_tanh(x)
    assert quantum_output.dtype == jnp.float64, "Quantum activation should be float64"


def test_comprehensive_workflow():
    """Test a complete Opifex workflow."""
    from opifex.core.problems import create_pde_problem
    from opifex.neural.base import StandardMLP

    # Create a simple PDE problem
    domain = {"x": (0.0, 1.0)}

    def simple_equation(x, u, u_derivatives):
        """Simple 1D diffusion equation."""
        return u_derivatives["u_t"] - 0.1 * u_derivatives["u_xx"]

    _ = create_pde_problem(
        domain=domain,
        equation=simple_equation,
        boundary_conditions={"type": "dirichlet", "value": 0.0},
        initial_conditions={"type": "gaussian", "center": 0.5, "width": 0.1},
    )

    # Create a neural network
    key = jax.random.key(123)
    rngs = nnx.Rngs(params=key)

    model = StandardMLP(layer_sizes=[1, 32, 32, 1], activation="tanh", rngs=rngs)

    # Test forward pass with sample data
    batch_size = 16
    spatial_dim = 1
    x = jax.random.uniform(key, (batch_size, spatial_dim))
    output = model(x)

    assert output.shape == (batch_size, 1), "Model output shape incorrect"
    assert jnp.isfinite(output).all(), "Model output should be finite"


def test_jax_integration():
    """Test JAX integration and automatic differentiation."""
    import jax

    # Test basic JAX operations
    def test_function(x):
        return jnp.sum(x**2)

    x = jnp.array([1.0, 2.0, 3.0])

    # Test function evaluation
    result = test_function(x)
    expected = 14.0  # 1 + 4 + 9
    assert jnp.isclose(result, expected), "Function evaluation incorrect"

    # Test gradient computation
    grad_fn = jax.grad(test_function)
    gradient = grad_fn(x)
    expected_grad = 2 * x  # derivative of x^2 is 2x
    assert jnp.allclose(gradient, expected_grad), "Gradient computation incorrect"

    # Test JIT compilation
    jit_fn = jax.jit(test_function)
    jit_result = jit_fn(x)
    assert jnp.isclose(jit_result, result), "JIT result should match regular result"


def test_flax_nnx_integration():
    """Test FLAX NNX integration."""
    from flax import nnx

    # Test basic NNX module
    class SimpleModule(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.linear = nnx.Linear(10, 5, rngs=rngs)

        def __call__(self, x):
            return nnx.relu(self.linear(x))

    key = jax.random.key(456)
    rngs = nnx.Rngs(params=key)
    module = SimpleModule(rngs=rngs)

    # Test forward pass
    x = jax.random.normal(key, (8, 10))
    output = module(x)
    assert output.shape == (8, 5), "Module output shape incorrect"

    # Test gradient computation
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(module, x)
    assert jnp.isfinite(loss), "Loss should be finite"
    assert grads is not None, "Gradients should be computed"


def test_environment_configuration():
    """Test environment and configuration."""
    # Test JAX configuration
    devices = jax.devices()
    assert len(devices) > 0, "Should have at least one device"

    # Test precision settings
    x32 = jnp.float32(1.0)
    x64 = jnp.float64(1.0)
    assert x32.dtype == jnp.float32, "Float32 precision"
    assert x64.dtype == jnp.float64, "Float64 precision"


def main() -> None:
    """Main function to run all documentation example tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify Opifex documentation examples")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--specific", type=str, help="Run specific test only")

    args = parser.parse_args()

    print("🧪 Opifex Documentation Examples Verification")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 60)

    tester = OpifexExampleTester(verbose=args.verbose)

    # Define all tests
    tests = [
        ("Problem Creation", test_problem_creation),
        ("Geometry Operations", test_geometry_operations),
        ("Boundary Conditions", test_boundary_conditions),
        ("Neural Networks", test_neural_networks),
        ("Training Infrastructure", test_training_infrastructure),
        ("Activation Functions", test_activation_functions),
        ("Comprehensive Workflow", test_comprehensive_workflow),
        ("JAX Integration", test_jax_integration),
        ("FLAX NNX Integration", test_flax_nnx_integration),
        ("Environment Configuration", test_environment_configuration),
    ]

    # Run specific test if requested
    if args.specific:
        matching_tests = [
            (name, func)
            for name, func in tests
            if args.specific.lower() in name.lower()
        ]
        if not matching_tests:
            print(f"❌ No tests found matching '{args.specific}'")
            sys.exit(1)
        tests = matching_tests

    # Run all tests
    print(f"\nRunning {len(tests)} documentation example tests...\n")

    for test_name, test_func in tests:
        tester.test(test_name, test_func)

    # Print summary and exit
    success = tester.print_summary()

    if success:
        print("\n🎉 All documentation examples work correctly!")
        print("   Opifex documentation is validated and up-to-date.")
    else:
        print("\n⚠️  Some documentation examples failed.")
        print("   Please check the documentation and fix the issues.")
        print("   Run with --verbose for detailed error information.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
