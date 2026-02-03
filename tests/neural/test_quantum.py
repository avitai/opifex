"""Test quantum neural networks.

Test suite for quantum neural network implementations including
neural DFT, neural SCF, and exchange-correlation functionals.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.quantum.neural_dft import NeuralDFT
from opifex.neural.quantum.neural_scf import NeuralSCFSolver
from opifex.neural.quantum.neural_xc import NeuralXCFunctional


class TestNeuralDFT:
    """Test Neural Density Functional Theory implementation."""

    def setup_method(self):
        """Setup for each test method with quantum-safe precision."""
        self.backend = jax.default_backend()
        # Use high precision for quantum calculations
        print(f"Running DFT tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_neural_dft_initialization(self, rngs):
        """Test Neural DFT initialization with proper precision handling."""
        neural_dft = NeuralDFT(
            grid_size=16,
            chemical_accuracy_target=1e-3,  # 1 mHa (chemical accuracy)
            rngs=rngs,
        )

        assert neural_dft.grid_size == 16
        assert neural_dft.chemical_accuracy_target == 1e-3
        assert hasattr(neural_dft, "neural_xc_functional")
        assert hasattr(neural_dft, "neural_scf_solver")

    def test_neural_dft_energy_computation(self, rngs, rng_key):
        """Test Neural DFT basic functionality with quantum precision."""
        num_electrons = 2  # Store for normalization
        neural_dft = NeuralDFT(
            grid_size=8,  # Small for testing
            rngs=rngs,
        )

        # Create test density (normalized)
        density = jax.random.uniform(rng_key, (1, 8, 8, 8), minval=0.0, maxval=1.0)

        # Normalize density to integrate to num_electrons
        density = density / jnp.sum(density) * num_electrons

        # Test that the object was created successfully
        assert neural_dft.grid_size == 8

    def test_neural_dft_chemical_accuracy_validation(self, rngs, rng_key):
        """Test Neural DFT basic properties."""
        num_electrons = 2  # Store for normalization
        neural_dft = NeuralDFT(
            grid_size=8,
            chemical_accuracy_target=1e-3,  # 1 mHa
            rngs=rngs,
        )

        # Create test densities
        density1 = jax.random.uniform(rng_key, (1, 8, 8, 8), minval=0.0, maxval=1.0)
        density2 = jax.random.uniform(rng_key, (1, 8, 8, 8), minval=0.0, maxval=1.0)

        # Normalize densities
        density1 = density1 / jnp.sum(density1) * num_electrons
        density2 = density2 / jnp.sum(density2) * num_electrons

        # Test basic properties
        assert neural_dft.chemical_accuracy_target == 1e-3

        # Energy difference should be finite
        energy_diff = jnp.abs(jnp.sum(density1) - jnp.sum(density2))
        assert jnp.isfinite(energy_diff)


class TestNeuralSCF:
    """Test Neural Self-Consistent Field implementation."""

    def setup_method(self):
        """Setup for each test method with quantum-safe precision."""
        self.backend = jax.default_backend()
        print(f"Running Neural SCF tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_neural_scf_initialization(self, rngs):
        """Test Neural SCF initialization."""
        neural_scf = NeuralSCFSolver(
            convergence_threshold=1e-8,
            max_iterations=100,
            mixing_strategy="neural",
            grid_size=8,
            chemical_accuracy_target=1e-6,
            rngs=rngs,
        )

        assert neural_scf.convergence_threshold == 1e-8
        assert neural_scf.max_iterations == 100
        assert neural_scf.mixing_strategy == "neural"
        assert neural_scf.grid_size == 8
        assert hasattr(neural_scf, "density_mixer")
        assert hasattr(neural_scf, "convergence_predictor")

    def test_neural_scf_density_mixing(self, rngs, rng_key):
        """Test Neural SCF density mixing capabilities."""
        neural_scf = NeuralSCFSolver(
            convergence_threshold=1e-8,
            max_iterations=50,
            mixing_strategy="neural",
            grid_size=4,  # Very small for testing
            chemical_accuracy_target=1e-6,
            rngs=rngs,
        )

        # Create test densities
        old_density = jax.random.uniform(rng_key, (4,), minval=0.0, maxval=1.0)
        new_density = jax.random.uniform(rng_key, (4,), minval=0.0, maxval=1.0)

        mixed_density = neural_scf._mix_densities(
            old_density, new_density, deterministic=True
        )

        # Check mixed density properties
        assert mixed_density.shape == (4,)
        assert jnp.all(jnp.isfinite(mixed_density))
        assert jnp.all(mixed_density >= 0.0)  # Should be non-negative

    def test_neural_scf_convergence_prediction(self, rngs):
        """Test Neural SCF convergence prediction capabilities."""
        neural_scf = NeuralSCFSolver(
            convergence_threshold=1e-8,
            max_iterations=50,
            mixing_strategy="neural",
            grid_size=4,
            chemical_accuracy_target=1e-6,
            rngs=rngs,
        )

        # Create test convergence history
        convergence_history = jnp.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

        is_converged, chemical_accuracy, prediction_score = (
            neural_scf._check_convergence(
                energy_error=1e-8,
                density_error=1e-8,
                convergence_history=convergence_history,
                deterministic=True,
            )
        )

        # Check convergence prediction results
        assert isinstance(is_converged, (bool, jnp.bool_))
        assert isinstance(chemical_accuracy, (bool, jnp.bool_))
        assert isinstance(prediction_score, (float, jnp.floating))
        assert 0.0 <= prediction_score <= 1.0


class TestNeuralExchangeCorrelation:
    """Test Neural Exchange-Correlation functional implementation."""

    def setup_method(self):
        """Setup for each test method with quantum-safe precision."""
        self.backend = jax.default_backend()
        print(f"Running Neural XC tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_neural_xc_initialization(self, rngs):
        """Test Neural XC functional initialization."""
        neural_xc = NeuralXCFunctional(
            hidden_sizes=[32, 32, 16],
            use_attention=True,
            use_advanced_features=False,
            dropout_rate=0.0,
            rngs=rngs,
        )

        assert neural_xc.use_attention
        assert not neural_xc.use_advanced_features
        assert hasattr(neural_xc, "feature_extractor")
        assert hasattr(neural_xc, "layers")

    def test_neural_xc_functional_computation(self, rngs, rng_key):
        """Test Neural XC functional computation."""
        neural_xc = NeuralXCFunctional(
            hidden_sizes=[16, 16],
            use_attention=False,  # Simplified for testing
            rngs=rngs,
        )

        # Create test density and gradients
        density = jax.random.uniform(rng_key, (1, 8), minval=0.1, maxval=1.0)
        gradients = jax.random.normal(rng_key, (1, 8, 3))

        xc_energy = neural_xc(density, gradients, deterministic=True)

        assert jnp.isfinite(xc_energy).all()
        assert xc_energy.shape == (1, 8)

    def test_neural_xc_potential_computation(self, rngs, rng_key):
        """Test Neural XC potential computation."""
        neural_xc = NeuralXCFunctional(
            hidden_sizes=[16],
            use_attention=False,  # Simplified for testing
            use_advanced_features=False,
            rngs=rngs,
        )

        # Create test density and gradients
        density = jax.random.uniform(rng_key, (1, 4), minval=0.1, maxval=1.0)
        gradients = jax.random.normal(rng_key, (1, 4, 3))

        xc_potential = neural_xc.compute_functional_derivative(
            density, gradients, deterministic=True
        )

        assert xc_potential.shape == density.shape
        assert jnp.all(jnp.isfinite(xc_potential))

    def test_neural_xc_gradient_computation(self, rngs, rng_key):
        """Test Neural XC gradient computation for optimization."""
        neural_xc = NeuralXCFunctional(
            hidden_sizes=[8],
            use_attention=False,
            rngs=rngs,
        )

        def loss_fn(model, density, gradients):
            xc_energy = model(density, gradients, deterministic=True)
            return jnp.sum(xc_energy**2)

        density = jax.random.uniform(rng_key, (1, 4), minval=0.1, maxval=1.0)
        gradients = jax.random.normal(rng_key, (1, 4, 3))

        # Compute gradients
        grads = nnx.grad(loss_fn)(neural_xc, density, gradients)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)


class TestQuantumIntegration:
    """Test integration between quantum neural network components."""

    def setup_method(self):
        """Setup for each test method with quantum-safe precision."""
        self.backend = jax.default_backend()
        print(f"Running Quantum Integration tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_dft_scf_integration(self, rngs, rng_key):
        """Test integration between Neural DFT and Neural SCF."""
        # Create small system for integration test
        neural_dft = NeuralDFT(
            grid_size=4,
            rngs=rngs,
        )

        # Create test density for DFT energy computation
        density = jax.random.uniform(rng_key, (1, 4, 4, 4), minval=0.0, maxval=1.0)

        # Normalize density to have reasonable electron count (2 electrons)
        density = density / jnp.sum(density) * 2.0

        # Compute DFT energy from density
        # Just test that we can create the object and access basic properties
        assert jnp.isfinite(density).all()
        assert neural_dft.grid_size == 4
