"""Tests for NTK wrapper with neural-tangents integration.

TDD: These tests define the expected behavior for NTK computation with NNX models.
"""

import jax.numpy as jnp
import pytest
from flax import nnx


# Check if neural-tangents is available and compatible
def _check_neural_tangents_available():
    import importlib.util

    if importlib.util.find_spec("neural_tangents") is None:
        return False
    try:
        # Check if it can be imported without compatibility issues
        import neural_tangents  # noqa: F401  # pyright: ignore[reportUnusedImport]

        return True
    except Exception:
        # Compatibility issues with JAX version
        return False


NEURAL_TANGENTS_AVAILABLE = _check_neural_tangents_available()


class TestNTKWrapperCreation:
    """Test NTK wrapper creation."""

    @pytest.mark.skipif(
        not NEURAL_TANGENTS_AVAILABLE,
        reason="neural-tangents not available or incompatible with JAX version",
    )
    def test_create_ntk_fn_from_nnx(self):
        """Should create NTK function from NNX model."""
        from opifex.core.physics.ntk.wrapper import create_ntk_fn_from_nnx

        # Simple NNX model
        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        ntk_fn = create_ntk_fn_from_nnx(model)

        assert ntk_fn is not None
        assert callable(ntk_fn)

    @pytest.mark.skipif(
        not NEURAL_TANGENTS_AVAILABLE,
        reason="neural-tangents not available or incompatible with JAX version",
    )
    def test_create_ntk_fn_with_deep_model(self):
        """Should work with deeper neural networks."""
        from opifex.core.physics.ntk.wrapper import create_ntk_fn_from_nnx

        class DeepModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.layers = nnx.List(
                    [
                        nnx.Linear(2, 32, rngs=rngs),
                        nnx.Linear(32, 16, rngs=rngs),
                        nnx.Linear(16, 1, rngs=rngs),
                    ]
                )

            def __call__(self, x):
                for layer in list(self.layers)[:-1]:
                    x = nnx.relu(layer(x))
                return list(self.layers)[-1](x)

        model = DeepModel(rngs=nnx.Rngs(0))
        ntk_fn = create_ntk_fn_from_nnx(model)

        assert ntk_fn is not None


class TestEmpiricalNTK:
    """Test empirical NTK computation."""

    def test_compute_empirical_ntk(self):
        """Should compute empirical NTK matrix."""
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        ntk = compute_empirical_ntk(model, x)

        # NTK should be (batch, batch) shaped
        assert ntk.shape == (3, 3)
        # NTK should be symmetric
        assert jnp.allclose(ntk, ntk.T, atol=1e-5)
        # NTK should be positive semi-definite (all eigenvalues >= 0)
        eigenvalues = jnp.linalg.eigvalsh(ntk)
        assert jnp.all(eigenvalues >= -1e-6)

    def test_compute_ntk_different_points(self):
        """Should compute NTK between different sets of points."""
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x1 = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        x2 = jnp.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])

        ntk = compute_empirical_ntk(model, x1, x2)

        # NTK should be (batch1, batch2) shaped
        assert ntk.shape == (2, 3)

    def test_ntk_finite_values(self):
        """NTK values should be finite."""
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        class DeepModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.layers = nnx.List(
                    [
                        nnx.Linear(2, 16, rngs=rngs),
                        nnx.Linear(16, 8, rngs=rngs),
                        nnx.Linear(8, 1, rngs=rngs),
                    ]
                )

            def __call__(self, x):
                for layer in list(self.layers)[:-1]:
                    x = nnx.tanh(layer(x))
                return list(self.layers)[-1](x)

        model = DeepModel(rngs=nnx.Rngs(0))
        x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        ntk = compute_empirical_ntk(model, x)

        assert jnp.all(jnp.isfinite(ntk))


class TestNTKConfig:
    """Test NTK configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.core.physics.ntk.wrapper import NTKConfig

        config = NTKConfig()
        assert config.implementation == 1  # Jacobian contraction
        assert config.trace_axes == ()
        assert config.diagonal_axes == ()

    def test_custom_implementation(self):
        """Should accept custom implementation setting."""
        from opifex.core.physics.ntk.wrapper import NTKConfig

        config = NTKConfig(implementation=2)
        assert config.implementation == 2


class TestNTKWrapper:
    """Test NTKWrapper class for NNX models."""

    def test_create_wrapper(self):
        """Should create wrapper for NNX model."""
        from opifex.core.physics.ntk.wrapper import NTKWrapper

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        wrapper = NTKWrapper(model)

        assert wrapper is not None
        assert wrapper.model is model

    def test_compute_ntk_via_wrapper(self):
        """Should compute NTK via wrapper."""
        from opifex.core.physics.ntk.wrapper import NTKWrapper

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        wrapper = NTKWrapper(model)

        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        ntk = wrapper.compute_ntk(x)

        assert ntk.shape == (2, 2)

    def test_wrapper_with_config(self):
        """Should accept custom configuration."""
        from opifex.core.physics.ntk.wrapper import NTKConfig, NTKWrapper

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = NTKConfig(implementation=2)
        wrapper = NTKWrapper(model, config=config)

        assert wrapper.config.implementation == 2


class TestJacobianComputation:
    """Test Jacobian computation utilities."""

    def test_compute_jacobian(self):
        """Should compute Jacobian of model output w.r.t. parameters."""
        from opifex.core.physics.ntk.wrapper import compute_jacobian

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x = jnp.array([[0.1, 0.2]])

        jacobian = compute_jacobian(model, x)

        # Jacobian should be a tree of arrays
        assert isinstance(jacobian, dict) or hasattr(jacobian, "__iter__")

    def test_jacobian_shape(self):
        """Jacobian should have correct shape relative to parameters."""
        from opifex.core.physics.ntk.wrapper import compute_jacobian, flatten_jacobian

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        jacobian = compute_jacobian(model, x)
        flat_jacobian = flatten_jacobian(jacobian)

        # Flat jacobian should be (batch * output_dim, num_params)
        assert flat_jacobian.ndim == 2


class TestNTKMultiOutput:
    """Test NTK computation for multi-output models."""

    def test_multi_output_ntk(self):
        """Should handle models with multiple outputs."""
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        class MultiOutputModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 3, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = MultiOutputModel(rngs=nnx.Rngs(0))
        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        ntk = compute_empirical_ntk(model, x)

        # For multi-output, NTK is (batch * out, batch * out)
        # or (batch, batch) if we sum over outputs
        assert ntk.shape[0] == ntk.shape[1]
        assert jnp.all(jnp.isfinite(ntk))
