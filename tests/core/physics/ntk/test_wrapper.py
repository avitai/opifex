"""Tests for JAX-native NTK wrapper utilities.

TDD: These tests define the expected behavior for NTK computation with NNX models.
"""

import warnings

import jax
import jax.numpy as jnp
from flax import nnx


class TestNTKWrapperCreation:
    """Test NTK wrapper creation."""

    def test_create_ntk_fn_from_nnx(self):
        """Should create NTK function from NNX model."""
        from opifex.core.physics.ntk.wrapper import create_ntk_fn_from_nnx

        # Simple NNX model
        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs) -> None:
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        ntk_fn = create_ntk_fn_from_nnx(model)

        assert ntk_fn is not None
        assert callable(ntk_fn)
        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        ntk = ntk_fn(x)
        assert ntk.shape == (2, 2)
        assert jnp.all(jnp.isfinite(ntk))

    def test_create_ntk_fn_with_deep_model(self):
        """Should work with deeper neural networks."""
        from opifex.core.physics.ntk.wrapper import create_ntk_fn_from_nnx

        class DeepModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs) -> None:
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
        x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ntk = ntk_fn(x)
        assert ntk.shape == (3, 3)
        assert jnp.all(jnp.isfinite(ntk))


class TestEmpiricalNTK:
    """Test empirical NTK computation."""

    def test_compute_empirical_ntk(self):
        """Should compute empirical NTK matrix."""
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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
            def __init__(self, rngs: nnx.Rngs) -> None:
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


class _BatchNormModel(nnx.Module):
    """Linear, BatchNorm, tanh, Linear: parameters plus running statistics."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(3, 8, rngs=rngs)
        self.norm = nnx.BatchNorm(8, rngs=rngs)
        self.out = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.out(nnx.tanh(self.norm(self.hidden(x))))


class _DropoutModel(nnx.Module):
    """Linear, tanh, Dropout, Linear: parameters plus RNG state."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(3, 8, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.out = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.out(self.dropout(nnx.tanh(self.hidden(x))))


class TestJacobianParametersOnly:
    """compute_jacobian differentiates the model's nnx.Param state and nothing else."""

    def test_no_deprecation_warning(self) -> None:
        """No deprecated flax.nnx.State API is used."""
        from opifex.core.physics.ntk.wrapper import compute_jacobian

        model = _BatchNormModel(rngs=nnx.Rngs(0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_jacobian(model, jnp.ones((2, 3)))
        deprecations = [
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
        ]
        assert deprecations == []

    def test_jacobian_leaves_are_the_parameters(self) -> None:
        """The Jacobian has one entry per nnx.Param and none for running statistics."""
        from opifex.core.physics.ntk.wrapper import compute_jacobian

        model = _BatchNormModel(rngs=nnx.Rngs(0))
        jacobian = compute_jacobian(model, jnp.ones((2, 3)))
        jacobian_paths = {path for path, _ in nnx.to_flat_state(jacobian)}
        param_paths = {path for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))}
        assert jacobian_paths == param_paths

    def test_batchnorm_eval_ntk_holds_running_statistics_fixed(self) -> None:
        """In eval mode the running statistics are inputs to the network, not parameters.

        The reference differentiates a hand-written forward pass with respect to the two
        Linear layers and the BatchNorm scale and shift only. Over seeds 0 to 3 the two
        agree within 1.9e-06 absolute on entries up to 9.3; the tolerance sits above that.
        """
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        model = _BatchNormModel(rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(10), (6, 3))
        model(3.0 * x + 1.0)  # train mode moves the running statistics off their initial values
        model.eval()
        mean, var, epsilon = model.norm.mean[...], model.norm.var[...], model.norm.epsilon
        hidden_bias, scale, shift, out_bias = (
            model.hidden.bias,
            model.norm.scale,
            model.norm.bias,
            model.out.bias,
        )
        assert hidden_bias is not None
        assert scale is not None
        assert shift is not None
        assert out_bias is not None
        params = {
            "hidden_kernel": model.hidden.kernel[...],
            "hidden_bias": hidden_bias[...],
            "scale": scale[...],
            "shift": shift[...],
            "out_kernel": model.out.kernel[...],
            "out_bias": out_bias[...],
        }

        def forward(p: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
            hidden = inputs @ p["hidden_kernel"] + p["hidden_bias"]
            normed = (hidden - mean) / jnp.sqrt(var + epsilon) * p["scale"] + p["shift"]
            return jnp.tanh(normed) @ p["out_kernel"] + p["out_bias"]

        rows = jnp.concatenate(
            [
                leaf.reshape(x.shape[0], -1)
                for leaf in jax.tree.leaves(jax.jacrev(forward)(params, x))
            ],
            axis=1,
        )
        reference = rows @ rows.T

        ntk = compute_empirical_ntk(model, x)
        assert float(jnp.max(jnp.abs(ntk - reference))) < 1e-5

    def test_dropout_model_has_an_ntk(self) -> None:
        """RNG state is carried through the forward pass rather than differentiated."""
        from opifex.core.physics.ntk.wrapper import compute_empirical_ntk

        model = _DropoutModel(rngs=nnx.Rngs(0))
        ntk = compute_empirical_ntk(model, jax.random.normal(jax.random.key(1), (4, 3)))
        assert ntk.shape == (4, 4)
        assert jnp.all(jnp.isfinite(ntk))
