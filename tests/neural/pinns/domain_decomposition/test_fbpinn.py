"""Tests for FBPINN (Finite Basis Physics-Informed Neural Network).

TDD: These tests define the expected behavior for FBPINN with window functions.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain


class TestWindowFunctions:
    """Test window function implementations."""

    def test_cosine_window_at_center(self):
        """Window should be 1.0 at subdomain center."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import CosineWindow

        subdomain = Subdomain(id=0, bounds=jnp.array([[0.0, 1.0]]))
        window = CosineWindow(subdomain)

        # At center (0.5), window should be maximum (1.0)
        center = jnp.array([[0.5]])
        value = window(center)
        assert jnp.allclose(value, 1.0, atol=1e-5)

    def test_cosine_window_at_boundary(self):
        """Window should be 0.0 at subdomain boundary."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import CosineWindow

        subdomain = Subdomain(id=0, bounds=jnp.array([[0.0, 1.0]]))
        window = CosineWindow(subdomain)

        # At boundary (0.0 or 1.0), window should be minimum (0.0)
        boundary = jnp.array([[0.0], [1.0]])
        values = window(boundary)
        assert jnp.allclose(values, 0.0, atol=1e-5)

    def test_cosine_window_smooth_transition(self):
        """Window should decrease smoothly from center to boundary."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import CosineWindow

        subdomain = Subdomain(id=0, bounds=jnp.array([[0.0, 1.0]]))
        window = CosineWindow(subdomain)

        # Sample points from center to boundary
        x = jnp.linspace(0.5, 1.0, 11).reshape(-1, 1)
        values = window(x)

        # Values should be monotonically decreasing
        diffs = jnp.diff(values)
        assert jnp.all(diffs <= 1e-6)  # Allow small numerical error

    def test_cosine_window_2d(self):
        """Window should work in 2D domains."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import CosineWindow

        subdomain = Subdomain(id=0, bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]))
        window = CosineWindow(subdomain)

        # At center (0.5, 0.5), window should be maximum
        center = jnp.array([[0.5, 0.5]])
        value = window(center)
        assert jnp.allclose(value, 1.0, atol=1e-5)

        # At corner (0.0, 0.0), window should be minimum
        corner = jnp.array([[0.0, 0.0]])
        value = window(corner)
        assert jnp.allclose(value, 0.0, atol=1e-5)

    def test_gaussian_window_at_center(self):
        """Gaussian window should be 1.0 at subdomain center."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import GaussianWindow

        subdomain = Subdomain(id=0, bounds=jnp.array([[0.0, 1.0]]))
        window = GaussianWindow(subdomain, sigma=0.3)

        center = jnp.array([[0.5]])
        value = window(center)
        assert jnp.allclose(value, 1.0, atol=1e-5)

    def test_gaussian_window_decay(self):
        """Gaussian window should decay away from center."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import GaussianWindow

        subdomain = Subdomain(id=0, bounds=jnp.array([[0.0, 1.0]]))
        window = GaussianWindow(subdomain, sigma=0.3)

        center = jnp.array([[0.5]])
        away = jnp.array([[0.8]])

        center_val = window(center)
        away_val = window(away)

        assert center_val > away_val

    def test_partition_of_unity_1d(self):
        """Window functions should both be non-zero in overlap interior."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import CosineWindow

        # Overlapping subdomains
        sub1 = Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]]))
        sub2 = Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]]))

        window1 = CosineWindow(sub1)
        window2 = CosineWindow(sub2)

        # In the INTERIOR of the overlap region (0.4, 0.6), both windows are non-zero
        # Note: At x=0.4, sub2's window is 0 (boundary); at x=0.6, sub1's window is 0
        # So we test the interior point at x=0.5
        interior_point = jnp.array([[0.5]])
        w1 = window1(interior_point)
        w2 = window2(interior_point)

        # Both windows should be non-zero at the overlap interior
        assert w1 > 0
        assert w2 > 0

        # The sum should be > 0 (partition of unity is enforced via normalization)
        assert (w1 + w2) > 0


class TestFBPINNConfig:
    """Test FBPINN configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINNConfig

        config = FBPINNConfig()
        assert config.window_type == "cosine"
        assert config.normalize_windows is True
        assert config.overlap_factor > 0

    def test_custom_window_type(self):
        """Should accept custom window type."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINNConfig

        config = FBPINNConfig(window_type="gaussian")
        assert config.window_type == "gaussian"

    def test_custom_overlap_factor(self):
        """Should accept custom overlap factor."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINNConfig

        config = FBPINNConfig(overlap_factor=0.3)
        assert config.overlap_factor == 0.3


class TestFBPINNCreation:
    """Test FBPINN model creation."""

    def test_create_fbpinn(self):
        """Should create FBPINN with subdomains and window functions."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]
        interfaces = []

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16, 16],
            rngs=nnx.Rngs(0),
        )

        assert model is not None
        assert len(model.subdomains) == 2

    def test_create_with_custom_config(self):
        """Should accept custom FBPINN configuration."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN, FBPINNConfig

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]
        config = FBPINNConfig(window_type="gaussian", overlap_factor=0.3)

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            config=config,
            rngs=nnx.Rngs(0),
        )

        assert model.config.window_type == "gaussian"

    def test_window_functions_created(self):
        """Should create window function for each subdomain."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        assert len(model.windows) == 2


class TestFBPINNForward:
    """Test FBPINN forward pass."""

    def test_forward_pass(self):
        """Should compute forward pass with window-weighted outputs."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.2], [0.5], [0.8]])
        y = model(x)

        assert y.shape == (3, 1)
        assert jnp.isfinite(y).all()

    def test_forward_pass_2d(self):
        """Should handle 2D domains."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6], [0.0, 1.0]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0], [0.0, 1.0]])),
        ]

        model = FBPINN(
            input_dim=2,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[32, 16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]])
        y = model(x)

        assert y.shape == (3, 1)
        assert jnp.isfinite(y).all()

    def test_smooth_output_in_overlap(self):
        """Output should be smooth in overlap region due to window blending."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        # Sample points in overlap region
        x = jnp.linspace(0.4, 0.6, 20).reshape(-1, 1)
        y = model(x)

        # Check that output is continuous (no large jumps)
        diffs = jnp.abs(jnp.diff(y, axis=0))
        max_diff = jnp.max(diffs)

        # The max difference should be reasonable (no discontinuities)
        assert max_diff < 1.0  # Reasonable bound for smooth transition


class TestFBPINNWindowWeights:
    """Test FBPINN window weight computation."""

    def test_window_weights_sum_to_one(self):
        """Window weights should sum to 1 when normalized for interior points."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN, FBPINNConfig

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]
        config = FBPINNConfig(normalize_windows=True)

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            config=config,
            rngs=nnx.Rngs(0),
        )

        # Test interior points only (excluding domain boundaries where windows are 0)
        # At x=0.0 and x=1.0, both windows are 0 (subdomain boundaries)
        x = jnp.linspace(0.1, 0.9, 17).reshape(-1, 1)
        weights = model.compute_window_weights(x)

        # Sum should be 1.0 for interior points (after normalization)
        weight_sums = jnp.sum(weights, axis=-1)
        assert jnp.allclose(weight_sums, 1.0, atol=1e-5)

    def test_window_weights_shape(self):
        """Window weights should have correct shape."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.2], [0.5], [0.8]])
        weights = model.compute_window_weights(x)

        assert weights.shape == (3, 2)  # (batch, num_subdomains)


class TestFBPINNGradients:
    """Test FBPINN gradient computation."""

    def test_gradient_computation(self):
        """Should compute gradients for training."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.2], [0.5], [0.8]])
        y_target = jnp.array([[1.0], [2.0], [1.5]])

        def loss_fn(model):
            y = model(x)
            return jnp.mean((y - y_target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)

        assert jnp.isfinite(loss)
        # Grads should exist and be finite
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0

    def test_jit_compatible(self):
        """Forward pass should be JIT compatible."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=[],
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        @nnx.jit
        def forward(model, x):
            return model(x)

        x = jnp.array([[0.5]])
        y = forward(model, x)

        assert y.shape == (1, 1)
        assert jnp.isfinite(y).all()


class TestFBPINNWithInterfaces:
    """Test FBPINN with interface conditions."""

    def test_with_interface_points(self):
        """Should work with interfaces defined (for consistency checking)."""
        from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN

        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
            Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = FBPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.5]])
        y = model(x)

        assert y.shape == (1, 1)
        assert jnp.isfinite(y).all()
