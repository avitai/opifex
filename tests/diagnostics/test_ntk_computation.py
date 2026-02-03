import jax
import jax.numpy as jnp
from flax import nnx

from opifex.diagnostics.ntk_computation import compute_gradient_jacobian, compute_ntk


class SimpleModel(nnx.Module):
    def __init__(self, key):
        self.linear = nnx.Linear(2, 1, rngs=nnx.Rngs(key))

    def __call__(self, x):
        return self.linear(x)


def test_compute_ntk_shape_and_properties():
    key = jax.random.key(0)
    model = SimpleModel(key=0)

    # Batch size 4, input dim 2
    x = jax.random.normal(key, (4, 2))

    # Compute NTK
    ntk = compute_ntk(model, x)

    # Check shape: (batch, batch)
    assert ntk.shape == (4, 4)

    # Check symmetry: NTK = NTK.T
    assert jnp.allclose(ntk, ntk.T, atol=1e-5)

    # Check positive semi-definiteness (diagonal >= 0)
    assert jnp.all(jnp.diag(ntk) >= -1e-5)


def test_ntk_consistency():
    """Verify NTK matches J * J.T / m formula."""
    key = jax.random.key(1)
    model = SimpleModel(key=1)
    x = jax.random.normal(key, (3, 2))

    ntk = compute_ntk(model, x)

    # Manually compute Gradient Jacobian
    # J = jac_fn(model, x) # Result is PyTree of gradients (unused)

    # Flatten Jacobian manually for comparison is hard due to PyTree structure
    # So we trust the helper function compute_gradient_jacobian if we write it

    # Check that manual J*J.T matches our NTK function
    J_flat = compute_gradient_jacobian(model, x)
    expected_ntk = (J_flat @ J_flat.T) / x.shape[0]

    assert jnp.allclose(ntk, expected_ntk, atol=1e-5)
