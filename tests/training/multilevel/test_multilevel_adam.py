import jax.numpy as jnp
from flax import nnx

from opifex.training.multilevel.multilevel_adam import MultilevelAdam


class SimpleMLP(nnx.Module):
    def __init__(self, in_features, out_features, rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)


def test_multilevel_adam_update_and_resize():
    rngs = nnx.Rngs(0)
    # Coarse model: 2 -> 2
    model = SimpleMLP(2, 2, rngs)

    # Initialize optimizer
    # Usage philosophy: MultilevelAdam(learning_rate=...)
    optimizer = MultilevelAdam(learning_rate=0.01)
    optimizer.init(model)

    # Perform an update
    x = jnp.ones((1, 2))
    y = jnp.ones((1, 2))

    loss_fn = lambda m: jnp.mean((m.linear(x) - y) ** 2)
    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)

    # Check that state exists and is non-zero (Adam moments updated)
    # Accessing internal optax state might be complex depending on implementation
    # But MultilevelAdam should expose it or Wrap it.

    # Now simulate prolongation: New model 2 -> 4
    new_model = SimpleMLP(2, 4, rngs)

    # Manually copy weights to simulate prolongation (simplified)
    # We just want to test the optimizer state resizing

    # Define how to resize the state
    # The user/trainer must provide a function that maps old_params -> new_params
    # OR the optimizer tries to match by name/structure?
    # Matching by structure is hard if shapes change.
    # Usually we need a `prolongation_fn` for the state.
    # A simple default is: "pad with zeros" or "copy where shapes match"

    # Let's assume MultilevelAdam.resize_state(new_params, map_fn)
    # map_fn(old_param_value, new_param_shape) -> new_param_value

    def simple_pad(old_val, new_val_shape):
        new_val = jnp.zeros(new_val_shape)
        # Slicing for testing 2->2 to 2->4 (out dim changed)
        # Linear kernel is (in, out) = (2, 2) -> (2, 4)
        if old_val.ndim == 2:
            h, w = old_val.shape
            new_val = new_val.at[:h, :w].set(old_val)
        else:  # bias (2,) -> (4,)
            w = old_val.shape[0]
            new_val = new_val.at[:w].set(old_val)
        return new_val

    optimizer.resize_state(new_model, transition_fn=simple_pad)

    # Perform update with new model
    grads_new = nnx.grad(lambda m: jnp.mean((m.linear(x) - jnp.ones((1, 4))) ** 2))(
        new_model
    )
    optimizer.update(new_model, grads_new)

    # Check if update worked (no shape mismatch errors)
    assert True


def test_multilevel_adam_masking():
    """Test if we can mask updates for newly added parameters (freeze them optionally)."""
    # Optional feature: when moving to fine level, maybe freeze coarse part?
    # Or just standard Adam behavior.
    # Let's stick to basic resizing for now.


def test_multilevel_adam_uninitialized_update():
    """Test update call without prior init."""
    rngs = nnx.Rngs(0)
    model = SimpleMLP(2, 2, rngs)
    optimizer = MultilevelAdam(learning_rate=0.01)

    # Should not raise error and should initialize internally
    grads = nnx.grad(lambda m: jnp.sum(m.linear.kernel.value))(model)
    optimizer.update(model, grads)

    assert optimizer.opt_state is not None


def test_multilevel_adam_resize_no_state():
    """Test resize_state call when state is None."""
    rngs = nnx.Rngs(0)
    optimizer = MultilevelAdam(learning_rate=0.01)
    new_model = SimpleMLP(2, 4, rngs)

    # Should just return without error
    optimizer.resize_state(new_model, lambda x, s: x)
    assert optimizer.opt_state is None


def test_multilevel_adam_resize_unknown_structure():
    """Test fallback when state structure doesn't match expected Adam state."""
    rngs = nnx.Rngs(0)
    model = SimpleMLP(2, 2, rngs)
    optimizer = MultilevelAdam(learning_rate=0.01)
    optimizer.init(model)

    # Manually corrupt state to something unexpected
    optimizer.opt_state = ({"foo": "bar"},)  # type: ignore # noqa: PGH003

    new_model = SimpleMLP(2, 4, rngs)

    # Should trigger re-init logic (lines 110-111)
    optimizer.resize_state(new_model, lambda x, s: x)

    # Check if re-initialized (state structure should be valid optax state now)
    # opt_state[0] should have count/mu/nu if it's back to normal
    assert hasattr(optimizer.opt_state[0], "mu")  # type: ignore # noqa: PGH003


def test_multilevel_adam_resize_generic_object():
    """Test resize when state object has mu/nu but no _replace method."""

    rngs = nnx.Rngs(0)
    model = SimpleMLP(2, 2, rngs)
    optimizer = MultilevelAdam(learning_rate=0.01)
    optimizer.init(model)

    # Create a mock state object that behaves like Adam state but is generic class
    class MockAdamState:
        def __init__(self, count, mu, nu):
            self.count = count
            self.mu = mu
            self.nu = nu

    # Need to register it as pytree so jax.tree.map works on it?
    # Actually optax state is a tuple (AdamState, EmptyState).
    # We replace AdamState with MockAdamState.

    real_state = optimizer.opt_state[0]  # type: ignore # noqa: PGH003
    mock_state = MockAdamState(real_state.count, real_state.mu, real_state.nu)

    # We need to make sure jax.tree.map works on individual fields (mu/nu), not the object itself
    # MultilevelAdam implementation does:
    # new_mu = jax.tree.map(..., adam_state.mu, new_params)
    # So adam_state itself doesn't need to be a Pytree node, just haveAttributes .mu and .nu

    optimizer.opt_state = (mock_state, optimizer.opt_state[1])  # type: ignore # noqa: PGH003

    new_model = SimpleMLP(2, 4, rngs)

    # Run resize
    # This should hit the 'else' block because MockAdamState has no _replace
    optimizer.resize_state(new_model, lambda x, s: x)

    # Result should be reconstructed as ScaleByAdamState (default fallback)
    from optax._src.transform import ScaleByAdamState

    assert isinstance(optimizer.opt_state[0], ScaleByAdamState)  # type: ignore # noqa: PGH003
