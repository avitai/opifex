import jax
import jax.numpy as jnp
from flax import nnx

from opifex.training.multilevel.cascade_training import CascadeTrainer
from opifex.training.multilevel.multilevel_adam import MultilevelAdam


class DummyModel(nnx.Module):
    def __init__(self, size, rngs):
        self.size = size
        self.param = nnx.Param(jax.random.uniform(rngs.params(), (size,)))

    def __call__(self, x):
        return x * jnp.mean(self.param.value)


def test_cascade_trainer_lifecycle():
    rngs = nnx.Rngs(0)
    # Hierarchy: dim 2 -> dim 4
    model_coarse = DummyModel(2, rngs)
    model_fine = DummyModel(4, rngs)
    hierarchy = [model_coarse, model_fine]

    # Prolongation fn: pad with zeros
    def prolongate_fn(coarse, fine):
        # simplistic copy
        fine.param.value = fine.param.value.at[:2].set(coarse.param.value)
        return fine

    optimizer = MultilevelAdam(learning_rate=0.1)

    trainer = CascadeTrainer(
        hierarchy=hierarchy, optimizer=optimizer, prolongate_fn=prolongate_fn
    )

    assert trainer.current_level == 0
    assert trainer.get_current_model().size == 2

    # Train step simulation
    # trainer.train_epoch(...) would be integration
    # Let's just check advancing

    # Modify coarse param to see if it carries over
    trainer.get_current_model().param.value = jnp.array([10.0, 10.0])

    advanced = trainer.advance_level()
    assert advanced
    assert trainer.current_level == 1
    assert trainer.get_current_model().size == 4

    # Check prolongation
    fine_vals = trainer.get_current_model().param.value
    assert jnp.allclose(fine_vals[:2], jnp.array([10.0, 10.0]))

    # Check optimization state resize
    # Opt state should have been resized
    # accessing optimizer internal state might be tricky but we check for errors during update

    # Mock update
    grads = nnx.grad(lambda m: jnp.sum(m.param.value))(trainer.get_current_model())
    trainer.optimizer.update(trainer.get_current_model(), grads)

    # Try advancing again (should fail/return False)
    assert not trainer.advance_level()


def test_cascade_trainer_custom_optimizer_resize():
    # Verify that we can pass a transition function for the optimizer state
    pass
