"""The shared time-stepping driver: a fixed number of steps, saved on a static schedule.

Oracles are exact. Repeated multiplication by a constant has a closed form that float32
reproduces bit for bit when the reference multiplies in the same order, and a step that
adds its own index sums to a triangular number, which pins the index each step receives.
The step count is static so the trajectory is a ``lax.scan`` and differentiates in
reverse mode; the checkpointing is asserted in the traced program, not assumed.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from opifex.core.solver import integrate, Snapshots


FACTOR = np.float32(0.9)


def _scale(state: jax.Array, index: jax.Array) -> jax.Array:
    return state * FACTOR


def _repeated(initial: np.ndarray, steps: int) -> np.ndarray:
    value = initial.astype(np.float32)
    for _ in range(steps):
        value = value * FACTOR
    return value


def _count_index(state: jax.Array, index: jax.Array) -> jax.Array:
    return state + index


class TestTheFinalState:
    """By default the driver returns the state after the last step."""

    def test_it_is_the_step_applied_num_steps_times(self) -> None:
        initial = np.asarray([1.0, -2.0, 3.5], dtype=np.float32)
        final = integrate(_scale, jnp.asarray(initial), num_steps=12)
        assert final.shape == (1, 3)
        np.testing.assert_array_equal(np.asarray(final[-1]), _repeated(initial, 12))

    def test_each_step_receives_its_global_index(self) -> None:
        final = integrate(_count_index, jnp.asarray(0, jnp.int32), num_steps=10)
        assert int(final[-1]) == sum(range(10))

    def test_the_index_is_an_int32_array(self) -> None:
        seen: list[jax.Array] = []

        def record(state: jax.Array, index: jax.Array) -> jax.Array:
            seen.append(index)
            return state

        integrate(record, jnp.zeros(()), num_steps=3)
        assert seen[0].dtype == jnp.int32
        assert seen[0].shape == ()


class TestSnapshots:
    """A static schedule of evenly spaced saves, the last being the final state."""

    def test_count_saves_evenly_spaced_states(self) -> None:
        saved = integrate(
            _count_index, jnp.asarray(0, jnp.int32), num_steps=8, save=Snapshots(count=4)
        )
        np.testing.assert_array_equal(np.asarray(saved), [sum(range(n)) for n in (2, 4, 6, 8)])

    def test_the_initial_state_can_be_prepended(self) -> None:
        initial = np.asarray([2.0], dtype=np.float32)
        saved = integrate(
            _scale,
            jnp.asarray(initial),
            num_steps=6,
            save=Snapshots(count=3, include_initial=True),
        )
        assert saved.shape == (4, 1)
        for row, steps in enumerate((0, 2, 4, 6)):
            np.testing.assert_array_equal(np.asarray(saved[row]), _repeated(initial, steps))

    def test_a_pytree_state_is_saved_leaf_by_leaf(self) -> None:
        def step(state: dict[str, jax.Array], index: jax.Array) -> dict[str, jax.Array]:
            return {"u": state["u"] * FACTOR, "n": state["n"] + index}

        saved = integrate(
            step,
            {"u": jnp.ones(2), "n": jnp.asarray(0, jnp.int32)},
            num_steps=4,
            save=Snapshots(count=2),
        )
        assert saved["u"].shape == (2, 2)
        np.testing.assert_array_equal(np.asarray(saved["n"]), [1, 6])

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_count_below_one_is_refused(self, count: int) -> None:
        with pytest.raises(ValueError, match="count"):
            Snapshots(count=count)


class TestTheStepCountIsValidated:
    """Checked on the host, where the count is a Python int."""

    @pytest.mark.parametrize("num_steps", [0, -3])
    def test_a_count_below_one_is_refused(self, num_steps: int) -> None:
        with pytest.raises(ValueError, match="num_steps"):
            integrate(_scale, jnp.ones(1), num_steps=num_steps)

    def test_a_count_the_schedule_does_not_divide_is_refused(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            integrate(_scale, jnp.ones(1), num_steps=10, save=Snapshots(count=4))


class TestTransforms:
    """jit, vmap and reverse-mode grad, with the step checkpointed by default."""

    def test_jit_compiles_once_across_initial_states(self) -> None:
        traces: list[int] = []

        @jax.jit
        def run(initial: jax.Array) -> jax.Array:
            traces.append(1)
            return integrate(_scale, initial, num_steps=5)

        run(jnp.ones(3))
        run(jnp.full(3, 7.0, dtype=jnp.float32))  # same dtype and weak type
        assert len(traces) == 1

    def test_vmap_over_initial_states(self) -> None:
        initials = jnp.arange(6.0).reshape(3, 2)
        saved = jax.vmap(lambda u: integrate(_scale, u, num_steps=4))(initials)
        assert saved.shape == (3, 1, 2)
        np.testing.assert_allclose(
            np.asarray(saved[:, -1]), np.asarray(initials) * float(FACTOR) ** 4, rtol=1e-6
        )

    @pytest.mark.parametrize("checkpoint", [True, False])
    def test_grad_matches_the_closed_form(self, checkpoint: bool) -> None:
        def final(factor: jax.Array) -> jax.Array:
            def step(state: jax.Array, index: jax.Array) -> jax.Array:
                return state * factor

            saved = integrate(step, jnp.asarray(2.0), num_steps=7, checkpoint=checkpoint)
            return saved[-1]

        expected = 7 * 2.0 * 0.9**6
        np.testing.assert_allclose(float(jax.grad(final)(jnp.asarray(0.9))), expected, rtol=1e-5)

    @pytest.mark.parametrize(("checkpoint", "present"), [(True, True), (False, False)])
    def test_the_step_is_checkpointed_on_request(self, checkpoint: bool, present: bool) -> None:
        def final(initial: jax.Array) -> jax.Array:
            return integrate(_scale, initial, num_steps=3, checkpoint=checkpoint)[-1]

        program = str(jax.make_jaxpr(jax.grad(final))(jnp.asarray(1.0)))
        assert ("remat" in program or "checkpoint" in program) is present


class TestNnxTransforms:
    """A step closing over an NNX module trains through the driver under NNX transforms."""

    def test_nnx_grad_under_nnx_jit_reaches_the_module(self) -> None:
        model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
        traces: list[int] = []

        @nnx.jit
        def loss_and_grads(model: nnx.Linear, initial: jax.Array) -> tuple[jax.Array, nnx.State]:
            traces.append(1)

            def loss(model: nnx.Linear) -> jax.Array:
                def step(state: jax.Array, index: jax.Array) -> jax.Array:
                    return state + 0.1 * jnp.tanh(model(state))

                final = integrate(step, initial, num_steps=8)[-1]
                return jnp.sum(final**2)

            return nnx.value_and_grad(loss)(model)

        value, grads = loss_and_grads(model, jnp.ones((3, 2)))
        loss_and_grads(model, jnp.full((3, 2), 0.5, dtype=jnp.float32))
        assert len(traces) == 1
        assert bool(jnp.isfinite(value))
        leaves = jax.tree.leaves(grads)
        assert {leaf.shape for leaf in leaves} == {(2,), (2, 2)}
        assert all(float(jnp.abs(leaf).sum()) > 0.0 for leaf in leaves)

    def test_an_optimizer_step_through_the_driver_lowers_the_loss(self) -> None:
        model = nnx.Linear(2, 2, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(model, optax.sgd(0.05), wrt=nnx.Param)
        initial = jnp.ones((4, 2))

        def loss(model: nnx.Linear) -> jax.Array:
            def step(state: jax.Array, index: jax.Array) -> jax.Array:
                return state + 0.1 * model(state)

            return jnp.mean(integrate(step, initial, num_steps=5)[-1] ** 2)

        @nnx.jit
        def train_step(model: nnx.Linear, optimizer: nnx.Optimizer) -> jax.Array:
            value, grads = nnx.value_and_grad(loss)(model)
            optimizer.update(model, grads)
            return value

        losses = [float(train_step(model, optimizer)) for _ in range(5)]
        assert all(later < earlier for earlier, later in itertools.pairwise(losses))
