r"""Tests for :class:`MultiTaskEnergyHead` -- task-conditioned energy readouts.

Load-bearing contracts (the UMA ``task_name`` design, arXiv:2506.23971; the MACE
multi-head fine-tuning per-head scale-shift, ``../mace`` ``multihead_tools.py`` /
``ScaleShiftBlock.forward(x, head)``):

* one backbone, many per-task readouts -- two tasks give *different* energies for
  the *same* input (independent MLP params and independent ``E0``/normaliser);
* each task's own :class:`AtomicScaleShift` (``E0`` + normaliser) is applied;
* selecting task A vs B routes to the correct per-task readout;
* tasks can be added / removed after construction (``with_task`` / ``without_task``);
* the head emits the total ``"energy"`` and is registered under
  ``"multitask_energy"`` -- it drops into ``AtomisticModel`` as the ``"energy"`` head;
* the forward pass is ``jit``/``grad``/``vmap`` clean for a *fixed (static)* task
  (REQUIRED) -- task selection is a Python-level string, never a tracer branch.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.registry import PropertyHeadRegistry
from opifex.neural.atomistic.multitask import MultiTaskEnergyHead, TASK_NAME_KEY
from opifex.neural.atomistic.scale_shift import AtomicScaleShift


_FEATURE_DIM = 6
_N_ATOMS = 4
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2, 3]), jnp.asarray([1, 2, 3, 0]))


def _system() -> MolecularSystem:
    """Return a tiny 4-atom system (CH3-like) for plumbing tests."""
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.9, 0.0],
            [-0.5, -0.9, 0.0],
        ],
        dtype=jnp.float64,
    )
    return MolecularSystem(atomic_numbers=jnp.asarray([6, 1, 1, 1]), positions=positions)


def _embeddings(seed: int = 0, *, task_name: str | None = None) -> dict[str, Array]:
    """Return per-atom ``node_features`` plus an optional active ``task_name``."""
    key = jax.random.PRNGKey(seed)
    embeddings: dict[str, Array] = {
        "node_features": jax.random.normal(key, (_N_ATOMS, _FEATURE_DIM), dtype=jnp.float64)
    }
    if task_name is not None:
        embeddings[TASK_NAME_KEY] = task_name  # type: ignore[assignment]
    return embeddings


def _with_task(features: Array, task_name: str) -> dict[str, Array]:
    """Build the per-call embeddings dict carrying the active (static) task label.

    The label rides through the standard ``dict[str, Array]`` embeddings mapping
    (the documented :data:`TASK_NAME_KEY` mechanism); the ``cast`` reflects that
    the value is a Python ``str`` selector, not an array leaf.
    """
    return {"node_features": features, TASK_NAME_KEY: cast("Array", task_name)}


def _head(*, tasks: tuple[str, ...] = ("qm9", "spice")) -> MultiTaskEnergyHead:
    """Build a two-task head with distinct per-task seeds and scale-shifts."""
    scale_shifts = {
        "qm9": AtomicScaleShift(scale=jnp.asarray(2.0), shift=jnp.asarray(-1.0)),
        "spice": AtomicScaleShift(scale=jnp.asarray(0.5), shift=jnp.asarray(3.0)),
    }
    return MultiTaskEnergyHead(
        feature_dim=_FEATURE_DIM,
        task_names=tasks,
        scale_shifts={name: scale_shifts[name] for name in tasks if name in scale_shifts},
        rngs=nnx.Rngs(0),
    )


class TestMultiTaskEnergyHead:
    def test_implemented_properties_is_energy(self) -> None:
        assert _head().implemented_properties == ("energy",)

    def test_lists_its_task_names(self) -> None:
        assert set(_head().task_names) == {"qm9", "spice"}

    def test_default_task_is_first_registered(self) -> None:
        head = _head(tasks=("qm9", "spice"))
        assert head.default_task == "qm9"

    def test_output_key_is_energy(self) -> None:
        head = _head()
        out = head(_system(), _GRAPH, _embeddings(task_name="qm9"))
        assert set(out.keys()) == {"energy"}
        assert out["energy"].shape == ()

    def test_two_tasks_give_different_energies_for_same_input(self) -> None:
        head = _head()
        node_features = _embeddings()["node_features"]
        energy_a = head(_system(), _GRAPH, _with_task(node_features, "qm9"))
        energy_b = head(_system(), _GRAPH, _with_task(node_features, "spice"))
        assert not jnp.allclose(energy_a["energy"], energy_b["energy"])

    def test_selecting_task_routes_to_that_readout(self) -> None:
        with jax.enable_x64(True):
            head = _head()
            node_features = _embeddings()["node_features"]
            embeddings = {"node_features": node_features}
            # Selecting a task must equal calling that task's underlying EnergyHead.
            for name in ("qm9", "spice"):
                routed = head(_system(), _GRAPH, _with_task(node_features, name))["energy"]
                direct = head.task_head(name)(_system(), _GRAPH, embeddings)["energy"]
                assert jnp.allclose(routed, direct, atol=1e-10)

    def test_missing_task_name_uses_default_task(self) -> None:
        with jax.enable_x64(True):
            head = _head()
            node_features = _embeddings()["node_features"]
            embeddings = {"node_features": node_features}
            implicit = head(_system(), _GRAPH, embeddings)["energy"]
            explicit = head(_system(), _GRAPH, _with_task(node_features, head.default_task))[
                "energy"
            ]
            assert jnp.allclose(implicit, explicit, atol=1e-10)

    def test_unknown_task_name_raises_with_available_names(self) -> None:
        head = _head()
        with pytest.raises(KeyError, match="qm9"):
            head(_system(), _GRAPH, _embeddings(task_name="not_a_task"))

    def test_per_task_scale_shift_is_applied(self) -> None:
        with jax.enable_x64(True):
            head = _head()
            node_features = _embeddings()["node_features"]
            system = _system()
            for name, scale_shift in (
                ("qm9", AtomicScaleShift(scale=jnp.asarray(2.0), shift=jnp.asarray(-1.0))),
                ("spice", AtomicScaleShift(scale=jnp.asarray(0.5), shift=jnp.asarray(3.0))),
            ):
                # Reconstruct the raw (pre scale-shift) readout for that task.
                raw_head = head.task_head(name)
                raw = jnp.sum(
                    raw_head.readout(nnx.silu(raw_head.hidden(node_features)))  # type: ignore[attr-defined]
                )
                expected = scale_shift.apply(raw, system.n_atoms)
                routed = head(system, _GRAPH, _with_task(node_features, name))["energy"]
                assert jnp.allclose(routed, expected, atol=1e-10)

    def test_with_task_adds_a_readout(self) -> None:
        head = _head().with_task(
            "omol",
            feature_dim=_FEATURE_DIM,
            scale_shift=AtomicScaleShift.identity(),
            rngs=nnx.Rngs(1),
        )
        assert "omol" in head.task_names
        out = head(_system(), _GRAPH, _embeddings(task_name="omol"))
        assert out["energy"].shape == ()

    def test_with_task_rejects_duplicate(self) -> None:
        with pytest.raises(ValueError, match="qm9"):
            _head().with_task("qm9", feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(1))

    def test_without_task_removes_a_readout(self) -> None:
        head = _head().without_task("spice")
        assert "spice" not in head.task_names
        assert set(head.task_names) == {"qm9"}

    def test_without_task_rejects_unknown(self) -> None:
        with pytest.raises(KeyError, match="spice"):
            _head().without_task("not_a_task")

    def test_without_last_task_rejected(self) -> None:
        head = _head(tasks=("qm9",))
        with pytest.raises(ValueError, match="at least one"):
            head.without_task("qm9")

    def test_registered_under_name(self) -> None:
        assert PropertyHeadRegistry().require("multitask_energy") is MultiTaskEnergyHead

    def test_jit_grad_vmap_smoke_with_static_task(self) -> None:
        head = _head()
        graphdef, state = nnx.split(head)
        system = _system()
        node_features = _embeddings()["node_features"]
        task_name = "spice"  # fixed (static) Python-level task selection

        def energy_for(features: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            return rebuilt(system, _GRAPH, _with_task(features, task_name))["energy"]

        jitted = jax.jit(energy_for)
        energy = jitted(node_features)
        assert energy.shape == ()
        assert bool(jnp.isfinite(energy))

        gradient = jax.grad(lambda f: energy_for(f) ** 2)(node_features)
        assert gradient.shape == node_features.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

        batch = jnp.stack([node_features, node_features + 0.1])
        batched = jax.vmap(energy_for)(batch)
        assert batched.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(batched)))

    def test_two_static_tasks_jit_to_different_energies(self) -> None:
        head = _head()
        graphdef, state = nnx.split(head)
        system = _system()
        node_features = _embeddings()["node_features"]

        def energy_for(features: Array, task_name: str) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            return rebuilt(system, _GRAPH, _with_task(features, task_name))["energy"]

        # task_name is static under jit (it changes the traced computation).
        jitted = jax.jit(energy_for, static_argnums=1)
        energy_a = jitted(node_features, "qm9")
        energy_b = jitted(node_features, "spice")
        assert not jnp.allclose(energy_a, energy_b)
