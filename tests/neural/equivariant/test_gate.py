r"""Tests for the equivariant gate nonlinearity.

Behaviour is specified against ``e3nn-jax``'s ``gate``
(``../e3nn-jax/e3nn_jax/_src/gate.py``): scalar channels are activated directly,
and each higher-``l`` multiplicity is scaled ("gated") by an activated scalar
gate.  The gate scalars are the *rightmost* scalars of the input (one per
non-scalar multiplicity).

The load-bearing test is equivariance: the gate is invariant on scalars and
equivariant on vectors, because each vector is scaled by an invariant scalar
gate.  ``jit``/``grad``/``vmap`` are also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Gate, gate, Irreps, IrrepsArray


_RNG = SO3Group()


def _rotate(x: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    rotated_chunks = []
    for (_, irrep), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        matrix = wigner_d(irrep.l, rotation).astype(chunk.dtype)
        rotated_chunks.append(jnp.einsum("ij,...uj->...ui", matrix, chunk))
    flat = [c.reshape(*c.shape[:-2], -1) for c in rotated_chunks]
    return IrrepsArray(x.irreps, jnp.concatenate(flat, axis=-1))


def _random_input(irreps: Irreps, seed: int, dtype: jnp.dtype = jnp.float32) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array)


class TestGate:
    def test_output_irreps_drop_gate_scalars(self) -> None:
        """Gate scalars are consumed; output keeps extra scalars + gated vectors."""
        # 5x0e: 3 extra + 2 gates (for 2x1o); output is 3x0e + 2x1o.
        x = _random_input(Irreps("5x0e+2x1o"), 0)
        result = gate(x)
        assert result.irreps == Irreps("3x0e+2x1o")

    def test_pure_scalars_are_all_activated(self) -> None:
        """With nothing to gate, all scalars pass through the activation."""
        x = _random_input(Irreps("4x0e"), 1)
        result = gate(x)
        assert result.irreps == Irreps("4x0e")
        expected = jax.nn.gelu(x.array)
        assert jnp.allclose(result.array, expected, atol=1e-5)

    def test_scalars_invariant(self) -> None:
        """Scalar outputs do not change under rotation (l=0 is invariant)."""
        x = _random_input(Irreps("4x0e"), 2)
        rotation = _RNG.random_element(jax.random.PRNGKey(3))
        assert jnp.allclose(gate(_rotate(x, rotation)).array, gate(x).array, atol=1e-5)

    def test_equivariance(self) -> None:
        """``gate(D . x) = D . gate(x)``: gates are invariant, so vectors rotate."""
        with jax.enable_x64(True):
            irreps = Irreps("6x0e+2x1o+1x2e")
            x = _random_input(irreps, 4, dtype=jnp.float64)
            rotation = _RNG.random_element(jax.random.PRNGKey(5)).astype(jnp.float64)
            left = gate(_rotate(x, rotation))
            right = _rotate(gate(x), rotation)
            assert left.irreps == right.irreps
            assert jnp.allclose(left.array, right.array, atol=1e-8)

    def test_gate_scales_vectors_by_activated_scalar(self) -> None:
        """A single 1o gated by one 0e equals ``sigmoid(gate) * vector``."""
        # 1x0e (extra) + 1x0e (gate) + 1x1o.
        array = jnp.asarray([0.5, 0.7, 1.0, 2.0, 3.0])
        x = IrrepsArray("2x0e+1x1o", array)
        result = gate(x)
        gate_value = jax.nn.sigmoid(jnp.asarray(0.7))
        expected_vector = gate_value * jnp.asarray([1.0, 2.0, 3.0])
        assert jnp.allclose(result.chunks[1].reshape(-1), expected_vector, atol=1e-5)

    def test_requires_enough_scalars(self) -> None:
        """Fewer scalars than non-scalar irreps is an error (e3nn assumption)."""
        x = _random_input(Irreps("1x0e+2x1o"), 6)
        try:
            gate(x)
        except ValueError:
            return
        raise AssertionError("expected ValueError when scalars < non-scalar irreps")

    def test_gate_module_wrapper(self) -> None:
        layer = Gate("5x0e+2x1o")
        x = _random_input(Irreps("5x0e+2x1o"), 7)
        assert layer(x).irreps == gate(x).irreps
        assert jnp.allclose(layer(x).array, gate(x).array, atol=1e-6)

    def test_jit_compatibility(self) -> None:
        x = _random_input(Irreps("5x0e+2x1o"), 8)
        jitted = jax.jit(lambda arr: gate(IrrepsArray("5x0e+2x1o", arr)).array)
        assert jnp.allclose(jitted(x.array), gate(x).array, atol=1e-5)

    def test_grad_compatibility(self) -> None:
        x = _random_input(Irreps("5x0e+2x1o"), 9)

        def loss(arr: jax.Array) -> jax.Array:
            return jnp.sum(gate(IrrepsArray("5x0e+2x1o", arr)).array ** 2)

        gradient = jax.grad(loss)(x.array)
        assert gradient.shape == x.array.shape
        assert jnp.all(jnp.isfinite(gradient))

    def test_vmap_compatibility(self) -> None:
        irreps = Irreps("5x0e+2x1o")
        batch = jax.random.normal(jax.random.PRNGKey(10), (4, irreps.dim))
        result = gate(IrrepsArray(irreps, batch))
        assert result.array.shape == (4, Irreps("3x0e+2x1o").dim)
        single = gate(IrrepsArray(irreps, batch[0]))
        assert jnp.allclose(result.array[0], single.array, atol=1e-5)
