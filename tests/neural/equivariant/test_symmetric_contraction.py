"""Tests for the MACE-style symmetric contraction module.

Validates the load-bearing contracts: correct output layout, rotational
equivariance of the per-channel contraction, per-element weight selection, and
jit / grad / vmap cleanliness (so conservative forces differentiate through it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant.irreps import Irreps, IrrepsArray
from opifex.neural.equivariant.symmetric_contraction import SymmetricContraction


_IRREPS_IN = "1x0e + 1x1o + 1x2e"
_IRREPS_OUT = "1x0e + 1x1o + 1x2e"
_CHANNELS = 4
_SPECIES = 3


def _build(correlation: int = 3) -> SymmetricContraction:
    return SymmetricContraction(
        _IRREPS_IN,
        _IRREPS_OUT,
        correlation=correlation,
        num_species=_SPECIES,
        num_channels=_CHANNELS,
        rngs=nnx.Rngs(0),
    )


def _inputs(n_nodes: int = 2, seed: int = 1) -> tuple[IrrepsArray, jax.Array]:
    irreps = Irreps(_IRREPS_IN)
    key = jax.random.PRNGKey(seed)
    array = jax.random.normal(key, (n_nodes, _CHANNELS, irreps.dim))
    species = jax.nn.one_hot(jnp.arange(n_nodes) % _SPECIES, _SPECIES)
    return IrrepsArray(irreps, array), species


def _block_wigner(irreps: Irreps, rotation: jax.Array) -> np.ndarray:
    """Block-diagonal Wigner-D over an irreps layout (numpy, for the test)."""
    full = [np.asarray(wigner_d(ir.l, rotation)) for mul, ir in irreps for _ in range(mul)]
    size = sum(b.shape[0] for b in full)
    out = np.zeros((size, size))
    pos = 0
    for b in full:
        out[pos : pos + b.shape[0], pos : pos + b.shape[1]] = b
        pos += b.shape[0]
    return out


class TestSymmetricContraction:
    def test_output_layout(self) -> None:
        module = _build()
        features, species = _inputs()
        out = module(features, species)
        assert out.irreps == Irreps(_IRREPS_OUT)
        assert out.array.shape == (2, _CHANNELS, Irreps(_IRREPS_OUT).dim)

    def test_rotation_equivariant(self) -> None:
        with jax.enable_x64(True):
            module = _build(correlation=3)
            features, species = _inputs()
            rotation = SO3Group().random_element(jax.random.PRNGKey(5))
            d_in = jnp.asarray(_block_wigner(Irreps(_IRREPS_IN), rotation))
            d_out = jnp.asarray(_block_wigner(Irreps(_IRREPS_OUT), rotation))

            base = module(features, species).array
            rotated_in = IrrepsArray(Irreps(_IRREPS_IN), features.array @ d_in.T)
            rotated_out = module(rotated_in, species).array
            np.testing.assert_allclose(rotated_out, base @ np.asarray(d_out).T, atol=1e-6)

    def test_per_element_weights_differ(self) -> None:
        module = _build()
        irreps = Irreps(_IRREPS_IN)
        array = jax.random.normal(jax.random.PRNGKey(2), (1, _CHANNELS, irreps.dim))
        features = IrrepsArray(irreps, array)
        out0 = module(features, jax.nn.one_hot(jnp.array([0]), _SPECIES))
        out1 = module(features, jax.nn.one_hot(jnp.array([1]), _SPECIES))
        assert not jnp.allclose(out0.array, out1.array)

    def test_jit_grad_vmap_clean(self) -> None:
        module = _build()
        graphdef, state = nnx.split(module)
        features, species = _inputs()

        def energy(array: jax.Array) -> jax.Array:
            rebuilt = nnx.merge(graphdef, state)
            out = rebuilt(IrrepsArray(Irreps(_IRREPS_IN), array), species)
            return jnp.sum(out.array**2)

        value = jax.jit(energy)(features.array)
        assert jnp.isfinite(value)
        grad = jax.grad(energy)(features.array)
        assert grad.shape == features.array.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        batch = jnp.stack([features.array, features.array + 0.1])
        out = jax.vmap(energy)(batch)
        assert out.shape == (2,)
