r"""Tests for the equivariant invariants ``norm`` / ``inner_product`` and the
per-multiplicity scalar scaling ``apply_scalar_weights``.

Behaviour is specified against ``e3nn-jax`` (``e3nn.norm`` / ``e3nn.dot``,
``../e3nn-jax/e3nn_jax/_src/basic.py``) and the QHNet ``InnerProduct``
(``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py``): ``norm`` returns the per-irrep
Euclidean norm as ``0e`` scalars, ``inner_product`` the per-multiplicity dot
product (component-normalised by ``1 / dim``). The load-bearing tests are
rotational invariance of both, the NaN-safe gradient of ``norm`` at zero, and
``jit``/``grad``/``vmap`` cleanliness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import (
    apply_scalar_weights,
    inner_product,
    Irreps,
    IrrepsArray,
    norm,
    rms_normalize,
)


_RNG = SO3Group()


def _rotate(x: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    rotated_chunks = []
    for (_, irrep), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        matrix = wigner_d(irrep.l, rotation).astype(chunk.dtype)
        rotated_chunks.append(jnp.einsum("ij,...uj->...ui", matrix, chunk))
    flat = [c.reshape(*c.shape[:-2], -1) for c in rotated_chunks]
    return IrrepsArray(x.irreps, jnp.concatenate(flat, axis=-1))


def _random_input(irreps: Irreps, seed: int, dtype: jnp.dtype = jnp.float64) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array)


class TestNorm:
    def test_output_irreps_all_scalar(self) -> None:
        """``norm`` maps every block to a ``0e`` scalar of the same multiplicity."""
        result = norm(_random_input(Irreps("2x0e + 3x1o + 1x2e"), 0))
        assert result.irreps == Irreps("2x0e + 3x0e + 1x0e")

    def test_matches_euclidean_norm(self) -> None:
        """Each scalar equals the L2 norm of its multiplicity's components."""
        x = _random_input(Irreps("4x1o"), 1)
        result = norm(x).array
        expected = jnp.linalg.norm(x.chunks[0], axis=-1)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_rotation_invariant(self) -> None:
        """The norm of each irrep is invariant under rotation."""
        x = _random_input(Irreps("2x0e + 2x1o + 1x2e"), 2)
        rotation = _RNG.random_element(jax.random.PRNGKey(3))
        assert jnp.allclose(norm(x).array, norm(_rotate(x, rotation)).array, atol=1e-6)

    def test_squared_option(self) -> None:
        """``squared=True`` returns the squared norm without the sqrt."""
        x = _random_input(Irreps("3x1o"), 4)
        assert jnp.allclose(norm(x, squared=True).array, norm(x).array ** 2, atol=1e-6)

    def test_gradient_finite_at_zero(self) -> None:
        """The NaN-safe sqrt yields a finite gradient at a zero vector."""

        def total_norm(array: jax.Array) -> jax.Array:
            return jnp.sum(norm(IrrepsArray(Irreps("1x1o"), array)).array)

        grad = jax.grad(total_norm)(jnp.zeros((3,)))
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_vmap(self) -> None:
        """``norm`` is jit- and vmap-clean over a batch axis."""
        batch = IrrepsArray(Irreps("2x1o"), jax.random.normal(jax.random.PRNGKey(5), (8, 6)))
        out = nnx.jit(jax.vmap(norm))(batch)
        assert out.array.shape == (8, 2)


class TestInnerProduct:
    def test_output_scalar_per_multiplicity(self) -> None:
        """``inner_product`` yields one ``0e`` scalar per input multiplicity."""
        irreps = Irreps("2x0e + 3x1o")
        result = inner_product(_random_input(irreps, 6), _random_input(irreps, 7))
        assert result.irreps == Irreps("2x0e + 3x0e")

    def test_component_normalised_dot(self) -> None:
        """Each scalar is the per-multiplicity dot product divided by ``dim``."""
        x = _random_input(Irreps("2x1o"), 8)
        y = _random_input(Irreps("2x1o"), 9)
        expected = jnp.sum(x.chunks[0] * y.chunks[0], axis=-1) / 3.0
        assert jnp.allclose(inner_product(x, y).array, expected, atol=1e-6)

    def test_rotation_invariant(self) -> None:
        """The inner product is invariant under a shared rotation of both inputs."""
        x = _random_input(Irreps("1x0e + 2x1o + 1x2e"), 10)
        y = _random_input(Irreps("1x0e + 2x1o + 1x2e"), 11)
        rotation = _RNG.random_element(jax.random.PRNGKey(12))
        base = inner_product(x, y).array
        rotated = inner_product(_rotate(x, rotation), _rotate(y, rotation)).array
        assert jnp.allclose(base, rotated, atol=1e-6)


class TestRmsNormalize:
    def test_unit_rms_output(self) -> None:
        """The output has unit root-mean-square over the feature axis."""
        x = _random_input(Irreps("4x0e + 3x1o + 2x2e"), 30)
        out = rms_normalize(x).array
        assert jnp.allclose(jnp.sqrt(jnp.mean(out**2)), 1.0, atol=1e-3)

    def test_equivariance(self) -> None:
        """Scaling by an invariant RMS commutes with rotation."""
        x = _random_input(Irreps("2x0e + 2x1o + 1x2e"), 31)
        rotation = _RNG.random_element(jax.random.PRNGKey(32))
        lhs = _rotate(rms_normalize(x), rotation).array
        rhs = rms_normalize(_rotate(x, rotation)).array
        assert jnp.allclose(lhs, rhs, atol=1e-6)

    def test_bounds_large_magnitude(self) -> None:
        """A feature scaled up 1e8x normalises back to unit RMS (bounds blow-up)."""
        x = _random_input(Irreps("4x0e + 2x1o"), 33)
        huge = IrrepsArray(x.irreps, x.array * 1e8)
        out = rms_normalize(huge).array
        assert jnp.abs(out).max() < 100.0


class TestApplyScalarWeights:
    def test_scales_each_multiplicity(self) -> None:
        """Each multiplicity chunk is scaled by its scalar weight; irreps unchanged."""
        x = _random_input(Irreps("2x0e + 1x1o"), 13)
        weights = jnp.array([2.0, 3.0, 4.0])  # 3 multiplicities total
        scaled = apply_scalar_weights(x, weights)
        assert scaled.irreps == x.irreps
        assert jnp.allclose(scaled.chunks[1], x.chunks[1] * 4.0, atol=1e-6)

    def test_preserves_equivariance(self) -> None:
        """Scaling by invariant weights commutes with rotation."""
        x = _random_input(Irreps("1x0e + 2x1o"), 14)
        weights = jnp.array([0.5, 1.5, 2.5])
        rotation = _RNG.random_element(jax.random.PRNGKey(15))
        lhs = _rotate(apply_scalar_weights(x, weights), rotation).array
        rhs = apply_scalar_weights(_rotate(x, rotation), weights).array
        assert jnp.allclose(lhs, rhs, atol=1e-6)
