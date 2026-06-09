r"""Tests for the eSCN SO(2)-frame edge convolution.

Validates the central guarantee of the SO(2)-frame trick (Passaro & Zitnick
2023, "Reducing SO(3) Convolutions to SO(2)", arXiv:2302.03655; QHNetV2, Yu et
al. 2023, arXiv:2306.04922): rotating each edge into a local frame aligned with
its direction turns the SO(3) edge tensor product into a per-``m`` SO(2)
operation, while preserving full SO(3) equivariance of the edge -> message map.

The reference contract the convolution must satisfy is the same drop-in
``TensorProduct``-style signature used by
:class:`opifex.neural.quantum.hamiltonian.predictor.HamiltonianPredictor` for
its ``edge_tensor_product``: ``__call__(node_features, edge_vectors) ->
IrrepsArray`` with ``irreps_out``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.quantum.hamiltonian.so2_convolution import SO2EdgeConvolution


HIDDEN_IRREPS = "4x0e + 4x1o + 2x2e"
SH_LMAX = 2


def _random_rotation(key: jax.Array) -> jax.Array:
    """Return a uniformly random ``SO(3)`` rotation matrix via QR of a Gaussian."""
    gaussian = jax.random.normal(key, (3, 3))
    q, r = jnp.linalg.qr(gaussian)
    q = q * jnp.sign(jnp.diag(r))
    return q * jnp.sign(jnp.linalg.det(q))


def _rotate_irreps(irreps: Irreps, array: jax.Array, rotation: jax.Array) -> jax.Array:
    """Apply the block-diagonal Wigner-D of ``rotation`` to an irreps feature array."""
    pieces: list[jax.Array] = []
    for mul, irrep in irreps.blocks:
        block = array[..., _block_slice(irreps, irrep, mul)]
        block = block.reshape(*array.shape[:-1], mul, irrep.dim)
        d = wigner_d(irrep.l, rotation)
        rotated = jnp.einsum("ij,...uj->...ui", d, block)
        pieces.append(rotated.reshape(*array.shape[:-1], mul * irrep.dim))
    return jnp.concatenate(pieces, axis=-1)


def _block_slice(irreps: Irreps, target_irrep: object, target_mul: int) -> slice:
    start = 0
    for mul, irrep in irreps.blocks:
        width = mul * irrep.dim
        if irrep == target_irrep and mul == target_mul and irrep.dim == (2 * irrep.l + 1):
            return slice(start, start + width)
        start += width
    raise AssertionError("block not found")


def _build_inputs(key: jax.Array, num_edges: int = 5) -> tuple[IrrepsArray, jax.Array]:
    """Return random node features and random edge vectors for ``num_edges`` edges."""
    feat_key, vec_key = jax.random.split(key)
    irreps = Irreps(HIDDEN_IRREPS)
    features = IrrepsArray(irreps, jax.random.normal(feat_key, (num_edges, irreps.dim)))
    vectors = jax.random.normal(vec_key, (num_edges, 3))
    return features, vectors


def _make_conv() -> SO2EdgeConvolution:
    return SO2EdgeConvolution(
        HIDDEN_IRREPS,
        sh_lmax=SH_LMAX,
        irreps_out=HIDDEN_IRREPS,
        rngs=nnx.Rngs(params=0),
    )


def test_output_irreps_match_configured_layout() -> None:
    """The convolution output carries exactly the requested ``irreps_out``."""
    conv = _make_conv()
    features, vectors = _build_inputs(jax.random.key(1))
    out = conv(features, vectors)
    assert out.irreps == Irreps(HIDDEN_IRREPS)
    assert out.array.shape == (features.array.shape[0], Irreps(HIDDEN_IRREPS).dim)


def test_rotational_equivariance() -> None:
    r"""Rotating geometry by ``R`` rotates the output by ``D(R)``: the eSCN guarantee.

    With ``x' = D(R) x`` on node features and ``v' = R v`` on edge vectors, the
    SO(2)-frame convolution must satisfy ``f(x', v') = D(R) f(x, v)`` because the
    edge frame co-rotates with the geometry.
    """
    # The map is exactly equivariant; the rotate -> mix -> rotate-back chain of
    # Wigner-D matrices accumulates float32 error, so the residual is checked in
    # float64 (the standard opifex pattern for tight equivariance assertions).
    with jax.enable_x64(True):
        conv = _make_conv()
        features, vectors = _build_inputs(jax.random.key(2))
        features = IrrepsArray(features.irreps, features.array.astype(jnp.float64))
        vectors = vectors.astype(jnp.float64)
        rotation = _random_rotation(jax.random.key(3)).astype(jnp.float64)

        baseline = conv(features, vectors)
        rotated_features = IrrepsArray(
            features.irreps, _rotate_irreps(features.irreps, features.array, rotation)
        )
        rotated_vectors = jnp.einsum("ij,nj->ni", rotation, vectors)
        rotated_out = conv(rotated_features, rotated_vectors)

        expected = _rotate_irreps(baseline.irreps, baseline.array, rotation)
        residual = float(jnp.max(jnp.abs(rotated_out.array - expected)))
    assert residual < 1e-5, f"equivariance residual {residual:.2e} exceeds 1e-5"


def test_rotational_equivariance_float32_bound() -> None:
    """In float32 the eSCN map stays equivariant to a looser, precision-set bound."""
    conv = _make_conv()
    features, vectors = _build_inputs(jax.random.key(2))
    rotation = _random_rotation(jax.random.key(3))

    baseline = conv(features, vectors)
    rotated_features = IrrepsArray(
        features.irreps, _rotate_irreps(features.irreps, features.array, rotation)
    )
    rotated_vectors = jnp.einsum("ij,nj->ni", rotation, vectors)
    rotated_out = conv(rotated_features, rotated_vectors)

    expected = _rotate_irreps(baseline.irreps, baseline.array, rotation)
    residual = float(jnp.max(jnp.abs(rotated_out.array - expected)))
    assert residual < 1e-3, f"float32 equivariance residual {residual:.2e} exceeds 1e-3"


def test_zero_edge_vector_is_finite() -> None:
    """A degenerate (zero-length) edge must not produce NaNs/Infs."""
    conv = _make_conv()
    irreps = Irreps(HIDDEN_IRREPS)
    features = IrrepsArray(irreps, jnp.ones((2, irreps.dim)))
    vectors = jnp.zeros((2, 3))
    out = conv(features, vectors)
    assert bool(jnp.all(jnp.isfinite(out.array)))


def test_agreement_both_maps_are_equivariant() -> None:
    r"""Consistency: SO(2) conv and the dense SO(3) tensor product share output irreps.

    Both the eSCN SO(2)-frame convolution and the dense
    :class:`FullyConnectedTensorProduct` realise an equivariant edge -> message
    map ``hidden (x) Y_l -> hidden``; this checks they expose the same output
    layout so the SO(2) conv is a drop-in (the learnable weights differ, so
    values are not expected to match).
    """
    from opifex.neural.equivariant import FullyConnectedTensorProduct, spherical_harmonics

    conv = _make_conv()
    features, vectors = _build_inputs(jax.random.key(4))
    sh = spherical_harmonics(SH_LMAX, vectors)
    dense = FullyConnectedTensorProduct(
        HIDDEN_IRREPS, sh.irreps, HIDDEN_IRREPS, rngs=nnx.Rngs(params=1)
    )
    so2_out = conv(features, vectors)
    dense_out = dense(features, sh)
    assert so2_out.irreps == dense_out.irreps


def test_jit_grad_vmap_smoke() -> None:
    """The convolution survives ``jit``, ``grad`` and ``vmap`` transforms."""
    conv = _make_conv()
    graphdef, state = nnx.split(conv)
    features, vectors = _build_inputs(jax.random.key(5))

    def apply(state: nnx.State, array: jax.Array, vecs: jax.Array) -> jax.Array:
        module = nnx.merge(graphdef, state)
        return module(IrrepsArray(Irreps(HIDDEN_IRREPS), array), vecs).array

    jitted = jax.jit(apply)
    out = jitted(state, features.array, vectors)
    assert out.shape == (features.array.shape[0], Irreps(HIDDEN_IRREPS).dim)

    def scalar_loss(array: jax.Array) -> jax.Array:
        return jnp.sum(apply(state, array, vectors) ** 2)

    grads = jax.grad(scalar_loss)(features.array)
    assert grads.shape == features.array.shape
    assert bool(jnp.all(jnp.isfinite(grads)))

    batched_features = jnp.broadcast_to(features.array, (3, *features.array.shape))
    batched_vectors = jnp.broadcast_to(vectors, (3, *vectors.shape))
    vmapped = jax.vmap(apply, in_axes=(None, 0, 0))(state, batched_features, batched_vectors)
    assert vmapped.shape == (3, *out.shape)


def test_invalid_input_irreps_raise() -> None:
    """Passing features whose irreps differ from the configured ones fails fast."""
    conv = _make_conv()
    wrong = IrrepsArray("2x0e", jnp.ones((4, 2)))
    with pytest.raises(ValueError, match="irreps"):
        conv(wrong, jnp.zeros((4, 3)))


def test_m0_axis_is_y() -> None:
    r"""Sanity: opifex's real basis quantises about ``y`` (the eSCN reference axis).

    Under a ``y``-axis rotation the Wigner-D matrix is block-diagonal in
    ``\pm m`` pairs with the central ``m = 0`` component fixed; this is the
    structural fact the SO(2) reduction relies on.
    """
    angle = 0.7
    cos, sin = np.cos(angle), np.sin(angle)
    r_y = jnp.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    d2 = np.array(wigner_d(2, r_y))
    assert np.isclose(d2[2, 2], 1.0, atol=1e-6)
    off_center = np.abs(d2[2, [0, 1, 3, 4]]).max()
    assert off_center < 1e-6
