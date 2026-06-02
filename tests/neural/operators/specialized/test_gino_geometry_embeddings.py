"""Tests for computed geometry embeddings in the GINO operator.

These tests pin the canonical geometry-embedding behaviour of the
Geometry-Informed Neural Operator (Li et al. 2023, arXiv:2309.00583):

* coordinates are lifted into a sinusoidal positional embedding whose
  transformer-style frequencies match the reference implementation
  (``neuraloperator/neuralop/layers/embeddings.py``);
* the lifted coordinates are encoded into geometry features that actually
  flow through the network (distinct geometries -> distinct embeddings and
  distinct operator outputs); and
* the embedding path is compatible with ``jit`` / ``grad`` / ``vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.neural.operators.common.embeddings import SinusoidalEmbedding
from opifex.neural.operators.specialized.gino import (
    GeometryEncoder,
    GeometryInformedNeuralOperator,
)


class TestSinusoidalTransformerFrequencies:
    """The transformer frequencies must match the cited reference formula."""

    def test_transformer_frequencies_match_reference(self) -> None:
        """Reference: freqs = (1/max_positions) ** (arange(L)/L * 2)."""
        num_frequencies = 4
        max_positions = 10000
        embedding = SinusoidalEmbedding(
            in_channels=1,
            num_frequencies=num_frequencies,
            embedding_type="transformer",
            max_positions=max_positions,
        )

        # Single coordinate p so the embedding is exactly sin/cos of p * freq.
        coord = jnp.asarray([[[0.5]]])  # (1, 1, 1)
        output = np.asarray(embedding(coord)).reshape(-1)

        ref_exponents = np.arange(0, num_frequencies) / num_frequencies * 2
        ref_freqs = (1.0 / max_positions) ** ref_exponents
        expected = np.empty(2 * num_frequencies)
        expected[0::2] = np.sin(0.5 * ref_freqs)
        expected[1::2] = np.cos(0.5 * ref_freqs)

        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-7)


class TestGeometryEncoder:
    """The geometry encoder lifts coordinates into geometry features."""

    def test_output_shape(self) -> None:
        """Encoder maps (batch, n, coord_dim) -> (batch, n, output_dim)."""
        encoder = GeometryEncoder(
            coord_dim=2,
            hidden_dim=32,
            output_dim=16,
            use_positional_encoding=True,
            rngs=nnx.Rngs(0),
        )
        coords = jax.random.normal(jax.random.PRNGKey(1), (3, 25, 2))
        out = encoder(coords)
        assert out.shape == (3, 25, 16)
        assert jnp.all(jnp.isfinite(out))

    def test_distinct_geometries_distinct_embeddings(self) -> None:
        """Different coordinates must produce different geometry embeddings."""
        encoder = GeometryEncoder(
            coord_dim=2,
            hidden_dim=32,
            output_dim=16,
            use_positional_encoding=True,
            rngs=nnx.Rngs(0),
        )
        coords_a = jnp.zeros((1, 4, 2))
        coords_b = jnp.linspace(-1.0, 1.0, 8).reshape(1, 4, 2)

        emb_a = encoder(coords_a)
        emb_b = encoder(coords_b)

        assert not jnp.allclose(emb_a, emb_b)

    def test_positional_encoding_changes_output(self) -> None:
        """Enabling positional encoding must change the encoder behaviour."""
        coords = jax.random.normal(jax.random.PRNGKey(2), (1, 6, 2))
        with_pe = GeometryEncoder(
            coord_dim=2,
            hidden_dim=16,
            output_dim=8,
            use_positional_encoding=True,
            rngs=nnx.Rngs(3),
        )
        without_pe = GeometryEncoder(
            coord_dim=2,
            hidden_dim=16,
            output_dim=8,
            use_positional_encoding=False,
            rngs=nnx.Rngs(3),
        )
        # Different input dimensionality means a genuinely different mapping.
        assert with_pe.embedded_coord_dim != without_pe.embedded_coord_dim
        assert without_pe.positional_embedding is None
        assert with_pe.positional_embedding is not None
        assert with_pe.embedded_coord_dim == 2 + with_pe.positional_embedding.out_channels
        # Both must still produce valid embeddings on the same coordinates.
        assert with_pe(coords).shape == (1, 6, 8)
        assert without_pe(coords).shape == (1, 6, 8)

    def test_jit_grad_vmap_smoke(self) -> None:
        """Encoder must compose with jit, grad and vmap."""
        encoder = GeometryEncoder(
            coord_dim=2,
            hidden_dim=16,
            output_dim=8,
            use_positional_encoding=True,
            rngs=nnx.Rngs(0),
        )
        graphdef, state = nnx.split(encoder)

        def apply(state_in: nnx.State, coords: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, state_in)
            return model(coords)

        coords = jax.random.normal(jax.random.PRNGKey(4), (2, 9, 2))

        # jit
        jitted = jax.jit(apply)
        out_jit = jitted(state, coords)
        assert out_jit.shape == (2, 9, 8)

        # grad (scalar loss w.r.t. coordinates)
        def loss(coords_in: jax.Array) -> jax.Array:
            return jnp.sum(apply(state, coords_in) ** 2)

        grads = jax.grad(loss)(coords)
        assert grads.shape == coords.shape
        assert jnp.all(jnp.isfinite(grads))

        # vmap over a batch of geometries (each (9, 2) -> (9, 8))
        single = jax.random.normal(jax.random.PRNGKey(5), (5, 9, 2))
        out_vmap = jax.vmap(lambda c: apply(state, c))(single)
        assert out_vmap.shape == (5, 9, 8)
        assert jnp.all(jnp.isfinite(out_vmap))


class TestGINOGeometryFlow:
    """Computed geometry embeddings must influence the operator output."""

    def _make_model(self) -> GeometryInformedNeuralOperator:
        return GeometryInformedNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            coord_dim=2,
            geometry_dim=16,
            num_layers=1,
            use_geometry_attention=True,
            use_spectral_conv=False,
            rngs=nnx.Rngs(0),
        )

    def test_distinct_geometries_distinct_outputs(self) -> None:
        """Different input coordinates must yield different operator outputs."""
        model = self._make_model()
        x = jax.random.normal(jax.random.PRNGKey(6), (1, 8, 8, 2))

        coords_grid = model._generate_default_coords((8, 8), 1)
        coords_warped = coords_grid * jnp.asarray([1.0, 2.0]) + 0.3

        out_grid = model(x, geometry_data={"coords": coords_grid})
        out_warped = model(x, geometry_data={"coords": coords_warped})

        assert out_grid.shape == (1, 8, 8, 1)
        assert jnp.all(jnp.isfinite(out_grid))
        assert jnp.all(jnp.isfinite(out_warped))
        assert not jnp.allclose(out_grid, out_warped)

    def test_geometry_not_dummy_zeros(self) -> None:
        """The geometry-attention path must use computed (non-zero) embeddings."""
        model = self._make_model()
        coords = model._generate_default_coords((8, 8), 1)
        embeddings = model.geometry_encoder(coords)
        # A real encoder of a non-trivial grid is not identically zero.
        assert jnp.any(jnp.abs(embeddings) > 1e-6)
        assert embeddings.shape == (1, 64, model.geometry_dim)

    def test_jit_forward(self) -> None:
        """The full forward pass must run under jax.jit."""
        model = self._make_model()
        graphdef, state = nnx.split(model)
        x = jax.random.normal(jax.random.PRNGKey(7), (1, 8, 8, 2))

        @jax.jit
        def forward(state_in: nnx.State, inputs: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, state_in)(inputs)

        out = forward(state, x)
        assert out.shape == (1, 8, 8, 1)
        assert jnp.all(jnp.isfinite(out))
