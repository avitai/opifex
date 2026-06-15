r"""PsiFormer self-attention neural-network wavefunction.

A Flax-NNX port of the PsiFormer ansatz (von Glehn, Spencer & Pfau, *A
Self-Attention Ansatz for Ab-initio Quantum Chemistry*, arXiv:2211.13672;
reference implementation ``../ferminet`` ``psiformer.py`` ``make_psiformer_layers``
and ``make_self_attention_block``).

The PsiFormer keeps the generalized-Slater determinant structure of FermiNet

.. math::

    \psi(r) = \sum_k w_k \det\!\big[\phi^k_i(r_j)\big],

but replaces FermiNet's pooled one-/two-electron equivariant streams with a
stack of multi-head **self-attention** blocks over the electrons. Each electron
attends to every other electron, so the backbone is permutation-equivariant by
construction (a permutation of the electrons permutes the per-electron outputs
identically), which in turn makes the determinant antisymmetric under same-spin
exchange.

Architecture (one self-attention layer, following the reference)
----------------------------------------------------------------
#. **Input features.** Only the one-electron electron-nucleus features
   ``[r_ae, ae]`` are used (the PsiFormer drops the explicit two-electron
   stream -- pair information re-enters through attention). The integer spin
   label of each electron (``+1`` up, ``-1`` down) is appended; the reference
   notes this spin feature is *required* for correct permutation equivariance.
#. **Embed.** A bias-free linear map lifts the features to the attention width
   ``attn_dim = num_heads * head_dim``.
#. **Self-attention block**, repeated ``num_layers`` times::

       x <- x + MultiHeadAttention(x, x, x)      # residual
       x <- LayerNorm(x)                          # optional
       x <- x + MLP(x)                            # residual, tanh MLP
       x <- LayerNorm(x)                          # optional

#. **Orbitals + envelope + determinant.** Identical to FermiNet: a per-spin
   linear projection to ``determinants * nelectron`` orbitals, an isotropic
   exponential envelope ``sum_atom pi * exp(-sigma * r_ae)``, and the shared
   log-domain :func:`~._blocks.logdet_matmul`. The PsiFormer requires a single
   dense ``(nelec, nelec)`` determinant (``full_det`` in the reference), so the
   orbital projection emits ``nelectron`` columns per determinant.

Design notes
------------
* The module exposes a **single-walker** ``__call__(positions) -> (sign,
  log|psi|)`` evaluated entirely in the log domain, so it is ``vmap``-able over
  walkers and ``grad``-able for the kinetic energy -- the same
  :class:`~opifex.neural.quantum.vmc.protocols.Wavefunction` contract as
  :class:`~.ferminet.FermiNet`.
* Spin counts are *static* attributes (``nspins``); the spin partition split
  happens at trace time, so the network is ``jit``-clean.
* :meth:`backbone_features` exposes the per-electron attention output for
  permutation-equivariance testing without re-running the orbital head.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.quantum.vmc.wavefunctions._blocks import (
    construct_input_features,
    logdet_matmul,
)


logger = logging.getLogger(__name__)


class _MultiHeadSelfAttention(nnx.Module):
    """FermiNet-style multi-head self-attention over the electron index.

    Scaled dot-product attention with separate bias-free query/key/value
    projections and a bias-free output projection, matching the reference
    ``make_multi_head_attention``. Attention is computed over the electron axis,
    so a permutation of the electrons permutes the outputs identically.

    Args:
        feature_dim: Input (and output) embedding width.
        num_heads: Number of attention heads.
        head_dim: Per-head embedding width.
        rngs: NNX random-number generators.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the bias-free Q/K/V and output projections."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float64))
        inner = num_heads * head_dim
        self.query = nnx.Linear(feature_dim, inner, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(feature_dim, inner, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(feature_dim, inner, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(inner, feature_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: Float[Array, "nelectron feature_dim"]) -> Array:
        """Apply self-attention over the electron axis.

        Args:
            x: Per-electron embeddings of shape ``(nelectron, feature_dim)``.

        Returns:
            The attended embeddings of shape ``(nelectron, feature_dim)``.
        """
        nelectron = x.shape[0]

        def split_heads(projection: Array) -> Array:
            return projection.reshape(nelectron, self.num_heads, self.head_dim)

        query = split_heads(self.query(x))
        key = split_heads(self.key(x))
        value = split_heads(self.value(x))

        # Logits over (head, query electron, key electron).
        logits = jnp.einsum("thd,Thd->htT", query, key) * self._scale
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("htT,Thd->thd", weights, value)
        attended = attended.reshape(nelectron, self.num_heads * self.head_dim)
        return self.out(attended)


class _AttentionLayer(nnx.Module):
    """One PsiFormer self-attention layer: attention + MLP with residuals.

    Args:
        feature_dim: Embedding width carried through the layer.
        num_heads: Number of attention heads.
        head_dim: Per-head embedding width.
        mlp_hidden: Hidden widths of the position-wise tanh MLP (the final
            projection back to ``feature_dim`` is appended automatically).
        use_layer_norm: If ``True`` apply LayerNorm after each residual.
        rngs: NNX random-number generators.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden: tuple[int, ...],
        use_layer_norm: bool,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the attention sublayer, the tanh MLP and optional LayerNorms."""
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.attention = _MultiHeadSelfAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            rngs=rngs,
        )
        self.mlp_layers = nnx.List([])
        width_in = feature_dim
        for width_out in (*mlp_hidden, feature_dim):
            self.mlp_layers.append(nnx.Linear(width_in, width_out, rngs=rngs))
            width_in = width_out
        if use_layer_norm:
            self.attention_norm = nnx.LayerNorm(feature_dim, rngs=rngs)
            self.mlp_norm = nnx.LayerNorm(feature_dim, rngs=rngs)

    def _mlp(self, x: Array) -> Array:
        """Position-wise tanh MLP applied independently per electron."""
        for layer in self.mlp_layers:
            x = jnp.tanh(layer(x))
        return x

    def __call__(self, x: Float[Array, "nelectron feature_dim"]) -> Array:
        """Apply attention + MLP with residual connections (and optional norms)."""
        x = x + self.attention(x)
        if self.use_layer_norm:
            x = self.attention_norm(x)
        x = x + self._mlp(x)
        if self.use_layer_norm:
            x = self.mlp_norm(x)
        return x


class PsiFormer(nnx.Module):
    """PsiFormer self-attention generalized-Slater wavefunction (single walker).

    Args:
        nspins: ``(n_up, n_down)`` electron counts (static).
        atoms: Nuclear coordinates of shape ``(natom, ndim)``.
        charges: Nuclear charges of shape ``(natom,)``.
        num_layers: Number of stacked self-attention layers.
        num_heads: Number of multi-head attention heads.
        head_dim: Per-head embedding width; the attention width is
            ``num_heads * head_dim``.
        mlp_hidden: Hidden widths of the position-wise tanh MLP in each layer.
        determinants: Number of dense Slater determinants in the sum.
        use_layer_norm: If ``True`` apply LayerNorm after each residual.
        rngs: NNX random-number generators.

    Raises:
        ValueError: If ``num_heads``, ``head_dim`` or ``num_layers`` are not
            positive, or if no electrons are present.

    Notes:
        The PsiFormer uses a single dense ``(nelec, nelec)`` determinant
        (``full_det`` in the reference); there is no block-diagonal variant.
    """

    def __init__(
        self,
        *,
        nspins: tuple[int, int],
        atoms: Float[Array, "natom ndim"],
        charges: Float[Array, " natom"],
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: int = 64,
        mlp_hidden: tuple[int, ...] = (256,),
        determinants: int = 4,
        use_layer_norm: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the embedding, self-attention stack, orbitals and envelopes."""
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                "num_heads and head_dim must be positive; got "
                f"num_heads={num_heads}, head_dim={head_dim}."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive; got {num_layers}.")
        if sum(nspins) == 0:
            raise ValueError("No electrons present in nspins.")

        self.nspins = nspins
        self.atoms = jnp.asarray(atoms)
        self.charges = jnp.asarray(charges)
        self.determinants = determinants
        self.ndim = int(self.atoms.shape[1])
        natom = int(self.atoms.shape[0])
        nelectron = sum(nspins)

        # Static cumulative split index for the spin partition.
        self._spin_split: tuple[int, ...] = (nspins[0],) if nspins[1] > 0 else ()
        active_spins = tuple(s for s in nspins if s > 0)
        self._active_spins = active_spins
        # Per-electron spin label (+1 up, -1 down): required for the PsiFormer's
        # permutation equivariance (reference ``make_psiformer_layers``).
        self._spins = jnp.array([1.0] * nspins[0] + [-1.0] * nspins[1])

        # One-electron input feature width: natom * (ndim + 1) for [r_ae, ae],
        # plus one channel for the spin label.
        feature_in = natom * (self.ndim + 1) + 1
        attn_dim = num_heads * head_dim
        self.embed = nnx.Linear(feature_in, attn_dim, use_bias=False, rngs=rngs)

        self.layers = nnx.List([])
        for _ in range(num_layers):
            self.layers.append(
                _AttentionLayer(
                    feature_dim=attn_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_hidden=mlp_hidden,
                    use_layer_norm=use_layer_norm,
                    rngs=rngs,
                )
            )

        # Orbital projections: per spin channel, full dense determinant needs
        # ``nelectron`` columns per determinant.
        orbital_projections = []
        envelope_pi = []
        envelope_sigma = []
        for _spin in active_spins:
            out_features = determinants * nelectron
            orbital_projections.append(nnx.Linear(attn_dim, out_features, rngs=rngs))
            envelope_pi.append(nnx.Param(jnp.ones((natom, out_features))))
            envelope_sigma.append(nnx.Param(jnp.ones((natom, out_features))))
        self.orbital_projections = nnx.List(orbital_projections)
        self.envelope_pi = nnx.List(envelope_pi)
        self.envelope_sigma = nnx.List(envelope_sigma)

        # Per-determinant mixing weights of the generalized-Slater sum.
        self.determinant_weights = nnx.Param(jnp.ones((determinants, 1)))

    def backbone_features(
        self, positions: Float[Array, "nelectron ndim"]
    ) -> Float[Array, "nelectron attn_dim"]:
        """Run the embedding + self-attention stack to per-electron features.

        Exposed for permutation-equivariance testing: a permutation of the
        electrons permutes these features identically.

        Args:
            positions: Electron coordinates of shape ``(nelectron, ndim)``.

        Returns:
            The per-electron attention output of shape ``(nelectron, attn_dim)``.
        """
        ae, _ee, r_ae, _r_ee = construct_input_features(positions, self.atoms)
        ae_features = jnp.concatenate([r_ae, ae], axis=-1)
        ae_features = ae_features.reshape(ae_features.shape[0], -1)
        features = jnp.concatenate([ae_features, self._spins[:, None]], axis=-1)

        x = self.embed(features)
        for layer in self.layers:
            x = layer(x)
        return x

    def _orbital_matrices(
        self,
        h_to_orbitals: Float[Array, "nelectron attn_dim"],
        r_ae: Float[Array, "nelectron natom 1"],
    ) -> list[Array]:
        """Project features to per-determinant orbital matrices with envelopes.

        Returns one matrix per active spin channel, each of shape
        ``(rows, determinants, nelectron)`` for the dense determinant.
        """
        spin_split = self._spin_split
        h_channels = jnp.split(h_to_orbitals, spin_split, axis=0)
        r_ae_channels = jnp.split(r_ae, spin_split, axis=0)
        nelectron = sum(self.nspins)

        matrices = []
        for index, spin in enumerate(self._active_spins):
            orbitals = self.orbital_projections[index](h_channels[index])
            # Isotropic envelope: sum_atom pi * exp(-sigma * r_ae).
            decay = jnp.exp(-r_ae_channels[index] * self.envelope_sigma[index].value)
            envelope = jnp.sum(decay * self.envelope_pi[index].value, axis=1)
            orbitals = orbitals * envelope
            matrices.append(orbitals.reshape(spin, self.determinants, nelectron))
        return matrices

    def __call__(self, positions: Float[Array, "nelectron ndim"]) -> tuple[Array, Array]:
        """Evaluate the wavefunction for a single walker.

        Args:
            positions: Electron coordinates of shape ``(nelectron, ndim)``.

        Returns:
            A ``(sign, log|psi|)`` tuple of scalars.
        """
        _ae, _ee, r_ae, _r_ee = construct_input_features(positions, self.atoms)
        h_to_orbitals = self.backbone_features(positions)
        matrices = self._orbital_matrices(h_to_orbitals, r_ae)

        # Stack spin channels along the row axis into one dense determinant of
        # shape (determinants, nelectron, nelectron).
        dense = jnp.concatenate([jnp.moveaxis(m, 0, 1) for m in matrices], axis=1)
        return logdet_matmul([dense], self.determinant_weights.value)


__all__ = ["PsiFormer"]
