r"""FermiNet-core generalized-Slater neural-network wavefunction.

A Flax-NNX port of the Fermionic Neural Network ansatz (Pfau, Spencer, Matthews
& Foulkes, *Phys. Rev. Research* **2**, 033429 (2020); reference implementation
``../ferminet`` ``networks.py`` ``make_orbitals`` and ``make_fermi_net_layers``).

The wavefunction is a sum of generalized Slater determinants

.. math::

    \psi(r) = \sum_k w_k \det\!\big[\phi^k_i(r_j)\big],

where each orbital :math:`\phi^k_i` is a permutation-equivariant function of all
electron coordinates (not a single-particle orbital), multiplied by a per-orbital
isotropic exponential envelope that enforces the correct asymptotic decay. The
equivariant backbone interleaves a one-electron stream ``h_one`` and a
two-electron stream ``h_two`` with FermiNet symmetric feature pooling.

Design notes
------------
* The module exposes a **single-walker** ``__call__(positions) -> (sign, log|psi|)``
  evaluated entirely in the log domain (:func:`~._blocks.logdet_matmul`), so it
  is ``vmap``-able over walkers and ``grad``-able for the kinetic energy.
* The determinant axis is handled by a ``vmap`` inside the module, matching the
  task's "vmap over the determinant axis" requirement.
* Spin is a *static* attribute (``nspins``); the equivariant feature pooling
  splits along the spin partition at trace time, so the network is jit-clean.
* A PsiFormer attention backbone can swap in by replacing
  :meth:`_equivariant_backbone` -- the orbital/envelope/determinant machinery is
  agnostic to how ``h_to_orbitals`` is produced.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.quantum.vmc.wavefunctions._blocks import (
    construct_input_features,
    logdet_matmul,
)


def _symmetric_features(
    h_one: Float[Array, "nelectron n1"],
    h_two: Float[Array, "nelectron nelectron n2"],
    spin_split: tuple[int, ...],
) -> Array:
    """Concatenate one-electron features with spin-pooled one/two-electron means.

    This is the FermiNet permutation-equivariant feature construction
    (``construct_symmetric_features``): every electron sees its own features plus
    the mean of the one- and two-electron features over each occupied spin
    channel.

    Args:
        h_one: One-electron stream of shape ``(nelectron, n1)``.
        h_two: Two-electron stream of shape ``(nelectron, nelectron, n2)``.
        spin_split: Cumulative split indices for the spin partition.

    Returns:
        The symmetric feature array of shape ``(nelectron, n1 + k*n1 + k*n2)``
        where ``k`` is the number of occupied spin channels.
    """
    nelectron = h_one.shape[0]
    h_ones = jnp.split(h_one, spin_split, axis=0)
    h_twos = jnp.split(h_two, spin_split, axis=0)

    g_one = [
        jnp.tile(jnp.mean(h, axis=0, keepdims=True), (nelectron, 1))
        for h in h_ones
        if h.shape[0] > 0
    ]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.shape[0] > 0]
    return jnp.concatenate([h_one, *g_one, *g_two], axis=1)


class FermiNet(nnx.Module):
    """FermiNet generalized-Slater wavefunction (single-walker evaluation).

    Args:
        nspins: ``(n_up, n_down)`` electron counts (static).
        atoms: Nuclear coordinates of shape ``(natom, ndim)``.
        charges: Nuclear charges of shape ``(natom,)``.
        hidden_one: Widths of the one-electron stream layers.
        hidden_two: Widths of the two-electron stream layers. Must have the same
            length as ``hidden_one``.
        determinants: Number of Slater determinants in the sum.
        full_det: If ``True`` use a single dense ``(nelec, nelec)`` determinant;
            if ``False`` use spin-factored block-diagonal determinants.
        rngs: NNX random-number generators.

    Raises:
        ValueError: If ``hidden_one`` and ``hidden_two`` differ in length, or if
            no electrons are present.
    """

    def __init__(
        self,
        *,
        nspins: tuple[int, int],
        atoms: Float[Array, "natom ndim"],
        charges: Float[Array, " natom"],
        hidden_one: tuple[int, ...] = (64, 64, 64),
        hidden_two: tuple[int, ...] = (16, 16, 16),
        determinants: int = 4,
        full_det: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the equivariant streams, orbital projections and envelopes."""
        super().__init__()
        if len(hidden_one) != len(hidden_two):
            raise ValueError(
                "hidden_one and hidden_two must have the same number of layers; "
                f"got {len(hidden_one)} and {len(hidden_two)}."
            )
        if sum(nspins) == 0:
            raise ValueError("No electrons present in nspins.")

        self.nspins = nspins
        self.atoms = jnp.asarray(atoms)
        self.charges = jnp.asarray(charges)
        self.determinants = determinants
        self.full_det = full_det
        self.ndim = int(self.atoms.shape[1])
        natom = int(self.atoms.shape[0])
        nelectron = sum(nspins)
        # Static cumulative split indices for the spin partition.
        self._spin_split: tuple[int, ...] = (nspins[0],) if nspins[1] > 0 else ()
        active_spins = tuple(s for s in nspins if s > 0)
        self._active_spins = active_spins
        nchannels = len(active_spins)

        # Raw input feature widths: ae -> (natom*(ndim+1)), ee -> (ndim+1).
        n1 = natom * (self.ndim + 1)
        n2 = self.ndim + 1

        self.single_layers = nnx.List([])
        self.double_layers = nnx.List([])
        for width_one, width_two in zip(hidden_one, hidden_two, strict=True):
            in_one = n1 + nchannels * n1 + nchannels * n2
            self.single_layers.append(
                nnx.Linear(in_one, width_one, rngs=rngs)  # type: ignore[arg-type]
            )
            self.double_layers.append(
                nnx.Linear(n2, width_two, rngs=rngs)  # type: ignore[arg-type]
            )
            n1, n2 = width_one, width_two
        self._orbital_in = n1

        # Orbital projections: per spin channel, (determinants * orbitals_out).
        orbital_projections = []
        envelope_pi = []
        envelope_sigma = []
        for spin in active_spins:
            orbitals_out = nelectron if full_det else spin
            out_features = determinants * orbitals_out
            orbital_projections.append(nnx.Linear(self._orbital_in, out_features, rngs=rngs))
            # Isotropic exponential envelope params, one per (atom, orbital).
            envelope_pi.append(nnx.Param(jnp.ones((natom, out_features))))
            envelope_sigma.append(nnx.Param(jnp.ones((natom, out_features))))
        self.orbital_projections = nnx.List(orbital_projections)
        self.envelope_pi = nnx.List(envelope_pi)
        self.envelope_sigma = nnx.List(envelope_sigma)

        # Per-determinant mixing weights of the generalized-Slater sum.
        self.determinant_weights = nnx.Param(jnp.ones((determinants, 1)))

    def _equivariant_backbone(
        self,
        ae: Float[Array, "nelectron natom ndim"],
        ee: Float[Array, "nelectron nelectron ndim"],
        r_ae: Float[Array, "nelectron natom 1"],
        r_ee: Float[Array, "nelectron nelectron 1"],
    ) -> Array:
        """Run the interleaved one/two-electron streams to per-electron features.

        Returns the ``h_to_orbitals`` array of shape ``(nelectron, orbital_in)``.
        """
        ae_features = jnp.concatenate([r_ae, ae], axis=-1)
        ae_features = ae_features.reshape(ae_features.shape[0], -1)
        ee_features = jnp.concatenate([r_ee, ee], axis=-1)

        h_one = ae_features
        h_two = ee_features
        for single, double in zip(self.single_layers, self.double_layers, strict=True):
            features = _symmetric_features(h_one, h_two, self._spin_split)
            h_one_next = jnp.tanh(single(features))
            h_two_next = jnp.tanh(double(h_two))
            # Residual connection where shapes match (FermiNet ``residual``).
            h_one = h_one_next + h_one if h_one_next.shape == h_one.shape else h_one_next
            h_two = h_two_next + h_two if h_two_next.shape == h_two.shape else h_two_next
        return h_one

    def _orbital_matrices(
        self,
        h_to_orbitals: Float[Array, "nelectron orbital_in"],
        ae: Float[Array, "nelectron natom ndim"],  # noqa: ARG002 - orbital builder interface receives atom-electron displacements
        r_ae: Float[Array, "nelectron natom 1"],
    ) -> list[Array]:
        """Project features to per-determinant orbital matrices with envelopes.

        Returns one matrix per active spin channel, each of shape
        ``(determinants, n_rows, n_cols)``.
        """
        spin_split = self._spin_split
        h_channels = jnp.split(h_to_orbitals, spin_split, axis=0)
        r_ae_channels = jnp.split(r_ae, spin_split, axis=0)
        nelectron = sum(self.nspins)

        matrices = []
        for index, spin in enumerate(self._active_spins):
            h_channel = h_channels[index]
            orbitals = self.orbital_projections[index](h_channel)
            # Isotropic envelope: sum_atom pi * exp(-sigma * r_ae).
            decay = jnp.exp(-r_ae_channels[index] * self.envelope_sigma[index].value)
            envelope = jnp.sum(decay * self.envelope_pi[index].value, axis=1)
            orbitals = orbitals * envelope
            cols = nelectron if self.full_det else spin
            matrices.append(orbitals.reshape(spin, self.determinants, cols))
        return matrices

    def __call__(self, positions: Float[Array, "nelectron ndim"]) -> tuple[Array, Array]:
        """Evaluate the wavefunction for a single walker.

        Args:
            positions: Electron coordinates of shape ``(nelectron, ndim)``.

        Returns:
            A ``(sign, log|psi|)`` tuple of scalars.
        """
        ae, ee, r_ae, r_ee = construct_input_features(positions, self.atoms)
        h_to_orbitals = self._equivariant_backbone(ae, ee, r_ae, r_ee)
        matrices = self._orbital_matrices(h_to_orbitals, ae, r_ae)

        # Reorder each channel to (determinants, rows, cols) for logdet_matmul.
        if self.full_det:
            # Stack channels along the row axis into one dense determinant.
            dense = jnp.concatenate([jnp.moveaxis(m, 0, 1) for m in matrices], axis=1)
            determinant_inputs = [dense]
        else:
            determinant_inputs = [jnp.moveaxis(m, 0, 1) for m in matrices]

        return logdet_matmul(determinant_inputs, self.determinant_weights.value)


__all__ = ["FermiNet"]
