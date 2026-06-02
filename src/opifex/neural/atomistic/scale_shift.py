r"""Per-atom energy scale-shift: reference energies (E0) + output normaliser.

This module closes the deferral documented in
:class:`~opifex.neural.atomistic.heads.energy.EnergyHead`: per-element reference
energies (``E0``) and an output normaliser. It implements the standard MACE /
NequIP affine readout transform

.. math:: E_{\text{phys}} = \text{scale} \cdot E_{\text{raw}}
          + n_{\text{atoms}} \cdot \text{shift},

where ``E_raw`` is the bare sum-of-atomic-energies the network predicts. The
``scale`` puts the network output on the (per-atom-energy) scale of the data and
the per-atom ``shift`` adds back the mean atomic reference energy, so the network
only has to learn the small *interaction* energy -- the dramatically easier and
better-conditioned target (Batzner et al. 2022, NequIP, arXiv:2101.03164; the
``../mace`` ``ScaleShiftBlock`` in ``mace/modules/blocks.py``).

The statistics fit by :func:`fit_atomic_scale_shift` are the MACE
``compute_mean_std_atomic_inter_energy`` recipe (``mace/modules/utils.py``):
``shift`` is the mean per-atom energy and ``scale`` is the standard deviation of
the per-configuration residual energy.

:class:`AtomicScaleShift` is a :func:`flax.struct.dataclass` so it is an
immutable, ``jit``/``grad``/``vmap``-traceable JAX PyTree (the same container
pattern as :mod:`opifex.uncertainty.types`).
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float  # noqa: TC002


@struct.dataclass(frozen=True, kw_only=True)
class AtomicScaleShift:
    r"""Affine per-atom energy transform ``scale * E_raw + n_atoms * shift``.

    Attributes:
        scale: Output normaliser multiplying the raw (summed) energy -- typically
            the standard deviation of the per-configuration residual energy.
        shift: Per-atom reference-energy (``E0``) offset added once per atom --
            typically the mean per-atom energy.
    """

    scale: Float[Array, ""]
    shift: Float[Array, ""]

    @classmethod
    def identity(cls) -> AtomicScaleShift:
        """Return the no-op transform (``scale=1``, ``shift=0``)."""
        return cls(scale=jnp.asarray(1.0), shift=jnp.asarray(0.0))

    def apply(self, raw_energy: Array, n_atoms: int) -> Array:
        """Map a raw (summed) energy to a physical energy.

        Args:
            raw_energy: The bare sum-of-atomic-energies from the network.
            n_atoms: Number of atoms in the configuration (static under ``jit``).

        Returns:
            ``scale * raw_energy + n_atoms * shift``.
        """
        return self.scale * raw_energy + n_atoms * self.shift

    def invert(self, physical_energy: Array, n_atoms: int) -> Array:
        """Map a physical energy back to the raw (network) scale.

        The inverse of :meth:`apply`; used to standardise training targets so the
        network fits the normalised residual energy.

        Args:
            physical_energy: A physical total energy.
            n_atoms: Number of atoms in the configuration (static under ``jit``).

        Returns:
            ``(physical_energy - n_atoms * shift) / scale``.
        """
        return (physical_energy - n_atoms * self.shift) / self.scale


def fit_atomic_scale_shift(
    energies: Float[Array, " n_configs"],
    atom_counts: Float[Array, " n_configs"],
) -> AtomicScaleShift:
    r"""Fit an :class:`AtomicScaleShift` from per-configuration energies.

    Implements the MACE ``compute_mean_std_atomic_inter_energy`` statistics
    (``mace/modules/utils.py``):

    .. math::
        \text{shift} = \operatorname{mean}_c (E_c / n_c), \qquad
        \text{scale} = \operatorname{std}_c (E_c - n_c \cdot \text{shift}),

    so ``shift`` is the mean atomic reference energy and ``scale`` is the spread
    of the per-configuration residual (interaction) energy.

    Args:
        energies: Total energy of each configuration, shape ``(n_configs,)``.
        atom_counts: Number of atoms in each configuration, shape ``(n_configs,)``.

    Returns:
        The fitted :class:`AtomicScaleShift`.
    """
    shift = jnp.mean(energies / atom_counts)
    scale = jnp.std(energies - atom_counts * shift)
    return AtomicScaleShift(scale=scale, shift=shift)


__all__ = ["AtomicScaleShift", "fit_atomic_scale_shift"]
