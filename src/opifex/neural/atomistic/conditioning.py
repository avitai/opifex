r"""Charge / spin conditioning for atomistic backbones (UMA / OrbMol).

A single interatomic-potential network can only describe charged or open-shell
systems if it is told the *global* total charge and spin state -- otherwise the
energy is ill-defined (the same nuclei + positions admit several charge/spin
states with different energies). UMA (Wood et al. 2025, "UMA: A Family of
Universal Models for Atoms", arXiv:2506.23971) and OrbMol therefore **reserve
input slots for charge and spin**: each is mapped to a feature vector of the
backbone's scalar width and *added to every atom's invariant* (``l = 0``)
*features*, so the global condition is broadcast identically to all atoms.

This module ports the fairchem UMA ``ChgSpinEmbedding`` recipe
(``fairchem/core/models/uma/nn/embedding.py``) and its application in
``escn_md.py`` (``x_message[:, 0, :] += csd_mixed_emb[batch]``): two separate
embeddings -- one for charge, one for spin -- are mixed by
``SiLU(Linear(concat(charge_emb, spin_emb)))`` down to ``feature_dim`` and
broadcast to ``(n_atoms, feature_dim)`` for addition onto the backbone's initial
node features.

Two embedding strategies are provided, mirroring UMA's ``embedding_type``:

* ``"table"`` -- a learned :class:`flax.nnx.Embed` table indexed by the integer
  charge / multiplicity (UMA's ``rand_emb``). Bounded to a documented integer
  range via an index offset (charge ``[-100, 100]`` -> 201 rows, multiplicity
  ``[1, 100]`` -> 100 rows), so out-of-range inputs fail fast.
* ``"fourier"`` -- a random-feature sinusoidal embedding (UMA's ``pos_emb``):
  ``[sin(2 pi w c), cos(2 pi w c)]`` with fixed random frequencies ``w``. This
  handles arbitrary (even unseen) integer values without an a-priori table
  size, at the cost of a learned table's per-value flexibility.

Charge and multiplicity are **static structural metadata** (small Python ints
off a :class:`~opifex.core.quantum.molecular_system.MolecularSystem`), not
arrays. Following the repo's numpy-static-metadata convention (see
:class:`~opifex.neural.atomistic.scale_shift.AtomicScaleShift` and
``MolecularSystem.n_electrons``), they are read concretely at trace time with
``np.asarray(...).item()`` and the indexing / featurisation uses NumPy so the
module stays ``jit`` / ``grad`` / ``vmap`` compatible when charge / spin are
passed as static arguments.

References:
    * Wood et al. 2025, arXiv:2506.23971 -- UMA reserves input slots for charge
      and spin, conditioning the node embedding on the global state.
    * fairchem ``ChgSpinEmbedding`` (``uma/nn/embedding.py``) and its use in
      ``escn_md.py`` -- the table / Fourier embeddings, the index ranges and the
      ``SiLU(Linear(concat(...)))`` mixing added to the ``l = 0`` channel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


logger = logging.getLogger(__name__)


_CHARGE_MIN = -100
"""Most negative total charge representable by the learned table (UMA range)."""

_CHARGE_MAX = 100
"""Most positive total charge representable by the learned table (UMA range)."""

_MULTIPLICITY_MIN = 1
"""Smallest spin multiplicity ``2S + 1`` (a singlet); multiplicity is positive."""

_MULTIPLICITY_MAX = 100
"""Largest spin multiplicity representable by the learned table (UMA range)."""

_FOURIER_FREQUENCY_SCALE = 1.0
"""Std of the fixed random Fourier frequencies ``w`` (UMA ``pos_emb`` ``scale``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeSpinConditioningConfig:
    """Hyper-parameters of a :class:`ChargeSpinConditioning` module.

    Attributes:
        feature_dim: Width ``F`` of the produced conditioning vector. Matches the
            backbone's scalar (``l = 0``) node-feature width so the conditioning
            can be added onto the initial node features.
        embedding_type: ``"table"`` for a learned per-integer embedding bounded to
            the documented range, or ``"fourier"`` for a random-feature sinusoidal
            embedding handling arbitrary integer values.
    """

    feature_dim: int = 64
    embedding_type: str = "table"


def _validate_charge(charge: int) -> None:
    """Raise if the total charge falls outside the documented table range."""
    if not _CHARGE_MIN <= charge <= _CHARGE_MAX:
        raise ValueError(
            f"charge {charge} is outside the supported range [{_CHARGE_MIN}, {_CHARGE_MAX}]"
        )


def _validate_multiplicity(multiplicity: int) -> None:
    """Raise if the spin multiplicity is non-positive or above the table range."""
    if not _MULTIPLICITY_MIN <= multiplicity <= _MULTIPLICITY_MAX:
        raise ValueError(
            f"multiplicity {multiplicity} is outside the supported range "
            f"[{_MULTIPLICITY_MIN}, {_MULTIPLICITY_MAX}]"
        )


class _IntegerEmbedding(nnx.Module):
    """Embed a single static integer to a ``feature_dim`` vector (one channel).

    Wraps either a learned :class:`flax.nnx.Embed` table (``"table"``) or a
    fixed-frequency random Fourier featuriser (``"fourier"``), mirroring UMA's
    ``rand_emb`` / ``pos_emb``. The integer is consumed as static metadata.

    Args:
        num_embeddings: Table size (number of representable integers).
        index_offset: Added to the integer to map it onto ``[0, num_embeddings)``.
        feature_dim: Output width ``F``.
        embedding_type: ``"table"`` or ``"fourier"``.
        rngs: Random number generators (keyword-only) seeding the weights.
    """

    def __init__(
        self,
        *,
        num_embeddings: int,
        index_offset: int,
        feature_dim: int,
        embedding_type: str,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the learned table or the fixed random Fourier frequencies."""
        super().__init__()
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim {feature_dim} must be even for conditioning")
        self.embedding_type = embedding_type
        self.index_offset = index_offset
        if embedding_type == "table":
            self.table = nnx.Embed(num_embeddings=num_embeddings, features=feature_dim, rngs=rngs)
        elif embedding_type == "fourier":
            # Fixed (non-learned) random frequencies, UMA pos_emb: half the width
            # is sin, half cos, so draw feature_dim // 2 frequencies.
            frequencies = _FOURIER_FREQUENCY_SCALE * jax.random.normal(
                rngs.params(), (feature_dim // 2,)
            )
            self.frequencies = nnx.Variable(frequencies)
        else:
            raise ValueError(f"embedding_type {embedding_type!r} must be 'table' or 'fourier'")

    def __call__(self, value: int) -> Float[Array, " feature_dim"]:
        """Embed the static integer ``value`` to a ``(feature_dim,)`` vector.

        Args:
            value: The (static) integer charge or multiplicity.

        Returns:
            The conditioning channel of shape ``(feature_dim,)``.
        """
        if self.embedding_type == "table":
            # Index built with NumPy so it stays a concrete static int under jit.
            index = jnp.asarray(int(np.asarray(value).item()) + self.index_offset)
            return self.table(index)
        # Fourier: project the scalar onto fixed frequencies, emit sin/cos pair.
        scalar = float(np.asarray(value).item())
        projection = scalar * self.frequencies.value * 2.0 * jnp.pi
        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)], axis=-1)


class ChargeSpinConditioning(nnx.Module):
    r"""Inject global total charge and spin multiplicity into per-atom features.

    Reproduces the UMA / OrbMol charge-spin conditioning (arXiv:2506.23971): the
    static integer total charge and spin multiplicity of a
    :class:`~opifex.core.quantum.molecular_system.MolecularSystem` are each
    embedded to ``feature_dim`` (learned table or random Fourier features), mixed
    by ``SiLU(Linear(concat(charge_emb, spin_emb)))`` to ``feature_dim`` and
    broadcast to ``(n_atoms, feature_dim)`` -- the identical global vector for
    every atom -- so a backbone can **add** it onto its initial invariant
    (``l = 0``) node features (fairchem ``escn_md.py``
    ``x_message[:, 0, :] += csd_mixed_emb[batch]``).

    Charge and multiplicity are static structural metadata; they are read with
    ``np.asarray(...).item()`` at trace time and must be passed as static
    arguments under ``jit`` (the numpy-static-metadata convention).

    Args:
        config: Conditioning hyper-parameters. Defaults to
            :class:`ChargeSpinConditioningConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.
    """

    def __init__(
        self,
        *,
        config: ChargeSpinConditioningConfig | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the charge / spin embeddings and the mixing linear."""
        super().__init__()
        self.config = config if config is not None else ChargeSpinConditioningConfig()
        feature_dim = self.config.feature_dim
        embedding_type = self.config.embedding_type
        self.charge_embedding = _IntegerEmbedding(
            num_embeddings=_CHARGE_MAX - _CHARGE_MIN + 1,
            index_offset=-_CHARGE_MIN,
            feature_dim=feature_dim,
            embedding_type=embedding_type,
            rngs=rngs,
        )
        self.spin_embedding = _IntegerEmbedding(
            num_embeddings=_MULTIPLICITY_MAX - _MULTIPLICITY_MIN + 1,
            index_offset=-_MULTIPLICITY_MIN,
            feature_dim=feature_dim,
            embedding_type=embedding_type,
            rngs=rngs,
        )
        self.mix = nnx.Linear(2 * feature_dim, feature_dim, rngs=rngs)

    def __call__(
        self, *, charge: int, multiplicity: int, n_atoms: int
    ) -> Float[Array, "n_atoms feature_dim"]:
        """Return the per-atom conditioning for a given charge / spin state.

        Args:
            charge: Total molecular charge (static int, bounded for ``"table"``).
            multiplicity: Spin multiplicity ``2S + 1`` (static positive int).
            n_atoms: Number of atoms to broadcast the conditioning over (static).

        Returns:
            The conditioning of shape ``(n_atoms, feature_dim)`` -- the identical
            global vector for every atom -- to be **added** to the backbone's
            initial node features.
        """
        _validate_charge(charge)
        _validate_multiplicity(multiplicity)
        charge_emb = self.charge_embedding(charge)
        spin_emb = self.spin_embedding(multiplicity)
        mixed = nnx.silu(self.mix(jnp.concatenate([charge_emb, spin_emb], axis=-1)))
        return jnp.broadcast_to(mixed, (n_atoms, mixed.shape[-1]))

    def from_system(self, system: MolecularSystem) -> Float[Array, "n_atoms feature_dim"]:
        """Return the per-atom conditioning for a :class:`MolecularSystem`.

        Reads the static ``charge`` and ``multiplicity`` (and ``n_atoms``) off the
        system and delegates to :meth:`__call__`.

        Args:
            system: The molecular system carrying the global charge / spin state.

        Returns:
            The conditioning of shape ``(n_atoms, feature_dim)``.
        """
        return self(
            charge=int(np.asarray(system.charge).item()),
            multiplicity=int(np.asarray(system.multiplicity).item()),
            n_atoms=system.n_atoms,
        )


__all__ = [
    "ChargeSpinConditioning",
    "ChargeSpinConditioningConfig",
]
