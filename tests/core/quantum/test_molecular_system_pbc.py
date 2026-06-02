r"""Tests for the periodic-boundary fields on :class:`MolecularSystem`.

Adding ``pbc`` (and reusing the existing ``cell``) must be additive: existing
keyword-only constructions without these fields keep working, and ``pbc`` defaults
to ``None`` (treated as non-periodic).
"""

from __future__ import annotations

import jax.numpy as jnp

from opifex.core.quantum.molecular_system import MolecularSystem


def _diatomic() -> dict[str, object]:
    return {
        "atomic_numbers": jnp.asarray([1, 1]),
        "positions": jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    }


class TestPbcField:
    def test_defaults_to_none(self) -> None:
        system = MolecularSystem(**_diatomic())  # type: ignore[arg-type]
        assert system.pbc is None

    def test_existing_construction_unaffected(self) -> None:
        """A system built without cell/pbc remains non-periodic."""
        system = MolecularSystem(**_diatomic())  # type: ignore[arg-type]
        assert system.is_periodic is False

    def test_pbc_accepts_tuple(self) -> None:
        system = MolecularSystem(
            **_diatomic(),  # type: ignore[arg-type]
            cell=jnp.eye(3) * 10.0,
            pbc=(True, True, True),
        )
        assert system.pbc == (True, True, True)

    def test_cell_with_pbc_is_periodic(self) -> None:
        system = MolecularSystem(
            **_diatomic(),  # type: ignore[arg-type]
            cell=jnp.eye(3) * 10.0,
            pbc=(True, True, True),
        )
        assert system.is_periodic is True

    def test_translate_preserves_pbc(self) -> None:
        system = MolecularSystem(
            **_diatomic(),  # type: ignore[arg-type]
            cell=jnp.eye(3) * 10.0,
            pbc=(True, True, True),
        )
        moved = system.translate(jnp.asarray([1.0, 0.0, 0.0]))
        assert moved.pbc == (True, True, True)
        assert moved.cell is not None
