"""A molecular system's electron count is static structural metadata.

:attr:`MolecularSystem.n_electrons` is a plain Python ``int``, so it can size an array or
drive control flow inside ``jax.jit`` instead of degrading to a tracer (the JAX sharp-bits
guide on static shapes under ``jit``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.core.problems import create_molecular_system


class TestMolecularSystemElectronCountStatic:
    """``n_electrons`` is static structural metadata, not a tracer."""

    def test_n_electrons_is_python_int(self) -> None:
        """A neutral hydrogen atom reports exactly one electron as a Python int."""
        hydrogen = create_molecular_system([("H", (0.0, 0.0, 0.0))])
        result = hydrogen.n_electrons
        assert isinstance(result, int)
        assert result == 1

    def test_charged_ion_reduces_electron_count(self) -> None:
        """A +1 helium cation has one fewer electron than neutral helium."""
        helium_cation = create_molecular_system([("He", (0.0, 0.0, 0.0))], charge=1)
        assert helium_cation.n_electrons == 1

    def test_n_electrons_is_static_under_jit(self) -> None:
        """``n_electrons`` stays a static int usable as an array shape under jit."""
        lithium = create_molecular_system([("Li", (0.0, 0.0, 0.0))])

        @jax.jit
        def build_with_count() -> jax.Array:
            # Sizing an array with ``n_electrons`` only works if it is static.
            return jnp.zeros((lithium.n_electrons,))

        result = build_with_count()
        assert result.shape == (3,)  # lithium has three electrons
