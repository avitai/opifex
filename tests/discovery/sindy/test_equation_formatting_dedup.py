"""Characterization tests for shared SINDy equation formatting (Task 12.3.11 A).

These tests pin the exact human-readable equation strings produced by the
three SINDy variants (``SINDy``, ``WeakSINDy``, ``EnsembleSINDy``). The string
output must remain byte-for-byte identical after the duplicated formatting
logic is extracted into ``format_sindy_equations``.

The fitted models use fixed deterministic data so the captured strings are
stable across runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.discovery.sindy.config import (
    EnsembleSINDyConfig,
    SINDyConfig,
    WeakSINDyConfig,
)
from opifex.discovery.sindy.ensemble_sindy import EnsembleSINDy
from opifex.discovery.sindy.sindy import SINDy
from opifex.discovery.sindy.weak_sindy import WeakSINDy


def _linear_2d_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic linear system dx/dt = A x with A fixed."""
    x = jax.random.normal(jax.random.PRNGKey(0), (300, 2))
    true_coef = jnp.array([[0.0, -2.0], [1.0, 0.0]])
    x_dot = x @ true_coef.T
    return x, x_dot


class TestSindyEquationStrings:
    """Pin exact equation strings for each SINDy variant."""

    def test_sindy_equations_string(self) -> None:
        """SINDy.equations produces a stable, fully-specified string list."""
        x, x_dot = _linear_2d_data()
        model = SINDy(SINDyConfig(polynomial_degree=1, threshold=0.05))
        model.fit(x, x_dot)

        eqs = model.equations(["x", "y"])

        # Structural pins: one equation per target, prefixed by d{name}/dt.
        assert len(eqs) == 2
        assert eqs[0].startswith("dx/dt = ")
        assert eqs[1].startswith("dy/dt = ")
        # Recovered linear structure: dx/dt ~ -2 y, dy/dt ~ x.
        assert "y" in eqs[0]
        assert "x" in eqs[1]
        # Pin full strings so extraction cannot alter formatting.
        assert eqs == _SINDY_EXPECTED

    def test_weak_sindy_equations_string(self) -> None:
        """WeakSINDy.equations produces a stable string list."""
        x = jnp.ones((200, 1))
        t = jnp.arange(200) * 0.01
        model = WeakSINDy(WeakSINDyConfig(polynomial_degree=1, n_subdomains=20))
        model.fit(x, t)

        eqs = model.equations(["x"])

        assert len(eqs) == 1
        assert eqs[0].startswith("dx/dt = ")
        assert eqs == _WEAK_EXPECTED

    def test_ensemble_sindy_equations_string(self) -> None:
        """EnsembleSINDy.equations keeps the ±std term formatting."""
        x, x_dot = _linear_2d_data()
        model = EnsembleSINDy(EnsembleSINDyConfig(polynomial_degree=1, threshold=0.05, n_models=10))
        model.fit(x, x_dot, key=jax.random.PRNGKey(42))

        eqs = model.equations(["x", "y"])

        assert len(eqs) == 2
        assert eqs[0].startswith("dx/dt = ")
        assert eqs[1].startswith("dy/dt = ")
        # Ensemble keeps the (mean±std) notation.
        assert "±" in eqs[0]
        assert eqs == _ENSEMBLE_EXPECTED

    def test_default_target_names_when_input_names_none(self) -> None:
        """Without input_names, targets fall back to x0, x1, ... ."""
        x, x_dot = _linear_2d_data()
        model = SINDy(SINDyConfig(polynomial_degree=1, threshold=0.05))
        model.fit(x, x_dot)

        eqs = model.equations()

        assert eqs[0].startswith("dx0/dt = ")
        assert eqs[1].startswith("dx1/dt = ")

    def test_precision_controls_decimal_places(self) -> None:
        """precision argument flows through to coefficient formatting."""
        x, x_dot = _linear_2d_data()
        model = SINDy(SINDyConfig(polynomial_degree=1, threshold=0.05))
        model.fit(x, x_dot)

        eqs = model.equations(["x", "y"], precision=1)

        # A coefficient near -2 formatted at precision=1 -> "-2.0".
        assert "-2.0" in eqs[0]


# Expected strings captured from the pre-refactor implementations on the
# deterministic fixtures above. Behaviour must be byte-identical afterwards.
_SINDY_EXPECTED: list[str] = [
    "dx/dt = -2.000 y",
    "dy/dt = 1.000 x",
]

_WEAK_EXPECTED: list[str] = [
    "dx/dt = 0",
]

_ENSEMBLE_EXPECTED: list[str] = [
    "dx/dt = (-2.000±0.000) y",
    "dy/dt = (1.000±0.000) x",
]
