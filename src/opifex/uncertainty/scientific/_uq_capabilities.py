"""UQ capability declarations for stochastic-field surrogates.

Static, module-level constants, collected into
:data:`opifex.capabilities.UQ_CAPABILITIES`; importing this module changes no global state.

Four registered surfaces:

* ``stochastic_fields:KLE`` — :class:`KarhunenLoeveExpansion` random
  field model + :func:`sample_kle_field` sampler. Default strategy
  :attr:`DefaultStrategy.KARHUNEN_LOEVE`.
* ``stochastic_fields:PCE`` — Polynomial chaos basis +
  :func:`fit_pce_coefficients` + :func:`pce_mean_variance` /
  :func:`pce_summary`. Default strategy
  :attr:`DefaultStrategy.POLYNOMIAL_CHAOS`.
* ``stochastic_fields:StochasticGalerkin`` —
  :class:`StochasticGalerkinSurrogate` + :func:`fit_galerkin_surrogate`
  / :func:`evaluate_galerkin_surrogate`. Default strategy
  :attr:`DefaultStrategy.STOCHASTIC_GALERKIN`.
* ``stochastic_fields:StochasticCollocation` —
  :class:`StochasticCollocationSurrogate` +
  :func:`fit_collocation_surrogate` / :func:`evaluate_collocation_surrogate`
  (sparse-grid Smolyak collocation as a stochastic-Galerkin variant).
  Default strategy :attr:`DefaultStrategy.STOCHASTIC_GALERKIN`.

All four advertise :attr:`supports_stochastic_field_input=True` so the
Phase 8 capability coverage tests (``test_phase8_capability_coverage.py``)
and the Phase 9 final audit can find the registered surfaces honestly.
"""

from __future__ import annotations

from opifex.core.capabilities import DefaultStrategy, UQCapability


_KLE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_stochastic_field_input=True,
    supports_function_space=True,
    default_strategy=DefaultStrategy.KARHUNEN_LOEVE,
    source_package="opifex",
    notes=(
        "Karhunen-Loève expansion of a stochastic input field — "
        ":class:`KarhunenLoeveExpansion` + :func:`sample_kle_field`. "
        "Mercer-theorem-based truncation with eigen-pair extraction in "
        "pure JAX (jit / vmap / grad compatible). Reference: Ghanem & "
        "Spanos (1991) 'Stochastic Finite Elements'."
    ),
)


_PCE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_stochastic_field_input=True,
    default_strategy=DefaultStrategy.POLYNOMIAL_CHAOS,
    source_package="opifex",
    notes=(
        "Polynomial chaos expansion — :class:`PolynomialChaosBasis`, "
        ":func:`fit_pce_coefficients`, :func:`pce_mean_variance`, "
        ":func:`pce_summary`, :func:`sample_pce_field`. Hermite / "
        "Legendre / Laguerre / Jacobi basis families via the "
        "Wiener-Askey scheme. Reference: Xiu & Karniadakis "
        "(SIAM J. Sci. Comput. 2002, doi:10.1137/S1064827501387826)."
    ),
)


_STOCHASTIC_GALERKIN_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_stochastic_field_input=True,
    supports_solver_uncertainty=True,
    default_strategy=DefaultStrategy.STOCHASTIC_GALERKIN,
    source_package="opifex",
    notes=(
        "Stochastic-Galerkin surrogate — :class:`StochasticGalerkinSurrogate`, "
        ":func:`fit_galerkin_surrogate`, :func:`evaluate_galerkin_surrogate`. "
        "Spectral projection onto the PCE basis with Galerkin "
        "orthogonality enforced exactly. Reference: Le Maître & Knio "
        "(2010) 'Spectral Methods for Uncertainty Quantification'."
    ),
)


_STOCHASTIC_COLLOCATION_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_stochastic_field_input=True,
    supports_solver_uncertainty=True,
    default_strategy=DefaultStrategy.STOCHASTIC_GALERKIN,
    source_package="opifex",
    notes=(
        "Stochastic collocation surrogate — "
        ":class:`StochasticCollocationSurrogate`, "
        ":func:`fit_collocation_surrogate`, "
        ":func:`evaluate_collocation_surrogate`, "
        ":func:`smolyak_sparse_grid`, :func:`tensor_grid_gauss_hermite`. "
        "Sparse-grid Smolyak collocation as the non-intrusive sibling of "
        "stochastic-Galerkin; shares the STOCHASTIC_GALERKIN strategy "
        "bucket. Reference: Xiu & Hesthaven (SIAM J. Sci. Comput. 2005)."
    ),
)


SCIENTIFIC_FIELD_CAPABILITIES: dict[str, UQCapability] = {
    "stochastic_fields:KLE": _KLE_CAPABILITY,
    "stochastic_fields:PCE": _PCE_CAPABILITY,
    "stochastic_fields:StochasticGalerkin": _STOCHASTIC_GALERKIN_CAPABILITY,
    "stochastic_fields:StochasticCollocation": _STOCHASTIC_COLLOCATION_CAPABILITY,
}


__all__ = ["SCIENTIFIC_FIELD_CAPABILITIES"]
