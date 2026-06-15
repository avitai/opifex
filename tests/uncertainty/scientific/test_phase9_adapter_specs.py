r"""Phase 9 ProbNum adapter-spec closure — Slice 20.

Verifies the five ProbNum-family adapter specs that the Phase 9
final-validation checklist (``09-phase-final-validation.md:751-756``,
line 755 names ``ExpEKSpec`` explicitly) and the Task 6.3 design
notes (``notes/04-task-6.3-expansion-design.md:178-200``) required
but that the audit confirmed absent:

* ``ExpEKSpec`` — exponential extended Kalman correction.
* ``RosenbrockExpEKSpec`` — Rosenbrock-Wanner ExpEK variant.
* ``DiagonalEK1Spec`` — diagonal-Jacobian EK1 correction.
* ``DataUpdateCallbackSpec`` — online assimilation callback.
* ``DynamicMVDiffusionSpec`` + ``FixedMVDiffusionSpec`` — split of
  the old single-class ``ApplyDiffusionSpec`` per the Julia
  ``diffusions/typedefs.jl:39-103`` separation.

References
----------
* Tronarp+ 2019 — *Bayesian ODE solvers: the maximum a posteriori
  estimate*, arXiv:1810.03440 (ExpEK).
* Krämer+ 2022 — *Probabilistic ODE solutions in millions of
  dimensions*, ICML (DiagonalEK1).
* Bosch+ 2021 — *Calibrated adaptive probabilistic ODE solvers*,
  AISTATS (Rosenbrock-style correction).
"""

from __future__ import annotations

from opifex.uncertainty.scientific.probabilistic_numerics import (
    DataUpdateCallbackSpec,
    DiagonalEK1Spec,
    DynamicMVDiffusionSpec,
    ExpEKSpec,
    FixedMVDiffusionSpec,
    RosenbrockExpEKSpec,
)


def test_phase9_named_probnum_specs_are_importable() -> None:
    """Phase 9 :751-756 + Task 6.3 design notes :178-200 named-spec gate."""
    import opifex.uncertainty.scientific.probabilistic_numerics as pn

    for name in (
        "ExpEKSpec",
        "RosenbrockExpEKSpec",
        "DiagonalEK1Spec",
        "DataUpdateCallbackSpec",
        "DynamicMVDiffusionSpec",
        "FixedMVDiffusionSpec",
    ):
        assert hasattr(pn, name), (
            f"ProbNum adapter spec '{name}' missing from "
            f"opifex.uncertainty.scientific.probabilistic_numerics."
        )


def test_exp_ek_spec_advertises_stiff_ivp_and_exponential_tags() -> None:
    """``ExpEKSpec`` family tags must include ``exp_ek`` and ``stiff_ivp``."""
    spec = ExpEKSpec()
    assert "exp_ek" in spec.family_tags
    assert "stiff_ivp" in spec.family_tags
    assert spec.source_package == "opifex"


def test_rosenbrock_exp_ek_spec_advertises_rosenbrock_wanner_tag() -> None:
    """``RosenbrockExpEKSpec`` family tags must include ``rosenbrock_wanner``."""
    spec = RosenbrockExpEKSpec()
    assert "rosenbrock_exp_ek" in spec.family_tags
    assert "rosenbrock_wanner" in spec.family_tags


def test_diagonal_ek1_spec_wrap_returns_diagonal_ek1_step() -> None:
    """``DiagonalEK1Spec.wrap`` returns the diagonal-EK1 single-step callable."""
    from opifex.uncertainty.registry import UQCapability
    from opifex.uncertainty.statespace.diagonal_ek1 import diagonal_ek1_step

    spec = DiagonalEK1Spec()
    capability = UQCapability()
    callable_returned = spec.wrap(model=object(), capability=capability)
    assert callable_returned is diagonal_ek1_step


def test_dynamic_and_fixed_mv_diffusion_specs_share_underlying_math() -> None:
    """Both diffusion specs wrap the same ``apply_diffusion`` callable."""
    from opifex.uncertainty.registry import UQCapability
    from opifex.uncertainty.scientific._specialised import apply_diffusion

    dynamic_callable = DynamicMVDiffusionSpec().wrap(model=object(), capability=UQCapability())
    fixed_callable = FixedMVDiffusionSpec().wrap(model=object(), capability=UQCapability())
    assert dynamic_callable is apply_diffusion
    assert fixed_callable is apply_diffusion


def test_dynamic_mv_diffusion_advertises_time_dependent_tag() -> None:
    """``DynamicMVDiffusionSpec`` family tags include ``time_dependent``."""
    assert "time_dependent" in DynamicMVDiffusionSpec().family_tags


def test_fixed_mv_diffusion_advertises_time_invariant_tag() -> None:
    """``FixedMVDiffusionSpec`` family tags include ``time_invariant``."""
    assert "time_invariant" in FixedMVDiffusionSpec().family_tags


def test_data_update_callback_spec_advertises_online_assimilation_tag() -> None:
    """``DataUpdateCallbackSpec`` family tags include ``online_assimilation``."""
    spec = DataUpdateCallbackSpec()
    assert "data_update_callback" in spec.family_tags
    assert "online_assimilation" in spec.family_tags
