"""Probabilistic-numerics adapter catalogue.

Pattern-A frozen dataclasses declaring metadata for the probnum
ecosystem of probabilistic ODE / SDE / finite-difference solvers and
auxiliary axes. Concrete algorithms either:

* are vendored into other opifex subpackages (statespace, linalg) and
  cited per the design notes; or
* point at a user-installed backend via ``required_capabilities``; or
* are pure metadata for ecosystem awareness (e.g. deprecated repos).

Catalogue (21 specs):

Solver / ecosystem adapters:
* :class:`ProbdiffeqAdapterSpec` — mature JAX-native solver suite with
  9-axis configuration (extended with 4 new spec axes plus
  ``pn_observation_noise``).
* :class:`ProbnumAdapterSpec` — reference NumPy implementation;
  individual algorithms vendored into ``opifex.uncertainty.statespace``.
* :class:`TornadoxAdapterSpec` — emits a ``DeprecationWarning`` pointing
  at :class:`ProbdiffeqAdapterSpec` (per tornadox's own README).
  ``DiagonalEK1`` itself is vendored into
  ``opifex.uncertainty.statespace.diagonal_ek1``.
* :class:`ProbfindiffAdapterSpec` — JAX-native scattered-grid finite
  differences.
* :class:`DiffeqzooAdapterSpec` — canonical ODE problem catalogue;
  problems vendored into the test-fixtures module.

Likelihood adapters:
* :class:`FenrirAdapterSpec` — Tronarp+ ICML 2022 post-solve smoothing
  likelihood (arXiv:2202.01287).
* :class:`DaltonAdapterSpec` — Wu+Lysy AISTATS 2024 two-solve
  likelihood (arXiv:2306.05566).

Prior adapters:
* :class:`IOUPPriorSpec` — Integrated Ornstein-Uhlenbeck prior with
  three rate-parameter modes (Bosch+ NeurIPS 2023, arXiv:2305.14978).
* :class:`MaternPriorSpec` — Matérn SDE construction.
* :class:`IWPPriorSpec` — Integrated Wiener Process prior.

Solver axis specs (each carries a ``Literal`` enumeration):
* :class:`SsmFactSpec`, :class:`InitSchemeSpec`,
  :class:`CorrectionSpec`, :class:`CubatureRuleSpec`,
  :class:`StrategySpec`, :class:`CalibrationSpec`,
  :class:`DiffusionSpec`.

Specialised algorithmic specs:
* :class:`ManifoldUpdateSpec` — manifold-constrained update with
  ``jax.jacrev`` residual Jacobian.
* :class:`PerturbedStepSolverSpec` — Conrad+ 2017 perturbed-step
  solver (deferred algorithm; spec is ecosystem-aware).
* :class:`DenseOutputSamplingSpec` — joint posterior sampling at
  arbitrary density via interpolate-then-sample (Tronarp+ 2019).
* :class:`DynamicMVDiffusionSpec` / :class:`FixedMVDiffusionSpec` —
  multivariate diffusion machinery (time-dependent / time-invariant)
  (valid only with EK0 or DiagonalEK1 plus blockdiag covariance).

References
----------
* Tronarp+ 2019 arXiv:1810.03440 — *Probabilistic Solutions to ODEs as
  Non-Linear Bayesian Filtering*.
* Krämer+Hennig 2020 arXiv:2012.10106 — *Stable implementation of
  probabilistic ODE solvers*.
* Bosch+ 2023 arXiv:2305.14978 — *Probabilistic Exponential Integrators*.
* Tronarp+ 2022 arXiv:2202.01287 — *Fenrir: physics-enhanced regression*.
* Wu+Lysy 2024 arXiv:2306.05566 — *DALTON: Data-Adaptive Latent Solver
  for Stiff Probabilistic ODEs*.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


# ---------------------------------------------------------------------------
# Axis specs (Literal-typed enumerations carrying validation)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SsmFactSpec:
    """State-space-model covariance factorisation axis.

    Trinity per ``probdiffeq/impl/impl.py:30-41`` and Julia
    ``covariance_structure.jl:1-13``:
    ``isotropic`` ↔ ``IsometricKroneckerCovariance``,
    ``blockdiag`` ↔ ``BlockDiagonalCovariance``,
    ``dense`` ↔ ``DenseCovariance``.
    """

    choice: Literal["dense", "isotropic", "blockdiag"] = "dense"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InitSchemeSpec:
    """Initialisation scheme for the solver state.

    Choices follow probdiffeq's ``taylor`` family (Krämer+Hennig 2020,
    arXiv:2012.10106): Taylor coefficients via automatic differentiation,
    forward-mode, classical interpolation, or a simple zero-derivative
    initialiser for fast prototyping.
    """

    choice: Literal["taylor", "forward_mode", "classical", "simple"] = "taylor"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CorrectionSpec:
    """ODE-residual correction rule.

    probdiffeq exposes ``correction_ts0/ts1/slr0/slr1`` at
    ``ivpsolvers.py:487,500,527,542``. Cite Tronarp+ 2019 arXiv:1810.03440.
    """

    choice: Literal["ts0", "ts1", "slr0", "slr1"] = "ts0"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CubatureRuleSpec:
    """Cubature rule used by SLR corrections.

    probdiffeq exposes
    ``cubature_third_order_spherical / cubature_unscented_transform /
    cubature_gauss_hermite`` at ``ivpsolvers.py:94,117,144``.
    """

    choice: Literal["spherical", "unscented", "gauss_hermite"] = "spherical"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StrategySpec:
    """Posterior-estimation strategy (forward filter / RTS smoother / fixedpoint)."""

    choice: Literal["smoother", "filter", "fixedpoint"] = "smoother"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationSpec:
    """Diffusion-parameter calibration mode (MLE-style, dynamic, or none)."""

    choice: Literal["mle", "dynamic", "none"] = "mle"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionSpec:
    """Diffusion-parameter parametrisation (scalar / dynamic MV / fixed MV).

    MV (multivariate) diffusion is only valid in combination with EK0
    or DiagonalEK1 plus the ``blockdiag`` covariance factorisation per
    Julia ``algorithms.jl:108-129``.
    """

    choice: Literal["scalar", "dynamic_mv", "fixed_mv"] = "scalar"


# ---------------------------------------------------------------------------
# Base class for probnum-ecosystem adapter specs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PNAdapterSpecBase:
    """Shared shape for probabilistic-numerics adapter specs."""

    default_strategy: DefaultStrategy = DefaultStrategy.PROBABILISTIC_NUMERICS
    source_package: str = "opifex"
    required_capabilities: tuple[str, ...] = ()
    family_tags: tuple[str, ...] = ()
    notes: str = ""

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Raise :class:`NotImplementedError` until the backend lands."""
        del model, capability
        raise NotImplementedError(
            f"Probabilistic-numerics adapter {type(self).__name__} is not "
            f"yet wired (source_package={self.source_package!r}). Family "
            f"tags: {self.family_tags!r}."
        )


# ---------------------------------------------------------------------------
# Solver / ecosystem adapter specs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProbdiffeqAdapterSpec(_PNAdapterSpecBase):
    """Mature JAX-native probabilistic ODE solver suite.

    Extended with the four solver-axis fields (``ssm_fact``,
    ``init_scheme``, ``correction``, ``cubature``) plus
    ``pn_observation_noise`` for residual regularisation on stiff
    problems (Julia ``algorithms.jl:108-129``).
    """

    source_package: str = "probdiffeq"
    required_capabilities: tuple[str, ...] = ("native_jax",)
    family_tags: tuple[str, ...] = (
        "ek0",
        "ek1",
        "ts0",
        "ts1",
        "slr0",
        "slr1",
        "smoother",
        "filter",
        "fixedpoint",
    )
    ssm_fact: SsmFactSpec = dataclasses.field(default_factory=SsmFactSpec)
    init_scheme: InitSchemeSpec = dataclasses.field(default_factory=InitSchemeSpec)
    correction: CorrectionSpec = dataclasses.field(default_factory=CorrectionSpec)
    cubature: CubatureRuleSpec = dataclasses.field(default_factory=CubatureRuleSpec)
    pn_observation_noise: float | jax.Array | None = None
    notes: str = (
        "Mature JAX-native suite. 9-axis configuration via "
        "ssm_fact / init_scheme / correction / cubature / strategy / "
        "calibration / diffusion / pn_observation_noise."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProbnumAdapterSpec(_PNAdapterSpecBase):
    """Reference NumPy ecosystem; algorithms vendored module-by-module."""

    source_package: str = "probnum"
    family_tags: tuple[str, ...] = ("ek0", "ek1", "ioup", "matern", "iwp")
    notes: str = (
        "Metadata-only — IOUP / Matérn / IWP priors and EK0 / EK1 "
        "correction references are vendored into "
        "opifex.uncertainty.statespace, citing probnum/randprocs/markov/"
        "integrator/* and probnum/diffeq/odefilter/approx_strategies/"
        "_ek.py module-by-module."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TornadoxAdapterSpec(_PNAdapterSpecBase):
    """Deprecated metadata-only adapter; DiagonalEK1 vendored separately.

    Emits a :class:`DeprecationWarning` at construction pointing users
    at :class:`ProbdiffeqAdapterSpec`. The DiagonalEK1 implementation
    is vendored under
    :func:`opifex.uncertainty.statespace.diagonal_ek1_step`.
    """

    source_package: str = "tornadox"
    family_tags: tuple[str, ...] = ("diagonal_ek1",)
    notes: str = (
        "Deprecated — use ProbdiffeqAdapterSpec. DiagonalEK1 vendored "
        "into opifex.uncertainty.statespace.diagonal_ek1_step citing "
        "tornadox/ek1.py:273-332."
    )

    def __post_init__(self) -> None:
        """Emit a DeprecationWarning pointing at the probdiffeq successor."""
        warnings.warn(
            "TornadoxAdapterSpec is deprecated; use ProbdiffeqAdapterSpec. "
            "tornadox itself declares the package superseded.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProbfindiffAdapterSpec(_PNAdapterSpecBase):
    """User-installed JAX-native scattered-grid finite differences."""

    source_package: str = "probfindiff"
    required_capabilities: tuple[str, ...] = ("native_jax",)
    family_tags: tuple[str, ...] = (
        "scattered_grid",
        "adaptive_collocation",
    )
    notes: str = (
        "JAX-native finite-difference stencils on scattered grids — "
        "useful for adaptive PINN collocation."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiffeqzooAdapterSpec(_PNAdapterSpecBase):
    """Canonical ODE problem catalogue (Lotka-Volterra, FitzHugh-Nagumo, …)."""

    source_package: str = "diffeqzoo"
    family_tags: tuple[str, ...] = ("problem_catalogue",)
    notes: str = (
        "Metadata-only — a small canonical ODE problem catalogue is "
        "vendored into tests/uncertainty/fixtures/canonical_odes.py "
        "with BibTeX annotations per problem."
    )


# ---------------------------------------------------------------------------
# Likelihood specs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FenrirAdapterSpec(_PNAdapterSpecBase):
    """Fenrir post-solve smoothing data-likelihood (Tronarp+ 2022).

    arXiv:2202.01287. The likelihood is vendored adjacent to this spec;
    cite ``ProbNumDiffEq.jl/src/data_likelihoods/fenrir.jl:30-128``.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("data_likelihood", "smoother_likelihood")
    notes: str = (
        "Fenrir log-likelihood: forward solve + backward smoothing with "
        "data conditioning. arXiv:2202.01287."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the JAX-native Fenrir log-likelihood callable."""
        from opifex.uncertainty.scientific._likelihoods import fenrir_data_loglik

        del model, capability
        return fenrir_data_loglik


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DaltonAdapterSpec(_PNAdapterSpecBase):
    """DALTON data-adaptive latent two-solve likelihood (Wu+Lysy 2024).

    arXiv:2306.05566. Computes ``data_ll + with_pn_ll − without_pn_ll``
    from two solver passes (one with and one without
    ``DataUpdateCallback``). Cite
    ``ProbNumDiffEq.jl/src/data_likelihoods/dalton.jl:23-76``.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("data_likelihood", "two_solve")
    notes: str = (
        "DALTON log-likelihood = data_ll + with_pn_ll - without_pn_ll. "
        "Two solver passes per evaluation. arXiv:2306.05566."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the JAX-native DALTON log-likelihood combinator."""
        from opifex.uncertainty.scientific._likelihoods import dalton_data_loglik

        del model, capability
        return dalton_data_loglik


# ---------------------------------------------------------------------------
# Prior specs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IOUPPriorSpec(_PNAdapterSpecBase):
    """Integrated Ornstein-Uhlenbeck prior.

    Three rate-parameter modes (scalar, vector, matrix) per Julia
    ``priors/ioup.jl:103-117``. Cite arXiv:2305.14978 (Bosch+ NeurIPS
    2023 "Probabilistic Exponential Integrators").
    """

    num_derivatives: int = 1
    wiener_process_dimension: int = 1
    rate_parameter: float | jax.Array = 1.0
    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "ioup",
        "scalar_rate",
        "vector_rate",
        "matrix_rate",
    )
    notes: str = (
        "IOUP SDE construction with three rate-parameter modes "
        "(scalar / vector / matrix). arXiv:2305.14978."
    )

    def __post_init__(self) -> None:
        """Validate ``rate_parameter`` shape against ``wiener_process_dimension``."""
        rate_array = jnp.asarray(self.rate_parameter)
        if rate_array.ndim == 1:
            if rate_array.shape[0] != self.wiener_process_dimension:
                raise ValueError(
                    "IOUPPriorSpec: rate_parameter vector length must equal "
                    f"wiener_process_dimension={self.wiener_process_dimension}; "
                    f"got shape {rate_array.shape!r}."
                )
        elif rate_array.ndim == 2:
            if rate_array.shape != (
                self.wiener_process_dimension,
                self.wiener_process_dimension,
            ):
                raise ValueError(
                    "IOUPPriorSpec: rate_parameter matrix must be "
                    f"({self.wiener_process_dimension}, "
                    f"{self.wiener_process_dimension}); "
                    f"got shape {rate_array.shape!r}."
                )
        elif rate_array.ndim > 2:
            raise ValueError(
                "IOUPPriorSpec: rate_parameter must be a scalar, 1-D vector, "
                f"or 2-D matrix; got ndim={rate_array.ndim}."
            )

    @property
    def rate_mode(self) -> str:
        """Return the rate-parameter mode (``scalar`` / ``vector`` / ``matrix``)."""
        rate_array = jnp.asarray(self.rate_parameter)
        if rate_array.ndim == 0:
            return "scalar"
        if rate_array.ndim == 1:
            return "vector"
        return "matrix"

    def build_sde(self) -> tuple[jax.Array, jax.Array]:
        """Build the ``(drift, dispersion)`` SDE matrices."""
        from opifex.uncertainty.scientific._priors_sde import ioup_sde

        return ioup_sde(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=self.wiener_process_dimension,
            rate_parameter=self.rate_parameter,
        )

    def wrap(self, model: Any, capability: UQCapability) -> tuple[jax.Array, jax.Array]:
        """Return the ``(drift, dispersion)`` SDE pair for downstream use."""
        del model, capability
        return self.build_sde()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MaternPriorSpec(_PNAdapterSpecBase):
    """Matérn SDE construction (binomial-coefficient SDE matrix building)."""

    num_derivatives: int = 1
    wiener_process_dimension: int = 1
    lengthscale: float = 1.0
    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("matern_prior",)
    notes: str = (
        "Matérn SDE construction via binomial-coefficient SDE matrix "
        "building. Cite Särkkä & Solin 2019 §12.3."
    )

    def build_sde(self) -> tuple[jax.Array, jax.Array]:
        """Build the ``(drift, dispersion)`` SDE matrices."""
        from opifex.uncertainty.scientific._priors_sde import matern_sde

        return matern_sde(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=self.wiener_process_dimension,
            lengthscale=self.lengthscale,
        )

    def wrap(self, model: Any, capability: UQCapability) -> tuple[jax.Array, jax.Array]:
        """Return the ``(drift, dispersion)`` SDE pair for downstream use."""
        del model, capability
        return self.build_sde()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IWPPriorSpec(_PNAdapterSpecBase):
    """Integrated Wiener Process prior (probabilistic ODE solver default)."""

    num_derivatives: int = 1
    wiener_process_dimension: int = 1
    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("iwp",)
    notes: str = (
        "Integrated Wiener Process prior — the canonical default for "
        "probabilistic ODE solvers. Cite probnum/randprocs/markov/"
        "integrator/_iwp.py."
    )

    def build_sde(self) -> tuple[jax.Array, jax.Array]:
        """Build the ``(drift, dispersion)`` SDE matrices."""
        from opifex.uncertainty.scientific._priors_sde import iwp_sde

        return iwp_sde(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=self.wiener_process_dimension,
        )

    def wrap(self, model: Any, capability: UQCapability) -> tuple[jax.Array, jax.Array]:
        """Return the ``(drift, dispersion)`` SDE pair for downstream use."""
        del model, capability
        return self.build_sde()


# ---------------------------------------------------------------------------
# Specialised algorithmic specs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ManifoldUpdateSpec(_PNAdapterSpecBase):
    """Manifold-constrained update using ``jax.jacrev`` for residual Jacobian."""

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("manifold_update", "residual_jacobian")
    notes: str = (
        "Projects the ODE solver state onto a manifold defined by a "
        "residual ``g(x) = 0``. Uses jax.jacrev to compute the "
        "residual Jacobian on-the-fly."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the JAX-native iterated EKF manifold-update callable."""
        from opifex.uncertainty.scientific._specialised import manifold_update

        del model, capability
        return manifold_update


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PerturbedStepSolverSpec(_PNAdapterSpecBase):
    """Perturbed-step solver (Conrad+ 2017) — deferred to Phase 8/9."""

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("perturbed_step", "stochastic_perturbation")
    notes: str = (
        "Conrad+ 2017 perturbed-step ODE solver. Algorithm deferred to "
        "Phase 8/9; spec is ecosystem-aware."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DenseOutputSamplingSpec(_PNAdapterSpecBase):
    """Joint posterior sampling at arbitrary density via interpolate-then-sample.

    probdiffeq's ``markov_sample`` is grid-locked to solver steps; this
    spec covers the interpolate-then-sample pattern from Tronarp+ 2019
    arXiv:1810.03440 §5 and Julia ``solution_sampling.jl:64-87``.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("dense_output", "interpolate_sample")
    notes: str = (
        "Vendored interpolate-then-sample for joint posterior samples "
        "at arbitrary density. Cite Tronarp+ 2019 arXiv:1810.03440 §5."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the Cholesky-based dense-output Gaussian sampler."""
        from opifex.uncertainty.scientific._specialised import dense_output_sample

        del model, capability
        return dense_output_sample


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DynamicMVDiffusionSpec(_PNAdapterSpecBase):
    """Time-dependent multivariate diffusion (Julia ``diffusions/typedefs.jl:39-67``).

    The diffusion matrix ``D(t)`` varies with the solver's grid time.
    Sibling spec :class:`FixedMVDiffusionSpec` covers the time-
    independent counterpart. Both share
    :func:`opifex.uncertainty.scientific._specialised.apply_diffusion`
    as the underlying math layer; the spec classes advertise the
    distinct capability metadata for the adapter registry.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "apply_diffusion",
        "multivariate_diffusion",
        "time_dependent",
    )
    notes: str = (
        "Time-dependent multivariate diffusion D(t). Valid with EK0 or "
        "DiagonalEK1 + blockdiag covariance factorisation."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the time-dependent diffusion scaling callable."""
        from opifex.uncertainty.scientific._specialised import apply_diffusion

        del model, capability
        return apply_diffusion


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FixedMVDiffusionSpec(_PNAdapterSpecBase):
    """Time-independent multivariate diffusion (Julia ``diffusions/typedefs.jl:68-103``).

    The diffusion matrix ``D`` is held constant across the solver's
    grid. Companion to :class:`DynamicMVDiffusionSpec`; preferred when
    the SDE driving the prior is genuinely homogeneous in time.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "apply_diffusion",
        "multivariate_diffusion",
        "time_invariant",
    )
    notes: str = (
        "Time-invariant multivariate diffusion D. Valid with EK0 or "
        "DiagonalEK1 + blockdiag covariance factorisation."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the time-invariant diffusion scaling callable."""
        from opifex.uncertainty.scientific._specialised import apply_diffusion

        del model, capability
        return apply_diffusion


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExpEKSpec(_PNAdapterSpecBase):
    """Exponential extended Kalman correction (Tronarp+ 2019 §5).

    Replaces the linearisation in the standard EK0/EK1 correction with
    an exponential integrator step for the residual Jacobian — yields
    higher-order convergence on stiff IVPs without a Rosenbrock
    quadrature step. Spec exists to advertise capability metadata to
    user-installed ProbNum-family adapters; the JAX-native math layer
    is in :mod:`opifex.uncertainty.scientific._specialised`.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "exp_ek",
        "exponential_correction",
        "stiff_ivp",
    )
    notes: str = (
        "Exponential extended Kalman correction (Tronarp+ 2019 §5). "
        "Higher-order convergence on stiff IVPs; preferred over EK1 "
        "for moderately stiff dynamics."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RosenbrockExpEKSpec(_PNAdapterSpecBase):
    """Rosenbrock-style exponential EK correction.

    Combines the Rosenbrock-Wanner one-step correction with the
    exponential EK linearisation (Bosch+ 2021 §4.2). Suited to very
    stiff IVPs where ExpEK still over-shoots and a full Newton step is
    too expensive.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "rosenbrock_exp_ek",
        "exponential_correction",
        "stiff_ivp",
        "rosenbrock_wanner",
    )
    notes: str = (
        "Rosenbrock-Wanner one-step correction combined with the "
        "exponential EK linearisation; targets very stiff IVPs."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiagonalEK1Spec(_PNAdapterSpecBase):
    """Diagonal EK1 correction with structured Jacobian (Krämer+ 2022).

    Approximates the EK1 Jacobian by its diagonal — preserves
    asymptotic convergence order while reducing per-step linear-algebra
    cost from ``O(d^2)`` to ``O(d)``. The underlying math is in
    :mod:`opifex.uncertainty.statespace.diagonal_ek1`; this spec
    advertises the capability for the ProbNum adapter registry.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "diagonal_ek1",
        "structured_jacobian",
        "linear_cost_correction",
    )
    notes: str = (
        "Diagonal-Jacobian EK1 correction (Krämer+ 2022). Reduces "
        "per-step cost to O(d) while preserving EK1 convergence order."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the diagonal-EK1 single-step callable."""
        from opifex.uncertainty.statespace.diagonal_ek1 import diagonal_ek1_step

        del model, capability
        return diagonal_ek1_step


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DataUpdateCallbackSpec(_PNAdapterSpecBase):
    """Solver-step callback for online data assimilation (probdiffeq pattern).

    Advertises the capability for adapters to register a callback that
    fires inside the solver loop and conditions the running posterior
    on incoming observations. Used to integrate Task 6.7 assimilation
    primitives (:func:`opifex.uncertainty.assimilation.update`) with
    ProbNum-family ODE solvers without modifying the solver core.
    """

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = (
        "data_update_callback",
        "online_assimilation",
        "solver_step_callback",
    )
    notes: str = (
        "Solver-step callback for assimilating observations inside a "
        "ProbNum-family ODE solver. Routes through Task 6.7 "
        "assimilation primitives."
    )


__all__ = [
    "CalibrationSpec",
    "CorrectionSpec",
    "CubatureRuleSpec",
    "DaltonAdapterSpec",
    "DataUpdateCallbackSpec",
    "DenseOutputSamplingSpec",
    "DiagonalEK1Spec",
    "DiffeqzooAdapterSpec",
    "DiffusionSpec",
    "DynamicMVDiffusionSpec",
    "ExpEKSpec",
    "FenrirAdapterSpec",
    "FixedMVDiffusionSpec",
    "IOUPPriorSpec",
    "IWPPriorSpec",
    "InitSchemeSpec",
    "ManifoldUpdateSpec",
    "MaternPriorSpec",
    "PerturbedStepSolverSpec",
    "ProbdiffeqAdapterSpec",
    "ProbfindiffAdapterSpec",
    "ProbnumAdapterSpec",
    "RosenbrockExpEKSpec",
    "SsmFactSpec",
    "StrategySpec",
    "TornadoxAdapterSpec",
]
