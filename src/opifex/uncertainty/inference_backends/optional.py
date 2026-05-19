"""Optional-backend adapter specs for the BackendRouter.

Each :class:`OptionalBackendSpec` is a frozen, slotted, hashable dataclass
describing a single optional inference / flow / distribution backend:

* ``name`` — display name shown to callers.
* ``family`` — ``"flow"`` / ``"sampler"`` / ``"distribution"``.
* ``source_package`` — the import root that owns the backend.
* ``import_module`` — the importable module name used by ``probe()`` to
  detect installation.
* ``install_hint`` — the pip / uv install command suffix shown when an
  unavailable backend is requested.
* ``method_names`` — supported sampler / inference method names (e.g.
  ``("hmc", "nuts")`` for a sampler family, ``("real_nvp",)`` for a flow).

``probe()`` returns ``True`` when the import root is importable.
``instantiate()`` either returns the backend's primary class (when
available) or raises :class:`ImportError` with the install hint AND a
pointer at the always-available Artifex alternative.

Adapter resolution order across the file is **Artifex-first**: every
optional spec ships alongside Artifex equivalents and the router selects
Artifex when no caller preference is expressed.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OptionalBackendSpec:
    """Capability declaration for one optional inference / flow / distribution backend."""

    name: str
    family: str
    source_package: str
    import_module: str
    install_hint: str
    method_names: tuple[str, ...] = ()
    artifex_alternative: str | None = None

    def probe(self) -> bool:
        """Return ``True`` when the optional package is installed."""
        return importlib.util.find_spec(self.import_module) is not None

    def instantiate(self) -> object:
        """Import and return the backend's primary class, or raise ``ImportError``.

        Always raises when the optional dependency is not installed; the
        message contains both the install hint and the canonical Artifex
        alternative so callers can decide between installing the optional
        package or using the Artifex default.
        """
        if not self.probe():
            artifex_hint = (
                f" Artifex always-available alternative: {self.artifex_alternative}."
                if self.artifex_alternative
                else ""
            )
            raise ImportError(
                f"Optional backend {self.name!r} ({self.source_package}) is "
                f"not installed. Install: {self.install_hint}.{artifex_hint}"
            )
        return importlib.import_module(self.import_module)


# ---------------------------------------------------------------------------
# Artifex flow family — always-available (Artifex is a hard dependency)
# ---------------------------------------------------------------------------

_ARTIFEX_FLOW_MODULE = "artifex.generative_models.models.flow"

ARTIFEX_FLOW_SPECS: tuple[OptionalBackendSpec, ...] = (
    OptionalBackendSpec(
        name="RealNVP",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.real_nvp",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("real_nvp",),
    ),
    OptionalBackendSpec(
        name="MAF",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.maf",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("masked_autoregressive_flow",),
    ),
    OptionalBackendSpec(
        name="IAF",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.iaf",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("inverse_autoregressive_flow",),
    ),
    OptionalBackendSpec(
        name="Glow",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.glow",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("glow",),
    ),
    OptionalBackendSpec(
        name="NeuralSplineFlow",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.neural_spline",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("neural_spline",),
    ),
    OptionalBackendSpec(
        name="ConditionalFlow",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.conditional",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("conditional",),
    ),
    OptionalBackendSpec(
        name="MADE",
        family="flow",
        source_package="artifex",
        import_module=f"{_ARTIFEX_FLOW_MODULE}.made",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("made",),
    ),
)


# ---------------------------------------------------------------------------
# Optional flow backends — install-gated
# ---------------------------------------------------------------------------

OPTIONAL_FLOW_SPECS: tuple[OptionalBackendSpec, ...] = (
    OptionalBackendSpec(
        name="bijx",
        family="flow",
        source_package="bijx",
        import_module="bijx",
        install_hint="uv pip install bijx",
        method_names=("real_nvp", "neural_spline", "iaf"),
        artifex_alternative="opifex.uncertainty.inference_backends.optional.ARTIFEX_FLOW_SPECS",
    ),
    OptionalBackendSpec(
        name="FlowJAX",
        family="flow",
        source_package="flowjax",
        import_module="flowjax",
        install_hint="uv pip install flowjax",
        method_names=("real_nvp", "neural_spline", "maf"),
        artifex_alternative="opifex.uncertainty.inference_backends.optional.ARTIFEX_FLOW_SPECS",
    ),
)


# ---------------------------------------------------------------------------
# Optional sampler / probabilistic-programming backends
# ---------------------------------------------------------------------------

OPTIONAL_SAMPLER_SPECS: tuple[OptionalBackendSpec, ...] = (
    OptionalBackendSpec(
        name="TFP-substrate",
        family="sampler",
        source_package="tensorflow_probability",
        import_module="tensorflow_probability.substrates.jax",
        install_hint="uv pip install tensorflow-probability",
        method_names=("hmc", "nuts", "replica_exchange_mc", "smc", "advi"),
        artifex_alternative=("opifex.uncertainty.inference_backends.blackjax.BlackJAXBackend"),
    ),
    OptionalBackendSpec(
        name="Bayeux",
        family="sampler",
        source_package="bayeux",
        import_module="bayeux",
        install_hint="uv pip install bayeux",
        method_names=("hmc", "nuts", "smc", "vi"),
        artifex_alternative=("opifex.uncertainty.inference_backends.blackjax.BlackJAXBackend"),
    ),
    OptionalBackendSpec(
        name="NumPyro",
        family="sampler",
        source_package="numpyro",
        import_module="numpyro",
        install_hint="uv pip install numpyro",
        method_names=("hmc", "nuts", "svi", "smc"),
        artifex_alternative=("opifex.uncertainty.inference_backends.blackjax.BlackJAXBackend"),
    ),
    OptionalBackendSpec(
        name="oryx",
        family="sampler",
        source_package="oryx",
        import_module="oryx",
        install_hint="uv pip install oryx",
        method_names=("inference",),
    ),
    OptionalBackendSpec(
        name="sbiax",
        family="sampler",
        source_package="sbiax",
        import_module="sbiax",
        install_hint="uv pip install sbiax",
        method_names=("npe", "nle", "nre"),
        artifex_alternative=("opifex.uncertainty.inference_backends.optional.ARTIFEX_FLOW_SPECS"),
    ),
    OptionalBackendSpec(
        name="flowMC",
        family="sampler",
        source_package="flowMC",
        import_module="flowMC",
        install_hint="uv pip install flowMC",
        method_names=("nf_mcmc",),
        artifex_alternative=("opifex.uncertainty.inference_backends.optional.ARTIFEX_FLOW_SPECS"),
    ),
    OptionalBackendSpec(
        name="traceax",
        family="sampler",
        source_package="traceax",
        import_module="traceax",
        install_hint="uv pip install traceax",
        method_names=("hutchinson_trace", "lanczos"),
    ),
    OptionalBackendSpec(
        name="matfree",
        family="sampler",
        source_package="matfree",
        import_module="matfree",
        install_hint="uv pip install matfree",
        method_names=("matvec_lanczos",),
    ),
    OptionalBackendSpec(
        name="kfac-jax",
        family="sampler",
        source_package="kfac_jax",
        import_module="kfac_jax",
        install_hint="uv pip install kfac-jax",
        method_names=("kfac",),
    ),
)


# ---------------------------------------------------------------------------
# Distribution-adapter specs (Artifex first, then GPJax / Distrax / TFP)
# ---------------------------------------------------------------------------

DISTRIBUTION_SPECS: tuple[OptionalBackendSpec, ...] = (
    OptionalBackendSpec(
        name="Artifex-Distribution",
        family="distribution",
        source_package="artifex",
        import_module="artifex.generative_models.core.distributions.base",
        install_hint="artifex (already installed as Opifex dependency)",
        method_names=("sample", "log_prob", "mean", "variance"),
    ),
    OptionalBackendSpec(
        name="GPJax",
        family="distribution",
        source_package="gpjax",
        import_module="gpjax",
        install_hint="uv pip install gpjax",
        method_names=("posterior", "predict"),
        artifex_alternative="artifex.generative_models.core.distributions.base.Distribution",
    ),
    OptionalBackendSpec(
        name="Distrax",
        family="distribution",
        source_package="distrax",
        import_module="distrax",
        install_hint="uv pip install distrax",
        method_names=("sample", "log_prob", "mean", "variance"),
        artifex_alternative="artifex.generative_models.core.distributions.base.Distribution",
    ),
    OptionalBackendSpec(
        name="TFP-substrate",
        family="distribution",
        source_package="tensorflow_probability",
        import_module="tensorflow_probability.substrates.jax",
        install_hint="uv pip install tensorflow-probability",
        method_names=("sample", "log_prob", "mean", "variance"),
        artifex_alternative="artifex.generative_models.core.distributions.base.Distribution",
    ),
)


__all__ = [
    "ARTIFEX_FLOW_SPECS",
    "DISTRIBUTION_SPECS",
    "OPTIONAL_FLOW_SPECS",
    "OPTIONAL_SAMPLER_SPECS",
    "OptionalBackendSpec",
]
